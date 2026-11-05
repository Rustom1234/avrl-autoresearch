"""Uniform model-call layer: Claude tiers via the Claude Code CLI, open-weight models via vLLM.

    from call import call
    r = call("haiku", "2+2?", schema={"type": "object", "properties": {"answer": {"type": "integer"}},
                                       "required": ["answer"]}, tag="smoke:math")
    r.ok, r.obj, r.cost_usd_plan, r.usage

All subprocess and HTTP work goes through `_run_claude`, `_http_post` and `_http_get` so tests can monkeypatch them.
See README.md for assumptions and limits.
"""
from __future__ import annotations

import fcntl
import http.client
import json
import os
import random
import re
import subprocess
import threading
import time
import urllib.error
import urllib.request
from contextlib import contextmanager
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path

try:  # optional, the minimal validator below covers required/type/properties/items/enum
    import jsonschema
except ImportError:  # pragma: no cover
    jsonschema = None

REGISTRY_PATH = Path(__file__).with_name("models.json")
DEFAULT_LOG = "calls.jsonl"
_ZERO_USAGE = {"input": 0, "output": 0, "cache_read": 0, "cache_write": 0}
# endpoint -> "response_format" | "structured_outputs": remembers which structured-output field the server accepts.
# (Top-level guided_json is deliberately never used: vLLM 0.30 accepts it but silently ignores it.)
_STRUCT_MODE: dict[str, str] = {}

MAX_HOPS = 2                                   # fallback hops per call()
RL_RETRIES = 8                                 # Claude rate-limit retries per call(), shared across hops
BACKOFF_START_S, BACKOFF_CAP_S = 60, 1800
BREAKER_POLL_S, SLOT_POLL_S = 10, 0.5          # seconds between breaker checks / free-slot checks
_SCRATCH = "/tmp/claude-22830/-homes-rdubash/2d8b70a3-2fb9-42a6-aac4-20bd12ab9c6d/scratchpad"
_RATE_LIMIT = re.compile(r"rate[ _-]?limit|usage limit|limit reached|overloaded|\b429\b|\b529\b|too many requests|quota",
                         re.I)
_INFRA_EXC = (OSError, subprocess.TimeoutExpired, http.client.HTTPException)  # refused, timeout, missing binary, ...


class ContextTooLong(Exception):
    """The vLLM server rejected the request because prompt plus max_tokens exceeds the context window."""


@dataclass
class CallResult:
    obj: object
    text: str
    ok: bool
    error: str | None
    model_name: str
    model_id: str
    provider: str
    usage: dict = field(default_factory=lambda: dict(_ZERO_USAGE))
    cost_usd_plan: float = 0.0
    cost_usd_nominal: float = 0.0
    latency_s: float = 0.0
    attempts: int = 0
    requested_model: str = ""
    served_model: str = ""
    error_class: str | None = None   # None when ok, else "infra" | "model" | "context"


@dataclass
class _Att:
    """Outcome of one attempt before schema validation."""
    text: str = ""
    obj: object = None
    usage: dict = field(default_factory=lambda: dict(_ZERO_USAGE))
    plan: float = 0.0
    nominal: float = 0.0
    error: str | None = None
    infra: bool = False          # infrastructure failure: eligible for fallback, never a model failure
    rate_limited: bool = False   # Claude rate or usage limit signature (implies infra)


# ---------------------------------------------------------------- registry

def load_registry(path=None) -> dict:
    """models.json is a list of entries; returns {name: entry}. Add entries to the file, no code change."""
    entries = json.loads(Path(path or REGISTRY_PATH).read_text())
    return {e["name"]: e for e in entries}


# ---------------------------------------------------------------- validation

_TYPES = {"object": dict, "array": list, "string": str, "boolean": bool, "null": type(None),
          "number": (int, float), "integer": int}


def _validate_minimal(obj, schema, path="$") -> str | None:
    """Checks type, enum, required, properties (recursive), items (recursive). Returns an error string or None."""
    t = schema.get("type")
    if t is not None:
        ok = False
        for name in (t if isinstance(t, list) else [t]):
            py = _TYPES.get(name)
            if py is None:
                return f"{path}: unsupported schema type {name!r}"
            if isinstance(obj, bool) and name in ("integer", "number"):
                continue
            if isinstance(obj, py):
                ok = True
        if not ok:
            return f"{path}: expected {t}, got {type(obj).__name__}"
    if "enum" in schema and obj not in schema["enum"]:
        return f"{path}: {obj!r} not in enum {schema['enum']}"
    if isinstance(obj, dict):
        for k in schema.get("required", []):
            if k not in obj:
                return f"{path}: missing required key {k!r}"
        for k, sub in schema.get("properties", {}).items():
            if k in obj:
                e = _validate_minimal(obj[k], sub, f"{path}.{k}")
                if e:
                    return e
    if isinstance(obj, list) and isinstance(schema.get("items"), dict):
        for i, v in enumerate(obj):
            e = _validate_minimal(v, schema["items"], f"{path}[{i}]")
            if e:
                return e
    return None


def validate(obj, schema) -> str | None:
    """Error string if obj violates schema, else None. Uses jsonschema when importable."""
    if jsonschema is not None:
        try:
            jsonschema.validate(obj, schema)
        except jsonschema.ValidationError as e:
            return e.message
        except jsonschema.SchemaError as e:
            return f"bad schema: {e.message}"
        return None
    return _validate_minimal(obj, schema)


def _parse_json(text: str):
    """json.loads, tolerating a ```json fence. Returns None if unparseable."""
    s = (text or "").strip()
    m = re.match(r"^```(?:json)?\s*(.*?)\s*```$", s, re.S)
    if m:
        s = m.group(1)
    try:
        return json.loads(s)
    except ValueError:
        return None


# ---------------------------------------------------------------- injectable I/O

def _run_claude(argv, stdin_text, timeout):
    """Run the claude CLI with the prompt on stdin. Returns (returncode, stdout, stderr). Raises on timeout."""
    p = subprocess.run(argv, input=stdin_text, capture_output=True, text=True, timeout=timeout)
    return p.returncode, p.stdout, p.stderr


def _http(req, timeout):
    """Returns (status, body) where body is the parsed JSON, or the raw text if not JSON."""
    try:
        with urllib.request.urlopen(req, timeout=timeout) as r:
            status, raw = r.status, r.read().decode("utf-8", "replace")
    except urllib.error.HTTPError as e:
        status, raw = e.code, e.read().decode("utf-8", "replace")
    try:
        return status, json.loads(raw)
    except ValueError:
        return status, raw


def _http_post(url, payload, timeout):
    return _http(urllib.request.Request(url, data=json.dumps(payload).encode(), method="POST",
                                        headers={"Content-Type": "application/json"}), timeout)


def _http_get(url, timeout):
    return _http(urllib.request.Request(url), timeout)


# ---------------------------------------------------------------- Claude coordination (breaker, slots)

def _breaker_path() -> Path:
    return Path(os.environ.get("AR_BREAKER_FILE") or f"{_SCRATCH}/claude_breaker.json")


def _breaker_until() -> float:
    try:
        return float(json.loads(_breaker_path().read_text())["until_epoch"])
    except (OSError, ValueError, KeyError, TypeError):
        return 0.0


def _wait_breaker():
    """Sleep (polling) while the shared breaker file says Claude is blocked."""
    while (rem := _breaker_until() - time.time()) > 0:
        time.sleep(min(BREAKER_POLL_S, rem))


def _trip_breaker(delay, reason):
    """Block all Claude callers for `delay` s (never shortens an existing block). Falls back to sleeping here."""
    until = time.time() + delay
    if until <= _breaker_until():
        return
    p = _breaker_path()
    try:
        p.parent.mkdir(parents=True, exist_ok=True)
        tmp = p.with_name(f"{p.name}.{os.getpid()}.{threading.get_ident()}.tmp")
        tmp.write_text(json.dumps({"until_epoch": until, "reason": reason[:200]}))
        os.replace(tmp, p)
    except OSError:
        time.sleep(delay)


def _backoff(i) -> float:
    """Sleep for rate-limit retry i (0-based): 60, 120, ... capped at 1800 s, jittered down to between 50% and 100%."""
    return min(BACKOFF_CAP_S, BACKOFF_START_S * 2 ** i) * (0.5 + 0.5 * random.random())


@contextmanager
def _claude_slot():
    """flock semaphore: at most AR_CLAUDE_MAX_CONCURRENT (default 4) holders across processes on this host."""
    n = max(1, int(os.environ.get("AR_CLAUDE_MAX_CONCURRENT", "4")))
    d = Path(os.environ.get("AR_LOCK_DIR") or _SCRATCH)
    d.mkdir(parents=True, exist_ok=True)
    while True:
        for i in range(n):
            fd = os.open(d / f"claude_slot_{i}.lock", os.O_RDWR | os.O_CREAT, 0o666)
            try:
                fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
            except OSError as e:
                os.close(fd)
                if not isinstance(e, BlockingIOError):
                    raise
                continue
            try:
                yield
            finally:
                os.close(fd)  # closing releases the flock
            return
        time.sleep(SLOT_POLL_S)


# ---------------------------------------------------------------- providers

def _claude_attempt(m, user, system, schema, timeout_s) -> _Att:
    argv = ["claude", "-p", "--model", m["model_id"], "--no-session-persistence", "--tools", "",
            "--strict-mcp-config", "--setting-sources", "", "--output-format", "json"]
    if schema:
        argv += ["--json-schema", json.dumps(schema)]
    if system:
        argv += ["--system-prompt", system]
    with _claude_slot():
        _wait_breaker()  # re-check: the breaker may have tripped while this call queued for a slot
        rc, out, err = _run_claude(argv, user, timeout_s)
    try:
        d = json.loads(out)
        assert isinstance(d, dict)
    except (ValueError, AssertionError):
        limited = rc != 0 and bool(_RATE_LIMIT.search(f"{err} {out}"))
        return _Att(text=out or "", error=f"claude CLI output is not a JSON object (rc={rc}): {(err or out or '')[:300]}",
                    infra=True, rate_limited=limited)
    u = d.get("usage") or {}
    att = _Att(text=d.get("result") or "", obj=d.get("structured_output"),
               usage={"input": u.get("input_tokens", 0), "output": u.get("output_tokens", 0),
                      "cache_read": u.get("cache_read_input_tokens", 0),
                      "cache_write": u.get("cache_creation_input_tokens", 0)},
               plan=float(d.get("total_cost_usd") or 0.0))
    att.nominal = att.plan
    if d.get("is_error") or rc != 0:
        status = d.get("api_error_status")
        att.error = f"claude CLI error (rc={rc}, status={status}): {(att.text or err or '')[:300]}"
        att.rate_limited = bool(d.get("is_error")) and bool(_RATE_LIMIT.search(f"{att.text} {err} {status}"))
        att.infra = att.rate_limited or (isinstance(status, int) and status >= 500)
    elif d.get("stop_reason") == "tool_use" and att.obj is None:
        # With --json-schema a successful call also ends on tool_use but carries structured_output.
        att.error = "stop_reason tool_use without structured_output (retryable)"
    return att


_UNKNOWN_FIELD = ("unknown", "unrecognized", "extra", "not permitted", "unsupported", "not supported", "unexpected")


def _msg(body) -> str:
    return (body if isinstance(body, str) else json.dumps(body)).lower()


def _is_context_error(status, body) -> bool:
    m = _msg(body)
    return status == 400 and ("context length" in m or "maximum context" in m or "context_length_exceeded" in m)


def _is_unknown_field(status, body) -> bool:
    m = _msg(body)
    return 400 <= status < 500 and any(w in m for w in _UNKNOWN_FIELD)


def _vllm_attempt(m, user, system, schema, temperature, max_tokens, timeout_s, extra=None) -> _Att:
    endpoint = m["endpoint"].rstrip("/")
    url = endpoint + "/v1/chat/completions"
    messages = ([{"role": "system", "content": system}] if system else []) + [{"role": "user", "content": user}]
    payload = {"model": m["model_id"], "messages": messages, "max_tokens": max_tokens}
    if temperature is not None:
        payload["temperature"] = temperature
    mode = _STRUCT_MODE.get(endpoint, "response_format")
    if schema and mode == "response_format":
        payload["response_format"] = {"type": "json_schema", "json_schema": {
            "name": schema.get("title") or "response", "schema": schema}}
    elif schema:
        payload["structured_outputs"] = {"json": schema}
    payload.update(extra or {})
    status, body = _http_post(url, payload, timeout_s)
    if schema and mode == "response_format" and _is_unknown_field(status, body) and not _is_context_error(status, body):
        payload = {**{k: v for k, v in payload.items() if k != "response_format"}, "structured_outputs": {"json": schema}}
        status, body = _http_post(url, payload, timeout_s)
        if status == 200:
            _STRUCT_MODE[endpoint] = "structured_outputs"
    if _is_context_error(status, body):
        raise ContextTooLong(_msg(body)[:300])
    if status != 200 or not isinstance(body, dict):
        return _Att(text=body if isinstance(body, str) else json.dumps(body), error=f"HTTP {status}: {_msg(body)[:300]}",
                    infra=status >= 500 or status == 404)  # 404: another model is loaded on this endpoint
    try:
        text = body["choices"][0]["message"]["content"] or ""
    except (KeyError, IndexError, TypeError):
        return _Att(text=json.dumps(body), error="vLLM reply has no choices[0].message.content")
    u = body.get("usage") or {}
    att = _Att(text=text, usage={"input": u.get("prompt_tokens", 0), "output": u.get("completion_tokens", 0),
                                 "cache_read": 0, "cache_write": 0})
    price = m.get("nominal_price_per_m")
    if price:
        att.nominal = (att.usage["input"] * price["input"] + att.usage["output"] * price["output"]) / 1e6
    return att


# ---------------------------------------------------------------- main entry

def _log(path, rec):
    with open(path, "a") as f:
        f.write(json.dumps(rec) + "\n")


def _call_model(m, requested, rl, *, user, system, schema, temperature, max_tokens, tag, timeout_s, log_path,
                attempts, extra) -> CallResult:
    """Up to `attempts` tries on one registry entry. Claude rate-limit retries (budget `rl`) do not use up attempts."""
    tot = dict(_ZERO_USAGE)
    plan = nominal = latency = 0.0
    obj, text, err, cls, ok, n, tries = None, "", None, None, False, 0, 0
    while n < attempts:
        n, tries = n + 1, tries + 1
        if m["provider"] == "claude":
            _wait_breaker()
        t0 = time.monotonic()
        ctx_err = None
        try:
            if m["provider"] == "claude":
                att = _claude_attempt(m, user, system, schema, timeout_s)
            else:
                att = _vllm_attempt(m, user, system, schema, temperature, max_tokens, timeout_s, extra)
        except ContextTooLong as e:
            ctx_err, att = e, _Att(error=f"ContextTooLong: {e}")
        except Exception as e:  # timeout, connection refused, E2BIG, missing binary, ...
            att = _Att(error=f"{type(e).__name__}: {e}", infra=isinstance(e, _INFRA_EXC))
        dt = time.monotonic() - t0
        obj, err = att.obj, att.error
        if err is None:
            if obj is None:
                obj = _parse_json(att.text)
            if schema and obj is None:
                err = "response is not parseable JSON"
            elif schema and (verr := validate(obj, schema)):
                err = f"schema validation failed: {verr}"
            elif not schema and obj is None and not att.text.strip():
                err = "empty response"
        text, ok = att.text, err is None
        cls = None if ok else "context" if ctx_err else "infra" if att.infra else "model"
        for k in tot:
            tot[k] += att.usage[k]
        plan += att.plan
        nominal += att.nominal
        latency += dt
        _log(log_path, {"ts": datetime.now(timezone.utc).isoformat(timespec="milliseconds"), "tag": tag,
                        "model_name": m["name"], "requested_model": requested, "served_model": m["name"],
                        "provider": m["provider"], "ok": ok, "error": err[:500] if err else None, "error_class": cls,
                        "usage": att.usage, "cost_usd_plan": att.plan, "cost_usd_nominal": att.nominal,
                        "latency_s": round(dt, 4), "prompt_chars": len(user) + len(system or ""),
                        "schema_name": (schema.get("title") or "response") if schema else None, "attempt": tries})
        if ctx_err:
            raise ctx_err
        if ok:
            break
        if att.rate_limited:
            _trip_breaker(_backoff(RL_RETRIES - rl["left"]), err)  # the next pass sleeps it off in _wait_breaker
            if rl["left"] == 0:
                break
            rl["left"] -= 1
            n -= 1
    return CallResult(obj=obj if ok else None, text=text, ok=ok, error=None if ok else err,
                      model_name=m["name"], model_id=m["model_id"], provider=m["provider"], usage=tot,
                      cost_usd_plan=plan, cost_usd_nominal=nominal, latency_s=latency, attempts=tries,
                      requested_model=requested, served_model=m["name"], error_class=cls)


def call(model_name, user, *, system=None, schema=None, temperature=None, max_tokens=2048, tag="",
         timeout_s=300, log_path=None, attempts=3, extra=None) -> CallResult:
    """One logical model call; returns CallResult, raises ContextTooLong (vllm only).

    Each model in the chain (requested, then its `fallback` entry, then that one's, at most MAX_HOPS=2 hops) gets up to
    `attempts` tries. Retried within a model: CLI/HTTP errors, timeouts, tool_use stop, unparseable JSON, schema
    violations. Moves to the next model only when the last failure was infrastructure (connection refused, timeout,
    HTTP 5xx, CLI not producing JSON, Claude rate limit after its retries); a schema/model failure or ContextTooLong
    never hops. Claude rate or usage limits trip the shared breaker file and are retried with backoff (RL_RETRIES per
    call, outside `attempts`). `healthy()` is not consulted here.
    temperature and max_tokens apply to vllm only (the Claude CLI exposes neither).
    extra (dict) is merged into the vllm request body last, so it can add or override fields, e.g.
    {"chat_template_kwargs": {"enable_thinking": False}} or {"reasoning_effort": "low"}. Ignored for claude.
    usage, costs, latency_s and attempts are summed over all tries and hops; the JSONL log has one line per try with
    requested_model, served_model and error_class. model_name/model_id/provider of the result describe the served model.
    """
    registry = load_registry()
    if model_name not in registry:
        raise KeyError(f"unknown model {model_name!r}; known: {sorted(registry)}")
    chain = [model_name]
    while len(chain) <= MAX_HOPS and (nxt := registry[chain[-1]].get("fallback")) and nxt not in chain:
        if nxt not in registry:
            raise KeyError(f"model {chain[-1]!r} names unknown fallback {nxt!r}")
        chain.append(nxt)
    kw = dict(user=user, system=system, schema=schema, temperature=temperature, max_tokens=max_tokens, tag=tag,
              timeout_s=timeout_s, log_path=log_path or DEFAULT_LOG, attempts=attempts, extra=extra)
    rl, prev = {"left": RL_RETRIES}, None
    for name in chain:
        r = _call_model(registry[name], model_name, rl, **kw)
        if prev:
            r.usage = {k: v + prev.usage[k] for k, v in r.usage.items()}
            r.cost_usd_plan += prev.cost_usd_plan
            r.cost_usd_nominal += prev.cost_usd_nominal
            r.latency_s += prev.latency_s
            r.attempts += prev.attempts
        if r.ok or r.error_class != "infra":
            break
        prev = r
    return r


def healthy(model_name, timeout=3) -> bool:
    """True for claude entries; for vllm GET {endpoint}/v1/models and require the served model id. False if disabled."""
    m = load_registry()[model_name]
    if m["provider"] == "claude":
        return True
    if m.get("disabled"):
        return False
    try:
        status, body = _http_get(m["endpoint"].rstrip("/") + "/v1/models", timeout)
    except Exception:
        return False
    return status == 200 and isinstance(body, dict) and any(
        isinstance(d, dict) and d.get("id") == m["model_id"] for d in body.get("data") or [])
