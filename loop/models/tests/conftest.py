import json
import sys
import time
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import call as call_mod  # noqa: E402

_REAL_SLEEP = time.sleep


@pytest.fixture(autouse=True)
def _isolate(monkeypatch, tmp_path):
    """Fresh structured-output cache, cwd in tmp (default calls.jsonl lands there), breaker and slot locks in tmp, no real I/O."""
    call_mod._STRUCT_MODE.clear()
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("AR_BREAKER_FILE", str(tmp_path / "breaker" / "claude_breaker.json"))
    monkeypatch.setenv("AR_LOCK_DIR", str(tmp_path / "locks"))
    monkeypatch.delenv("AR_CLAUDE_MAX_CONCURRENT", raising=False)

    def no_cli(*a, **k):
        raise AssertionError("real _run_claude called")

    def no_http(*a, **k):
        raise AssertionError("real _http_post called")

    def no_get(*a, **k):
        raise AssertionError("real _http_get called")

    monkeypatch.setattr(call_mod, "_run_claude", no_cli)
    monkeypatch.setattr(call_mod, "_http_post", no_http)
    monkeypatch.setattr(call_mod, "_http_get", no_get)

    def guarded_sleep(s):  # a forgotten `clock` fixture must fail fast, not sleep 60 s
        assert s <= 2, f"real time.sleep({s}) in a test; use the clock fixture"
        _REAL_SLEEP(s)

    monkeypatch.setattr(call_mod.time, "sleep", guarded_sleep)


def cli_json(result="", structured=None, *, is_error=False, stop_reason="end_turn", cost=0.001,
             usage=(2, 5, 3, 1100), api_error_status=None):
    """Canned `claude -p --output-format json` stdout. Key names are the ones observed on 2026-10-21."""
    d = {"type": "result", "subtype": "error_during_execution" if is_error else "success", "is_error": is_error,
         "api_error_status": api_error_status, "result": result, "stop_reason": stop_reason,
         "total_cost_usd": cost, "num_turns": 1,
         "usage": {"input_tokens": usage[0], "output_tokens": usage[1], "cache_read_input_tokens": usage[2],
                   "cache_creation_input_tokens": usage[3]},
         "modelUsage": {"claude-haiku-5-5": {"costUSD": cost, "contextWindow": 1000000}}}
    if structured is not None:
        d["structured_output"] = structured
    return json.dumps(d)


class Script:
    """Replays queued replies. Items: (rc, stdout, stderr) / (status, body) tuples, or an Exception to raise."""

    def __init__(self, items):
        self.items, self.calls = list(items), []

    def __call__(self, *args):
        self.calls.append(args)
        item = self.items.pop(0)
        if isinstance(item, Exception):
            raise item
        return item


@pytest.fixture
def claude(monkeypatch):
    def install(items):
        s = Script(items)
        monkeypatch.setattr(call_mod, "_run_claude", s)
        return s
    return install


@pytest.fixture
def http(monkeypatch):
    def install(items):
        s = Script(items)
        monkeypatch.setattr(call_mod, "_http_post", s)
        return s
    return install


def ok_cli(stdout, rc=0):
    return (rc, stdout, "")


def vllm_body(content, prompt_tokens=10, completion_tokens=5):
    return {"choices": [{"message": {"content": content}, "finish_reason": "stop"}],
            "usage": {"prompt_tokens": prompt_tokens, "completion_tokens": completion_tokens}}


class Clock:
    """Fake wall clock: sleep() advances time() and records the request."""

    def __init__(self):
        self.now, self.sleeps = 1_800_000_000.0, []

    def time(self):
        return self.now

    def sleep(self, s):
        self.sleeps.append(s)
        self.now += s


@pytest.fixture
def clock(monkeypatch):
    c = Clock()
    monkeypatch.setattr(call_mod.time, "time", c.time)
    monkeypatch.setattr(call_mod.time, "sleep", c.sleep)
    return c


def rate_limited_cli(text="Claude AI usage limit reached"):
    return ok_cli(cli_json(text, is_error=True, cost=0.0, api_error_status=429), rc=1)
