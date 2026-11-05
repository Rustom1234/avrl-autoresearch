"""Fallback chains, Claude rate-limit breaker and backoff, concurrency slots, healthy(), error_class."""
import json
import os
import subprocess
import threading
import time
from pathlib import Path

import pytest

import call as call_mod
from call import ContextTooLong, call, healthy
from conftest import cli_json, ok_cli, rate_limited_cli, vllm_body

SCHEMA = {"type": "object", "properties": {"answer": {"type": "integer"}}, "required": ["answer"]}
GOOD = ok_cli(cli_json("", {"answer": 4}, cost=0.002))
REFUSED = ConnectionRefusedError("refused")


def log_lines(path):
    return [json.loads(x) for x in open(path).read().splitlines()]


def model_of(argv):
    return argv[argv.index("--model") + 1]


def breaker_file():
    return Path(os.environ["AR_BREAKER_FILE"])


def set_breaker(until):
    breaker_file().parent.mkdir(parents=True, exist_ok=True)
    breaker_file().write_text(json.dumps({"until_epoch": until, "reason": "test"}))


# ---------------------------------------------------------------- fallback chains

def test_fallback_hop_on_connection_refused_records_served_model(http, claude):
    h = http([REFUSED] * 3)
    c = claude([GOOD])
    r = call("qwen35-9b", "q", schema=SCHEMA, log_path="l.jsonl")
    assert r.ok and r.obj == {"answer": 4} and r.error_class is None
    assert r.requested_model == "qwen35-9b" and r.served_model == "haiku"
    assert (r.model_name, r.model_id, r.provider) == ("haiku", "haiku", "claude")
    assert len(h.calls) == 3 and len(c.calls) == 1 and r.attempts == 4
    assert r.cost_usd_plan == pytest.approx(0.002)
    lines = log_lines("l.jsonl")
    assert [(x["requested_model"], x["served_model"], x["ok"], x["error_class"]) for x in lines] == \
        [("qwen35-9b", "qwen35-9b", False, "infra")] * 3 + [("qwen35-9b", "haiku", True, None)]
    assert [x["attempt"] for x in lines] == [1, 2, 3, 1]
    assert lines[-1]["provider"] == "claude" and lines[0]["provider"] == "vllm"


def test_two_hops_vllm_to_haiku_to_sonnet_and_costs_are_summed(http, claude):
    http([(503, "down")] * 3)
    c = claude([ok_cli(cli_json("internal server error", is_error=True, api_error_status=500, cost=0.01,
                                usage=(1, 1, 1, 1)), rc=1)] * 3
               + [ok_cli(cli_json("", {"answer": 4}, cost=0.02, usage=(10, 20, 30, 40)))])
    r = call("qwen35-9b", "q", schema=SCHEMA, log_path="l.jsonl")
    assert r.ok and r.requested_model == "qwen35-9b" and r.served_model == "sonnet"
    assert [model_of(a[0]) for a in c.calls] == ["haiku"] * 3 + ["sonnet"]
    assert r.attempts == 7 and r.cost_usd_plan == pytest.approx(0.05)
    assert r.usage == {"input": 13, "output": 23, "cache_read": 33, "cache_write": 43}


def test_all_infra_stops_after_two_hops(http, claude, tmp_path, monkeypatch):
    reg = tmp_path / "reg.json"
    chain = ["a", "b", "c", "d"]
    reg.write_text(json.dumps([{"name": n, "provider": "vllm", "model_id": n, "endpoint": f"http://{n}:1",
                                "nominal_price_per_m": None, "max_context_tokens": 1, "roles_hint": "",
                                "fallback": (chain[i + 1] if i < 3 else None)} for i, n in enumerate(chain)]))
    monkeypatch.setattr(call_mod, "REGISTRY_PATH", reg)
    h = http([REFUSED] * 3)
    r = call("a", "q", attempts=1)
    assert [c[0] for c in h.calls] == ["http://a:1/v1/chat/completions", "http://b:1/v1/chat/completions",
                                       "http://c:1/v1/chat/completions"]
    assert not r.ok and r.error_class == "infra" and r.requested_model == "a" and r.served_model == "c"


def test_fallback_cycle_and_unknown_fallback(http, tmp_path, monkeypatch):
    reg = tmp_path / "reg.json"
    base = {"provider": "vllm", "endpoint": "http://x:1", "nominal_price_per_m": None, "max_context_tokens": 1,
            "roles_hint": ""}
    reg.write_text(json.dumps([{**base, "name": "a", "model_id": "a", "fallback": "b"},
                               {**base, "name": "b", "model_id": "b", "fallback": "a"},
                               {**base, "name": "z", "model_id": "z", "fallback": "ghost"}]))
    monkeypatch.setattr(call_mod, "REGISTRY_PATH", reg)
    h = http([REFUSED] * 2)
    r = call("a", "q", attempts=1)
    assert len(h.calls) == 2 and r.served_model == "b"                    # a -> b, never back to a
    with pytest.raises(KeyError, match="ghost"):
        call("z", "q")


@pytest.mark.parametrize("status,body", [(404, {"object": "error", "message": "The model `qwen35-9b` does not exist.",
                                                  "code": 404}), (503, "service unavailable")])
def test_vllm_404_and_503_are_infra_and_hop(http, claude, status, body):
    h = http([(status, body)] * 3)
    c = claude([GOOD])
    r = call("qwen35-9b", "q", schema=SCHEMA, log_path="l.jsonl")
    assert r.ok and r.served_model == "haiku" and len(h.calls) == 3 and len(c.calls) == 1
    assert [x["error_class"] for x in log_lines("l.jsonl")] == ["infra"] * 3 + [None]


@pytest.mark.parametrize("status", [400, 401, 422])
def test_other_vllm_4xx_stay_model_class_and_do_not_hop(http, claude, status):
    h = http([(status, "rejected")] * 3)
    c = claude([])
    r = call("qwen35-9b", "q")
    assert not r.ok and r.error_class == "model" and r.served_model == "qwen35-9b" and len(h.calls) == 3 and c.calls == []


def test_slot_poll_default_is_half_a_second():
    assert call_mod.SLOT_POLL_S == 0.5 and call_mod.BREAKER_POLL_S == 10


def test_no_fallback_on_context_too_long(http, claude):
    h = http([(400, {"error": {"message": "maximum context length is 8192 tokens"}})])
    c = claude([])
    with pytest.raises(ContextTooLong):
        call("qwen35-9b", "q", log_path="l.jsonl")
    assert len(h.calls) == 1 and c.calls == []
    (line,) = log_lines("l.jsonl")
    assert line["error_class"] == "context" and line["served_model"] == "qwen35-9b"


def test_no_fallback_on_schema_failure(http, claude):
    h = http([(200, vllm_body('{"answer": "x"}'))] * 3)
    c = claude([])
    r = call("qwen35-9b", "q", schema=SCHEMA, log_path="l.jsonl")
    assert not r.ok and r.error_class == "model" and r.served_model == "qwen35-9b" and r.requested_model == "qwen35-9b"
    assert len(h.calls) == 3 and c.calls == []
    assert {x["error_class"] for x in log_lines("l.jsonl")} == {"model"}


def test_claude_schema_failure_does_not_hop_to_sonnet(claude):
    c = claude([ok_cli(cli_json("", {"answer": "x"}))] * 3)
    r = call("haiku", "q", schema=SCHEMA)
    assert not r.ok and r.error_class == "model" and r.served_model == "haiku"
    assert [model_of(a[0]) for a in c.calls] == ["haiku"] * 3


def test_error_class_values(http, claude):
    claude([GOOD])
    assert call("haiku", "q", schema=SCHEMA).error_class is None
    claude([ok_cli(cli_json("", {"answer": "x"}))] * 3)
    assert call("sonnet", "q", schema=SCHEMA).error_class == "model"
    claude([subprocess.TimeoutExpired("claude", 1)] * 3)
    assert call("sonnet", "q").error_class == "infra"                    # sonnet has no fallback
    http([(400, {"error": {"message": "maximum context length is 8192 tokens"}})])
    with pytest.raises(ContextTooLong):                                  # raised, so "context" is only in the log
        call("qwen35-9b", "q", log_path="l.jsonl")
    assert log_lines("l.jsonl")[-1]["error_class"] == "context"


def test_error_class_infra_when_whole_chain_down(http, claude):
    http([REFUSED] * 3)
    claude([(0, "garbage", "")] * 3 + [subprocess.TimeoutExpired("claude", 1)] * 3)
    r = call("qwen35-9b", "q")
    assert not r.ok and r.error_class == "infra" and r.served_model == "sonnet"


def test_call_does_not_consult_healthy(http, claude, monkeypatch):
    def boom(*a, **k):
        raise AssertionError("healthy called")
    monkeypatch.setattr(call_mod, "healthy", boom)
    http([(200, vllm_body("hi"))])
    assert call("qwen35-9b", "q").ok


# ---------------------------------------------------------------- breaker and backoff

def test_breaker_in_future_is_slept_off_before_claude(claude, clock):
    set_breaker(clock.now + 25)
    s = claude([ok_cli(cli_json("hi"))])
    assert call("haiku", "q").ok
    assert clock.sleeps == [10, 10, 5] and len(s.calls) == 1


@pytest.mark.parametrize("content", [None, "not json", json.dumps([1]), json.dumps({"until_epoch": "x"})])
def test_breaker_missing_or_garbage_or_past_does_not_wait(claude, clock, content):
    if content is not None:
        breaker_file().parent.mkdir(parents=True, exist_ok=True)
        breaker_file().write_text(content)
    claude([ok_cli(cli_json("hi"))])
    assert call("haiku", "q").ok and clock.sleeps == []


def test_breaker_in_past_does_not_wait_and_vllm_ignores_breaker(claude, http, clock):
    set_breaker(clock.now - 1)
    claude([ok_cli(cli_json("hi"))])
    assert call("haiku", "q").ok and clock.sleeps == []
    set_breaker(clock.now + 500)
    http([(200, vllm_body("hi"))])
    assert call("qwen35-9b", "q").ok and clock.sleeps == []


def test_breaker_rechecked_after_slot_acquired(claude, clock, monkeypatch):
    real_slot = call_mod._claude_slot

    def slot():  # another process trips the breaker while this call queued for a slot
        set_breaker(clock.now + 30)
        return real_slot()
    monkeypatch.setattr(call_mod, "_claude_slot", slot)
    s = claude([ok_cli(cli_json("hi"))])
    assert call("haiku", "q").ok and sum(clock.sleeps) == pytest.approx(30) and len(s.calls) == 1


def test_backoff_schedule(monkeypatch):
    monkeypatch.setattr(call_mod.random, "random", lambda: 1.0)
    assert [call_mod._backoff(i) for i in range(10)] == [60, 120, 240, 480, 960, 1800, 1800, 1800, 1800, 1800]
    monkeypatch.setattr(call_mod.random, "random", lambda: 0.0)
    assert [call_mod._backoff(i) for i in (0, 1, 5, 9)] == [30, 60, 900, 900]
    monkeypatch.undo()
    assert all(30 <= call_mod._backoff(0) <= 60 for _ in range(50))


def test_rate_limit_trips_breaker_backs_off_and_succeeds_without_using_attempts(claude, clock, monkeypatch):
    monkeypatch.setattr(call_mod.random, "random", lambda: 1.0)
    s = claude([rate_limited_cli()] * 3 + [ok_cli(cli_json("fine"))])
    r = call("haiku", "q", attempts=1, log_path="l.jsonl")                # attempts=1: rate-limit retries are separate
    assert r.ok and r.error_class is None and r.attempts == 4 and len(s.calls) == 4
    assert sum(clock.sleeps) == pytest.approx(60 + 120 + 240) and max(clock.sleeps) <= 10
    b = json.loads(breaker_file().read_text())
    assert set(b) == {"until_epoch", "reason"} and "usage limit" in b["reason"]
    assert b["until_epoch"] == pytest.approx(clock.now)                    # last trip (240 s) was slept to its end
    lines = log_lines("l.jsonl")
    assert [x["error_class"] for x in lines] == ["infra"] * 3 + [None] and [x["attempt"] for x in lines] == [1, 2, 3, 4]


def test_breaker_written_before_sleeping(claude, clock, monkeypatch):
    monkeypatch.setattr(call_mod.random, "random", lambda: 1.0)
    seen = []
    claude([rate_limited_cli(), ok_cli(cli_json("fine"))])
    orig = clock.sleep
    monkeypatch.setattr(call_mod.time, "sleep", lambda s: (seen.append(json.loads(breaker_file().read_text())), orig(s)))
    t0 = clock.now
    call("haiku", "q")
    assert seen[0]["until_epoch"] == pytest.approx(t0 + 60)


def test_rate_limit_gives_up_after_8_retries_with_infra_and_budget_is_shared_across_hops(claude, clock, monkeypatch):
    monkeypatch.setattr(call_mod.random, "random", lambda: 1.0)
    s = claude([rate_limited_cli()] * 30)
    r = call("haiku", "q", log_path="l.jsonl")
    # haiku: 1 try + 8 retries; the hop to sonnet gets the exhausted budget: one try, no retries
    assert [model_of(a[0]) for a in s.calls] == ["haiku"] * 9 + ["sonnet"]
    assert not r.ok and r.error_class == "infra" and r.requested_model == "haiku" and r.served_model == "sonnet"
    assert r.attempts == 10
    # 60,120,240,480,960 then 1800 x3 slept between haiku tries, then one 1800 s breaker wait before the sonnet try
    assert sum(clock.sleeps) == pytest.approx(60 + 120 + 240 + 480 + 960 + 1800 * 3 + 1800)
    assert {x["error_class"] for x in log_lines("l.jsonl")} == {"infra"}


@pytest.mark.parametrize("text", ["Rate limit exceeded", "rate_limit_error", "Claude AI usage limit reached|1760000000",
                                  "API Error: 529 overloaded_error", "Too Many Requests", "You exceeded your quota",
                                  "Request failed with status 429", "weekly LIMIT REACHED"])
def test_rate_limit_signatures_detected(claude, clock, text):
    s = claude([ok_cli(cli_json(text, is_error=True, cost=0.0), rc=1), ok_cli(cli_json("fine"))])
    r = call("haiku", "q", attempts=1)
    assert r.ok and len(s.calls) == 2 and sum(clock.sleeps) > 0


@pytest.mark.parametrize("text", ["invalid_request_error: bad schema", "error code 4290 unrelated", "boom"])
def test_non_limit_errors_are_model_class_and_not_slept(claude, clock, text):
    s = claude([ok_cli(cli_json(text, is_error=True), rc=1)])
    r = call("haiku", "q", attempts=1)
    assert not r.ok and r.error_class == "model" and len(s.calls) == 1 and clock.sleeps == []


def test_limit_words_in_successful_answer_are_ignored(claude, clock):
    claude([ok_cli(cli_json("Rate limit and quota explained: error 429, 529, overloaded"))])
    r = call("haiku", "q")
    assert r.ok and clock.sleeps == []


def test_rate_limit_text_on_stderr_without_json_is_detected(claude, clock):
    s = claude([(1, "", "Error: usage limit reached"), ok_cli(cli_json("fine"))])
    assert call("haiku", "q", attempts=1).ok and len(s.calls) == 2 and sum(clock.sleeps) > 0


def test_trip_never_shortens_existing_block(clock):
    call_mod._trip_breaker(600, "long")
    call_mod._trip_breaker(60, "short")
    b = json.loads(breaker_file().read_text())
    assert b["until_epoch"] == pytest.approx(clock.now + 600) and b["reason"] == "long"


# ---------------------------------------------------------------- concurrency slots

@pytest.mark.parametrize("env,limit", [("2", 2), (None, 4)])
def test_slot_semaphore_limits_concurrency(monkeypatch, env, limit):
    if env:
        monkeypatch.setenv("AR_CLAUDE_MAX_CONCURRENT", env)
    monkeypatch.setattr(call_mod, "SLOT_POLL_S", 0.005)
    lock, full = threading.Lock(), threading.Event()
    st = {"now": 0, "max": 0}

    def run(argv, stdin, timeout):
        with lock:
            st["now"] += 1
            st["max"] = max(st["max"], st["now"])
            if st["now"] == limit:
                full.set()
        full.wait(5)                       # hold until `limit` callers are inside at once
        time.sleep(0.05)
        with lock:
            st["now"] -= 1
        return ok_cli(cli_json("x"))

    monkeypatch.setattr(call_mod, "_run_claude", run)
    out = []
    ts = [threading.Thread(target=lambda: out.append(call("haiku", "q", log_path="l.jsonl"))) for _ in range(limit * 3)]
    for t in ts:
        t.start()
    for t in ts:
        t.join(30)
    assert len(out) == limit * 3 and all(r.ok for r in out)
    assert st["max"] == limit


def test_slot_released_on_error_and_locks_live_in_lock_dir(claude):
    claude([subprocess.TimeoutExpired("claude", 1)] * 3)
    r = call("sonnet", "q", attempts=3)
    assert not r.ok
    with call_mod._claude_slot():           # would block forever (guarded sleep asserts) if a slot leaked
        pass
    assert (Path(os.environ["AR_LOCK_DIR"]) / "claude_slot_0.lock").exists()


def test_vllm_does_not_take_a_slot(http, monkeypatch):
    def no_slot():
        raise AssertionError("slot taken")
    monkeypatch.setattr(call_mod, "_claude_slot", no_slot)
    http([(200, vllm_body("hi"))])
    assert call("qwen35-9b", "q").ok


# ---------------------------------------------------------------- healthy()

def test_healthy(monkeypatch):
    seen = []

    def get(url, timeout):
        seen.append((url, timeout))
        return 200, {"object": "list", "data": [{"id": "qwen35-9b"}, {"id": "other"}]}
    monkeypatch.setattr(call_mod, "_http_get", get)
    assert healthy("qwen35-9b") is True and seen == [("http://cherry:8000/v1/models", 3)]
    assert healthy("qwen35-9b", timeout=1) is True and seen[-1] == ("http://cherry:8000/v1/models", 1)
    assert healthy("qwen35-4b") is False                          # served model id not in the listing
    assert healthy("haiku") is True and healthy("sonnet") is True and healthy("opus") is True
    assert len(seen) == 3                                         # claude entries never touch the network
    assert healthy("gptoss-20b") is False and healthy("qwen7b-awq") is False and len(seen) == 3  # disabled


@pytest.mark.parametrize("reply", [(500, "boom"), (200, "not a dict"), (200, {"data": None}), (200, {"nodata": 1}),
                                   (200, {"data": ["qwen35-9b"]}), ConnectionRefusedError("refused"),
                                   TimeoutError("slow")])
def test_healthy_false_on_bad_replies(monkeypatch, reply):
    def get(url, timeout):
        if isinstance(reply, Exception):
            raise reply
        return reply
    monkeypatch.setattr(call_mod, "_http_get", get)
    assert healthy("qwen35-9b") is False


def test_healthy_unknown_model():
    with pytest.raises(KeyError):
        healthy("nope")
