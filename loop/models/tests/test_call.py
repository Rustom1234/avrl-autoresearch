import json
import subprocess

import pytest

import call as call_mod
from call import ContextTooLong, call, validate, _validate_minimal
from conftest import cli_json, ok_cli, vllm_body

SCHEMA = {"type": "object", "properties": {"answer": {"type": "integer"}}, "required": ["answer"]}


def log_lines(path):
    return [json.loads(x) for x in open(path).read().splitlines()]


# ---------------------------------------------------------------- claude

def test_claude_success_structured_output_and_argv(claude):
    # real shape: with --json-schema the call ends on tool_use but carries structured_output
    s = claude([ok_cli(cli_json('{"answer":4}', {"answer": 4}, stop_reason="tool_use", cost=0.0002462,
                               usage=(2, 52, 0, 1100)))])
    r = call("haiku", "what is 2+2", system="be terse", schema=SCHEMA, tag="t:x")
    assert r.ok and r.obj == {"answer": 4} and r.error is None and r.attempts == 1
    assert r.provider == "claude" and r.model_id == "haiku" and r.model_name == "haiku"
    assert r.usage == {"input": 2, "output": 52, "cache_read": 0, "cache_write": 1100}
    assert r.cost_usd_plan == pytest.approx(0.0002462) and r.cost_usd_nominal == r.cost_usd_plan
    argv, stdin, timeout = s.calls[0]
    assert stdin == "what is 2+2" and "what is 2+2" not in argv      # prompt on stdin, never argv
    assert argv[:4] == ["claude", "-p", "--model", "haiku"]
    assert "--no-session-persistence" in argv and "--strict-mcp-config" in argv
    assert argv[argv.index("--tools") + 1] == "" and argv[argv.index("--setting-sources") + 1] == ""
    assert argv[argv.index("--output-format") + 1] == "json"
    assert json.loads(argv[argv.index("--json-schema") + 1]) == SCHEMA
    assert argv[argv.index("--system-prompt") + 1] == "be terse"
    assert "--bare" not in argv and timeout == 300


def test_claude_no_schema_no_system_flags(claude):
    s = claude([ok_cli(cli_json("hello"))])
    r = call("sonnet", "hi")
    assert r.ok and r.text == "hello" and r.obj is None
    argv = s.calls[0][0]
    assert "--json-schema" not in argv and "--system-prompt" not in argv


def test_claude_result_text_json_without_structured_key(claude):
    claude([ok_cli(cli_json('```json\n{"answer": 7}\n```'))])
    r = call("haiku", "q", schema=SCHEMA)
    assert r.ok and r.obj == {"answer": 7}


def test_claude_is_error_retried_then_fails_with_raw_text(claude):
    s = claude([ok_cli(cli_json("bad request body", is_error=True, cost=0.01), rc=1)] * 3)
    r = call("haiku", "q", schema=SCHEMA, attempts=3)
    assert not r.ok and r.obj is None and r.attempts == 3 and len(s.calls) == 3
    assert "claude CLI error" in r.error and r.text == "bad request body" and r.error_class == "model"
    assert r.cost_usd_plan == pytest.approx(0.03)                    # summed over attempts


def test_claude_malformed_cli_json_retried_then_success(claude):
    claude([(0, "not json at all", ""), ok_cli(cli_json("{}", {"answer": 1}))])
    r = call("haiku", "q", schema=SCHEMA)
    assert r.ok and r.attempts == 2 and r.obj == {"answer": 1}


def test_claude_tool_use_without_structured_is_retryable(claude):
    s = claude([ok_cli(cli_json("", stop_reason="tool_use")), ok_cli(cli_json("fine"))])
    r = call("haiku", "q")
    assert r.ok and r.text == "fine" and r.attempts == 2 and len(s.calls) == 2


def test_schema_failure_then_success_counts_attempts_and_logs(claude):
    claude([ok_cli(cli_json("", {"answer": "four"}, cost=0.002)), ok_cli(cli_json("", {"answer": 4}, cost=0.003))])
    r = call("haiku", "q", schema=SCHEMA, log_path="log.jsonl")
    assert r.ok and r.obj == {"answer": 4} and r.attempts == 2
    assert r.cost_usd_plan == pytest.approx(0.005)
    a, b = log_lines("log.jsonl")
    assert (a["ok"], a["attempt"], b["ok"], b["attempt"]) == (False, 1, True, 2)
    assert "schema validation failed" in a["error"] and b["error"] is None


def test_unparseable_json_exhausts_attempts(claude):
    s = claude([ok_cli(cli_json("sorry, no json"))] * 2)
    r = call("haiku", "q", schema=SCHEMA, attempts=2)
    assert not r.ok and r.attempts == 2 and len(s.calls) == 2
    assert r.text == "sorry, no json" and r.obj is None and "not parseable" in r.error


def test_timeout_is_retried(claude):
    claude([subprocess.TimeoutExpired("claude", 1), ok_cli(cli_json("ok"))])
    r = call("haiku", "q", timeout_s=1)
    assert r.ok and r.attempts == 2


def test_missing_required_key(claude):
    claude([ok_cli(cli_json("", {"other": 1}))] * 3)
    r = call("haiku", "q", schema=SCHEMA)
    assert not r.ok and "missing required key" in r.error


def test_cost_extraction_all_tiers(claude):
    for name, cost in (("haiku", 0.0013), ("sonnet", 0.0235), ("opus", 0.047)):
        claude([ok_cli(cli_json("x", cost=cost))])
        r = call(name, "q")
        assert r.cost_usd_plan == pytest.approx(cost) and r.cost_usd_nominal == pytest.approx(cost)


# ---------------------------------------------------------------- vllm

def test_vllm_success_payload_usage_and_nominal_cost(http):
    s = http([(200, vllm_body('{"answer": 4}', 1000, 200))])
    r = call("qwen7b-awq", "q", system="sys", schema=SCHEMA, temperature=0.0, max_tokens=64)
    assert r.ok and r.obj == {"answer": 4} and r.provider == "vllm" and r.cost_usd_plan == 0.0
    assert r.usage == {"input": 1000, "output": 200, "cache_read": 0, "cache_write": 0}
    assert r.cost_usd_nominal == pytest.approx((1000 * 0.10 + 200 * 0.30) / 1e6)
    url, payload, timeout = s.calls[0]
    assert url == "http://cherry:8000/v1/chat/completions"
    assert payload["model"] == "qwen7b-awq" and payload["max_tokens"] == 64 and payload["temperature"] == 0.0
    assert payload["messages"] == [{"role": "system", "content": "sys"}, {"role": "user", "content": "q"}]
    assert payload["response_format"]["type"] == "json_schema"
    assert payload["response_format"]["json_schema"]["schema"] == SCHEMA
    assert "guided_json" not in payload and "structured_outputs" not in payload


def test_vllm_omits_temperature_and_structured_fields_when_unset(http):
    s = http([(200, vllm_body("plain text"))])
    r = call("qwen7b-awq", "q")
    assert r.ok and r.text == "plain text" and r.obj is None
    payload = s.calls[0][1]
    assert "temperature" not in payload and "response_format" not in payload \
        and "guided_json" not in payload and "structured_outputs" not in payload


def test_vllm_context_error_raises_and_is_not_retried(http):
    msg = {"error": {"message": "This model's maximum context length is 8192 tokens. However, you requested 9000 tokens",
                     "code": 400}}
    s = http([(400, msg)])
    with pytest.raises(ContextTooLong):
        call("qwen7b-awq", "q" * 10, schema=SCHEMA, log_path="log.jsonl")
    assert len(s.calls) == 1
    (line,) = log_lines("log.jsonl")
    assert line["ok"] is False and "ContextTooLong" in line["error"]


def test_vllm_fallback_to_structured_outputs_and_cached(http):
    unknown = {"error": {"message": "Unknown field: response_format is not supported by this server", "code": 400}}
    s = http([(400, unknown), (200, vllm_body('{"answer": 1}')), (200, vllm_body('{"answer": 2}'))])
    r = call("qwen7b-awq", "q", schema=SCHEMA)
    assert r.ok and r.obj == {"answer": 1} and r.attempts == 1
    first, second = s.calls[0][1], s.calls[1][1]
    assert "response_format" in first and "structured_outputs" not in first
    assert second["structured_outputs"] == {"json": SCHEMA} and "response_format" not in second
    r2 = call("qwen7b-awq", "q", schema=SCHEMA)                       # remembered: no wasted request
    assert r2.obj == {"answer": 2} and len(s.calls) == 3
    assert s.calls[2][1]["structured_outputs"] == {"json": SCHEMA} and "response_format" not in s.calls[2][1]
    assert all("guided_json" not in c[1] for c in s.calls)            # guided_json is accepted but ignored by vLLM 0.30


def test_vllm_422_extra_inputs_also_falls_back(http):
    body = {"detail": [{"loc": ["body", "response_format"], "msg": "Extra inputs are not permitted"}]}
    s = http([(422, body), (200, vllm_body('{"answer": 3}'))])
    assert call("qwen7b-awq", "q", schema=SCHEMA).obj == {"answer": 3}
    assert s.calls[1][1]["structured_outputs"] == {"json": SCHEMA} and "guided_json" not in s.calls[1][1]


def test_vllm_extra_is_merged(http):
    extra = {"chat_template_kwargs": {"enable_thinking": False}, "reasoning_effort": "low"}
    s = http([(200, vllm_body('{"answer": 1}'))])
    r = call("qwen7b-awq", "q", schema=SCHEMA, extra=extra)
    assert r.ok
    p = s.calls[0][1]
    assert p["chat_template_kwargs"] == {"enable_thinking": False} and p["reasoning_effort"] == "low"
    assert p["model"] == "qwen7b-awq" and "response_format" in p       # base fields untouched
    assert extra == {"chat_template_kwargs": {"enable_thinking": False}, "reasoning_effort": "low"}  # not mutated


def test_vllm_extra_survives_fallback(http):
    unknown = {"error": {"message": "Unknown field response_format"}}
    s = http([(400, unknown), (200, vllm_body('{"answer": 1}'))])
    call("qwen7b-awq", "q", schema=SCHEMA, extra={"reasoning_effort": "low"})
    assert s.calls[0][1]["reasoning_effort"] == "low" and s.calls[1][1]["reasoning_effort"] == "low"
    assert "structured_outputs" in s.calls[1][1]


def test_extra_can_override_and_is_ignored_for_claude(http, claude):
    s = http([(200, vllm_body("hi"))])
    call("qwen7b-awq", "q", extra={"max_tokens": 7})
    assert s.calls[0][1]["max_tokens"] == 7
    c = claude([ok_cli(cli_json("hi"))])
    r = call("haiku", "q", extra={"reasoning_effort": "low"})
    assert r.ok and "reasoning_effort" not in " ".join(c.calls[0][0]) and c.calls[0][1] == "q"


def test_vllm_http_500_and_connection_error_retried(http):
    s = http([(500, "boom"), ConnectionRefusedError("refused"), (200, vllm_body('{"answer": 5}'))])
    r = call("qwen7b-awq", "q", schema=SCHEMA, log_path="log.jsonl")
    assert r.ok and r.attempts == 3 and len(s.calls) == 3
    errs = [x["error"] for x in log_lines("log.jsonl")]
    assert errs[0].startswith("HTTP 500") and errs[1].startswith("ConnectionRefusedError") and errs[2] is None


def test_vllm_schema_failure_then_success(http):
    http([(200, vllm_body('{"answer": "x"}')), (200, vllm_body('{"answer": 9}', 7, 3))])
    r = call("qwen7b-awq", "q", schema=SCHEMA)
    assert r.ok and r.obj == {"answer": 9} and r.attempts == 2
    assert r.usage["input"] == 17 and r.usage["output"] == 8          # summed over attempts


def test_vllm_reply_without_choices(http):
    http([(200, {"weird": 1})] * 2)
    r = call("qwen7b-awq", "q", attempts=2)
    assert not r.ok and "no choices" in r.error


# ---------------------------------------------------------------- logging, registry, validation

def test_log_line_fields_and_default_path(claude, tmp_path):
    claude([ok_cli(cli_json("", {"answer": 4}, cost=0.5, usage=(1, 2, 3, 4)))])
    call("opus", "abc", system="de", schema={**SCHEMA, "title": "Ans"}, tag="phase1:judge")
    (line,) = log_lines(tmp_path / "calls.jsonl")                     # default: calls.jsonl in cwd
    assert set(line) >= {"ts", "tag", "model_name", "provider", "ok", "error", "usage", "cost_usd_plan",
                         "cost_usd_nominal", "latency_s", "prompt_chars", "schema_name", "attempt"}
    assert line["tag"] == "phase1:judge" and line["model_name"] == "opus" and line["provider"] == "claude"
    assert line["ok"] is True and line["error"] is None and line["prompt_chars"] == 5
    assert line["schema_name"] == "Ans" and line["attempt"] == 1 and line["cost_usd_plan"] == 0.5
    assert line["usage"] == {"input": 1, "output": 2, "cache_read": 3, "cache_write": 4}


def test_log_appends_across_calls(claude):
    claude([ok_cli(cli_json("a")), ok_cli(cli_json("b"))])
    call("haiku", "1", log_path="l.jsonl")
    call("haiku", "2", log_path="l.jsonl")
    assert len(log_lines("l.jsonl")) == 2


def test_unknown_model_name():
    with pytest.raises(KeyError, match="nope"):
        call("nope", "q")


def test_extra_registry_entry_needs_no_code_change(http, monkeypatch, tmp_path):
    reg = tmp_path / "reg.json"
    reg.write_text(json.dumps([{"name": "big14b", "provider": "vllm", "model_id": "qwen14b", "endpoint": "http://x:9/",
                                "nominal_price_per_m": {"input": 0.20, "output": 0.60}, "max_context_tokens": 8192,
                                "roles_hint": "test"}]))
    monkeypatch.setattr(call_mod, "REGISTRY_PATH", reg)
    s = http([(200, vllm_body("hi", 1_000_000, 1_000_000))])
    r = call("big14b", "q")
    assert s.calls[0][0] == "http://x:9/v1/chat/completions" and s.calls[0][1]["model"] == "qwen14b"
    assert r.cost_usd_nominal == pytest.approx(0.80)


def test_shipped_registry():
    reg = call_mod.load_registry()
    assert {"haiku", "sonnet", "opus", "qwen7b-awq", "qwen35-9b", "gptoss-20b", "qwen35-4b", "r1-qwen3-8b",
            "gemma4-12b"} <= set(reg)
    for name, ep, price in (("qwen35-9b", "http://cherry:8000", (0.10, 0.30)), ("gptoss-20b", "http://pine:8000", (0.15, 0.60)),
                            ("qwen35-4b", "http://walnut:8000", (0.05, 0.20)), ("r1-qwen3-8b", "http://willow:8000", (0.10, 0.30))):
        e = reg[name]
        assert e["provider"] == "vllm" and e["model_id"] == name and e["endpoint"] == ep
        assert (e["nominal_price_per_m"]["input"], e["nominal_price_per_m"]["output"]) == price
    assert reg["qwen7b-awq"]["endpoint"] == "http://cherry:8000"
    assert reg["qwen7b-awq"]["nominal_price_per_m"] == {"input": 0.10, "output": 0.30}
    for n in ("haiku", "sonnet", "opus"):
        assert reg[n]["provider"] == "claude" and reg[n]["nominal_price_per_m"] is None
    for e in reg.values():
        assert {"name", "provider", "model_id", "endpoint", "nominal_price_per_m", "max_context_tokens",
                "roles_hint", "fallback"} <= set(e)
        assert e["fallback"] is None or e["fallback"] in reg


def test_shipped_fallback_chains_and_disabled_flags():
    reg = call_mod.load_registry()
    assert {n: e["fallback"] for n, e in reg.items() if not e.get("disabled")} == {
        "qwen35-9b": "haiku", "qwen35-4b": "haiku", "r1-qwen3-8b": "haiku", "gemma4-12b": "haiku",
        "haiku": "sonnet", "sonnet": None, "opus": "sonnet"}
    assert reg["gptoss-20b"]["fallback"] == "haiku"
    g = reg["gemma4-12b"]
    assert g["provider"] == "vllm" and g["endpoint"] == "http://pine:8000"
    assert g["nominal_price_per_m"] == {"input": 0.15, "output": 0.60}
    assert {n for n, e in reg.items() if e.get("disabled")} == {"gptoss-20b", "qwen7b-awq"}
    assert reg["gptoss-20b"]["endpoint"] == "http://pine:8000"


@pytest.mark.parametrize("obj,schema,bad", [
    ({"a": 1}, {"type": "object", "required": ["a"]}, False),
    ({}, {"type": "object", "required": ["a"]}, True),
    (True, {"type": "integer"}, True),                                # bool is not an integer
    (1, {"type": "number"}, False),
    ("x", {"type": ["string", "null"]}, False),
    (None, {"type": ["string", "null"]}, False),
    (3, {"type": ["string", "null"]}, True),
    ({"a": {"b": "x"}}, {"type": "object", "properties": {"a": {"type": "object", "properties": {"b": {"type": "integer"}}}}}, True),
    ([1, 2, "3"], {"type": "array", "items": {"type": "integer"}}, True),
    ([1, 2], {"type": "array", "items": {"type": "integer"}}, False),
    ("c", {"enum": ["a", "b"]}, True),
    ("a", {"enum": ["a", "b"]}, False),
    ([1], {"type": "object"}, True),
])
def test_minimal_validator(obj, schema, bad):
    assert (_validate_minimal(obj, schema) is not None) == bad
    assert (validate(obj, schema) is not None) == bad
