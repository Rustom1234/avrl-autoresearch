import json

import pytest

from budget import over_ceiling, tokens_used


def line(provider, i, o, cr=0, cw=0):
    return json.dumps({"provider": provider, "usage": {"input": i, "output": o, "cache_read": cr, "cache_write": cw}})


@pytest.fixture
def log(tmp_path):
    p = tmp_path / "c.jsonl"
    p.write_text("\n".join([line("claude", 10, 5, 100, 1000), line("claude", 1, 2, 3, 4), line("vllm", 500, 50),
                            json.dumps({"provider": "claude", "usage": None}), json.dumps({"provider": "claude"}),
                            json.dumps([1, 2]), "", "{broken"]) + "\n")
    return p


def test_tokens_used_sums_per_provider(log):
    assert tokens_used(log) == {"claude": {"input": 11, "output": 7, "cache_read": 103, "cache_write": 1004},
                                "vllm": {"input": 500, "output": 50, "cache_read": 0, "cache_write": 0}}


def test_tokens_used_missing_log_is_empty(tmp_path):
    assert tokens_used(tmp_path / "nope.jsonl") == {}
    assert over_ceiling(tmp_path / "nope.jsonl", 0) is False


def test_over_ceiling_counts_all_four_kinds_for_claude_only(log):
    assert sum(tokens_used(log)["claude"].values()) == 1125
    assert over_ceiling(log, 1124) is True
    assert over_ceiling(log, 1125) is False                       # strictly over
    assert over_ceiling(log, 10**6) is False                      # vllm's 550 tokens never count toward the claude ceiling
    assert over_ceiling(log, 549, provider="vllm") is True and over_ceiling(log, 550, provider="vllm") is False
    assert over_ceiling(log, 0, provider="gemini") is False
