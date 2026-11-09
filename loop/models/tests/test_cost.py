import json
import subprocess
import sys
from pathlib import Path

import pytest

from cost import summarize


def line(model, tag, ok=True, inp=10, out=5, plan=0.0, nominal=0.0, lat=1.0, provider="claude"):
    return json.dumps({"ts": "t", "tag": tag, "model_name": model, "provider": provider, "ok": ok, "error": None,
                       "usage": {"input": inp, "output": out, "cache_read": 1, "cache_write": 2},
                       "cost_usd_plan": plan, "cost_usd_nominal": nominal, "latency_s": lat,
                       "prompt_chars": 3, "schema_name": None, "attempt": 1})


@pytest.fixture
def log(tmp_path):
    p = tmp_path / "c.jsonl"
    p.write_text("\n".join([
        line("haiku", "judge:r1", plan=0.01, nominal=0.01, lat=1.0),
        line("haiku", "judge:r2", plan=0.02, nominal=0.02, lat=3.0),
        line("sonnet", "plan:r1", ok=False, plan=0.10, nominal=0.10, lat=2.0),
        line("qwen7b-awq", "judge:r1", nominal=0.001, lat=0.5, provider="vllm"),
        line("qwen7b-awq", "", nominal=0.001, lat=0.5, provider="vllm"),
        "", "{broken json",
    ]) + "\n")
    return p


def test_group_by_model(log):
    s = summarize(log)
    h = s["by_model"]["haiku"]
    assert h["attempts"] == 2 and h["ok"] == 2 and h["input"] == 20 and h["output"] == 10
    assert h["cache_read"] == 2 and h["cache_write"] == 4
    assert h["cost_usd_plan"] == pytest.approx(0.03) and h["mean_latency_s"] == pytest.approx(2.0)
    assert s["by_model"]["sonnet"]["ok"] == 0 and s["by_model"]["sonnet"]["attempts"] == 1
    q = s["by_model"]["qwen7b-awq"]
    assert q["cost_usd_plan"] == 0.0 and q["cost_usd_nominal"] == pytest.approx(0.002)


def test_group_by_tag_prefix(log):
    t = summarize(log)["by_tag"]
    assert set(t) == {"judge", "plan", "(none)"}
    assert t["judge"]["attempts"] == 3 and t["judge"]["cost_usd_plan"] == pytest.approx(0.03)
    assert t["plan"]["cost_usd_plan"] == pytest.approx(0.10) and t["(none)"]["attempts"] == 1


def test_total_and_bad_lines(log):
    s = summarize(log)
    assert s["total"]["attempts"] == 5 and s["bad_lines"] == 1
    assert s["total"]["cost_usd_plan"] == pytest.approx(0.13)
    assert s["total"]["cost_usd_nominal"] == pytest.approx(0.132)


def test_main_prints_table(log):
    out = subprocess.run([sys.executable, str(Path(__file__).resolve().parent.parent / "cost.py"), str(log)],
                         capture_output=True, text=True, check=True).stdout
    assert "haiku" in out and "TOTAL" in out and "judge" in out and "plan$" in out


def fb_line(req, served, tag="judge:x", ok=True, plan=0.0, provider="claude"):
    return json.dumps({"ts": "t", "tag": tag, "model_name": served, "requested_model": req, "served_model": served,
                       "provider": provider, "ok": ok, "error": None, "error_class": None if ok else "infra",
                       "usage": {"input": 10, "output": 5, "cache_read": 0, "cache_write": 0},
                       "cost_usd_plan": plan, "cost_usd_nominal": plan, "latency_s": 1.0, "prompt_chars": 3,
                       "schema_name": None, "attempt": 1})


def test_group_by_served_and_requested_and_count_fallbacks(tmp_path):
    p = tmp_path / "f.jsonl"
    p.write_text("\n".join([
        fb_line("qwen35-9b", "qwen35-9b", ok=False, provider="vllm"),          # refused, then served by haiku
        fb_line("qwen35-9b", "haiku", plan=0.01),
        fb_line("qwen35-9b", "qwen35-9b", provider="vllm"),                    # served normally
        fb_line("haiku", "haiku", plan=0.02),
        fb_line("haiku", "sonnet", plan=0.10),
        fb_line("haiku", "sonnet", plan=0.10),
        line("opus", "old:t", plan=0.5),                                       # pre-fallback line: model_name only
    ]) + "\n")
    s = summarize(p)
    sv, rq = s["by_served_model"], s["by_requested_model"]
    assert {k: v["attempts"] for k, v in sv.items()} == {"qwen35-9b": 2, "haiku": 2, "sonnet": 2, "opus": 1}
    assert {k: v["attempts"] for k, v in rq.items()} == {"qwen35-9b": 3, "haiku": 3, "opus": 1}
    assert sv["sonnet"]["cost_usd_plan"] == pytest.approx(0.20) and rq["haiku"]["cost_usd_plan"] == pytest.approx(0.22)
    assert rq["qwen35-9b"]["ok"] == 2 and sv["qwen35-9b"]["ok"] == 1
    assert s["by_model"] is sv                                                 # back-compat alias
    assert s["fallbacks"] == {"attempts": 3, "by_route": {"qwen35-9b->haiku": 1, "haiku->sonnet": 2}}
    assert s["total"]["attempts"] == 7 and s["total"]["cost_usd_plan"] == pytest.approx(0.73)


def test_old_logs_have_no_fallbacks(log):
    s = summarize(log)
    assert s["fallbacks"] == {"attempts": 0, "by_route": {}}
    assert s["by_requested_model"].keys() == s["by_served_model"].keys()


def test_main_prints_served_requested_and_fallbacks(tmp_path):
    p = tmp_path / "f.jsonl"
    p.write_text(fb_line("haiku", "sonnet", plan=0.1) + "\n")
    out = subprocess.run([sys.executable, str(Path(__file__).resolve().parent.parent / "cost.py"), str(p)],
                         capture_output=True, text=True, check=True).stdout
    assert "served model" in out and "requested model" in out and "haiku->sonnet" in out
