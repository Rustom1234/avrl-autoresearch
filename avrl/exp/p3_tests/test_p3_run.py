"""p3_effort_run: item selection, prompt hash guard, resume, max-calls guard, fallback and env handling (call() is mocked)."""
import json
import os
import shutil

import pytest

import call as call_mod
import p3_effort_run as P
from p3_helpers import fake_result, good_output, make_items

AR3 = P.AR3_DIR
needs_ar3 = pytest.mark.skipif(not (AR3 / "items.jsonl").exists(), reason="AR3 data not present")


class Recorder:
    """Stand-in for call(): records every invocation and returns a canned CallResult."""

    def __init__(self, result=None):
        self.calls, self.result = [], result

    def __call__(self, model, user, **kw):
        self.calls.append((model, user, kw))
        return self.result(model, user, kw) if callable(self.result) else (self.result or fake_result())


def rows(run_dir, level):
    p = P.calls_path(run_dir, level)
    return [json.loads(ln) for ln in p.read_text().splitlines()] if p.exists() else []


# ---------------------------------------------------------------- inputs

@needs_ar3
def test_prompt_hashes_match_the_recorded_ar3_hashes(tmp_path):
    got = P.check_prompt_hashes()
    assert got["system_sha256"].startswith("4282aeb0") and got["schema_sha256"].startswith("cc43db81")
    shutil.copy(AR3 / "critic_hashes.json", tmp_path / "critic_hashes.json")
    rec = json.loads((tmp_path / "critic_hashes.json").read_text())
    rec["system_sha256"] = "0" * 64
    (tmp_path / "critic_hashes.json").write_text(json.dumps(rec))
    with pytest.raises(RuntimeError, match="system_sha256 mismatch"):
        P.check_prompt_hashes(tmp_path)


@needs_ar3
def test_select_items_counts_and_exclusions():
    items, c = P.select_items()
    assert c == {"all_items": 328, "probe_orig": 6, "natural_controls_moved_to_probe": 23, "burn": 75, "test": 224, "selected": 299, "entries": 64}
    assert len(items) == 299 and len({it["item_id"] for it in items}) == 299
    assert {it["split"] for it in items} == {"burn", "test"}
    assert all(it["truth"] in ("clean", "N-big", "N-mid", "N-stale", "V-body", "V-coh") for it in items)  # no natural, no unknown
    assert [it["item_id"] for it in items] == [it["item_id"] for it in P.select_items()[0]]  # fixed order


@needs_ar3
def test_select_items_rejects_a_modified_items_file(tmp_path):
    for f in ("items.jsonl", "items.sha256", "control_labels.json"):
        shutil.copy(AR3 / f, tmp_path / f)
    P.select_items(tmp_path)
    with open(tmp_path / "items.jsonl", "a") as f:
        f.write("\n")
    with pytest.raises(RuntimeError, match="sha256"):
        P.select_items(tmp_path)


@needs_ar3
def test_day0_items_are_twenty_fixed_burn_items_stratified_by_kind():
    items, _ = P.select_items()
    d0 = P.day0_items(items)
    assert len(d0) == 20 and len({it["item_id"] for it in d0}) == 20 and {it["split"] for it in d0} == {"burn"}
    by_kind = {}
    for it in d0:
        by_kind[it["kind"]] = by_kind.get(it["kind"], 0) + 1
    assert by_kind == P.DAY0_ALLOC
    assert [it["item_id"] for it in d0] == [it["item_id"] for it in P.day0_items(list(reversed(items)))]  # independent of input order


def test_day0_items_needs_enough_burn_items():
    with pytest.raises(ValueError):
        P.day0_items(make_items(6))


def test_level_effort_and_valid_output():
    assert P.level_effort("default") is None and P.level_effort("high") == "high"
    with pytest.raises(ValueError):
        P.level_effort("xhigh")
    assert P.valid_output(good_output(0)) and P.valid_output(good_output(100))
    assert not P.valid_output({"problems": []})
    assert not P.valid_output({"problems": [], "judgment_confidence": 101})
    assert not P.valid_output({"problems": [], "judgment_confidence": 70.5})
    assert not P.valid_output({"problems": [], "judgment_confidence": True})
    assert not P.valid_output(None)


# ---------------------------------------------------------------- sweep

def test_run_sweep_calls_haiku_with_the_frozen_prompt_and_effort(tmp_path):
    items, rec = make_items(2), Recorder()
    s = P.run_sweep(items, P.LEVELS, tmp_path, 100, workers=1, call_fn=rec, cli_version="9.9")
    assert s["started"] == s["ok"] == 8 and s["invalid"] == 0
    for model, user, kw in rec.calls:
        assert model == "haiku" and kw["system"] == P.R.SYSTEM and kw["schema"] == P.SCHEMA_OBJ and kw["attempts"] == 3
        assert kw["log_path"] == str(tmp_path / "calllog.jsonl")
    assert sorted(kw["effort"] or "default" for _, _, kw in rec.calls) == sorted(["default", "low", "medium", "high"] * 2)
    assert rec.calls[0][1] == items[0]["text"]
    r = rows(tmp_path, "low")[0]
    assert r["level"] == "low" and r["ok"] and r["split"] == "burn" and r["served_model"] == "haiku" and r["cli_version"] == "9.9"
    assert r["usage"]["output"] == 100 and r["cost_usd"] == 0.002 and r["structured_output"]["judgment_confidence"] == 70
    assert {"item_id", "entry_key", "level", "split", "ok", "structured_output", "usage", "cost_usd", "latency_s", "served_model", "model_id", "ts"} <= set(r)


def test_resume_skips_finished_keys(tmp_path):
    items = make_items(3)
    P.run_sweep(items, ["low", "high"], tmp_path, 100, workers=1, call_fn=Recorder())
    rec = Recorder()
    s = P.run_sweep(items, ["low", "high"], tmp_path, 100, workers=1, call_fn=rec)
    assert rec.calls == [] and s["pending"] == 0 and len(rows(tmp_path, "low")) == 3
    s = P.run_sweep(items, ["low", "high", "medium"], tmp_path, 100, workers=1, call_fn=rec)  # a new level only adds that level
    assert s["started"] == 3 and {kw["effort"] for _, _, kw in rec.calls} == {"medium"}


def test_max_calls_is_a_hard_cap_and_the_rest_stays_pending(tmp_path):
    items, rec = make_items(5), Recorder()
    s = P.run_sweep(items, ["low", "high"], tmp_path, 3, workers=2, call_fn=rec)
    assert len(rec.calls) == 3 and s["started"] == 3 and s["pending"] == 10 and s["planned"] == 3
    s = P.run_sweep(items, ["low", "high"], tmp_path, 4, workers=1, call_fn=rec)
    assert len(rec.calls) == 7 and s["pending"] == 7
    done = P.done_keys(tmp_path, ["low", "high"])
    assert len(done) == 7
    P.run_sweep(items, ["low", "high"], tmp_path, 1000, workers=1, call_fn=rec)
    assert len(rec.calls) == 10  # exactly one call per key over the three invocations


def test_item_major_order_so_a_partial_run_covers_whole_items(tmp_path):
    rec = Recorder()
    P.run_sweep(make_items(4), ["low", "medium", "high"], tmp_path, 6, workers=1, call_fn=rec)
    assert [u for _, u, _ in rec.calls] == ["L001| text 0"] * 3 + ["L001| text 1"] * 3


def test_retries_count_against_the_cap(tmp_path):
    rec = Recorder(lambda m, u, kw: fake_result(attempts=3))
    s = P.run_sweep(make_items(6), ["low"], tmp_path, 4, workers=1, call_fn=rec)
    assert len(rec.calls) == 2 and s["attempts"] == 6  # the second call reaches the attempts cap; no third call starts


def test_fallback_to_sonnet_is_recorded_not_ok_and_retried(tmp_path):
    items = make_items(1)
    P.run_sweep(items, ["high"], tmp_path, 10, workers=1, call_fn=Recorder(fake_result(served="sonnet")))
    r = rows(tmp_path, "high")[0]
    assert r["fallback"] is True and r["ok"] is False and r["served_model"] == "sonnet"
    assert P.done_keys(tmp_path, ["high"]) == set()
    rec = Recorder()
    s = P.run_sweep(items, ["high"], tmp_path, 10, workers=1, call_fn=rec)
    assert s["started"] == 1 and P.done_keys(tmp_path, ["high"]) == {("it00", "high")}


def test_invalid_output_is_recorded_not_ok_and_retried(tmp_path):
    items = make_items(1)
    bad = fake_result(obj={"problems": []})  # no judgment_confidence
    s = P.run_sweep(items, ["low"], tmp_path, 10, workers=1, call_fn=Recorder(bad))
    assert s["invalid"] == 1 and rows(tmp_path, "low")[0]["ok"] is False
    failed = fake_result(ok=False)
    failed.obj = None
    P.run_sweep(items, ["low"], tmp_path, 10, workers=1, call_fn=Recorder(failed))
    assert P.done_keys(tmp_path, ["low"]) == set() and len(rows(tmp_path, "low")) == 2


def test_effort_env_overrides_are_removed_before_any_call(tmp_path, monkeypatch):
    monkeypatch.setenv("CLAUDE_EFFORT", "xhigh")
    monkeypatch.setenv("CLAUDE_CODE_EFFORT_LEVEL", "max")
    seen = []

    def spy(model, user, **kw):
        seen.append({v: os.environ.get(v) for v in P.EFFORT_ENV_VARS})
        return fake_result()

    P.run_sweep(make_items(1), ["low"], tmp_path, 5, workers=1, call_fn=spy)
    assert seen == [{"CLAUDE_EFFORT": None, "CLAUDE_CODE_EFFORT_LEVEL": None}]


def test_dry_run_makes_no_calls_and_writes_nothing(tmp_path):
    rec, run_dir = Recorder(), tmp_path / "p3"
    s = P.run_sweep(make_items(3), ["low", "high"], run_dir, 4, call_fn=rec, dry_run=True)
    assert rec.calls == [] and s["dry_run"] and s["pending"] == 6 and s["planned"] == 4 and not run_dir.exists()


def test_a_crashing_call_stops_the_sweep_and_raises(tmp_path):
    def boom(model, user, **kw):
        raise ValueError("bad effort")

    with pytest.raises(ValueError, match="bad effort"):
        P.run_sweep(make_items(5), ["low"], tmp_path, 5, workers=1, call_fn=boom)
    assert rows(tmp_path, "low") == []


def test_truncated_last_line_is_ignored_on_resume(tmp_path):
    P.run_sweep(make_items(2), ["low"], tmp_path, 5, workers=1, call_fn=Recorder())
    with open(P.calls_path(tmp_path, "low"), "a") as f:
        f.write('{"item_id": "it01", "level": "lo')  # killed mid-write
    assert P.done_keys(tmp_path, ["low"]) == {("it00", "low"), ("it01", "low")}


def test_cli_metadata_capture_reaches_the_row(tmp_path, monkeypatch):
    out = json.dumps({"result": "", "modelUsage": {"claude-haiku-5-5": {}}, "usage": {"output_tokens": 321}})
    monkeypatch.setattr(call_mod, "_run_claude", lambda argv, stdin, timeout: (0, out, ""))
    P.install_capture()
    call_mod._run_claude(["claude", "-p", "--effort", "high"], "x", 1)

    def via_capture(model, user, **kw):
        call_mod._run_claude(["claude", "-p", "--effort", kw["effort"]] if kw["effort"] else ["claude", "-p"], user, 1)
        return fake_result()

    P.run_sweep(make_items(1), ["default", "high"], tmp_path, 5, workers=1, call_fn=via_capture)
    d, h = rows(tmp_path, "default")[0], rows(tmp_path, "high")[0]
    assert h["argv_effort"] == "high" and d["argv_effort"] is None
    assert h["model_id"] == "claude-haiku-5-5" and h["model_ids"] == ["claude-haiku-5-5"] and h["raw_usage"] == {"output_tokens": 321}


def test_main_requires_a_cap_and_the_shared_lock_env(monkeypatch, capsys):
    monkeypatch.delenv("AR_LOCK_DIR", raising=False)
    monkeypatch.delenv("AR_BREAKER_FILE", raising=False)
    with pytest.raises(SystemExit):
        P.main(["--items", "day0"])  # no --max-calls
    with pytest.raises(SystemExit):
        P.main(["--items", "day0", "--max-calls", "5"])  # no lock env
    if (AR3 / "items.jsonl").exists():
        s = P.main(["--items", "day0", "--dry-run", "--run-dir", "/nonexistent/p3"])
        assert s["pending"] == 80 and s["dry_run"]


def test_enter_neutral_cwd_leaves_the_project_directory(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    d = P.enter_neutral_cwd()
    try:
        assert os.getcwd() == os.path.realpath(d) and os.listdir(d) == []
        assert not os.path.exists(os.path.join(d, "CLAUDE.md"))
    finally:
        os.chdir(tmp_path)
        os.rmdir(d)
    P.run_sweep(make_items(1), ["low"], tmp_path / "r", 5, workers=1, call_fn=Recorder())
    assert rows(tmp_path / "r", "low")[0]["cwd"] == str(tmp_path)
