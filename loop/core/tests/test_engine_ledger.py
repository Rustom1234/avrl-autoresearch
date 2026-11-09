"""ledger.py: schema, append-only file, results table, claimed_best."""
import json

import pytest

import ledger as L


def rec(slot, kind="candidate", decision="reject", **kw):
    d = dict(slot=slot, attempt=0, kind=kind, idea_id="c%d" % slot, description="try %d" % slot, edit_class="constant",
             job_id="job%d" % slot, gpu=0, status="ok", official_val_bpb=1.14, adjusted_val_bpb=1.14,
             incumbent_before={"mean": 1.1436, "n": 24}, incumbent_after={"mean": 1.1436, "n": 24}, ts="t")
    d.update(kw, kind=kind, decision=decision)
    return L.new_record(**d)


def test_new_record_has_every_schema_key_and_rejects_unknown_ones():
    r = L.new_record(slot=0, kind="candidate", decision="reject")
    assert set(r) == set(L.FIELDS)
    assert r["model_calls"] == [] and r["flags"] == [] and r["charged"] is True and r["lint"] is None
    for f in ("slot", "attempt", "kind", "idea_id", "description", "edit_class", "model_calls", "job_id", "gpu",
              "status", "official_val_bpb", "adjusted_val_bpb", "sealed_val_bpb", "flags", "z1", "z2", "decision",
              "incumbent_before", "incumbent_after", "lint", "critic", "ts"):
        assert f in r                                         # the field names shared with the analysis agent
    with pytest.raises(KeyError):
        L.new_record(slot=0, kind="candidate", decision="reject", surprise=1)
    with pytest.raises(ValueError):
        L.new_record(slot=0, kind="candidate", decision="maybe")


def test_ledger_is_append_only_jsonl(tmp_path):
    led = L.Ledger(tmp_path / "ledger.jsonl")
    assert led.read() == []
    led.append(rec(0))
    led.append(rec(1, decision="accept"))
    lines = (tmp_path / "ledger.jsonl").read_text().splitlines()
    assert len(lines) == 2 and json.loads(lines[1])["decision"] == "accept"
    assert [r["slot"] for r in L.Ledger(tmp_path / "ledger.jsonl").read()] == [0, 1]


def test_results_table_numbers_come_from_records_and_groups_runs():
    records = [
        rec(0, decision="replicate", official_val_bpb=1.1400, adjusted_val_bpb=1.1370),
        rec(1, kind="replicate", decision="accept", official_val_bpb=1.1390, adjusted_val_bpb=1.1360, idea_id="c0"),
        rec(2, decision="invalid", official_val_bpb=None, adjusted_val_bpb=None, flags=["target_leak", "no_official"],
            description="bad | idea\nwith newline"),
        rec(3, decision="abort", job_id=None, official_val_bpb=None, adjusted_val_bpb=None, charged=False,
            flags=["engine:abort"]),
        rec(4, decision="infra", job_id=None, official_val_bpb=None, adjusted_val_bpb=None, charged=False),
    ]
    inc = {"mean_adjusted": 1.1365, "n": 2}
    t = L.results_table(records, inc)
    rows = [ln for ln in t.splitlines() if ln.startswith("| ") and "| slot |" not in ln]
    assert len(rows) == 3                                     # replicate folded into its candidate, infra left out
    assert "1.1400/1.1390" in rows[0] and "accept" in rows[0] and "| 2 |" in rows[0]
    assert "-0.0071" in rows[0]                               # mean adjusted 1.1365 minus incumbent before 1.1436
    assert "target_leak" in rows[1] and "invalid" in rows[1]
    assert "bad / idea with newline" in rows[1]               # model text is flattened, so no fake table rows
    assert "abort" in rows[2] and "| - |" in rows[2]
    assert "1.1365 over 2 run(s)" in t


def test_claimed_best_is_the_final_incumbents_mean_official():
    assert L.claimed_best([]) is None
    assert L.claimed_best([rec(0, decision="reject")]) is None
    records = [
        rec(0, decision="accept", official_val_bpb=1.130),
        rec(1, decision="replicate", official_val_bpb=1.128),
        rec(2, kind="replicate", decision="accept", official_val_bpb=1.134),
        rec(3, decision="reject", official_val_bpb=1.100),
    ]
    assert L.claimed_best(records) == pytest.approx(1.131)    # mean of 1.128 and 1.134, the last accepted experiment
