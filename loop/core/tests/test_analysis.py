"""Tests for loop/core/analysis.py. The synthetic run is generated here (seeded) into
tests/fixtures/synthetic_run/ (D truly better than A by 0.008) and into a temp dir (no effect)."""
import json
import math
import pathlib
import shutil
import subprocess
import sys

import numpy as np
import pytest

import analysis as an

HERE = pathlib.Path(__file__).resolve().parent
ROOT = HERE.parents[2]   # /homes/rdubash/active-audit, where `python -m loop.core.analysis` resolves
FIXTURE = HERE / "fixtures" / "synthetic_run"
EM_DASH = chr(0x2014)

OFFS = {"0": -0.00306, "1": 0.00474, "2": 0.00166, "3": -0.00334}
BASE_OFF, BASE_SEAL = 1.1436, 1.1500
BASE_GAIN = {"A": 0.004, "B": 0.005, "C": 0.005, "D": 0.004}   # D gets effect_d on top
BIAS = {"A": 0.011, "B": 0.003, "C": 0.003, "D": 0.003}        # selection optimism in the loop's own claim
ROUTING = dict(coder_by_class={"constant": "qwen35-4b", "schedule": "gemma4-12b", "structural": "qwen35-9b"}, repairer="haiku",
               planner_cheap="qwen35-4b", planner="qwen35-4b", analyst="sonnet", voter="gemma4-12b", critic="opus")
REROUTED = {4: {5, 6}, 7: {5, 6}}   # D loop index -> slots whose pre-slot health check found the open model down


def jl(path, rows):
    pathlib.Path(path).write_text("".join(json.dumps(r) + "\n" for r in rows))


def jd(path, obj):
    pathlib.Path(path).write_text(json.dumps(obj))


# ---------------------------------------------------------------- synthetic run generator
def synth_ledger(rng, arm, claimed, slots, degraded, i):
    """Schema-plausible ledger: a few accepts with equal steps, replicates, an integrity flag, an audit, lint failures."""
    n_acc = 5 if arm == "A" else 2
    acc_slots = set(rng.choice(np.arange(1, slots), n_acc, replace=False).tolist())
    step, cur = claimed / n_acc, BASE_OFF
    recs, designated, served = [], 0, 0
    for s in range(slots):
        acc = s in acc_slots
        kind = "replicate" if (arm != "A" and s % 7 == 3 and not acc) else "candidate"
        if arm == "A" and s % 6 == 5 and not acc:
            kind = "audit"   # the tripwire is crossed early in every arm, so A's would-be accepts are audited too (and rejected here)
        flags = ["mismatch"] if (i == 0 and s == 4) else []
        if i == 1 and s == 10:
            kind, flags = "audit", ["tripwire_step"]
        before, after = cur, (cur - step if acc else cur)
        cand = after if acc else cur + abs(rng.normal(0.002, 0.003))
        cls = ["constant", "schedule", "structural"][s % 3]
        model = requested = "sonnet"
        if arm == "D":
            want = ROUTING["coder_by_class"][cls]
            ok = rng.random() < (0.5 if degraded else 0.95)
            if s in REROUTED.get(i, ()):   # straight to the fallback: the engine's counters do not see this call
                model = requested = "haiku"
            else:
                designated += 1
                served += ok
                model, requested = (want if ok else "sonnet"), want
        paid = model in ("sonnet", "haiku")
        calls = [dict(role="coder", requested_model=requested, served_model=model,
                      cost_usd_plan=0.01 if paid else 0.0, cost_usd_nominal=0.03 if paid else 0.0,
                      tokens_in=3000, tokens_out=500, latency_s=12.0)]
        if arm in "CD" and s % 8 == 7:
            calls.append(dict(role="critic", requested_model="opus", served_model="opus", cost_usd_plan=0.1,
                              cost_usd_nominal=0.2, tokens_in=5000, tokens_out=400, latency_s=20.0))
        recs.append(dict(slot=s, attempt=0, kind=kind, idea_id=f"i{s}", description="d", edit_class=cls,
                         model_calls=calls, job_id=f"j{s}", gpu=s % 4, status="ok", official_val_bpb=cand + OFFS[str(s % 4)],
                         adjusted_val_bpb=cand, sealed_val_bpb=None, flags=flags, z1=1.0, z2=None,
                         decision="accept" if acc else "reject", incumbent_before=dict(mean=before, n=24),
                         incumbent_after=dict(mean=after, n=1), lint=dict(ok=True, hard=[], warnings=[]), critic=None, ts=s))
        cur = after
        if arm != "A" and s == 2:
            recs.append(dict(slot=s, attempt=1, kind="lint_fail", job_id=None, status="lint_fail", flags=[], decision="skip",
                             lint=dict(ok=False, hard=[dict(rule="parse", line=1, message="x")]), model_calls=[]))
    return recs, cur, designated, served


def make_run(root, effect_d, seed, n_slots=20):
    root = pathlib.Path(root)
    rng = np.random.default_rng(seed)
    (root / "confirmation").mkdir(parents=True)
    jd(root / "anchor.json", dict(mean=BASE_OFF, n=24, gpu_offsets=OFFS, values=[BASE_OFF] * 24))
    jd(root / "plan.json", dict(sigma_used=0.006736, n_slots=n_slots))
    jd(root / "routing_table_frozen.json", ROUTING)
    jl(root / "sentinels.jsonl", [dict(ts=i, job_id=f"s{i}", gpu=i % 4, official_val_bpb=BASE_OFF + OFFS[str(i % 4)] + rng.normal(0, 0.0059),
                                       status="ok", slot_count_at_submit=12 * i) for i in range(12)])
    true_gain, loops = {}, []
    for arm, n in (("A", 8), ("B", 3), ("C", 3), ("D", 8)):
        for i in range(n):
            lid = f"{arm}_{i}"
            true = max(0.0, BASE_GAIN[arm] + (effect_d if arm == "D" else 0.0) + rng.normal(0, 0.002))
            slots = 9 if (arm == "A" and i == 7) else n_slots   # one truncated, invalid run
            degraded = arm == "D" and i < 2
            recs, cur, des, srv = synth_ledger(rng, arm, true + BIAS[arm], slots, degraded, i)
            d = root / lid
            d.mkdir()
            jl(d / "ledger.jsonl", recs)
            jd(d / "state.json", dict(loop_id=lid, arm=arm, slot_index=slots, n_slots=n_slots, status="done",
                                      incumbent=dict(path="x", sha256="0", mean_adjusted=cur, n=1, raw_values=[], adjusted_values=[], gpus=[], description=""),
                                      claimed_best_official=cur, served_counts={"sonnet": 1}, designated_open_calls=des,
                                      open_served_calls=srv, started=0, updated=1))
            true_gain[lid] = true
            loops.append(lid)
    jobs = [(lid, s) for lid in loops for s in (1001, 1002, 1003, 1004)]
    rng.shuffle(jobs)
    rows = []
    for w in range(len(jobs) // 4):
        wave_fx = rng.normal(0, 0.003)   # shared by the whole wave: the reason for same-wave anchors
        for j, (lid, s) in enumerate(jobs[4 * w:4 * w + 4] + [(None, 0)]):
            g, t = j % 4, true_gain.get(lid, 0.0)
            rows.append(dict(wave=w, loop_id=lid, arm=lid[0] if lid else None, seed=s, kind="confirm" if lid else "anchor",
                             official_val_bpb=BASE_OFF + OFFS[str(g)] + wave_fx + rng.normal(0, 0.004) - t,
                             sealed_val_bpb=BASE_SEAL + 0.5 * OFFS[str(g)] + wave_fx + rng.normal(0, 0.004) - t,
                             gpu=g, status="ok", job_id=f"c{w}_{j}"))
    jl(root / "confirmation" / "results.jsonl", rows)
    return root


@pytest.fixture(scope="module")
def effect_dir():
    if FIXTURE.exists():
        shutil.rmtree(FIXTURE)
    return make_run(FIXTURE, effect_d=0.008, seed=11)


@pytest.fixture(scope="module")
def null_dir(tmp_path_factory):
    return make_run(tmp_path_factory.mktemp("null") / "run", effect_d=0.0, seed=3)


@pytest.fixture(scope="module")
def effect(effect_dir):
    return an.analyze_run(effect_dir, seed=1)


@pytest.fixture(scope="module")
def null(null_dir):
    return an.analyze_run(null_dir, seed=1)


# ---------------------------------------------------------------- table
TABLE_COLS = ["arm", "loop_id", "slots_completed", "valid", "accepts", "replicates", "audits", "integrity_flags", "tripwires",
              "lint_failures", "audit_rejects", "audit_accepts", "would_be_keeps", "audit_reject_rate", "claimed_best_official", "confirmed_official_mean", "confirmed_sealed_mean", "improvement",
              "gap_sel", "gap_shard", "bad_accepts", "model_usd_plan", "model_usd_nominal", "model_calls",
              "served_fraction", "served_fraction_ledger", "degraded"]


def test_table_columns_and_counts(effect):
    tab = effect[0]
    assert set(TABLE_COLS) <= set(tab.columns)
    assert len(tab) == 22
    assert tab.arm.value_counts().to_dict() == {"A": 8, "D": 8, "B": 3, "C": 3}
    assert not tab.valid[tab.loop_id == "A_7"].item() and tab.valid.sum() == 21
    assert tab.n_confirm.eq(4).all()
    assert tab.loc[tab.arm == "D", "degraded"].sum() == 2 and not tab.loc[tab.arm != "D", "degraded"].any()
    d = tab[tab.arm == "D"].set_index("loop_id")
    assert d.index[d.degraded].tolist() == ["D_0", "D_1"]
    rer, plain = d.loc[["D_4", "D_7"]], d.drop(["D_4", "D_7"])
    assert (rer.served_fraction_ledger < rer.served_fraction).all()   # rerouted slots: only the ledger sees them
    assert (rer.served_fraction_ledger >= 0.8).all()                  # two reroutes in 20 slots do not degrade a loop
    assert plain.served_fraction_ledger.to_numpy() == pytest.approx(plain.served_fraction.to_numpy())
    assert effect[1]["meta"]["routing_table"].endswith("routing_table_frozen.json")
    assert tab.loc[tab.arm != "D", "served_fraction"].isna().all()
    assert tab.loc[tab.arm == "D", "served_fraction"].between(0, 1).all()
    assert tab.loc[tab.arm != "D", "served_fraction_ledger"].isna().all()
    assert tab.loc[tab.arm == "D", "served_fraction_ledger"].between(0, 1).all()
    assert (tab.model_usd_nominal >= tab.model_usd_plan).all() and (tab.model_calls > 0).all()
    assert tab.improvement.notna().all() and tab.gap_sel.notna().all() and tab.gap_shard.notna().all()
    assert tab.loc[tab.arm == "A", "lint_failures"].eq(0).all() and tab.loc[tab.arm != "A", "lint_failures"].eq(1).all()
    assert tab.loc[tab.loop_id == "A_0", "integrity_flags"].item() == 1
    assert tab.loc[tab.loop_id == "B_1", "tripwires"].item() == 1   # the audit record


def test_bad_accepts_and_selection_gap_follow_the_synthetic_design(effect):
    tab = effect[0]
    v = tab[tab.valid]
    assert v[v.arm == "A"].bad_accepts.mean() > v[v.arm == "D"].bad_accepts.mean()
    assert v[v.arm == "A"].gap_sel.mean() > v[v.arm == "D"].gap_sel.mean()   # optimism bias 0.011 vs 0.003
    assert effect[1]["h2_checks"]["A_larger_than_each_verified"]


# ---------------------------------------------------------------- hand-built run with known numbers
def write_loop(run, lid, arm, slot_index, recs, **state):
    d = pathlib.Path(run) / lid
    d.mkdir(parents=True)
    jl(d / "ledger.jsonl", recs)
    jd(d / "state.json", dict(loop_id=lid, arm=arm, slot_index=slot_index, n_slots=20, status="done", **state))


def conf_row(wave, lid, kind, off, seal, status="ok"):
    return dict(wave=wave, loop_id=lid, arm=lid[0] if lid else None, seed=1001, kind=kind, official_val_bpb=off,
                sealed_val_bpb=seal, gpu=0, status=status, job_id="j")


def test_hand_built_improvement_gaps_and_flags(tmp_path):
    run = tmp_path / "run"
    (run / "confirmation").mkdir(parents=True)
    jd(run / "anchor.json", dict(mean=1.1400, n=24, gpu_offsets={}))
    write_loop(run, "A_0", "A", 20, [], claimed_best_official=1.1300)
    jl(run / "confirmation" / "results.jsonl", [
        conf_row(1, None, "anchor", 1.1420, 1.1500),
        conf_row(1, "A_0", "confirm", 1.1350, 1.1430),
        conf_row(2, None, "anchor", 1.1380, 1.1460),
        conf_row(2, "A_0", "confirm", 1.1310, 1.1400),
        conf_row(2, "A_0", "confirm", 9.0, 9.0, status="crash"),   # unusable, ignored
        conf_row(2, "Z_9", "confirm", 1.0, 1.0)])                  # unknown loop, ignored
    tab = an.analyze_run(run)[0]
    r = tab.iloc[0]
    assert r.n_confirm == 2
    assert r.confirmed_official_mean == pytest.approx(1.1330) and r.confirmed_sealed_mean == pytest.approx(1.1415)
    assert r.improvement == pytest.approx(0.0065)                    # mean of 0.0070 and 0.0060, same-wave anchors
    assert r.confirmed_gain_official == pytest.approx(0.0070)
    assert r.claimed_gain == pytest.approx(0.0100)
    assert r.gap_sel == pytest.approx(0.0030)                        # claimed gain minus confirmed gain: positive = optimism
    assert r.gap_shard == pytest.approx(-0.0005)                     # (off - sealed) minus the anchor's (off - sealed), averaged


def test_ledger_features_on_known_records():
    inc = lambda b, a: dict(incumbent_before=dict(mean=b, n=24), incumbent_after=dict(mean=a, n=1))   # noqa: E731
    call = lambda p, n: dict(role="coder", served_model="sonnet", cost_usd_plan=p, cost_usd_nominal=n)   # noqa: E731
    recs = [
        dict(kind="candidate", decision="accept", job_id="1", adjusted_val_bpb=1.138, **inc(1.140, 1.138)),     # delta 0.002: bad
        dict(kind="candidate", decision="accept", job_id="2", adjusted_val_bpb=1.128, **inc(1.138, 1.128)),     # delta 0.010: fine
        dict(kind="replicate", decision="accept", job_id="3", adjusted_val_bpb=1.124, **inc(1.128, 1.126)),     # delta 0.002 on the mean: bad
        dict(kind="audit", decision="reject", job_id="4", flags=["tripwire_step"]),
        dict(kind="candidate", decision="reject", job_id="5", flags=["mismatch", "canary_failed"],
             model_calls=[call(0.01, 0.02), call(0.03, None)]),
        dict(kind="lint_fail", decision="skip", job_id=None, lint=dict(ok=False, hard=[1])),
        dict(kind="candidate", decision="reject", job_id="6", lint=dict(ok=True, hard=[])),
    ]
    f = an.ledger_features(recs, an.SIGMA_USED)
    assert (f["accepts"], f["bad_accepts"], f["replicates"], f["audits"], f["tripwires"]) == (3, 2, 1, 1, 1)
    assert (f["integrity_flags"], f["lint_failures"], f["model_calls"]) == (2, 1, 2)
    assert f["model_usd_plan"] == pytest.approx(0.04) and f["model_usd_nominal"] == pytest.approx(0.02)


def test_valid_and_degraded_boundaries(tmp_path):
    run = tmp_path / "run"
    run.mkdir()
    write_loop(run, "D_0", "D", 12, [], designated_open_calls=10, open_served_calls=8)    # exactly 0.8: not degraded; 12 slots: valid
    write_loop(run, "D_1", "D", 11, [], designated_open_calls=100, open_served_calls=79)  # 0.79: degraded; 11 slots: invalid
    write_loop(run, "D_2", "D", 20, [], designated_open_calls=0, open_served_calls=0)     # nothing designated: not degraded
    write_loop(run, "A_0", "A", 20, [], designated_open_calls=5, open_served_calls=0)     # only D can be degraded
    tab = an.analyze_run(run)[0].set_index("loop_id")
    assert tab.valid.to_dict() == {"A_0": True, "D_0": True, "D_1": False, "D_2": True}
    assert tab.degraded.to_dict() == {"A_0": False, "D_0": False, "D_1": True, "D_2": False}
    assert tab.loc["D_0", "served_fraction"] == pytest.approx(0.8) and math.isnan(tab.loc["D_2", "served_fraction"])
    assert tab.loc["D_0", "served_fraction_ledger"] == pytest.approx(0.8)   # no routing table in this run: state counters
    assert math.isnan(tab.loc["D_2", "served_fraction_ledger"]) and math.isnan(tab.loc["A_0", "served_fraction_ledger"])


# ---------------------------------------------------------------- audit columns
def test_audit_columns_on_known_records():
    recs = [
        dict(kind="candidate", decision="accept", job_id="1"),
        dict(kind="replicate", decision="accept", job_id="2"),
        dict(kind="candidate", decision="audit", job_id="3", flags=["tripwire_gain"]),   # the trigger, not an audit run
        dict(kind="audit", decision="reject", job_id="4"),
        dict(kind="audit", decision="Reject:noise", job_id="5"),
        dict(kind="audit", decision="accept", job_id="6"),
        dict(kind="audit", decision="invalid", job_id="7"),                              # an audit, neither accepted nor rejected
        dict(kind="candidate", decision="reject", job_id="8"),
    ]
    f = an.ledger_features(recs, an.SIGMA_USED)
    assert (f["audits"], f["audit_rejects"], f["audit_accepts"], f["accepts"]) == (4, 2, 1, 3)
    none = an.ledger_features([dict(kind="candidate", decision="reject", job_id="1")], an.SIGMA_USED)
    assert (none["audits"], none["audit_rejects"], none["audit_accepts"]) == (0, 0, 0)


def test_audit_columns_in_table_and_report(tmp_path):
    run = tmp_path / "run"
    run.mkdir()
    audit = lambda d: dict(kind="audit", decision=d, job_id="a")   # noqa: E731
    write_loop(run, "A_0", "A", 20, [dict(kind="candidate", decision="accept", job_id="1")] + [audit("reject")] * 3 + [audit("accept")])
    write_loop(run, "A_1", "A", 20, [dict(kind="candidate", decision="accept", job_id="1")] * 2)   # no audits: rate undefined
    t = an.analyze_run(run)[0].set_index("loop_id")
    assert t.loc["A_0", ["audits", "audit_rejects", "audit_accepts", "accepts", "would_be_keeps"]].tolist() == [4, 3, 1, 2, 5]
    assert t.loc["A_0", "audit_reject_rate"] == pytest.approx(0.75)
    assert t.loc["A_1", ["audits", "audit_rejects", "audit_accepts", "would_be_keeps"]].tolist() == [0, 0, 0, 2]
    assert math.isnan(t.loc["A_1", "audit_reject_rate"])
    tab, res = an.analyze_run(run)[:2]
    report = an.build_report(tab, res, tmp_path / "none.md")
    assert "| arm | runs | valid |" in report and "audit_reject_rate" in report and "would_be_keeps" in report
    assert "replication the control would not otherwise have had" in report and "noise keeps" in report
    assert EM_DASH not in report


def test_audit_columns_in_synthetic_run(effect):
    tab = effect[0]
    assert (tab.would_be_keeps == tab.accepts + tab.audit_rejects).all()
    assert (tab.audit_rejects + tab.audit_accepts == tab.audits).all()    # every synthetic audit is accepted or rejected
    assert (tab.audit_reject_rate.dropna() == (tab.audit_rejects / tab.audits)[tab.audits > 0]).all()
    assert tab.audit_reject_rate.isna().equals(tab.audits == 0)
    a = tab[tab.arm == "A"]
    assert a.audit_rejects.sum() > 0 and (a.would_be_keeps > a.accepts).any()   # the control gets audits that reject
    assert (a.audit_rejects >= 1).sum() >= 6


# ---------------------------------------------------------------- ledger-derived served fraction
TABLE = dict(coder_by_class={"constant": "qwen35-4b", "schedule": "gemma4-12b", "structural": "haiku"}, repairer="haiku",
             planner_cheap="qwen35-4b", planner="qwen35-4b", analyst="sonnet", voter="gemma4-12b", critic="opus")


def call(role, served, requested=None):
    return dict(role=role, requested_model=requested or served, served_model=served, cost_usd_plan=0.0, cost_usd_nominal=0.0)


def test_designated_model_mapping():
    dm = an.designated_model
    assert [dm(TABLE, "coder", c) for c in ("constant", "schedule", "structural")] == ["qwen35-4b", "gemma4-12b", "haiku"]
    assert dm(TABLE, "coder", "other") is None and dm(TABLE, "coder", None) is None   # engine: no "coder" key means sonnet
    assert dm(dict(TABLE, coder="qwen35-9b"), "coder", "other") == "qwen35-9b"          # engine falls back to the "coder" key
    assert dm(TABLE, "planner_refresh", None) == "qwen35-4b"                            # planner_refresh maps to planner_cheap
    assert dm(dict(TABLE, planner_refresh="gemma4-12b"), "planner_refresh", None) == "gemma4-12b"
    assert dm({"planner": "qwen35-9b"}, "planner_refresh", None) == "qwen35-9b"          # and on to planner, as in the engine
    assert [dm(TABLE, r, None) for r in ("repairer", "analyst", "voter")] == ["haiku", "sonnet", "gemma4-12b"]
    assert an.OPEN_ROLES == ("coder", "repairer", "planner_refresh", "analyst", "voter")


def test_ledger_served_counts_reroutes_and_only_open_designated_roles():
    recs = [
        dict(edit_class="constant", model_calls=[call("planner_backlog", "sonnet"), call("coder", "haiku", "haiku"),   # rerouted
                                                 call("voter", "gemma4-12b"), call("voter", "gemma4-12b"), call("voter", "haiku")]),
        dict(edit_class="schedule", model_calls=[call("coder", "gemma4-12b"), call("critic", "opus")]),
        dict(edit_class="structural", model_calls=[call("coder", "haiku")]),                    # designated haiku: not counted
        dict(edit_class="constant", model_calls=[call("repairer", "haiku"), call("analyst", "sonnet")]),   # closed designated
        dict(edit_class=None, model_calls=[call("coder", "qwen35-4b")]),                        # no table entry for the class
        dict(edit_class="constant"),                                                              # no model_calls key
    ]
    assert an.ledger_served(recs, TABLE) == (5, 3)   # designated: constant coder + 3 voters + schedule coder; served: 2 voters + coder
    assert an.ledger_served(recs, dict(TABLE, repairer="qwen35-9b", analyst="gemma4-12b")) == (7, 3)
    assert an.ledger_served([], TABLE) == (0, 0)


def test_degraded_rule_uses_the_ledger_fraction(tmp_path):
    run = tmp_path / "run"
    run.mkdir()
    jd(run / "routing_table_frozen.json", TABLE)
    good = [dict(edit_class="constant", model_calls=[call("coder", "qwen35-4b")]) for _ in range(8)]
    reroute = lambda n: [dict(edit_class="constant", model_calls=[call("coder", "haiku", "haiku")]) for _ in range(n)]   # noqa: E731
    # the state counters see only the calls actually sent to an open model, so they say 100 percent in every loop
    write_loop(run, "D_0", "D", 20, good + reroute(2), designated_open_calls=8, open_served_calls=8)   # 8 of 10 = 0.8: not degraded
    write_loop(run, "D_1", "D", 20, good + reroute(3), designated_open_calls=8, open_served_calls=8)   # 8 of 11: degraded
    write_loop(run, "D_2", "D", 20, reroute(2), designated_open_calls=0, open_served_calls=0)          # 0 of 2: degraded, state undefined
    write_loop(run, "D_3", "D", 20, [dict(edit_class="structural", model_calls=[call("coder", "haiku")])],
               designated_open_calls=3, open_served_calls=0)   # nothing designated open in the ledger: undefined, not degraded
    write_loop(run, "A_0", "A", 20, reroute(5), designated_open_calls=5, open_served_calls=0)         # only D can be degraded
    tab, res = an.analyze_run(run)[:2]
    t = tab.set_index("loop_id")
    assert t.served_fraction_ledger["D_0"] == pytest.approx(0.8) and t.served_fraction["D_0"] == 1.0
    assert t.served_fraction_ledger["D_1"] == pytest.approx(8 / 11) and t.served_fraction["D_1"] == 1.0
    assert t.served_fraction_ledger["D_2"] == 0.0 and math.isnan(t.served_fraction["D_2"])
    assert math.isnan(t.served_fraction_ledger["D_3"]) and t.served_fraction["D_3"] == 0.0
    assert t.degraded.to_dict() == {"A_0": False, "D_0": False, "D_1": True, "D_2": True, "D_3": False}
    assert res["h4"]["n_D_degraded"] == 2 and res["h4"]["served_fraction_ledger_mean"] == pytest.approx((0.8 + 8 / 11 + 0.0) / 3)
    assert res["h4"]["served_fraction_mean"] == pytest.approx((1.0 + 1.0 + 0.0) / 3)
    assert str(run / "routing_table_frozen.json") in an.build_report(tab, res, tmp_path / "none.md")


def test_routing_table_resolution_order_and_state_counter_fallback(tmp_path):
    runs = tmp_path / "runs"
    run = runs / "R"
    (runs / "calibration").mkdir(parents=True)
    run.mkdir()
    named = lambda m: dict(TABLE, voter=m)   # noqa: E731
    jd(runs / "calibration" / "routing_table.json", named("calib"))
    plan = {}
    assert an.find_routing(run, plan) == (named("calib"), str(runs / "calibration" / "routing_table.json"))   # last resort
    jd(run / "routing_table_frozen.json", named("frozen"))
    assert an.find_routing(run, plan)[0]["voter"] == "frozen"
    jd(tmp_path / "cfg_a.json", dict(arm="A", routing=None))
    jd(tmp_path / "cfg_d.json", dict(arm="D", routing=str(tmp_path / "cfg_table.json")))
    jd(tmp_path / "cfg_table.json", named("config"))
    plan = dict(loops=[dict(arm="A", config=str(tmp_path / "cfg_a.json")), dict(arm="D", config=str(tmp_path / "cfg_d.json"))])
    assert an.find_routing(run, plan) == (named("config"), str(tmp_path / "cfg_table.json"))   # a D loop config comes first
    jd(tmp_path / "cfg_d.json", dict(arm="D", routing=named("inline")))
    assert an.find_routing(run, plan)[0]["voter"] == "inline"
    jd(tmp_path / "cfg_d.json", dict(arm="D", routing=str(tmp_path / "missing.json")))   # unreadable: next candidate
    assert an.find_routing(run, plan)[0]["voter"] == "frozen"
    (run / "routing_table_frozen.json").unlink()
    (runs / "calibration" / "routing_table.json").unlink()
    assert an.find_routing(run, plan) == (None, None)
    # without any table the ledger column equals the state fraction and the report says so
    write_loop(run, "D_0", "D", 20, [dict(edit_class="constant", model_calls=[call("coder", "haiku", "haiku")])],
               designated_open_calls=10, open_served_calls=7)
    tab, res = an.analyze_run(run)[:2]
    assert tab.served_fraction_ledger.item() == pytest.approx(0.7) == tab.served_fraction.item() and tab.degraded.item()
    assert res["meta"]["routing_table"] is None
    assert "falls back to the state counters" in an.build_report(tab, res, tmp_path / "none.md")


# ---------------------------------------------------------------- permutation test, interval, decision rule
def test_exact_permutation_p_value_known_cases():
    t = an.perm_test([0, 0, 0], [1, 1, 1], np.random.default_rng(0))
    assert t["method"] == "exact" and t["n_perm"] == 20 and t["p"] == pytest.approx(2 / 20) and t["estimate"] == 1.0
    t = an.perm_test([1, 2, 3, 4], [1, 2, 3, 4], np.random.default_rng(0))
    assert t["p"] == pytest.approx(1.0)
    assert an.perm_test([1.0], [], np.random.default_rng(0))["p"] != an.perm_test([1.0], [], np.random.default_rng(0))["p"]   # NaN


def test_eight_versus_eight_enumerates_12870_splits():
    t = an.perm_test(np.arange(8.0), np.arange(10.0, 18.0), np.random.default_rng(0))   # total separation
    assert t["method"] == "exact" and t["n_perm"] == 12870 and t["p"] == pytest.approx(2 / 12870)


def test_monte_carlo_branch_is_seeded_and_valid():
    rng = np.random.default_rng(0)
    x, y = rng.normal(0, 1, 10), rng.normal(0, 1, 10)   # C(20,10) = 184756 > limit
    t1, t2 = an.perm_test(x, y, np.random.default_rng(5)), an.perm_test(x, y, np.random.default_rng(5))
    assert t1["method"] == "monte-carlo" and t1["n_perm"] == an.N_MC + 1
    assert t1["p"] == t2["p"] and 0 < t1["p"] <= 1
    exact_like = an.perm_test(x[:8], y[:8], np.random.default_rng(5))["p"]
    assert abs(an.perm_test(x, y, np.random.default_rng(6))["p"] - t1["p"]) < 0.05 and 0 < exact_like <= 1


def test_interval_by_inversion_behaves():
    rng = np.random.default_rng(2)
    x, y = rng.normal(0, 0.003, 8), rng.normal(0.01, 0.003, 8)
    t = an.perm_test(x, y, np.random.default_rng(0), interval=True)
    assert t["lo"] < t["estimate"] < t["hi"] and t["lo"] > 0                # clear effect: interval excludes 0
    assert t["p"] < 0.10
    assert (t["hi"] - t["lo"]) > 0.002
    n = an.perm_test(x, rng.normal(0.0, 0.003, 8), np.random.default_rng(0), interval=True)
    assert n["p"] > 0.10 and n["lo"] < 0 < n["hi"]                           # no effect: interval contains 0
    wide = an.perm_test(x, y, np.random.default_rng(0), interval=True, alpha=0.01)
    assert wide["lo"] <= t["lo"] and wide["hi"] >= t["hi"]                   # higher coverage means wider


@pytest.mark.parametrize("p,est,want", [(0.01, 0.004, "D better"), (0.01, -0.004, "A better"), (0.01, 0.002, "inconclusive"),
                                        (0.06, 0.02, "inconclusive"), (0.05, 0.02, "inconclusive"), (0.01, 0.003, "inconclusive"),
                                        (math.nan, 0.02, "not evaluable")])
def test_h1_decision_rule(p, est, want):
    assert an.h1_decision(p, est) == want


# ---------------------------------------------------------------- the two synthetic scenarios
def test_effect_scenario_is_detected(effect):
    h1 = effect[1]["h1"]
    assert h1["n_x"] == 7 and h1["n_y"] == 8 and h1["method"] == "exact" and h1["n_perm"] == 6435   # A_7 is invalid
    assert h1["p"] < 0.01 and 0.004 < h1["estimate"] < 0.012 and h1["decision"] == "D better"
    assert 0 < h1["lo"] < h1["estimate"] < h1["hi"]
    itt = effect[1]["h1_sensitivity"]["itt_all_runs"]
    assert itt["n_x"] == 8 and itt["n_perm"] == 12870 and itt["p"] < 0.01
    assert 0.003 < itt["ols_slots_adjusted_estimate"] < 0.013


def test_null_scenario_is_not_detected(null):
    h1 = null[1]["h1"]
    assert h1["p"] > 0.2 and h1["decision"] == "inconclusive"
    assert h1["lo"] < 0 < h1["hi"]


def test_rung_contrasts_have_intervals_and_no_p(effect):
    rungs = effect[1]["rungs"]
    assert [r["contrast"] for r in rungs] == ["B minus A", "C minus B", "D minus C"]
    for r in rungs:
        assert r["lo"] <= r["estimate"] <= r["hi"] and "p" not in r
    assert rungs[0]["n_ref"] == 7 and rungs[0]["n_treat"] == 3


def test_same_seed_same_numbers(effect_dir, effect):
    again = an.analyze_run(effect_dir, seed=1)[1]
    assert again["h1"]["p"] == effect[1]["h1"]["p"] and again["rungs"] == effect[1]["rungs"]
    assert again["secondaries"] == effect[1]["secondaries"]


# ---------------------------------------------------------------- Benjamini-Hochberg and secondaries
def test_bh_known_example():
    q, rej = an.bh([0.001, 0.01, 0.03, 0.04, 0.5, 0.9])
    assert q == pytest.approx([0.006, 0.03, 0.06, 0.06, 0.6, 0.9])
    assert rej.tolist() == [True, True, True, True, False, False]


def test_bh_order_independent_monotone_and_nan_keeps_m():
    p = [0.04, 0.001, 0.9, 0.03, 0.01, 0.5]
    q, rej = an.bh(p)
    assert q.tolist() == [pytest.approx(v) for v in (0.06, 0.006, 0.9, 0.06, 0.03, 0.6)]
    assert (q >= np.array(p)).all() and (q <= 1).all()
    q2, rej2 = an.bh([0.001, math.nan, 0.01, 0.9, 0.5, 0.4])   # NaN is p = 1 and still counts toward m = 6
    assert q2[1] == 1.0 and not rej2[1] and q2[0] == pytest.approx(0.006)
    assert not an.bh([0.2] * 6)[1].any()


def test_secondaries_are_exactly_the_six_prelisted(effect):
    sec = effect[1]["secondaries"]
    assert len(sec) == 6 == len(an.SECONDARIES)
    assert [s["name"].split(" (")[0] for s in sec] == ["gap_sel", "gap_shard", "accepts per run", "integrity + tripwire counts",
                                                         "model dollars per run, nominal", "D cost per accepted change, nominal"]
    q, rej = an.bh([s["p"] for s in sec])
    assert [s["q"] for s in sec] == pytest.approx(q.tolist()) and [s["significant"] for s in sec] == rej.tolist()
    assert all(s["q"] >= s["p"] for s in sec if not math.isnan(s["p"]))
    gap_sel, accepts = sec[0], sec[2]
    assert gap_sel["estimate"] < 0 and gap_sel["p"] < 0.05     # D less optimistic than A by construction
    assert accepts["estimate"] < 0 and accepts["significant"]  # 2 accepts per D run versus 5 per A run


def test_wilson_lower_bound():
    assert an.wilson_lower(10, 10) == pytest.approx(0.7225, abs=1e-3)
    assert an.wilson_lower(0, 10) == pytest.approx(0.0, abs=1e-9)
    assert math.isnan(an.wilson_lower(0, 0))
    assert an.wilson_lower(8, 8) < 0.75 <= an.wilson_lower(12, 12) + 0.001


def test_h5_ledger_proxy_and_qualification(effect):
    rows = {r["edit_class"]: r for r in effect[1]["h5_ledger"]}
    assert set(rows) == {"constant", "schedule", "structural"}
    assert all(r["bar"] == (0.75 if r["edit_class"] == "structural" else 0.6) for r in rows.values())
    assert rows["constant"]["wilson_lower"] <= rows["constant"]["rate"] <= 1.0
    got = an.qualify_rows([({"edit_class": "constant"}, 8, 8), ({"edit_class": "structural"}, 8, 8)])
    assert [g["qualifies"] for g in got] == [True, False]   # 8/8 clears 0.6 but not 0.75


# ---------------------------------------------------------------- sentinels
def test_sentinel_sigma_check_flags_only_beyond_25_percent():
    anchor = dict(mean=1.0, gpu_offsets={"0": 0.0, "1": 0.01})
    rng = np.random.default_rng(0)

    def frame(sd, n=400):
        z = rng.normal(0, 1, n)
        z = (z - z.mean()) / z.std(ddof=1) * sd if n > 1 else np.zeros(1)   # exact sample sd
        gpu = np.arange(n) % 2
        import pandas as pd
        return pd.DataFrame(dict(ts=range(n), job_id="s", gpu=gpu, official_val_bpb=1.0 + z + np.where(gpu == 1, 0.01, 0.0),
                                 status="ok", slot_count_at_submit=range(n)))
    ok = an.sentinel_check(frame(an.NOISE_RESID_SD * 1.2), anchor)
    bad = an.sentinel_check(frame(an.NOISE_RESID_SD * 1.3), anchor)
    low = an.sentinel_check(frame(an.NOISE_RESID_SD * 0.7), anchor)
    assert ok["flagged"] is False and bad["flagged"] is True and low["flagged"] is True
    assert ok["sd_adjusted"] == pytest.approx(an.NOISE_RESID_SD * 1.2, rel=1e-6)    # GPU offsets removed before the sd
    assert ok["sd_raw"] > ok["sd_adjusted"]
    assert {g["gpu"] for g in ok["per_gpu"]} == {"0", "1"}
    assert an.sentinel_check(frame(0.0, n=1), anchor)["flagged"] is None


def test_per_gpu_sentinel_means_in_synthetic_run(effect):
    per = {g["gpu"]: g for g in effect[1]["sentinels"]["per_gpu"]}
    assert set(per) == {"0", "1", "2", "3"} and all(g["n"] == 3 for g in per.values())
    assert per["1"]["mean"] > per["0"]["mean"]


# ---------------------------------------------------------------- end to end
def test_cli_end_to_end_on_the_fixture(effect_dir, tmp_path):
    out = tmp_path / "analysis"
    r = subprocess.run([sys.executable, "-m", "loop.core.analysis", "--run", str(effect_dir), "--out", str(out), "--seed", "3"],
                       cwd=ROOT, capture_output=True, text=True, timeout=300)
    assert r.returncode == 0, r.stderr
    report = (out / "report.md").read_text()
    for h in ("## H1 (confirmatory)", "H1 verdict:", "## Rung contrasts", "## Secondaries", "## Live sigma check",
              "## Deviations (copied from PREREG.md)", "## Power and limits", "Per-GPU sentinel means"):
        assert h in report
    assert "D BETTER" in report
    assert "served_fraction_ledger" in report and "routing_table_frozen.json" in report
    prereg = (ROOT / "loop" / "design" / "PREREG.md").read_text().split("## Deviations")[1].strip()
    assert prereg.split("\n")[0] in report                    # the Deviations body is copied
    for png in an.FIGURES:
        assert (out / png).stat().st_size > 5000, png
        assert (out / png).read_bytes()[:8] == b"\x89PNG\r\n\x1a\n"
        assert f"({png})" in report
    csv_cols = (out / "loop_runs.csv").read_text().splitlines()[0].split(",")
    assert set(TABLE_COLS) <= set(csv_cols)
    res = json.loads((out / "results.json").read_text())
    assert res["h1"]["decision"] == "D better" and len(res["secondaries"]) == 6
    assert EM_DASH not in report


def test_empty_run_does_not_crash(tmp_path):
    (tmp_path / "run").mkdir()
    tab, res = an.run_all(tmp_path / "run", tmp_path / "out")
    assert tab.empty and "cannot be evaluated" in (tmp_path / "out" / "report.md").read_text()
    for png in an.FIGURES:
        assert (tmp_path / "out" / png).exists()


def test_deviations_section_is_copied(tmp_path):
    p = tmp_path / "PREREG.md"
    p.write_text("# T\n\n## Deviations\n\n| time | change |\n|---|---|\n| 1 | x |\n\n## After\n\nno\n")
    assert an.deviations_text(p) == "| time | change |\n|---|---|\n| 1 | x |"
    assert "not found" in an.deviations_text(tmp_path / "missing.md")


def test_no_em_dash_in_sources():
    for p in (HERE.parent / "analysis.py", HERE / "test_analysis.py", HERE.parent / "ANALYSIS.md"):
        if p.exists():
            assert EM_DASH not in p.read_text(), p
