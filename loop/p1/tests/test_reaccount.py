"""Tests for loop/p1/reaccount.py on tiny synthetic ledgers and run directories."""
import json
import math

import numpy as np
import pytest

import reaccount as r

EM_DASH = chr(0x2014)
BASE = {"official": 1.1149, "n": 16, "tbs": 2 ** 19}
R1_IDS = ["A_%02d" % i for i in range(1, 9)] + ["B_01", "B_02", "B_03", "C_01", "C_02", "C_03"] + ["D_%02d" % i for i in range(1, 9)]


# ---------------------------------------------------------------- builders
def mkL(changes=(), n_slots=20, lid="A_01", gpu_h=1.0, claude=0.1, opn=0.01, rows=None):
    """A loop dict with uniform costs per slot. changes = [(slot, official, n_runs, tbs_exponent)]."""
    ch = [{"slot": s, "official": o, "n": n, "tbs": 2 ** e} for s, o, n, e in changes]
    off, n, tbs = r.build_tracks(ch, n_slots, BASE)
    return {"id": lid, "arm": lid[0], "n_slots": n_slots, "jobs": [1] * n_slots, "gpu_s": [gpu_h * 3600] * n_slots,
            "claude": [claude] * n_slots, "open": [opn] * n_slots, "change_slots": sorted(c["slot"] for c in ch),
            "inc_official": off, "inc_n": n, "inc_tbs": tbs, "final_tbs": tbs[-1], "final_depth": 8, "rows": rows or []}


def rec(slot, kind="candidate", decision="reject", official=1.10, job=True, charged=True, calls=None):
    """A ledger row with the fields reaccount reads."""
    return {"slot": slot, "attempt": 0, "kind": kind, "decision": decision, "charged": charged,
            "job_id": ("job_%d_%s" % (slot, kind)) if job else None, "official_val_bpb": official if job else None,
            "model_calls": calls or []}


CALLS = [{"role": "coder", "served_model": "qwen35-4b", "tokens_in": 1000, "tokens_out": 100, "cost_usd_plan": 0.0},
         {"role": "planner_backlog", "served_model": "sonnet", "tokens_in": 5000, "tokens_out": 200, "cost_usd_plan": 0.02}]
PER_M = {"qwen35-4b": {"input": 0.17, "output": 0.25}}


def write_loop(run, lid, rows, n_slots=8, tbs_exp=None, final_exp=19, pending=None, wall=400.0):
    """Write ledger, state, per-slot train.py and result.json, and incumbent.py for a synthetic loop."""
    d = run / lid
    (d / "slots").mkdir(parents=True)
    (d / "ledger.jsonl").write_text("".join(json.dumps(x) + "\n" for x in rows))
    (d / "state.json").write_text(json.dumps({"n_slots": n_slots, "pending_calls": pending or []}))
    for x in rows:
        if x["job_id"]:
            sd = d / "slots" / str(x["slot"])
            sd.mkdir()
            e = (tbs_exp or {}).get(x["slot"], 19)
            (sd / "train.py").write_text("DEPTH = 8\nTOTAL_BATCH_SIZE = 2**%d\n" % e)
            (sd / "result.json").write_text(json.dumps({"job_id": x["job_id"], "wall_seconds": wall}))
    (d / "incumbent.py").write_text("DEPTH = 8\nTOTAL_BATCH_SIZE = 2**%d\n" % final_exp)


def a_rows():
    """Arm A style loop, 8 slots: direct accept at 0 (2^18), reject, accept at 2 (2^17), a keep audited and rejected, rejects."""
    return [rec(0, decision="accept", official=1.08, calls=CALLS), rec(1, official=1.09, calls=CALLS),
            rec(2, decision="accept", official=1.05, calls=CALLS), rec(3, decision="audit", official=1.04, calls=CALLS),
            rec(4, kind="audit", decision="reject", official=1.052, calls=CALLS), rec(5, official=1.06, calls=CALLS),
            rec(6, official=1.055, calls=CALLS), rec(7, official=1.058, calls=CALLS)]


def d_rows():
    """Arm D style loop, 8 slots: accept at 1 (2^18) and at 6 (2^17); slot 3 has an uncharged abort row riding on it."""
    out = [rec(s, official=1.09 - 0.001 * s, calls=CALLS) for s in range(8)]
    out[1] = rec(1, decision="accept", official=1.07, calls=CALLS)
    out[6] = rec(6, decision="accept", official=1.05, calls=CALLS)
    out.insert(3, rec(3, decision="abort", job=False, charged=False, calls=CALLS))
    return out


def make_run(tmp_path):
    """A synthetic run: A_01..A_04 and D_01..D_04. Burn-in loops are A_01, A_03, D_01, D_03; the test loops hold garbage."""
    run = tmp_path / "R1"
    for lid in ("A_01", "A_03"):
        write_loop(run, lid, a_rows(), tbs_exp={0: 18, 2: 17}, final_exp=17)
    for lid in ("D_01", "D_03"):
        write_loop(run, lid, d_rows(), tbs_exp={1: 18, 6: 17}, final_exp=17, pending=[CALLS[1]])
    for lid in ("A_02", "A_04", "D_02", "D_04"):
        (run / lid).mkdir(parents=True)
        (run / lid / "ledger.jsonl").write_text("this is not json, the test half must not be opened")
    (run / "anchor.json").write_text(json.dumps({"values": [{"val": 1.11}, {"val": 1.12}], "n": 2, "sigma_used": 0.004743}))
    sent = [{"official_val_bpb": 1.110 + 0.002 * i, "gpu": i % 4, "status": "ok"} for i in range(12)]
    (run / "launch_sentinels.jsonl").write_text("".join(json.dumps(x) + "\n" for x in sent[:4]))
    (run / "sentinels.jsonl").write_text("".join(json.dumps(x) + "\n" for x in sent[4:]))
    conf = [{"wave": w, "loop_id": None, "arm": None, "kind": "anchor", "status": "ok", "sealed_val_bpb": 1.115} for w in (1, 2)]
    for lid in ("A_01", "A_03", "D_01", "D_03"):
        conf += [{"wave": 1 + k % 2, "loop_id": lid, "arm": lid[0], "kind": "confirm", "status": "ok", "sealed_val_bpb": 1.052}
                 for k in range(4)]
    (run / "confirmation").mkdir()
    (run / "confirmation" / "results.jsonl").write_text("".join(json.dumps(x) + "\n" for x in conf))
    return run


# ---------------------------------------------------------------- reading train.py constants
def test_parse_consts_last_assignment_wins():
    src = "TOTAL_BATCH_SIZE = 2**17\nDEPTH = 8\nTOTAL_BATCH_SIZE = 2**18\nASPECT_RATIO = 64\nHEAD_DIM = 128\n"
    c = r.parse_consts(src)
    assert c == {"TOTAL_BATCH_SIZE": 2 ** 18, "DEPTH": 8, "ASPECT_RATIO": 64, "HEAD_DIM": 128}


def test_parse_consts_expressions():
    c = r.parse_consts("TOTAL_BATCH_SIZE = 1 << 17\nDEPTH = 4 + 2\nASPECT_RATIO = -(-85)\nHEAD_DIM = 256 // 2 * 1 - 0\n")
    assert c == {"TOTAL_BATCH_SIZE": 131072, "DEPTH": 6, "ASPECT_RATIO": 85, "HEAD_DIM": 128}


def test_parse_consts_only_module_level_tracked_names():
    src = "OTHER = 5\ndef f():\n    DEPTH = 99\nif True:\n    TOTAL_BATCH_SIZE = 7\nDEPTH = 8\n"
    assert r.parse_consts(src) == {"DEPTH": 8}


def test_parse_consts_unsupported_expression():
    with pytest.raises(ValueError):
        r.parse_consts("TOTAL_BATCH_SIZE = foo()\n")


def test_check_consts():
    c = {"DEPTH": 8, "TOTAL_BATCH_SIZE": 2 ** 17}
    ok = {"printed": {"depth": 8, "num_steps": 1000, "total_tokens_M": 131.072}}
    r.check_consts(c, ok)
    r.check_consts(c, {})
    r.check_consts(c, None)
    with pytest.raises(ValueError):
        r.check_consts(c, {"printed": {"depth": 6}})
    with pytest.raises(ValueError):
        r.check_consts(c, {"printed": {"num_steps": 1000, "total_tokens_M": 262.144}})


# ---------------------------------------------------------------- split and guard
def test_split_loops_matches_the_fixed_split():
    sp = r.split_loops(R1_IDS)
    burn = sorted(l for l, h in sp.items() if h == "burnin")
    test = sorted(l for l, h in sp.items() if h == "test")
    assert burn == ["A_01", "A_03", "A_05", "A_07", "B_01", "B_03", "C_01", "C_03", "D_01", "D_03", "D_05", "D_07"]
    assert test == ["A_02", "A_04", "A_06", "A_08", "B_02", "C_02", "D_02", "D_04", "D_06", "D_08"]


def test_split_loops_ignores_input_order():
    assert r.split_loops(["D_02", "D_01", "D_03"]) == {"D_01": "burnin", "D_02": "test", "D_03": "burnin"}


def test_check_half_refuses_test_and_all_without_env():
    r.check_half("burnin", {})
    for half in ("test", "all"):
        with pytest.raises(SystemExit):
            r.check_half(half, {})
        with pytest.raises(SystemExit):
            r.check_half(half, {"P1_ALLOW_TEST": "0"})
        r.check_half(half, {"P1_ALLOW_TEST": "1"})
    with pytest.raises(SystemExit):
        r.check_half("nonsense", {"P1_ALLOW_TEST": "1"})


def test_main_refuses_test_half(monkeypatch, tmp_path):
    run = make_run(tmp_path)
    monkeypatch.delenv("P1_ALLOW_TEST", raising=False)
    for half in ("test", "all"):
        with pytest.raises(SystemExit):
            r.main(["--half", half, "--run", str(run), "--rule", "cap:5"])
    assert not (run / "p1").exists()


def test_main_needs_frozen_rule_for_test_half(monkeypatch, tmp_path):
    monkeypatch.setenv("P1_ALLOW_TEST", "1")
    with pytest.raises(SystemExit) as e:
        r.main(["--half", "test", "--run", str(make_run(tmp_path))])
    assert "rule" in str(e.value)


def test_rule_helpers():
    assert r.parse_rule("quiet:5") == ("quiet", 5)
    assert r.parse_rule("after_first:10") == ("after_first", 10)
    assert r.rule_label(("cap", 8)) == "cap_n8"
    assert r.rule_label(("after_first", 3)) == "after_first_k3"
    with pytest.raises(ValueError):
        r.parse_rule("cap:7")


# ---------------------------------------------------------------- ledger readers
def test_groups_and_change_events():
    recs = [rec(0), rec(1, decision="audit", official=1.05), rec(2, kind="audit", decision="accept", official=1.06),
            rec(3, official=1.2), rec(4, decision="accept", official=1.0), rec(5, decision="abort", job=False, charged=False),
            rec(5, kind="candidate", decision="infra", job=False, charged=False), rec(6, decision="crash", job=True, official=None),
            rec(7, kind="crash_fix", decision="accept", official=1.01)]
    assert [len(g) for g in r.groups(recs)] == [1, 2, 1, 1, 1, 1, 1]
    ev = r.change_events(recs)
    assert [e["slot"] for e in ev] == [2, 4, 7]
    assert ev[0]["first_slot"] == 1 and ev[0]["n_runs"] == 2 and ev[0]["official"] == pytest.approx(1.055)
    assert ev[1]["n_runs"] == 1 and ev[1]["official"] == 1.0


def test_call_dollars():
    assert r.call_dollars(CALLS[1], PER_M) == (0.02, 0.0)
    claude, opn = r.call_dollars(CALLS[0], PER_M)
    assert claude == 0.0 and opn == pytest.approx((1000 * 0.17 + 100 * 0.25) / 1e6)


def test_slot_costs_counts_uncharged_rows_clamps_and_adds_tail():
    recs = [rec(0, calls=CALLS), rec(1, decision="abort", job=False, charged=False, calls=CALLS), rec(1, calls=CALLS),
            rec(5, calls=[CALLS[1]])]
    claude, opn = r.slot_costs(recs, PER_M, 3, tail_calls=[CALLS[1]])
    assert claude == pytest.approx([0.02, 0.04, 0.02 + 0.02])   # slot 5 clamps to the last slot, the tail also lands there
    assert opn[1] == pytest.approx(2 * 195e-6) and opn[2] == 0.0


def test_gpu_seconds(tmp_path):
    write_loop(tmp_path, "A_01", [rec(0), rec(1, job=False), rec(2)], n_slots=3, wall=400.0)
    recs = r.jl(tmp_path / "A_01" / "ledger.jsonl")
    assert r.gpu_seconds(tmp_path / "A_01", recs, 3) == [400.0, 0.0, 400.0]
    recs[2]["job_id"] = "someone_else"
    with pytest.raises(ValueError):
        r.gpu_seconds(tmp_path / "A_01", recs, 3)
    with pytest.raises(ValueError):
        r.gpu_seconds(tmp_path / "A_01", [rec(0), rec(0)], 3)


def test_build_tracks():
    ch = [{"slot": 1, "official": 1.07, "n": 1, "tbs": 2 ** 18}, {"slot": 3, "official": 1.05, "n": 2, "tbs": 2 ** 17}]
    off, n, tbs = r.build_tracks(ch, 5, BASE)
    assert off == [1.1149, 1.07, 1.07, 1.05, 1.05]
    assert n == [16, 1, 1, 2, 2]
    assert tbs == [2 ** 19, 2 ** 18, 2 ** 18, 2 ** 17, 2 ** 17]


def test_load_loop(tmp_path):
    write_loop(tmp_path, "D_01", d_rows(), tbs_exp={1: 18, 6: 17}, final_exp=17, pending=[CALLS[1]])
    L = r.load_loop(tmp_path, "D_01", PER_M, BASE)
    assert L["n_slots"] == 8 and L["arm"] == "D" and L["jobs"] == [1] * 8
    assert L["gpu_s"] == [400.0] * 8
    assert L["change_slots"] == [1, 6]
    assert L["inc_tbs"] == [2 ** 19, 2 ** 18, 2 ** 18, 2 ** 18, 2 ** 18, 2 ** 18, 2 ** 17, 2 ** 17]
    assert L["final_tbs"] == 2 ** 17 and L["final_depth"] == 8
    assert sum(L["claude"]) == pytest.approx(9 * 0.02 + 0.02)    # 9 rows including the abort, plus the tail call
    assert L["claude"][3] == pytest.approx(0.04)                 # abort row rides on slot 3
    assert len(L["rows"]) == 8 and L["rows"][1]["decision"] == "accept"


def test_load_loop_rejects_broken_loops(tmp_path):
    rows = [rec(0), rec(2)]
    write_loop(tmp_path, "A_01", rows, n_slots=3)
    with pytest.raises(ValueError):        # slot 1 is not charged
        r.load_loop(tmp_path, "A_01", PER_M, BASE)
    write_loop(tmp_path, "A_02", [rec(0, decision="accept", official=1.0)], n_slots=1, tbs_exp={0: 17}, final_exp=18)
    with pytest.raises(ValueError):        # incumbent.py disagrees with the last accepted slot
        r.load_loop(tmp_path, "A_02", PER_M, BASE)


def test_load_loop_needs_a_change(tmp_path):
    write_loop(tmp_path, "A_01", [rec(0), rec(1)], n_slots=2)
    with pytest.raises(ValueError):
        r.load_loop(tmp_path, "A_01", PER_M, BASE)


# ---------------------------------------------------------------- cost math
def test_cost_upto_and_loop_total():
    L = mkL()
    c = r.cost_upto(L, 4, 2.0)
    assert c["jobs"] == 5 and c["gpu_h"] == pytest.approx(5.0)
    assert c["claude"] == pytest.approx(0.5) and c["open"] == pytest.approx(0.05) and c["model"] == pytest.approx(0.55)
    assert c["total"] == pytest.approx(0.55 + 10.0)
    assert r.cost_upto(L, 99, 1.0) == r.loop_total(L, 1.0)
    assert r.loop_total(L, 0.5)["total"] == pytest.approx(2.2 + 10.0)


def test_crossover_price():
    assert r.crossover_price(1.0, 1.0, 0.5, 2.0) == pytest.approx(0.5)   # 1 + p = 0.5 + 2p at p = 0.5
    assert r.crossover_price(1.0, 2.0, 1.0, 2.0) is None
    assert r.crossover_price(0.2, 1.0, 0.1, 0.5) == pytest.approx(-0.2)


def test_crossover_keeps_only_positive_prices_and_gives_gaps():
    c = r.crossover(1.0, 1.0, 0.5, 2.0)
    assert c["price"] == pytest.approx(0.5)
    assert c["x_minus_y_at_0"] == pytest.approx(0.5) and c["x_minus_y_at_4"] == pytest.approx(5.0 - 8.5)
    neg = r.crossover(0.2, 1.0, 0.1, 0.5)
    assert neg["price"] is None and neg["raw_price"] == pytest.approx(-0.2)
    assert "dearer" in r.xdescribe("D_vs_A", neg)
    assert "0.500 USD per A100-hour" in r.xdescribe("D_vs_A", c)
    cheap = r.crossover(0.1, 0.5, 0.2, 1.0)
    assert "cheaper" in r.xdescribe("D_vs_C", cheap)


# ---------------------------------------------------------------- plateau, reach, Kaplan-Meier
@pytest.mark.parametrize("last,censored,slot", [(2, False, 2), (14, False, 14), (15, True, 20), (19, True, 20)])
def test_plateau_observed_or_censored(last, censored, slot):
    L = mkL(changes=[(0, 1.08, 1, 18), (last, 1.05, 2, 17)] if last else [(0, 1.08, 1, 18)])
    p = r.plateau(L)
    assert p == {"last_change": last, "slot": slot, "censored": censored}


def test_plateau_quiet_parameter_and_short_loops():
    L = mkL(changes=[(3, 1.05, 1, 17)], n_slots=8)
    assert r.plateau(L)["censored"] is True       # only 4 slots follow
    assert r.plateau(L, quiet=4) == {"last_change": 3, "slot": 3, "censored": False}


def test_reach():
    L = mkL(changes=[(0, 1.08, 1, 18), (4, 1.05, 2, 17), (9, 1.04, 1, 16)])
    assert r.reach(L) == {"slot": 4, "censored": False}
    assert r.reach(mkL(changes=[(0, 1.08, 1, 18)])) == {"slot": 20, "censored": True}
    assert r.reach(mkL(changes=[(7, 1.08, 1, 18)]), thr=2 ** 18) == {"slot": 7, "censored": False}


def test_km_median():
    assert r.km_median([1, 2, 3], [True, True, True]) == 2
    assert r.km_median([5, 8, 10, 20], [True, True, True, False]) == 8
    assert r.km_median([5, 20, 20], [True, False, False]) is None
    assert r.km_median([2, 2, 2, 2], [True] * 4) == 2
    curve = r.km_curve([1, 2, 2, 5], [True, True, False, True])
    assert curve == [(1, 0.75), (2, 0.5), (5, 0.0)]


def test_km_censored_before_event_counts_only_while_at_risk():
    curve = r.km_curve([3, 4, 6, 9], [False, True, True, True])
    assert curve[0] == (4, pytest.approx(2 / 3)) and curve[1] == (6, pytest.approx(1 / 3)) and curve[2] == (9, 0.0)


def test_boot_diff():
    rng = np.random.default_rng(0)
    sep = r.boot_diff([2, 2, 3, 2], [10, 12, 9, 11], 2000, rng)
    assert sep["diff"] == pytest.approx(2.25 - 10.5) and sep["excludes_zero"] and sep["hi"] < 0
    same = r.boot_diff([1, 5, 9, 3], [1, 5, 9, 3], 2000, rng)
    assert same["diff"] == 0 and not same["excludes_zero"] and same["lo"] < 0 < same["hi"]


# ---------------------------------------------------------------- halting rules
def naive_halt(cs, n, rule):
    """Independent window version of the rules."""
    kind, p = rule
    if kind == "cap":
        return min(p - 1, n - 1)
    start = 0 if kind == "quiet" else (min(cs) + 1 if cs else n)
    for s in range(start + p - 1, n):
        if not any(c in cs for c in range(s - p + 1, s + 1)):
            return s
    return n - 1


def test_halt_slot_hand_cases():
    cs = [2, 5]
    assert r.halt_slot(cs, 20, ("quiet", 3)) == 8
    assert r.halt_slot(cs, 20, ("quiet", 2)) == 1
    assert r.halt_slot(cs, 20, ("after_first", 3)) == 8
    assert r.halt_slot(cs, 20, ("after_first", 2)) == 4
    assert r.halt_slot(cs, 20, ("cap", 5)) == 4
    assert r.halt_slot(cs, 20, ("cap", 30)) == 19
    assert r.halt_slot([], 20, ("quiet", 3)) == 2
    assert r.halt_slot([], 20, ("after_first", 3)) == 19
    assert r.halt_slot([0, 2, 4, 6, 8, 10, 12, 14, 16, 18], 20, ("quiet", 3)) == 19      # never fires


def test_halt_slot_matches_naive_windows():
    rng = np.random.default_rng(3)
    for _ in range(300):
        cs = sorted(set(rng.choice(20, rng.integers(1, 6), replace=False).tolist()))
        for rule in r.RULES:
            assert r.halt_slot(cs, 20, rule) == naive_halt(set(cs), 20, rule), (cs, rule)


def test_halting_outcome_fraction_and_loss():
    L = mkL(changes=[(2, 1.06, 1, 18), (6, 1.04, 2, 17)], n_slots=10)
    o = r.halting_outcome(L, ("cap", 4))
    assert o["halt"] == 3 and o["frac"] == pytest.approx(0.4) and o["loss"] == pytest.approx(0.02)
    o = r.halting_outcome(L, ("quiet", 3))
    assert o["halt"] == 5 and o["frac"] == pytest.approx(0.6) and o["loss"] == pytest.approx(0.02)
    o = r.halting_outcome(L, ("quiet", 4))                 # never fires: runs to the end, spends everything, loses nothing
    assert o["halt"] == 9 and o["frac"] == pytest.approx(1.0) and o["loss"] == 0.0
    early = mkL(changes=[(5, 1.0, 1, 17)], n_slots=10)
    assert r.halting_outcome(early, ("quiet", 3))["loss"] == pytest.approx(1.1149 - 1.0)   # halt at slot 2 with the baseline
    low = mkL(changes=[(1, 1.00, 1, 18), (6, 1.05, 1, 17)], n_slots=10)    # final raw mean worse than the halted one
    assert r.halting_outcome(low, ("cap", 4))["loss"] == pytest.approx(-0.05)


def test_summarize_and_select_rule():
    outs = [{"frac": 0.5, "loss": 0.0}, {"frac": 0.3, "loss": 0.004}, {"frac": 0.4, "loss": 0.0041}, {"frac": 0.2, "loss": -0.01}]
    s = r.summarize_rule(outs)
    assert s["share_within"] == 0.75 and s["mean_frac"] == pytest.approx(0.35) and s["mean_loss"] == pytest.approx(-0.000475)
    sums = {"a": {"share_within": 0.9, "mean_frac": 0.6}, "b": {"share_within": 0.8, "mean_frac": 0.5},
            "c": {"share_within": 0.79, "mean_frac": 0.1}, "d": {"share_within": 1.0, "mean_frac": 0.5}}
    assert r.select_rule(sums) == "b"                      # lowest fraction among those at 0.8 or more, tie goes to the earlier rule
    assert r.select_rule({"c": sums["c"]}) is None


# ---------------------------------------------------------------- improvements, sentinels, ANOVA
def test_final_improvements():
    conf = [{"wave": 1, "loop_id": None, "kind": "anchor", "status": "ok", "sealed_val_bpb": 1.10},
            {"wave": 2, "loop_id": None, "kind": "anchor", "status": "ok", "sealed_val_bpb": 1.20},
            {"wave": 3, "loop_id": None, "kind": "anchor", "status": "crash", "sealed_val_bpb": None},
            {"wave": 1, "loop_id": "A_01", "kind": "confirm", "status": "ok", "sealed_val_bpb": 1.04},
            {"wave": 2, "loop_id": "A_01", "kind": "confirm", "status": "ok", "sealed_val_bpb": 1.12},
            {"wave": 3, "loop_id": "A_01", "kind": "confirm", "status": "ok", "sealed_val_bpb": 1.00},
            {"wave": 1, "loop_id": "B_01", "kind": "confirm", "status": "ok", "sealed_val_bpb": 1.00}]
    out = r.final_improvements(conf, ["A_01"])
    assert list(out) == ["A_01"]
    a = out["A_01"]
    assert a["n_runs"] == 3 and a["sealed_mean"] == pytest.approx((1.04 + 1.12 + 1.00) / 3)
    assert a["paired"] == pytest.approx(((1.10 - 1.04) + (1.20 - 1.12)) / 2)       # wave 3 has no usable anchor
    assert a["global"] == pytest.approx(1.15 - a["sealed_mean"])


def test_sentinel_rows(tmp_path):
    (tmp_path / "launch_sentinels.jsonl").write_text(json.dumps({"official_val_bpb": 1.1, "gpu": 0, "status": "ok"}) + "\n")
    (tmp_path / "sentinels.jsonl").write_text(json.dumps({"official_val_bpb": 1.2, "gpu": 1, "status": "ok"}) + "\n"
                                              + json.dumps({"official_val_bpb": None, "gpu": 2, "status": "crash"}) + "\n")
    assert r.sentinel_rows(tmp_path) == [(1.1, 0), (1.2, 1)]


def test_anova_gpu():
    rows = [(1.0, 0), (1.1, 0), (1.2, 0), (2.0, 1), (2.1, 1), (2.2, 1)]
    a = r.anova_gpu(rows)
    grand = np.mean([v for v, _ in rows])
    ssb = 3 * (1.1 - grand) ** 2 + 3 * (2.1 - grand) ** 2
    ssw = 4 * 0.01 * 1.0
    assert a["F"] == pytest.approx((ssb / 1) / (ssw / 4))
    assert a["df_between"] == 1 and a["df_within"] == 4 and a["p"] < 0.001
    assert a["per_gpu"]["0"]["n"] == 3 and a["per_gpu"]["1"]["mean"] == pytest.approx(2.1)
    flat = r.anova_gpu([(1.0, 0), (1.2, 0), (1.1, 1), (1.1, 1)])
    assert flat["p"] > 0.5


# ---------------------------------------------------------------- noise replay
def converged_loop():
    rows = [{"slot": 0, "kind": "candidate", "decision": "accept", "official": 1.08},
            {"slot": 2, "kind": "candidate", "decision": "accept", "official": 1.05},
            {"slot": 3, "kind": "candidate", "decision": "audit", "official": 1.04},
            {"slot": 4, "kind": "audit", "decision": "reject", "official": 1.052},
            {"slot": 5, "kind": "candidate", "decision": "reject", "official": 1.06},
            {"slot": 6, "kind": "candidate", "decision": "accept", "official": 1.049},
            {"slot": 7, "kind": "candidate", "decision": "reject", "official": 1.07}]
    return mkL(changes=[(0, 1.08, 1, 18), (2, 1.05, 2, 17)], n_slots=8, rows=rows)


def test_converged_view_and_observed_counts():
    v = r.converged_view(converged_loop())
    assert v["reach"] == 2 and v["bar0"] == 1.05 and v["n0"] == 2 and v["slots_left"] == 5
    assert [c["slot"] for c in v["cands"]] == [3, 5, 6, 7]
    assert [c["keep"] for c in v["cands"]] == [True, False, True, False]
    assert [c["below_bar"] for c in v["cands"]] == [True, False, True, False]
    assert v["audits"] == [{"slot": 4, "reject": True}]
    o = r.observed_counts([v])
    assert (o["candidates"], o["keeps"], o["audits"], o["audit_rejects"], o["audit_accepts"], o["bar_rule_agree"]) == (4, 2, 1, 1, 0, 4)
    assert o["first_run_mean"] == pytest.approx(np.mean([1.04, 1.06, 1.049, 1.07]))
    assert r.converged_view(mkL(changes=[(0, 1.08, 1, 18)])) is None


def test_make_noise():
    rng = np.random.default_rng(0)
    g = r.make_noise("gaussian", None)(rng, 100000, 0.5)
    assert abs(g.mean()) < 0.01 and g.std() == pytest.approx(0.5, rel=0.02)
    resid = np.array([-1.0, 0.0, 3.0])
    b = r.make_noise("bootstrap", resid)(rng, 1000, None)
    assert set(np.unique(b)) <= {-1.0, 0.0, 3.0}


def test_sim_converged_extremes_and_bookkeeping():
    rng = np.random.default_rng(1)
    resid = np.zeros(30)
    # level far below the bar, almost no noise: every candidate is kept, audited and accepted, then the bar drops to the level
    t = r.sim_converged([(2.0, 2, 6)], 1.0, 1e-6, 30, 30, 0.004743, "gaussian", resid, 200, rng)
    assert t["keeps"].min() >= 1 and (t["audit_accepts"] >= 1).all()
    assert (t["candidates"] + t["audits"] == 6).all()                    # every slot is a candidate or an audit
    assert (t["audits"] <= t["keeps"]).all() and (t["audit_accepts"] + t["audit_rejects"] == t["audits"]).all()
    # level far above the bar: nothing is ever kept
    t = r.sim_converged([(1.0, 2, 6)], 2.0, 1e-6, 30, 30, 0.004743, "gaussian", resid, 200, rng)
    assert t["keeps"].sum() == 0 and t["audits"].sum() == 0 and (t["candidates"] == 6).all()
    # a keep in the last slot gets no audit
    t = r.sim_converged([(2.0, 2, 1)], 1.0, 1e-6, 30, 30, 0.004743, "gaussian", resid, 50, rng)
    assert (t["keeps"] == 1).all() and t["audits"].sum() == 0
    # bootstrap noise runs too
    t = r.sim_converged([(1.0, 2, 4)], 2.0, 0.0038, 30, 30, 0.004743, "bootstrap", np.array([-0.001, 0.001]), 50, rng)
    assert t["keeps"].sum() == 0


def test_pred_interval():
    assert r.pred_interval(np.arange(101)) == (5, 95)
    lo, hi = r.pred_interval(np.array([3, 3, 3, 3, 3]))
    assert (lo, hi) == (3, 3)


def test_noise_replay_structure():
    v = r.converged_view(converged_loop())
    vals = list(1.11 + 0.003 * np.random.default_rng(2).standard_normal(40))
    out = r.noise_replay([v], vals, 0.004743, 400, np.random.default_rng(0))
    assert out["observed"]["audit_rejects"] == 1 and out["sentinel_n"] == 40
    assert set(out["predicted"]) == {"gaussian", "bootstrap", "gaussian_conv"}
    ar = out["predicted"]["gaussian"]["audit_rejects"]
    assert ar["lo90"] <= ar["mean"] <= ar["hi90"] and ar["inside"] == (ar["lo90"] <= 1 <= ar["hi90"])
    assert 0 <= ar["p_ge"] <= 1 and 0 <= ar["p_le"] <= 1


# ---------------------------------------------------------------- frontier
def test_sim_frontier_limits():
    rng = np.random.default_rng(5)
    draw = r.make_noise("gaussian", None)
    big = {rule: r.sim_frontier(rule, 1.0, draw, 0.004, 0.004743, 2, 400, rng) for rule in r.FRONTIER_RULES}
    for rule in r.FRONTIER_RULES:
        assert big[rule]["first_keep"] == 1.0 and big[rule]["keeps"] == 20       # a huge true effect is always kept
    assert [big[k]["runs_per_slot"] for k in ("keep_if_lower", "zband", "mean2", "median3")] == [1.0, 1.0, 2.0, 3.0]
    assert big["zband"]["first_runs"] == 1.0
    with pytest.raises(ValueError):
        r.sim_frontier("nope", 0.0, draw, 0.004, 0.004743, 2, 10, rng)


def test_sim_frontier_null_behaviour():
    rng = np.random.default_rng(6)
    draw = r.make_noise("gaussian", None)
    kil = r.sim_frontier("keep_if_lower", 0.0, draw, 0.004, 0.004743, 1, 20000, rng)
    assert kil["first_keep"] == pytest.approx(0.5, abs=0.02)           # incumbent and candidate are exchangeable
    assert 2.0 < kil["keeps"] < 3.6                                    # ratchet: roughly the number of records in 20 draws
    zb = r.sim_frontier("zband", 0.0, draw, 0.004, 0.004743, 2, 20000, rng)
    assert zb["keeps"] < 0.25 * kil["keeps"] and 1.0 < zb["runs_per_slot"] < 1.5
    m2 = r.sim_frontier("mean2", 0.0, draw, 0.004, 0.004743, 2, 20000, rng)
    assert m2["runs_per_slot"] == 2.0 and abs(m2["keeps"] - kil["keeps"]) < 0.6   # more runs do not cut false keeps of a ratchet


def test_sim_frontier_power_grows_with_effect():
    rng = np.random.default_rng(7)
    draw = r.make_noise("gaussian", None)
    p = [r.sim_frontier("zband", d, draw, 0.004, 0.004743, 2, 5000, rng)["first_keep"] for d in (0.0, 0.004, 0.012)]
    assert p[0] < p[1] < p[2]


def test_frontier_structure():
    out = r.frontier(list(1.11 + 0.003 * np.random.default_rng(0).standard_normal(30)), 0.004743, 2, 50, np.random.default_rng(0))
    assert set(out) == {"gaussian", "bootstrap"}
    assert set(out["gaussian"]) == set(r.FRONTIER_RULES) and set(out["gaussian"]["zband"]) == {str(d) for d in r.FRONTIER_EFFECTS}
    out = r.frontier(list(1.11 + 0.003 * np.random.default_rng(0).standard_normal(30)), 0.004743, 2, 50, np.random.default_rng(0), conv_sd=0.002)
    assert "gaussian_conv" in out


def test_slot_noise():
    s = r.slot_noise([5, 8, 10, 20])
    assert s["sd"] == pytest.approx(np.std([5, 8, 10, 20], ddof=1)) and s["n"] == 4
    assert s["se_diff"] == pytest.approx(s["sd"] * math.sqrt(2 / 24)) and s["half_width_90"] == pytest.approx(1.645 * s["se_diff"])


# ---------------------------------------------------------------- whole pipeline on a synthetic run
def test_main_burnin_end_to_end(tmp_path, monkeypatch):
    monkeypatch.delenv("P1_ALLOW_TEST", raising=False)
    run = make_run(tmp_path)
    opened = []
    real = r.jl
    monkeypatch.setattr(r, "jl", lambda p: opened.append(str(p)) or real(p))
    r.main(["--half", "burnin", "--run", str(run), "--sims", "300", "--boot", "300"])
    assert not any(t in p for p in opened for t in ("A_02", "A_04", "D_02", "D_04"))     # test loops never opened
    res = json.loads((run / "p1" / "burnin" / "results.json").read_text())
    rep = (run / "p1" / "burnin" / "report.md").read_text()
    assert res["loops"] == ["A_01", "A_03", "D_01", "D_03"] and set(res["split"].values()) == {"burnin"}
    assert EM_DASH not in rep and "Project 1 re-accounting of R1: burnin half" in rep
    # costs: A loop has 8 jobs of 400 s; Claude 8 * 0.02; D loop has 8 jobs, 9 rows of calls plus one tail call
    a = res["cost"]["per_loop"]["A_01"]
    assert a["jobs"] == 8 and a["gpu_h"] == pytest.approx(8 * 400 / 3600) and a["claude"] == pytest.approx(0.16) and a["accepts"] == 2
    assert res["cost"]["per_loop"]["D_01"]["claude"] == pytest.approx(0.2)
    assert a["total"]["2.0"] == pytest.approx(0.16 + a["open"] + 2 * 8 * 400 / 3600)
    assert a["per_accept"]["2.0"] == pytest.approx(a["total"]["2.0"] / 2)
    # plateau: A last change at slot 2 with 5 slots after it is observed, D last change at 6 is censored at n_slots
    assert res["plateau"]["per_loop"]["A_01"]["censored"] is False and res["plateau"]["per_loop"]["A_01"]["slot"] == 2
    assert res["plateau"]["per_loop"]["D_01"]["censored"] is True
    assert res["reach"]["per_loop"]["A_01"]["slot"] == 2 and res["reach"]["per_loop"]["D_01"]["slot"] == 6
    assert res["stall"]["n_stalled"] == 0 and res["stall"]["draft_p2_flag"]["n_flagged"] == 4     # the draft rule flags depth 8
    assert len(res["params"]["code_sha256"]) == 64
    assert res["noise"]["replay"]["observed"]["candidates"] == 8 and res["noise"]["replay"]["observed"]["bar_rule_agree"] == 8
    assert res["halting"]["selected"] in res["halting"]["summary"] or res["halting"]["selected"] is None
    assert set(res["decisions"]) >= {"claim1", "halting", "n2_test1"}
    assert res["params"]["half"] == "burnin"


def test_analyse_stall_flags_and_arm_summaries():
    a = converged_loop()
    d = mkL(changes=[(3, 1.06, 1, 18)], n_slots=8, lid="D_01")
    d["final_depth"] = 6
    imp = {"A_01": {"n_runs": 4, "sealed_mean": 1.052, "paired": 0.063, "global": 0.064},
           "D_01": {"n_runs": 4, "sealed_mean": 1.065, "paired": 0.052, "global": 0.050}}
    vals = list(1.11 + 0.003 * np.random.default_rng(2).standard_normal(30))
    sent = [(v, i % 4) for i, v in enumerate(vals)]
    res = r.analyse([a, d], sent, imp, 0.004743, 200, 200, 1)
    st = res["stall"]
    assert st["n_stalled"] == 1 and st["per_loop"]["D_01"]["stalled"] and not st["per_loop"]["A_01"]["stalled"]
    assert st["draft_p2_flag"]["n_flagged"] == 1 and st["draft_p2_flag"]["n_stalled_missed"] == 1
    assert st["draft_p2_flag"]["flagged_mean_paired"] == pytest.approx(0.063)          # only the depth 8 loop is flagged
    assert st["groups"]["paired"]["separated"] is True and st["groups"]["global"]["stalled_mean"] == pytest.approx(0.050)
    assert res["plateau"]["per_arm"]["D"]["n_censored"] == 1 and res["plateau"]["per_arm"]["A"]["km_median"] == 2
    assert res["halting"]["outcomes"]["cap_n5"]["A_01"]["halt"] == 4
    assert "handoff" not in res                    # needs more than one arm D loop
    frozen = r.analyse([a, d], sent, imp, 0.004743, 200, 200, 1, frozen_rule="cap_n5")
    assert frozen["decisions"]["halting"]["rule"] == "cap_n5"


def test_report_has_no_em_dash_and_lists_every_section(tmp_path):
    run = make_run(tmp_path)
    r.main(["--half", "burnin", "--run", str(run), "--sims", "200", "--boot", "200", "--out", str(tmp_path / "out")])
    rep = (tmp_path / "out" / "report.md").read_text()
    for head in ("## 1. Cost reconciliation", "## 2. Plateau", "## 3. Reach 2^17", "## 4. Stall", "## 5. Halting rules",
                 "## 6. Noise replay", "## 7. GPU offsets", "## Decision lines"):
        assert head in rep
    assert EM_DASH not in rep
    assert EM_DASH not in (tmp_path / "out" / "results.json").read_text()


def test_all_loop_ids_lists_only_loop_dirs(tmp_path):
    run = make_run(tmp_path)
    assert r.all_loop_ids(run) == ["A_01", "A_02", "A_03", "A_04", "D_01", "D_02", "D_03", "D_04"]


def test_table_and_fm():
    assert r.fm(None) == "n/a" and r.fm(1.23456, 2) == "1.23"
    assert r.table(["a", "b"], [[1, 2]]) == "| a | b |\n|---|---|\n| 1 | 2 |"


def test_mean_of_and_by_arm():
    assert r.mean_of([{"x": 1.0}, {"x": 4.0}], "x") == 2.5
    arms = r.by_arm([mkL(lid="D_01"), mkL(lid="A_01"), mkL(lid="D_03")])
    assert list(arms) == ["A", "D"] and [L["id"] for L in arms["D"]] == ["D_01", "D_03"]


# ---------------------------------------------------------------- regressions found by the independent verifier
def test_analyse_uses_frozen_frontier_n_inc():
    a = converged_loop()                                   # its converged incumbent rests on 2 runs
    sent = [(v, i % 4) for i, v in enumerate(1.11 + 0.003 * np.random.default_rng(2).standard_normal(30))]
    imp = {"A_01": {"n_runs": 4, "sealed_mean": 1.052, "paired": 0.063, "global": 0.064}}
    assert r.analyse([a], sent, imp, 0.004743, 50, 50, 1)["noise"]["frontier_n_inc"] == 2
    assert r.analyse([a], sent, imp, 0.004743, 50, 50, 1, frozen_n_inc=3)["noise"]["frontier_n_inc"] == 3


def test_main_freezes_frontier_n_inc_outside_burnin(monkeypatch, tmp_path):
    """The test and all halves use the burn-in frontier incumbent size instead of re-deriving it from their own loops."""
    run = make_run(tmp_path)
    seen = {}

    def fake_analyse(*args):
        seen["n_inc"] = args[-1]
        raise SystemExit(0)
    monkeypatch.setenv("P1_ALLOW_TEST", "1")
    monkeypatch.setattr(r, "load_loop", lambda run, lid, per_m, base: lid)
    monkeypatch.setattr(r, "final_improvements", lambda conf, ids: {})
    monkeypatch.setattr(r, "analyse", fake_analyse)
    for half, want in (("burnin", None), ("test", r.FRONTIER_N_INC), ("all", r.FRONTIER_N_INC)):
        with pytest.raises(SystemExit):
            r.main(["--half", half, "--run", str(run), "--rule", "after_first:5"])
        assert seen.pop("n_inc") == want


def test_burnin_output_ignores_test_rows_in_the_shared_confirmation_file(monkeypatch, tmp_path):
    monkeypatch.delenv("P1_ALLOW_TEST", raising=False)
    run = make_run(tmp_path)
    args = ["--half", "burnin", "--run", str(run), "--sims", "100", "--boot", "100", "--out"]
    r.main(args + [str(tmp_path / "before")])
    with open(run / "confirmation" / "results.jsonl", "a") as f:
        for lid in ("A_02", "D_04"):
            f.write(json.dumps({"wave": 1, "loop_id": lid, "arm": lid[0], "kind": "confirm", "status": "ok", "sealed_val_bpb": 0.5}) + "\n")
    r.main(args + [str(tmp_path / "after")])
    assert (tmp_path / "before" / "results.json").read_text() == (tmp_path / "after" / "results.json").read_text()


def test_test_half_requires_the_frozen_rule(monkeypatch):
    import reaccount as R
    monkeypatch.setenv("P1_ALLOW_TEST", "1")
    with pytest.raises(SystemExit) as e:
        R.main(["--half", "test", "--rule", "cap:10"])
    assert "after_first:5" in str(e.value)


def test_fixed_split_matches_r1_directory():
    import reaccount as R
    split = R.split_loops(R.all_loop_ids(R.RUN))
    assert {l for l, h in split.items() if h == "burnin"} == R.FIXED_BURNIN
    assert {l for l, h in split.items() if h == "test"} == R.FIXED_TEST
