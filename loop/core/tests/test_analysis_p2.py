"""Tests of loop/core/analysis_p2.py on a synthetic P2 run built in a temp dir (54 loops: SS, ST, QS, QT, H)."""
import json
import math
import statistics
from fractions import Fraction
from pathlib import Path

import numpy as np
import pytest

import analysis_p2 as P
import plan_p2

PRICES = json.loads((Path(P.DEFAULT_PRICES)).read_text())
EM_DASH = chr(0x2014)
OPEN_IN = {"qwen35-4b": 0.17, "gemma4-12b": 0.05}


def jl(path, rows):
    Path(path).write_text("".join(json.dumps(r) + "\n" for r in rows))


# ---------------------------------------------------------------- synthetic run
def src(tbs_exp=19, depth=8):
    return "ASPECT_RATIO = 64\nHEAD_DIM = 128\nTOTAL_BATCH_SIZE = 2**%d\nDEPTH = %d\n" % (tbs_exp, depth)


def write_loop(root, lid, cell, changes=(), claude=0.2, open_tok=(0, 0), gpus=(0, 1), wall=400.0, served=(10, 10),
               fallbacks=0, harm_noops=0, routing=True, slots=20, status="done", block=1, with_cell_in_config=True,
               flagged=(), planner_served=None):
    """changes: [(slot, tbs_exponent, depth, runs_behind_the_accept)]. The ledger has one row per slot. flagged: slots whose
    job has load_flagged. planner_served: served models of the two planner calls on the first ledger row (default: the
    model the cell's table names)."""
    d = root / lid
    (d / "slots").mkdir(parents=True)
    by_slot = {c[0]: c for c in changes}
    cur = (19, 8)
    rows = []
    for k in range(slots):
        if k in by_slot:
            cur = by_slot[k][1:3]
        (d / "slots" / str(k)).mkdir()
        (d / "slots" / str(k) / "train.py").write_text(src(*cur))
        (d / "slots" / str(k) / "result.json").write_text(json.dumps(
            {"status": "ok", "gpu": gpus[k % len(gpus)], "wall_seconds": wall, "load_flagged": k in flagged}))
        acc = k in by_slot
        calls = [{"role": "coder", "served_model": "qwen35-4b", "requested_model": "qwen35-4b"}]
        if k == 0:
            want = "qwen35-4b" if cell[0] == "Q" else "sonnet"
            calls += [{"role": r, "served_model": m, "requested_model": want}
                      for r, m in zip(("planner_backlog", "planner_widen"), planner_served or (want, want))]
        rows.append({"slot": k, "attempt": 0, "kind": "candidate", "decision": "accept" if acc else "reject",
                     "edit_class": "constant", "charged": True, "job_id": "j%d" % k,
                     "incumbent_after": {"mean": 1.1, "n": by_slot[k][3] if acc else 16},
                     "model_calls": calls})
    (d / "incumbent.py").write_text(src(*cur))
    jl(d / "ledger.jsonl", rows)
    jl(d / "calls.jsonl", [
        {"served_model": "sonnet", "provider": "claude", "cost_usd_plan": claude, "usage": {"input": 10, "output": 10}},
        {"served_model": "qwen35-4b", "provider": "vllm", "cost_usd_plan": 0.0,
         "usage": {"input": open_tok[0], "output": open_tok[1]}}])
    state = {"loop_id": lid, "arm": "D", "status": status, "slot_index": slots, "n_slots": 20,
             "designated_open_calls": served[0], "open_served_calls": served[1], "fallbacks": [{}] * fallbacks}
    if harm_noops is not None:
        state["harm_noops"] = [{"slot": i} for i in range(harm_noops)]
    (d / "state.json").write_text(json.dumps(state))
    cfg = {"loop_id": lid}
    if with_cell_in_config:
        cfg["cell"] = cell
    if routing:
        table = root / "routing" / ("%s.json" % ("SS" if cell == "H" else cell))
        table.parent.mkdir(exist_ok=True)
        table.write_text(json.dumps(plan_p2.routing_table("SS" if cell == "H" else cell)))
        cfg["routing"] = str(table)
    (d / "config.json").write_text(json.dumps(cfg))


def reach(i, base=3):
    return base + i % 3


def world(root, *, planner_stall=False, coder_stall=False, harm_delay=10, cost=None, n=12, n_harm=6, stall_ids=(),
          flagged_ids=()):
    """54 loops. Default: no factor effects, H loops slower by harm_delay slots, QT much cheaper than SS. stall_ids: loops
    that stall whatever their cell. flagged_ids: loops with one load-flagged job."""
    cost = cost or {"SS": (0.2, (0, 0)), "ST": (0.1, (50_000, 50_000)), "QS": (0.1, (20_000, 20_000)),
                    "QT": (0.01, (20_000, 20_000)), "H": (0.2, (0, 0))}
    root.mkdir(parents=True)
    plan_loops = []
    for cell, count in (("SS", n), ("ST", n), ("QS", n), ("QT", n), ("H", n_harm)):
        for i in range(count):
            lid = "%s_%02d" % (cell, i + 1)
            stall = (planner_stall and cell[0] == "Q") or (coder_stall and cell[-1] == "T") or lid in stall_ids
            t17 = reach(i) + (harm_delay if cell == "H" else 0)
            changes = [(1, 19, 8, 1), (5, 18, 6, 2)] if stall else [(1, 19, 8, 1), (t17, 17, 8, 2)]
            claude, tok = cost[cell]
            write_loop(root, lid, cell, changes, claude=claude, open_tok=tok, harm_noops=3 if cell == "H" else 0,
                       block=i + 1, with_cell_in_config=False, flagged=(4,) if lid in flagged_ids else ())
            plan_loops.append({"loop_id": lid, "arm": "D", "cell": cell, "block": i + 1})
    (root / "plan.json").write_text(json.dumps({"seed": 2, "loops": plan_loops, "sigma_used": 0.005}))
    (root / "anchor.json").write_text(json.dumps({"anchor_mean": 1.115, "n": 16}))
    return root


def run(root):
    df, res, skipped = P.analyze(root, PRICES)
    return df, res, skipped


# ---------------------------------------------------------------- the AST rule
@pytest.mark.parametrize("text, tbs, depth, width", [
    ("TOTAL_BATCH_SIZE = 2**19\nDEPTH = 8\nASPECT_RATIO = 64\nHEAD_DIM = 128\n", 2 ** 19, 8, 512),
    ("TOTAL_BATCH_SIZE = 2**19\nTOTAL_BATCH_SIZE = 2**18\nDEPTH = 6\nASPECT_RATIO = 85\nHEAD_DIM = 128\n", 2 ** 18, 6, 512),
    ("TOTAL_BATCH_SIZE = 1 << 17\nDEPTH = 6\nASPECT_RATIO = 64\nHEAD_DIM = 128\n", 2 ** 17, 6, 384),
    ("TOTAL_BATCH_SIZE = 2**19 // 4\nDEPTH = -(-8)\nASPECT_RATIO = 64\nHEAD_DIM = 128\n", 2 ** 17, 8, 512),
    ("TOTAL_BATCH_SIZE = 131072\n", 131072, None, None),
])
def test_parse_big_changes(text, tbs, depth, width):
    r = P.parse_big_changes(text)
    assert (r["TOTAL_BATCH_SIZE"], r["DEPTH"], r["width"]) == (tbs, depth, width)


def test_parse_big_changes_unusable_sources():
    for text in (None, "", "def (:", "TOTAL_BATCH_SIZE = other * 2\n", "TOTAL_BATCH_SIZE = 2**9999\n",
                 "if True:\n    TOTAL_BATCH_SIZE = 2**17\n", "TOTAL_BATCH_SIZE = 2**17\nTOTAL_BATCH_SIZE = foo()\n"):
        assert P.parse_big_changes(text)["TOTAL_BATCH_SIZE"] is None


# ---------------------------------------------------------------- log-rank
def test_logrank_matches_a_hand_computed_example():
    # a: events at 1, 3, 5; b: events at 2, 4, 6. Per event time (at risk a, at risk b, deaths, a's deaths):
    # t1 (3,3,1,1) t2 (2,3,1,0) t3 (2,2,1,1) t4 (1,2,1,0) t5 (1,1,1,1) t6 (0,1,1,0)
    # O_a = 3, E_a = 1/2 + 2/5 + 1/2 + 1/3 + 1/2 + 0 = 67/30, so O - E = 23/30
    # V = 1/4 + 6/25 + 1/4 + 2/9 + 1/4 + 0 = 1091/900, z = (23/30) / sqrt(1091/900) = 23 / sqrt(1091)
    r = P.logrank([1, 3, 5], [1, 1, 1], [2, 4, 6], [1, 1, 1])
    assert r["o_minus_e"] == pytest.approx(float(Fraction(23, 30)))
    assert r["var"] == pytest.approx(float(Fraction(1091, 900)))
    assert r["z"] == pytest.approx(23 / math.sqrt(1091))
    assert r["p_two"] == pytest.approx(math.erfc(r["z"] / math.sqrt(2)))
    assert r["p_two"] == pytest.approx(0.4862218, abs=1e-6)
    assert r["p_a_slower"] == pytest.approx(0.7568891, abs=1e-6)       # a is the faster group here, so slower is unlikely


def test_logrank_swapping_groups_flips_the_sign_only():
    a = P.logrank([1, 3, 5, 7], [1, 1, 0, 1], [2, 4, 6, 8, 9], [1, 1, 1, 0, 1])
    b = P.logrank([2, 4, 6, 8, 9], [1, 1, 1, 0, 1], [1, 3, 5, 7], [1, 1, 0, 1])
    assert a["z"] == pytest.approx(-b["z"]) and a["p_two"] == pytest.approx(b["p_two"])
    assert a["p_a_slower"] == pytest.approx(1 - b["p_a_slower"])


def test_logrank_agrees_with_scipy_with_ties_and_censoring():
    st = pytest.importorskip("scipy.stats")
    rng = np.random.default_rng(3)
    ta, tb = rng.integers(1, 21, 24), rng.integers(1, 21, 24)
    ea, eb = ta < 20, tb < 20          # everything at 20 is censored, as in the P2 analysis
    ours = P.logrank(ta, ea, tb, eb)
    cen = lambda t, e: st.CensoredData(uncensored=t[e], right=t[~e])
    theirs = st.logrank(cen(ta, ea), cen(tb, eb))
    assert ours["z"] == pytest.approx(theirs.statistic) and ours["p_two"] == pytest.approx(theirs.pvalue)
    assert ours["p_a_slower"] == pytest.approx(st.logrank(cen(ta, ea), cen(tb, eb), alternative="less").pvalue)


def test_logrank_no_events_or_identical_groups_is_nan():
    assert math.isnan(P.logrank([20, 20], [0, 0], [20, 20], [0, 0])["z"])
    assert P.logrank([1, 2, 3], [1, 1, 1], [1, 2, 3], [1, 1, 1])["z"] == pytest.approx(0.0)


def test_km_median_hand_computed():
    assert P.km_median([1, 2, 3, 4], [1, 1, 1, 1]) == 2                       # S: 0.75, 0.5 -> second value
    assert P.km_median([1, 2, 3, 4], [0, 1, 1, 1]) == 3                       # S(2) = 2/3, S(3) = 1/3
    assert P.km_median([2, 2, 3, 3], [1, 1, 1, 1]) == 2                       # ties: S(2) = 0.5 exactly
    assert P.km_median(list(range(1, 7)) + [20] * 6, [1] * 6 + [0] * 6) == 6  # exactly half: floating point must not hide it
    assert P.km_median([3, 5, 20, 20, 20, 20], [1, 1, 0, 0, 0, 0]) == math.inf    # S never gets to 0.5: not reached
    assert math.isnan(P.km_median([], []))


def test_survival_reports_median_and_mean_and_they_can_disagree():
    import pandas as pd
    grp = lambda ts, es: pd.DataFrame({"t_2e17": ts, "event_2e17": es})
    lo = grp([3] * 12, [True] * 12)
    hi = grp([3] * 9 + [20] * 3, [True] * 9 + [False] * 3)             # a quarter censored at 20: the mean moves, the median does not
    r = P.survival(hi, lo, "2e17")
    assert r["diff"] == pytest.approx(3 * 17 / 12) and r["median_hi"] == r["median_lo"] == 3 and r["median_diff"] == 0
    hi = grp([3] * 5 + [20] * 7, [True] * 5 + [False] * 7)             # more than half censored: the median is not reached
    assert P.survival(hi, lo, "2e17")["median_diff"] == math.inf
    assert P.survival(lo, hi, "2e17")["median_diff"] == -math.inf
    assert math.isnan(P.survival(hi, hi, "2e17")["median_diff"])


def test_fisher_and_permutation_helpers():
    assert P.fisher(5, 5, 0, 5) == pytest.approx(2 / math.comb(10, 5))
    assert P.fisher(5, 5, 0, 5, "greater") == pytest.approx(1 / math.comb(10, 5))
    assert math.isnan(P.fisher(0, 0, 1, 2))
    r = P.perm(np.zeros(12), np.ones(12), seed=1)
    assert r["estimate"] == 1.0 and r["p"] < 1e-3 and r["n_perm"] == 10001   # 10000 random draws plus the observed split
    r = P.perm([0, 1, 2], [3, 4, 5], seed=1, interval=True)
    assert r["method"] == "exact" and r["n_perm"] == 20 and r["lo"] <= 3 <= r["hi"]


# ---------------------------------------------------------------- one loop, hand-checked
def test_loop_features(tmp_path):
    root = tmp_path / "run"
    root.mkdir()
    changes = [(2, 19, 8, 1), (6, 18, 6, 2), (9, 17, 6, 1), (14, 17, 6, 2)]
    write_loop(root, "ST_01", "ST", changes, claude=0.30, open_tok=(1_000_000, 2_000_000), gpus=(0, 1, 2), wall=3600.0,
               served=(10, 8), fallbacks=2, harm_noops=None)
    (root / "ST_01" / "slots" / "5" / "harm.json").write_text("{}")
    (root / "ST_01" / "slots" / "7" / "harm.json").write_text("{}")
    (root / "ST_01" / "calls.jsonl").write_text((root / "ST_01" / "calls.jsonl").read_text().replace(
        '"served_model": "qwen35-4b"', '"served_model": "gemma4-12b"'))
    df, gpu, anchor, skipped = P.load_run(root, PRICES)
    (r,) = df.to_dict("records")
    assert (r["cell"], r["planner"], r["coder"], r["harm"]) == ("ST", "S", "T", False)
    assert (r["final_tbs"], r["final_depth"], r["stall"]) == (2 ** 17, 6, 0.0)
    assert r["last_change_slot"] == 14 and r["first_2e17_slot"] == 9
    assert (r["t_last_change"], r["event_last_change"], r["t_2e17"], r["event_2e17"]) == (15, True, 10, True)   # slots used
    assert (r["accepts"], r["accepts_backed"]) == (4, 2)
    assert r["claude_usd"] == pytest.approx(0.30)
    assert r["open_usd"] == pytest.approx(1.0 * OPEN_IN["gemma4-12b"] + 2.0 * PRICES["per_million_tokens"]["gemma4-12b"]["output"])
    assert r["model_usd"] == pytest.approx(r["claude_usd"] + r["open_usd"])
    assert (r["n_jobs"], r["gpu_hours"]) == (20, pytest.approx(20.0))
    for p in (0.5, 1, 2, 4):
        assert r["total_usd_gpu_%s" % str(p).replace(".", "p")] == pytest.approx(r["model_usd"] + 20.0 * p)
    assert r["usd_per_backed_change"] == pytest.approx(r["model_usd"] / 2)
    assert r["served_fraction_state"] == pytest.approx(0.8) and r["n_fallbacks"] == 2
    assert r["served_fraction_ledger"] == pytest.approx(1.0)      # table sends constant edits to qwen35-4b, served by qwen
    assert r["harm_noops"] == 2                                   # no state key: counted from harm.json files
    assert (r["n_load_flagged"], r["planner_calls"], r["planner_served_fraction"]) == (0, 2, 1.0)
    assert sorted(gpu.gpu.unique()) == [0, 1, 2] and len(gpu) == 20


def test_never_reaching_2e17_is_censored_and_a_loop_without_changes_too(tmp_path):
    root = tmp_path / "run"
    root.mkdir()
    write_loop(root, "QS_01", "QS", [(1, 19, 8, 1), (4, 18, 6, 2)])
    write_loop(root, "QS_02", "QS", [])
    write_loop(root, "QS_03", "QS", [(1, 17, 8, 2)], slots=12, status="stopped")
    df, *_ = P.load_run(root, PRICES)
    a, b, c = (df[df.loop_id == i].iloc[0] for i in ("QS_01", "QS_02", "QS_03"))
    assert (a.stall, a.event_2e17, a.t_2e17, a.last_change_slot, a.t_last_change) == (1.0, False, 20, 4, 5)
    assert (b.stall, b.event_2e17, b.t_2e17, b.event_last_change, b.t_last_change, b.accepts) == (1.0, False, 20, False, 20, 0)
    assert pd_isna(b.last_change_slot) and pd_isna(b.first_2e17_slot)
    assert (c.slots_completed, c.t_2e17, c.event_2e17, c.stall) == (12, 2, True, 0.0)
    # a stopped loop that never got there would be censored at its own slot count
    write_loop(root, "QS_04", "QS", [], slots=12, status="stopped")
    df, *_ = P.load_run(root, PRICES)
    assert df[df.loop_id == "QS_04"].iloc[0].t_2e17 == 12


def test_event_times_are_slots_used_so_a_stopped_loop_leaves_the_risk_set_first(tmp_path):
    root = tmp_path / "run"
    root.mkdir()
    write_loop(root, "QS_01", "QS", [], slots=3, status="stopped")              # ran slots 0 to 2, no event
    write_loop(root, "QS_02", "QS", [(3, 17, 8, 2)])                             # event in slot 3, its fourth slot
    write_loop(root, "QS_03", "QS", [(0, 17, 8, 2)])                             # event in the very first slot
    df, *_ = P.load_run(root, PRICES)
    t = dict(zip(df.loop_id, df.t_2e17))
    assert t == {"QS_01": 3, "QS_02": 4, "QS_03": 1}                              # the stopped loop is censored before slot 3's event


def pd_isna(x):
    import pandas as pd
    return bool(pd.isna(x))


def test_cells_come_from_plan_then_config_then_id_and_others_are_skipped(tmp_path):
    root = tmp_path / "run"
    root.mkdir()
    write_loop(root, "SS_01", "SS", [(1, 17, 8, 2)], with_cell_in_config=False)       # id prefix
    write_loop(root, "weird", "QT", [(1, 17, 8, 2)])                                   # config cell
    write_loop(root, "A_01", "ST", [(1, 17, 8, 2)], with_cell_in_config=False)       # nothing says what it is
    write_loop(root, "ST_09", "ST", [(1, 17, 8, 2)])
    (root / "plan.json").write_text(json.dumps({"loops": [{"loop_id": "ST_09", "cell": "QS", "block": 4}]}))   # plan wins
    df, _, _, skipped = P.load_run(root, PRICES)
    assert dict(zip(df.loop_id, df.cell)) == {"SS_01": "SS", "ST_09": "QS", "weird": "QT"} and skipped == ["A_01"]
    assert dict(zip(df.loop_id, df.planner)) == {"SS_01": "S", "ST_09": "Q", "weird": "Q"}
    assert df[df.loop_id == "ST_09"].block.iloc[0] == 4


# ---------------------------------------------------------------- sealed gain, two versions
def test_sealed_gain_two_versions(tmp_path):
    root = tmp_path / "run"
    root.mkdir()
    write_loop(root, "SS_01", "SS", [(1, 17, 8, 2)])
    write_loop(root, "SS_02", "SS", [(1, 17, 8, 2)])
    rows = [  # wave, kind, loop, sealed
        (1, "anchor", None, 1.1160), (1, "confirm", "SS_01", 1.0500), (1, "confirm", "SS_02", 1.0600),
        (2, "anchor", None, 1.1180), (2, "confirm", "SS_01", 1.0540), (2, "confirm", "SS_02", 1.0620),
        (3, "anchor", None, None), (3, "confirm", "SS_01", 1.0520)]
    (root / "confirmation").mkdir()
    jl(root / "confirmation" / "results.jsonl", [
        {"wave": w, "kind": k, "loop_id": lid, "arm": "D", "seed": 1001, "official_val_bpb": None if s is None else s - 0.0007,
         "sealed_val_bpb": s, "gpu": 0, "status": "ok" if s is not None else "crash"} for w, k, lid, s in rows])
    rows_c = [  # the unusable wave 3 anchor leaves SS_01's third run to the global anchor mean for the paired version
        (1.1160 - 1.0500), (1.1180 - 1.0540), (statistics.mean([1.1160, 1.1180]) - 1.0520)]
    df, _, anchor, _ = P.load_run(root, PRICES)
    a = df[df.loop_id == "SS_01"].iloc[0]
    glob = statistics.mean([1.1160, 1.1180])
    assert anchor == pytest.approx(glob)
    assert a.sealed_gain_global == pytest.approx(glob - statistics.mean([1.0500, 1.0540, 1.0520]))
    assert a.sealed_gain_paired == pytest.approx(statistics.mean(rows_c))
    b = df[df.loop_id == "SS_02"].iloc[0]
    assert b.sealed_gain_global == pytest.approx(glob - 1.0610)
    assert b.sealed_gain_paired == pytest.approx(statistics.mean([1.1160 - 1.0600, 1.1180 - 1.0620]))


# ---------------------------------------------------------------- the decision rules
GOOD = dict(stall_diff=0.0, slots_diff=0.5)


def test_decide_pass_kill_pivot_and_no_verdict():
    d = P.decide(0.01, GOOD, GOOD, 8.0)
    assert d["verdict"] == "PASS" and d["passed"] and not d["kill"] and not d["pivot"] and all(d["conditions"].values())
    assert P.decide(0.20, GOOD, GOOD, 8.0)["verdict"] == "KILL"                                  # harm not detected
    assert P.decide(0.05, GOOD, GOOD, 8.0)["verdict"] == "KILL"                                  # p must be below 0.05
    assert P.decide(0.01, dict(stall_diff=0.26, slots_diff=0), GOOD, 8.0)["verdict"] == "PIVOT"  # cheap planner stalls more
    assert P.decide(0.01, dict(stall_diff=0, slots_diff=5.5), GOOD, 8.0)["verdict"] == "PIVOT"   # and delays 2^17
    assert P.decide(0.01, dict(stall_diff=0.25, slots_diff=5.0), GOOD, 8.0)["verdict"] == "NO VERDICT"   # margins are strict
    d = P.decide(0.01, GOOD, dict(stall_diff=0.5, slots_diff=0), 8.0)
    assert d["verdict"] == "CODER HURTS" and not d["pivot"] and d["roles"] == dict(planner="cheap tier holds", coder="cheap tier hurts")
    assert P.decide(0.01, GOOD, dict(stall_diff=0, slots_diff=6), 8.0)["verdict"] == "CODER HURTS"
    assert P.decide(0.01, dict(stall_diff=0.5, slots_diff=0), dict(stall_diff=0.5, slots_diff=0), 8.0)["verdict"] == "PIVOT"
    assert P.decide(0.01, GOOD, dict(stall_diff=0, slots_diff=4), 8.0)["verdict"] == "PASS"
    assert P.decide(0.01, GOOD, GOOD, 4.99)["verdict"] == "NO VERDICT" and P.decide(0.01, GOOD, GOOD, 5.0)["verdict"] == "PASS"
    both = P.decide(0.5, dict(stall_diff=0.9, slots_diff=9), GOOD, 8.0)
    assert both["verdict"] == "KILL" and both["pivot"]                                          # KILL has precedence
    nan = math.nan
    n = P.decide(nan, dict(stall_diff=nan, slots_diff=nan), dict(stall_diff=nan, slots_diff=nan), nan)
    assert n["verdict"] == "KILL" and not any(n["conditions"].values())
    n = P.decide(0.01, dict(stall_diff=nan, slots_diff=nan), GOOD, 8.0)
    assert n["verdict"] == "NO VERDICT" and not n["pivot"]


def test_decide_treats_a_median_that_is_not_reached_as_a_delay():
    inf = math.inf
    assert P.decide(0.01, dict(stall_diff=0, slots_diff=inf), GOOD, 8.0)["verdict"] == "PIVOT"      # Q never gets there, S does
    assert P.decide(0.01, dict(stall_diff=0, slots_diff=-inf), GOOD, 8.0)["verdict"] == "PASS"      # S never gets there, Q does
    assert P.decide(0.01, dict(stall_diff=0, slots_diff=inf - inf), GOOD, 8.0)["verdict"] == "NO VERDICT"   # neither does


def test_constants_are_the_prereg_margins():
    c = P.CONSTANTS
    assert (c["harm_alpha"], c["stall_margin"], c["slots_margin"], c["cost_ratio_min"]) == (0.05, 0.25, 5.0, 5.0)
    assert c["n_perm"] == 10000 and c["censor_slot"] == 20 and c["stall_cut"] == 2 ** 17 and c["gpu_prices"] == [0.5, 1, 2, 4]


# ---------------------------------------------------------------- whole-run scenarios
def test_pass_scenario(tmp_path):
    df, res, _ = run(world(tmp_path / "run"))
    assert len(df) == 54 and df.cell.value_counts().to_dict() == {"SS": 12, "ST": 12, "QS": 12, "QT": 12, "H": 6}
    pl, co = res["effects"]["planner"], res["effects"]["coder"]
    assert (pl["n_hi"], pl["n_lo"], co["n_hi"], co["n_lo"]) == (24, 24, 24, 24)        # harm loops are not in the main effects
    assert pl["stall"]["diff"] == 0 and co["stall"]["diff"] == 0 and pl["stall"]["p"] == 1.0
    assert pl["time_2e17"]["diff"] == pytest.approx(0) and co["time_2e17"]["diff"] == pytest.approx(0)
    h = res["harm"]
    assert h["n_h"] == 6 and h["noops_total"] == 18 and h["noops_per_loop"] == 3
    assert h["time_2e17"]["mean_hi"] - h["time_2e17"]["mean_lo"] == pytest.approx(10) and h["p_h_slower"] < 0.01
    assert h["stall"]["rate_h"] == 0 and h["stall"]["p_two"] == 1.0
    cr = res["cost_ratio"]
    assert cr["usd_per_change_ss"] == pytest.approx(0.2) and cr["ratio"] > 5
    gpu_h = 20 * 400 / 3600           # every loop ran 20 jobs of 400 s; one backed change per loop
    assert cr["ratio_with_gpu"]["1"] == pytest.approx((0.2 + gpu_h) / (cr["usd_per_change_qt"] + gpu_h))
    assert cr["ratio_with_gpu"]["4"] < cr["ratio_with_gpu"]["0.5"] < cr["ratio"]
    assert res["decision"]["verdict"] == "PASS"
    cost = pl["log_cost"]["model_usd"]
    assert cost["ratio"] < 1 and cost["p"] < 0.01                                      # Q is cheaper, tested on log dollars


def test_pivot_scenario_when_the_cheap_planner_stalls(tmp_path):
    df, res, _ = run(world(tmp_path / "run", planner_stall=True))
    pl = res["effects"]["planner"]
    assert pl["stall"]["rate_hi"] == 1.0 and pl["stall"]["rate_lo"] == 0.0 and pl["stall"]["diff"] == 1.0
    assert pl["stall"]["p"] == pytest.approx(2 / math.comb(48, 24), abs=1e-9)
    assert pl["time_2e17"]["diff"] > 3 and pl["time_2e17"]["p_two"] < 1e-3 and pl["time_2e17"]["events_hi"] == 0
    assert pl["time_2e17"]["median_hi"] == math.inf and pl["time_2e17"]["median_diff"] == math.inf   # Q never gets there
    co = res["effects"]["coder"]["stall"]
    assert (co["rate_hi"], co["rate_lo"], co["diff"]) == (0.5, 0.5, 0.0)            # QT and QS stall, SS and ST do not
    assert res["decision"]["verdict"] == "PIVOT" and res["decision"]["pivot"]


def test_kill_scenario_when_the_harm_control_is_not_detected(tmp_path):
    df, res, _ = run(world(tmp_path / "run", harm_delay=0))
    assert res["harm"]["p_h_slower"] > 0.05 and res["decision"]["verdict"] == "KILL"
    text = P.build_report(df, res, [])
    assert "with 6 H against 12 SS loops" in text and "weak evidence" in text     # a miss at 6 loops is not proof of a blind substrate


def test_coder_hurts_when_only_the_coder_hurts(tmp_path):
    df, res, _ = run(world(tmp_path / "run", coder_stall=True))
    assert res["effects"]["coder"]["stall"]["diff"] == 1.0 and res["effects"]["planner"]["stall"]["diff"] == 0.0
    assert res["decision"]["verdict"] == "CODER HURTS" and not res["decision"]["pivot"]


def test_cost_condition_can_fail(tmp_path):
    cost = {c: (0.2, (0, 0)) for c in ("SS", "ST", "QS", "QT", "H")}
    df, res, _ = run(world(tmp_path / "run", cost=cost))
    assert res["cost_ratio"]["ratio"] == pytest.approx(1.0)
    assert res["decision"]["verdict"] == "NO VERDICT" and not res["decision"]["conditions"]["qt_cheaper"]


def test_slot_margin_uses_the_median_not_the_mean(tmp_path):
    """P2_PREREG says median slots to 2^17. Five of the 24 Q loops stall (0.21, under the stall margin): their slot 20
    censoring lifts the mean difference over 3 slots, the median difference stays 0, so the verdict is PASS."""
    ids = ("QS_01", "QS_02", "QS_03", "QT_01", "QT_02")
    df, res, _ = run(world(tmp_path / "run", stall_ids=ids))
    pl = res["effects"]["planner"]
    assert pl["time_2e17"]["diff"] > 3 and pl["time_2e17"]["median_diff"] == 0
    assert 0.2 < pl["stall"]["diff"] < 0.25
    d = res["decision"]
    assert d["verdict"] == "PASS" and not d["pivot"] and d["conditions"]["planner_slots_ok"]
    text = P.build_report(df, res, [])
    assert "median 5.0 vs 5.0: 0.0; mean" in text


def test_planner_served_fraction_sees_a_q_loop_whose_backlog_fell_back(tmp_path):
    root = world(tmp_path / "run")
    path = root / "QS_01" / "ledger.jsonl"
    rows = [json.loads(l) for l in path.read_text().splitlines()]
    (backlog,) = [c for c in rows[0]["model_calls"] if c["role"] == "planner_backlog"]
    backlog["served_model"] = "haiku"                                   # walnut was down at slot 0
    jl(path, rows)
    df, res, _ = run(root)
    q = df[df.loop_id == "QS_01"].iloc[0]
    assert (q.planner_calls, q.planner_served_fraction) == (2, 0.5)
    assert math.isnan(q.served_fraction_ledger)                          # QS has no designated open call outside the planner: blind
    assert df[df.loop_id != "QS_01"].planner_served_fraction.eq(1.0).all()
    text = P.build_report(df, res, [])
    assert "not all served by the designated model" in text and "QS_01 (0.50)" in text


def test_planner_served_matches_by_role_and_defaults_to_sonnet():
    table = {"planner_backlog": "qwen35-4b", "planner_widen": "qwen35-4b", "planner": "qwen35-4b", "coder": "sonnet"}
    recs = [{"model_calls": [{"role": "planner_backlog", "served_model": "qwen35-4b"},
                             {"role": "planner_widen", "served_model": "haiku"},
                             {"role": "planner_refresh", "served_model": "qwen35-4b"},       # falls back to the 'planner' key
                             {"role": "coder", "served_model": "haiku"}]}, {"model_calls": None}]
    assert P.planner_served(recs, table) == (3, 2)
    assert P.planner_served([{"model_calls": [{"role": "planner_backlog", "served_model": "claude-sonnet-4"}]}], {}) == (1, 1)
    assert P.planner_served([{"model_calls": [{"role": "planner_backlog", "served_model": ""}]}], {}) == (1, 0)


def test_load_flagged_jobs_are_counted_and_the_sensitivity_verdict_drops_their_loops(tmp_path):
    h_ids = tuple("H_%02d" % i for i in range(1, 7))                       # every harm loop has one flagged job
    df, res, _ = run(world(tmp_path / "run", flagged_ids=h_ids))
    assert df.n_load_flagged.sum() == 6 and df[df.cell == "H"].n_load_flagged.eq(1).all()
    assert res["gpu"]["per_gpu"]["0"]["load_flagged_jobs"] == 6 and res["gpu"]["per_gpu"]["1"]["load_flagged_jobs"] == 0
    s = res["load_flag_sensitivity"]
    assert res["decision"]["verdict"] == "PASS"
    assert (s["n_dropped"], s["n_loops"], s["decision"]["verdict"]) == (6, 48, "KILL")   # without the H loops no harm control is left
    text = P.build_report(df, res, [])
    assert "Without the 6 loops that have a load-flagged job (48 loops left)" in text and "**KILL** (the run itself: PASS)" in text
    df, res, _ = run(world(tmp_path / "clean"))
    assert df.n_load_flagged.sum() == 0 and res["load_flag_sensitivity"]["n_dropped"] == 0
    assert res["load_flag_sensitivity"]["decision"]["verdict"] == "PASS"
    assert "No job was load-flagged" in P.build_report(df, res, [])


def test_quick_decision_agrees_with_the_verdict_on_each_scenario(tmp_path):
    for name, kw in (("a", {}), ("b", dict(planner_stall=True)), ("c", dict(harm_delay=0)), ("d", dict(coder_stall=True))):
        df, res, _ = run(world(tmp_path / name, **kw))
        assert P.quick_decision(df) == res["decision"]


# ---------------------------------------------------------------- per GPU, CLI, report
def test_gpu_breakdown(tmp_path):
    root = world(tmp_path / "run", n=2, n_harm=1)
    jl(root / "launch_sentinels.jsonl", [{"status": "ok", "gpu": g, "official_val_bpb": v} for g, v in
                                         ((0, 1.110), (0, 1.112), (1, 1.120), (1, 1.122), (1, None), (2, 1.0))])
    jl(root / "sentinels.jsonl", [{"status": "ok", "gpu": 0, "official_val_bpb": 1.114}, {"status": "crash", "gpu": 1}])
    df, res, _ = run(root)
    g = res["gpu"]
    assert set(g["per_gpu"]) == {"0", "1", "2"} and g["n_sentinels"] == 6
    assert g["per_gpu"]["0"]["sentinels"] == 3 and g["per_gpu"]["0"]["sentinel_mean"] == pytest.approx((1.110 + 1.112 + 1.114) / 3)
    assert g["per_gpu"]["1"]["sentinel_sd"] == pytest.approx(statistics.stdev([1.120, 1.122]))
    assert sum(v["loop_jobs"] for v in g["per_gpu"].values()) == 9 * 20
    assert g["per_gpu"]["0"]["wall_s_mean"] == 400.0 
    from scipy import stats   # GPU 2 has one sentinel and is left out of the F test
    assert g["sentinel_anova_p"] == pytest.approx(stats.f_oneway([1.110, 1.112, 1.114], [1.120, 1.122])[1])
    assert sum(g["per_gpu"]["0"]["jobs_by_cell"].values()) == g["per_gpu"]["0"]["loop_jobs"]


def test_cli_writes_the_three_files_and_no_em_dash(tmp_path, capsys):
    root = world(tmp_path / "run")
    out = tmp_path / "out"
    assert P.main(["--run", str(root), "--out", str(out)]) == 0
    assert sorted(p.name for p in out.iterdir()) == ["loops.csv", "report.md", "results.json"]
    import pandas as pd
    csv = pd.read_csv(out / "loops.csv")
    assert len(csv) == 54 and {"loop_id", "cell", "planner", "coder", "harm", "final_tbs", "final_depth", "stall",
                               "last_change_slot", "first_2e17_slot", "accepts", "accepts_backed", "claude_usd", "open_usd",
                               "gpu_hours", "total_usd_gpu_0p5", "total_usd_gpu_1", "total_usd_gpu_2", "total_usd_gpu_4",
                               "served_fraction_state", "served_fraction_ledger", "n_fallbacks", "harm_noops",
                               "sealed_gain_global", "sealed_gain_paired"} <= set(csv.columns)
    res = json.loads((out / "results.json").read_text())
    assert res["decision"]["verdict"] == "PASS" and res["constants"]["harm_alpha"] == 0.05
    text = (out / "report.md").read_text()
    assert EM_DASH not in text and EM_DASH not in (out / "results.json").read_text()
    assert "## Verdict: PASS" in text and "Harm control" in text and "Per GPU" in text and "Sealed held-out gain" in text
    assert "Load-flagged jobs (sensitivity)" in text and "median slots to 2^17" in text and "planner served fraction" in text
    assert "load_flag_sensitivity" in res
    assert "verdict PASS" in capsys.readouterr().out
    default_out = P.main(["--run", str(root)])
    assert default_out == 0 and (root / "analysis" / "report.md").exists()


def test_cell_map_labels_an_old_run_and_drops_the_rest(tmp_path):
    root = tmp_path / "old"
    root.mkdir()
    for lid in ("A_01", "C_01", "C_02", "D_01", "D_02", "D_03"):
        write_loop(root, lid, "SS", [(1, 19, 8, 1), (4, 17, 8, 2)], with_cell_in_config=False)
    (root / "plan.json").write_text(json.dumps({"seed": 1, "loops": [{"loop_id": "C_01", "arm": "C"}]}))   # no cells in it
    out = tmp_path / "out"
    assert P.main(["--run", str(root), "--out", str(out), "--cell-map", "C=SS,D=ST"]) == 0
    import pandas as pd
    csv = pd.read_csv(out / "loops.csv")
    assert dict(zip(csv.loop_id, csv.cell)) == {"C_01": "SS", "C_02": "SS", "D_01": "ST", "D_02": "ST", "D_03": "ST"}
    assert "left out: A_01" in (out / "report.md").read_text()
    with pytest.raises(SystemExit):
        P.main(["--run", str(root), "--out", str(out), "--cell-map", "C=XX"])
    with pytest.raises(SystemExit):
        P.main(["--run", str(root), "--out", str(out)])    # nothing has a cell without the map


def test_missing_cells_do_not_crash(tmp_path):
    root = tmp_path / "run"
    root.mkdir()
    for i in range(4):
        write_loop(root, "SS_%02d" % (i + 1), "SS", [(1, 17, 8, 2)])
        write_loop(root, "ST_%02d" % (i + 1), "ST", [(2, 17, 8, 2)])
    df, res, _ = run(root)
    assert res["decision"]["verdict"] == "KILL" and math.isnan(res["effects"]["planner"]["stall"]["diff"])
    assert math.isnan(res["harm"]["p_h_slower"]) and res["effects"]["coder"]["time_2e17"]["diff"] == pytest.approx(1.0)
    out = tmp_path / "out"
    assert P.main(["--run", str(root), "--out", str(out)]) == 0
    assert "could not be evaluated" in (out / "report.md").read_text()
