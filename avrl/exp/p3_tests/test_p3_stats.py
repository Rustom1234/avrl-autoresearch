"""Judging, AUROC, escalation frontier, interpolation and entry-clustered bootstrap of p3_effort_analyze."""
import json
import math

import numpy as np
import pandas as pd
import pytest
from sklearn.metrics import roc_auc_score

import ar3_features as F
import p3_effort_analyze as A
import p3_effort_run as P
from p3_helpers import make_frame, make_items

AR3 = P.AR3_DIR


# ---------------------------------------------------------------- judging

def test_correct_k1_clean_control_needs_no_lines():
    ctl = make_items(3)[0]
    assert A.correct_k1(ctl, set()) is True
    assert A.correct_k1(ctl, {7}) is False


def test_correct_k1_plant_needs_a_line_in_accept_lines():
    plant = make_items(3)[1]  # accept_lines 4, 5, 6
    assert A.correct_k1(plant, {5}) is True
    assert A.correct_k1(plant, {4}) is True
    assert A.correct_k1(plant, {9}) is False  # flagged elsewhere
    assert A.correct_k1(plant, set()) is False  # missed
    assert A.correct_k1(plant, {9, 6}) is True  # any reported line in the accept set


def test_correct_k1_matches_ar3_parquet_on_real_items():
    """k=1 correctness recomputed from the stored AR3 pass 1 calls equals AR3's own h1_correct and s1_correct."""
    if not (AR3 / "items.parquet").exists():
        pytest.skip("AR3 data not present")
    items, _ = P.select_items(AR3)
    ids = {it["item_id"] for it in items}
    sonnet = A.load_sonnet_pass1(AR3, ids)
    df = pd.read_parquet(AR3 / "items.parquet").set_index("item_id")
    assert set(sonnet) == ids
    for it in items:
        assert float(A.correct_k1(it, sonnet[it["item_id"]][0]["lines"])) == df.loc[it["item_id"], "s1_correct"]
    hp = F.load_calls(AR3, ["burn", "test"], ids)
    for it in items:
        assert float(A.correct_k1(it, hp[it["item_id"]]["haiku"][1]["lines"])) == df.loc[it["item_id"], "h1_correct"]


# ---------------------------------------------------------------- AUROC

def test_auroc_wrong_perfect_reversed_and_tied():
    cor = np.array([1, 1, 1, 0, 0])
    assert A.auroc_wrong(cor, [90, 85, 80, 40, 30]) == 1.0  # wrong verdicts have the lowest confidence
    assert A.auroc_wrong(cor, [30, 40, 50, 80, 90]) == 0.0
    assert A.auroc_wrong(cor, [70] * 5) == 0.5
    assert math.isnan(A.auroc_wrong([1, 1, 1], [60, 70, 80]))  # no wrong verdicts


def test_auroc_wrong_matches_sklearn_with_and_without_weights():
    rng = np.random.default_rng(3)
    cor = (rng.random(80) < 0.7).astype(int)
    conf = rng.integers(40, 95, 80)
    w = rng.uniform(0.1, 2.0, 80)
    assert A.auroc_wrong(cor, conf) == pytest.approx(roc_auc_score(1 - cor, 100 - conf))
    assert A.auroc_wrong(cor, conf, w) == pytest.approx(roc_auc_score(1 - cor, 100 - conf, sample_weight=w))


# ---------------------------------------------------------------- frontier

def test_escalation_probs_without_ties_takes_the_lowest():
    e = A.escalation_probs([50, 90, 70, 60], 0.5)
    assert list(e) == [1, 0, 0, 1]


def test_escalation_probs_splits_ties_equally_and_sums_to_q_times_n():
    conf = [60, 60, 60, 60, 80, 90]
    e = A.escalation_probs(conf, 0.5)  # 3 of 6: all four tied items share 3
    assert list(e[:4]) == [0.75] * 4 and list(e[4:]) == [0, 0]
    rng = np.random.default_rng(1)
    c = rng.integers(50, 60, 40).astype(float)
    for q in A.Q_GRID:
        assert A.escalation_probs(c, q).sum() == pytest.approx(q * 40)
    assert A.escalation_probs(c, 0).sum() == 0 and (A.escalation_probs(c, 1) == 1).all()


def test_frontier_ends_and_hand_example():
    h_cor, h_cost, conf = [1, 0, 0, 1], [0.01, 0.01, 0.01, 0.01], [90, 40, 50, 80]
    s_cor, s_cost = [1, 1, 0, 1], [0.1, 0.1, 0.1, 0.1]
    f = A.frontier(h_cor, h_cost, conf, s_cor, s_cost, qs=(0, 0.25, 0.5, 1))
    assert f["cost"] == pytest.approx([0.01, 0.035, 0.06, 0.11])
    assert f["acc"] == pytest.approx([0.5, 0.75, 0.75, 0.75])  # q=.25 swaps item 2 (wrong->right); q=.5 swaps item 3 (wrong->wrong)
    full = A.frontier(h_cor, h_cost, conf, s_cor, s_cost)
    assert full["cost"][0] == pytest.approx(0.01) and full["acc"][-1] == pytest.approx(np.mean(s_cor))
    assert np.all(np.diff(full["cost"]) > 0)


def test_frontier_weighted_normalises_weights():
    f1 = A.frontier([1, 0], [0.01, 0.01], [90, 50], [1, 1], [0.1, 0.1], w=[1, 3], qs=(0, 1))
    assert f1["acc"] == pytest.approx([0.25, 1.0])
    f2 = A.frontier([1, 0], [0.01, 0.01], [90, 50], [1, 1], [0.1, 0.1], w=[10, 30], qs=(0, 1))
    assert f2["acc"] == pytest.approx(f1["acc"])


def test_acc_at_cost_interpolates_and_returns_nan_outside():
    costs, accs = [0.01, 0.02, 0.04], [0.5, 0.6, 0.9]
    assert A.acc_at_cost(costs, accs, 0.01) == pytest.approx(0.5)
    assert A.acc_at_cost(costs, accs, 0.03) == pytest.approx(0.75)
    assert A.acc_at_cost(costs, accs, 0.04) == pytest.approx(0.9)
    assert math.isnan(A.acc_at_cost(costs, accs, 0.005)) and math.isnan(A.acc_at_cost(costs, accs, 0.05))


def test_matched_cost_diff_sign_follows_which_arm_is_better():
    fr = pd.DataFrame({"w": [1.0] * 4, "s_cor": [1.0] * 4, "s_cost": [0.1] * 4,
                       "cor_low": [1.0, 1.0, 0.0, 0.0], "conf_low": [90, 90, 40, 40], "cost_low": [0.01] * 4,
                       "cor_high": [1.0, 1.0, 0.0, 0.0], "conf_high": [90, 90, 40, 40], "cost_high": [0.06] * 4})
    # effort-first costs 0.06 and has accuracy 0.5; model-first at 0.06 escalates q=0.5, both wrong items, accuracy 1.0
    assert A.matched_cost_diff(fr, "low", "high") == pytest.approx(0.5)
    fr["s_cor"] = 0.0  # Sonnet is wrong everywhere: escalating the two wrong items gains nothing
    assert A.matched_cost_diff(fr, "low", "high") == pytest.approx(0.0)
    assert math.isnan(A.matched_cost_diff(fr.assign(cost_high=0.5), "low", "high"))  # outside the frontier range


# ---------------------------------------------------------------- bootstrap

def test_cluster_ci_resamples_whole_entries_and_is_deterministic():
    fr = make_frame(40)
    sizes = fr.groupby("entry_key").size().to_dict()

    def whole_clusters(f):
        counts = f.groupby("entry_key").size()
        return float(any(c % sizes[k] for k, c in counts.items()))

    ci = A.cluster_ci(fr, whole_clusters, 60, 1)
    assert ci["est"] == ci["lo"] == ci["hi"] == 0.0 and ci["n_clusters"] == 20 and ci["draws_ok"] == 60

    def mean_w(f):
        return float(f.w.mean())

    a, b = A.cluster_ci(fr, mean_w, 200, 5), A.cluster_ci(fr, mean_w, 200, 5)
    assert a == b
    assert a["lo"] < a["est"] < a["hi"]


def test_cluster_ci_draws_entries_not_items():
    """With two perfectly correlated items per entry the clustered interval is as wide as the interval on one item per entry,
    and wider than an item-level bootstrap would give."""
    rng = np.random.default_rng(0)
    x = np.repeat(rng.normal(size=30), 2)
    fr = pd.DataFrame({"entry_key": np.repeat(np.arange(30), 2), "x": x})
    ci = A.cluster_ci(fr, lambda f: float(f.x.mean()), 1000, 2)
    iid_sd = fr.x.std() / math.sqrt(len(fr))
    assert (ci["hi"] - ci["lo"]) / 3.29 > 1.2 * iid_sd


def test_verdicts():
    ci = lambda est, lo, hi: {"est": est, "lo": lo, "hi": hi}
    assert A.h1_verdict(ci(0.08, 0.02, 0.14)) == "high_better"
    assert A.h1_verdict(ci(-0.08, -0.15, -0.01)) == "high_worse"
    assert A.h1_verdict(ci(0.01, -0.03, 0.04)) == "no_effect_within_margin"
    assert A.h1_verdict(ci(0.03, -0.05, 0.11)) == "inconclusive"
    assert A.h1_verdict(ci(0.03, 0.01, 0.045)) == "no_effect_within_margin"
    assert A.h1_verdict(ci(0.04, 0.01, 0.12)) == "inconclusive"  # excludes 0 but smaller than the effect of interest
    assert A.h1_verdict(ci(float("nan"), float("nan"), float("nan"))) == "undefined"
    assert A.h2_verdict(ci(0.05, 0.01, 0.09)) == "model_first_better"
    assert A.h2_verdict(ci(-0.05, -0.09, -0.01)) == "effort_first_better"
    assert A.h2_verdict(ci(0.0, -0.04, 0.05)) == "no_difference"


def test_analyze_split_end_to_end_on_synthetic_frame():
    fr = make_frame(80, seed=4)
    res = A.analyze_split(fr, ["low", "high"], B=40, seed=1)
    assert res["n_items"] == 80 and res["n_entries"] == 40
    lo, hi = res["levels"]["low"], res["levels"]["high"]
    assert hi["auroc"] > lo["auroc"]  # high has informative confidence by construction
    assert res["h1"]["est"] == pytest.approx(hi["auroc_w"] - lo["auroc_w"])  # primary is the weighted contrast
    assert res["h1"]["unweighted"]["est"] == pytest.approx(hi["auroc"] - lo["auroc"])
    assert res["h1"]["est"] > 0.2 and res["h1"]["verdict"] in ("high_better", "inconclusive")
    assert res["h2"]["est"] == pytest.approx(A.matched_cost_diff(fr, "low", "high", True))
    assert res["token_ratio_median_high_over_low"] == pytest.approx(900 / 400)
    assert res["kill_no_effort_spread_and_null"] is False
    assert set(res["ci90"]) >= {"auroc_low", "auroc_high_w", "auroc_high_minus_low", "auroc_high_minus_low_w", "matched_low_to_high", "matched_low_to_high_w"}
    assert len(res["frontier"]["low"]["unweighted"]["q"]) == 11
    assert set(res["matched_cost_grid_point_estimates"]) == {"low_to_high", "high_to_low"}
    json.dumps(A._clean(res))  # serialisable with nan mapped to null


def test_analyze_split_handles_an_empty_frame():
    fr = A.build_frame([], {"low": {}, "high": {}}, {}, {}, ["low", "high"])
    res = A.analyze_split(fr, ["low", "high"], B=10)
    assert res["n_items"] == 0 and "h1" not in res
