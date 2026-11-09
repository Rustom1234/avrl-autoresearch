"""Unit tests for avrl/exp/n3_signal_stage0.py on synthetic strata with known ICC and wrong-output collision.

  cd <repository root> && PYTHONDONTWRITEBYTECODE=1 .venv/bin/python -m pytest -q -p no:cacheprovider avrl/exp/n3_tests
"""
import sys
from itertools import combinations
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import ar3_features as F  # noqa: E402
import ar3_stats as S  # noqa: E402
import n3_signal_stage0 as N  # noqa: E402

# ------------------------------------------------------------------ synthetic generator


def synth(n_items=6000, P=8, a=0.6, icc=0.4, c_w=0.3, c_r=0.9, seed=0):
    """Items with Beta item accuracy (mean a, ICC = icc) and outputs: a right pass emits the shared right label with
    probability sqrt(c_r) (else a unique label), a wrong pass the shared wrong label with probability sqrt(c_w) (else unique).
    Two right passes then collide with probability c_r, two wrong passes with probability c_w."""
    rng = np.random.default_rng(seed)
    k = 1.0 / icc - 1.0
    p = rng.beta(a * k, (1 - a) * k, size=n_items)
    right = rng.random((n_items, P)) < p[:, None]
    shared = rng.random((n_items, P)) < np.where(right, np.sqrt(c_r), np.sqrt(c_w))
    uniq = np.arange(n_items * P).reshape(n_items, P) + 10
    ids = np.where(shared, np.where(right, 0, 1), uniq)
    wrong = ~right
    same = np.stack([ids[:, i] == ids[:, j] for i, j in combinations(range(P), 2)], axis=1)
    ww, sww, rr, srr = N.pair_stats(wrong, same)
    correct = right.astype(int)
    dec = np.array([N.plurality_correct(ids[i, :3].tolist(), correct[i, :3].tolist()) for i in range(n_items)])
    agree3 = np.mean([ids[:, i] == ids[:, j] for i, j in combinations(range(3), 2)], axis=0)
    return N.Stratum("syn", "verifier", P, wrong.sum(1), ww, sww, rr, srr, (1 - dec).astype(int), np.arange(n_items), {"AGREE": -agree3})


# ------------------------------------------------------------------ batch statistics


def test_auroc_batch_matches_reference_with_ties_and_weights():
    rng = np.random.default_rng(1)
    y = rng.integers(0, 2, 200)
    sc = rng.integers(0, 6, 200).astype(float)  # many ties
    W = rng.integers(0, 4, (5, 200)).astype(float)
    got = N.auroc_batch(sc, y, W)
    for b in range(5):
        assert got[b] == pytest.approx(S.weighted_auroc(y, sc, W[b]), abs=1e-12)
    from sklearn.metrics import roc_auc_score
    assert got[0] == pytest.approx(roc_auc_score(y, sc, sample_weight=W[0]), abs=1e-12)


def test_auroc_batch_extremes_and_single_class():
    one = np.ones((1, 6))
    assert N.auroc_batch(np.array([1, 2, 3, 4, 5, 6.0]), np.array([0, 0, 0, 1, 1, 1]), one)[0] == 1.0
    assert N.auroc_batch(np.array([1, 2, 3, 4, 5, 6.0]), np.array([1, 1, 1, 0, 0, 0]), one)[0] == 0.0
    assert N.auroc_batch(np.zeros(6), np.array([0, 0, 0, 1, 1, 1]), one)[0] == 0.5
    assert np.isnan(N.auroc_batch(np.arange(6.0), np.zeros(6), one)[0])


def test_recall_batch_hand_example_with_fractional_ties():
    sc, y = np.array([3.0, 2.0, 2.0, 1.0]), np.array([1, 0, 1, 1])
    one = np.ones((1, 4))
    assert N.recall_batch(sc, y, one, 0.25)[0] == pytest.approx(1 / 3)  # budget 1 item: only the top one
    assert N.recall_batch(sc, y, one, 0.5)[0] == pytest.approx(1.5 / 3)  # top one plus half of the tied pair (one of which is wrong)
    assert N.recall_batch(sc, y, one, 1.0)[0] == pytest.approx(1.0)
    assert np.isnan(N.recall_batch(sc, np.zeros(4), one)[0])


def test_recall_batch_integer_weights_equal_duplicated_items():
    rng = np.random.default_rng(2)
    sc, y = rng.integers(0, 5, 30).astype(float), rng.integers(0, 2, 30)
    w = rng.integers(0, 4, 30)
    rep = np.repeat(np.arange(30), w)
    a = N.recall_batch(sc, y, w[None, :].astype(float))[0]
    b = N.recall_batch(sc[rep], y[rep], np.ones((1, len(rep))))[0]
    assert a == pytest.approx(b, abs=1e-12)


def test_icc_batch_matches_anova_on_duplicated_items():
    rng = np.random.default_rng(3)
    n, m = 7, rng.integers(0, 8, 40)
    w = rng.integers(0, 3, 40)
    rep = np.repeat(np.arange(40), w)
    x = np.zeros((len(rep), n))
    for r, i in enumerate(rep):
        x[r, : m[i]] = 1
    N_, grand = len(rep), x.mean()
    msb = n * ((x.mean(1) - grand) ** 2).sum() / (N_ - 1)
    msw = ((x - x.mean(1, keepdims=True)) ** 2).sum() / (N_ * (n - 1))
    want = (msb - msw) / (msb + (n - 1) * msw)
    assert N.icc_batch(m, n, w[None, :].astype(float))[0] == pytest.approx(want, abs=1e-12)


# ------------------------------------------------------------------ pass level


def test_pair_stats_hand_example():
    wrong = np.array([[1, 1, 0], [1, 0, 0]], bool)  # pairs in order (0,1), (0,2), (1,2)
    same = np.array([[1, 0, 0], [0, 0, 1]], bool)
    ww, sww, rr, srr = N.pair_stats(wrong, same)
    assert ww.tolist() == [1, 0] and sww.tolist() == [1, 0]
    assert rr.tolist() == [0, 1] and srr.tolist() == [0, 1]


def test_plurality_correct_ties_and_rounding():
    assert N.plurality_correct([5, 5, 7], [1, 1, 0]) == 1
    assert N.plurality_correct([5, 6, 7], [0, 1, 1]) == 0  # all differ: earliest pass wins
    assert N.plurality_correct([9, 9, 4], [1, 0, 1]) == 0  # members mean 0.5 rounds to 0, as code_rewards does
    assert N.plurality_correct([2, 3, 3], [1, 1, 1]) == 1


def test_critic_output_equality_uses_the_plus_minus_one_matching():
    assert F.sim({5}, {6}) == 1.0 and F.sim({5}, {5}) == 1.0
    assert F.sim(set(), set()) == 1.0
    assert F.sim({5}, {7}) < 1.0 and F.sim({5}, set()) < 1.0 and F.sim({5, 9}, {5}) < 1.0


def test_merge_kinds_merges_a_family_with_a_small_kind():
    counts = {"N-big": 63, "N-mid": 92, "N-stale": 54, "V-body": 26, "V-coh": 23}
    m = N.merge_kinds(counts)
    assert m["V-body"] == m["V-coh"] == "V" and m["N-big"] == "N-big" and m["N-stale"] == "N-stale"
    assert N.merge_kinds({"N-big": 63, "N-mid": 41, "V-body": 40, "V-coh": 50})["V-body"] == "V-body"


def test_ids_treat_empty_and_nan_strings_as_values():
    ids = N._ids(["", "nan", "", "3"])
    assert ids[0] == ids[2] and len(set(ids.tolist())) == 3


def test_keep_stratum_floors():
    def mk(n, wrong):
        y = np.r_[np.ones(wrong), np.zeros(n - wrong)].astype(int)
        z = np.zeros(n)
        return N.Stratum("s", "critic", 5, z, z, z, z, z, y, np.arange(n))
    assert N.keep_stratum(mk(60, 20))[0]
    assert not N.keep_stratum(mk(39, 20))[0]
    assert not N.keep_stratum(mk(60, 9))[0]
    assert not N.keep_stratum(mk(60, 52))[0]  # only 8 correct decisions


# ------------------------------------------------------------------ synthetic strata with known ICC and collision


@pytest.mark.parametrize("a,icc,c_w,c_r", [(0.6, 0.4, 0.3, 0.9), (0.4, 0.6, 0.8, 0.5), (0.8, 0.2, 0.05, 1.0)])
def test_stratum_stats_recover_known_icc_and_collision(a, icc, c_w, c_r):
    st = synth(a=a, icc=icc, c_w=c_w, c_r=c_r, seed=11)
    p = N.point_stats(st)
    assert p["a"] == pytest.approx(a, abs=0.02)
    assert p["icc"] == pytest.approx(icc, abs=0.03)
    assert p["c_w"] == pytest.approx(c_w, abs=0.03)
    assert p["c_r"] == pytest.approx(c_r, abs=0.03)


def test_collision_not_icc_decides_whether_agreement_finds_errors():
    scatter = N.point_stats(synth(a=0.6, icc=0.4, c_w=0.02, c_r=1.0, seed=5))
    collide = N.point_stats(synth(a=0.6, icc=0.4, c_w=0.95, c_r=1.0, seed=6))
    assert scatter["icc"] == pytest.approx(collide["icc"], abs=0.05)  # same within-item error correlation
    assert scatter["auc_AGREE"] > 0.75
    assert collide["auc_AGREE"] < scatter["auc_AGREE"] - 0.15  # unanimous wrong answers hide the errors
    assert collide["auc_AGREE"] < 0.65
    assert scatter["rec_AGREE"] > collide["rec_AGREE"] + 0.3


def test_bootstrap_is_seeded_and_brackets_the_point_estimate():
    st = synth(n_items=400, seed=7)
    a = N.bootstrap_stratum(st, 300, seed=3, chunk=100)
    b = N.bootstrap_stratum(st, 300, seed=3, chunk=100)
    assert np.array_equal(a["c_w"], b["c_w"], equal_nan=True)
    p = N.point_stats(st)
    for k in ("a", "icc", "c_w", "auc_AGREE", "rec_AGREE"):
        lo, hi = N.ci(a[k])
        assert lo <= p[k] <= hi


def test_cluster_bootstrap_draws_whole_clusters():
    st = synth(n_items=60, seed=8)
    st.cluster = np.repeat(np.arange(20), 3)  # 20 entries of 3 items
    rng = np.random.default_rng(0)
    W = rng.multinomial(20, np.full(20, 0.05), size=4)[:, st.cluster]
    assert (W.reshape(4, 20, 3)[:, :, 0] == W.reshape(4, 20, 3)[:, :, 2]).all()
    out = N.bootstrap_stratum(st, 50, seed=1, chunk=25)
    assert np.isfinite(out["auc_AGREE"]).sum() > 40


# ------------------------------------------------------------------ T1, T2, T3 rules


def test_loso_recovers_an_exact_linear_law_and_penalises_noise():
    rng = np.random.default_rng(4)
    X = rng.normal(size=(9, 2))
    mae, r2 = N.loso_scores(X, 0.5 + 0.1 * X[:, 0] - 0.2 * X[:, 1])
    assert mae < 1e-9 and r2 == pytest.approx(1.0)
    _, r2n = N.loso_scores(rng.normal(size=(9, 3)), rng.normal(size=9))
    assert r2n < 0.2


def test_t1_models_collision_beats_icc_when_collision_is_the_cause():
    rng = np.random.default_rng(9)
    S_ = 12
    a, c_r, c_w, icc = rng.uniform(0.3, 0.9, S_), rng.uniform(0.5, 1, S_), rng.uniform(0, 1, S_), rng.uniform(0.2, 0.7, S_)
    auc = 0.9 - 0.5 * c_w + 0.05 * a
    m = N.t1_models(a, c_r, c_w, icc, auc)
    assert m["mae_coll"] < 0.01 and m["r2_coll"] > 0.99 and m["gap"] > 0.2
    assert N.t1_verdict(m, S_) == "PASS"


def test_t1_verdict_rules():
    ok = {"mae_coll": 0.05, "r2_coll": 0.8, "r2_icc": 0.5, "gap": 0.3}
    assert N.t1_verdict(ok, 8) == "PASS"
    assert N.t1_verdict({**ok, "mae_coll": 0.07}, 8) == "INCONCLUSIVE"
    assert N.t1_verdict({**ok, "r2_icc": 0.75, "gap": 0.05}, 8) == "KILL"
    assert N.t1_verdict({**ok, "r2_icc": 0.65, "gap": 0.15}, 8) == "INCONCLUSIVE"
    assert N.t1_verdict(ok, 5) == "INCONCLUSIVE"


def test_t3_verdict_needs_more_than_half_wide():
    assert N.t3_verdict([0.2, 0.2, 0.2, 0.05, 0.05])[0] == "UNPOWERED"
    assert N.t3_verdict([0.2, 0.2, 0.05, 0.05])[0] == "POWERED"  # exactly half is not more than half
    assert N.t3_verdict([0.2, 0.2, 0.05, 0.05])[1] == 2


def _boot(rng, mean, B=400, sd=0.01):
    return mean + sd * rng.standard_normal(B)


def test_t2_pass_fail_and_ties():
    rng = np.random.default_rng(10)
    pv, bv = {"AGREE": 0.9, "DIFF": 0.3}, {"AGREE": _boot(rng, 0.9), "DIFF": _boot(rng, 0.3)}
    pc, bc = {"AGREE": 0.3, "VERB": 0.8, "DIFF": 0.35}, {"AGREE": _boot(rng, 0.3), "VERB": _boot(rng, 0.8), "DIFF": _boot(rng, 0.35)}
    r = N.t2_verdict(pv, bv, pc, bc)
    assert r["verdict"] == "PASS" and r["best_verifier"] == "AGREE" and r["best_critic"] == "VERB"
    # agreement also best in the critic: no reversal
    pc2, bc2 = {"AGREE": 0.8, "VERB": 0.5, "DIFF": 0.3}, {"AGREE": _boot(rng, 0.8), "VERB": _boot(rng, 0.5), "DIFF": _boot(rng, 0.3)}
    assert N.t2_verdict(pv, bv, pc2, bc2)["verdict"] == "FAIL"
    # VERB leads on the point estimate but the interval includes zero
    pc3, bc3 = {"AGREE": 0.5, "VERB": 0.52, "DIFF": 0.3}, {"AGREE": _boot(rng, 0.5, sd=0.1), "VERB": _boot(rng, 0.52, sd=0.1), "DIFF": _boot(rng, 0.3)}
    assert N.t2_verdict(pv, bv, pc3, bc3)["verdict"] == "FAIL"
    # agreement does not clearly beat the difficulty control in the verifier
    pv4, bv4 = {"AGREE": 0.5, "DIFF": 0.49}, {"AGREE": _boot(rng, 0.5, sd=0.1), "DIFF": _boot(rng, 0.49, sd=0.1)}
    assert N.t2_verdict(pv4, bv4, pc, bc)["verdict"] == "FAIL"


def test_role_signals_is_a_macro_average():
    point = {"s1": {"rec_AGREE": 0.2}, "s2": {"rec_AGREE": 0.6}}
    boot = {"s1": {"rec_AGREE": np.array([0.1, 0.3])}, "s2": {"rec_AGREE": np.array([0.5, 0.7])}}
    p, b = N.role_signals(point, boot, ["s1", "s2"], ["AGREE"])
    assert p["AGREE"] == pytest.approx(0.4) and b["AGREE"].tolist() == pytest.approx([0.3, 0.5])


def test_final_verdict():
    assert N.final_verdict("PASS", "PASS", "POWERED") == "PROMOTE"
    assert N.final_verdict("PASS", "PASS", "UNPOWERED").startswith("PROMOTE (to enlarge n only")
    assert N.final_verdict("PASS", "FAIL", "POWERED") == "STOP"
    assert N.final_verdict("KILL", "PASS", "POWERED") == "STOP"
    assert N.final_verdict("INCONCLUSIVE", "PASS", "POWERED") == "STOP"


# ------------------------------------------------------------------ real data gate and prereg lock


def test_prereg_hash_matches():
    assert len(N.verify_prereg()) == 64


def test_loader_reproduces_the_published_numbers():
    p0 = N.load_verifier("p0", "p0", "answer")
    crit, chk = N.load_critic()
    r = N.repro_checks(chk, p0)
    assert r["all_ok"], r
    assert p0.n_items == 3000 and p0.n_pass == 8
    assert sorted(s.name for s in crit) == ["N-big", "N-mid", "N-stale", "V", "clean"]
    assert {s.name: s.n_items for s in crit} == {"N-big": 63, "N-mid": 92, "N-stale": 54, "V": 49, "clean": 41}


# ------------------------------------------------------------------ exploratory helpers and the small-n guard


def _mini(name, y, entry, agree=None, weight=None):
    n = len(y)
    z = np.zeros(n)
    agree = np.full(n, -1.0) if agree is None else np.asarray(agree, float)
    return N.Stratum(name, "critic", 5, z + 1, z + 1, z, z + 1, z + 1, np.asarray(y), np.unique(entry, return_inverse=True)[1],
                     {"AGREE": agree, "VERB": z.copy(), "DIFF": z.copy()}, {"item_ids": [f"{name}{i}" for i in range(n)], "entry": np.asarray(entry)}, weight)


def test_pool_strata_adds_counts_and_shares_entry_clusters():
    a = _mini("a", [1, 0, 0], entry=[10, 11, 12])
    b = _mini("b", [1, 1], entry=[11, 13])  # entry 11 appears in both strata
    p = N.pool_strata("ab", [a, b])
    assert p.n_items == 5 and int(p.y.sum()) == 3 and p.n_clusters == 4
    assert p.cluster[1] == p.cluster[3]  # same entry, same cluster
    assert "DIFF" not in p.scores and set(p.scores) == {"AGREE", "VERB"}
    assert p.extra["item_ids"] == ["a0", "a1", "a2", "b0", "b1"]


def test_unanimity_hand_example_and_weights():
    st = _mini("u", y=[1, 1, 0, 0], entry=[0, 1, 2, 3], agree=[-1.0, -0.5, -1.0, -1.0])
    u = N.unanimity(st)
    assert u["unanimous_given_wrong"] == pytest.approx(0.5) and u["unanimous_given_right"] == pytest.approx(1.0)
    assert u["wrong_given_unanimous"] == pytest.approx(1 / 3)
    st.weight = np.array([3.0, 1.0, 1.0, 1.0])
    assert N.unanimity(st)["unanimous_given_wrong"] == pytest.approx(0.75)


def test_fixed_weight_equals_duplicated_items():
    st = synth(n_items=50, seed=12)
    w = np.random.default_rng(0).integers(1, 4, 50).astype(float)
    rep = np.repeat(np.arange(50), w.astype(int))
    dup = N.Stratum("dup", "verifier", st.n_pass, st.m[rep], st.ww[rep], st.sww[rep], st.rr[rep], st.srr[rep], st.y[rep], np.arange(len(rep)),
                    {"AGREE": st.scores["AGREE"][rep]})
    st.weight = w
    a, b = N.point_stats(st), N.point_stats(dup)
    for k in ("a", "icc", "c_w", "c_r", "auc_AGREE", "rec_AGREE"):
        assert a[k] == pytest.approx(b[k], abs=1e-9)


def test_run_t1_refuses_to_fit_with_fewer_than_six_strata():
    sts = [synth(n_items=300, seed=i) for i in range(3)]
    for i, s in enumerate(sts):
        s.name = f"s{i}"
    point = {s.name: N.point_stats(s) for s in sts}
    boot = {s.name: N.bootstrap_stratum(s, 20, i, chunk=20) for i, s in enumerate(sts)}
    r = N.run_t1(sts, point, boot, 20)
    assert r["verdict"] == "INCONCLUSIVE" and "r2_coll" not in r


def test_run_t1_fits_with_enough_strata():
    sts = []
    for i, c_w in enumerate([0.02, 0.1, 0.3, 0.5, 0.7, 0.9]):
        s = synth(n_items=600, a=0.5 + 0.05 * (i % 3), icc=0.3 + 0.05 * (i % 2), c_w=c_w, c_r=0.9, seed=20 + i)
        s.name = f"s{i}"
        sts.append(s)
    point = {s.name: N.point_stats(s) for s in sts}
    boot = {s.name: N.bootstrap_stratum(s, 30, i, chunk=30) for i, s in enumerate(sts)}
    r = N.run_t1(sts, point, boot, 30)
    assert r["verdict"] in ("PASS", "KILL", "INCONCLUSIVE") and np.isfinite(r["r2_coll"]) and r["n_boot_ok"] > 0


def test_explore_table_renders_a_row_per_stratum():
    st = synth(n_items=200, seed=13)
    row = {"name": "x", "role": "verifier", "items": 200, "clusters": 200, "wrong_dec": int(st.y.sum()), "right_dec": 200 - int(st.y.sum()),
           "point": N.point_stats(st), "ci90": {k: N.ci(v) for k, v in N.bootstrap_stratum(st, 40, 1, chunk=40).items()}, **N.unanimity(st)}
    out = N.build_explore_table([row, row])
    assert out.count("\n") == 4 and out.splitlines()[2].startswith("| x |")


def test_spearman_vs_auc_hand_example():
    mk = lambda n, cw, icc, a, auc: {"name": n, "point": {"c_w": cw, "icc": icc, "a": a, "c_r": 1.0 - cw, "auc_AGREE": auc}}
    rows = [mk("x1", 0.1, 0.5, 0.9, 0.9), mk("x2", 0.2, 0.1, 0.8, 0.8), mk("x3", 0.3, 0.4, 0.7, 0.7), mk("x4", 0.4, 0.2, 0.6, 0.6), mk("other", 0.9, 0.9, 0.9, 0.1)]
    r = N.spearman_vs_auc(rows, ["x1", "x2", "x3", "x4"])
    assert r["strata"] == ["x1", "x2", "x3", "x4"]
    assert r["c_w"] == pytest.approx(-1.0) and r["a"] == pytest.approx(1.0) and r["c_r"] == pytest.approx(1.0)
    assert r["icc"] == pytest.approx(0.4)
