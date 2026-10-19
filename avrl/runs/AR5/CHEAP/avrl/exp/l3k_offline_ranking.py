#!/usr/bin/env python3
"""L3k: full-label offline ranking diagnostic, no sequential loop.

Question: with all 3,000 gold labels available, does a logistic regression over the cheap features rank the
proxy errors into a global top 750 at least as well as the agreement threshold order? This is a ceiling
diagnostic for the ranking itself, not a deployable allocator. Gold labels are used for out-of-fold training
and for scoring, so no number here is available to an online allocator.

Arms (one global selection of 750 of the 3,000 groups per arm; no sequential loop, no verifier budget):
  threshold           sort by agreement ascending, then n_distinct descending (the ThresholdAllocator key). No model.
  six                 out-of-fold probability from a regression on the six cheap features.
  twenty_two          out-of-fold probability from a regression on the six cheap features plus pc0 to pc15.
  pc_only             out-of-fold probability from a regression on pc0 to pc15 alone.
  six_scaled          secondary, not pre-registered: six, inside a StandardScaler pipeline.
  twenty_two_scaled   secondary, not pre-registered: twenty_two, inside a StandardScaler pipeline.

Model: LogisticRegression(max_iter=1000), unscaled for the primary arms, as in allocators.LearnedAllocator.
Folds: StratifiedKFold(n_splits=5, shuffle=True, random_state=fold_seed) on proxy_error. Every group gets one
out-of-fold probability. Fold seed 0 is the primary read; fold seeds 0 to 7 give the robustness mean, min and max.
Ranking: global top 750 by out-of-fold probability, unclipped, stable sort (equal scores keep group index order).

Pool: runs/p0/feats.parquet, read only. CPU only. No GPU, no ssh, no pip. allocators.py is not imported.
The context numbers (E03 per-step threshold, L3i greedy6) are read from their result.json files, not rerun.
Writes result.json into --out; the runner redirects stdout to run.log.
"""
import argparse
import json
import os
import sys
import time
import warnings

sys.dont_write_bytecode = True
AVRL_DIR = "/homes/rdubash/avrl_live/CHEAP/avrl"
E03_RESULT = os.path.join(AVRL_DIR, "runs", "E03", "result.json")
L3I_RESULT = os.path.join(AVRL_DIR, "runs", "L3i", "result.json")

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from sklearn.exceptions import ConvergenceWarning  # noqa: E402
from sklearn.linear_model import LogisticRegression  # noqa: E402
from sklearn.metrics import roc_auc_score  # noqa: E402
from sklearn.model_selection import StratifiedKFold  # noqa: E402
from sklearn.pipeline import make_pipeline  # noqa: E402
from sklearn.preprocessing import StandardScaler  # noqa: E402

N_GROUPS = 3000
N_ERR = 436
K = 750
CHANCE = K * N_ERR / N_GROUPS
SIX_COLS = ["agreement", "n_distinct", "mean_tokens", "std_tokens", "any_truncated", "log_qlen"]
PC_COLS = [f"pc{i}" for i in range(16)]
FOLD_SEEDS = list(range(8))
# arm -> (feature set key, scaled)
MODEL_ARMS = {
    "six": ("six", False),
    "twenty_two": ("twenty_two", False),
    "pc_only": ("pc_only", False),
    "six_scaled": ("six", True),
    "twenty_two_scaled": ("twenty_two", True),
}
PRIMARY_ARMS = ["six", "twenty_two", "pc_only"]
SECONDARY_ARMS = ["six_scaled", "twenty_two_scaled"]


def load_pool(path):
    df = pd.read_parquet(path)
    assert len(df) == N_GROUPS, f"expected {N_GROUPS} groups, got {len(df)}"
    y = df["proxy_error"].to_numpy(dtype=np.int64)
    assert int(y.sum()) == N_ERR, f"expected {N_ERR} proxy errors, got {int(y.sum())}"
    six = df[SIX_COLS].to_numpy(dtype=np.float64)
    pcs = df[PC_COLS].to_numpy(dtype=np.float64)
    twenty_two = np.hstack([six, pcs])
    assert twenty_two.shape == (N_GROUPS, 22), twenty_two.shape
    assert np.isfinite(twenty_two).all(), "non-finite feature values"
    return {
        "y": y,
        "agreement": df["agreement"].to_numpy(dtype=np.float64),
        "n_distinct": df["n_distinct"].to_numpy(dtype=np.int64),
        "six": six,
        "twenty_two": twenty_two,
        "pc_only": pcs,
    }


def threshold_selection(pool):
    """ThresholdAllocator's key applied globally: agreement ascending, then n_distinct descending.
    np.lexsort is stable and uses its last key as the primary key."""
    sel = np.lexsort((-pool["n_distinct"], pool["agreement"]))[:K]
    assert len(np.unique(sel)) == K, "threshold selection must be K distinct groups"
    return sel


def top_k(scores):
    """Global top K by score, descending. Stable: equal scores keep group index order."""
    sel = np.argsort(-scores, kind="stable")[:K]
    assert len(np.unique(sel)) == K, "selection must be K distinct groups"
    return sel


def oof_probability(X, y, fold_seed, scaled):
    p = np.full(len(y), np.nan)
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=fold_seed)
    for tr, te in skf.split(X, y):
        model = LogisticRegression(max_iter=1000)
        if scaled:
            model = make_pipeline(StandardScaler(), model)
        model.fit(X[tr], y[tr])
        p[te] = model.predict_proba(X[te])[:, 1]
    assert not np.isnan(p).any(), "every group must get an out-of-fold probability"
    return p


def evaluate(pool, arm, fold_seed):
    key, scaled = MODEL_ARMS[arm]
    p = oof_probability(pool[key], pool["y"], fold_seed, scaled)
    sel = top_k(p)
    caught = int(pool["y"][sel].sum())
    return {"arm": arm, "fold_seed": fold_seed, "selected": int(len(sel)), "caught": caught,
            "admitted": (N_ERR - caught) / N_ERR, "auc": float(roc_auc_score(pool["y"], p))}


def summarize(rows):
    caught = [r["caught"] for r in rows]
    return {
        "caught_per_fold_seed": caught,
        "caught_seed0": caught[0],
        "caught_mean": float(np.mean(caught)),
        "caught_min": int(min(caught)),
        "caught_max": int(max(caught)),
        "admitted_seed0": rows[0]["admitted"],
        "admitted_mean": float(np.mean([r["admitted"] for r in rows])),
        "auc_seed0": rows[0]["auc"],
        "auc_mean": float(np.mean([r["auc"] for r in rows])),
        "auc_per_fold_seed": [round(r["auc"], 6) for r in rows],
    }


def count_conv(wrec):
    return sum(issubclass(w.category, ConvergenceWarning) for w in wrec)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pool", required=True)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)
    t0 = time.time()

    def log(msg):
        print(msg, flush=True)

    pool = load_pool(args.pool)
    n_err = int(pool["y"].sum())
    assert n_err == N_ERR and K == 750 and CHANCE == 109.0, (n_err, K, CHANCE)
    log(f"L3k pool {args.pool}: groups={N_GROUPS} proxy_errors={n_err} selection={K} chance={CHANCE:.1f}")
    log(f"six features {SIX_COLS}; pc features {PC_COLS[0]} to {PC_COLS[-1]}")
    log("model LogisticRegression(max_iter=1000), unscaled for primary arms; "
        "folds StratifiedKFold(5, shuffle=True, random_state=fold_seed) on proxy_error; fold seeds 0-7")

    thr_sel = threshold_selection(pool)
    thr_caught = int(pool["y"][thr_sel].sum())
    thr_admitted = (n_err - thr_caught) / n_err
    thr_auc_ref = float(roc_auc_score(pool["y"], -pool["agreement"]))
    log(f"threshold (global offline, no model): caught {thr_caught} admitted {thr_admitted:.4f} "
        f"AUC of -agreement (reference) {thr_auc_ref:.4f}")

    with open(E03_RESULT) as f:
        e03_thr_caught = json.load(f)["per_budget"]["0.25"]["arms"]["threshold"]["caught_per_seed"]
    with open(L3I_RESULT) as f:
        l3i_greedy6_mean = json.load(f)["caught_mean_seeds_0_to_7"]["greedy6"]
    e03_thr_mean = float(np.mean(e03_thr_caught))
    log(f"context: E03 per-step threshold caught mean seeds 0-7 {e03_thr_mean:.2f}; "
        f"L3i greedy6 caught mean seeds 0-7 {l3i_greedy6_mean:.2f}")

    rows = {}
    conv = {}
    with warnings.catch_warnings(record=True) as wrec:
        warnings.simplefilter("always")
        for arm in PRIMARY_ARMS + SECONDARY_ARMS:
            before = count_conv(wrec)
            rows[arm] = [evaluate(pool, arm, s) for s in FOLD_SEEDS]
            conv[arm] = count_conv(wrec) - before
            caught = [r["caught"] for r in rows[arm]]
            tag = "secondary" if arm in SECONDARY_ARMS else "primary"
            log(f"{arm} ({tag}): caught per fold seed 0-7 {caught} mean {np.mean(caught):.2f} "
                f"ConvergenceWarnings {conv[arm]}")
        conv_total = count_conv(wrec)

    summ = {arm: summarize(rows[arm]) for arm in rows}
    summ["threshold"] = {"caught": thr_caught, "admitted": thr_admitted, "auc_of_neg_agreement_reference": thr_auc_ref}

    six_mean = summ["six"]["caught_mean"]
    gap_mean = six_mean - thr_caught
    gap_seed0 = summ["six"]["caught_seed0"] - thr_caught
    pc_dev = summ["pc_only"]["caught_mean"] - CHANCE
    if abs(pc_dev) <= 10:
        pc_read = "read iii: pc0 to pc15 alone within 10 of chance, no proxy-error signal on their own on this pool"
    elif pc_dev > 10:
        pc_read = "pc0 to pc15 alone above chance by more than 10: some signal alone, claim 3 bar not met by this test"
    else:
        pc_read = "pc0 to pc15 alone below chance by more than 10: anti-signal (not pre-registered)"
    if gap_mean >= 0:
        verdict = "PASS (read ii)"
        verdict_text = ("six-feature full-label ranking reaches the threshold order (fold-seed mean); the greedy-loop "
                        "gap is attributed to the online loop, and L3l is the next test")
    else:
        verdict = "KILL (read i)"
        verdict_text = ("six-feature full-label ranking is below the threshold order (fold-seed mean); the regression "
                        "ranking is worse than agreement on this pool even with full labels, and L3j is not justified "
                        "for the regression family")
    log(f"six minus threshold, fold-seed mean: {gap_mean:+.2f} (seed 0: {gap_seed0:+d})")
    log(f"pc_only minus chance (109.0), fold-seed mean: {pc_dev:+.2f}; {pc_read}")
    log(f"overall L3k verdict: {verdict}; {verdict_text}")

    checks = {
        "groups_3000": len(pool["y"]) == N_GROUPS,
        "proxy_errors_436": n_err == N_ERR,
        "selection_size_750_all_arms": all(r["selected"] == K for arm in rows for r in rows[arm]) and len(thr_sel) == K,
        "chance_109": CHANCE == 109.0,
        "twenty_two_shape_3000x22": pool["twenty_two"].shape == (N_GROUPS, 22),
    }
    assert all(checks.values()), checks

    result = {
        "id": "L3k",
        "pool": args.pool,
        "n_groups": N_GROUPS,
        "n_proxy_errors": n_err,
        "selection_size_groups": K,
        "chance_caught": CHANCE,
        "six_cols": SIX_COLS,
        "pc_cols": PC_COLS,
        "folds": {"n_splits": 5, "shuffle": True, "stratified_on": "proxy_error", "fold_seeds": FOLD_SEEDS},
        "context": {
            "e03_per_step_threshold_caught_mean_seeds_0_to_7": e03_thr_mean,
            "l3i_greedy6_caught_mean_seeds_0_to_7": l3i_greedy6_mean,
            "note": "E03 threshold is per-step (chunks of 16); this script's threshold is a global offline selection. "
                    "Compare within this script only.",
        },
        "arms": summ,
        "metric": {
            "name": "proxy errors caught (of 436) in the global top-750 selection, unscaled logistic regression, "
                    "five-fold out-of-fold probabilities; fold seed 0 and fold-seed mean over seeds 0-7",
            "threshold_caught": thr_caught,
            "six_caught_seed0": summ["six"]["caught_seed0"],
            "six_caught_mean_seeds_0_to_7": six_mean,
            "twenty_two_caught_seed0": summ["twenty_two"]["caught_seed0"],
            "twenty_two_caught_mean_seeds_0_to_7": summ["twenty_two"]["caught_mean"],
            "pc_only_caught_seed0": summ["pc_only"]["caught_seed0"],
            "pc_only_caught_mean_seeds_0_to_7": summ["pc_only"]["caught_mean"],
            "chance_caught": CHANCE,
        },
        "reads": {
            "six_minus_threshold_fold_seed_mean": gap_mean,
            "six_minus_threshold_seed0": gap_seed0,
            "pc_only_minus_chance_fold_seed_mean": pc_dev,
            "pc_only_read_iii": pc_read,
        },
        "verdict": verdict,
        "verdict_text": verdict_text,
        "verdict_rule": "PASS (read ii) iff six-feature fold-seed mean caught >= threshold caught; "
                        "KILL (read i) otherwise. Read iii: pc_only fold-seed mean within 10 of 109.0 means no signal alone.",
        "convergence_warnings": {"per_arm": conv, "total": conv_total},
        "checks": checks,
        "assumptions": [
            "Spec says five feature sets but lists four pre-registered arms (threshold, six, 22 features, pc0 to pc15 alone). "
            "Run: those four, plus two secondary StandardScaler variants (six_scaled, twenty_two_scaled). Count mismatch noted.",
            "threshold is a global offline selection (lexsort on agreement ascending, n_distinct descending), not the per-step E03 selection (334.25 context).",
            "out-of-fold probabilities are unclipped (the online loop clips to [1e-3, 1 - 1e-3]); ties in probability resolved by group index via a stable sort.",
            "features are raw (unscaled) for primary arms; pc0 to pc15 are float32 in the parquet and cast to float64.",
            "gold (proxy_error) is used for out-of-fold training and scoring only. This is a full-label ceiling diagnostic, not a deployable allocator.",
            "AUC of the threshold arm is a reference computed on -agreement alone (no tie-break on n_distinct).",
        ],
        "wall_s": round(time.time() - t0, 1),
    }
    out_path = os.path.join(args.out, "result.json")
    with open(out_path, "w") as f:
        json.dump(result, f, indent=1)
    log(f"\nwrote {out_path} (wall {result['wall_s']} s)")


if __name__ == "__main__":
    main()
