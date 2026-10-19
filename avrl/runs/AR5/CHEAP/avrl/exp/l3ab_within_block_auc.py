#!/usr/bin/env python3
"""L3ab: does the six-feature regression rank corrupted groups within (agreement, n_distinct) blocks at all?

Diagnostic only (no claim credit). CPU only, no GPU, no pip, no ssh. Reads runs/p0/feats.parquet read-only and writes
result.json into --out; the runner (avrl/exp/L3ab.sh) redirects stdout to run.log.

Score (ranker ceiling with full labels, not an online arm): out-of-fold logistic regression on the six l3i FEAT_COLS (no pc
columns), KFold(5, shuffle=True, random_state=0) over all 3000 groups, Pipeline(StandardScaler, LogisticRegression(max_iter=1000)),
target proxy_error. Each group is scored by the fold model that did not see it.

Full-pool read: blocks are exact (agreement, n_distinct) values. Kept blocks contain both labels. The within-block AUC is the
block-size-weighted mean of per-block ROC AUCs against proxy_error (weight = groups in the block). The 95% CI is a block
bootstrap: kept blocks resampled with replacement, B=2000, rng default_rng(20000), percentile interval.

Cut read (same AUC restricted to the cut): positions 4 and 5 (counted from 1) of each 16-group step under the threshold order
(allocators.ThresholdAllocator._order, unchanged), steps taken from the seed-0 permutation default_rng(1000).permutation(3000).
Blocks are rebuilt among the cut groups only. Same B and seed as the full-pool read. Reported, does not set the reading.
"""
import argparse
import json
import math
import os
import sys
import time

sys.dont_write_bytecode = True
AVRL_DIR = "/homes/rdubash/avrl_live/CHEAP/avrl"
sys.path.insert(0, AVRL_DIR)

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from sklearn.linear_model import LogisticRegression  # noqa: E402
from sklearn.metrics import roc_auc_score  # noqa: E402
from sklearn.model_selection import KFold  # noqa: E402
from sklearn.pipeline import make_pipeline  # noqa: E402
from sklearn.preprocessing import StandardScaler  # noqa: E402

import allocators as A  # noqa: E402

STEP = 16
N_EXPECT = 3000
N_ERR_EXPECT = 436
FEAT_COLS = ["agreement", "n_distinct", "mean_tokens", "std_tokens", "any_truncated", "log_qlen"]
CV_SPLITS = 5
CV_SEED = 0
PERM_SEED = 1000 + 0          # seed-0 step permutation, default_rng(1000 + seed) as in L3aa and L3h
CUT_POSITIONS = (3, 4)        # 0-based indices, i.e. positions 4 and 5 counted from 1
BOOT_SEED = 20000
READING_LO = 0.5
READING_RULE = (
    "Reading, fixed before the run (diagnostic, no claim credit). A = block-size-weighted within-block AUC of the out-of-fold "
    "score against proxy_error over all kept (mixed) blocks on the full pool, with its 95% block-bootstrap percentile CI "
    "(B=2000, rng default_rng(20000)). If the CI includes 0.5 (lower <= 0.5 <= upper), read KILL: the within-block ranker is not "
    "the lever on this pool, no within-block re-specification is built, and the next test is L3ae. If the CI lower bound is above "
    "0.5, read SURVIVE: the regression ranks within blocks even with full labels, so the L3h null comes from the online loop, and "
    "the next test is L3n. If the CI lies entirely below 0.5, read ANTI_SIGNAL: report it as anti-signal and build nothing on "
    "that score. The cut read (positions 4 and 5) is reported with its own CI and does not set the reading. This entry cannot "
    "change claim 2 or claim 3."
)
ASSUMPTIONS = [
    "score: KFold(5, shuffle=True, random_state=0) over all 3000 groups; Pipeline(StandardScaler, LogisticRegression(max_iter=1000)) "
    "fitted on the four training folds, predict_proba[:, 1] on the held-out fold; target proxy_error; six FEAT_COLS, no pc columns",
    "blocks are exact (agreement, n_distinct) values; a kept block contains both labels (at least one error and one non-error)",
    "within-block AUC is sklearn roc_auc_score per kept block; ties in the score get average rank",
    "weighted mean weights = number of groups in the block (kept blocks only), so the full-pool value is sum(n_b * AUC_b) / sum(n_b)",
    "block bootstrap: K kept blocks resampled with replacement per draw (rng default_rng(20000), B=2000), same weighted mean per draw; "
    "95% CI is the 2.5th and 97.5th percentile of the draws",
    "cut read: positions 4 and 5 counted from 1 are 0-based indices 3 and 4 of each step ordered by ThresholdAllocator._order "
    "(agreement ascending, n_distinct descending, stable on chunk position); all steps are included, including the last 8-group step",
    "cut read blocks are rebuilt among the cut groups only; its bootstrap uses the same B and seed (default_rng(20000))",
    "label set is {0, 1} and proxy_error is used only as the target and the AUC label, never as a feature",
    "reference values: all-group AUC of the out-of-fold score, and the share of blocks (and of groups) that contain both labels",
]


def load_pool(path):
    assert len(FEAT_COLS) == 6 and not any(c.startswith("pc") for c in FEAT_COLS), FEAT_COLS
    df = pd.read_parquet(path)
    assert len(df) == N_EXPECT, f"expected {N_EXPECT} groups, got {len(df)}"
    y = df["proxy_error"].to_numpy(dtype=np.int64)
    assert set(np.unique(y).tolist()) <= {0, 1}, "proxy_error must be binary"
    assert int(y.sum()) == N_ERR_EXPECT, f"expected {N_ERR_EXPECT} proxy errors, got {int(y.sum())}"
    return {
        "X": df[FEAT_COLS].to_numpy(dtype=np.float64),
        "agreement": df["agreement"].to_numpy(dtype=np.float64),
        "n_distinct": df["n_distinct"].to_numpy(dtype=np.int64),
        "y": y,
    }


def oof_score(X, y):
    """Out-of-fold logistic score. Every group is scored by the fold model that did not train on it."""
    oof = np.full(len(y), np.nan)
    kf = KFold(n_splits=CV_SPLITS, shuffle=True, random_state=CV_SEED)
    for tr, te in kf.split(X):
        model = make_pipeline(StandardScaler(), LogisticRegression(max_iter=1000))
        model.fit(X[tr], y[tr])
        oof[te] = model.predict_proba(X[te])[:, 1]
    assert not np.isnan(oof).any(), "out-of-fold score has NaN entries"
    return oof


def cut_groups(n, agree, ndist):
    """Pool indices at positions 4 and 5 (from 1) of each 16-group step under the threshold order, steps from the seed-0
    permutation. Each step contributes two groups; the last step has 8 groups and still contributes two."""
    perm = np.random.default_rng(PERM_SEED).permutation(n)
    picked = []
    for t in range(math.ceil(n / STEP)):
        chunk = perm[t * STEP:(t + 1) * STEP]
        records = [{"idx": j, "agreement": float(agree[g]), "n_distinct": int(ndist[g])} for j, g in enumerate(chunk)]
        order = A.ThresholdAllocator._order(records)
        assert len(order) > max(CUT_POSITIONS), (t, len(order))
        for pos in CUT_POSITIONS:
            picked.append(int(chunk[order[pos]["idx"]]))
    assert len(set(picked)) == len(picked), "cut groups must be distinct"
    return np.asarray(sorted(picked), dtype=np.int64)


def kept_blocks(agree, ndist, y, score):
    """Exact (agreement, n_distinct) blocks. Returns arrays for the kept (mixed) blocks and the count of all blocks."""
    by = {}
    for i, key in enumerate(zip(agree.tolist(), ndist.tolist())):
        by.setdefault(key, []).append(i)
    sizes, aucs, errs = [], [], []
    for key in sorted(by):
        ids = np.asarray(by[key], dtype=np.int64)
        yy = y[ids]
        if yy.min() == yy.max():
            continue
        sizes.append(len(ids))
        aucs.append(float(roc_auc_score(yy, score[ids])))
        errs.append(int(yy.sum()))
    return (np.asarray(sizes, dtype=np.float64), np.asarray(aucs, dtype=np.float64),
            np.asarray(errs, dtype=np.int64), len(by))


def weighted_mean(sizes, aucs):
    return float((sizes * aucs).sum() / sizes.sum())


def block_bootstrap(sizes, aucs, n_boot, seed):
    """Resample the kept blocks with replacement; each draw is the size-weighted mean over the drawn blocks."""
    rng = np.random.default_rng(seed)
    K = len(sizes)
    draws = np.empty(n_boot, dtype=np.float64)
    for b in range(n_boot):
        idx = rng.choice(K, size=K, replace=True)
        draws[b] = (sizes[idx] * aucs[idx]).sum() / sizes[idx].sum()
    return draws


def read_block(name, agree, ndist, y, score, n_boot, log):
    sizes, aucs, errs, n_blocks = kept_blocks(agree, ndist, y, score)
    K = len(sizes)
    assert K > 0, f"{name}: no mixed blocks"
    point = weighted_mean(sizes, aucs)
    draws = block_bootstrap(sizes, aucs, n_boot, BOOT_SEED)
    lo, hi = np.percentile(draws, [2.5, 97.5])
    res = {
        "name": name,
        "groups_total": int(len(y)),
        "errors_total": int(y.sum()),
        "blocks_total": int(n_blocks),
        "mixed_blocks": int(K),
        "share_blocks_mixed": float(K / n_blocks),
        "groups_in_mixed_blocks": int(sizes.sum()),
        "errors_in_mixed_blocks": int(errs.sum()),
        "share_groups_in_mixed_blocks": float(sizes.sum() / len(y)),
        "within_block_auc_weighted": point,
        "boot_mean": float(draws.mean()),
        "boot_sd": float(draws.std(ddof=1)),
        "ci95_percentile": [float(lo), float(hi)],
        "B": int(n_boot),
        "boot_seed": BOOT_SEED,
        "includes_0_5": bool(lo <= 0.5 <= hi),
        "per_block_auc_min": float(aucs.min()),
        "per_block_auc_max": float(aucs.max()),
    }
    log(f"{name}: groups total {res['groups_total']} errors {res['errors_total']}; blocks total {n_blocks}, mixed {K} "
        f"({res['share_blocks_mixed']:.3f} of blocks); groups in mixed blocks {res['groups_in_mixed_blocks']}, "
        f"errors in mixed blocks {res['errors_in_mixed_blocks']}")
    log(f"{name}: within-block AUC (size-weighted) {point:.4f}; block bootstrap B={n_boot} mean {res['boot_mean']:.4f} "
        f"sd {res['boot_sd']:.4f}; 95% CI [{lo:.4f}, {hi:.4f}]; includes 0.5: {res['includes_0_5']}")
    return res


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pool", required=True)
    ap.add_argument("--boot", type=int, default=2000)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    assert args.boot == 2000, "spec: block bootstrap B=2000"
    os.makedirs(args.out, exist_ok=True)

    def log(msg):
        print(msg, flush=True)

    t0 = time.time()
    pool = load_pool(args.pool)
    X, agree, ndist, y = pool["X"], pool["agreement"], pool["n_distinct"], pool["y"]
    n = len(y)
    log(f"L3ab pool {args.pool}: groups={n} proxy_errors={int(y.sum())} features={FEAT_COLS}")
    log("PRE-REGISTERED READING (written into the log before the run):")
    log(f"  {READING_RULE}")

    score = oof_score(X, y)
    auc_all = float(roc_auc_score(y, score))
    log(f"out-of-fold score (KFold 5, shuffle, random_state=0): all-group AUC {auc_all:.4f}")

    full = read_block("full_pool", agree, ndist, y, score, args.boot, log)

    cut_ids = cut_groups(n, agree, ndist)
    cut = read_block("cut_positions_4_5", agree[cut_ids], ndist[cut_ids], y[cut_ids], score[cut_ids], args.boot, log)
    log(f"cut read: {len(cut_ids)} groups (positions 4 and 5 over {math.ceil(n / STEP)} steps), "
        f"{int(y[cut_ids].sum())} errors in the cut")

    lo, hi = full["ci95_percentile"]
    if lo > READING_LO:
        reading, next_test = "SURVIVE", "L3n"
    elif hi < READING_LO:
        reading, next_test = "ANTI_SIGNAL", None
    else:
        reading, next_test = "KILL", "L3ae"
    log(f"reading (fixed rule): {reading}; full-pool 95% CI [{lo:.4f}, {hi:.4f}] against 0.5; next test: "
        f"{next_test if next_test else 'none, build nothing on this score'}")

    result = {
        "id": "L3ab",
        "pool": args.pool,
        "n_groups": n,
        "n_proxy_errors": int(y.sum()),
        "features": FEAT_COLS,
        "score": {"model": "Pipeline(StandardScaler, LogisticRegression(max_iter=1000))",
                  "cv": f"KFold(n_splits={CV_SPLITS}, shuffle=True, random_state={CV_SEED}), out-of-fold",
                  "target": "proxy_error"},
        "assumptions": ASSUMPTIONS,
        "reference": {"all_group_auc": auc_all,
                      "share_blocks_mixed_full_pool": full["share_blocks_mixed"],
                      "share_groups_in_mixed_blocks_full_pool": full["share_groups_in_mixed_blocks"]},
        "metric": {
            "name": "block-size-weighted within-block AUC of the out-of-fold score against proxy_error, full pool, "
                    "block bootstrap 95% percentile CI (B=2000, default_rng(20000))",
            "within_block_auc_weighted": full["within_block_auc_weighted"],
            "ci95_percentile": full["ci95_percentile"],
            "groups_used": full["groups_in_mixed_blocks"],
            "errors_used": full["errors_in_mixed_blocks"],
            "mixed_blocks": full["mixed_blocks"],
            "cut_within_block_auc_weighted": cut["within_block_auc_weighted"],
            "cut_ci95_percentile": cut["ci95_percentile"],
            "cut_groups_used": cut["groups_in_mixed_blocks"],
            "cut_errors_used": cut["errors_in_mixed_blocks"],
            "cut_mixed_blocks": cut["mixed_blocks"],
            "cut_groups_total": cut["groups_total"],
            "cut_errors_total": cut["errors_total"],
        },
        "full_pool": full,
        "cut_read": cut,
        "reading_threshold": READING_LO,
        "reading": reading,
        "next_test": next_test,
        "reading_rule": READING_RULE,
        "checks": {"proxy_error_binary": True, "n_groups_is_3000": n == N_EXPECT,
                   "n_errors_is_436": int(y.sum()) == N_ERR_EXPECT, "oof_no_nan": True,
                   "cut_groups_are_2_per_step": len(cut_ids) == 2 * math.ceil(n / STEP)},
        "wall_s": round(time.time() - t0, 1),
    }
    path = os.path.join(args.out, "result.json")
    with open(path, "w") as f:
        json.dump(result, f, indent=1)
    log(f"\nwrote {path} (wall {result['wall_s']} s)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
