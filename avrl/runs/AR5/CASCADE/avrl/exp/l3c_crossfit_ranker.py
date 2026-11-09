"""L3c: 5-fold cross-fit labeled logistic ranker on the E03 sim pool (CPU only).

Derived from avrl/exp/e03_threshold_bootstrap.py. Only the pool build is kept: cell100 buckets with at least MIN groups,
seed-0 shuffle of bucket values, first quarter of buckets dropped as the fit split, leaving the 2,180-group sim pool.
The 75 sim buckets go into 5 folds by a seeded shuffle, so no bucket straddles folds. Each fold is scored by a
StandardScaler + LogisticRegression(class_weight="balanced") fit on the other four folds (label proxy_error), so every
sim group gets one out-of-fold score. All groups are ranked by that pooled score and the top budgets are admitted.

The E03 references are recomputed in the same run (label-free threshold: 8 seeds, random tie-break, same code path as
E03; labeled static reference: fit on the fit split, same as E03), so the verdict is checked against numbers this run
reproduces, not only against constants copied from the E03 table.

Metric: caught proxy errors at the 25% budget (545 of 2,180 sim groups). admitted = 1 - caught / total_errors (309).
"""
import os

os.environ.setdefault("OMP_NUM_THREADS", "1")
import json
import time
import argparse

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline

G, MIN, LABEL = "cell100", 3, "proxy_error"
CHEAP = ["agreement", "n_distinct", "mean_tokens", "std_tokens", "any_truncated", "log_qlen"]
COLS = CHEAP + [f"pc{i}" for i in range(16)]  # the 22 cheap+emb columns used by E03
N_FOLDS, FOLD_SEED = 5, 0
BUDGETS = [0.10, 0.25, 0.50]
MAIN = "0.25"
N_THRESH_SEEDS = 8  # same seeds as the E03 threshold arm
E03_THRESH_CAUGHT, E03_STATIC_CAUGHT = 252.5, 263.0  # E03 table, 25% budget, same sim pool


def make_clf():
    return make_pipeline(StandardScaler(), LogisticRegression(max_iter=1000, class_weight="balanced"))


def threshold_order(ag, nd, rng):
    """Copied from E03 (ThresholdAllocator._order): agreement ascending, then n_distinct descending, exact ties in seeded random order."""
    tb = rng.random(len(ag))
    return np.lexsort((tb, -nd, ag))  # lexsort: last key is primary, sort is stable


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--feats", required=True)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)
    t0 = time.time()

    # Pool build: same as runs/p0/sim_alloc.py and E03.
    df = pd.read_parquet(args.feats)
    cnt = df[G].value_counts(); df = df[df[G].isin(cnt[cnt >= MIN].index)].reset_index(drop=True)
    rng0 = np.random.default_rng(0); groups = df[G].unique(); rng0.shuffle(groups)
    fit_g = set(groups[: len(groups) // 4])
    fit = df[df[G].isin(fit_g)]
    sim = df[~df[G].isin(fit_g)].reset_index(drop=True)
    X, y = sim[COLS].fillna(0).values, sim[LABEL].values.astype(int)
    n_sim, total, n_b = len(sim), int(y.sum()), int(sim[G].nunique())

    # 5 folds over sim buckets: seeded shuffle, then round-robin (15 buckets per fold, no bucket straddles folds).
    buckets = np.sort(sim[G].unique())
    perm = np.random.default_rng(FOLD_SEED).permutation(len(buckets))
    fold_of = {buckets[j]: i % N_FOLDS for i, j in enumerate(perm)}
    fold = sim[G].map(fold_of).values

    # Cross-fit: each fold is scored by a classifier trained on the other four folds.
    oof = np.full(n_sim, np.nan)
    train_n, test_n, fold_b = [], [], []
    for k in range(N_FOLDS):
        te = fold == k; tr = ~te
        oof[te] = make_clf().fit(X[tr], y[tr]).predict_proba(X[te])[:, 1]
        train_n.append(int(tr.sum())); test_n.append(int(te.sum())); fold_b.append(int(len(np.unique(sim[G].values[te]))))
    n_nan = int(np.isnan(oof).sum())
    if n_nan:
        raise ValueError(f"{n_nan} NaN out-of-fold scores")

    # Rank all sim groups by the pooled out-of-fold score and admit the top budgets.
    order = np.argsort(-oof, kind="stable")
    cross = {}
    for bf in BUDGETS:
        b = int(bf * n_sim); c = int(y[order[:b]].sum())
        cross[str(bf)] = {"budget": b, "caught": c, "admitted": 1 - c / total}

    # E03 references, recomputed on the same pool.
    def adm(c):
        return 1 - c / total

    b25 = int(0.25 * n_sim)
    ag, nd = sim["agreement"].values, sim["n_distinct"].values
    thr_per_seed = [int(y[threshold_order(ag, nd, np.random.default_rng(s))[:b25]].sum()) for s in range(N_THRESH_SEEDS)]
    thr_mean = float(np.mean(thr_per_seed))
    static = make_clf().fit(fit[COLS].fillna(0).values, fit[LABEL]).predict_proba(X)[:, 1]  # labeled reference only, as in E03
    static_caught = int(y[np.argsort(-static)[:b25]].sum())
    repro = {"threshold_recomputed_mean": thr_mean, "threshold_e03": E03_THRESH_CAUGHT, "threshold_match": thr_mean == E03_THRESH_CAUGHT,
             "static_recomputed": static_caught, "static_e03": E03_STATIC_CAUGHT, "static_match": float(static_caught) == E03_STATIC_CAUGHT}

    c25 = cross[MAIN]["caught"]
    if c25 >= E03_STATIC_CAUGHT:
        verdict = "PASS"
        reason = f"caught {c25} >= 263.0 at 25%: labeled cross-fit ranker matches the labeled static reference, so the labeled ceiling has room above the label-free threshold; L3b worth running"
    elif c25 <= E03_THRESH_CAUGHT:
        verdict = "KILL"
        reason = f"caught {c25} <= 252.5 at 25%: no labeled fixed ranker on these 22 features beats the label-free threshold out of sample on this pair; phase 1 not funded on this pair"
    else:
        verdict = "INCONCLUSIVE"
        reason = f"caught {c25} strictly between 252.5 and 263.0 at 25%: reported as inconclusive, not a pass or a kill"

    print(f"feats {os.path.basename(args.feats)} cell {G} | sim groups {n_sim} buckets {n_b} total_errors {total} | "
          f"{N_FOLDS} folds fold_seed {FOLD_SEED}, buckets per fold {fold_b}, train sizes {train_n}, test sizes {test_n}", flush=True)
    print("\n| budget | budget_n | caught | admitted |\n|---|---|---|---|", flush=True)
    for bf in BUDGETS:
        r = cross[str(bf)]
        print(f"| {bf:.0%} | {r['budget']} | {r['caught']} | {r['admitted']:.4f} |", flush=True)
    print(f"\nE03 references at 25% (same pool): threshold label-free 252.5 caught (8-seed mean; recomputed {thr_mean}, admitted {adm(thr_mean):.4f}); "
          f"static labeled 263.0 caught (recomputed {static_caught}, admitted {adm(static_caught):.4f})", flush=True)
    print(f"reproduction: threshold match {repro['threshold_match']}, static match {repro['static_match']}", flush=True)
    print(f"\nL3c 25%: cross-fit caught {c25}, admitted {cross[MAIN]['admitted']:.4f} | verdict {verdict}: {reason}", flush=True)

    out = {
        "exp": "L3c", "verdict": verdict, "verdict_reason": reason,
        "metric": {"name": "caught proxy errors at 25% budget, 5-fold cross-fit labeled logistic ranker", "budget": MAIN, "caught": c25,
                   "admitted": cross[MAIN]["admitted"], "total_errors": total},
        "crossfit": cross,
        "e03_references_25": {
            "threshold_label_free": {"e03_caught_mean": E03_THRESH_CAUGHT, "recomputed_caught_mean": thr_mean,
                                     "recomputed_per_seed": thr_per_seed, "recomputed_admitted": adm(thr_mean)},
            "static_labeled": {"e03_caught": E03_STATIC_CAUGHT, "recomputed_caught": static_caught, "recomputed_admitted": adm(static_caught)},
        },
        "reproduction": repro,
        "config": {"feats": args.feats, "cell": G, "min_bucket_size": MIN, "feature_list": COLS, "n_features": len(COLS), "label": LABEL,
                   "model": "StandardScaler + LogisticRegression(max_iter=1000, class_weight='balanced')",
                   "n_folds": N_FOLDS, "fold_seed": FOLD_SEED,
                   "fold_rule": "sorted sim bucket values, rng(fold_seed).permutation, round-robin into folds, no bucket straddles folds",
                   "fit_split": "seed-0 shuffle of bucket values, first quarter = fit split (used only by the static reference)",
                   "rank_rule": "pooled out-of-fold score, descending, stable sort",
                   "per_fold_train_size": train_n, "per_fold_test_size": test_n, "per_fold_buckets": fold_b},
        "pool": {"n_sim": n_sim, "n_buckets": n_b, "total_errors": total, "fit_groups": int(len(fit))},
        "wall_seconds": round(time.time() - t0, 2),
    }
    with open(os.path.join(args.out, "result.json"), "w") as f:
        json.dump(out, f, indent=1)
    print(f"wrote {os.path.join(args.out, 'result.json')}", flush=True)


if __name__ == "__main__":
    main()
