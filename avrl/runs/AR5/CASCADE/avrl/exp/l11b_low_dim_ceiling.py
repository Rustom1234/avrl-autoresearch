"""L11b (copy of avrl/exp/l10a_oracle_ceiling.py): low-dimensional online learner and labeled static ranker, per-seed refit.

Arms, all compared with the label-free agreement threshold on the same pool, split, budgets and seeds as L10a:
- threshold: label-free, agreement ascending then n_distinct descending, seeded random tie-break (unchanged from L10a).
- uniform: random order per seed (context).
- greedy_online: top-scored unseen docs, classifier refit every R labels on the --features columns.
- static: labeled logistic ranker on the same columns. Refit per seed on a seeded 80% bucket subsample of the seed-0
  fit quarter (deviation 1 in the L11b pre-registration). Its per-seed scores are fixed inside the bootstrap (deviation 2).
- hand: post hoc rule, threshold order for the first (budget - 100) docs, then the next 100 e01_proxy_error == 1 docs
  in threshold order. Runs only with --e01_feats.

--features selects the column set for greedy_online and static: cheap (six rollout statistics), cheap+emb (cheap plus
16 MiniLM PCs), agree_nd (agreement and n_distinct only). --e01_feats appends e01_proxy_error (cross-policy oracle feature)
to that set, as in L10a.

Metric: admitted = 1 - caught / total_errors. D = admitted(threshold) - admitted(arm). Positive D means the arm admits
fewer corrupted rewards than the label-free threshold.

Bootstrap: resample sim docs with replacement (BOOT_SEED 0, sorted indices, pool order kept). Per resample every arm is
recomputed on that resample's own error total, except that static uses the per-seed scores fixed on the full sim pool.
"""
import os

os.environ.setdefault("OMP_NUM_THREADS", "1")
import json
import time
import argparse
from concurrent.futures import ProcessPoolExecutor

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline

MIN, K = 3, 1
LABEL = "proxy_error"
E01_COL = "e01_proxy_error"
CHEAP = ["agreement", "n_distinct", "mean_tokens", "std_tokens", "any_truncated", "log_qlen"]
EMB = [f"pc{i}" for i in range(16)]
FEATS = {"cheap": CHEAP, "cheap+emb": CHEAP + EMB, "agree_nd": ["agreement", "n_distinct"]}
BOOT_SEED = 0  # rng for bootstrap resample indices (separate from the allocator seeds 0..SEEDS-1)
STATIC_SEED_BASE = 100  # static seed s: numpy default_rng(STATIC_SEED_BASE + s) draws its 80% bucket subsample
STATIC_FRAC = 0.8
HAND_FLAGGED = 100

_BASE, _G, _COLS, _S, _ARMS = None, None, None, None, None  # set in each worker by _init_worker


def make_pool(d, G, cols):
    codes, uniq = pd.factorize(d[G].values)
    P = {
        "X": d[cols].fillna(0).values,
        "y": d[LABEL].values.astype(int),
        "codes": codes,
        "n_b": len(uniq),
        "members": [np.where(codes == b)[0] for b in range(len(uniq))],
        "ag": d["agreement"].values,
        "nd": d["n_distinct"].values,
    }
    if E01_COL in d.columns:
        P["e01"] = d[E01_COL].values.astype(int)
    return P


def threshold_order(ag, nd, rng=None):
    """ThresholdAllocator._order key: agreement ascending, then n_distinct descending.
    Exact ties keep input order (rng None) or get a seeded random order (rng given)."""
    tb = rng.random(len(ag)) if rng is not None else np.arange(len(ag))
    return np.lexsort((tb, -nd, ag))  # lexsort: last key is primary, sort is stable


def hand_selection(P, budget, rng):
    """Post hoc hand rule: threshold order for the first budget-100 docs, then the next 100 e01-flagged docs
    in threshold order. If fewer than 100 flagged docs remain, the rest is filled in threshold order."""
    order = threshold_order(P["ag"], P["nd"], rng)
    n_thr = budget - HAND_FLAGGED
    head, rest = order[:n_thr], order[n_thr:]
    flag = P["e01"][rest] == 1
    tail = np.concatenate([rest[flag][:HAND_FLAGGED], rest[~flag]])[:HAND_FLAGGED]
    return np.concatenate([head, tail])


def run_online(P, budget, seed, greedy=False, R=None):
    """Copied from avrl/exp/l10a_oracle_ceiling.py (and runs/p0/sim_alloc.py) run_online, pool passed in as P.
    greedy_online: top-scored unseen docs anywhere, no buckets."""
    X, y, codes, members, n_b = P["X"], P["y"], P["codes"], P["members"], P["n_b"]
    rng = np.random.default_rng(seed); R = R or max(budget // 8, 40)
    seen = np.zeros(len(y), bool); score = rng.random(len(y)) * 1e-3; bprior = None
    alpha = np.ones(n_b); beta = np.ones(n_b); remaining = np.array([len(m) for m in members]); hits = spent = 0; next_fit = R
    while spent < budget:
        take = min(K, budget - spent); avail = remaining > 0
        if greedy:
            mu = np.where(~seen)[0]; c = mu[np.argsort(-score[mu])][:take]; seen[c] = True; hits += int(y[c].sum()); spent += len(c)
        else:
            s = rng.beta(alpha + 10 * bprior, beta + 10 * (1 - bprior)) if bprior is not None else rng.beta(alpha, beta)
            s[~avail] = -1; b = int(np.argmax(s)); mu = members[b][~seen[members[b]]]; c = mu[np.argsort(-score[mu])][:take]
            seen[c] = True; remaining[b] -= len(c); h = int(y[c].sum()); alpha[b] += h; beta[b] += len(c) - h; hits += h; spent += len(c)
        if spent >= next_fit and 0 < y[seen].sum() < seen.sum():
            score = make_pipeline(StandardScaler(), LogisticRegression(max_iter=500, class_weight="balanced")).fit(X[seen], y[seen]).predict_proba(X)[:, 1]
            bp = pd.Series(score).groupby(codes).mean().values; bprior = bp / bp.max(); next_fit += R
    return hits


def arm_caught(P, arm, budget, seeds, S):
    """Caught-error counts per seed for one arm. S: (n_docs, n_seeds) per-seed static scores for this pool."""
    y = P["y"]
    if arm == "uniform":
        return np.array([y[np.random.default_rng(s).permutation(len(y))[:budget]].sum() for s in seeds], float)
    if arm == "threshold":  # label-free, random tie-break per seed
        return np.array([y[threshold_order(P["ag"], P["nd"], np.random.default_rng(s))[:budget]].sum() for s in seeds], float)
    if arm == "greedy_online":
        return np.array([run_online(P, budget, s, greedy=True) for s in seeds], float)
    if arm == "static":  # labeled; seed i ranks with its own refit scores
        return np.array([y[np.argsort(-S[:, i], kind="stable")[:budget]].sum() for i in range(len(seeds))], float)
    if arm == "hand":  # post hoc; same seeded tie-break as threshold
        return np.array([y[hand_selection(P, budget, np.random.default_rng(s))].sum() for s in seeds], float)
    raise ValueError(arm)


def _init_worker(base, G, cols, S, arms):
    global _BASE, _G, _COLS, _S, _ARMS
    _BASE, _G, _COLS, _S, _ARMS = base, G, cols, S, arms


def _one_resample(task):
    idx, budget, seeds = task
    P = make_pool(_BASE.iloc[idx], _G, _COLS)
    tot = int(P["y"].sum())
    S_r = _S[idx]
    h = {arm: float(arm_caught(P, arm, budget, seeds, S_r).mean()) for arm in _ARMS}
    return tot, h


def pct_ci(x):
    x = np.asarray(x, float)
    return [float(np.percentile(x, 2.5)), float(np.percentile(x, 97.5))]


def status(d_pt, lo, hi):
    if d_pt > 0 and lo > 0:
        return "PASS"
    if d_pt <= 0:
        return "KILL"
    return "INCONCLUSIVE"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--feats", required=True)
    ap.add_argument("--cell", default="cell100")
    ap.add_argument("--features", default="cheap+emb", choices=list(FEATS))
    ap.add_argument("--budgets", default="0.10,0.25,0.50")
    ap.add_argument("--seeds", type=int, default=8)
    ap.add_argument("--boot", type=int, default=200)
    ap.add_argument("--boot_budget", type=float, default=0.25)
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--e01_feats", default=None, help="p0 feats.parquet: adds proxy_error as e01_proxy_error to the greedy and static columns, enables the hand rule")
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    G, cols = args.cell, list(FEATS[args.features])
    budgets = [float(b) for b in args.budgets.split(",")]
    seeds = list(range(args.seeds))
    os.makedirs(args.out, exist_ok=True)
    t0 = time.time()

    # Same split as runs/p0/sim_alloc.py and L10a: keep buckets with >= MIN docs, seed-0 shuffle of bucket values, first quarter = fit.
    df = pd.read_parquet(args.feats)
    if args.e01_feats:  # cross-policy feature, left join keeps pool row order
        e01 = pd.read_parquet(args.e01_feats, columns=["pid", "proxy_error"]).rename(columns={"proxy_error": E01_COL})
        df = df.merge(e01, on="pid", how="left", validate="one_to_one")
        n_match = int(df[E01_COL].notna().sum())
        assert len(df) == 3000 and n_match == 3000, f"E01 join: {n_match} matched of {len(df)} pids"
        assert not df[E01_COL].isna().any()
        cols = cols + [E01_COL]
        print(f"E01 join: {n_match} pids matched, {E01_COL} mean {df[E01_COL].mean():.4f}", flush=True)
    arms = ["uniform", "threshold", "greedy_online", "static"] + (["hand"] if args.e01_feats else [])
    cnt = df[G].value_counts(); df = df[df[G].isin(cnt[cnt >= MIN].index)].reset_index(drop=True)
    rng0 = np.random.default_rng(0); groups = df[G].unique(); rng0.shuffle(groups)
    n_fit = len(groups) // 4; fit_groups = groups[:n_fit]
    fit = df[df[G].isin(fit_groups)]; sim = df[~df[G].isin(fit_groups)].reset_index(drop=True)

    # Static arm, refit per seed (deviation 1): seed s fits on a seeded 80% subsample of the fit-quarter buckets.
    S = np.zeros((len(sim), len(seeds)))
    fit_info = []
    for s in seeds:
        rs = np.random.default_rng(STATIC_SEED_BASE + s)
        k = int(round(STATIC_FRAC * n_fit))
        sub = rs.choice(fit_groups, size=k, replace=False)
        f = fit[fit[G].isin(sub)]
        assert 0 < f[LABEL].sum() < len(f), f"seed {s}: fit subsample has one class"
        clf = make_pipeline(StandardScaler(), LogisticRegression(max_iter=1000, class_weight="balanced")).fit(f[cols].fillna(0).values, f[LABEL].values)
        S[:, s] = clf.predict_proba(sim[cols].fillna(0).values)[:, 1]
        fit_info.append({"seed": s, "fit_buckets": k, "fit_docs": int(len(f)), "fit_errors": int(f[LABEL].sum())})

    P = make_pool(sim, G, cols)
    n_sim = len(sim); total = int(P["y"].sum())
    print(f"feats {os.path.basename(args.feats)} cell {G} features {args.features} ({len(cols)} cols) e01 {bool(args.e01_feats)} | "
          f"sim docs {n_sim} buckets {P['n_b']} base {P['y'].mean():.4f} total_errors {total} "
          f"median bucket {int(np.median([len(m) for m in P['members']]))} | fit quarter {n_fit} buckets", flush=True)

    point = {}
    for bf in budgets:
        budget = int(bf * n_sim)
        point[str(bf)] = {}
        for arm in arms:
            h = arm_caught(P, arm, budget, seeds, S)
            point[str(bf)][arm] = {"budget": budget, "caught_mean": float(h.mean()), "caught_sd": float(h.std()),
                                   "caught_per_seed": h.tolist(), "admitted": 1 - float(h.mean()) / total, "n_runs": len(h)}
    D_point = {str(bf): {arm: point[str(bf)]["threshold"]["admitted"] - point[str(bf)][arm]["admitted"] for arm in arms} for bf in budgets}

    print("\n| budget | budget_n | " + " | ".join(arms) + " |\n|---|---|" + "---|" * len(arms), flush=True)
    for bf in budgets:
        r = point[str(bf)]
        print(f"| {bf:.0%} | {r['uniform']['budget']} | " + " | ".join(f"{r[a]['caught_mean']:.2f}+-{r[a]['caught_sd']:.2f} (adm {r[a]['admitted']:.3f})" for a in arms) + " |", flush=True)
    print("\nD = admitted(threshold) - admitted(arm), point estimates:", flush=True)
    for bf in budgets:
        print(f"  {bf:.0%}: " + "  ".join(f"D({a}) {D_point[str(bf)][a]:+.4f}" for a in arms if a != "threshold"), flush=True)

    # Bootstrap at boot_budget: resample sim docs with replacement (sorted, pool order), recompute every arm per resample.
    bkey = str(args.boot_budget)
    assert bkey in point, "boot_budget must be one of --budgets"
    boot_budget = int(args.boot_budget * n_sim)
    rng_b = np.random.default_rng(BOOT_SEED)
    tasks = [(np.sort(rng_b.integers(0, n_sim, size=n_sim)), boot_budget, seeds) for _ in range(args.boot)]
    base = sim[list(dict.fromkeys([G, LABEL, "agreement", "n_distinct"] + cols))].reset_index(drop=True)
    t1 = time.time()
    with ProcessPoolExecutor(max_workers=args.workers, initializer=_init_worker, initargs=(base, G, cols, S, arms)) as ex:
        res = list(ex.map(_one_resample, tasks, chunksize=4))
    boot_secs = time.time() - t1
    adm = {arm: np.array([1 - h[arm] / tot for tot, h in res]) for arm in arms}
    D_boot = {arm: adm["threshold"] - adm[arm] for arm in arms if arm != "threshold"}
    D_boot_ci = {arm: {"ci95": pct_ci(D_boot[arm]), "boot_mean": float(D_boot[arm].mean()),
                       "frac_resamples_D_gt_0": float((D_boot[arm] > 0).mean()), "n_resamples": len(D_boot[arm])} for arm in D_boot}
    print(f"\nbootstrap {len(res)} resamples at {args.boot_budget:.0%} ({boot_budget} docs), {args.workers} workers, {boot_secs:.1f}s", flush=True)
    status_row = {}
    for arm in D_boot:
        c = D_boot_ci[arm]["ci95"]; d_pt = D_point[bkey][arm]
        status_row[arm] = status(d_pt, c[0], c[1])
        print(f"  D({arm}) point {d_pt:+.4f}  boot mean {D_boot[arm].mean():+.4f}  95% CI [{c[0]:+.4f}, {c[1]:+.4f}]  "
              f"frac>0 {D_boot_ci[arm]['frac_resamples_D_gt_0']:.3f}  status {status_row[arm]}", flush=True)

    summary = (f"L11b {G} features={args.features} e01={bool(args.e01_feats)} {args.boot_budget:.0%}: " +
               "; ".join(f"D({a}) {D_point[bkey][a]:+.4f} CI [{D_boot_ci[a]['ci95'][0]:+.4f}, {D_boot_ci[a]['ci95'][1]:+.4f}]"
                         for a in D_boot))
    print("\n" + summary, flush=True)

    out = {
        "exp": "L11b",
        "summary": summary,
        "config": {"feats": args.feats, "cell": G, "features": args.features, "e01_feats": args.e01_feats,
                   "feature_list": cols, "n_features": len(cols), "arms": arms, "min_bucket_size": MIN, "K": K,
                   "seeds": args.seeds, "fit_split": "seed-0 shuffle of bucket values, first quarter = fit",
                   "static_refit": {"seed_rng": f"default_rng({STATIC_SEED_BASE} + seed)", "bucket_fraction": STATIC_FRAC},
                   "threshold_tiebreak": "seeded random among exact (agreement, n_distinct) ties",
                   "hand_rule": {"flagged": HAND_FLAGGED, "rule": "threshold order for budget - 100 docs, then next 100 e01_proxy_error == 1 docs in threshold order"},
                   "boot_resamples": args.boot, "boot_budget_frac": args.boot_budget, "boot_budget_docs": boot_budget, "boot_seed": BOOT_SEED,
                   "boot_unit": "sim docs with replacement; resample keeps pool order; each arm recomputed on the resample's own error total; static scores fixed per seed (no refit)"},
        "pool": {"n_sim": n_sim, "n_buckets": P["n_b"], "total_errors": total, "budget_docs": {str(bf): int(bf * n_sim) for bf in budgets},
                 "e01_flagged_in_sim": int(P["e01"].sum()) if "e01" in P else None},
        "static_fit_info": fit_info,
        "point": point,
        "D_point": D_point,
        "D_bootstrap": D_boot_ci,
        "D_bootstrap_draws": {arm: D_boot[arm].tolist() for arm in D_boot},
        "status_at_boot_budget": status_row,
        "wall_seconds": round(time.time() - t0, 1),
        "bootstrap_wall_seconds": round(boot_secs, 1),
    }
    with open(os.path.join(args.out, "result.json"), "w") as f:
        json.dump(out, f, indent=1)
    print(f"wrote {os.path.join(args.out, 'result.json')}", flush=True)


if __name__ == "__main__":
    main()
