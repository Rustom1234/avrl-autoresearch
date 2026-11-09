"""E03: label-free agreement threshold vs. learned allocators on the E01 GSM8K replay (CPU only).

Copies the replay functions from runs/p0/sim_alloc.py (run_online, same seed-0 quarter fit split, same sim pool,
same RNG streams) and adds one label-free arm, `threshold`, ordered like ThresholdAllocator._order in allocators.py:
agreement ascending, then n_distinct descending. This arm reads no labels. The `static` arm (logistic classifier fit
on labeled fit groups) is kept for reference only.

Metric: corrupted-reward admitted rate = 1 - caught / total_errors, where total_errors is the number of proxy errors
in the pool being replayed. D = admitted(threshold) - admitted(arm). Positive D means the arm admits fewer corrupted
rewards than the label-free threshold.

Bootstrap: resample the sim pool (groups) with replacement. Per resample, recompute the threshold order, rerun
ts_online and greedy_online over all seeds, and compute admitted rates against that resample's own error total.
"""
import os

os.environ.setdefault("OMP_NUM_THREADS", "1")
import sys
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
LABEL = "judge_error"
CHEAP = ["agreement", "n_distinct", "mean_tokens", "std_tokens", "any_truncated", "log_qlen"]
EMB = [f"pc{i}" for i in range(16)]
FEATS = {"cheap": CHEAP, "cheap+emb": CHEAP + EMB}
E01_JSON = "/homes/rdubash/avrl_live/CASCADE/avrl/runs/p0/sim_cell100_cheap+emb.json"
BOOT_SEED = 0  # rng for bootstrap resample indices (separate from the allocator seeds 0..SEEDS-1)
ARMS = ["uniform", "static", "threshold", "threshold_input_order", "greedy_online", "ts_online"]

_BASE, _G, _COLS = None, None, None  # set in each worker by _init_worker


def make_pool(d, G, cols):
    codes, uniq = pd.factorize(d[G].values)
    return {
        "X": d[cols].fillna(0).values,
        "y": d[LABEL].values.astype(int),
        "codes": codes,
        "n_b": len(uniq),
        "members": [np.where(codes == b)[0] for b in range(len(uniq))],
        "ag": d["agreement"].values,
        "nd": d["n_distinct"].values,
    }


def threshold_order(ag, nd, rng=None):
    """ThresholdAllocator._order key: agreement ascending, then n_distinct descending.
    Exact ties keep input order (rng None) or get a seeded random order (rng given)."""
    tb = rng.random(len(ag)) if rng is not None else np.arange(len(ag))
    return np.lexsort((tb, -nd, ag))  # lexsort: last key is primary, sort is stable


def run_online(P, budget, seed, greedy=False, R=None):
    """Copied from runs/p0/sim_alloc.py run_online, with the pool passed in as P.
    ts_online: Thompson over buckets, classifier refit every R labels on labels collected so far.
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


def arm_caught(P, arm, budget, seeds, static_order=None):
    """Caught-error counts per seed for one arm. Deterministic arms return a single value."""
    y = P["y"]
    if arm == "uniform":
        return np.array([y[np.random.default_rng(s).permutation(len(y))[:budget]].sum() for s in seeds], float)
    if arm == "static":
        return np.array([y[static_order[:budget]].sum()], float)
    if arm == "threshold":  # label-free, random tie-break per seed
        return np.array([y[threshold_order(P["ag"], P["nd"], np.random.default_rng(s))[:budget]].sum() for s in seeds], float)
    if arm == "threshold_input_order":  # label-free, deterministic tie-break (pool order), sensitivity row
        return np.array([y[threshold_order(P["ag"], P["nd"])[:budget]].sum()], float)
    if arm == "greedy_online":
        return np.array([run_online(P, budget, s, greedy=True) for s in seeds], float)
    if arm == "ts_online":
        return np.array([run_online(P, budget, s) for s in seeds], float)
    raise ValueError(arm)


def _init_worker(base, G, cols):
    global _BASE, _G, _COLS
    _BASE, _G, _COLS = base, G, cols


def _one_resample(task):
    idx, budget, seeds = task
    P = make_pool(_BASE.iloc[idx], _G, _COLS)
    tot = int(P["y"].sum())
    h = {arm: float(arm_caught(P, arm, budget, seeds).mean()) for arm in ["threshold", "ts_online", "greedy_online"]}
    return tot, h


def pct_ci(x):
    x = np.asarray(x, float)
    return [float(np.percentile(x, 2.5)), float(np.percentile(x, 97.5))]


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
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    G, cols = args.cell, FEATS[args.features]
    budgets = [float(b) for b in args.budgets.split(",")]
    seeds = list(range(args.seeds))
    os.makedirs(args.out, exist_ok=True)
    t0 = time.time()

    # Same split as runs/p0/sim_alloc.py: keep buckets with >= MIN groups, seed-0 shuffle of bucket values, first quarter = fit.
    df = pd.read_parquet(args.feats)
    cnt = df[G].value_counts(); df = df[df[G].isin(cnt[cnt >= MIN].index)].reset_index(drop=True)
    rng0 = np.random.default_rng(0); groups = df[G].unique(); rng0.shuffle(groups)
    fit_g = set(groups[: len(groups) // 4]); fit = df[df[G].isin(fit_g)]; sim = df[~df[G].isin(fit_g)].reset_index(drop=True)
    clf = make_pipeline(StandardScaler(), LogisticRegression(max_iter=1000, class_weight="balanced")).fit(fit[cols].fillna(0).values, fit[LABEL])
    sim["static"] = clf.predict_proba(sim[cols].fillna(0).values)[:, 1]  # reference arm only (uses labels on fit split)
    static_order = np.argsort(-sim["static"].values)

    P = make_pool(sim, G, cols)
    n_sim = len(sim); total = int(P["y"].sum())
    print(f"feats {os.path.basename(args.feats)} cell {G} features {args.features} ({len(cols)} cols) | sim docs {n_sim} buckets {P['n_b']} "
          f"base {P['y'].mean():.4f} total_errors {total} median bucket {int(np.median([len(m) for m in P['members']]))}", flush=True)

    point = {}
    for bf in budgets:
        budget = int(bf * n_sim)
        point[str(bf)] = {}
        for arm in ARMS:
            h = arm_caught(P, arm, budget, seeds, static_order)
            point[str(bf)][arm] = {"budget": budget, "caught_mean": float(h.mean()), "caught_sd": float(h.std()),
                                   "admitted": 1 - float(h.mean()) / total, "n_runs": len(h)}
    D_point = {str(bf): {arm: point[str(bf)]["threshold"]["admitted"] - point[str(bf)][arm]["admitted"] for arm in ["ts_online", "greedy_online"]} for bf in budgets}

    print("\n| budget | budget_n | " + " | ".join(ARMS) + " |\n|---|---|" + "---|" * len(ARMS), flush=True)
    for bf in budgets:
        r = point[str(bf)]
        print(f"| {bf:.0%} | {r['uniform']['budget']} | " + " | ".join(f"{r[a]['caught_mean']:.2f}+-{r[a]['caught_sd']:.2f} (adm {r[a]['admitted']:.3f})" for a in ARMS) + " |", flush=True)
    print("\nD = admitted(threshold) - admitted(arm), point estimates:", flush=True)
    for bf in budgets:
        print(f"  {bf:.0%}: D(ts_online) {D_point[str(bf)]['ts_online']:+.4f}   D(greedy_online) {D_point[str(bf)]['greedy_online']:+.4f}", flush=True)

    # Reproduction check: same code path as E01 must give E01's stored numbers.
    repro = None
    if args.features == "cheap+emb" and G == "cell100" and os.path.exists(E01_JSON):
        ref = json.load(open(E01_JSON))
        diffs = [abs(ref[f"budget_{bf}"][arm]["mean"] - point[str(bf)][arm]["caught_mean"]) for bf in budgets for arm in ["uniform", "static", "greedy_online", "ts_online"]]
        repro = {"ref": E01_JSON, "max_abs_diff_caught_mean": float(max(diffs)), "matches_E01": bool(max(diffs) < 1e-9)}
        print(f"\nreproduction vs E01 json: max abs diff {max(diffs):.3e}, matches {repro['matches_E01']}", flush=True)

    # Bootstrap at boot_budget: resample sim groups with replacement, recompute everything per resample.
    bkey = str(args.boot_budget)
    assert bkey in point, "boot_budget must be one of --budgets"
    boot_budget = int(args.boot_budget * n_sim)
    rng_b = np.random.default_rng(BOOT_SEED)
    tasks = [(np.sort(rng_b.integers(0, n_sim, size=n_sim)), boot_budget, seeds) for _ in range(args.boot)]
    base = sim[list(dict.fromkeys([G, LABEL, "agreement", "n_distinct"] + cols))].reset_index(drop=True)  # cols already holds agreement/n_distinct
    t1 = time.time()
    with ProcessPoolExecutor(max_workers=args.workers, initializer=_init_worker, initargs=(base, G, cols)) as ex:
        res = list(ex.map(_one_resample, tasks, chunksize=4))
    boot_secs = time.time() - t1
    adm = {k: np.array([1 - h[k] / tot for tot, h in res]) for k in ["threshold", "ts_online", "greedy_online"]}
    D_boot = {arm: adm["threshold"] - adm[arm] for arm in ["ts_online", "greedy_online"]}
    D_boot_ci = {arm: {"ci95": pct_ci(D_boot[arm]), "boot_mean": float(D_boot[arm].mean()), "frac_resamples_D_gt_0": float((D_boot[arm] > 0).mean()), "n_resamples": len(D_boot[arm])} for arm in D_boot}
    print(f"\nbootstrap {len(res)} resamples at {args.boot_budget:.0%} ({boot_budget} docs), {args.workers} workers, {boot_secs:.1f}s", flush=True)
    for arm in D_boot:
        c = D_boot_ci[arm]["ci95"]
        print(f"  D({arm}) point {D_point[bkey][arm]:+.4f}  boot mean {D_boot[arm].mean():+.4f}  95% CI [{c[0]:+.4f}, {c[1]:+.4f}]  frac>0 {D_boot_ci[arm]['frac_resamples_D_gt_0']:.3f}", flush=True)

    # Verdict (primary arm only): PASS iff the 95% CI of D(ts_online) excludes 0 and D > 0 at 25%.
    d_pt, (lo, hi) = D_point[bkey]["ts_online"], D_boot_ci["ts_online"]["ci95"]
    passed = lo > 0 and d_pt > 0
    verdict = "PASS" if passed else "KILL"
    if passed:
        reason = "95% CI of D(ts_online) excludes 0 and D > 0 at 25%"
    elif d_pt <= 0:
        reason = "D(ts_online) <= 0 at 25%: the label-free threshold admits as few or fewer corrupted rewards as ts_online"
    else:
        reason = "95% CI of D(ts_online) includes 0 at 25%"
    summary = (f"E03 {G} {args.features} 25%: admitted threshold {point[bkey]['threshold']['admitted']:.3f}, ts_online {point[bkey]['ts_online']['admitted']:.3f}, "
               f"greedy_online {point[bkey]['greedy_online']['admitted']:.3f}; D(ts_online) {d_pt:+.3f} CI [{lo:+.3f}, {hi:+.3f}]; "
               f"D(greedy_online) {D_point[bkey]['greedy_online']:+.3f} CI {D_boot_ci['greedy_online']['ci95'][0]:+.3f} to {D_boot_ci['greedy_online']['ci95'][1]:+.3f}; verdict {verdict}")
    print("\n" + summary, flush=True)

    out = {
        "exp": "E03",
        "verdict": verdict,
        "verdict_reason": reason,
        "summary": summary,
        "config": {"feats": args.feats, "cell": G, "features": args.features, "n_features": len(cols), "feature_list": cols,
                   "min_bucket_size": MIN, "K": K, "seeds": args.seeds, "fit_split": "seed-0 shuffle of bucket values, first quarter = fit (labels used only by static reference)",
                   "threshold_tiebreak": "primary: seeded random among exact (agreement, n_distinct) ties; sensitivity row: pool input order",
                   "boot_resamples": args.boot, "boot_budget_frac": args.boot_budget, "boot_budget_docs": boot_budget, "boot_seed": BOOT_SEED,
                   "boot_unit": "sim groups with replacement; resample keeps pool order; admitted rates use each resample's own error total"},
        "pool": {"n_sim": n_sim, "n_buckets": P["n_b"], "total_errors": total, "budget_docs": {str(bf): int(bf * n_sim) for bf in budgets}},
        "point": point,
        "D_point": D_point,
        "D_bootstrap": D_boot_ci,
        "D_bootstrap_draws": {arm: D_boot[arm].tolist() for arm in D_boot},
        "reproduction_vs_E01": repro,
        "wall_seconds": round(time.time() - t0, 1),
        "bootstrap_wall_seconds": round(boot_secs, 1),
    }
    with open(os.path.join(args.out, "result.json"), "w") as f:
        json.dump(out, f, indent=1)
    print(f"wrote {os.path.join(args.out, 'result.json')}", flush=True)


if __name__ == "__main__":
    main()
