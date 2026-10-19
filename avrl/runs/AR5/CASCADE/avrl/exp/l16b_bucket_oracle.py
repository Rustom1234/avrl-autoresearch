"""L16b readout: bucket oracle vs. label-free agreement threshold on the L14b gate-passing GSM8K pool (CPU only).

Copy of avrl/exp/l14b_claim2_readout.py (L14b itself is not edited). Pre-registered in avrl/RESEARCH_LOG.md
("L16b pre-registration"). Changes from L14b: one arm added, bucket_oracle (labels used, upper bound, deterministic).
Every other arm, bootstrap and output line is unchanged, so the control reproduces avrl/runs/L14b/result.json. The primary
interval is the bucket-cluster bootstrap (cell100 buckets resampled with replacement, seed 123, repeated buckets relabeled as
new clusters, bucket rates rebuilt per resample); the group bootstrap (BOOT_SEED 0) is the secondary interval.

Metric: corrupted-reward admitted rate = 1 - caught / total_errors, where total_errors is the number of proxy errors
in the sim pool being replayed. D = admitted(threshold) - admitted(arm). Positive D means the arm admits fewer corrupted
rewards than the label-free threshold.

Verdict (pre-registered, L16b): PASS iff point D(bucket_oracle) > 0 at 25% and the lower bound of the primary bucket-cluster
interval > 0. KILL iff point D(bucket_oracle) <= 0 at 25%. INCONCLUSIVE iff point D > 0 and the primary interval includes 0.
bucket_oracle (labels used, upper bound): buckets ordered by true proxy_error rate, descending (equal rates by bucket code), kept
contiguous; inside a bucket, the threshold key (agreement ascending, n_distinct descending), then pool order. Caught = sum of
proxy_error over the first budget docs in that order.

Bootstraps at boot_budget (25%), each resample recomputes the threshold order, bucket_oracle, ts_online and greedy_online over all seeds,
with admitted rates computed against that resample's own error total.
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
LABEL = "proxy_error"
CHEAP = ["agreement", "n_distinct", "mean_tokens", "std_tokens", "any_truncated", "log_qlen"]
EMB = [f"pc{i}" for i in range(16)]
FEATS = {"cheap": CHEAP, "cheap+emb": CHEAP + EMB}
BOOT_SEED = 0  # rng for group bootstrap resample indices (separate from the allocator seeds 0..SEEDS-1), same draws as E03
CLUSTER_SEED = 123  # rng for the bucket-cluster bootstrap (primary interval), as avrl/exp/l5a_learned_class.py
ARMS = ["uniform", "static", "threshold", "threshold_input_order", "greedy_online", "ts_online", "bucket_oracle"]

_BASE, _G, _COLS, _MEMBERS = None, None, None, None  # set in each worker by _init_worker


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


def cluster_df(base, G, members, draws):
    """Bucket-cluster resample as avrl/exp/l5a_learned_class.py cluster_pool: draw k keeps every group of bucket draws[k] and
    gets cell code k, so a repeated bucket is a new cluster. Rows keep the sim-pool order (stable sort on the source row index)."""
    rows = np.concatenate([members[b] for b in draws])
    codes = np.concatenate([np.full(len(members[b]), k) for k, b in enumerate(draws)])
    o = np.argsort(rows, kind="stable")
    d = base.iloc[rows[o]].reset_index(drop=True).copy()
    d[G] = codes[o]
    return d


def threshold_order(ag, nd, rng=None):
    """ThresholdAllocator._order key: agreement ascending, then n_distinct descending.
    Exact ties keep input order (rng None) or get a seeded random order (rng given)."""
    tb = rng.random(len(ag)) if rng is not None else np.arange(len(ag))
    return np.lexsort((tb, -nd, ag))  # lexsort: last key is primary, sort is stable


def bucket_oracle_order(P):
    """bucket_oracle (labels used, upper bound): buckets by true proxy_error rate, descending, equal rates by bucket code (first
    appearance in the pool). Buckets stay contiguous. Inside a bucket: threshold key (agreement ascending, n_distinct descending),
    then pool order."""
    y, codes, n_b = P["y"], P["codes"], P["n_b"]
    rate = np.bincount(codes, weights=y, minlength=n_b) / np.bincount(codes, minlength=n_b)
    rank = np.empty(n_b, int)
    rank[np.lexsort((np.arange(n_b), -rate))] = np.arange(n_b)  # rank[b] = position of bucket b in the oracle order
    return np.lexsort((np.arange(len(y)), -P["nd"], P["ag"], rank[codes]))  # last key is primary


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
    if arm == "bucket_oracle":  # labels used, upper bound, deterministic (one value per budget)
        return np.array([y[bucket_oracle_order(P)[:budget]].sum()], float)
    raise ValueError(arm)


def _init_worker(base, G, cols, members):
    global _BASE, _G, _COLS, _MEMBERS
    _BASE, _G, _COLS, _MEMBERS = base, G, cols, members


def _eval(P, budget, seeds):
    tot = int(P["y"].sum())
    h = {arm: float(arm_caught(P, arm, budget, seeds).mean()) for arm in ["threshold", "ts_online", "greedy_online", "bucket_oracle"]}
    return tot, h


def _one_resample(task):
    idx, budget, seeds = task
    return _eval(make_pool(_BASE.iloc[idx], _G, _COLS), budget, seeds)


def _cluster_resample(task):
    draws, budget, seeds = task
    return _eval(make_pool(cluster_df(_BASE, _G, _MEMBERS, draws), _G, _COLS), budget, seeds)


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

    # Same split as runs/p0/sim_alloc.py (and E03): keep buckets with >= MIN groups, seed-0 shuffle of bucket values, first quarter = fit.
    df = pd.read_parquet(args.feats)
    n_input, n_buckets_input = len(df), int(df[G].nunique())
    split_counts = {k: int(v) for k, v in df["split"].value_counts().items()} if "split" in df.columns else None
    cnt = df[G].value_counts(); n_buckets_kept = int((cnt >= MIN).sum())
    df = df[df[G].isin(cnt[cnt >= MIN].index)].reset_index(drop=True)
    rng0 = np.random.default_rng(0); groups = df[G].unique(); rng0.shuffle(groups)
    fit_g = set(groups[: len(groups) // 4]); fit = df[df[G].isin(fit_g)]; sim = df[~df[G].isin(fit_g)].reset_index(drop=True)
    clf = make_pipeline(StandardScaler(), LogisticRegression(max_iter=1000, class_weight="balanced")).fit(fit[cols].fillna(0).values, fit[LABEL])
    sim["static"] = clf.predict_proba(sim[cols].fillna(0).values)[:, 1]  # reference arm only (uses labels on fit split)
    static_order = np.argsort(-sim["static"].values)

    P = make_pool(sim, G, cols)
    n_sim = len(sim); total = int(P["y"].sum())
    print(f"feats {os.path.basename(args.feats)} cell {G} features {args.features} ({len(cols)} cols) | input rows {n_input} buckets {n_buckets_input} "
          f"| sim docs {n_sim} buckets {P['n_b']} base {P['y'].mean():.4f} total_errors {total} median bucket {int(np.median([len(m) for m in P['members']]))}", flush=True)

    point = {}
    for bf in budgets:
        budget = int(bf * n_sim)
        point[str(bf)] = {}
        for arm in ARMS:
            h = arm_caught(P, arm, budget, seeds, static_order)
            point[str(bf)][arm] = {"budget": budget, "caught_mean": float(h.mean()), "caught_sd": float(h.std()),
                                   "admitted": 1 - float(h.mean()) / total, "n_runs": len(h)}
    D_point = {str(bf): {arm: point[str(bf)]["threshold"]["admitted"] - point[str(bf)][arm]["admitted"] for arm in ["bucket_oracle", "ts_online", "greedy_online"]} for bf in budgets}

    print("\n| budget | budget_n | " + " | ".join(ARMS) + " |\n|---|---|" + "---|" * len(ARMS), flush=True)
    for bf in budgets:
        r = point[str(bf)]
        print(f"| {bf:.0%} | {r['uniform']['budget']} | " + " | ".join(f"{r[a]['caught_mean']:.2f}+-{r[a]['caught_sd']:.2f} (adm {r[a]['admitted']:.3f})" for a in ARMS) + " |", flush=True)
    print("\nD = admitted(threshold) - admitted(arm), point estimates:", flush=True)
    for bf in budgets:
        print(f"  {bf:.0%}: D(bucket_oracle) {D_point[str(bf)]['bucket_oracle']:+.4f}   D(ts_online) {D_point[str(bf)]['ts_online']:+.4f}   D(greedy_online) {D_point[str(bf)]['greedy_online']:+.4f}", flush=True)

    # No E01 reproduction here (avrl/runs/p0 is a different pool). The code check against E03 is avrl/exp/l14b_code_check.py.
    repro = None

    # Bootstraps at boot_budget. Primary: bucket-cluster (seed 123). Secondary: group (BOOT_SEED 0, same draws as E03).
    bkey = str(args.boot_budget)
    assert bkey in point, "boot_budget must be one of --budgets"
    boot_budget = int(args.boot_budget * n_sim)
    base = sim[list(dict.fromkeys([G, LABEL, "agreement", "n_distinct"] + cols))].reset_index(drop=True)  # cols already holds agreement/n_distinct
    members = P["members"]  # sim-pool rows per bucket, same row order as base
    rng_b = np.random.default_rng(BOOT_SEED)
    group_tasks = [(np.sort(rng_b.integers(0, n_sim, size=n_sim)), boot_budget, seeds) for _ in range(args.boot)]
    rng_c = np.random.default_rng(CLUSTER_SEED)
    cluster_tasks = [(rng_c.integers(0, P["n_b"], size=P["n_b"]), boot_budget, seeds) for _ in range(args.boot)]
    t1 = time.time()
    with ProcessPoolExecutor(max_workers=args.workers, initializer=_init_worker, initargs=(base, G, cols, members)) as ex:
        res_g = list(ex.map(_one_resample, group_tasks, chunksize=4))
        res_c = list(ex.map(_cluster_resample, cluster_tasks, chunksize=4))
    boot_secs = time.time() - t1

    def summarize(res):
        adm = {k: np.array([1 - h[k] / tot for tot, h in res]) for k in ["threshold", "bucket_oracle", "ts_online", "greedy_online"]}
        draws = {arm: adm["threshold"] - adm[arm] for arm in ["bucket_oracle", "ts_online", "greedy_online"]}
        ci = {arm: {"ci95": pct_ci(draws[arm]), "boot_mean": float(draws[arm].mean()), "frac_resamples_D_gt_0": float((draws[arm] > 0).mean()),
                    "n_resamples": len(draws[arm])} for arm in draws}
        return ci, draws

    D_grp_ci, D_grp = summarize(res_g)
    D_clu_ci, D_clu = summarize(res_c)
    print(f"\nbootstrap {args.boot} bucket-cluster resamples (CLUSTER_SEED {CLUSTER_SEED}, primary) and {args.boot} group resamples "
          f"(BOOT_SEED {BOOT_SEED}, secondary) at {args.boot_budget:.0%} ({boot_budget} docs), {args.workers} workers, {boot_secs:.1f}s", flush=True)
    for name, Dci, Dd in [("cluster (primary)", D_clu_ci, D_clu), ("group (secondary)", D_grp_ci, D_grp)]:
        for arm in Dd:
            c = Dci[arm]["ci95"]
            print(f"  {name} D({arm}) point {D_point[bkey][arm]:+.4f}  boot mean {Dd[arm].mean():+.4f}  95% CI [{c[0]:+.4f}, {c[1]:+.4f}]  frac>0 {Dci[arm]['frac_resamples_D_gt_0']:.3f}", flush=True)

    # Verdict (pre-registered): PASS iff point D > 0 and the lower bound of the primary bucket-cluster interval > 0; KILL iff point D <= 0.
    d_pt = D_point[bkey]["bucket_oracle"]
    lo, hi = D_clu_ci["bucket_oracle"]["ci95"]
    if d_pt <= 0:
        verdict, reason = "KILL", "point D(bucket_oracle) <= 0 at 25%: the label-free threshold admits as few or fewer corrupted rewards as the bucket oracle"
    elif lo > 0:
        verdict, reason = "PASS", "point D(bucket_oracle) > 0 at 25% and the 95% bucket-cluster interval excludes 0"
    else:
        verdict, reason = "INCONCLUSIVE", "point D(bucket_oracle) > 0 at 25% and the 95% bucket-cluster interval includes 0"
    gl, gh = D_grp_ci["bucket_oracle"]["ci95"]
    summary = (f"L16b {G} {args.features} 25%: admitted threshold {point[bkey]['threshold']['admitted']:.3f}, bucket_oracle {point[bkey]['bucket_oracle']['admitted']:.3f}, "
               f"ts_online {point[bkey]['ts_online']['admitted']:.3f}, greedy_online {point[bkey]['greedy_online']['admitted']:.3f}; "
               f"D(bucket_oracle) {d_pt:+.3f} cluster CI [{lo:+.3f}, {hi:+.3f}] group CI [{gl:+.3f}, {gh:+.3f}]; "
               f"D(ts_online) {D_point[bkey]['ts_online']:+.3f}; D(greedy_online) {D_point[bkey]['greedy_online']:+.3f}; verdict {verdict}")
    print("\n" + summary, flush=True)

    out = {
        "exp": "L16b",
        "verdict": verdict,
        "verdict_reason": reason,
        "summary": summary,
        "config": {"feats": args.feats, "cell": G, "features": args.features, "n_features": len(cols), "feature_list": cols,
                   "min_bucket_size": MIN, "K": K, "seeds": args.seeds, "budgets": budgets,
                   "input_rows": n_input, "input_split_counts": split_counts, "input_buckets": n_buckets_input, "buckets_kept_min3": n_buckets_kept,
                   "fit_split": "seed-0 shuffle of bucket values, first quarter = fit (held out of the sim pool; labels used only by the static reference arm)",
                   "fit_buckets_held_out": len(fit_g),
                   "threshold_tiebreak": "threshold: seeded random among exact (agreement, n_distinct) ties; threshold_input_order: pool order",
                   "boot_cluster": {"resamples": args.boot, "seed": CLUSTER_SEED, "unit": "sim-pool cell100 buckets with replacement, duplicates relabeled", "role": "primary"},
                   "boot_group": {"resamples": args.boot, "seed": BOOT_SEED, "unit": "sim groups with replacement, pool order kept", "role": "secondary"},
                   "boot_budget_frac": args.boot_budget, "boot_budget_docs": boot_budget},
        "pool": {"n_sim": n_sim, "n_buckets": P["n_b"], "total_errors": total, "budget_docs": {str(bf): int(bf * n_sim) for bf in budgets}},
        "point": point,
        "D_point": D_point,
        "D_bootstrap_cluster": D_clu_ci,
        "D_bootstrap_group": D_grp_ci,
        "D_bootstrap_cluster_draws": {arm: D_clu[arm].tolist() for arm in D_clu},
        "D_bootstrap_group_draws": {arm: D_grp[arm].tolist() for arm in D_grp},
        "reproduction_vs_E01": repro,
        "wall_seconds": round(time.time() - t0, 1),
        "bootstrap_wall_seconds": round(boot_secs, 1),
    }
    with open(os.path.join(args.out, "result.json"), "w") as f:
        json.dump(out, f, indent=1)
    print(f"wrote {os.path.join(args.out, 'result.json')}", flush=True)


if __name__ == "__main__":
    main()
