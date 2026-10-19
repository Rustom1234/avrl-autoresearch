"""L17d readout: labeled static ablation by feature family on the L14b sim pool, K = 8 majority proxy (CPU only).

Copy of avrl/exp/l14b_claim2_readout.py (L14b itself is not edited). Pre-registered in avrl/RESEARCH_LOG.md ("L17d
pre-registration"). Changes from L14b: (1) the one static reference arm becomes three fixed static logistic models, each fit
once on the fit quarter: static (cheap six plus pc0 to pc15, the L14b reference), static_cheap6, static_pc; (2) the bootstrap
also recomputes the threshold and the three static arms at 25% and 50%; (3) the verdict is on
Delta = admitted(static_cheap6) - admitted(static) at 25% and 50%. The other L14b arms are unchanged and are checked by
avrl/exp/L17d_control.py. The L14b docstring follows.

L14b readout: label-free agreement threshold vs. learned allocators on the L14a gate-passing GSM8K pool (CPU only).

Copy of avrl/exp/e03_threshold_bootstrap.py (E03 itself is not edited). Pre-registered in avrl/RESEARCH_LOG.md
("L14b pre-registration"). Changes from E03: (1) the primary interval is a bucket-cluster bootstrap (cell100 buckets resampled
with replacement, seed 123, repeated buckets relabeled as new clusters, as avrl/exp/l5a_learned_class.py); the group bootstrap
(BOOT_SEED 0) is kept as the secondary interval; (2) the verdict uses the primary interval; (3) the E01 reproduction block is
removed because avrl/runs/p0 is not this pool. The code check against E03 is a separate command (avrl/exp/l14b_code_check.py).

Metric: corrupted-reward admitted rate = 1 - caught / total_errors, where total_errors is the number of proxy errors
in the sim pool being replayed. D = admitted(threshold) - admitted(arm). Positive D means the arm admits fewer corrupted
rewards than the label-free threshold.

Verdict (L17d, pre-registered): PASS iff at 25% or at 50% point Delta > 0 and the lower bound of the primary bucket-cluster
interval of Delta > 0. KILL iff point Delta <= 0 at both 25% and 50%. INCONCLUSIVE otherwise. (The L14b D(ts_online) verdict
is no longer computed here; its draws are kept for the control.)

Bootstraps at boot_budget (25%) for ts_online and greedy_online (control), and at 25% and 50% for the threshold and the static
arms. Each resample recomputes the threshold order, ts_online and greedy_online over all seeds,
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
ARMS = ["uniform", "static", "static_cheap6", "static_pc", "threshold", "threshold_input_order", "greedy_online", "ts_online"]
STATIC_SETS = {"static": CHEAP + EMB, "static_cheap6": CHEAP, "static_pc": EMB}  # fixed static feature sets (L17d)
STATIC_ARMS = list(STATIC_SETS)
BOOT_ARMS = {"0.25": ["threshold", "ts_online", "greedy_online"] + STATIC_ARMS, "0.5": ["threshold"] + STATIC_ARMS}
METRICS = ["Delta_cheap6_minus_cheapPC", "Delta_pc_minus_cheapPC", "D_static_cheapPC", "D_static_cheap6", "D_static_pc"]

_BASE, _G, _COLS, _MEMBERS = None, None, None, None  # set in each worker by _init_worker


def static_metrics(a):
    """a: arm -> admitted (float or numpy array). Delta is the L17d decision metric; the others are descriptive."""
    return {"Delta_cheap6_minus_cheapPC": a["static_cheap6"] - a["static"], "Delta_pc_minus_cheapPC": a["static_pc"] - a["static"],
            "D_static_cheapPC": a["threshold"] - a["static"], "D_static_cheap6": a["threshold"] - a["static_cheap6"],
            "D_static_pc": a["threshold"] - a["static_pc"]}


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
        "score": {arm: d["s_" + arm].values for arm in STATIC_ARMS},
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


def arm_caught(P, arm, budget, seeds):
    """Caught-error counts per seed for one arm. Deterministic arms return a single value."""
    y = P["y"]
    if arm == "uniform":
        return np.array([y[np.random.default_rng(s).permutation(len(y))[:budget]].sum() for s in seeds], float)
    if arm in STATIC_ARMS:  # fixed model scores, ranked as in L14b (np.argsort(-score)); no refit inside a resample
        return np.array([y[np.argsort(-P["score"][arm])[:budget]].sum()], float)
    if arm == "threshold":  # label-free, random tie-break per seed
        return np.array([y[threshold_order(P["ag"], P["nd"], np.random.default_rng(s))[:budget]].sum() for s in seeds], float)
    if arm == "threshold_input_order":  # label-free, deterministic tie-break (pool order), sensitivity row
        return np.array([y[threshold_order(P["ag"], P["nd"])[:budget]].sum()], float)
    if arm == "greedy_online":
        return np.array([run_online(P, budget, s, greedy=True) for s in seeds], float)
    if arm == "ts_online":
        return np.array([run_online(P, budget, s) for s in seeds], float)
    raise ValueError(arm)


def _init_worker(base, G, cols, members):
    global _BASE, _G, _COLS, _MEMBERS
    _BASE, _G, _COLS, _MEMBERS = base, G, cols, members


def _eval(P, budgets, seeds):
    """budgets: bkey -> docs. Returns the pool's error total and the caught means of BOOT_ARMS[bkey] at each budget."""
    tot = int(P["y"].sum())
    h = {bk: {arm: float(arm_caught(P, arm, n, seeds).mean()) for arm in BOOT_ARMS[bk]} for bk, n in budgets.items()}
    return tot, h


def _one_resample(task):
    idx, budgets, seeds = task
    return _eval(make_pool(_BASE.iloc[idx], _G, _COLS), budgets, seeds)


def _cluster_resample(task):
    draws, budgets, seeds = task
    return _eval(make_pool(cluster_df(_BASE, _G, _MEMBERS, draws), _G, _COLS), budgets, seeds)


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
    for arm, fc in STATIC_SETS.items():  # three fixed static models, fit on the fit split only (labels used there, nowhere else)
        clf = make_pipeline(StandardScaler(), LogisticRegression(max_iter=1000, class_weight="balanced")).fit(fit[fc].fillna(0).values, fit[LABEL])
        sim["s_" + arm] = clf.predict_proba(sim[fc].fillna(0).values)[:, 1]

    P = make_pool(sim, G, cols)
    n_sim = len(sim); total = int(P["y"].sum())
    print(f"feats {os.path.basename(args.feats)} cell {G} features {args.features} ({len(cols)} cols) | input rows {n_input} buckets {n_buckets_input} "
          f"| sim docs {n_sim} buckets {P['n_b']} base {P['y'].mean():.4f} total_errors {total} median bucket {int(np.median([len(m) for m in P['members']]))}", flush=True)

    point = {}
    for bf in budgets:
        budget = int(bf * n_sim)
        point[str(bf)] = {}
        for arm in ARMS:
            h = arm_caught(P, arm, budget, seeds)
            point[str(bf)][arm] = {"budget": budget, "caught_mean": float(h.mean()), "caught_sd": float(h.std()),
                                   "admitted": 1 - float(h.mean()) / total, "n_runs": len(h)}
            if arm in STATIC_ARMS:  # diagnostic only: other sim docs whose score equals the cutoff doc's score
                s = P["score"][arm]
                point[str(bf)][arm]["cutoff_ties"] = int((s == s[np.argsort(-s)[budget - 1]]).sum()) - 1
    D_point = {str(bf): {arm: point[str(bf)]["threshold"]["admitted"] - point[str(bf)][arm]["admitted"] for arm in ["ts_online", "greedy_online"]} for bf in budgets}
    Delta_point = {str(bf): static_metrics({arm: point[str(bf)][arm]["admitted"] for arm in ARMS}) for bf in budgets}

    print("\n| budget | budget_n | " + " | ".join(ARMS) + " |\n|---|---|" + "---|" * len(ARMS), flush=True)
    for bf in budgets:
        r = point[str(bf)]
        print(f"| {bf:.0%} | {r['uniform']['budget']} | " + " | ".join(f"{r[a]['caught_mean']:.2f}+-{r[a]['caught_sd']:.2f} (adm {r[a]['admitted']:.3f})" for a in ARMS) + " |", flush=True)
    print("\nD = admitted(threshold) - admitted(arm), point estimates:", flush=True)
    for bf in budgets:
        print(f"  {bf:.0%}: D(ts_online) {D_point[str(bf)]['ts_online']:+.4f}   D(greedy_online) {D_point[str(bf)]['greedy_online']:+.4f}", flush=True)
    print("\nL17d metrics, point estimates (Delta = admitted(static_cheap6) - admitted(static); positive = cheap six admit more):", flush=True)
    for bf in budgets:
        m = Delta_point[str(bf)]
        print(f"  {bf:.0%}: Delta(cheap6 - cheap+PC) {m['Delta_cheap6_minus_cheapPC']:+.4f}   Delta(pc - cheap+PC) {m['Delta_pc_minus_cheapPC']:+.4f}   "
              f"D(static cheap+PC) {m['D_static_cheapPC']:+.4f}   D(static cheap6) {m['D_static_cheap6']:+.4f}   D(static pc) {m['D_static_pc']:+.4f}", flush=True)

    # No E01 reproduction here (avrl/runs/p0 is a different pool). The code check against E03 is avrl/exp/l14b_code_check.py.
    repro = None

    # Bootstraps at boot_budget. Primary: bucket-cluster (seed 123). Secondary: group (BOOT_SEED 0, same draws as E03).
    bkey = "0.25"  # the ts_online and greedy_online control draws stay at 25%, as in L14b
    assert args.boot_budget == 0.25 and bkey in point and "0.5" in point, "L17d needs --boot_budget 0.25 and budgets 0.25 and 0.5"
    boot_budgets = {"0.25": int(0.25 * n_sim), "0.5": int(0.5 * n_sim)}  # fixed docs per resample, as in L14b
    base = sim[list(dict.fromkeys([G, LABEL, "agreement", "n_distinct"] + cols + ["s_" + a for a in STATIC_ARMS]))].reset_index(drop=True)
    members = P["members"]  # sim-pool rows per bucket, same row order as base
    rng_b = np.random.default_rng(BOOT_SEED)
    group_tasks = [(np.sort(rng_b.integers(0, n_sim, size=n_sim)), boot_budgets, seeds) for _ in range(args.boot)]
    rng_c = np.random.default_rng(CLUSTER_SEED)
    cluster_tasks = [(rng_c.integers(0, P["n_b"], size=P["n_b"]), boot_budgets, seeds) for _ in range(args.boot)]
    t1 = time.time()
    with ProcessPoolExecutor(max_workers=args.workers, initializer=_init_worker, initargs=(base, G, cols, members)) as ex:
        res_g = list(ex.map(_one_resample, group_tasks, chunksize=4))
        res_c = list(ex.map(_cluster_resample, cluster_tasks, chunksize=4))
    boot_secs = time.time() - t1

    def interval(x):
        return {"ci95": pct_ci(x), "boot_mean": float(x.mean()), "frac_resamples_D_gt_0": float((x > 0).mean()), "n_resamples": len(x)}

    def summarize(res):
        adm = {bk: {k: np.array([1 - h[bk][k] / tot for tot, h in res]) for k in BOOT_ARMS[bk]} for bk in boot_budgets}
        draws = {arm: adm[bkey]["threshold"] - adm[bkey][arm] for arm in ["ts_online", "greedy_online"]}
        ci = {arm: interval(draws[arm]) for arm in draws}
        sdraws = {bk: static_metrics(adm[bk]) for bk in boot_budgets}
        sci = {bk: {m: interval(sdraws[bk][m]) for m in METRICS} for bk in boot_budgets}
        return ci, draws, sci, sdraws

    D_grp_ci, D_grp, S_grp_ci, S_grp = summarize(res_g)
    D_clu_ci, D_clu, S_clu_ci, S_clu = summarize(res_c)
    print(f"\nbootstrap {args.boot} bucket-cluster resamples (CLUSTER_SEED {CLUSTER_SEED}, primary) and {args.boot} group resamples "
          f"(BOOT_SEED {BOOT_SEED}, secondary) at 25% ({boot_budgets['0.25']} docs) and 50% ({boot_budgets['0.5']} docs), {args.workers} workers, {boot_secs:.1f}s", flush=True)
    for name, Dci, Dd in [("cluster (primary)", D_clu_ci, D_clu), ("group (secondary)", D_grp_ci, D_grp)]:
        for arm in Dd:
            c = Dci[arm]["ci95"]
            print(f"  {name} D({arm}) point {D_point[bkey][arm]:+.4f}  boot mean {Dd[arm].mean():+.4f}  95% CI [{c[0]:+.4f}, {c[1]:+.4f}]  frac>0 {Dci[arm]['frac_resamples_D_gt_0']:.3f}", flush=True)
    for name, Sci, Sd in [("cluster (primary)", S_clu_ci, S_clu), ("group (secondary)", S_grp_ci, S_grp)]:
        for bf in boot_budgets:
            for m in METRICS:
                c = Sci[bf][m]["ci95"]
                print(f"  {name} {m} at {float(bf):.0%} point {Delta_point[bf][m]:+.4f}  boot mean {Sd[bf][m].mean():+.4f}  "
                      f"95% CI [{c[0]:+.4f}, {c[1]:+.4f}]  frac>0 {Sci[bf][m]['frac_resamples_D_gt_0']:.3f}", flush=True)

    # Verdict (L17d, pre-registered): PASS iff at 25% or 50% point Delta > 0 and the lower bound of its primary bucket-cluster
    # interval > 0; KILL iff point Delta <= 0 at both budgets; INCONCLUSIVE otherwise. Delta = admitted(static_cheap6) - admitted(static).
    M = "Delta_cheap6_minus_cheapPC"
    dpt = {bf: Delta_point[bf][M] for bf in boot_budgets}
    lows = {bf: S_clu_ci[bf][M]["ci95"][0] for bf in boot_budgets}
    highs = {bf: S_clu_ci[bf][M]["ci95"][1] for bf in boot_budgets}
    if any(dpt[bf] > 0 and lows[bf] > 0 for bf in boot_budgets):
        verdict = "PASS"
    elif all(dpt[bf] <= 0 for bf in boot_budgets):
        verdict = "KILL"
    else:
        verdict = "INCONCLUSIVE"
    reason = (f"point Delta(cheap6 - cheap+PC) {dpt['0.25']:+.4f} at 25% and {dpt['0.5']:+.4f} at 50%; primary bucket-cluster 95% "
              f"lower bounds {lows['0.25']:+.4f} at 25% and {lows['0.5']:+.4f} at 50%")
    pt, cl = point[bkey], S_clu_ci
    summary = (f"L17d {G} {args.features} (K = 8 pool): admitted at 25% threshold {pt['threshold']['admitted']:.3f}, static (cheap+PC) "
               f"{pt['static']['admitted']:.3f}, static_cheap6 {pt['static_cheap6']['admitted']:.3f}, static_pc {pt['static_pc']['admitted']:.3f}; "
               f"Delta(cheap6 - cheap+PC) 25% {dpt['0.25']:+.3f} cluster CI [{lows['0.25']:+.3f}, {highs['0.25']:+.3f}] "
               f"group CI [{S_grp_ci['0.25'][M]['ci95'][0]:+.3f}, {S_grp_ci['0.25'][M]['ci95'][1]:+.3f}]; "
               f"50% {dpt['0.5']:+.3f} cluster CI [{lows['0.5']:+.3f}, {highs['0.5']:+.3f}]; "
               f"D(static cheap+PC) 25% {Delta_point['0.25']['D_static_cheapPC']:+.3f} cluster CI "
               f"[{cl['0.25']['D_static_cheapPC']['ci95'][0]:+.3f}, {cl['0.25']['D_static_cheapPC']['ci95'][1]:+.3f}]; verdict {verdict}")
    print("\n" + summary, flush=True)

    out = {
        "exp": "L17d",
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
                   "boot_budget_frac": args.boot_budget, "boot_budget_docs": boot_budgets, "static_feature_sets": STATIC_SETS},
        "pool": {"n_sim": n_sim, "n_buckets": P["n_b"], "total_errors": total, "budget_docs": {str(bf): int(bf * n_sim) for bf in budgets}},
        "point": point,
        "D_point": D_point,
        "Delta_point": Delta_point,
        "static_bootstrap_cluster": S_clu_ci,
        "static_bootstrap_group": S_grp_ci,
        "static_bootstrap_cluster_draws": {bf: {m: S_clu[bf][m].tolist() for m in METRICS} for bf in boot_budgets},
        "static_bootstrap_group_draws": {bf: {m: S_grp[bf][m].tolist() for m in METRICS} for bf in boot_budgets},
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
