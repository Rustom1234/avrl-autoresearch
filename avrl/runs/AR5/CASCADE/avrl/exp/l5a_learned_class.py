"""L5a: the shipped LearnedAllocator (avrl/allocators.py) vs the label-free agreement threshold on the E03 pool (CPU only).

Copied from avrl/exp/e03_threshold_bootstrap.py. E03, allocators.py and the p0 files are not edited; allocators.py is
imported read-only. Arms:
  threshold              label-free ThresholdAllocator order, seeded random tie-break (as E03)
  threshold_input_order  label-free, pool-order tie-break (the class's own cold start order, as E03)
  learned_class          shipped LearnedAllocator with class defaults (refit_every=20, prior_strength=2.0)

learned_class replays the class as the phase-1 loop would call it, at K = 1 per visit: each visit builds records for every
still-unverified pool group, calls select(records, 1, rng), marks the chosen group verified, and calls update() with that
one group only. The label (proxy_error) therefore reaches the allocator only for groups it chose.

Metric: admitted = 1 - caught / total_errors, caught = mean over allocator seeds. D = admitted(threshold) - admitted(arm),
so a positive D means the arm admits fewer corrupted rewards than the threshold.

Bootstraps at boot_budget (25%), each resample recomputes threshold and learned_class against its own error total:
  group    200 resamples of sim groups with replacement, BOOT_SEED 0 (same draws as E03, so D is paired with E03).
  cluster  200 resamples of the cell100 buckets with replacement, seed 123. A repeated bucket is relabeled as a new
           cluster. Rows keep pool order within a resample.
sklearn ConvergenceWarnings are counted in the point runs and in each bootstrap.
"""
import os

os.environ.setdefault("OMP_NUM_THREADS", "1")
import sys
import json
import time
import argparse
import warnings
from concurrent.futures import ProcessPoolExecutor

import numpy as np
import pandas as pd
from sklearn.exceptions import ConvergenceWarning

sys.path.insert(0, "/homes/rdubash/avrl_live/CASCADE/avrl")
from allocators import LearnedAllocator  # shipped class, imported read-only

MIN, K = 3, 1
LABEL = "proxy_error"
CHEAP = ["agreement", "n_distinct", "mean_tokens", "std_tokens", "any_truncated", "log_qlen"]
EMB = [f"pc{i}" for i in range(16)]
FEATS = {"cheap": CHEAP, "cheap+emb": CHEAP + EMB}  # cheap+emb = 22 columns (the class docstring says 21; see RESEARCH_LOG)
E03_JSON = "/homes/rdubash/avrl_live/CASCADE/avrl/runs/E03/result.json"
BOOT_SEED = 0  # group bootstrap rng, same as E03
CLUSTER_SEED = 123  # bucket-cluster bootstrap rng
ARMS = ["threshold", "threshold_input_order", "learned_class"]

_BASE, _G, _COLS, _P = None, None, None, None  # set in each worker by _init_worker


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


def cluster_pool(P, draws):
    """Bucket-cluster resample: draws holds bucket ids with repeats. Each draw keeps all groups of its bucket, and draw k
    gets cell code k, so a repeated bucket is a new cluster. Rows are returned in the source pool order."""
    rows, codes = [], []
    for k, b in enumerate(draws):
        m = P["members"][b]
        rows.append(m)
        codes.append(np.full(len(m), k))
    rows, codes = np.concatenate(rows), np.concatenate(codes)
    o = np.argsort(rows, kind="stable")
    rows, codes = rows[o], codes[o]
    return {"X": P["X"][rows], "y": P["y"][rows], "codes": codes, "n_b": len(draws), "members": None,
            "ag": P["ag"][rows], "nd": P["nd"][rows]}


def threshold_order(ag, nd, rng=None):
    """ThresholdAllocator._order key: agreement ascending, then n_distinct descending.
    Exact ties keep input order (rng None) or get a seeded random order (rng given)."""
    tb = rng.random(len(ag)) if rng is not None else np.arange(len(ag))
    return np.lexsort((tb, -nd, ag))  # lexsort: last key is primary, sort is stable


def threshold_caught(P, budget, seeds, input_order=False):
    y = P["y"]
    if input_order:
        return np.array([y[threshold_order(P["ag"], P["nd"])[:budget]].sum()], float)
    return np.array([y[threshold_order(P["ag"], P["nd"], np.random.default_rng(s))[:budget]].sum() for s in seeds], float)


def learned_caught(P, budget, seeds):
    """Shipped LearnedAllocator, replayed as the phase-1 loop calls it: K = 1 visit per group, candidates = all
    still-unverified pool groups, update() receives only the group just verified.
    Returns (caught per seed, sklearn ConvergenceWarnings raised during these runs)."""
    X, y, codes, ag, nd = P["X"], P["y"], P["codes"], P["ag"], P["nd"]
    records = [{"idx": i, "feat_vec": X[i], "cell": int(codes[i]), "agreement": ag[i], "n_distinct": nd[i]}
               for i in range(len(y))]
    caught, n_conv = [], 0
    for s in seeds:
        alloc = LearnedAllocator()  # class defaults: refit_every=20, prior_strength=2.0
        rng = np.random.default_rng(s)
        unverified, hits = list(range(len(y))), 0
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            for _ in range(budget):
                (i,) = alloc.select([records[j] for j in unverified], K, rng)
                unverified.remove(i)
                lab = int(y[i])
                hits += lab
                alloc.update([(X[i], lab, int(codes[i]))])
        n_conv += sum(issubclass(x.category, ConvergenceWarning) for x in w)
        caught.append(float(hits))
    return np.array(caught, float), n_conv


def boot_eval(P, budget, seeds):
    tot = int(P["y"].sum())
    thr = float(threshold_caught(P, budget, seeds).mean())
    lc, n_conv = learned_caught(P, budget, seeds)
    return tot, thr, float(lc.mean()), n_conv


def _init_worker(base, G, cols, P_full):
    global _BASE, _G, _COLS, _P
    _BASE, _G, _COLS, _P = base, G, cols, P_full


def _group_task(task):
    idx, budget, seeds = task
    return boot_eval(make_pool(_BASE.iloc[idx], _G, _COLS), budget, seeds)


def _cluster_task(task):
    draws, budget, seeds = task
    return boot_eval(cluster_pool(_P, draws), budget, seeds)


def pct_ci(x):
    x = np.asarray(x, float)
    return [float(np.percentile(x, 2.5)), float(np.percentile(x, 97.5))]


def summarize(res):
    tot = np.array([r[0] for r in res], float)
    thr = np.array([r[1] for r in res], float)
    lc = np.array([r[2] for r in res], float)
    D = (lc - thr) / tot  # admitted(threshold) - admitted(learned_class) = (caught_learned - caught_threshold) / total
    assert np.all(np.isfinite(D)) and len(D) == len(res)
    return {"ci95": pct_ci(D), "boot_mean": float(D.mean()), "frac_resamples_D_gt_0": float((D > 0).mean()),
            "n_resamples": int(len(D)), "conv_warnings": int(sum(r[3] for r in res)), "draws": D.tolist()}


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
    ap.add_argument("--limit", type=int, default=0, help="smoke test only: keep the first N sim groups (0 = full pool)")
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    G, cols = args.cell, FEATS[args.features]
    budgets = [float(b) for b in args.budgets.split(",")]
    seeds = list(range(args.seeds))
    os.makedirs(args.out, exist_ok=True)
    t0 = time.time()

    # Same pool as E03: keep buckets with >= MIN groups, seed-0 shuffle of bucket values, first quarter = fit split (dropped).
    df = pd.read_parquet(args.feats)
    cnt = df[G].value_counts(); df = df[df[G].isin(cnt[cnt >= MIN].index)].reset_index(drop=True)
    rng0 = np.random.default_rng(0); groups = df[G].unique(); rng0.shuffle(groups)
    fit_g = set(groups[: len(groups) // 4]); sim = df[~df[G].isin(fit_g)].reset_index(drop=True)
    if args.limit:
        sim = sim.iloc[: args.limit].reset_index(drop=True)
    P = make_pool(sim, G, cols)
    n_sim, total = len(sim), int(P["y"].sum())
    print(f"feats {os.path.basename(args.feats)} cell {G} features {args.features} ({len(cols)} cols) | sim groups {n_sim} "
          f"buckets {P['n_b']} base {P['y'].mean():.4f} total_errors {total}", flush=True)

    point = {}
    conv_point = 0
    for bf in budgets:
        budget = int(bf * n_sim)
        point[str(bf)] = {}
        for arm in ARMS:
            if arm == "learned_class":
                h, conv = learned_caught(P, budget, seeds)
            else:
                h, conv = threshold_caught(P, budget, seeds, input_order=(arm == "threshold_input_order")), 0
            conv_point += conv
            point[str(bf)][arm] = {"budget": budget, "caught_mean": float(h.mean()), "caught_sd": float(h.std()),
                                   "caught_per_seed": h.tolist(), "admitted": 1 - float(h.mean()) / total,
                                   "n_runs": len(h), "conv_warnings": int(conv)}
    D_point = {str(bf): {arm: point[str(bf)]["threshold"]["admitted"] - point[str(bf)][arm]["admitted"]
                         for arm in ["threshold_input_order", "learned_class"]} for bf in budgets}

    print("\n| budget | budget_n | " + " | ".join(ARMS) + " |\n|---|---|" + "---|" * len(ARMS), flush=True)
    for bf in budgets:
        r = point[str(bf)]
        print(f"| {bf:.0%} | {r['threshold']['budget']} | " + " | ".join(
            f"{r[a]['caught_mean']:.2f}+-{r[a]['caught_sd']:.2f} (adm {r[a]['admitted']:.3f})" for a in ARMS) + " |", flush=True)
    print("\nD = admitted(threshold) - admitted(arm), point estimates:", flush=True)
    for bf in budgets:
        print(f"  {bf:.0%}: D(learned_class) {D_point[str(bf)]['learned_class']:+.4f}   "
              f"D(threshold_input_order) {D_point[str(bf)]['threshold_input_order']:+.4f}", flush=True)
    print(f"sklearn ConvergenceWarnings in point runs: {conv_point}", flush=True)

    # Reproduction: the threshold arms must match E03's stored point values (same pool, same tie-breaks).
    repro = None
    if os.path.exists(E03_JSON) and args.limit == 0:
        ref = json.load(open(E03_JSON))
        diffs = [abs(ref["point"][str(bf)][arm]["caught_mean"] - point[str(bf)][arm]["caught_mean"])
                 for bf in budgets for arm in ["threshold", "threshold_input_order"]]
        repro = {"ref": E03_JSON, "max_abs_diff_caught_mean": float(max(diffs)), "matches_E03": bool(max(diffs) < 1e-9),
                 "pool_matches_E03": bool(ref["pool"]["n_sim"] == n_sim and ref["pool"]["total_errors"] == total)}
        print(f"\nreproduction vs E03 json: max abs diff {max(diffs):.3e}, matches {repro['matches_E03']}, "
              f"pool matches {repro['pool_matches_E03']}", flush=True)

    # Bootstraps at boot_budget.
    bkey = str(args.boot_budget)
    assert bkey in point, "boot_budget must be one of --budgets"
    boot_budget = int(args.boot_budget * n_sim)
    base = sim[list(dict.fromkeys([G, LABEL, "agreement", "n_distinct"] + cols))].reset_index(drop=True)
    rng_b = np.random.default_rng(BOOT_SEED)
    group_tasks = [(np.sort(rng_b.integers(0, n_sim, size=n_sim)), boot_budget, seeds) for _ in range(args.boot)]
    rng_c = np.random.default_rng(CLUSTER_SEED)
    cluster_tasks = [(rng_c.integers(0, P["n_b"], size=P["n_b"]), boot_budget, seeds) for _ in range(args.boot)]
    t1 = time.time()
    with ProcessPoolExecutor(max_workers=args.workers, initializer=_init_worker, initargs=(base, G, cols, P)) as ex:
        res_g = list(ex.map(_group_task, group_tasks, chunksize=4))
        res_c = list(ex.map(_cluster_task, cluster_tasks, chunksize=4))
    boot_secs = time.time() - t1
    Dg, Dc = summarize(res_g), summarize(res_c)
    print(f"\nbootstrap {args.boot} resamples each at {args.boot_budget:.0%} ({boot_budget} docs), {args.workers} workers, {boot_secs:.1f}s", flush=True)
    for name, Db in [("group", Dg), ("cluster", Dc)]:
        print(f"  {name}: D(learned_class) point {D_point[bkey]['learned_class']:+.4f}  boot mean {Db['boot_mean']:+.4f}  "
              f"95% CI [{Db['ci95'][0]:+.4f}, {Db['ci95'][1]:+.4f}]  frac>0 {Db['frac_resamples_D_gt_0']:.3f}  "
              f"ConvergenceWarnings {Db['conv_warnings']}", flush=True)

    # Verdict (pre-registered): PASS iff point D > 0 and the lower 95% bound is > 0 under BOTH bootstraps.
    d_pt = D_point[bkey]["learned_class"]
    reasons = []
    if d_pt <= 0:
        reasons.append("point D(learned_class) <= 0 at 25%")
    for name, Db in [("group", Dg), ("cluster", Dc)]:
        lo, hi = Db["ci95"]
        if lo <= 0:
            reasons.append(f"{name} 95% CI includes 0" if hi >= 0 else f"{name} 95% CI lies entirely below 0")
    verdict = "PASS" if not reasons else "KILL"
    reason = "point D > 0 and both 95% CIs exclude 0 on the positive side" if not reasons else "; ".join(reasons)
    p = point[bkey]
    summary = (f"L5a {G} {args.features} 25%: admitted threshold {p['threshold']['admitted']:.3f} "
               f"({p['threshold']['caught_mean']:.1f} caught), learned_class {p['learned_class']['admitted']:.3f} "
               f"({p['learned_class']['caught_mean']:.1f} caught); D(learned_class) {d_pt:+.3f} "
               f"group CI [{Dg['ci95'][0]:+.3f}, {Dg['ci95'][1]:+.3f}] cluster CI [{Dc['ci95'][0]:+.3f}, {Dc['ci95'][1]:+.3f}]; "
               f"verdict {verdict}")
    print("\n" + summary, flush=True)

    out = {
        "exp": "L5a",
        "verdict": verdict,
        "verdict_reason": reason,
        "summary": summary,
        "config": {"feats": args.feats, "cell": G, "features": args.features, "n_features": len(cols), "feature_list": cols,
                   "min_bucket_size": MIN, "K": K, "seeds": args.seeds, "budgets": budgets,
                   "fit_split": "seed-0 shuffle of bucket values, first quarter = fit, dropped (no static arm)",
                   "learned_class": "avrl/allocators.py LearnedAllocator() class defaults, candidates = all unverified groups per visit",
                   "threshold_tiebreak": "threshold: seeded random among exact ties; threshold_input_order: pool order",
                   "boot_group": {"resamples": args.boot, "seed": BOOT_SEED, "unit": "sim groups with replacement, pool order kept"},
                   "boot_cluster": {"resamples": args.boot, "seed": CLUSTER_SEED, "unit": "cell100 buckets with replacement, duplicates relabeled"},
                   "boot_budget_frac": args.boot_budget, "boot_budget_docs": boot_budget, "limit": args.limit},
        "pool": {"n_sim": n_sim, "n_buckets": P["n_b"], "total_errors": total,
                 "budget_docs": {str(bf): int(bf * n_sim) for bf in budgets}},
        "point": point,
        "D_point": D_point,
        "D_bootstrap_group": Dg,
        "D_bootstrap_cluster": Dc,
        "conv_warnings_point": int(conv_point),
        "reproduction_vs_E03": repro,
        "wall_seconds": round(time.time() - t0, 1),
        "bootstrap_wall_seconds": round(boot_secs, 1),
    }
    with open(os.path.join(args.out, "result.json"), "w") as f:
        json.dump(out, f, indent=1)
    print(f"wrote {os.path.join(args.out, 'result.json')}", flush=True)


if __name__ == "__main__":
    main()
