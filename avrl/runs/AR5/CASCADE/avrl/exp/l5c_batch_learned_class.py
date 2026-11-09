"""L5c: the shipped LearnedAllocator vs the label-free agreement threshold in phase-1 batch shape (CPU only).

Copied from avrl/exp/l5a_learned_class.py. L5a replayed the class over the whole unverified pool at K = 1. L5c replays it the
way the phase-1 loop calls it: the pool is cut into consecutive batches of 16 groups (the measured phase-1 step), and each
batch offers only its own groups as candidates. E03, L5a, allocators.py and the p0 files are not edited. allocators.py is
imported read-only.

Batches, per allocator seed s: perm = np.random.default_rng(s).permutation(n), then consecutive chunks of 16 (the last chunk
has n % 16 groups). A group not verified in its batch is accepted on the proxy and leaves the pool.

Budget: batch i verifies b_i = floor(f*N_i + 1/2) - floor(f*N_(i-1) + 1/2), where N_i is the number of groups consumed through
batch i. Exact rational arithmetic, so the half-up rounding is exact. Totals are floor(f*n + 1/2): 218 / 545 / 1,090.

Arms, on identical batches per seed:
  threshold_batch  the b_i groups ranked highest by threshold_order (agreement ascending, n_distinct descending, exact ties in
                   seeded random order with rng default_rng(s), one rng per seed run).
  learned_batch    LearnedAllocator() with class defaults (refit_every 20, prior_strength 2.0). Per batch: records for the
                   batch's groups only, select(records, b_i, rng) with rng default_rng(s), then update() with the b_i verified
                   (X, label, cell) tuples and nothing else.

Metric: admitted = 1 - caught / total_errors, caught = mean over allocator seeds 0..7. D = admitted(threshold_batch) -
admitted(learned_batch) = (caught_learned - caught_threshold) / total_errors. Positive D means the learned class admits fewer
corrupted rewards.

Bootstraps at boot_budget (25%) only: group (200 resamples, BOOT_SEED 0, same draws as E03 and L5a) and bucket-cluster (200
resamples of the 75 buckets with replacement, seed 123, repeated buckets relabeled as new clusters). Each resample rebuilds the
pool, its batches (permutation seed 0) and both arms (allocator seed 0), and recomputes D against its own error total.

sklearn ConvergenceWarnings are counted in the point runs and in each bootstrap.
"""
import os

os.environ.setdefault("OMP_NUM_THREADS", "1")
import sys
import json
import time
import argparse
import warnings
from fractions import Fraction
from math import floor
from concurrent.futures import ProcessPoolExecutor

import numpy as np
import pandas as pd
from sklearn.exceptions import ConvergenceWarning

sys.path.insert(0, "/homes/rdubash/avrl_live/CASCADE/avrl")
from allocators import LearnedAllocator  # shipped class, imported read-only

MIN = 3
G = "cell100"
LABEL = "proxy_error"
CHEAP = ["agreement", "n_distinct", "mean_tokens", "std_tokens", "any_truncated", "log_qlen"]
EMB = [f"pc{i}" for i in range(16)]
COLS = CHEAP + EMB  # cheap+emb, 22 columns, as in L5a
BOOT_SEED = 0  # group bootstrap rng, same as E03 and L5a
CLUSTER_SEED = 123  # bucket-cluster bootstrap rng, same as L5a
BOOT_ALLOC_SEED = 0  # allocator seed used inside every bootstrap resample
ARMS = ["threshold_batch", "learned_batch"]

_BASE, _G, _COLS, _P = None, None, None, None  # set in each worker by _init_worker
HALF = Fraction(1, 2)


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


def batch_schedule(n, f, batch):
    """[(start, end, b_i)] for consecutive batches of `batch` groups over n groups.
    b_i = floor(f*N_i + 1/2) - floor(f*N_(i-1) + 1/2), N_i = end of batch i (exact rational arithmetic)."""
    fr = Fraction(str(f))
    out, prev, start = [], 0, 0
    while start < n:
        end = min(start + batch, n)
        cum = floor(fr * end + HALF)
        out.append((start, end, cum - prev))
        prev, start = cum, end
    assert prev == floor(fr * n + HALF), "schedule total is not floor(f*n + 1/2)"
    return out


def batch_plan(n, f, batch, seed):
    """Batches for allocator seed `seed`: list of (row indices into the pool, b_i)."""
    perm = np.random.default_rng(seed).permutation(n)
    return [(perm[s:e], b) for s, e, b in batch_schedule(n, f, batch)]


def threshold_batch_caught(P, f, seeds, batch):
    """Returns (caught per seed, verified count per seed)."""
    y, ag, nd = P["y"], P["ag"], P["nd"]
    caught, verified = [], []
    for s in seeds:
        rng = np.random.default_rng(s)
        hits = n_ver = 0
        for rows, b in batch_plan(len(y), f, batch, s):
            chosen = rows[threshold_order(ag[rows], nd[rows], rng)[:b]]
            assert len(chosen) == b, "threshold_batch chose the wrong count"
            hits += int(y[chosen].sum())
            n_ver += len(chosen)
        caught.append(float(hits))
        verified.append(n_ver)
    return np.array(caught, float), verified


def learned_batch_caught(P, f, seeds, batch):
    """Shipped LearnedAllocator, replayed batch by batch as the phase-1 loop calls it: records for the batch's groups only,
    select(records, b_i, rng), update() with the verified tuples only. Returns (caught per seed, verified count per seed,
    sklearn ConvergenceWarnings raised during these runs)."""
    X, y, codes, ag, nd = P["X"], P["y"], P["codes"], P["ag"], P["nd"]
    caught, verified, n_conv = [], [], 0
    for s in seeds:
        alloc = LearnedAllocator()  # class defaults: refit_every=20, prior_strength=2.0
        rng = np.random.default_rng(s)
        hits = n_ver = 0
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            for rows, b in batch_plan(len(y), f, batch, s):
                records = [{"idx": int(i), "feat_vec": X[i], "cell": int(codes[i]), "agreement": ag[i],
                            "n_distinct": nd[i]} for i in rows]
                chosen = np.asarray(alloc.select(records, b, rng), dtype=int)
                assert len(chosen) == b, "learned_batch chose the wrong count"
                hits += int(y[chosen].sum())
                n_ver += len(chosen)
                alloc.update([(X[i], int(y[i]), int(codes[i])) for i in chosen])
        n_conv += sum(issubclass(x.category, ConvergenceWarning) for x in w)
        caught.append(float(hits))
        verified.append(n_ver)
    return np.array(caught, float), verified, n_conv


def boot_eval(P, f, batch, seeds):
    tot = int(P["y"].sum())
    thr, thr_v = threshold_batch_caught(P, f, seeds, batch)
    lc, lc_v, n_conv = learned_batch_caught(P, f, seeds, batch)
    exp_total = floor(Fraction(str(f)) * len(P["y"]) + HALF)
    assert all(v == exp_total for v in thr_v + lc_v), "budget not exact in a bootstrap resample"
    return tot, float(thr.mean()), float(lc.mean()), n_conv


def _init_worker(base, G, cols, P_full):
    global _BASE, _G, _COLS, _P
    _BASE, _G, _COLS, _P = base, G, cols, P_full


def _group_task(task):
    idx, f, seeds, batch = task
    return boot_eval(make_pool(_BASE.iloc[idx], _G, _COLS), f, batch, seeds)


def _cluster_task(task):
    draws, f, seeds, batch = task
    return boot_eval(cluster_pool(_P, draws), f, batch, seeds)


def pct_ci(x):
    x = np.asarray(x, float)
    return [float(np.percentile(x, 2.5)), float(np.percentile(x, 97.5))]


def summarize(res):
    tot = np.array([r[0] for r in res], float)
    thr = np.array([r[1] for r in res], float)
    lc = np.array([r[2] for r in res], float)
    D = (lc - thr) / tot  # admitted(threshold) - admitted(learned) = (caught_learned - caught_threshold) / total
    assert np.all(np.isfinite(D)) and len(D) == len(res)
    return {"ci95": pct_ci(D), "boot_mean": float(D.mean()), "frac_resamples_D_gt_0": float((D > 0).mean()),
            "n_resamples": int(len(D)), "conv_warnings": int(sum(r[3] for r in res)), "draws": D.tolist()}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--feats", required=True)
    ap.add_argument("--batch", type=int, default=16)
    ap.add_argument("--budgets", default="0.10,0.25,0.50")
    ap.add_argument("--seeds", type=int, default=8)
    ap.add_argument("--boot", type=int, default=200)
    ap.add_argument("--boot_budget", type=float, default=0.25)
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--limit", type=int, default=0, help="smoke test only: keep the first N sim groups (0 = full pool)")
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    budgets = [float(b) for b in args.budgets.split(",")]
    seeds = list(range(args.seeds))
    os.makedirs(args.out, exist_ok=True)
    t0 = time.time()

    # Same pool as E03 and L5a: keep buckets with >= MIN groups, seed-0 shuffle of bucket values, first quarter = fit split (dropped).
    df = pd.read_parquet(args.feats)
    cnt = df[G].value_counts(); df = df[df[G].isin(cnt[cnt >= MIN].index)].reset_index(drop=True)
    rng0 = np.random.default_rng(0); groups = df[G].unique(); rng0.shuffle(groups)
    fit_g = set(groups[: len(groups) // 4]); sim = df[~df[G].isin(fit_g)].reset_index(drop=True)
    if args.limit:
        sim = sim.iloc[: args.limit].reset_index(drop=True)
    P = make_pool(sim, G, COLS)
    n_sim, total = len(sim), int(P["y"].sum())
    print(f"feats {os.path.basename(args.feats)} cell {G} cheap+emb ({len(COLS)} cols) batch {args.batch} | sim groups {n_sim} "
          f"buckets {P['n_b']} base {P['y'].mean():.4f} total_errors {total}", flush=True)

    point, conv_point = {}, 0
    for bf in budgets:
        exp_total = floor(Fraction(str(bf)) * n_sim + HALF)
        thr, thr_v = threshold_batch_caught(P, bf, seeds, args.batch)
        lc, lc_v, conv = learned_batch_caught(P, bf, seeds, args.batch)
        conv_point += conv
        n_batches = len(batch_schedule(n_sim, bf, args.batch))
        point[str(bf)] = {}
        for arm, h, ver in [("threshold_batch", thr, thr_v), ("learned_batch", lc, lc_v)]:
            point[str(bf)][arm] = {"budget": exp_total, "n_batches": n_batches, "caught_mean": float(h.mean()),
                                   "caught_sd": float(h.std()), "caught_per_seed": h.tolist(), "verified_per_seed": ver,
                                   "budget_exact": all(v == exp_total for v in ver),
                                   "admitted": 1 - float(h.mean()) / total, "n_runs": len(h),
                                   "conv_warnings": int(conv) if arm == "learned_batch" else 0}
    D_point = {str(bf): point[str(bf)]["threshold_batch"]["admitted"] - point[str(bf)]["learned_batch"]["admitted"]
               for bf in budgets}

    print("\n| budget | budget_n | " + " | ".join(ARMS) + " |\n|---|---|" + "---|" * len(ARMS), flush=True)
    for bf in budgets:
        r = point[str(bf)]
        print(f"| {bf:.0%} | {r['threshold_batch']['budget']} | " + " | ".join(
            f"{r[a]['caught_mean']:.2f}+-{r[a]['caught_sd']:.2f} (adm {r[a]['admitted']:.3f})" for a in ARMS) + " |", flush=True)
    print("\nD = admitted(threshold_batch) - admitted(learned_batch), point estimates:", flush=True)
    for bf in budgets:
        print(f"  {bf:.0%}: D(learned_batch) {D_point[str(bf)]:+.4f}", flush=True)
    print(f"sklearn ConvergenceWarnings in point runs: {conv_point}", flush=True)

    # Bootstraps at boot_budget. Each resample: its own pool, batches (seed 0) and both arms (seed 0).
    bkey = str(args.boot_budget)
    assert bkey in point, "boot_budget must be one of --budgets"
    boot_budget = floor(Fraction(str(args.boot_budget)) * n_sim + HALF)
    base = sim[list(dict.fromkeys([G, LABEL, "agreement", "n_distinct"] + COLS))].reset_index(drop=True)
    rng_b = np.random.default_rng(BOOT_SEED)
    group_tasks = [(np.sort(rng_b.integers(0, n_sim, size=n_sim)), args.boot_budget, [BOOT_ALLOC_SEED], args.batch)
                   for _ in range(args.boot)]
    rng_c = np.random.default_rng(CLUSTER_SEED)
    cluster_tasks = [(rng_c.integers(0, P["n_b"], size=P["n_b"]), args.boot_budget, [BOOT_ALLOC_SEED], args.batch)
                     for _ in range(args.boot)]
    t1 = time.time()
    with ProcessPoolExecutor(max_workers=args.workers, initializer=_init_worker, initargs=(base, G, COLS, P)) as ex:
        res_g = list(ex.map(_group_task, group_tasks, chunksize=4))
        res_c = list(ex.map(_cluster_task, cluster_tasks, chunksize=4))
    boot_secs = time.time() - t1
    Dg, Dc = summarize(res_g), summarize(res_c)
    print(f"\nbootstrap {args.boot} resamples each at {args.boot_budget:.0%} ({boot_budget} docs), batch {args.batch}, "
          f"allocator seed {BOOT_ALLOC_SEED}, {args.workers} workers, {boot_secs:.1f}s", flush=True)
    for name, Db in [("group", Dg), ("cluster", Dc)]:
        print(f"  {name}: D(learned_batch) point {D_point[bkey]:+.4f}  boot mean {Db['boot_mean']:+.4f}  "
              f"95% CI [{Db['ci95'][0]:+.4f}, {Db['ci95'][1]:+.4f}]  frac>0 {Db['frac_resamples_D_gt_0']:.3f}  "
              f"ConvergenceWarnings {Db['conv_warnings']}", flush=True)

    # Verdict (pre-registered): PASS iff point D > 0 and the lower 95% bound is > 0 under BOTH bootstraps.
    d_pt = D_point[bkey]
    reasons = []
    if d_pt <= 0:
        reasons.append("point D(learned_batch) <= 0 at 25%")
    for name, Db in [("group", Dg), ("cluster", Dc)]:
        lo, hi = Db["ci95"]
        if lo <= 0:
            reasons.append(f"{name} 95% CI includes 0" if hi >= 0 else f"{name} 95% CI lies entirely below 0")
    verdict = "PASS" if not reasons else "KILL"
    reason = "point D > 0 and both 95% CIs exclude 0 on the positive side" if not reasons else "; ".join(reasons)
    p = point[bkey]
    summary = (f"L5c {G} cheap+emb batch {args.batch} 25%: admitted threshold_batch {p['threshold_batch']['admitted']:.3f} "
               f"({p['threshold_batch']['caught_mean']:.1f} caught), learned_batch {p['learned_batch']['admitted']:.3f} "
               f"({p['learned_batch']['caught_mean']:.1f} caught); D(learned_batch) {d_pt:+.3f} "
               f"group CI [{Dg['ci95'][0]:+.3f}, {Dg['ci95'][1]:+.3f}] cluster CI [{Dc['ci95'][0]:+.3f}, {Dc['ci95'][1]:+.3f}]; "
               f"verdict {verdict}")
    print("\n" + summary, flush=True)

    all_budget_exact = all(point[str(bf)][a]["budget_exact"] for bf in budgets for a in ARMS)
    all_finite = all(np.isfinite(point[str(bf)][a]["admitted"]) and np.isfinite(point[str(bf)][a]["caught_mean"])
                     for bf in budgets for a in ARMS) and all(np.isfinite(D_point[str(bf)]) for bf in budgets)
    out = {
        "exp": "L5c",
        "verdict": verdict,
        "verdict_reason": reason,
        "summary": summary,
        "config": {"feats": args.feats, "cell": G, "features": "cheap+emb", "n_features": len(COLS), "feature_list": COLS,
                   "min_bucket_size": MIN, "batch": args.batch, "seeds": args.seeds, "budgets": budgets,
                   "batch_rule": "permutation default_rng(seed) of the sim groups, consecutive chunks of `batch`, last chunk short",
                   "budget_rule": "b_i = floor(f*N_i + 1/2) - floor(f*N_(i-1) + 1/2), N_i = groups consumed through batch i, exact",
                   "fit_split": "seed-0 shuffle of bucket values, first quarter = fit, dropped (no static arm)",
                   "learned_batch": "avrl/allocators.py LearnedAllocator() class defaults, records for the batch only, select(b_i), "
                                    "update() with the b_i verified tuples",
                   "threshold_batch": "top b_i by threshold_order, exact ties seeded random with default_rng(seed) per seed run",
                   "boot_group": {"resamples": args.boot, "seed": BOOT_SEED, "unit": "sim groups with replacement, pool order kept",
                                  "allocator_seed": BOOT_ALLOC_SEED},
                   "boot_cluster": {"resamples": args.boot, "seed": CLUSTER_SEED,
                                    "unit": "cell100 buckets with replacement, duplicates relabeled",
                                    "allocator_seed": BOOT_ALLOC_SEED},
                   "boot_budget_frac": args.boot_budget, "boot_budget_docs": boot_budget, "limit": args.limit},
        "pool": {"n_sim": n_sim, "n_buckets": P["n_b"], "total_errors": total,
                 "budget_docs": {str(bf): floor(Fraction(str(bf)) * n_sim + HALF) for bf in budgets}},
        "point": point,
        "D_point": D_point,
        "D_bootstrap_group": Dg,
        "D_bootstrap_cluster": Dc,
        "checks": {"budget_exact_all_arms_seeds": all_budget_exact, "all_values_finite": bool(all_finite),
                   "conv_warnings_point": int(conv_point),
                   "conv_warnings_boot": {"group": Dg["conv_warnings"], "cluster": Dc["conv_warnings"]}},
        "wall_seconds": round(time.time() - t0, 1),
        "bootstrap_wall_seconds": round(boot_secs, 1),
    }
    with open(os.path.join(args.out, "result.json"), "w") as f:
        json.dump(out, f, indent=1)
    print(f"wrote {os.path.join(args.out, 'result.json')}", flush=True)


if __name__ == "__main__":
    main()
