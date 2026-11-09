#!/usr/bin/env python3
"""E03: phase-1-style per-step replay of the real allocator classes in avrl/allocators.py.

Pool: runs/p0/feats.parquet (E01 pool: 3000 GSM8K groups, 8 rollouts each), read only.
Arms: random, threshold (ThresholdAllocator), learned (LearnedAllocator), greedy (GreedyOnlineAllocator,
diagnostic: same online logistic regression, ranked without Thompson sampling).

Per (arm, budget, seed): permute the groups with default_rng(1000+seed), walk them in steps of STEP=16
groups, k = alloc.n_verify(len(step_groups), budget, step=t), select(records, k, rng) on cheap fields only,
then update() with (feat_vec, proxy_error, cell) for the verified groups only. Gold (proxy_error) is used
for scoring and as the update label of verified groups, never as a selection input.

CPU only. No GPU, no pip, no ssh. Writes result.json into --out. Log goes to stdout (the runner redirects it).
"""
import argparse
import json
import math
import os
import sys
import time
import warnings

sys.dont_write_bytecode = True
AVRL_DIR = "/homes/rdubash/avrl_live/CHEAP/avrl"
sys.path.insert(0, AVRL_DIR)

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from sklearn.exceptions import ConvergenceWarning  # noqa: E402

import allocators as A  # noqa: E402

STEP = 16
PC_COLS = [f"pc{i}" for i in range(16)]
FEAT_COLS = ["agreement", "n_distinct", "mean_tokens", "std_tokens", "any_truncated", "log_qlen"] + PC_COLS
ARMS = {
    "random": A.RandomAllocator,
    "threshold": A.ThresholdAllocator,
    "learned": A.LearnedAllocator,
    "greedy": A.GreedyOnlineAllocator,
}
BOOT_ARMS = ["threshold", "learned", "greedy"]


def load_pool(path):
    df = pd.read_parquet(path)
    assert len(df) == 3000, f"expected 3000 groups, got {len(df)}"
    return {
        "X": df[FEAT_COLS].to_numpy(dtype=np.float64),
        "agreement": df["agreement"].to_numpy(dtype=np.float64),
        "n_distinct": df["n_distinct"].to_numpy(dtype=np.int64),
        "cell": df["cell100"].to_numpy(dtype=np.int64),
        "y": df["proxy_error"].to_numpy(dtype=np.int64),
    }


def subset(pool, ids):
    return {k: v[ids] for k, v in pool.items()}


def run_arm(pool, arm, budget, seed):
    n = len(pool["y"])
    perm = np.random.default_rng(1000 + seed).permutation(n)
    alloc = ARMS[arm]()
    sel_rng = np.random.default_rng(2000 + seed)
    caught = 0
    calls = 0
    for t in range(math.ceil(n / STEP)):
        chunk = perm[t * STEP:(t + 1) * STEP]
        k = alloc.n_verify(len(chunk), budget, step=t)
        # cheap, gold-free fields only
        records = [
            {
                "idx": j,
                "feat_vec": pool["X"][g],
                "cell": int(pool["cell"][g]),
                "agreement": float(pool["agreement"][g]),
                "n_distinct": int(pool["n_distinct"][g]),
            }
            for j, g in enumerate(chunk)
        ]
        chosen = alloc.select(records, k, sel_rng)
        assert len(chosen) == k and len(set(chosen)) == k, (arm, t, k, chosen)
        verified = []
        for j in chosen:
            g = int(chunk[j])
            verified.append((pool["X"][g], int(pool["y"][g]), int(pool["cell"][g])))
            caught += int(pool["y"][g])
        alloc.update(verified)
        calls += k
    errors = int(pool["y"].sum())
    return {"arm": arm, "budget": budget, "seed": seed, "errors": errors, "caught": caught,
            "admitted": (errors - caught) / errors, "calls": calls}


def run_budget(pool, budget, seeds):
    out = {arm: [run_arm(pool, arm, budget, s) for s in seeds] for arm in ARMS}
    ref = [r["calls"] for r in out["random"]]
    for arm in ARMS:
        got = [r["calls"] for r in out[arm]]
        assert got == ref, f"verifier-call mismatch at budget {budget}: {arm} {got} vs {ref}"
    return out


def ci95(x):
    x = np.asarray(x, dtype=np.float64)
    lo, hi = np.percentile(x, [2.5, 97.5])
    return {"mean": float(x.mean()), "ci95": [float(lo), float(hi)],
            "frac_below_zero": float((x < 0).mean()), "n_resamples": int(len(x))}


def bootstrap(pool, budget, seeds, n_boot, log):
    n = len(pool["y"])
    diff_learned, diff_greedy = [], []
    for b in range(n_boot):
        ids = np.random.default_rng(10000 + b).choice(n, size=n, replace=True)
        sub = subset(pool, ids)
        adm = {arm: [] for arm in BOOT_ARMS}
        for s in seeds:
            rs = {arm: run_arm(sub, arm, budget, s) for arm in BOOT_ARMS}
            assert len({r["calls"] for r in rs.values()}) == 1, "verifier-call mismatch in bootstrap"
            for arm in BOOT_ARMS:
                adm[arm].append(rs[arm]["admitted"])
        diff_learned.append(float(np.mean([l - t for l, t in zip(adm["learned"], adm["threshold"])])))
        diff_greedy.append(float(np.mean([g - t for g, t in zip(adm["greedy"], adm["threshold"])])))
        if (b + 1) % 10 == 0:
            log(f"bootstrap {b + 1}/{n_boot} learned-thr {diff_learned[-1]:+.4f} greedy-thr {diff_greedy[-1]:+.4f}")
    return diff_learned, diff_greedy


def arm_stats(rows):
    return {
        "caught_per_seed": [r["caught"] for r in rows],
        "caught_mean": float(np.mean([r["caught"] for r in rows])),
        "admitted_per_seed": [round(r["admitted"], 6) for r in rows],
        "admitted_mean": float(np.mean([r["admitted"] for r in rows])),
        "calls_per_seed": [r["calls"] for r in rows],
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pool", required=True)
    ap.add_argument("--budgets", type=float, nargs="+", default=[0.10, 0.25, 0.50])
    ap.add_argument("--seeds", type=int, default=8)
    ap.add_argument("--boot", type=int, default=100)
    ap.add_argument("--boot_budget", type=float, default=0.25)
    ap.add_argument("--boot_seeds", type=int, default=4)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    assert args.boot_budget in args.budgets, "boot_budget must be one of --budgets"
    os.makedirs(args.out, exist_ok=True)

    def log(msg):
        print(msg, flush=True)

    t0 = time.time()
    pool = load_pool(args.pool)
    n = len(pool["y"])
    n_err = int(pool["y"].sum())
    log(f"E03 pool {args.pool}: groups={n} proxy_errors={n_err} step={STEP} "
        f"steps={math.ceil(n / STEP)} (last step {n - STEP * (math.ceil(n / STEP) - 1)} groups)")
    log(f"feature columns ({len(FEAT_COLS)}): {FEAT_COLS}")
    log(f"allocator defaults: learned refit_every=20 prior_strength=2.0; cell=cell100")

    seeds = list(range(args.seeds))
    with warnings.catch_warnings(record=True) as wrec:
        warnings.simplefilter("always")

        main_rows = {}
        per_budget = {}
        for b in args.budgets:
            rows = run_budget(pool, b, seeds)
            main_rows[b] = rows
            calls = rows["random"][0]["calls"]
            log(f"\n== budget {b:.2f}: verifier calls per run = {calls} (identical across arms and seeds)")
            for arm in ARMS:
                st = arm_stats(rows[arm])
                log(f"  {arm:9s} caught per seed {st['caught_per_seed']} mean {st['caught_mean']:.1f} "
                    f"admitted mean {st['admitted_mean']:.4f}")
            thr = arm_stats(rows["threshold"])
            diff_l = [r["admitted"] - t["admitted"] for r, t in zip(rows["learned"], rows["threshold"])]
            diff_g = [r["admitted"] - t["admitted"] for r, t in zip(rows["greedy"], rows["threshold"])]
            per_budget[b] = {
                "verifier_calls_per_run": calls,
                "verifier_calls_equal_across_arms": True,
                "errors_per_seed": [r["errors"] for r in rows["threshold"]],
                "arms": {arm: arm_stats(rows[arm]) for arm in ARMS},
                "learned_minus_threshold_admitted_mean": float(np.mean(diff_l)),
                "greedy_minus_threshold_admitted_mean": float(np.mean(diff_g)),
                "greedy_minus_threshold_caught_mean": float(
                    np.mean([r["caught"] - t["caught"] for r, t in zip(rows["greedy"], rows["threshold"])])),
                "learned_minus_threshold_caught_mean": float(
                    np.mean([r["caught"] - t["caught"] for r, t in zip(rows["learned"], rows["threshold"])])),
            }
        n_conv_main = sum(issubclass(w.category, ConvergenceWarning) for w in wrec)
        log(f"\nConvergenceWarnings during main runs: {n_conv_main}")

        bs = args.boot_seeds
        point_rows = main_rows[args.boot_budget]
        pt_l = [r["admitted"] - t["admitted"] for r, t in
                zip(point_rows["learned"][:bs], point_rows["threshold"][:bs])]
        pt_g = [r["admitted"] - t["admitted"] for r, t in
                zip(point_rows["greedy"][:bs], point_rows["threshold"][:bs])]
        log(f"\npoint estimate at budget {args.boot_budget:.2f}, seeds 0-{bs - 1}: "
            f"learned-thr admitted {np.mean(pt_l):+.4f}, greedy-thr admitted {np.mean(pt_g):+.4f}")

        boot = None
        if args.boot > 0:
            log(f"\nbootstrap: B={args.boot} resamples of {n} groups with replacement "
                f"(default_rng(10000+b)), seeds 0-{bs - 1}, budget {args.boot_budget:.2f}")
            d_l, d_g = bootstrap(pool, args.boot_budget, list(range(bs)), args.boot, log)
            boot = {"B": args.boot, "seeds": bs, "budget": args.boot_budget,
                    "learned_minus_threshold_admitted": ci95(d_l),
                    "greedy_minus_threshold_admitted": ci95(d_g)}
        n_conv_total = sum(issubclass(w.category, ConvergenceWarning) for w in wrec)

    n_conv_boot = n_conv_total - n_conv_main
    log(f"ConvergenceWarnings during bootstrap: {n_conv_boot}; total {n_conv_total}")

    if boot is not None:
        hi = boot["learned_minus_threshold_admitted"]["ci95"][1]
        verdict = "PASS" if hi < 0 else "KILL"
        log(f"\nCI95 (learned - threshold) admitted at {args.boot_budget:.2f}: "
            f"{boot['learned_minus_threshold_admitted']['ci95']} -> {verdict}")
    else:
        verdict = None

    result = {
        "id": "E03",
        "pool": args.pool,
        "n_groups": n,
        "n_proxy_errors": n_err,
        "step_size_groups": STEP,
        "budgets": args.budgets,
        "seeds": args.seeds,
        "assumptions": [
            "steps of 16 groups over a seed-permuted pool (last step 8 groups); n_verify(step_groups, budget, step=t) with cumulative rounding",
            "feat_vec = [agreement, n_distinct, mean_tokens, std_tokens, any_truncated (stands in for frac_truncated), "
            "log_qlen (stands in for q_len)] + pc0..pc15 (22 dims)",
            "cell = cell100 (KMeans k=100 over question embeddings from p0_analyze.py)",
            "LearnedAllocator defaults refit_every=20, prior_strength=2.0; GreedyOnlineAllocator inherits them",
            "permutation rng default_rng(1000+seed); selection rng default_rng(2000+seed) shared across steps of one run",
            "LogisticRegression in allocators.py used unscaled with max_iter=1000 as written",
            "gold (proxy_error) used only for scoring and as the update label of verified groups",
        ],
        "per_budget": {f"{b:.2f}": v for b, v in per_budget.items()},
        "metric": {
            "name": "admitted rate at 25% budget, learned minus threshold, paired per bootstrap resample, "
                    "mean over seeds, percentile 95% CI over B resamples",
            "budget": args.boot_budget,
            "point_estimate_seeds_0_to_boot_seeds_minus_1": {
                "learned_minus_threshold_admitted": float(np.mean(pt_l)),
                "greedy_minus_threshold_admitted": float(np.mean(pt_g)),
            },
            "bootstrap": boot,
        },
        "verdict": verdict,
        "verdict_rule": "PASS iff the 95% CI upper bound of (learned admitted - threshold admitted) at the boot budget is < 0",
        "convergence_warnings": {"main": n_conv_main, "bootstrap": n_conv_boot, "total": n_conv_total},
        "wall_s": round(time.time() - t0, 1),
    }
    with open(os.path.join(args.out, "result.json"), "w") as f:
        json.dump(result, f, indent=1)
    log(f"\nwrote {os.path.join(args.out, 'result.json')} (wall {result['wall_s']} s)")


if __name__ == "__main__":
    main()
