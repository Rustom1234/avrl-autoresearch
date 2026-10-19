#!/usr/bin/env python3
"""L3i: does the regression ranking admit fewer corrupted rewards without the 16 embedding components?

Copied from avrl/exp/l3a_thompson_ablation.py and reduced (the L3a file is not edited). Same pool, permutation,
step size, selection RNG, paired bootstrap resampling and verdict rule as L3a. Budget 0.25 only.

Arms:
  threshold  ThresholdAllocator (allocators.py), the baseline. Re-run, and checked against E03 before greedy6 runs.
  greedy6    GreedyOnlineAllocator (allocators.py, unchanged) on the six non-embedding cheap features
             (agreement, n_distinct, mean_tokens, std_tokens, any_truncated, log_qlen). No noise, no cells in
             selection, refit_every 20, unscaled LogisticRegression(max_iter=1000) as in E03.
  greedy22   E03 reference (22 features including pc0..pc15), read from runs/E03/result.json. Not rerun.

Pool: runs/p0/feats.parquet (3000 groups, 8 rollouts each), read only. Gold (proxy_error) is used only for
scoring and as the update label of verified groups, never as a selection input.

CPU only. No GPU, no pip, no ssh. Writes result.json into --out; the runner redirects stdout to run.log.
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
E03_RESULT = os.path.join(AVRL_DIR, "runs", "E03", "result.json")
sys.path.insert(0, AVRL_DIR)

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from sklearn.exceptions import ConvergenceWarning  # noqa: E402

import allocators as A  # noqa: E402

STEP = 16
FEAT_COLS = ["agreement", "n_distinct", "mean_tokens", "std_tokens", "any_truncated", "log_qlen"]
SPEC_THRESHOLD_CAUGHT_025 = [340, 341, 344, 332, 331, 328, 325, 333]
ARMS = {"threshold": A.ThresholdAllocator, "greedy6": A.GreedyOnlineAllocator}


def load_pool(path):
    assert len(FEAT_COLS) == 6 and not any(c.startswith("pc") for c in FEAT_COLS), FEAT_COLS
    df = pd.read_parquet(path)
    assert len(df) == 3000, f"expected 3000 groups, got {len(df)}"
    return {
        "X": df[FEAT_COLS].to_numpy(dtype=np.float64),
        "agreement": df["agreement"].to_numpy(dtype=np.float64),
        "n_distinct": df["n_distinct"].to_numpy(dtype=np.int64),
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
        # cheap, gold-free fields only. cell is a constant: greedy selection never reads cells, and the key
        # only satisfies LearnedAllocator.update's (feat_vec, label, cell) signature.
        records = [
            {
                "idx": j,
                "feat_vec": pool["X"][g],
                "cell": 0,
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
            verified.append((pool["X"][g], int(pool["y"][g]), 0))
            caught += int(pool["y"][g])
        alloc.update(verified)
        calls += k
    assert calls == round(budget * n), (arm, seed, calls)
    errors = int(pool["y"].sum())
    return {"arm": arm, "budget": budget, "seed": seed, "errors": errors, "caught": caught,
            "admitted": (errors - caught) / errors, "calls": calls}


def arm_stats(rows):
    return {
        "caught_per_seed": [r["caught"] for r in rows],
        "caught_mean": float(np.mean([r["caught"] for r in rows])),
        "admitted_per_seed": [round(r["admitted"], 6) for r in rows],
        "admitted_mean": float(np.mean([r["admitted"] for r in rows])),
        "calls_per_seed": [r["calls"] for r in rows],
    }


def summarize(x):
    x = np.asarray(x, dtype=np.float64)
    lo, hi = np.percentile(x, [2.5, 97.5])
    return {"mean": float(x.mean()), "ci95": [float(lo), float(hi)],
            "upper_below_0": bool(hi < 0), "frac_below_zero": float((x < 0).mean()),
            "n_below_zero": int((x < 0).sum()), "n_resamples": int(len(x))}


def load_e03():
    with open(E03_RESULT) as f:
        return json.load(f)


def check_threshold_vs_e03(thr_rows, e03, n_err):
    """Stop before any greedy6 number is read if the threshold arm does not reproduce E03."""
    e = e03["per_budget"]["0.25"]
    ref = e["arms"]["threshold"]
    got_caught = [r["caught"] for r in thr_rows]
    got_adm = [round(r["admitted"], 6) for r in thr_rows]
    checks = {
        "threshold_caught_per_seed_equals_spec": got_caught == SPEC_THRESHOLD_CAUGHT_025,
        "threshold_caught_per_seed_equals_e03": got_caught == ref["caught_per_seed"],
        "threshold_admitted_per_seed_equals_e03": got_adm == ref["admitted_per_seed"],
        "errors_per_seed_equal_e03": all(x == n_err for x in e["errors_per_seed"]),
    }
    if not all(checks.values()):
        sys.exit("STOP: threshold does not reproduce E03 at 25%. "
                 f"got caught {got_caught}, got admitted {got_adm}, "
                 f"E03 caught {ref['caught_per_seed']}, E03 admitted {ref['admitted_per_seed']}, "
                 f"checks {checks}")
    return checks


def bootstrap(pool, budget, seeds, n_boot, e_fixed, log):
    """Paired bootstrap over pool groups. diffs uses each resample's own error count as the admitted
    denominator (as in L3a, the primary metric). fixed uses the main pool's error count e_fixed (436),
    the literal formula in the L3i spec, reported alongside."""
    n = len(pool["y"])
    diffs, fixed, thr_boot = [], [], []
    for b in range(n_boot):
        ids = np.random.default_rng(10000 + b).choice(n, size=n, replace=True)
        sub = subset(pool, ids)
        thr = [run_arm(sub, "threshold", budget, s) for s in seeds]
        grd = [run_arm(sub, "greedy6", budget, s) for s in seeds]
        thr_boot.append(float(np.mean([r["admitted"] for r in thr])))
        diffs.append(float(np.mean([g["admitted"] - t["admitted"] for g, t in zip(grd, thr)])))
        fixed.append(float(np.mean([t["caught"] - g["caught"] for g, t in zip(grd, thr)])) / e_fixed)
        if (b + 1) % 10 == 0:
            log(f"bootstrap {b + 1}/{n_boot} greedy6-threshold {diffs[-1]:+.4f}")
    return diffs, fixed, thr_boot


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pool", required=True)
    ap.add_argument("--budgets", type=float, nargs="+", default=[0.25])
    ap.add_argument("--seeds", type=int, default=8)
    ap.add_argument("--boot", type=int, default=100)
    ap.add_argument("--boot_budget", type=float, default=0.25)
    ap.add_argument("--boot_seeds", type=int, default=4)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    assert args.budgets == [0.25] and args.boot_budget == 0.25, "L3i is defined at budget 0.25 only"
    assert args.seeds == 8 and args.boot_seeds == 4 and args.boot == 100, "spec: seeds 0-7, bootstrap seeds 0-3, B=100"
    os.makedirs(args.out, exist_ok=True)

    def log(msg):
        print(msg, flush=True)

    t0 = time.time()
    budget = 0.25
    bs = args.boot_seeds
    seeds = list(range(args.seeds))
    pool = load_pool(args.pool)
    n = len(pool["y"])
    n_err = int(pool["y"].sum())
    e03 = load_e03()
    e_greedy22 = e03["per_budget"]["0.25"]["arms"]["greedy"]
    log(f"L3i pool {args.pool}: groups={n} proxy_errors={n_err} step={STEP} "
        f"steps={math.ceil(n / STEP)} budget={budget} verifier_calls_per_run={round(budget * n)}")
    log(f"feature columns ({len(FEAT_COLS)}): {FEAT_COLS} (pc0..pc15 dropped)")
    log("greedy6 = allocators.GreedyOnlineAllocator unchanged: refit_every=20 prior_strength=2.0 "
        "LogisticRegression(max_iter=1000) unscaled; no noise, no cells in selection")
    log("arms: threshold (baseline), greedy6 (this run), greedy22 (E03 reference, not rerun)")

    with warnings.catch_warnings(record=True) as wrec:
        warnings.simplefilter("always")

        thr_rows = [run_arm(pool, "threshold", budget, s) for s in seeds]
        thr_checks = check_threshold_vs_e03(thr_rows, e03, n_err)
        log(f"threshold check vs E03 passed. caught per seed {[r['caught'] for r in thr_rows]}")

        g_rows = [run_arm(pool, "greedy6", budget, s) for s in seeds]
        st = arm_stats(g_rows)
        log(f"  greedy6 caught per seed {st['caught_per_seed']} mean {st['caught_mean']:.2f} "
            f"admitted mean {st['admitted_mean']:.4f}")
        n_conv_main = sum(issubclass(w.category, ConvergenceWarning) for w in wrec)
        log(f"ConvergenceWarnings during main runs: {n_conv_main}")

        main_diff = [g["admitted"] - t["admitted"] for g, t in zip(g_rows, thr_rows)]
        point = float(np.mean(main_diff[:bs]))
        log(f"point estimate (greedy6 - threshold admitted) at 0.25, seeds 0-{bs - 1}: {point:+.4f}")

        log(f"\nbootstrap: B={args.boot} resamples of {n} groups with replacement (default_rng(10000+b)), "
            f"seeds 0-{bs - 1}, budget {budget:.2f}")
        diffs, fixed, thr_boot = bootstrap(pool, budget, list(range(bs)), args.boot, n_err, log)
        n_conv_total = sum(issubclass(w.category, ConvergenceWarning) for w in wrec)

    n_conv_boot = n_conv_total - n_conv_main
    log(f"ConvergenceWarnings during bootstrap: {n_conv_boot}; total {n_conv_total}")

    s = summarize(diffs)
    s_fixed = summarize(fixed)
    verdict = "PASS" if s["upper_below_0"] else "KILL"
    log(f"CI95 (greedy6 - threshold) admitted at 0.25, per-resample denominator: {[round(x, 4) for x in s['ci95']]}, "
        f"resamples below 0: {s['n_below_zero']}/{s['n_resamples']}")
    log(f"CI95 (greedy6 - threshold) admitted at 0.25, fixed denominator {n_err}: "
        f"{[round(x, 4) for x in s_fixed['ci95']]}, resamples below 0: {s_fixed['n_below_zero']}/{s_fixed['n_resamples']}")
    log(f"caught mean seeds 0-7: greedy6 {st['caught_mean']:.2f}, threshold {np.mean([r['caught'] for r in thr_rows]):.2f}, "
        f"greedy22 (E03) {e_greedy22['caught_mean']:.2f}")
    log(f"overall L3i verdict: {verdict}")

    result = {
        "id": "L3i",
        "pool": args.pool,
        "n_groups": n,
        "n_proxy_errors": n_err,
        "step_size_groups": STEP,
        "budget": budget,
        "verifier_calls_per_run": round(budget * n),
        "feature_cols": FEAT_COLS,
        "seeds_main": args.seeds,
        "seeds_bootstrap": bs,
        "B": args.boot,
        "assumptions": [
            "steps of 16 groups over a seed-permuted pool (last step 8 groups); n_verify(step_groups, 0.25, step=t) as in E03 and L3a",
            "feat_vec = [agreement, n_distinct, mean_tokens, std_tokens, any_truncated, log_qlen] (6 dims); pc0..pc15 dropped, the only change from the E03 22-dim vector",
            "greedy6 = allocators.GreedyOnlineAllocator unchanged: regression ranks groups by predicted proxy_error; cold start before the first fit is the threshold order",
            "cell key is a constant 0 in every record and verified tuple; greedy selection never reads cells",
            "admitted = (errors - caught) / errors. Main table: errors = 436. Primary bootstrap: errors of each resample (as L3a). Supplementary bootstrap: fixed 436 denominator (literal L3i spec formula)",
            "permutation rng default_rng(1000+seed); selection rng default_rng(2000+seed) shared across steps of one run; bootstrap resample default_rng(10000+b), as in E03 and L3a",
            "gold (proxy_error) used only for scoring and as the update label of verified groups",
            "greedy22 is the E03 reference read from runs/E03/result.json, not rerun",
            "threshold per-resample values are not checkable against E03 (E03 stores CI summaries only)",
        ],
        "checks": {
            **thr_checks,
            "verifier_calls_equal_750_all_arms": True,
        },
        "main_25": {
            "threshold": arm_stats(thr_rows),
            "greedy6": st,
            "greedy22_e03_reference": {
                "caught_per_seed": e_greedy22["caught_per_seed"],
                "caught_mean": e_greedy22["caught_mean"],
                "admitted_per_seed": e_greedy22["admitted_per_seed"],
                "admitted_mean": e_greedy22["admitted_mean"],
                "source": E03_RESULT,
            },
        },
        "caught_mean_seeds_0_to_7": {
            "threshold": float(np.mean([r["caught"] for r in thr_rows])),
            "greedy6": st["caught_mean"],
            "greedy22_e03": e_greedy22["caught_mean"],
            "greedy6_minus_threshold": st["caught_mean"] - float(np.mean([r["caught"] for r in thr_rows])),
        },
        "main_paired_greedy6_minus_threshold_admitted_per_seed": [round(x, 6) for x in main_diff],
        "metric": {
            "name": "paired bootstrap of (greedy6 admitted - threshold admitted) at 25%, mean over seeds 0-3, "
                    "percentile 95% CI over B resamples of the 3000 groups, with the count of resamples below 0",
            "budget": budget,
            "point_estimate_seeds_0_to_3": point,
            "bootstrap": {"B": args.boot, "seeds": bs, "budget": budget, "denominator": "per-resample errors (primary)", **s},
            "bootstrap_fixed_denominator_436": {"denominator": n_err, **s_fixed},
            "threshold_admitted_per_resample_mean_seeds_0_to_3": thr_boot,
            "greedy6_minus_threshold_admitted_per_resample": diffs,
        },
        "verdict": verdict,
        "verdict_rule": "PASS iff the 95% percentile CI upper bound of (greedy6 admitted - threshold admitted) at 0.25 is < 0; "
                        "otherwise KILL (CI includes 0 or lies above 0). A PASS is a candidate re-specification, not a confirmed claim: "
                        "claim-2 credit needs re-pre-registration and a fresh pool (L3f).",
        "convergence_warnings": {"main": n_conv_main, "bootstrap": n_conv_boot, "total": n_conv_total},
        "wall_s": round(time.time() - t0, 1),
    }
    out_path = os.path.join(args.out, "result.json")
    with open(out_path, "w") as f:
        json.dump(result, f, indent=1)
    log(f"\nwrote {out_path} (wall {result['wall_s']} s)")


if __name__ == "__main__":
    main()
