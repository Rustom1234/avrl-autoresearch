#!/usr/bin/env python3
"""L3l: refit cadence and per-step gap trace for greedy6 (diagnostic, not a pass/kill test).

Copied from avrl/exp/l3i_no_embedding_regression.py (L3i itself is not edited). Same pool, permutation, step size,
selection RNG, six non-embedding features and unscaled LogisticRegression(max_iter=1000) as L3i. Budget 0.25 only.
The bootstrap is dropped: this run traces the cumulative caught curve, it produces no CI.

Arms (main runs, seeds 0-7, 750 verifier calls each):
  threshold    ThresholdAllocator. Checked against E03 (per-seed caught 340 341 344 332 331 328 325 333, mean 334.25)
               before any greedy number is reported.
  greedy6_r    GreedyOnlineAllocator(refit_every=r, prior_strength=2.0) for r in --refit_every (default 10 20 40).
               Control: r=20 must reproduce L3i's greedy6 per-seed caught (runs/L3i/result.json, main_25.greedy6),
               otherwise the script stops before writing anything.

Recorded per run: cumulative caught after every step (188 steps) and every refit (step index, verified count).
Gold (proxy_error) is used only for scoring and as the update label of verified groups, never for selection.

Diagnostic quantities (result.json). The script computes no verdict; the reading is done from the curve.
  gap curve        mean over seeds of (greedy6_r cumulative caught - threshold cumulative caught) after each step
  first_nonzero    first step where the per-seed gap is nonzero (the gap opens here)
  first_below      first step where greedy6_r cumulative caught is below threshold, per seed
  fraction_of_final_gap_at_mid   gap at the midpoint step (step 93 of 0-187) / gap at the final step, from the
                   mean curve. Reference points: about 1.0 if the gap is flat after it opens (loss in place by
                   mid-run); about 0.49 if it grows linearly from the first refit to the end, (93 - 4) / (187 - 4)
                   for refit_every 20.

CPU only. No GPU, no pip, no ssh.
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
L3I_RESULT = os.path.join(AVRL_DIR, "runs", "L3i", "result.json")
sys.path.insert(0, AVRL_DIR)

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from sklearn.exceptions import ConvergenceWarning  # noqa: E402

import allocators as A  # noqa: E402

STEP = 16
FEAT_COLS = ["agreement", "n_distinct", "mean_tokens", "std_tokens", "any_truncated", "log_qlen"]
SPEC_THRESHOLD_CAUGHT_025 = [340, 341, 344, 332, 331, 328, 325, 333]
SPEC_GREEDY6_R20_CAUGHT_025 = [332, 327, 323, 323, 317, 317, 318, 317]
SPEC_GREEDY6_R20_MEAN = 321.75
CHANCE_CAUGHT_025 = 109.0


def load_json(path):
    with open(path) as f:
        return json.load(f)


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


def run_arm(pool, make_alloc, budget, seed):
    """As L3i run_arm, plus cumulative caught after every step and the steps where the regression refits."""
    n = len(pool["y"])
    perm = np.random.default_rng(1000 + seed).permutation(n)
    alloc = make_alloc()
    sel_rng = np.random.default_rng(2000 + seed)
    caught = 0
    calls = 0
    cum = []
    refits = []
    last_fit = 0
    for t in range(math.ceil(n / STEP)):
        chunk = perm[t * STEP:(t + 1) * STEP]
        k = alloc.n_verify(len(chunk), budget, step=t)
        # cell is a constant: greedy selection never reads cells (same as L3i).
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
        assert len(chosen) == k and len(set(chosen)) == k, (t, k, chosen)
        verified = []
        for j in chosen:
            g = int(chunk[j])
            verified.append((pool["X"][g], int(pool["y"][g]), 0))
            caught += int(pool["y"][g])
        alloc.update(verified)
        calls += k
        cum.append(caught)
        fit_n = getattr(alloc, "last_fit_n", 0)
        if fit_n != last_fit:
            refits.append({"step": t, "n_verified": int(fit_n)})
            last_fit = fit_n
    assert calls == round(budget * n), (seed, calls)
    errors = int(pool["y"].sum())
    return {"seed": seed, "errors": errors, "caught": caught, "calls": calls,
            "admitted": (errors - caught) / errors, "cum_caught": cum, "refits": refits}


def check_threshold_vs_e03(thr_rows, e03, n_err):
    """Stop before any greedy6 number is reported if the threshold arm does not reproduce E03 (as L3i)."""
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


def check_greedy_r20_vs_l3i(g20_rows, l3i):
    """Stop before any greedy6 number is reported if refit_every=20 does not reproduce L3i's greedy6."""
    got = [r["caught"] for r in g20_rows]
    ref = l3i["main_25"]["greedy6"]["caught_per_seed"]
    checks = {
        "greedy6_r20_caught_per_seed_equals_l3i": got == ref,
        "greedy6_r20_caught_per_seed_equals_spec": got == SPEC_GREEDY6_R20_CAUGHT_025,
        "greedy6_r20_caught_mean_equals_321_75": float(np.mean(got)) == SPEC_GREEDY6_R20_MEAN,
    }
    if not all(checks.values()):
        sys.exit("STOP: greedy6 at refit_every=20 does not reproduce L3i at 25%. "
                 f"got caught {got}, L3i caught {ref}, spec {SPEC_GREEDY6_R20_CAUGHT_025}, checks {checks}")
    return checks


def gap_summary(thr_rows, g_rows):
    thr = np.array([r["cum_caught"] for r in thr_rows], dtype=np.float64)
    grd = np.array([r["cum_caught"] for r in g_rows], dtype=np.float64)
    gap = grd - thr  # (seeds, steps)
    T = gap.shape[1]
    mid = T // 2 - 1
    curve = gap.mean(axis=0)
    gm, gl = float(curve[mid]), float(curve[-1])
    frac_mid = gm / gl if abs(gl) >= 1 else None
    first_nonzero = [int(np.nonzero(row != 0)[0][0]) if np.any(row != 0) else None for row in gap]
    first_below = [int(np.nonzero(row < 0)[0][0]) if np.any(row < 0) else None for row in gap]
    refit_steps = [r["step"] for r in g_rows[0]["refits"]]
    for r in g_rows:
        assert [x["step"] for x in r["refits"]] == refit_steps, "refit schedule should not depend on the seed"
    return {
        "caught_per_seed": [r["caught"] for r in g_rows],
        "caught_mean": float(np.mean([r["caught"] for r in g_rows])),
        "refit_steps": refit_steps,
        "refits": g_rows[0]["refits"],
        "first_nonzero_step_per_seed": first_nonzero,
        "first_below_threshold_step_per_seed": first_below,
        "gap_mean_curve": [round(float(x), 4) for x in curve],
        "gap_at_mid_step": gm,
        "gap_at_final_step": gl,
        "mid_step": mid,
        "fraction_of_final_gap_at_mid": frac_mid,
        "cum_caught": [r["cum_caught"] for r in g_rows],
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pool", required=True)
    ap.add_argument("--budgets", type=float, nargs="+", default=[0.25])
    ap.add_argument("--seeds", type=int, default=8)
    ap.add_argument("--refit_every", type=int, nargs="+", default=[10, 20, 40])
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    assert args.budgets == [0.25], "L3l is defined at budget 0.25 only"
    assert args.seeds == 8 and args.refit_every == [10, 20, 40], "spec: seeds 0-7, refit_every 10 20 40"
    os.makedirs(args.out, exist_ok=True)

    def log(msg):
        print(msg, flush=True)

    t0 = time.time()
    budget = 0.25
    seeds = list(range(args.seeds))
    pool = load_pool(args.pool)
    n = len(pool["y"])
    n_err = int(pool["y"].sum())
    e03 = load_json(E03_RESULT)
    l3i = load_json(L3I_RESULT)
    log(f"L3l pool {args.pool}: groups={n} proxy_errors={n_err} step={STEP} "
        f"steps={math.ceil(n / STEP)} budget={budget} verifier_calls_per_run={round(budget * n)}")
    log(f"features ({len(FEAT_COLS)}): {FEAT_COLS}")
    log(f"greedy6_r = allocators.GreedyOnlineAllocator(refit_every=r, prior_strength=2.0), r in {args.refit_every}; "
        "LogisticRegression(max_iter=1000) unscaled; cold start = threshold order")

    with warnings.catch_warnings(record=True) as wrec:
        warnings.simplefilter("always")
        thr_rows = [run_arm(pool, A.ThresholdAllocator, budget, s) for s in seeds]
        thr_checks = check_threshold_vs_e03(thr_rows, e03, n_err)
        log(f"threshold check vs E03 passed. caught per seed {[r['caught'] for r in thr_rows]}")

        g_rows = {}
        for r in args.refit_every:
            make = (lambda r=r: A.GreedyOnlineAllocator(refit_every=r, prior_strength=2.0))
            g_rows[r] = [run_arm(pool, make, budget, s) for s in seeds]
        ctl = check_greedy_r20_vs_l3i(g_rows[20], l3i)
        log("greedy6 r=20 check vs L3i passed.")
        n_conv = sum(issubclass(w.category, ConvergenceWarning) for w in wrec)

    T = len(thr_rows[0]["cum_caught"])
    summaries = {r: gap_summary(thr_rows, g_rows[r]) for r in args.refit_every}

    log(f"\nthreshold caught per seed {[r['caught'] for r in thr_rows]} mean "
        f"{np.mean([r['caught'] for r in thr_rows]):.2f} (E03 334.25), chance {CHANCE_CAUGHT_025}")
    for r in args.refit_every:
        s = summaries[r]
        log(f"\nr={r} greedy6 caught per seed {s['caught_per_seed']} mean {s['caught_mean']:.2f}")
        log(f"  refits at steps {s['refit_steps'][:10]}{' ...' if len(s['refit_steps']) > 10 else ''} "
            f"(total {len(s['refit_steps'])})")
        log(f"  first nonzero gap step per seed {s['first_nonzero_step_per_seed']}")
        log(f"  first step below threshold per seed {s['first_below_threshold_step_per_seed']}")
        log(f"  gap at mid step {s['mid_step']}: {s['gap_at_mid_step']:+.2f}, final: {s['gap_at_final_step']:+.2f}, "
            f"fraction_of_final_gap_at_mid {s['fraction_of_final_gap_at_mid']}")
        log("  mean gap every 10 steps: " + " ".join(f"{s['gap_mean_curve'][t]:+.1f}" for t in range(0, T, 10))
            + f" {s['gap_mean_curve'][-1]:+.1f}")
    log(f"\nConvergenceWarnings total: {n_conv}")

    result = {
        "id": "L3l",
        "pool": args.pool,
        "n_groups": n,
        "n_proxy_errors": n_err,
        "step_size_groups": STEP,
        "steps": T,
        "budget": budget,
        "verifier_calls_per_run": round(budget * n),
        "feature_cols": FEAT_COLS,
        "seeds": args.seeds,
        "refit_every_values": args.refit_every,
        "assumptions": [
            "steps of 16 groups over a seed-permuted pool (last step 8 groups); n_verify(step_groups, 0.25, step=t) as in L3i",
            "feature set and unscaled LogisticRegression(max_iter=1000) as in L3i; the only change is refit_every",
            "greedy6_r is allocators.GreedyOnlineAllocator unchanged with refit_every=r and prior_strength=2.0",
            "refit schedule is deterministic (the trigger counts verified groups), so it is the same for every seed",
            "threshold and greedy6 pick identically until the first refit (cold start), so the gap is 0 before it",
            "permutation rng default_rng(1000+seed); selection rng default_rng(2000+seed), as in L3i",
            "gold (proxy_error) used only for scoring and as the update label of verified groups",
            "no verdict label is computed; fraction_of_final_gap_at_mid is a summary of the mean curve (reference points in the script docstring)",
        ],
        "checks": {**thr_checks, **ctl, "verifier_calls_equal_750_all_arms": True},
        "chance_caught_025": CHANCE_CAUGHT_025,
        "threshold": {
            "caught_per_seed": [r["caught"] for r in thr_rows],
            "caught_mean": float(np.mean([r["caught"] for r in thr_rows])),
            "cum_caught": [r["cum_caught"] for r in thr_rows],
        },
        "greedy6": {str(r): summaries[r] for r in args.refit_every},
        "references": {
            "threshold_e03_caught_mean": e03["per_budget"]["0.25"]["arms"]["threshold"]["caught_mean"],
            "greedy6_r20_l3i_caught_mean": l3i["main_25"]["greedy6"]["caught_mean"],
            "greedy22_e03_caught_mean": e03["per_budget"]["0.25"]["arms"]["greedy"]["caught_mean"],
        },
        "convergence_warnings_total": n_conv,
        "wall_s": round(time.time() - t0, 1),
    }
    out_path = os.path.join(args.out, "result.json")
    with open(out_path, "w") as f:
        json.dump(result, f, indent=1)
    log(f"\nwrote {out_path} (wall {result['wall_s']} s)")


if __name__ == "__main__":
    main()
