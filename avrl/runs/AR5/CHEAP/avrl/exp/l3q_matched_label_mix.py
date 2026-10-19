#!/usr/bin/env python3
"""L3q: does mixing uniformly drawn verified groups into the regression's training set beat a matched-budget control?

Redesign of L3m (entries L3m and L3q in lit/avrl/EXPERIMENT_BACKLOG.md). Copied from
avrl/exp/l3l_refit_cadence_trace.py, which is not edited. Same pool, step size, permutations, selection RNG,
six non-embedding features, unscaled LogisticRegression(max_iter=1000), refit_every 20 and prior_strength 2.0
as in L3l's greedy6 arm. Bootstrap design as in L3b: B resamples of the pool with default_rng(10000+b),
seeds 0-3, paired across arms.

Mixing share e in {0.25, 0.5}. Total verifier calls stay at 750 per run.
  U = round(e * 750) uniform calls per run (Python round, so 187.5 gives 188).
  In step t with k_t verifier calls (n_verify, as in allocators.py) and C_t cumulative calls through step t,
  the uniform count is u_t = floor(U * C_t / 750) - floor(U * C_(t-1) / 750). This is the same cumulative
  rounding as allocators.n_verify, it sums to U exactly, and 0 <= u_t <= k_t.
  The u_t uniform groups are drawn without replacement from the step's own groups (sel_rng). The other
  k_t - u_t calls go to the arm's ranking among the groups not drawn.

Arms:
  mixed_threshold(e)  uniform draws, then ThresholdAllocator order on the rest. Matched control.
  greedy6_mix(e)      uniform draws, then GreedyOnlineAllocator(refit_every=20, prior_strength=2.0) on the rest.
                      Training uses all verified labels, uniform ones included.

Controls at e=0 (no uniform calls), checked per seed before any mixed number is read:
  mixed_threshold(0) reproduces E03's threshold per seed (340 341 344 332 331 328 325 333, mean 334.25).
  greedy6_mix(0) reproduces L3l's refit_every=20 greedy6 per seed (332 327 323 323 317 317 318 317, mean 321.75).

Primary metric: paired caught difference greedy6_mix(e) minus mixed_threshold(e) at 750 calls, mean over seeds,
with the percentile 95% CI over B bootstrap resamples (seeds 0-3).
Pass (PASS_CANDIDATE): at some e in {0.25, 0.5} the CI lower bound is above 0. Re-specification candidate only;
it must be re-pre-registered and tested on a second pool (L3r) before any claim verdict changes.
Kill (KILL): at both e the CI lower bound is at or below 0 (the CI includes 0 or lies below 0).

Gold (proxy_error) is used only for scoring and as the update label of verified groups, never for selection.
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
L3I_RESULT = os.path.join(AVRL_DIR, "runs", "L3i", "result.json")
L3L_RESULT = os.path.join(AVRL_DIR, "runs", "L3l", "result.json")
sys.path.insert(0, AVRL_DIR)

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from sklearn.exceptions import ConvergenceWarning  # noqa: E402

import allocators as A  # noqa: E402

STEP = 16
FEAT_COLS = ["agreement", "n_distinct", "mean_tokens", "std_tokens", "any_truncated", "log_qlen"]
E_MIX = [0.25, 0.5]
E_CONTROL = 0.0
ARMS = ["mixed_threshold", "greedy6_mix"]
SPEC_THRESHOLD_CAUGHT_025 = [340, 341, 344, 332, 331, 328, 325, 333]
SPEC_THRESHOLD_MEAN = 334.25
SPEC_GREEDY6_R20_CAUGHT_025 = [332, 327, 323, 323, 317, 317, 318, 317]
SPEC_GREEDY6_R20_MEAN = 321.75
CHANCE_CAUGHT_025 = 109.0


def make_arm(name):
    if name == "mixed_threshold":
        return A.ThresholdAllocator()
    if name == "greedy6_mix":
        return A.GreedyOnlineAllocator(refit_every=20, prior_strength=2.0)
    raise KeyError(name)


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


def subset(pool, ids):
    return {k: v[ids] for k, v in pool.items()}


def run_arm(pool, arm, e, budget, seed):
    """One run at budget 0.25. e is the uniform share of the verifier calls (0 gives the plain arm)."""
    n = len(pool["y"])
    total = round(budget * n)
    U = int(round(e * total))
    perm = np.random.default_rng(1000 + seed).permutation(n)
    alloc = make_arm(arm)
    sel_rng = np.random.default_rng(2000 + seed)
    caught = 0
    caught_uniform = 0
    calls = 0
    uniform_calls = 0
    cum_calls = 0
    cum_uniform = 0
    for t in range(math.ceil(n / STEP)):
        chunk = perm[t * STEP:(t + 1) * STEP]
        k = alloc.n_verify(len(chunk), budget, step=t)
        cum_calls += k
        target = (U * cum_calls) // total
        u = target - cum_uniform
        cum_uniform = target
        assert 0 <= u <= k, (arm, e, seed, t, u, k)
        # cheap, gold-free fields only. cell is a constant: greedy selection never reads cells (as in L3l).
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
        if u > 0:
            uni = [int(x) for x in sel_rng.choice(len(records), size=u, replace=False)]
            uni_set = set(uni)
            rest = [r for r in records if r["idx"] not in uni_set]
            chosen = uni + alloc.select(rest, k - u, sel_rng)
        else:
            uni = []
            chosen = alloc.select(records, k, sel_rng)
        assert len(chosen) == k and len(set(chosen)) == k, (arm, e, seed, t, k, chosen)
        verified = []
        for j in chosen:
            g = int(chunk[j])
            verified.append((pool["X"][g], int(pool["y"][g]), 0))
            caught += int(pool["y"][g])
        for j in uni:
            caught_uniform += int(pool["y"][int(chunk[j])])
        alloc.update(verified)
        calls += k
        uniform_calls += u
    assert calls == total, (arm, e, seed, calls, total)
    assert uniform_calls == U, (arm, e, seed, uniform_calls, U)
    errors = int(pool["y"].sum())
    return {"arm": arm, "e": e, "seed": seed, "errors": errors, "caught": caught, "calls": calls,
            "uniform_calls": uniform_calls, "caught_uniform": caught_uniform}


def arm_stats(rows):
    return {
        "caught_per_seed": [r["caught"] for r in rows],
        "caught_mean": float(np.mean([r["caught"] for r in rows])),
        "caught_uniform_mean": float(np.mean([r["caught_uniform"] for r in rows])),
        "uniform_calls_per_run": rows[0]["uniform_calls"],
        "calls_per_seed": [r["calls"] for r in rows],
    }


def paired(g_rows, t_rows):
    d = [g["caught"] - t["caught"] for g, t in zip(g_rows, t_rows)]
    return {"caught_diff_per_seed": d, "caught_diff_mean": float(np.mean(d))}


def ci95(x):
    x = np.asarray(x, dtype=np.float64)
    lo, hi = np.percentile(x, [2.5, 97.5])
    return {"mean": float(x.mean()), "ci95": [float(lo), float(hi)],
            "lower_above_0": bool(lo > 0), "frac_below_zero": float((x < 0).mean()),
            "n_resamples": int(len(x))}


def check_controls(thr_rows, g_rows, e03, l3i, l3l):
    """Stop before any mixed number is read unless both e=0 arms reproduce their references per seed."""
    got_thr = [r["caught"] for r in thr_rows]
    got_g = [r["caught"] for r in g_rows]
    ref_thr = e03["per_budget"]["0.25"]["arms"]["threshold"]["caught_per_seed"]
    ref_g = l3l["greedy6"]["20"]["caught_per_seed"]
    checks = {
        "threshold_e0_caught_per_seed_equals_spec": got_thr == SPEC_THRESHOLD_CAUGHT_025,
        "threshold_e0_caught_per_seed_equals_e03": got_thr == ref_thr,
        "threshold_e0_mean_equals_334_25": float(np.mean(got_thr)) == SPEC_THRESHOLD_MEAN,
        "greedy6_mix_e0_caught_per_seed_equals_spec": got_g == SPEC_GREEDY6_R20_CAUGHT_025,
        "greedy6_mix_e0_caught_per_seed_equals_l3l_r20": got_g == ref_g,
        "greedy6_mix_e0_caught_per_seed_equals_l3i": got_g == l3i["main_25"]["greedy6"]["caught_per_seed"],
        "greedy6_mix_e0_mean_equals_321_75": float(np.mean(got_g)) == SPEC_GREEDY6_R20_MEAN,
        "uniform_calls_zero_at_e0": all(r["uniform_calls"] == 0 for r in thr_rows + g_rows),
    }
    if not all(checks.values()):
        sys.exit("STOP: e=0 controls do not reproduce the references. "
                 f"threshold caught {got_thr} (E03 {ref_thr}); greedy6 caught {got_g} (L3l r=20 {ref_g}); "
                 f"checks {checks}")
    return checks


def bootstrap(pool, budget, seeds, n_boot, log):
    n = len(pool["y"])
    diffs = {e: [] for e in E_MIX}
    for b in range(n_boot):
        ids = np.random.default_rng(10000 + b).choice(n, size=n, replace=True)
        sub = subset(pool, ids)
        for e in E_MIX:
            g_rows = [run_arm(sub, "greedy6_mix", e, budget, s) for s in seeds]
            t_rows = [run_arm(sub, "mixed_threshold", e, budget, s) for s in seeds]
            diffs[e].append(paired(g_rows, t_rows)["caught_diff_mean"])
        if (b + 1) % 10 == 0:
            log(f"bootstrap {b + 1}/{n_boot} greedy6_mix minus mixed_threshold caught: "
                + " ".join(f"e={e} {diffs[e][-1]:+.2f}" for e in E_MIX))
    return diffs


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pool", required=True)
    ap.add_argument("--budgets", type=float, nargs="+", default=[0.25])
    ap.add_argument("--seeds", type=int, default=8)
    ap.add_argument("--boot", type=int, default=100)
    ap.add_argument("--boot_seeds", type=int, default=4)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    assert args.budgets == [0.25], "L3q is defined at budget 0.25 only"
    assert args.seeds == 8 and args.boot_seeds == 4 and args.boot == 100, \
        "spec: seeds 0-7, bootstrap seeds 0-3, B=100"
    os.makedirs(args.out, exist_ok=True)

    def log(msg):
        print(msg, flush=True)

    t0 = time.time()
    budget = 0.25
    seeds = list(range(args.seeds))
    bs = args.boot_seeds
    pool = load_pool(args.pool)
    n = len(pool["y"])
    n_err = int(pool["y"].sum())
    total = round(budget * n)
    e03 = load_json(E03_RESULT)
    l3i = load_json(L3I_RESULT)
    l3l = load_json(L3L_RESULT)
    log(f"L3q pool {args.pool}: groups={n} proxy_errors={n_err} step={STEP} "
        f"steps={math.ceil(n / STEP)} budget={budget} verifier_calls_per_run={total}")
    log(f"features ({len(FEAT_COLS)}): {FEAT_COLS}")
    log("mixed_threshold(e): uniform draws then ThresholdAllocator on the rest. "
        "greedy6_mix(e): uniform draws then GreedyOnlineAllocator(refit_every=20, prior_strength=2.0) on the rest, "
        "trained on all verified labels.")
    log(f"uniform totals U = round(e*{total}): " + ", ".join(f"e={e} -> {int(round(e * total))}" for e in E_MIX))

    with warnings.catch_warnings(record=True) as wrec_main:
        warnings.simplefilter("always")
        ctl_thr = [run_arm(pool, "mixed_threshold", E_CONTROL, budget, s) for s in seeds]
        ctl_g = [run_arm(pool, "greedy6_mix", E_CONTROL, budget, s) for s in seeds]
        ctl_checks = check_controls(ctl_thr, ctl_g, e03, l3i, l3l)
        log(f"e=0 controls passed. threshold caught {[r['caught'] for r in ctl_thr]} (E03 334.25), "
            f"greedy6 caught {[r['caught'] for r in ctl_g]} (L3l r=20 321.75)")
        rows = {e: {arm: [run_arm(pool, arm, e, budget, s) for s in seeds] for arm in ARMS} for e in E_MIX}
        n_conv_main = sum(issubclass(w.category, ConvergenceWarning) for w in wrec_main)

    main_stats = {}
    main_pair = {}
    log("")
    for e in E_MIX:
        main_stats[e] = {arm: arm_stats(rows[e][arm]) for arm in ARMS}
        main_pair[e] = paired(rows[e]["greedy6_mix"], rows[e]["mixed_threshold"])
        main_pair[e]["caught_diff_mean_seeds_0_3"] = float(np.mean(main_pair[e]["caught_diff_per_seed"][:bs]))
        log(f"e={e}: uniform calls per run {rows[e]['mixed_threshold'][0]['uniform_calls']} of {total}")
        for arm in ARMS:
            st = main_stats[e][arm]
            log(f"  {arm:15s} caught per seed {st['caught_per_seed']} mean {st['caught_mean']:.2f} "
                f"(from uniform draws {st['caught_uniform_mean']:.1f})")
        log(f"  paired greedy6_mix minus mixed_threshold, seeds 0-7: {main_pair[e]['caught_diff_mean']:+.2f}; "
            f"seeds 0-{bs - 1}: {main_pair[e]['caught_diff_mean_seeds_0_3']:+.2f}")

    log(f"\nbootstrap: B={args.boot} resamples of {n} groups with replacement (default_rng(10000+b)), "
        f"seeds 0-{bs - 1}, budget {budget}")
    with warnings.catch_warnings(record=True) as wrec_boot:
        warnings.simplefilter("always")
        diffs = bootstrap(pool, budget, list(range(bs)), args.boot, log)
    n_conv_boot = sum(issubclass(w.category, ConvergenceWarning) for w in wrec_boot)
    boot = {e: ci95(diffs[e]) for e in E_MIX}
    passing = [e for e in E_MIX if boot[e]["lower_above_0"]]
    verdict = "PASS_CANDIDATE" if passing else "KILL"
    for e in E_MIX:
        ci = boot[e]["ci95"]
        reading = "lower bound above 0 (pass at this e)" if boot[e]["lower_above_0"] else \
            "lower bound at or below 0 (kill at this e)"
        log(f"CI95 (greedy6_mix minus mixed_threshold) caught at e={e}: [{ci[0]:+.2f}, {ci[1]:+.2f}], {reading}")
    log(f"ConvergenceWarnings: main (incl. controls) {n_conv_main}, bootstrap {n_conv_boot}")
    log(f"verdict: {verdict}" + (f" at e={passing} (re-specification candidate only, L3r needed)" if passing else ""))

    result = {
        "id": "L3q",
        "pool": args.pool,
        "n_groups": n,
        "n_proxy_errors": n_err,
        "step_size_groups": STEP,
        "steps": math.ceil(n / STEP),
        "budget": budget,
        "verifier_calls_per_run": total,
        "feature_cols": FEAT_COLS,
        "seeds": args.seeds,
        "boot_B": args.boot,
        "boot_seeds": bs,
        "mix_shares_e": E_MIX,
        "uniform_total_per_e": {str(e): int(round(e * total)) for e in E_MIX},
        "assumptions": [
            "steps of 16 groups over a seed-permuted pool (last step 8 groups); k_t = n_verify(step_groups, 0.25, step=t) as in L3l",
            "uniform total U = round(e * 750) (Python round, 187.5 gives 188 at e=0.25); per-step uniform count u_t = floor(U*C_t/750) - floor(U*C_(t-1)/750), C_t = cumulative calls through step t; sums to U, 0 <= u_t <= k_t",
            "uniform groups drawn without replacement from the step's own groups with sel_rng (default_rng(2000+seed)); the rest of the step's calls go to the arm's ranking among the groups not drawn",
            "mixed_threshold rest order: ThresholdAllocator (agreement ascending, then -n_distinct, stable ties)",
            "greedy6_mix rest order: GreedyOnlineAllocator(refit_every=20, prior_strength=2.0) unchanged; threshold cold start before the first fit; trained on all verified labels including uniform ones",
            "features and LogisticRegression(max_iter=1000) unscaled as in L3l and L3i; cell is a constant (no Thompson step)",
            "permutation rng default_rng(1000+seed); bootstrap resamples default_rng(10000+b) with replacement, seeds 0-3, paired across arms",
            "gold (proxy_error) used only for scoring and as the update label of verified groups",
        ],
        "checks": {**ctl_checks, "verifier_calls_equal_750_all_arms": True,
                   "uniform_calls_equal_U_all_runs": True},
        "controls_e0": {
            "mixed_threshold": arm_stats(ctl_thr),
            "greedy6_mix": arm_stats(ctl_g),
        },
        "arms": {str(e): main_stats[e] for e in E_MIX},
        "paired_main": {str(e): main_pair[e] for e in E_MIX},
        "bootstrap": {str(e): boot[e] for e in E_MIX},
        "bootstrap_diffs": {str(e): diffs[e] for e in E_MIX},
        "verdict": verdict,
        "verdict_rule": "PASS_CANDIDATE iff at some e in {0.25, 0.5} the percentile 95% CI lower bound of "
                        "caught(greedy6_mix(e)) minus caught(mixed_threshold(e)) over B resamples (seeds 0-3) is above 0; "
                        "KILL otherwise (both lower bounds at or below 0). PASS_CANDIDATE is a re-specification candidate "
                        "only and needs L3r before any claim verdict changes.",
        "passing_e": passing,
        "references": {
            "threshold_e03_caught_mean": e03["per_budget"]["0.25"]["arms"]["threshold"]["caught_mean"],
            "greedy6_r20_l3l_caught_mean": l3l["greedy6"]["20"]["caught_mean"],
            "chance_caught_025": CHANCE_CAUGHT_025,
            "random_e03_caught_mean": e03["per_budget"]["0.25"]["arms"]["random"]["caught_mean"],
        },
        "convergence_warnings": {"main_incl_controls": n_conv_main, "bootstrap": n_conv_boot},
        "wall_s": round(time.time() - t0, 1),
    }
    out_path = os.path.join(args.out, "result.json")
    with open(out_path, "w") as f:
        json.dump(result, f, indent=1)
    log(f"\nwrote {out_path} (wall {result['wall_s']} s)")


if __name__ == "__main__":
    main()
