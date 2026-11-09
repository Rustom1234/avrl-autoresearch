#!/usr/bin/env python3
"""L3aa: ceiling for within-block ordering. If the threshold order's exact (agreement, n_distinct) blocks were sorted by gold
proxy_error (error groups first), how many more corrupted rewards would 25% verification admit on the E01 GSM8K pool?

Derived from avrl/exp/l3h_tiebreak_threshold.py (that file and allocators.py are not edited). Same pool, seed permutations,
16-group steps, block order, control check, bootstrap designs A and B and metric definitions as L3h. Budget 0.25 only.
The reading (READING_RULE) is diagnostic: it decides which experiment is next, and it is not a claim verdict.

Arms:
  threshold        ThresholdAllocator (allocators.py, unchanged; the control). Per seed checked against the spec list and
                   runs/L3s/result.json main_25.threshold before the oracle arm runs.
  oracle_tiebreak  CEILING ONLY, not an allocator. Each 16-group step sorts by (agreement ascending, n_distinct descending,
                   proxy_error descending), stable, and verifies the first k. Gold (proxy_error) is read only here, and only to
                   order groups inside a block that shares (agreement, n_distinct). No regression is fitted, because nothing in
                   this arm's selection reads one. The gold order applies from step 0 (L3h's cold-start gate is not used, so
                   the ceiling holds in every step). Within a step the oracle's first k hold at least as many errors as the
                   threshold's first k, so per seed oracle caught >= threshold caught. main() asserts this.

Design A (primary, sets U_A): full pool resampled WITH replacement, B = 100, rng default_rng(10000 + b), seeds 0-3, 750 calls
per run, each resample's own error count as the denominator. U_A is the 97.5th percentile of (oracle admitted minus threshold
admitted).
Design B (the L3s design): half-pool subsamples WITHOUT replacement, size n // 2 = 1500, B = 200, rng default_rng(10000 + b),
seeds 0-3, 375 calls per run. Percentile and basic 95% bounds and bias (bootstrap mean minus the seeds 0-3 point) as in L3h.

The six pool feature columns are loaded and carried as in L3h, but neither arm's selection reads them.
Pool: runs/p0/feats.parquet (3000 groups, 8 rollouts each), read only.
CPU only. No GPU, no pip, no ssh. Writes result.json into --out; the runner redirects stdout to run.log.
"""
import argparse
import json
import math
import os
import sys
import time

sys.dont_write_bytecode = True
AVRL_DIR = "/homes/rdubash/avrl_live/CHEAP/avrl"
L3S_RESULT = os.path.join(AVRL_DIR, "runs", "L3s", "result.json")
sys.path.insert(0, AVRL_DIR)

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

import allocators as A  # noqa: E402

STEP = 16
FEAT_COLS = ["agreement", "n_distinct", "mean_tokens", "std_tokens", "any_truncated", "log_qlen"]
SPEC_THRESHOLD_CAUGHT_025 = [340, 341, 344, 332, 331, 328, 325, 333]
SPEC_THRESHOLD_CAUGHT_MEAN_025 = 334.25
B_A = 100
B_B = 200
READING_THRESHOLD = -0.005
READING_RULE = (
    "Reading, fixed before the run (diagnostic, no claim verdict). D = mean over seeds 0-7 of (oracle admitted minus threshold "
    "admitted) at 25%, 750 verifier calls per run. If D > -0.005, read CAPPED: within-block ordering is capped on this pool "
    "whatever the ranker, the tie-break family is closed on this pool, and L3ac is not run. If D <= -0.005, read HEADROOM: "
    "within-block ordering has room on this pool, and L3ab (within-block discrimination of the regression) decides whether "
    "the ranker is the cause. INVALID_CONTROL (no reading) if the threshold arm does not reproduce the per-seed caught counts "
    "340, 341, 344, 332, 331, 328, 325, 333 (mean 334.25) and runs/L3s/result.json; no oracle number is read then. "
    "Design A (U_A) and design B (percentile and basic upper bounds, bias) are reported and do not change the reading. "
    "This entry cannot change claim 2 or claim 3."
)
ASSUMPTIONS = [
    "steps of 16 groups over a seed-permuted pool (last step 8 groups); n_verify(step_groups, 0.25, step=t) as in E03, L3h and L3s",
    "block order is ThresholdAllocator._order (agreement ascending, n_distinct descending); the oracle adds proxy_error descending "
    "as a third key, stable on chunk position",
    "oracle_tiebreak has no regression, no RNG and no cells in selection; gold is read only through the proxy_error field that "
    "run_arm adds to that arm's records",
    "gold order from step 0 (no cold-start gate), so the oracle is a ceiling in every step; the threshold top-k is the same set "
    "as the oracle's whenever the cut falls on a block boundary",
    "diagnostic (no reading effect): steps where the oracle verified set differs from the threshold top-k",
    "admitted = (errors - caught) / errors. Main table: errors = 436. Design A denominator: errors of each resample. "
    "Design B denominator: errors of each subsample",
    "permutation rng default_rng(1000+seed); selection rng default_rng(2000+seed) per run (unused by both arms)",
    "design B draws are default_rng(10000+b).choice(n, size=n//2, replace=False), the same draws as L3h and L3s; a cross-check of the "
    "threshold subsample means against runs/L3s is recorded in checks and does not gate the reading",
    "gold (proxy_error) is used for scoring and as the oracle sort key inside blocks, and nowhere else",
    "gating: the oracle arm runs only if the threshold reproduces the spec list and runs/L3s/result.json per seed",
]


class OracleTieBreak(A.BaseAllocator):
    """Ceiling only. Threshold block order with exact (agreement, n_distinct) ties broken by gold proxy_error, error groups first.
    Nothing here reads a fitted model. Gold enters only through the proxy_error field that run_arm adds for this arm."""
    name = "oracle_tiebreak"

    def __init__(self):
        self.n_steps = 0     # steps compared with the threshold top-k (diagnostic only)
        self.n_changed = 0   # of those, steps where the oracle verified set differs from the threshold top-k

    def select(self, group_records, k, rng):
        if k == 0:
            return []
        threshold_top = [r["idx"] for r in A.ThresholdAllocator._order(group_records)[:k]]
        order = sorted(group_records, key=lambda r: (r["agreement"], -r["n_distinct"], -r["proxy_error"]))
        chosen = [r["idx"] for r in order[:k]]
        self.n_steps += 1
        self.n_changed += int(set(chosen) != set(threshold_top))
        return chosen


ARMS = {"threshold": A.ThresholdAllocator, "oracle_tiebreak": OracleTieBreak}


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
        # cheap, gold-free fields for every arm. The oracle arm alone also gets gold proxy_error, as its ceiling sort key.
        records = []
        for j, g in enumerate(chunk):
            rec = {
                "idx": j,
                "feat_vec": pool["X"][g],
                "cell": 0,
                "agreement": float(pool["agreement"][g]),
                "n_distinct": int(pool["n_distinct"][g]),
            }
            if arm == "oracle_tiebreak":
                rec["proxy_error"] = int(pool["y"][g])
            records.append(rec)
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
            "admitted": (errors - caught) / errors, "calls": calls,
            "changed_steps": getattr(alloc, "n_changed", 0), "compared_steps": getattr(alloc, "n_steps", 0)}


def arm_stats(rows):
    return {
        "caught_per_seed": [r["caught"] for r in rows],
        "caught_mean": float(np.mean([r["caught"] for r in rows])),
        "admitted_per_seed": [round(r["admitted"], 6) for r in rows],
        "admitted_mean": float(np.mean([r["admitted"] for r in rows])),
        "calls_per_seed": [r["calls"] for r in rows],
    }


def summarize(x, point):
    """Bootstrap summary. point is the seeds 0-3 full-pool paired mean. The percentile CI comes from the draws; the basic CI
    is the reflected interval [2 * point - q97.5, 2 * point - q2.5]."""
    x = np.asarray(x, dtype=np.float64)
    lo, hi = np.percentile(x, [2.5, 97.5])
    return {"n_resamples": int(len(x)), "mean": float(x.mean()), "bias": float(x.mean() - point),
            "sd": float(x.std(ddof=1)), "ci95_percentile": [float(lo), float(hi)],
            "ci95_basic": [float(2 * point - hi), float(2 * point - lo)],
            "n_below_zero": int((x < 0).sum())}


def check_threshold_vs_l3s(rows, l3s):
    """Threshold must reproduce the spec list and runs/L3s/result.json per seed at 25%. Checked before the oracle arm runs."""
    ref = l3s["main_25"]["threshold"]
    got = [r["caught"] for r in rows]
    checks = {
        "caught_per_seed_equals_spec": got == SPEC_THRESHOLD_CAUGHT_025,
        "caught_mean_equals_spec": float(np.mean(got)) == SPEC_THRESHOLD_CAUGHT_MEAN_025,
        "caught_per_seed_equals_l3s": got == ref["caught_per_seed"],
        "admitted_per_seed_equals_l3s": [round(r["admitted"], 6) for r in rows] == ref["admitted_per_seed"],
    }
    return all(checks.values()), checks


def bootstrap(pool, budget, seeds, n_boot, replace, size, label, log):
    """Paired bootstrap over pool groups, threshold and oracle only. replace=True: full-pool resamples (design A, size n).
    replace=False: subsamples without replacement (design B, size n // 2). Each draw's own error count is the denominator."""
    n = len(pool["y"])
    diffs, thr_boot = [], []
    for b in range(n_boot):
        ids = np.random.default_rng(10000 + b).choice(n, size=size, replace=replace)
        if not replace:
            assert len(np.unique(ids)) == size, b
        sub = subset(pool, ids)
        thr = [run_arm(sub, "threshold", budget, s) for s in seeds]
        ora = [run_arm(sub, "oracle_tiebreak", budget, s) for s in seeds]
        thr_boot.append(float(np.mean([r["admitted"] for r in thr])))
        diffs.append(float(np.mean([o["admitted"] - t["admitted"] for o, t in zip(ora, thr)])))
        if (b + 1) % 20 == 0:
            log(f"{label} {b + 1}/{n_boot} oracle-threshold {diffs[-1]:+.4f}")
    return diffs, thr_boot


def write_result(out, result):
    with open(os.path.join(out, "result.json"), "w") as f:
        json.dump(result, f, indent=1)
    return os.path.join(out, "result.json")


def load_json(path):
    with open(path) as f:
        return json.load(f)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pool", required=True)
    ap.add_argument("--seeds", type=int, default=8)
    ap.add_argument("--boot_a", type=int, default=B_A)
    ap.add_argument("--boot_b", type=int, default=B_B)
    ap.add_argument("--boot_seeds", type=int, default=4)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    assert args.seeds == 8 and args.boot_seeds == 4 and args.boot_a == 100 and args.boot_b == 200, \
        "spec: seeds 0-7, bootstrap seeds 0-3, design A B=100, design B B=200"
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
    m = n // 2
    l3s = load_json(L3S_RESULT)
    log(f"L3aa pool {args.pool}: groups={n} proxy_errors={n_err} step={STEP} steps={math.ceil(n / STEP)} "
        f"budget={budget} verifier_calls_per_run={round(budget * n)}")
    log("oracle_tiebreak = OracleTieBreak (CEILING ONLY): block order (agreement asc, n_distinct desc), then proxy_error desc, "
        "stable; no regression, no noise, no cells, gold order from step 0")
    log("PRE-REGISTERED READING (written into the log before the run):")
    log(f"  {READING_RULE}")

    thr_rows = [run_arm(pool, "threshold", budget, s) for s in seeds]
    thr_ok, thr_checks = check_threshold_vs_l3s(thr_rows, l3s)
    log(f"threshold vs spec and runs/L3s per seed: {'PASS' if thr_ok else 'FAIL'}; caught {[r['caught'] for r in thr_rows]}")

    if not thr_ok:
        reading = "INVALID_CONTROL"
        log("INVALID_CONTROL: the threshold control did not reproduce. The oracle arm is not run and no oracle number is read.")
        result = {"id": "L3aa", "pool": args.pool, "checks": {"threshold_vs_spec_and_l3s": thr_checks, "controls_pass": False},
                  "main_25": {"threshold": arm_stats(thr_rows)}, "reading_rule": READING_RULE, "reading": reading,
                  "metric": None, "wall_s": round(time.time() - t0, 1)}
        path = write_result(args.out, result)
        log(f"reading {reading}; wrote {path}")
        return 3

    ora_rows = [run_arm(pool, "oracle_tiebreak", budget, s) for s in seeds]
    assert [r["calls"] for r in ora_rows] == [r["calls"] for r in thr_rows], "equal calls across arms"
    ceiling_ok = all(o["caught"] >= t["caught"] for o, t in zip(ora_rows, thr_rows))
    assert ceiling_ok, "oracle caught fewer errors than threshold on some seed: the ceiling property is violated"
    st_t, st_o = arm_stats(thr_rows), arm_stats(ora_rows)
    diag = {"changed_steps_per_seed": [r["changed_steps"] for r in ora_rows],
            "changed_steps_total_seeds_0_to_7": int(sum(r["changed_steps"] for r in ora_rows)),
            "compared_steps_per_run": ora_rows[0]["compared_steps"],
            "steps_per_run": math.ceil(n / STEP)}
    log(f"  oracle caught per seed {st_o['caught_per_seed']} mean {st_o['caught_mean']:.3f} "
        f"admitted mean {st_o['admitted_mean']:.4f}")
    log(f"diagnostic (no reading effect): steps where the oracle verified set differs from the threshold top-k "
        f"{diag['changed_steps_per_seed']} of {diag['compared_steps_per_run']} per run")

    main_diff = [o["admitted"] - t["admitted"] for o, t in zip(ora_rows, thr_rows)]
    main_caught_diff = [o["caught"] - t["caught"] for o, t in zip(ora_rows, thr_rows)]
    main_mean_8 = float(np.mean(main_diff))
    point = float(np.mean(main_diff[:bs]))
    reading = "CAPPED" if main_mean_8 > READING_THRESHOLD else "HEADROOM"
    log(f"paired (oracle - threshold) admitted: seeds 0-7 mean {main_mean_8:+.4f}; seeds 0-{bs - 1} point {point:+.4f}; "
        f"caught diff per seed {main_caught_diff}")
    log(f"reading (fixed rule, diagnostic): {reading}")

    log(f"\nDesign A: B={args.boot_a} full-pool resamples WITH replacement (default_rng(10000+b)), seeds 0-{bs - 1}, 750 calls")
    diffs_a, _ = bootstrap(pool, budget, list(range(bs)), args.boot_a, True, n, "bootA", log)
    log(f"\nDesign B: B={args.boot_b} subsamples of {m} groups WITHOUT replacement (default_rng(10000+b)), "
        f"seeds 0-{bs - 1}, 375 calls")
    diffs_b, thr_b = bootstrap(pool, budget, list(range(bs)), args.boot_b, False, m, "bootB", log)

    sA = summarize(diffs_a, point)
    sB = summarize(diffs_b, point)
    U_A = sA["ci95_percentile"][1]
    draws_match = bool(np.allclose(thr_b, l3s["metric"]["threshold_subsample_mean_seeds_0_to_3"], rtol=0, atol=1e-12))
    draws_nonpositive = bool(max(diffs_a) <= 0 and max(diffs_b) <= 0)
    log(f"design A: U_A (97.5th percentile, per-resample denominator) {U_A:+.4f}; "
        f"95% percentile CI [{sA['ci95_percentile'][0]:+.4f}, {U_A:+.4f}]; resamples below 0: {sA['n_below_zero']}/{args.boot_a}")
    log(f"design B: percentile CI [{sB['ci95_percentile'][0]:+.4f}, {sB['ci95_percentile'][1]:+.4f}]; "
        f"basic CI [{sB['ci95_basic'][0]:+.4f}, {sB['ci95_basic'][1]:+.4f}]; bias {sB['bias']:+.4f}; "
        f"resamples below 0: {sB['n_below_zero']}/{args.boot_b}")
    log(f"design B threshold draws match runs/L3s threshold subsample means: {draws_match}")
    log(f"all bootstrap draws non-positive (ceiling property): {draws_nonpositive}")
    log(f"caught mean seeds 0-7: oracle {st_o['caught_mean']:.3f}, threshold {st_t['caught_mean']:.3f}")
    log(f"reading: {reading} (D = {main_mean_8:+.4f}, threshold -0.005)")

    result = {
        "id": "L3aa",
        "pool": args.pool,
        "n_groups": n,
        "n_proxy_errors": n_err,
        "step_size_groups": STEP,
        "budget": budget,
        "verifier_calls_per_run": round(budget * n),
        "design_B_subsample_size": m,
        "seeds_main": args.seeds,
        "seeds_bootstrap": bs,
        "B_design_A": args.boot_a,
        "B_design_B": args.boot_b,
        "assumptions": ASSUMPTIONS,
        "checks": {
            "threshold_vs_spec_and_l3s": thr_checks,
            "controls_pass": True,
            "verifier_calls_equal_across_arms": True,
            "oracle_caught_ge_threshold_each_seed": ceiling_ok,
            "all_bootstrap_draws_nonpositive": draws_nonpositive,
            "design_B_threshold_draws_match_l3s": draws_match,
        },
        "main_25": {"threshold": st_t, "oracle_tiebreak": st_o},
        "oracle_diagnostic": diag,
        "metric": {
            "name": "admitted rate at 25% for oracle_tiebreak minus threshold, paired over seeds 0-7 at 750 calls "
                    "(main_mean_admitted_diff_seeds_0_to_7); design A U_A (97.5th percentile, with replacement, B=100, seeds 0-3, "
                    "per-resample denominator) and design B percentile and basic upper bounds and bias (without replacement, "
                    "1500-group subsamples, B=200, seeds 0-3, per-subsample denominator)",
            "budget": budget,
            "main_mean_admitted_diff_seeds_0_to_7": main_mean_8,
            "main_caught_diff_per_seed_0_to_7": main_caught_diff,
            "main_caught_diff_mean_seeds_0_to_7": float(np.mean(main_caught_diff)),
            "point_estimate_seeds_0_to_3": point,
            "U_A": U_A,
            "reading_threshold": READING_THRESHOLD,
            "reading": reading,
            "design_A": {"B": args.boot_a, "replacement": True, "denominator": "per-resample errors", "seeds": bs,
                         "per_resample_diff": diffs_a, **sA},
            "design_B": {"B": args.boot_b, "subsample_size": m, "replacement": False,
                         "denominator": "per-subsample errors", "seeds": bs,
                         "per_subsample_diff": diffs_b, "threshold_subsample_mean_seeds_0_to_3": thr_b, **sB},
        },
        "reading_rule": READING_RULE,
        "reading": reading,
        "wall_s": round(time.time() - t0, 1),
    }
    path = write_result(args.out, result)
    log(f"\nwrote {path} (wall {result['wall_s']} s)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
