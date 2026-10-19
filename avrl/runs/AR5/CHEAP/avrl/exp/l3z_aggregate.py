#!/usr/bin/env python3
"""L3z aggregate: reads the 20 planted-pool result.json files, writes runs/L3z/summary.json (per pool plus aggregate) and
runs/L3z/result.json (metric values and the reading).

Pre-registered in avrl/RESEARCH_LOG.md, section "L3z 2026-10-19 pre-registration":
  PASS count out of 20 is the primary metric. bounds-only count (both upper bounds below 0, bias ignored) routes the reading.
  Achieved check: A = mean over the 20 pools of the seeds 0-7 paired admitted difference (greedy6s minus threshold).
  On target iff |A - (-0.03)| <= 0.005; otherwise OFF_TARGET and no reading is applied.
  R2 on target and PASS >= 10: usable for L3w under this design.
  R3 on target, PASS < 10, bounds-only >= 10: bias gate binding, next L3y.
  R4 on target, PASS < 10, bounds-only < 10: bounds underpowered, re-specify before GPU spend.
Denominator is always 20. INVALID_CONTROL pools count as not PASS and not bounds-only.
A missing result.json (crash or 45-minute cap) fails the aggregate; nothing is written in that case.
CPU only. No GPU, no pip, no ssh.
"""
import json
import math
import os
import sys

sys.dont_write_bytecode = True

import numpy as np  # noqa: E402

RUN = "/homes/rdubash/avrl_live/CHEAP/avrl/runs/L3z"
N_POOLS = 20
TARGET = -0.03
TARGET_TOL = 0.005
PASS_CUT = 10
BIAS_BIN = 0.003
ARMS = ("threshold", "greedy6u", "greedy6s")
VERDICTS = ("PASS", "KILL", "INCONCLUSIVE", "INVALID_CONTROL")


def load_pool(k):
    path = os.path.join(RUN, f"planted_{k}", "result.json")
    if not os.path.exists(path):
        return {"k": k, "verdict": "MISSING"}
    with open(path) as f:
        r = json.load(f)
    main = r["main_25"]
    diffs = [g - t for g, t in zip(main["greedy6s"]["admitted_per_seed"], main["threshold"]["admitted_per_seed"])]
    det = {arm: bool(r["checks"]["same_pool_determinism"][arm]["rerun_equals_main"]) for arm in ARMS}
    row = {"k": k, "verdict": r["verdict"], "pool_seed": r.get("pool_seed"), "strength": r.get("strength"),
           "achieved_diff_seeds_0_to_7": float(np.mean(diffs)), "determinism": det, "wall_s": r.get("wall_s")}
    if r["verdict"] == "INVALID_CONTROL":
        return row
    b = r["metric"]["bootstrap"]
    row.update({
        "point_estimate_seeds_0_to_3": r["metric"]["point_estimate_seeds_0_to_3"],
        "bootstrap_mean": b["mean"],
        "bias": b["bias"],
        "bootstrap_sd": b["sd"],
        "pct_upper": b["ci95_percentile"][1],
        "basic_upper": b["ci95_basic"][1],
    })
    return row


def reading_for(on_target, pass_n, bounds_only):
    if not on_target:
        return "R1 OFF_TARGET: no reading applies; a new pre-registration is needed before any further run"
    if pass_n >= PASS_CUT:
        return "R2 USABLE_FOR_L3W: PASS >= 10 of 20 at the achieved target (L3w still waits for the owed L3v calibration re-run and the bias reading)"
    if bounds_only >= PASS_CUT:
        return "R3 BIAS_GATE_BINDING: PASS < 10 and bounds-only >= 10; next L3y with a pre-registered alternative bias reference"
    return "R4 BOUNDS_UNDERPOWERED: PASS < 10 and bounds-only < 10; re-specify before any GPU spend"


def main():
    pools = [load_pool(k) for k in range(N_POOLS)]
    missing = [p["k"] for p in pools if p["verdict"] == "MISSING"]
    if missing:
        print(f"FAILED: no result.json for planted pools {missing} (crash or 45-minute cap). Nothing written.")
        return 1

    counts = {v: sum(p["verdict"] == v for p in pools) for v in VERDICTS}
    valid = [p for p in pools if p["verdict"] != "INVALID_CONTROL"]
    n_valid = len(valid)
    pass_n = counts["PASS"]
    bounds_only = sum(1 for p in valid if p["pct_upper"] < 0 and p["basic_upper"] < 0)
    bias = np.array([p["bias"] for p in valid], dtype=np.float64)
    mean_bias = float(bias.mean())
    sd_bias = float(bias.std(ddof=1))
    se_bias = sd_bias / math.sqrt(n_valid)
    achieved = np.array([p["achieved_diff_seeds_0_to_7"] for p in pools], dtype=np.float64)
    A = float(achieved.mean())
    A_sd = float(achieved.std(ddof=1))
    on_target = abs(A - TARGET) <= TARGET_TOL
    det_counts = {arm: sum(p["determinism"][arm] for p in pools) for arm in ARMS}
    all_det = sum(all(p["determinism"].values()) for p in pools)
    reading = reading_for(on_target, pass_n, bounds_only)

    metric = {
        "name": "L3z planted-effect power of the unchanged L3s gate at achieved greedy6s minus threshold = -0.03 (20 planted pools)",
        "strength": pools[0].get("strength"),
        "target": TARGET,
        "target_tolerance": TARGET_TOL,
        "achieved_mean_diff_seeds_0_to_7": A,
        "achieved_sd_across_pools": A_sd,
        "on_target": on_target,
        "n_pools": N_POOLS,
        "n_valid_pools": n_valid,
        "pass_count": pass_n,
        "bounds_only_count": bounds_only,
        "counts": counts,
        "mean_bias": mean_bias,
        "sd_bias": sd_bias,
        "se_mean_bias": se_bias,
        "bias_tolerance": BIAS_BIN,
        "pools_abs_bias_below_tol": int(sum(abs(p["bias"]) < BIAS_BIN for p in valid)),
        "determinism_pools_all_arms_reproduce": all_det,
        "determinism_pools_per_arm": det_counts,
        "reading": reading,
    }
    summary = {"id": "L3z", "pre_registration": "avrl/RESEARCH_LOG.md, section 'L3z 2026-10-19 pre-registration'",
               "reading": reading, "metric": metric, "per_pool": pools}
    result = {"id": "L3z", "reading": reading, "metric": metric, "summary": os.path.join(RUN, "summary.json")}
    with open(os.path.join(RUN, "summary.json"), "w") as f:
        json.dump(summary, f, indent=1)
    with open(os.path.join(RUN, "result.json"), "w") as f:
        json.dump(result, f, indent=1)

    print(f"{'pool':>5} {'seed':>6} {'verdict':>14} {'bias':>10} {'pct_upper':>11} {'basic_upper':>12} {'achieved':>10} {'det':>5}")
    for p in pools:
        det_ok = all(p["determinism"].values())
        if p["verdict"] == "INVALID_CONTROL":
            print(f"{p['k']:>5} {p['pool_seed']:>6} {p['verdict']:>14} {'':>10} {'':>11} {'':>12} "
                  f"{p['achieved_diff_seeds_0_to_7']:+10.5f} {str(det_ok):>5}")
            continue
        print(f"{p['k']:>5} {p['pool_seed']:>6} {p['verdict']:>14} {p['bias']:+10.5f} {p['pct_upper']:+11.5f} "
              f"{p['basic_upper']:+12.5f} {p['achieved_diff_seeds_0_to_7']:+10.5f} {str(det_ok):>5}")
    print(f"valid pools {n_valid}; mean bias {mean_bias:+.5f}; SD {sd_bias:.5f}; SE {se_bias:.5f}; "
          f"|bias|<{BIAS_BIN} in {metric['pools_abs_bias_below_tol']} pools")
    print(f"counts PASS {counts['PASS']} KILL {counts['KILL']} INCONCLUSIVE {counts['INCONCLUSIVE']} "
          f"INVALID_CONTROL {counts['INVALID_CONTROL']}")
    print(f"PASS count {pass_n}/20; bounds-only count {bounds_only}/20")
    print(f"achieved mean diff (greedy6s minus threshold, seeds 0-7) A = {A:+.5f} (SD across pools {A_sd:.5f}); "
          f"target {TARGET:+.3f} +/- {TARGET_TOL} -> on_target={on_target}")
    print(f"determinism: pools with all arms reproducing {all_det}/20; per arm {det_counts}")
    print(f"L3z reading: {reading}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
