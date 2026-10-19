#!/usr/bin/env python3
"""L3v aggregate: reads the 20 null-pool result.json files, writes runs/L3v/summary.json (per pool plus aggregate) and
runs/L3v/result.json (metric values and verdict).

Bins are the ones pre-registered in avrl/RESEARCH_LOG.md (L3v pre-registration, written before launch):
  bias bin: calibrated if |mean null bias| < 0.003, miscalibrated otherwise.
  PASS bin: 0 null PASS is the calibration outcome; exactly 1 triggers a rerun with more null pools (not a rule change);
            2 or more is too permissive.
  CALIBRATED iff the bias bin is calibrated, n_pass = 0, and no pool is INVALID_CONTROL.
A missing result.json (crash or 45-minute cap) fails the aggregate; nothing is written in that case.
CPU only. No GPU, no pip, no ssh.
"""
import json
import math
import os
import sys

sys.dont_write_bytecode = True

import numpy as np  # noqa: E402

RUN = "/homes/rdubash/avrl_live/CHEAP/avrl/runs/L3v"
N_POOLS = 20
BIAS_BIN = 0.003
VERDICTS = ("PASS", "KILL", "INCONCLUSIVE", "INVALID_CONTROL")


def load_pool(k):
    path = os.path.join(RUN, f"null_{k}", "result.json")
    if not os.path.exists(path):
        return {"k": k, "verdict": "MISSING"}
    with open(path) as f:
        r = json.load(f)
    row = {"k": k, "verdict": r["verdict"], "wall_s": r.get("wall_s")}
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


def main():
    pools = [load_pool(k) for k in range(N_POOLS)]
    missing = [p["k"] for p in pools if p["verdict"] == "MISSING"]
    if missing:
        print(f"FAILED: no result.json for null pools {missing} (crash or 45-minute cap). Nothing written.")
        return 1

    counts = {v: sum(p["verdict"] == v for p in pools) for v in VERDICTS}
    valid = [p for p in pools if p["verdict"] != "INVALID_CONTROL"]
    n = len(valid)
    bias = np.array([p["bias"] for p in valid], dtype=np.float64)
    mean_bias = float(bias.mean())
    sd_bias = float(bias.std(ddof=1))
    se_bias = sd_bias / math.sqrt(n)
    n_pct_below0 = sum(p["pct_upper"] < 0 for p in valid)
    n_pass = counts["PASS"]

    bias_ok = abs(mean_bias) < BIAS_BIN
    flags = []
    if not bias_ok:
        flags.append("MISCALIBRATED_BIAS")
    if n_pass >= 2:
        flags.append("TOO_PERMISSIVE")
    if n_pass == 1:
        flags.append("RERUN_MORE_NULL_POOLS")
    if counts["INVALID_CONTROL"]:
        flags.append("INVALID_CONTROL_POOLS")
    overall = "CALIBRATED" if not flags else " + ".join(flags)

    metric = {
        "name": "L3v null calibration of the L3s design: per-pool bias = seeds 0-3 bootstrap mean minus the seeds 0-3 "
                "point (greedy6s minus threshold admitted at 25%), aggregated over 20 label-independent null pools",
        "n_pools": N_POOLS,
        "n_valid_pools": n,
        "mean_bias": mean_bias,
        "sd_bias": sd_bias,
        "se_mean_bias": se_bias,
        "bias_tolerance": BIAS_BIN,
        "counts": counts,
        "n_pct_upper_below_0": n_pct_below0,
        "share_pct_upper_below_0": n_pct_below0 / n,
        "bins": {"bias_calibrated": bias_ok, "n_pass": n_pass, "flags": flags},
    }
    summary = {
        "id": "L3v",
        "pre_registration": "avrl/RESEARCH_LOG.md, section 'L3v 2026-10-18 pre-registration'",
        "verdict": overall,
        "metric": metric,
        "per_pool": pools,
    }
    result = {"id": "L3v", "verdict": overall, "metric": metric, "summary": os.path.join(RUN, "summary.json")}
    with open(os.path.join(RUN, "summary.json"), "w") as f:
        json.dump(summary, f, indent=1)
    with open(os.path.join(RUN, "result.json"), "w") as f:
        json.dump(result, f, indent=1)

    print(f"{'pool':>6} {'verdict':>14} {'bias':>10} {'pct_upper':>11} {'basic_upper':>12} {'point':>10}")
    for p in pools:
        if p["verdict"] == "INVALID_CONTROL":
            print(f"{p['k']:>6} {p['verdict']:>14}")
            continue
        print(f"{p['k']:>6} {p['verdict']:>14} {p['bias']:+10.5f} {p['pct_upper']:+11.5f} {p['basic_upper']:+12.5f} "
              f"{p['point_estimate_seeds_0_to_3']:+10.5f}")
    print(f"valid pools {n}; mean bias {mean_bias:+.5f}; SD {sd_bias:.5f}; SE {se_bias:.5f}")
    print(f"counts PASS {counts['PASS']} KILL {counts['KILL']} INCONCLUSIVE {counts['INCONCLUSIVE']} "
          f"INVALID_CONTROL {counts['INVALID_CONTROL']}")
    print(f"share of pools with percentile upper bound below 0: {n_pct_below0}/{n} = {n_pct_below0 / n:.3f} "
          f"(pre-registered calibrated rate 0.025)")
    print(f"L3v overall: {overall}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
