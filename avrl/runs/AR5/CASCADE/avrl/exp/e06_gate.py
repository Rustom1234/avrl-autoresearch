"""E06 gate readout (CPU, read-only on its inputs). For a run dir with groups.parquet and rollouts.parquet it returns:
the confident-subset group count n and proxy error count k at cutoff 0.75 (primary, the gate) and 0.625 (descriptive),
the rate k/n with a Wilson 95% interval, and the truncated-rollout share (mean of the truncated column).
Usage: e06_gate.py <run_dir> [out.json]. Reproduction, must match the logged values: avrl/runs/p0 (E01), avrl/runs/E10 (E10)."""
import json, math, os, sys
import pandas as pd

Z = 1.959963984540054


def wilson95(k, n):
    if n == 0:
        return [float("nan"), float("nan")]
    p, z2 = k / n, Z * Z
    centre = (p + z2 / (2 * n)) / (1 + z2 / n)
    half = Z * math.sqrt(p * (1 - p) / n + z2 / (4 * n * n)) / (1 + z2 / n)
    return [centre - half, centre + half]


def gate(run_dir):
    G = pd.read_parquet(os.path.join(run_dir, "groups.parquet"))
    R = pd.read_parquet(os.path.join(run_dir, "rollouts.parquet"))
    out = {"run_dir": os.path.abspath(run_dir), "n_groups": int(len(G)), "n_rollouts": int(len(R))}
    for cut in (0.75, 0.625):
        c = G[G.agreement >= cut]
        n, k = int(len(c)), int(c.proxy_error.sum())
        lo, hi = wilson95(k, n)
        out[f"cut_{cut}"] = {"n": n, "k": k, "rate": k / n if n else float("nan"), "wilson95": [lo, hi]}
    out["truncated_rollouts"] = int(R.truncated.sum())
    out["truncated_share"] = float(R.truncated.mean())
    return out


if __name__ == "__main__":
    res = gate(sys.argv[1])
    print(json.dumps(res, indent=1))
    if len(sys.argv) > 2:
        json.dump(res, open(sys.argv[2], "w"), indent=1)
