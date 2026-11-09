"""Independent recomputation of the L12b primary and sensitivity rho with a groupby, not the L12b script.

Read-only. Prints the numbers; writes nothing. Used to check the run report before the research log entry.
Usage: /homes/rdubash/active-audit/.venv/bin/python -I l12b_verify.py
"""
import json

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

ROOT = "/homes/rdubash/avrl_live/ROUTED/avrl"
P0 = f"{ROOT}/runs/p0/feats.parquet"
L4B = f"{ROOT}/runs/L4b/feats.parquet"
L13A = f"{ROOT}/runs/L13a"
S = ["agreement", "n_distinct", "any_truncated"]
CUT = 1318

p0 = pd.read_parquet(P0)
l4 = pd.read_parquet(L4B)
assert "proxy_error" in p0.columns and "proxy_error" in l4.columns, "label column missing"
assert set(S) <= set(p0.columns) and set(S) <= set(l4.columns), "stratum columns missing"

universe = set(p0.groupby(S).size().index)
test = l4[l4["pid"] <= CUT]
rem = l4[l4["pid"] > CUT]
print(f"p0 rows {len(p0)} strata {len(universe)}; l4b rows {len(l4)}; test {len(test)}; remaining {len(rem)}")
print(f"proxy error rate: test {test['proxy_error'].mean() * 100:.2f}%  remaining {rem['proxy_error'].mean() * 100:.2f}%")

gt = test.groupby(S)["proxy_error"].agg(["size", "mean"])
gr = rem.groupby(S)["proxy_error"].agg(["size", "mean"])
assert set(gt.index) <= universe and set(gr.index) <= universe, "row set has a stratum outside p0"
print(f"strata present: test {len(gt)}, remaining {len(gr)}, union {len(set(gt.index) | set(gr.index))}")
both = gt.index.intersection(gr.index)
print(f"strata on both sides: {len(both)}")
absent = sorted(universe - (set(gt.index) | set(gr.index)))
print(f"absent from L4b: {absent}")

rho_all = spearmanr(gt.loc[both, "mean"], gr.loc[both, "mean"]).correlation
print(f"PRIMARY rho (both sides, n={len(both)}): {rho_all:.4f}")

for m in (1, 5, 10, 20):
    idx = [i for i in both if gt.loc[i, "size"] >= m and gr.loc[i, "size"] >= m]
    if len(idx) >= 3:
        r = spearmanr(gt.loc[idx, "mean"], gr.loc[idx, "mean"]).correlation
        print(f"min docs per side >= {m:>2}: rho {r:.4f} over n={len(idx)}")
    else:
        print(f"min docs per side >= {m:>2}: fewer than 3 strata")

ref = {}
for side, d in (("test", "L4b_pid_le1318"), ("remaining", "L4b_pid_gt1318")):
    with open(f"{L13A}/{d}/result.json") as f:
        ref[side] = json.load(f)["pools"]["L4b"]["frozen_vs_fixed_mean_pp"]
print(f"L13a frozen minus fixed: test {ref['test']:+.4f} pp, remaining {ref['remaining']:+.4f} pp")

# Stratum-level check on the largest error gap, for the write-up.
d = (gt.loc[both, "mean"] - gr.loc[both, "mean"]).abs().sort_values(ascending=False)
print("largest absolute test vs remaining stratum gaps (top 5):")
for i in d.index[:5]:
    print(f"  {i}: test {gt.loc[i, 'mean']:.3f} (n={int(gt.loc[i, 'size'])}), remaining {gr.loc[i, 'mean']:.3f} (n={int(gr.loc[i, 'size'])})")

# Share of primary strata with fewer than 5 docs on either side.
small = sum(1 for i in both if min(gt.loc[i, "size"], gr.loc[i, "size"]) < 5)
print(f"primary strata with fewer than 5 docs on one side: {small} of {len(both)}")
