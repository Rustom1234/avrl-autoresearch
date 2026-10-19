"""L5a check: independent recompute of the frozen table and coverage, run before the verdict is logged. CPU only.

Recomputes the p0 frozen table with plain pandas (no L1B import), compares it with runs/L5a/result.json, and reports
how much of each eval pool the table labels. Also checks for prompt-text overlap between p0 and the eval pools.
Usage: /homes/rdubash/active-audit/.venv/bin/python -I /homes/rdubash/avrl_live/ROUTED/avrl/exp/l5a_verify.py
"""
import json
import numpy as np
import pandas as pd

R = "/homes/rdubash/avrl_live/ROUTED/avrl/runs"
S = ["agreement", "n_distinct", "any_truncated"]
K0 = 50.0

p0 = pd.read_parquet(f"{R}/p0/feats.parquet")
L2 = pd.read_parquet(f"{R}/L2b/feats.parquet")
L4 = pd.read_parquet(f"{R}/L4b/feats.parquet")
print("rows: p0", len(p0), "L2b", len(L2), "L4b", len(L4))
print("p0 proxy_error sum", int(p0["proxy_error"].sum()))
print("p0 columns", list(p0.columns))

y = p0["proxy_error"].astype(int)
g = p0.assign(_y=y).groupby(S)["_y"].agg(["sum", "size"]).reset_index()
g = g.rename(columns={"sum": "errors", "size": "count"})
g["prior"] = 1 - g["agreement"]
g["post"] = (g["errors"] + K0 * g["prior"]) / (g["count"] + K0)
print("strata", len(g), "count sum", int(g["count"].sum()), "error sum", int(g["errors"].sum()))

tab = pd.DataFrame(json.load(open(f"{R}/L5a/result.json"))["table"])
m = g.merge(tab, on=S, suffixes=("", "_r"), how="outer", indicator=True)
print("table merge status", m["_merge"].value_counts().to_dict())
print("max abs post diff", float(np.abs(m["post"] - m["post_r"]).max()),
      "count match", bool((m["count"] == m["count_r"]).all()),
      "error match", bool((m["errors"] == m["errors_r"]).all()))

for name, d in [("L2b", L2), ("L4b", L4)]:
    k = d[S].merge(g[S], on=S, how="left", indicator=True)
    print(name, "docs in p0 table strata", round(float((k["_merge"] == "both").mean()), 4))

for col in ["question", "prompt", "question_text", "problem", "input"]:
    if col in p0.columns and col in L2.columns and col in L4.columns:
        a = set(p0[col])
        print(col, "overlap p0-L2b", len(a & set(L2[col])), "p0-L4b", len(a & set(L4[col])))
