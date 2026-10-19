"""L16b independent check (CPU only, pandas, separate from the readout's numpy code).

Recomputes from avrl/runs/L14b/pool_feats.parquet: the sim pool (cell100 buckets with at least 3 groups, seed-0 fit quarter
removed), the 25% threshold caught mean over 8 tie-break seeds, and the bucket_oracle caught count. Compares both, and the D
point, with avrl/runs/L16b/result.json to 1e-9. Checks the sanity ceiling (D <= threshold admitted minus the perfect-label floor).
Writes avrl/runs/L16b/check.json and exits 1 on any mismatch or ceiling violation.
"""
import json
import sys

import numpy as np
import pandas as pd

CASCADE = "/homes/rdubash/avrl_live/CASCADE"
FEATS = f"{CASCADE}/avrl/runs/L14b/pool_feats.parquet"
RESULT = f"{CASCADE}/avrl/runs/L16b/result.json"
OUT = f"{CASCADE}/avrl/runs/L16b/check.json"
G, FRAC, SEEDS, TOL = "cell100", 0.25, 8, 1e-9

df = pd.read_parquet(FEATS)
df = df[df[G].map(df[G].value_counts()) >= 3].reset_index(drop=True)
groups = df[G].unique()
np.random.default_rng(0).shuffle(groups)
fit = set(groups[: len(groups) // 4])
sim = df[~df[G].isin(fit)].reset_index(drop=True)

n = len(sim)
y = sim["proxy_error"].astype(int).to_numpy()
total = int(y.sum())
budget = int(FRAC * n)
n_buckets = int(sim[G].nunique())

thr = []
for s in range(SEEDS):
    tb = np.random.default_rng(s).random(n)
    t = pd.DataFrame({"agreement": sim["agreement"], "n_distinct": sim["n_distinct"], "tb": tb, "y": y})
    t = t.sort_values(["agreement", "n_distinct", "tb"], ascending=[True, False, True], kind="mergesort")
    thr.append(int(t["y"].to_numpy()[:budget].sum()))
thr_caught = float(np.mean(thr))

rate = sim.groupby(G, sort=False)["proxy_error"].mean()
order = pd.DataFrame({"rate": rate.to_numpy(), "appear": np.arange(len(rate))}, index=rate.index)
order = order.sort_values(["rate", "appear"], ascending=[False, True])
pos = pd.Series(np.arange(len(order)), index=order.index)
t = sim.assign(_rank=sim[G].map(pos).to_numpy(), _pos=np.arange(n), _y=y)
t = t.sort_values(["_rank", "agreement", "n_distinct", "_pos"], ascending=[True, True, False, True])
oracle_caught = float(t["_y"].to_numpy()[:budget].sum())

res = json.load(open(RESULT))
p = res["point"]["0.25"]
thr_adm = 1 - thr_caught / total
oracle_adm = 1 - oracle_caught / total
d_indep = thr_adm - oracle_adm
ceiling = thr_adm - (1 - min(budget, total) / total)

pairs = {
    "n_sim": (n, res["pool"]["n_sim"]),
    "n_buckets": (n_buckets, res["pool"]["n_buckets"]),
    "total_errors": (total, res["pool"]["total_errors"]),
    "budget_docs": (budget, p["threshold"]["budget"]),
    "threshold_caught_mean": (thr_caught, p["threshold"]["caught_mean"]),
    "threshold_admitted": (thr_adm, p["threshold"]["admitted"]),
    "bucket_oracle_caught": (oracle_caught, p["bucket_oracle"]["caught_mean"]),
    "bucket_oracle_admitted": (oracle_adm, p["bucket_oracle"]["admitted"]),
    "D_bucket_oracle_25": (d_indep, res["D_point"]["0.25"]["bucket_oracle"]),
}
diffs = {k: abs(float(a) - float(b)) for k, (a, b) in pairs.items()}
matched = all(v <= TOL for v in diffs.values())
ceiling_ok = d_indep <= ceiling + TOL

out = {
    "exp": "L16b independent check",
    "sim_groups": n,
    "sim_buckets": n_buckets,
    "total_errors": total,
    "budget_docs": budget,
    "threshold_per_seed_caught": thr,
    "independent": {"threshold_caught_mean": thr_caught, "bucket_oracle_caught": oracle_caught,
                    "threshold_admitted": thr_adm, "bucket_oracle_admitted": oracle_adm, "D_bucket_oracle_25": d_indep},
    "result_json": {k: b for k, (_, b) in pairs.items()},
    "max_abs_diff": max(diffs.values()),
    "sanity_ceiling_D_25": ceiling,
    "ceiling_ok": ceiling_ok,
    "passed": matched and ceiling_ok,
}
with open(OUT, "w") as f:
    json.dump(out, f, indent=1)
print(json.dumps({"sim_groups": n, "sim_buckets": n_buckets, "total_errors": total, "max_abs_diff": out["max_abs_diff"],
                  "ceiling_ok": ceiling_ok, "passed": out["passed"]}))
sys.exit(0 if out["passed"] else 1)
