"""L17d independent check (CPU only, pandas; separate from the readout's numpy code).

Recomputes from avrl/runs/L14b/pool_feats.parquet: the sim pool (cell100 buckets with at least 3 groups, seed-0 fit quarter
removed); the three static logistic models, refit with sklearn on the fit quarter with the L14b settings; the caught counts at
25% and 50% for the three static arms and the threshold (mean over 8 tie-break seeds, pandas sort); the point values of the five
L17d metrics; and the 200 bucket-cluster and 200 group bootstrap draws of the five metrics at 25% and 50%. The resampled frames
are built with pandas (concat and a stable sort on the source row). Compares everything with avrl/runs/L17d/result.json to
1e-9 and writes avrl/runs/L17d/check.json. Exits 1 on any mismatch.
"""
import json
import sys

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

CASCADE = "/homes/rdubash/avrl_live/CASCADE"
FEATS = f"{CASCADE}/avrl/runs/L14b/pool_feats.parquet"
RESULT = f"{CASCADE}/avrl/runs/L17d/result.json"
OUT = f"{CASCADE}/avrl/runs/L17d/check.json"
G, LABEL, SEEDS, TOL, NBOOT = "cell100", "proxy_error", 8, 1e-9, 200
CLUSTER_SEED, BOOT_SEED = 123, 0
CHEAP = ["agreement", "n_distinct", "mean_tokens", "std_tokens", "any_truncated", "log_qlen"]
PCS = [f"pc{i}" for i in range(16)]
SETS = {"static": CHEAP + PCS, "static_cheap6": CHEAP, "static_pc": PCS}
METRICS = ["Delta_cheap6_minus_cheapPC", "Delta_pc_minus_cheapPC", "D_static_cheapPC", "D_static_cheap6", "D_static_pc"]

# Split: same rule as the readout (buckets with >= 3 groups, seed-0 shuffle of the bucket values, first quarter = fit).
df = pd.read_parquet(FEATS)
df = df[df[G].map(df[G].value_counts()) >= 3].reset_index(drop=True)
groups = df[G].unique()
np.random.default_rng(0).shuffle(groups)
fit_g = set(groups[: len(groups) // 4])
fit = df[df[G].isin(fit_g)]
sim = df[~df[G].isin(fit_g)].reset_index(drop=True)

n = len(sim)
y = sim[LABEL].to_numpy().astype(int)
total = int(y.sum())
n_b = int(sim[G].nunique())
codes, uniq = pd.factorize(sim[G])  # bucket index in order of first appearance, as the readout's make_pool
assert len(uniq) == n_b
budgets = {"0.25": int(0.25 * n), "0.5": int(0.5 * n)}  # fixed budgets, as in the readout

# Static models: refit here with sklearn on the fit quarter (labels from the fit split only).
for arm, cols in SETS.items():
    clf = make_pipeline(StandardScaler(), LogisticRegression(max_iter=1000, class_weight="balanced"))
    clf.fit(fit[cols].fillna(0).values, fit[LABEL])
    sim["s_" + arm] = clf.predict_proba(sim[cols].fillna(0).values)[:, 1]


def threshold_caught(frame, yv, budget_list):
    """Mean over tie-break seeds of the caught count at each budget. Order: agreement asc, n_distinct desc, seeded random tie-break."""
    acc = {b: [] for b in budget_list}
    for s in range(SEEDS):
        t = pd.DataFrame({"agreement": frame["agreement"].values, "n_distinct": frame["n_distinct"].values,
                          "tb": np.random.default_rng(s).random(len(frame)), "y": yv})
        t = t.sort_values(["agreement", "n_distinct", "tb"], ascending=[True, False, True])
        for b in budget_list:
            acc[b].append(int(t["y"].to_numpy()[:b].sum()))
    return {b: float(np.mean(v)) for b, v in acc.items()}


def static_caught(score, yv, b):
    return float(yv[np.argsort(-score)[:b]].sum())


def metrics(adm):
    """adm: arm -> admitted. The five L17d metrics."""
    return {"Delta_cheap6_minus_cheapPC": adm["static_cheap6"] - adm["static"],
            "Delta_pc_minus_cheapPC": adm["static_pc"] - adm["static"],
            "D_static_cheapPC": adm["threshold"] - adm["static"],
            "D_static_cheap6": adm["threshold"] - adm["static_cheap6"],
            "D_static_pc": adm["threshold"] - adm["static_pc"]}


def evaluate(frame):
    """Return {budget key: {metric: value}} for a (possibly resampled) pool frame, with that frame's own error total."""
    yv = frame[LABEL].to_numpy().astype(int)
    tot = int(yv.sum())
    tc = threshold_caught(frame, yv, list(budgets.values()))
    out = {}
    for bk, b in budgets.items():
        adm = {"threshold": 1 - tc[b] / tot}
        for arm in SETS:
            adm[arm] = 1 - static_caught(frame["s_" + arm].values, yv, b) / tot
        out[bk] = metrics(adm)
    return out


# Point estimates on the sim pool.
point = {}
for bk, b in budgets.items():
    caught = {"threshold": threshold_caught(sim, y, [b])[b]}
    for arm in SETS:
        caught[arm] = static_caught(sim["s_" + arm].values, y, b)
    point[bk] = caught
adm_pt = {bk: {arm: 1 - point[bk][arm] / total for arm in point[bk]} for bk in budgets}
delta_pt = {bk: metrics(adm_pt[bk]) for bk in budgets}

# Bootstrap draws. Cluster: rng 123, one integers call per resample, bucket draws relabelled as new clusters, rows in source order.
rng_c = np.random.default_rng(CLUSTER_SEED)
cluster_draws = [rng_c.integers(0, n_b, size=n_b) for _ in range(NBOOT)]
rng_b = np.random.default_rng(BOOT_SEED)
group_draws = [np.sort(rng_b.integers(0, n, size=n)) for _ in range(NBOOT)]

simc = sim.assign(_row=np.arange(n), _code=codes)
by_code = {b: simc[simc["_code"] == b] for b in range(n_b)}


def cluster_frame(draw):
    return pd.concat([by_code[b] for b in draw]).sort_values("_row", kind="mergesort").reset_index(drop=True)


cl = {bk: {m: [] for m in METRICS} for bk in budgets}
for draw in cluster_draws:
    r = evaluate(cluster_frame(draw))
    for bk in budgets:
        for m in METRICS:
            cl[bk][m].append(r[bk][m])
gr = {bk: {m: [] for m in METRICS} for bk in budgets}
for idx in group_draws:
    r = evaluate(sim.iloc[idx].reset_index(drop=True))
    for bk in budgets:
        for m in METRICS:
            gr[bk][m].append(r[bk][m])

# Compare with the readout.
res = json.load(open(RESULT))
pairs = {"n_sim": (n, res["pool"]["n_sim"]), "n_buckets": (n_b, res["pool"]["n_buckets"]), "total_errors": (total, res["pool"]["total_errors"])}
for bk in budgets:
    pairs[f"budget/{bk}"] = (budgets[bk], res["point"][bk]["threshold"]["budget"])
    for arm in point[bk]:
        pairs[f"caught/{bk}/{arm}"] = (point[bk][arm], res["point"][bk][arm]["caught_mean"])
    for m in METRICS:
        pairs[f"Delta_point/{bk}/{m}"] = (delta_pt[bk][m], res["Delta_point"][bk][m])
        for i in range(NBOOT):
            pairs[f"cluster/{bk}/{m}/{i}"] = (cl[bk][m][i], res["static_bootstrap_cluster_draws"][bk][m][i])
            pairs[f"group/{bk}/{m}/{i}"] = (gr[bk][m][i], res["static_bootstrap_group_draws"][bk][m][i])
diffs = {k: abs(float(a) - float(b)) for k, (a, b) in pairs.items()}
worst = sorted(diffs.items(), key=lambda kv: -kv[1])[:5]
passed = max(diffs.values()) <= TOL and len(cl["0.25"]["D_static_cheapPC"]) == NBOOT

out = {
    "exp": "L17d independent check",
    "sim_groups": n,
    "sim_buckets": n_b,
    "total_errors": total,
    "budget_docs": budgets,
    "compared": len(pairs),
    "tolerance": TOL,
    "max_abs_diff": max(diffs.values()),
    "worst_5": worst,
    "independent_point_delta": delta_pt,
    "independent_caught": point,
    "passed": passed,
}
with open(OUT, "w") as f:
    json.dump(out, f, indent=1)
print(json.dumps({"sim_groups": n, "sim_buckets": n_b, "total_errors": total, "compared": len(pairs),
                  "max_abs_diff": out["max_abs_diff"], "passed": passed}))
sys.exit(0 if passed else 1)
