"""L10a verification (CPU, read-only): checks the stored numbers against the raw E10 replay file and against an
independent recompute from avrl/runs/E10/feats.parquet (no import of the L10a or e03 script).
Writes avrl/runs/L10a/verify.json."""
import json
import numpy as np
import pandas as pd

BASE = "/homes/rdubash/avrl_live/CASCADE/avrl/runs"
prim = json.load(open(f"{BASE}/L10a/result.json"))
ctrl = json.load(open(f"{BASE}/L10a/control_no_e01/result.json"))
e10 = json.load(open(f"{BASE}/E10/replay/result.json"))
out = {}

# 1. Control (feature omitted) must reproduce the E10 replay numbers.
out["control_vs_E10_replay"] = {
    "max_abs_diff_point_caught_mean": max(abs(ctrl["point"][b][a]["caught_mean"] - e10["point"][b][a]["caught_mean"])
                                          for b in ctrl["point"] for a in ctrl["point"][b]),
    "max_abs_diff_D_point": max(abs(ctrl["D_point"][b][a] - e10["D_point"][b][a]) for b in ctrl["D_point"] for a in ctrl["D_point"][b]),
    "max_abs_diff_boot_draws": max(float(np.max(np.abs(np.array(ctrl["D_bootstrap_draws"][a]) - np.array(e10["D_bootstrap_draws"][a]))))
                                   for a in ["ts_online", "greedy_online"]),
    "E10_D_ts_25_point_and_ci": [e10["D_point"]["0.25"]["ts_online"], e10["D_bootstrap"]["ts_online"]["ci95"]],
}

# 2. Primary D recomputed from the stored caught means and total errors.
tot = prim["pool"]["total_errors"]
pt = prim["point"]["0.25"]
D_re = {a: (1 - pt["threshold"]["caught_mean"] / tot) - (1 - pt[a]["caught_mean"] / tot) for a in ["ts_online", "greedy_online"]}
out["primary_D25_recomputed"] = D_re
out["primary_D25_stored"] = {a: prim["D_point"]["0.25"][a] for a in D_re}

# 3. Bootstrap CI and share above 0 recomputed from the 200 stored draws.
for name, r in [("primary", prim), ("control", ctrl)]:
    for a in ["ts_online", "greedy_online"]:
        d = np.array(r["D_bootstrap_draws"][a])
        out[f"{name}_{a}_boot"] = {"n": int(len(d)), "ci95_recomputed": [float(np.percentile(d, 2.5)), float(np.percentile(d, 97.5))],
                                   "ci95_stored": r["D_bootstrap"][a]["ci95"], "share_gt0": float((d > 0).mean()),
                                   "max": float(d.max()), "min": float(d.min())}

# 4. Independent threshold recompute at 25% from the raw E10 feats (own code, same split rule as e03).
df = pd.read_parquet(f"{BASE}/E10/feats.parquet")
cnt = df["cell100"].value_counts()
df = df[df["cell100"].isin(cnt[cnt >= 3].index)].reset_index(drop=True)
rng0 = np.random.default_rng(0)
groups = df["cell100"].unique()
rng0.shuffle(groups)
fit_g = set(groups[: len(groups) // 4])
sim = df[~df["cell100"].isin(fit_g)].reset_index(drop=True)
y = sim["proxy_error"].values.astype(int)
ag, nd = sim["agreement"].values, sim["n_distinct"].values
budget = int(0.25 * len(sim))
caught = []
for s in range(8):
    tb = np.random.default_rng(s).random(len(sim))
    order = np.lexsort((tb, -nd, ag))
    caught.append(int(y[order[:budget]].sum()))
out["independent_threshold_25"] = {"n_sim": len(sim), "budget": budget, "total_errors": int(y.sum()), "caught_per_seed": caught,
                                   "caught_mean": float(np.mean(caught)), "stored_caught_mean": pt["threshold"]["caught_mean"],
                                   "stored_total_errors": tot, "stored_budget": pt["threshold"]["budget"]}

# 5. Feature list and E01 join checks on the raw file.
out["feature_counts"] = {"primary_n_features": prim["config"]["n_features"], "control_n_features": ctrl["config"]["n_features"],
                         "primary_has_e01": "e01_proxy_error" in prim["config"]["feature_list"],
                         "control_has_e01": "e01_proxy_error" in ctrl["config"]["feature_list"]}
e01 = pd.read_parquet(f"{BASE}/p0/feats.parquet", columns=["pid", "proxy_error"])
out["e01_proxy_error_mean_raw"] = float(e01["proxy_error"].mean())
out["e01_proxy_error_sum_raw"] = int(e01["proxy_error"].sum())

with open(f"{BASE}/L10a/verify.json", "w") as f:
    json.dump(out, f, indent=1, default=float)
print(json.dumps(out, indent=1, default=float))
