"""L11b assembly: checks against the L10a control, then the metric values and verdicts into avrl/runs/L11b/result.json.

Reads only the five L11b invocations and the L10a control (avrl/runs/L10a/control_no_e01/result.json). Exits non-zero
if any check fails. Verdict rules are the pre-registered ones (PASS, KILL, INCONCLUSIVE), applied in the copy script.
"""
import json
import sys

import numpy as np

ROOT = "/homes/rdubash/avrl_live/CASCADE"
OUT = f"{ROOT}/avrl/runs/L11b"
L10A_CONTROL = f"{ROOT}/avrl/runs/L10a/control_no_e01/result.json"
RUNS = ["control_cheap_emb", "primary", "ceiling", "ref_agree_nd", "ref_cheap_e01"]
BUDGETS = ["0.1", "0.25", "0.5"]
TOL = 1e-9

R = {n: json.load(open(f"{OUT}/{n}/result.json")) for n in RUNS}
L10A = json.load(open(L10A_CONTROL))
checks, failures = {}, []


def check(name, ok, detail):
    checks[name] = {"pass": bool(ok), "detail": detail}
    if not ok:
        failures.append(name)


# 1. Control reproduces L10a control exactly (greedy_online and threshold).
ctrl = R["control_cheap_emb"]
maxd = 0.0
for bf in BUDGETS:
    for arm in ["threshold", "greedy_online"]:
        a, b = L10A["point"][bf][arm], ctrl["point"][bf][arm]
        for k in ["caught_mean", "caught_sd", "admitted"]:
            maxd = max(maxd, abs(a[k] - b[k]))
    maxd = max(maxd, abs(L10A["D_point"][bf]["greedy_online"] - ctrl["D_point"][bf]["greedy_online"]))
da = np.array(L10A["D_bootstrap_draws"]["greedy_online"])
db = np.array(ctrl["D_bootstrap_draws"]["greedy_online"])
draw_d = float(np.abs(da - db).max())
ci_d = float(np.abs(np.array(L10A["D_bootstrap"]["greedy_online"]["ci95"]) - np.array(ctrl["D_bootstrap"]["greedy_online"]["ci95"])).max())
check("control_reproduces_L10a", maxd <= TOL and draw_d <= TOL and ci_d <= TOL and len(db) == 200,
      {"max_abs_diff_point_caught_adm_D": maxd, "max_abs_diff_greedy_draws": draw_d, "max_abs_diff_greedy_ci": ci_d,
       "n_draws": len(db), "L10a_D_greedy_25": L10A["D_point"]["0.25"]["greedy_online"],
       "L11b_D_greedy_25": ctrl["D_point"]["0.25"]["greedy_online"]})

# 2. Pool identical across the five invocations and equal to L10a.
pools = {n: (R[n]["pool"]["n_sim"], R[n]["pool"]["total_errors"], R[n]["pool"]["n_buckets"]) for n in RUNS}
pool_ok = all(v == (2180, 286, 75) for v in pools.values()) and L10A["pool"]["total_errors"] == 286
check("pool_matches_L10a", pool_ok, pools)

# 3. Uniform and threshold do not depend on the feature set or E01, so they must match across invocations.
same = True
for arm in ["uniform", "threshold"]:
    for bf in BUDGETS:
        vals = [R[n]["point"][bf][arm]["caught_mean"] for n in RUNS]
        same = same and max(vals) - min(vals) <= TOL
check("uniform_threshold_identical_across_runs", same, "caught means compared at 10/25/50% over all five invocations")

# 4. Hand rule identical in both E01 invocations (same rule, same seeds).
hand_ok = max(abs(R["ceiling"]["point"][bf]["hand"]["caught_mean"] - R["ref_cheap_e01"]["point"][bf]["hand"]["caught_mean"]) for bf in BUDGETS) <= TOL
check("hand_identical_in_e01_runs", hand_ok, "ceiling vs ref_cheap_e01, caught means at 10/25/50%")

# 5. Static per-seed fits: 8 fits, each with both classes (asserted in the copy), sizes logged.
fit = R["ceiling"]["static_fit_info"]
fit_ok = len(fit) == 8 and all(f["fit_errors"] > 0 and f["fit_docs"] - f["fit_errors"] > 0 for f in fit)
check("static_fits_both_classes", fit_ok, {"n_fits": len(fit), "fit_docs_range": [min(f["fit_docs"] for f in fit), max(f["fit_docs"] for f in fit)],
                                             "fit_errors_range": [min(f["fit_errors"] for f in fit), max(f["fit_errors"] for f in fit)]})

# 6. Bootstrap draw counts.
boot_ok = all(R[n]["config"]["boot_resamples"] == 200 and all(len(v) == 200 for v in R[n]["D_bootstrap_draws"].values()) for n in RUNS)
check("bootstrap_200_draws", boot_ok, "200 draws per gated and reference arm in every invocation")

# 7. E01 join: invocations with E01 flag 3,000 matched pids (asserted in the copy); no-E01 invocations have no e01 column.
check("e01_only_where_intended", R["primary"]["config"]["e01_feats"] is None and R["control_cheap_emb"]["config"]["e01_feats"] is None
      and R["ceiling"]["config"]["e01_feats"] and R["ref_cheap_e01"]["config"]["e01_feats"],
      "primary and control: no E01; ceiling and ref_cheap_e01: E01")

primary_cols = R["primary"]["config"]["feature_list"]
check("primary_has_no_e01_column", "e01_proxy_error" not in primary_cols and primary_cols == ["agreement", "n_distinct", "mean_tokens", "std_tokens", "any_truncated", "log_qlen"],
      primary_cols)


def row(run, arm, note):
    r = R[run]
    out = {"run": run, "arm": arm, "note": note,
           "caught_25": r["point"]["0.25"][arm]["caught_mean"], "caught_sd_25": r["point"]["0.25"][arm]["caught_sd"],
           "admitted_25": r["point"]["0.25"][arm]["admitted"],
           "D_point": {bf: r["D_point"][bf][arm] for bf in BUDGETS if arm in r["D_point"][bf]}}
    if arm in r["D_bootstrap"]:
        out["D_25_ci95"] = r["D_bootstrap"][arm]["ci95"]
        out["frac_draws_D_gt_0"] = r["D_bootstrap"][arm]["frac_resamples_D_gt_0"]
        out["status_25"] = r["status_at_boot_budget"][arm]
    return out


result = {
    "exp": "L11b",
    "sample": {"n_sim_docs": R["ceiling"]["pool"]["n_sim"], "total_errors": R["ceiling"]["pool"]["total_errors"],
               "budget_25": R["ceiling"]["point"]["0.25"]["uniform"]["budget"], "buckets": R["ceiling"]["pool"]["n_buckets"],
               "seeds": R["ceiling"]["config"]["seeds"], "boot_resamples": R["ceiling"]["config"]["boot_resamples"],
               "static_fit_buckets_per_seed": R["ceiling"]["static_fit_info"][0]["fit_buckets"],
               "static_fit_docs_range": [min(f["fit_docs"] for f in R["ceiling"]["static_fit_info"]),
                                         max(f["fit_docs"] for f in R["ceiling"]["static_fit_info"])]},
    "primary": row("primary", "greedy_online", "PRIMARY, label-free, greedy_online on cheap six, no E01"),
    "ceiling": row("ceiling", "static", "CEILING, labeled, static on cheap+16PCs+e01_proxy_error, refit per seed"),
    "reference": {
        "static_cheap_emb_no_e01": row("control_cheap_emb", "static", "reference: static on cheap+16PCs, no E01, refit per seed"),
        "greedy_agree_nd": row("ref_agree_nd", "greedy_online", "reference: greedy_online on agreement and n_distinct only"),
        "greedy_cheap_plus_e01": row("ref_cheap_e01", "greedy_online", "reference: greedy_online on cheap six plus e01_proxy_error"),
        "hand_rule": row("ceiling", "hand", "reference: post hoc hand rule, threshold 445 then 100 E01-flagged"),
        "uniform": row("primary", "uniform", "reference: uniform random order"),
        "threshold": row("primary", "threshold", "reference: label-free agreement threshold (baseline, D = 0 by construction)"),
    },
    "descriptive_extra": {
        "greedy_cheap_emb_no_e01_L10a_control": row("control_cheap_emb", "greedy_online", "L10a control arm (22 columns)"),
        "static_cheap_no_e01": row("primary", "static", "static on cheap six, no E01"),
        "static_agree_nd_no_e01": row("ref_agree_nd", "static", "static on agreement and n_distinct, no E01"),
        "static_cheap_plus_e01": row("ref_cheap_e01", "static", "static on cheap six plus E01"),
        "greedy_cheap_emb_plus_e01": row("ceiling", "greedy_online", "greedy_online on cheap+16PCs plus E01 (22+1 columns)"),
    },
    "verdicts": {
        "primary": R["primary"]["status_at_boot_budget"]["greedy_online"],
        "ceiling": R["ceiling"]["status_at_boot_budget"]["static"],
    },
    "checks": checks,
    "checks_failed": failures,
    "caveat": ("L9b: the 3% confident-subset gate is unmet on both GSM8K families at the pre-registered 0.75 cutoff "
               "(E10 1.62%, E01 1.10%), so this is a finding, not a verdict. One task, one policy (OLMo-2-0425-1B), "
               "8 seeds, 200 group-bootstrap resamples. E01 proxy_error is a cross-policy oracle scored against gold."),
}
with open(f"{OUT}/result.json", "w") as f:
    json.dump(result, f, indent=1)

print("checks:")
for k, v in checks.items():
    print(f"  {k}: {'PASS' if v['pass'] else 'FAIL'}  {v['detail']}")
p, c = result["primary"], result["ceiling"]
print(f"PRIMARY greedy_online cheap six, no E01, 25%: caught {p['caught_25']:.2f}, admitted {p['admitted_25']:.4f}, "
      f"D {p['D_point']['0.25']:+.4f} CI [{p['D_25_ci95'][0]:+.4f}, {p['D_25_ci95'][1]:+.4f}], status {p['status_25']}")
print(f"CEILING static cheap+emb+E01 (seed mean), 25%: caught {c['caught_25']:.2f}, admitted {c['admitted_25']:.4f}, "
      f"D {c['D_point']['0.25']:+.4f} CI [{c['D_25_ci95'][0]:+.4f}, {c['D_25_ci95'][1]:+.4f}], status {c['status_25']}")
for k, v in {**result["reference"], **result["descriptive_extra"]}.items():
    ci = f"CI [{v['D_25_ci95'][0]:+.4f}, {v['D_25_ci95'][1]:+.4f}]" if "D_25_ci95" in v else ""
    print(f"  {k}: caught {v['caught_25']:.2f} adm {v['admitted_25']:.4f} D10 {v['D_point'].get('0.1', float('nan')):+.4f} "
          f"D25 {v['D_point'].get('0.25', float('nan')):+.4f} D50 {v['D_point'].get('0.5', float('nan')):+.4f} {ci} {v.get('status_25', '')}")
print(f"wrote {OUT}/result.json")
sys.exit(1 if failures else 0)
