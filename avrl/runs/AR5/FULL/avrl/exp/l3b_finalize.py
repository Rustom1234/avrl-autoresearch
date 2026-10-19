"""L3b stage 3 (CPU): replay metrics and the final result.json. PRE-REGISTERED with l3b_analyze.py.
Reads <run_dir>/result_analyze.json, gates.json, boot_J_f0.25.json, boot_J_f0.1.json (sim_warm_boot_j.py, pool = all 2,000 groups, label judge_error) and, if present,
boot_J_held_f0.25.json (robustness: E21 held-out-cells quarter pool, not used for the verdict). Per resample the ratio is mean over the NS seeds of greedy_warm over mean over the
NS seeds of the best fixed rule (best = highest full-pool mean hits at 25% among agree_thr, judge_thr, blend50, chosen inside sim_warm_boot_j.py, the same rule at both budgets).
Percentile 95% CI of the per-resample ratios is the verdict interval; the basic CI (2 * plug-in - q97.5, 2 * plug-in - q2.5, plug-in = full-pool 8-seed ratio) is reported beside it.
H = (oracle - best fixed) / oracle on the full pool. Claim 2 PASS (secondary): percentile CI lower end > 1.0 at the 25% budget. Claim 2 offline KILL: best fixed rule >= 0.98 x greedy_warm
at 25% (full-pool point estimate) AND the percentile CI includes 1.0. Otherwise inconclusive. Writes <run_dir>/result.json and replay_table.txt."""
import os, sys, json, argparse
import numpy as np
ap = argparse.ArgumentParser(); ap.add_argument("--run_dir", required=True); a = ap.parse_args(); Q = a.run_dir
ALL = ("uniform", "agree_thr", "judge_thr", "blend50", "greedy_warm", "greedy_blend", "oracle"); FIXED = ("agree_thr", "judge_thr", "blend50")
A = json.load(open(os.path.join(Q, "result_analyze.json"))); gates = json.load(open(os.path.join(Q, "gates.json")))

def replay(path):
    bj = json.load(open(path)); fp = bj["full_pool"]; best = fp["best_fixed"]; rs = bj["resamples"]; mean = {p: np.array([np.mean(r[p]) for r in rs]) for p in ALL}
    full = {p: float(np.mean(fp[p])) for p in ALL}; out = {"file": os.path.basename(path), "pool_mode": bj["pool_mode"], "n_pool": bj["n_pool"], "n_pool_errors": fp["n_err"], "budget_frac": bj["budget_frac"],
                                                       "budget": fp["budget"], "n_boot": bj["n_boot"], "seeds_per_resample": bj["seeds_per_resample"], "sanity": bj["sanity"], "best_fixed": best,
                                                       "fixed_mean_at_25_full_pool": fp["fixed_mean_at_25"], "caught_full_pool_8seed": full, "caught_boot_mean": {p: float(mean[p].mean()) for p in ALL}, "caught_boot_sd": {p: float(mean[p].std(ddof=1)) for p in ALL}}
    def ci(num, den):
        r = mean[num] / np.maximum(mean[den], 1e-9); plug = full[num] / max(full[den], 1e-9); q = np.percentile(r, [2.5, 97.5])
        return {"plug_in": float(plug), "boot_mean": float(r.mean()), "boot_sd": float(r.std(ddof=1)), "percentile_ci95": [float(q[0]), float(q[1])], "basic_ci95": [float(2 * plug - q[1]), float(2 * plug - q[0])], "frac_above_1": float((r > 1).mean())}
    out["ratio_greedy_warm_over_best_fixed"] = ci("greedy_warm", best); out["ratio_greedy_blend_over_best_fixed"] = ci("greedy_blend", best)
    out["ratio_greedy_warm_over"] = {p: ci("greedy_warm", p) for p in FIXED}; out["ratio_best_fixed_over_agree_thr"] = ci(best, "agree_thr")
    out["H_full_pool"] = float((full["oracle"] - full[best]) / full["oracle"]); out["H_boot_mean"] = float(((mean["oracle"] - mean[best]) / mean["oracle"]).mean())
    d = mean["greedy_warm"] - mean[best]; out["gap_greedy_warm_minus_best_fixed"] = {"boot_mean": float(d.mean()), "percentile_ci95": [float(np.percentile(d, 2.5)), float(np.percentile(d, 97.5))]}
    return out

R = {}
for tag, f in (("f0.25", "boot_J_f0.25.json"), ("f0.1", "boot_J_f0.1.json"), ("held_f0.25", "boot_J_held_f0.25.json")):
    if os.path.exists(os.path.join(Q, f)): R[tag] = replay(os.path.join(Q, f))
c2 = {"available": "f0.25" in R}
if c2["available"]:
    r = R["f0.25"]; rt = r["ratio_greedy_warm_over_best_fixed"]; bf = r["best_fixed"]; full = r["caught_full_pool_8seed"]
    c2.update(best_fixed=bf, ratio_plug_in=rt["plug_in"], percentile_ci95=rt["percentile_ci95"], basic_ci95=rt["basic_ci95"], pass_ci_lower_gt_1=bool(rt["percentile_ci95"][0] > 1.0),
              kill_best_fixed_ge_098_greedy_warm=bool(full[bf] >= 0.98 * full["greedy_warm"]), kill_ci_includes_1=bool(rt["percentile_ci95"][0] <= 1.0 <= rt["percentile_ci95"][1]))
    c2["verdict"] = "claim 2 PASS (offline)" if c2["pass_ci_lower_gt_1"] else ("claim 2 offline KILL" if (c2["kill_best_fixed_ge_098_greedy_warm"] and c2["kill_ci_includes_1"]) else "inconclusive")
else: c2["verdict"] = "not run"
res = {"experiment": "L3b-judge-proxy", "gates_score": gates, "claim3": A["claim3"], "claim2": c2, "primary_tau0": A["primary_tau0"], "sensitivity_tau_match": A["sensitivity_tau_match"], "judge_overall": A["judge_overall"],
       "e21_reproduction_gate": A["e21_reproduction_gate"], "replay": R}
json.dump(res, open(os.path.join(Q, "result.json"), "w"), indent=1, default=float)
L = [f"claim 3: {A['claim3']['verdict']}", f"claim 2: {c2['verdict']}", ""]
for tag, r in R.items():
    L += [f"## replay {tag} (pool {r['n_pool']}, {r['n_pool_errors']} J errors, budget {r['budget']}, {r['n_boot']} resamples x {r['seeds_per_resample']} seeds, best fixed = {r['best_fixed']}, fixed means at 25% {json.dumps({k: round(v, 1) for k, v in r['fixed_mean_at_25_full_pool'].items()})})", "",
          "| policy | caught full pool (8 seeds) | caught bootstrap mean (SD) |", "|---|---|---|"]
    L += [f"| {p} | {r['caught_full_pool_8seed'][p]:.1f} | {r['caught_boot_mean'][p]:.1f} ({r['caught_boot_sd'][p]:.1f}) |" for p in ALL]
    for nm in ("ratio_greedy_warm_over_best_fixed", "ratio_greedy_blend_over_best_fixed", "ratio_best_fixed_over_agree_thr"):
        x = r[nm]; L.append(f"{nm}: plug-in {x['plug_in']:.3f}, boot mean {x['boot_mean']:.3f} (SD {x['boot_sd']:.3f}), percentile CI [{x['percentile_ci95'][0]:.3f}, {x['percentile_ci95'][1]:.3f}], basic CI [{x['basic_ci95'][0]:.3f}, {x['basic_ci95'][1]:.3f}], frac above 1 {x['frac_above_1']:.2f}")
    for p in FIXED:
        x = r["ratio_greedy_warm_over"][p]; L.append(f"  greedy_warm / {p}: plug-in {x['plug_in']:.3f}, percentile CI [{x['percentile_ci95'][0]:.3f}, {x['percentile_ci95'][1]:.3f}]")
    L += [f"H full pool {r['H_full_pool']:.3f}, H bootstrap mean {r['H_boot_mean']:.3f}; gap greedy_warm - best fixed: boot mean {r['gap_greedy_warm_minus_best_fixed']['boot_mean']:+.2f}, CI {r['gap_greedy_warm_minus_best_fixed']['percentile_ci95']}", ""]
L += ["## claim 2", json.dumps(c2, indent=1, default=float), "", "## claim 3", json.dumps(A["claim3"], indent=1, default=float)]
open(os.path.join(Q, "replay_table.txt"), "w").write("\n".join(L)); print("\n".join(L))
