"""L6a-cell-rate-usability finalizer (CPU). Forms the verdict from <root>/<pool>/result_cv.json (exp/l6a_cell_rate.py) and <root>/<pool>/boot_cellrerank_f0.25.json (exp/sim_warm_boot_j3.py), writes <root>/result.json and <root>/table.txt.
The bins are REGISTERED in the docstrings of exp/l6a_cell_rate.py (CV side, PASS / KILL / INCONCLUSIVE) and exp/sim_warm_boot_j3.py (replay side); this file only applies them. Per pool: A_p (best cell set delta AUC >= 0.03 with paired bootstrap CI lower end > 0 on ACC), K_p (all three cell sets delta < 0.02 and CI upper end < 0.05),
R_p (cell_rerank / agree_acc percentile 95% CI lower end > 1.0 at the 25% budget). PASS = A_p and R_p on the same pool for at least two of three pools. KILL = K_p on all three pools (discordant if R_p holds on two or more pools). Else INCONCLUSIVE (loose reading, A on >= 2 and R on >= 2 pools but not jointly, is flagged).
FAILED if any gate failed or was skipped on a real run: CV gates (E21 reproduction, JA AUC reproduction), replay gates (norefit, cell_rerank with no audited error equals agree_acc), and reproduction of every j2 arm that is deterministic (uniform, five fixed rules, oracle) plus the two learner arms against the L5c boot JSON of the same pool.
usage: l6a_finalize.py [--root DIR] [--pools p1 p2 p3] [--smoke]  (--smoke skips the j2-reproduction and skipped-gate requirements)."""
import os, sys, json, argparse
import numpy as np
ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))); RUNS = os.path.join(ROOT, "avrl", "runs")
ap = argparse.ArgumentParser(); ap.add_argument("--root", default=os.path.join(RUNS, "L6a-cell-rate-usability")); ap.add_argument("--pools", nargs="*", default=["rm", "judge3b", "L3b-judge-proxy"]); ap.add_argument("--smoke", action="store_true"); a = ap.parse_args()
NAMES = {"rm": "Skywork rm (K = 1,715, gold-calibrated)", "judge3b": "Qwen2.5-3B judge", "L3b-judge-proxy": "Qwen3-4B judge"}
J2REF = {"rm": "L5c-proxies/rm/boot_f0.25.json", "judge3b": "L5c-proxies/judge3b/boot_f0.25.json", "L3b-judge-proxy": "L5c-proxies/qwen3_4b_ref/boot_f0.25.json"}
FIXED = ("agree_thr", "judge_thr", "blend50", "agree_acc", "blend50_acc"); ALL = ("uniform",) + FIXED + ("cell_rerank", "greedy_warm", "greedy_blend", "oracle"); J2ARMS = [p for p in ALL if p != "cell_rerank"]; J2DET = ("uniform",) + FIXED + ("oracle",)
CELL_SETS = ("JA+cell100", "JA+cell300", "JA+both")

def replay(path):
    bj = json.load(open(path)); fp = bj["full_pool"]; best = fp["best_fixed"]; rs = bj["resamples"]; mean = {p: np.array([np.mean(r[p]) for r in rs]) for p in ALL}; full = {p: float(np.mean(fp[p])) for p in ALL}
    def ci(num, den):
        r = mean[num] / np.maximum(mean[den], 1e-9); plug = full[num] / max(full[den], 1e-9); q = np.percentile(r, [2.5, 97.5])
        return {"plug_in": float(plug), "boot_mean": float(r.mean()), "boot_sd": float(r.std(ddof=1)), "percentile_ci95": [float(q[0]), float(q[1])], "basic_ci95": [float(2 * plug - q[1]), float(2 * plug - q[0])], "frac_above_1": float((r > 1).mean())}
    d = mean["cell_rerank"] - mean["agree_acc"]; qd = np.percentile(d, [2.5, 97.5])
    return {"file": path, "accept_col": bj.get("accept_col"), "n_pool": bj["n_pool"], "n_pool_errors": fp["n_err"], "budget": fp["budget"], "n_boot": bj["n_boot"], "seeds_per_resample": bj["seeds_per_resample"], "sanity": bj["sanity"], "best_fixed": best,
            "fixed_mean_at_25_full_pool": fp["fixed_mean_at_25"], "caught_full_pool_8seed": full, "caught_boot_mean": {p: float(mean[p].mean()) for p in ALL}, "caught_boot_sd": {p: float(mean[p].std(ddof=1)) for p in ALL},
            "ratio_cell_rerank_over_agree_acc": ci("cell_rerank", "agree_acc"), "diff_cell_rerank_minus_agree_acc": {"plug_in": float(full["cell_rerank"] - full["agree_acc"]), "boot_mean": float(d.mean()), "percentile_ci95": [float(qd[0]), float(qd[1])]},
            "ratio_cell_rerank_over_best_fixed": ci("cell_rerank", best), "ratio_greedy_warm_over_agree_acc": ci("greedy_warm", "agree_acc"), "ratio_greedy_blend_over_agree_acc": ci("greedy_blend", "agree_acc"),
            "H_full_pool": float((full["oracle"] - full[best]) / full["oracle"]), "H_boot_mean": float(((mean["oracle"] - mean[best]) / mean["oracle"]).mean()), "gate_ok": bool(all(v for k, v in bj["sanity"].items() if k.startswith("gate_") and v is not None))}

def j2_repro(path, ref_rel):
    p = os.path.join(RUNS, ref_rel)
    if not os.path.exists(p): return {"pass": None, "note": f"no reference {ref_rel}"}
    new, old = json.load(open(path)), json.load(open(p))
    if (new["n_boot"], new["seeds_per_resample"], new["budget_frac"], new["n_pool"]) != (old["n_boot"], old["seeds_per_resample"], old["budget_frac"], old["n_pool"]): return {"pass": None, "note": "reference run has different B / seeds / frac / pool size"}
    eq = lambda arms: all(new["full_pool"][q] == old["full_pool"][q] for q in arms) and all(rn[q] == ro[q] for rn, ro in zip(new["resamples"], old["resamples"]) for q in arms)
    det = eq(J2DET); lrn = eq(("greedy_warm", "greedy_blend")); return {"pass": bool(det), "deterministic_arms_equal": bool(det), "learner_arms_equal": bool(lrn), "best_fixed_equal": new["full_pool"]["best_fixed"] == old["full_pool"]["best_fixed"], "reference": ref_rel}

R, gate_msgs = {}, []
for p in a.pools:
    d = os.path.join(a.root, p); cvp = os.path.join(d, "result_cv.json"); bp = os.path.join(d, "boot_cellrerank_f0.25.json"); x = {"name": NAMES.get(p, p)}
    if not os.path.exists(cvp): gate_msgs.append(f"{p}: no result_cv.json"); R[p] = x; continue
    cv = json.load(open(cvp)); x["cv_gates"] = cv["gates"]; x["cv_gates_all_pass"] = cv["gates_all_pass"]; x["cv_gates_any_skipped"] = cv["gates_any_skipped"]
    if not cv["gates_all_pass"] or (cv["gates_any_skipped"] and not a.smoke): gate_msgs.append(f"{p}: CV gate failed or skipped {json.dumps(cv['gates'])}")
    for tag in ("primary", "sensitivity", "consensus_post_hoc"):
        if tag in cv: x[tag] = cv[tag]
    if os.path.exists(bp):
        x["replay"] = replay(bp)
        if not x["replay"]["gate_ok"]: gate_msgs.append(f"{p}: replay gate flags {json.dumps(x['replay']['sanity'])}")
        if not a.smoke and p in J2REF:
            x["j2_reproduction"] = j2_repro(bp, J2REF[p])
            if x["j2_reproduction"]["pass"] is not True: gate_msgs.append(f"{p}: j2 reproduction {json.dumps(x['j2_reproduction'])}")
    else: gate_msgs.append(f"{p}: no boot_cellrerank_f0.25.json")
    sp = os.path.join(d, "boot_cellrerank_sens_f0.25.json")
    if os.path.exists(sp): x["replay_sensitivity"] = replay(sp)
    if "primary" in x and "deltas" in x["primary"]:
        pr = x["primary"]; x["A_pass"] = pr["A_pass"]; x["K_kill_side"] = pr["K_kill_side"]; x["best_cell_set"] = pr["best_cell_set"]; x["best_delta"] = pr["deltas"][pr["best_cell_set"]]
    if "replay" in x: x["R_pass"] = bool(x["replay"]["ratio_cell_rerank_over_agree_acc"]["percentile_ci95"][0] > 1.0)
    R[p] = x

have = [p for p in a.pools if "A_pass" in R.get(p, {}) and "R_pass" in R.get(p, {})]; joint = [p for p in have if R[p]["A_pass"] and R[p]["R_pass"]]
nA, nR = sum(R[p]["A_pass"] for p in have), sum(R[p]["R_pass"] for p in have); allK = len(have) == len(a.pools) and all(R[p]["K_kill_side"] for p in have)
if gate_msgs: line = "FAILED (gate or missing input): " + " | ".join(gate_msgs)
elif len(have) < len(a.pools): line = f"FAILED: only {len(have)} of {len(a.pools)} pools complete"
elif len(joint) >= 2: line = f"PASS: A and R both hold on {len(joint)} pools ({', '.join(joint)}); next L6c (fresh-prompt out-of-sample test)"
elif allK: line = "KILL (registered AUC bin: all three cell sets below +0.02 with CI upper < 0.05 on all three pools)" + (f"; DISCORDANT: R holds on {nR} pools" if nR >= 2 else "; L6c drops to a descriptive-only check")
else: line = f"INCONCLUSIVE (A on {nA} pools, R on {nR} pools, joint on {len(joint)}" + ("; loose reading met, not jointly" if nA >= 2 and nR >= 2 else "") + ")"
if a.smoke: line = "[SMOKE, no verdict] " + line
def brief(x):
    o = {}
    for tag in ("primary", "sensitivity", "consensus_post_hoc"):
        r = x.get(tag)
        if r and "deltas" in r: o[tag] = {"n_acc": r["n_acc"], "n_err": r["n_err"], "n_top": r["n_top"], "n_err_top": r["n_err_top"], "auc_JA": r["sets"]["JA"]["auc"], "best_cell_set": r["best_cell_set"],
                                         "delta_acc": {k: {"delta": r["deltas"][k]["delta"], "fold_sd": r["deltas"][k]["delta_fold_sd"], "boot_ci95": r["deltas"][k]["boot_ci95"]} for k in CELL_SETS},
                                         "delta_top_half": {k: {"delta": r["top_half"]["deltas"][k]["delta"], "boot_ci95": r["top_half"]["deltas"][k]["boot_ci95"]} for k in CELL_SETS},
                                         "cell_rate_alone_auc_acc": {k: v["auc_fold_mean"] for k, v in r["cell_rate_alone"].items()}, "cell_rate_alone_auc_top_half": {k: v["auc_fold_mean"] for k, v in r["top_half"]["cell_rate_alone"].items()}}
    for tag in ("replay", "replay_sensitivity"):
        r = x.get(tag)
        if r: o[tag] = {"cell_rerank_caught": r["caught_full_pool_8seed"]["cell_rerank"], "agree_acc_caught": r["caught_full_pool_8seed"]["agree_acc"], "greedy_warm_caught": r["caught_full_pool_8seed"]["greedy_warm"], "greedy_blend_caught": r["caught_full_pool_8seed"]["greedy_blend"], "oracle_caught": r["caught_full_pool_8seed"]["oracle"],
                        "ratio_cell_rerank_over_agree_acc": {k: r["ratio_cell_rerank_over_agree_acc"][k] for k in ("plug_in", "percentile_ci95", "basic_ci95")}, "H_full_pool": r["H_full_pool"], "best_fixed": r["best_fixed"], "n_boot": r["n_boot"]}
    return o

res = {"experiment": "L6a-cell-rate-usability", "verdict_line": line, "summary": {p: brief(R[p]) for p in a.pools}, "A_pass": {p: R[p].get("A_pass") for p in a.pools}, "K_kill_side": {p: R[p].get("K_kill_side") for p in a.pools}, "R_pass": {p: R[p].get("R_pass") for p in a.pools}, "joint_pass_pools": joint,
       "gate_messages": gate_msgs, "pools": R}
json.dump(res, open(os.path.join(a.root, "result.json"), "w"), indent=1, default=float)

f4 = lambda v: f"{v:.4f}"; L = [line, ""]
for p in a.pools:
    x = R[p]; L.append(f"## {x['name']}")
    for tag in ("primary", "sensitivity", "consensus_post_hoc"):
        r = x.get(tag)
        if not r or "deltas" not in r: continue
        L.append(f"[{tag}] ACC {r['n_acc']}, n_err {r['n_err']} (top half {r['n_top']} / {r['n_err_top']}), JA AUC {f4(r['sets']['JA']['auc'])} (fold SD {f4(r['sets']['JA']['auc_fold_sd'])}); top-half JA AUC {f4(r['top_half']['sets']['JA']['auc_fold_mean'])}")
        for k in CELL_SETS:
            dl, t = r["deltas"][k], r["top_half"]["deltas"][k]
            L.append(f"  {k}: ACC delta {dl['delta']:+.4f} (fold SD {f4(dl['delta_fold_sd'])}) boot CI [{dl['boot_ci95'][0]:+.4f}, {dl['boot_ci95'][1]:+.4f}] | top half delta {t['delta']:+.4f} CI [{t['boot_ci95'][0]:+.4f}, {t['boot_ci95'][1]:+.4f}]")
        L.append(f"  best {r['best_cell_set']}, A_pass {r['A_pass']}, K_kill_side {r['K_kill_side']}; cell-rate-alone AUC ACC {json.dumps({k: round(v['auc_fold_mean'], 4) for k, v in r['cell_rate_alone'].items()})} top half {json.dumps({k: round(v['auc_fold_mean'], 4) for k, v in r['top_half']['cell_rate_alone'].items()})}")
    for tag in ("replay", "replay_sensitivity"):
        r = x.get(tag)
        if not r: continue
        c = r["ratio_cell_rerank_over_agree_acc"]; df = r["diff_cell_rerank_minus_agree_acc"]
        L.append(f"[{tag}] pool {r['n_pool']}, J errors {r['n_pool_errors']}, budget {r['budget']}, {r['n_boot']} resamples x {r['seeds_per_resample']} seeds, best fixed {r['best_fixed']}, H {r['H_full_pool']:.3f} (boot mean {r['H_boot_mean']:.3f}), accept_col {r['accept_col']}")
        L.append("  caught full pool 8 seeds | boot mean (SD): " + "; ".join(f"{q} {r['caught_full_pool_8seed'][q]:.1f} | {r['caught_boot_mean'][q]:.1f} ({r['caught_boot_sd'][q]:.1f})" for q in ALL))
        L.append(f"  cell_rerank / agree_acc: plug-in {c['plug_in']:.3f}, boot mean {c['boot_mean']:.3f} (SD {c['boot_sd']:.3f}), percentile CI [{c['percentile_ci95'][0]:.3f}, {c['percentile_ci95'][1]:.3f}] (verdict), basic CI [{c['basic_ci95'][0]:.3f}, {c['basic_ci95'][1]:.3f}], frac above 1 {c['frac_above_1']:.2f}; difference {df['plug_in']:+.1f} caught, percentile CI [{df['percentile_ci95'][0]:+.1f}, {df['percentile_ci95'][1]:+.1f}]")
        for nm in ("ratio_cell_rerank_over_best_fixed", "ratio_greedy_warm_over_agree_acc", "ratio_greedy_blend_over_agree_acc"):
            c2 = r[nm]; L.append(f"  {nm}: plug-in {c2['plug_in']:.3f}, percentile CI [{c2['percentile_ci95'][0]:.3f}, {c2['percentile_ci95'][1]:.3f}], basic CI [{c2['basic_ci95'][0]:.3f}, {c2['basic_ci95'][1]:.3f}]")
    L.append(f"  A_pass {x.get('A_pass')}, K_kill_side {x.get('K_kill_side')}, R_pass {x.get('R_pass')}; j2 reproduction {json.dumps(x.get('j2_reproduction'))}"); L.append("")
open(os.path.join(a.root, "table.txt"), "w").write("\n".join(L)); print("\n".join(L))
