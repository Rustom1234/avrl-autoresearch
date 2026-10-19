"""L7a-tiebreak finalizer (CPU). Applies the bins REGISTERED in the docstring of exp/sim_warm_boot_j4.py (read that first; this file only applies them) to <root>/<pool>/boot_tb_f{0.25,0.1}.json (main mode, mode (i) context arms and mode (ii) seven-rule best fixed)
and <root>/<pool>/boot_xfit_f{0.25,0.1}.json (cross-fit mode), writes <root>/result.json and <root>/table.txt. Pools in the verdict: rm (K = 1,715), judge3b, L3b-judge-proxy (Qwen3-4B). rm_sens (K = 1,840), the 10% budget and every basic CI are reported, never in the verdict.
Per pool at 25%: X_p (cross-fit point ratio tb_xfit / greedy_blend >= 0.98), C_p (greedy_blend / tb_xfit percentile CI includes 1.0), V_p (non-vacuity: plug-in agree_acc / greedy_blend < 0.98 on the cross-fit halves), P_p = X_p and C_p and V_p (P_literal = without V_p), K_p (greedy_blend7 / BEST7 percentile CI lower end > 1.0).
PASS = P_p on >= 2 of 3 pools; KILL = K_p on >= 2 of 3; both = INCONCLUSIVE (DISCORDANT); else INCONCLUSIVE. FAILED (no verdict) if any gate failed, was skipped or is missing: j3 reproduction (every j3 column, full pool and every resample, against runs/L6a-cell-rate-usability/<pool>/boot_cellrerank_f0.25.json for 25% and against an
unmodified j3 run at 10% in <root>/_ref/<pool>/boot_j3_f0.1.json), all norefit gates of both modes, the j3 cell_rerank gates, and the cross-fit gates.
usage: l7a_finalize.py [--root DIR] [--pools p1 p2 p3] [--fracs 0.25 0.1] [--smoke] [--no-sens]  (--smoke skips the j3 reproduction gates and requires no 10% or sens files)."""
import os, sys, json, argparse
import numpy as np
ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))); RUNS = os.path.join(ROOT, "avrl", "runs")
ap = argparse.ArgumentParser(); ap.add_argument("--root", default=os.path.join(RUNS, "L7a-tiebreak")); ap.add_argument("--pools", nargs="*", default=["rm", "judge3b", "L3b-judge-proxy"]); ap.add_argument("--fracs", nargs="*", default=["0.25", "0.1"])
ap.add_argument("--smoke", action="store_true"); ap.add_argument("--no-sens", action="store_true"); ap.add_argument("--l6a", default=os.path.join(RUNS, "L6a-cell-rate-usability")); a = ap.parse_args()
NAMES = {"rm": "Skywork rm (K = 1,715, gold-calibrated)", "judge3b": "Qwen2.5-3B judge", "L3b-judge-proxy": "Qwen3-4B judge", "rm_sens": "rm sensitivity (K = 1,840, never in the verdict)"}
FIXED = ("agree_thr", "judge_thr", "blend50", "agree_acc", "blend50_acc"); TB = ("agree_acc_soth", "agree_acc_sdiff"); J3ARMS = ("uniform",) + FIXED + ("cell_rerank", "greedy_warm", "greedy_blend", "oracle")
MAINARMS = J3ARMS + TB + ("greedy_warm7", "greedy_blend7"); XARMS = ("uniform",) + FIXED + TB + ("tb_xfit", "greedy_warm", "greedy_blend", "oracle")
PRIMARY = a.fracs[0]; gate_msgs = []

def ci_of(mean, full, num, den):
    r = mean[num] / np.maximum(mean[den], 1e-9); plug = full[num] / max(full[den], 1e-9); q = np.percentile(r, [2.5, 97.5])
    return {"plug_in": float(plug), "boot_mean": float(r.mean()), "boot_sd": float(r.std(ddof=1)), "percentile_ci95": [float(q[0]), float(q[1])], "basic_ci95": [float(2 * plug - q[1]), float(2 * plug - q[0])], "frac_above_1": float((r > 1).mean())}

def j3_repro(new_path, ref_path):
    if not os.path.exists(ref_path): return {"pass": None, "note": f"no reference {ref_path}"}
    new, old = json.load(open(new_path)), json.load(open(ref_path)); key = lambda j: (j["n_boot"], j["seeds_per_resample"], j["budget_frac"], j["n_pool"], j["accept_col"], j["label"])
    if key(new) != key(old): return {"pass": None, "note": f"reference run has different B / seeds / frac / pool / accept / label: {key(new)} vs {key(old)}"}
    eq_full = all(new["full_pool"][q] == old["full_pool"][q] for q in J3ARMS) and new["full_pool"]["best_fixed"] == old["full_pool"]["best_fixed"] and new["full_pool"]["fixed_mean_at_25"] == old["full_pool"]["fixed_mean_at_25"]
    eq_res = len(new["resamples"]) == len(old["resamples"]) and all(rn[q] == ro[q] for rn, ro in zip(new["resamples"], old["resamples"]) for q in J3ARMS)
    return {"pass": bool(eq_full and eq_res), "full_pool_equal": bool(eq_full), "resamples_equal": bool(eq_res), "reference": ref_path, "arms_checked": list(J3ARMS)}

def main_stats(path):
    bj = json.load(open(path)); fp, rs = bj["full_pool"], bj["resamples"]; b5, b7 = fp["best_fixed"], fp["best_fixed7"]
    mean = {p: np.array([np.mean(r[p]) for r in rs]) for p in MAINARMS}; full = {p: float(np.mean(fp[p])) for p in MAINARMS}; ci = lambda n, d: ci_of(mean, full, n, d)
    R = {"greedy_blend7_over_best7": ci("greedy_blend7", b7), "greedy_warm7_over_best7": ci("greedy_warm7", b7), "greedy_blend_over_best7": ci("greedy_blend", b7), "greedy_blend_over_agree_acc": ci("greedy_blend", "agree_acc"), "greedy_warm_over_agree_acc": ci("greedy_warm", "agree_acc"),
         "soth_over_agree_acc": ci("agree_acc_soth", "agree_acc"), "sdiff_over_agree_acc": ci("agree_acc_sdiff", "agree_acc"), "soth_over_greedy_blend": ci("agree_acc_soth", "greedy_blend"), "sdiff_over_greedy_blend": ci("agree_acc_sdiff", "greedy_blend"), "best7_over_greedy_blend": ci(b7, "greedy_blend")}
    return {"file": path, "accept_col": bj["accept_col"], "n_pool": bj["n_pool"], "n_pool_errors": fp["n_err"], "budget": fp["budget"], "n_boot": bj["n_boot"], "seeds_per_resample": bj["seeds_per_resample"], "sanity": bj["sanity"], "best_fixed5": b5, "best_fixed7": b7,
            "fixed_mean_at_25_7_full_pool": fp["fixed_mean_at_25_7"], "caught_full_pool_8seed": full, "caught_boot_mean": {p: float(mean[p].mean()) for p in MAINARMS}, "caught_boot_sd": {p: float(mean[p].std(ddof=1)) for p in MAINARMS}, "ratios": R,
            "H5_full_pool": float((full["oracle"] - full[b5]) / full["oracle"]), "H7_full_pool": float((full["oracle"] - full[b7]) / full["oracle"]), "H7_boot_mean": float(((mean["oracle"] - mean[b7]) / mean["oracle"]).mean()),
            "K_pass": bool(R["greedy_blend7_over_best7"]["percentile_ci95"][0] > 1.0), "gate_ok": bool(all(v for k, v in bj["sanity"].items() if k.startswith("gate_") and v is not None)) and all(v is not None for k, v in bj["sanity"].items() if k.startswith("gate_"))}

def xfit_stats(path):
    xj = json.load(open(path)); H = xj["halves"]; assert len(H) == 2 and len(H[0]["resamples"]) == len(H[1]["resamples"]) == xj["n_boot"]
    full = {p: float(sum(np.mean(h["full_half"][p]) for h in H)) for p in XARMS}; mean = {p: sum(np.array([np.mean(r[p]) for r in h["resamples"]]) for h in H) for p in XARMS}; ci = lambda n, d: ci_of(mean, full, n, d)
    R = {"tb_xfit_over_greedy_blend": ci("tb_xfit", "greedy_blend"), "greedy_blend_over_tb_xfit": ci("greedy_blend", "tb_xfit"), "tb_xfit_over_agree_acc": ci("tb_xfit", "agree_acc"), "greedy_blend_over_agree_acc": ci("greedy_blend", "agree_acc"), "greedy_warm_over_agree_acc": ci("greedy_warm", "agree_acc"),
         "tb_xfit_over_greedy_warm": ci("tb_xfit", "greedy_warm"), "soth_over_greedy_blend": ci("agree_acc_soth", "greedy_blend"), "sdiff_over_greedy_blend": ci("agree_acc_sdiff", "greedy_blend")}
    inv = R["greedy_blend_over_tb_xfit"]; lo, hi = inv["percentile_ci95"]
    X = bool(R["tb_xfit_over_greedy_blend"]["plug_in"] >= 0.98); Cc = bool(lo <= 1.0 <= hi); V = bool(full["agree_acc"] / full["greedy_blend"] < 0.98)
    return {"file": path, "accept_col": xj["accept_col"], "n_boot": xj["n_boot"], "seeds_per_resample": xj["seeds_per_resample"], "sanity": xj["sanity"], "budget_per_half": [h["full_half"]["budget"] for h in H], "n_err_per_half": [h["full_half"]["n_err"] for h in H],
            "best_fixed5_per_half": [h["full_half"]["best_fixed"] for h in H], "selection": [{"held_out_parity": h["held_out_parity"], "tb_xfit_is": h["tb_xfit_is"], "selection": h["selection"], "in_sample_context": h["selection_in_sample_context"]} for h in H],
            "caught_cross_fit_8seed": full, "caught_cross_fit_boot_mean": {p: float(mean[p].mean()) for p in XARMS}, "caught_cross_fit_boot_sd": {p: float(mean[p].std(ddof=1)) for p in XARMS}, "ratios": R,
            "X": X, "C": Cc, "V": V, "P": bool(X and Cc and V), "P_literal": bool(X and Cc), "TB_ABOVE_LEARNER": bool(hi < 1.0), "gate_ok": bool(all(v is True for v in xj["sanity"].values()))}

R = {}
for p in a.pools + ([] if a.no_sens or a.smoke else ["rm_sens"]):
    d = os.path.join(a.root, p); x = {"name": NAMES.get(p, p), "main": {}, "xfit": {}}
    for f in a.fracs:
        mp, xp = os.path.join(d, f"boot_tb_f{f}.json"), os.path.join(d, f"boot_xfit_f{f}.json")
        if os.path.exists(mp):
            x["main"][f] = m = main_stats(mp)
            if not m["gate_ok"]: gate_msgs.append(f"{p} f{f} main: gate flag false or missing {json.dumps(m['sanity'])}")
            if not a.smoke:
                ref = os.path.join(a.l6a, "rm" if p == "rm_sens" else p, "boot_cellrerank_sens_f0.25.json" if p == "rm_sens" else "boot_cellrerank_f0.25.json") if f == "0.25" else os.path.join(a.root, "_ref", p, "boot_j3_f0.1.json")
                m["j3_reproduction"] = j3_repro(mp, ref)
                if m["j3_reproduction"]["pass"] is not True: gate_msgs.append(f"{p} f{f}: j3 reproduction {json.dumps(m['j3_reproduction'])}")
        else: gate_msgs.append(f"{p} f{f}: no {os.path.basename(mp)}")
        if os.path.exists(xp):
            x["xfit"][f] = xs = xfit_stats(xp)
            if not xs["gate_ok"]: gate_msgs.append(f"{p} f{f} xfit: gate flag false {json.dumps(xs['sanity'])}")
        else: gate_msgs.append(f"{p} f{f}: no {os.path.basename(xp)}")
    R[p] = x

vp = [p for p in a.pools if PRIMARY in R[p]["main"] and PRIMARY in R[p]["xfit"]]
P = {p: R[p]["xfit"][PRIMARY]["P"] for p in vp}; PL = {p: R[p]["xfit"][PRIMARY]["P_literal"] for p in vp}; K = {p: R[p]["main"][PRIMARY]["K_pass"] for p in vp}; TBA = {p: R[p]["xfit"][PRIMARY]["TB_ABOVE_LEARNER"] for p in vp}
nP, nPL, nK = sum(P.values()), sum(PL.values()), sum(K.values())
if gate_msgs: line = "FAILED (gate or missing input): " + " | ".join(gate_msgs)
elif len(vp) < len(a.pools): line = f"FAILED: only {len(vp)} of {len(a.pools)} pools complete"
elif nP >= 2 and nK >= 2: line = f"INCONCLUSIVE (DISCORDANT: both bins fire, PASS on {[p for p in vp if P[p]]}, KILL on {[p for p in vp if K[p]]})"
elif nP >= 2: line = f"PASS: cross-fitted tie-break within 0.98 of greedy_blend with the CI including 1.0 and a non-vacuous gap on {[p for p in vp if P[p]]}; claim 2 as worded ('any fixed rule') fails in this regime; L7b would need to freeze the tie-break"
elif nK >= 2: line = f"KILL: greedy_blend / best-of-seven-fixed percentile CI lower end above 1.0 on {[p for p in vp if K[p]]}; the learner survives the completed fixed set; L7b is worth GPU time"
else: line = f"INCONCLUSIVE (P on {nP} pools, P_literal on {nPL}, K on {nK})"
if not gate_msgs and len(vp) == len(a.pools):
    if nP != nPL: line += f"; NOTE: the literal bin (without the non-vacuity guard V_p) gives P on {[p for p in vp if PL[p]]}"
    if any(TBA.values()): line += f"; flag TB_ABOVE_LEARNER on {[p for p in vp if TBA[p]]} (tie-break significantly above greedy_blend)"
if a.smoke: line = "[SMOKE, no verdict] " + line

head = {p: {"best_fixed7": R[p]["main"][PRIMARY]["best_fixed7"], "H5": R[p]["main"][PRIMARY]["H5_full_pool"], "H7": R[p]["main"][PRIMARY]["H7_full_pool"], "caught_full_pool_8seed": {q: R[p]["main"][PRIMARY]["caught_full_pool_8seed"][q] for q in ("agree_acc", "agree_acc_soth", "agree_acc_sdiff", "greedy_warm", "greedy_blend", "greedy_blend7", "oracle")},
            "greedy_blend7_over_best7_pctCI": R[p]["main"][PRIMARY]["ratios"]["greedy_blend7_over_best7"]["percentile_ci95"], "tb_xfit_over_greedy_blend_plug_in": R[p]["xfit"][PRIMARY]["ratios"]["tb_xfit_over_greedy_blend"]["plug_in"], "tb_xfit_over_greedy_blend_basicCI": R[p]["xfit"][PRIMARY]["ratios"]["tb_xfit_over_greedy_blend"]["basic_ci95"],
            "greedy_blend_over_tb_xfit_pctCI": R[p]["xfit"][PRIMARY]["ratios"]["greedy_blend_over_tb_xfit"]["percentile_ci95"], "X": R[p]["xfit"][PRIMARY]["X"], "C": R[p]["xfit"][PRIMARY]["C"], "V": R[p]["xfit"][PRIMARY]["V"], "P": R[p]["xfit"][PRIMARY]["P"], "P_literal": R[p]["xfit"][PRIMARY]["P_literal"], "K": R[p]["main"][PRIMARY]["K_pass"]} for p in vp}
res = {"experiment": "L7a-tiebreak", "verdict_line": line, "headline_25pct": head, "primary_frac": PRIMARY, "P_pass": P, "P_literal_pass": PL, "K_pass": K, "TB_ABOVE_LEARNER": TBA, "gate_messages": gate_msgs, "pools": R}
json.dump(res, open(os.path.join(a.root, "result.json"), "w"), indent=1, default=float)

L = [line, ""]
for p in list(R):
    x = R[p]; L.append(f"## {x['name']}")
    for f in a.fracs:
        m, xs = x["main"].get(f), x["xfit"].get(f)
        if m:
            c, sd, fb = m["caught_full_pool_8seed"], m["caught_boot_sd"], m["caught_boot_mean"]
            L.append(f"[main, budget frac {f}] pool {m['n_pool']}, J errors {m['n_pool_errors']}, budget {m['budget']}, {m['n_boot']} resamples x {m['seeds_per_resample']} seeds, accept_col {m['accept_col']}; best fixed (5 rules) {m['best_fixed5']}, BEST7 {m['best_fixed7']}, H5 {m['H5_full_pool']:.3f}, H7 {m['H7_full_pool']:.3f} (boot mean {m['H7_boot_mean']:.3f})")
            L.append("  caught full pool 8 seeds | boot mean (SD): " + "; ".join(f"{q} {c[q]:.1f} | {fb[q]:.1f} ({sd[q]:.1f})" for q in ("agree_acc", "agree_acc_soth", "agree_acc_sdiff", "greedy_warm", "greedy_blend", "greedy_warm7", "greedy_blend7", "oracle")))
            L.append("  seven-rule fixed means at 25% (full pool): " + ", ".join(f"{q} {v:.1f}" for q, v in m["fixed_mean_at_25_7_full_pool"].items()))
            for nm, r in m["ratios"].items(): L.append(f"  {nm}: plug-in {r['plug_in']:.3f}, boot mean {r['boot_mean']:.3f} (SD {r['boot_sd']:.3f}), percentile CI [{r['percentile_ci95'][0]:.3f}, {r['percentile_ci95'][1]:.3f}], basic CI [{r['basic_ci95'][0]:.3f}, {r['basic_ci95'][1]:.3f}]")
            L.append(f"  K_p (greedy_blend7 / BEST7 percentile CI lower > 1.0): {m['K_pass']}; j3 reproduction: {json.dumps(m.get('j3_reproduction'))}")
        if xs:
            c = xs["caught_cross_fit_8seed"]; fb = xs["caught_cross_fit_boot_mean"]; sd = xs["caught_cross_fit_boot_sd"]
            L.append(f"[cross-fit, budget frac {f}] budget per half {xs['budget_per_half']}, errors per half {xs['n_err_per_half']}, best of 5 j2 rules per half {xs['best_fixed5_per_half']}")
            for s in xs["selection"]:
                q, ins = s["selection"], s["in_sample_context"]; fm = lambda v: "n/a" if v is None else f"{v:.3f}"
                L.append(f"  held-out half {s['held_out_parity']} (chosen on the other half): tie-break {s['tb_xfit_is']}; tier agreement {q['tier_agreement']}, {q['n_tier']} groups, {q['n_err_tier']} errors, AUC s_oth {fm(q['auc_soth'])}, -s_diff {fm(q['auc_sdiff'])}, fallback {q['fallback']}; in-sample context on the held-out half itself: AUC s_oth {fm(ins['auc_soth'])}, -s_diff {fm(ins['auc_sdiff'])}")
            L.append("  caught summed over the two held-out halves, 8 seeds | boot mean (SD): " + "; ".join(f"{q} {c[q]:.1f} | {fb[q]:.1f} ({sd[q]:.1f})" for q in ("agree_acc", "agree_acc_soth", "agree_acc_sdiff", "tb_xfit", "greedy_warm", "greedy_blend", "oracle")))
            for nm, r in xs["ratios"].items(): L.append(f"  {nm}: plug-in {r['plug_in']:.3f}, boot mean {r['boot_mean']:.3f} (SD {r['boot_sd']:.3f}), percentile CI [{r['percentile_ci95'][0]:.3f}, {r['percentile_ci95'][1]:.3f}] (basic [{r['basic_ci95'][0]:.3f}, {r['basic_ci95'][1]:.3f}])")
            L.append(f"  X_p {xs['X']}, C_p {xs['C']}, V_p {xs['V']}, P_p {xs['P']}, P_literal {xs['P_literal']}, TB_ABOVE_LEARNER {xs['TB_ABOVE_LEARNER']}")
    L.append("")
open(os.path.join(a.root, "table.txt"), "w").write("\n".join(L)); print("\n".join(L))
