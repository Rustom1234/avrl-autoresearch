"""L7a summary (CPU), adapted from exp/l6a_summarize.py. Usage: l7a_summarize.py <runs/L7a> [<runs/L6a>]
Reads l7a_d<direction>_B<B>_o<order seed>_<mode>.json (direction 1 = 3B then 7B-AWQ, direction 2 = 7B-AWQ then 3B; per-step B in 62, 125; order seeds 0..15; 12 disjoint batches of 250; mode drift / control_exact /
control_z) and writes result.json and ledger_table.md. Ratios are ratios of means over order seeds (sums of hits), CI = percentile bootstrap (10,000 resamples) over order seeds.

Regression gate (VOID if it fails): control_exact (single table A, arms in exactly their L6a form), order seeds 0..7, per-batch hits of thr_agreement, restart_offset, carry_plain, carry_offset_lam1 and batch_base_errors
must equal runs/L6a/amort_<judge A>_B<B>_o<seed>.json, and (B = 62 only) freeze_plain_k3 must equal L6a freeze_plain_125 (k = ceil(125 / 62) = 3). Also void: a missing file, any failed in-job self-check, or a drift run whose
batches 1..6 (all arms, base errors) differ from the control_z run of the same cell and seed (nothing may differ before the switch).

Selection protocol (as the L6a run): the best PERSISTENT arm per cell (carry_plain, carry_plain_window_125/250/500, carry_plain_decay, carry_offset_lam1; never reset_at_switch, never the freeze arms) is chosen by
cumulative hits over batches 7..12 (mean over the selection seeds, ties to the first in that order) on order seeds 0..7 and reported on order seeds 8..15 (VERDICT fold). Swapped fold (select on 8..15, report on 0..7)
and pooled 16 (each seed reported with the arm chosen on the other half) are cross-checks.

Pre-registered verdict (L7a spec), per cell = direction x B (4 cells), verdict fold:
  HEADLINE      best persistent / freeze_plain_k6 over batches 7..12 (ratio of means, 95% percentile bootstrap CI over order seeds)
  VOID          if the regression gate / self-checks did not pass
  PASS          if best persistent / freeze_plain_k6 >= 1.03 with CI lower bound > 1 in at least 3 of 4 cells and, in those cells, best persistent >= thr_agreement in batches 10..12
  KILL          if (i) freeze_plain_k6 >= 0.99 x best persistent over batches 7..12 in at least 3 of 4 cells, or (ii) best persistent < thr_agreement in both batches 7..9 and 10..12 in at least 3 of 4 cells
  INCONCLUSIVE  otherwise"""
import sys, os, json
import numpy as np

rd = sys.argv[1]; r6 = sys.argv[2] if len(sys.argv) > 2 else os.path.join(os.path.dirname(os.path.abspath(rd)), "L6a")
DIRS = {"d1 3B->7B-AWQ": (1, "Qwen2.5-3B-Instruct"), "d2 7B-AWQ->3B": (2, "Qwen2.5-7B-Instruct-AWQ")}; BUDS = (62, 125); NS, NB, NBOOT, SW = 16, 12, 10000, 6
MODES = ("drift", "control_exact", "control_z"); PASS_R, KILL_R, RET_R = 1.03, 0.99, 1.03
PERSIST = ["carry_plain", "carry_plain_window_125", "carry_plain_window_250", "carry_plain_window_500", "carry_plain_decay", "carry_offset_lam1"]; WIN = PERSIST[1:5]
ARMS = ["thr_agreement", "restart_offset"] + PERSIST + ["freeze_plain_k3", "freeze_plain_k6", "reset_at_switch"]
GRP = {"4-6 pre-switch": slice(3, 6), "7-9 post-switch": slice(6, 9), "10-12 recovery": slice(9, 12), "7-12": slice(6, 12)}
GATE_KEYS = ["thr_agreement", "restart_offset", "carry_plain", "carry_offset_lam1", "batch_base_errors"]
FA, FB = slice(0, 8), slice(8, 16)
rng = np.random.default_rng(0); IDX = {n: rng.integers(0, n, (NBOOT, n)) for n in (8, 16)}

def rat(num, den):
    """ratio of means of per-seed vectors with a percentile bootstrap CI over the order seeds (len 8 or 16)."""
    num, den = np.asarray(num, float), np.asarray(den, float); idx = IDX[len(num)]; rb = num[idx].mean(1) / den[idx].mean(1)
    return {"ratio": float(num.mean() / den.mean()), "ci_lo": float(np.percentile(rb, 2.5)), "ci_hi": float(np.percentile(rb, 97.5)), "n_order_seeds": int(len(num))}
inv = lambda r: {"ratio": 1 / r["ratio"], "ci_lo": 1 / r["ci_hi"], "ci_hi": 1 / r["ci_lo"], "n_order_seeds": r["n_order_seeds"]}

# ---------------- load + regression gate + self-checks
DATA, missing, mism, ncomp, nvals, chk_fail, pre_bad = {}, [], [], 0, 0, [], []
for dn, (d, jA) in DIRS.items():
    for B in BUDS:
        cell = {m: {"A": {}, "base": [], "info": {}} for m in MODES}
        for m in MODES:
            for o in range(NS):
                f = os.path.join(rd, f"l7a_d{d}_B{B}_o{o}_{m}.json")
                if not os.path.exists(f): missing.append(f); continue
                x = json.load(open(f)); assert x["budget"] == B and x["order_seed"] == o and x["n_batches"] == NB and x["m"] == 250 and x["mode"] == m
                for k, v in x["arms"].items(): cell[m]["A"].setdefault(k, []).append(v)
                cell[m]["base"].append(x["batch_base_errors"]); cell[m]["info"][o] = x["arm_info"]
                bad = [k for k, v in x["checks"].items() if not v]
                if bad: chk_fail.append({"cell": f"{dn} B={B}", "mode": m, "order_seed": o, "failed_selfchecks": bad})
                if m == "control_exact" and o < 8:
                    g = os.path.join(r6, f"amort_{jA}_B{B}_o{o}.json")
                    if not os.path.exists(g): mism.append({"cell": f"{dn} B={B}", "order_seed": o, "arm": "L6a file missing", "file": g}); continue
                    ref = json.load(open(g)); pairs = [(k, x["batch_base_errors"] if k == "batch_base_errors" else x["arms"][k], ref["batch_base_errors"] if k == "batch_base_errors" else ref["arms"][k]) for k in GATE_KEYS]
                    if B == 62: pairs.append(("freeze_plain_k3_vs_L6a_freeze_plain_125", x["arms"]["freeze_plain_k3"], ref["arms"]["freeze_plain_125"]))
                    for k, a_, r_ in pairs:
                        ncomp += 1; nvals += len(r_); diffs = [i for i in range(len(r_)) if a_[i] != r_[i]] if len(a_) == len(r_) else list(range(max(len(a_), len(r_))))
                        if diffs: mism.append({"cell": f"{dn} B={B}", "order_seed": o, "arm": k, "batches_0idx": diffs, "l7a": a_, "l6a": r_})
        for k, v in cell["drift"]["A"].items():  # nothing may differ before the switch: drift vs control_z, batches 1..6, every arm, every seed
            if k in cell["control_z"]["A"] and len(v) == NS and len(cell["control_z"]["A"][k]) == NS:
                bad = [o for o in range(NS) if v[o][:SW] != cell["control_z"]["A"][k][o][:SW]]
                if bad: pre_bad.append({"cell": f"{dn} B={B}", "arm": k, "order_seeds": bad})
        if cell["drift"]["base"] and cell["control_z"]["base"] and len(cell["drift"]["base"]) == NS == len(cell["control_z"]["base"]):
            bad = [o for o in range(NS) if cell["drift"]["base"][o][:SW] != cell["control_z"]["base"][o][:SW]]
            if bad: pre_bad.append({"cell": f"{dn} B={B}", "arm": "batch_base_errors", "order_seeds": bad})
        DATA[(dn, B)] = {m: {"A": {k: np.array(v, float) for k, v in cell[m]["A"].items()}, "base": np.array(cell[m]["base"], float), "info": cell[m]["info"]} for m in MODES}
nd = sum(len(m.get("batches_0idx", [1])) for m in mism); n_expected = 2 * (8 * len(GATE_KEYS) * 2 + 8)  # 2 directions x (2 B x 8 seeds x 5 keys + B = 62 x 8 seeds freeze)
gate = {"passed": bool(not missing and not mism and not chk_fail and not pre_bad and ncomp == n_expected), "vector_comparisons": ncomp, "vector_comparisons_expected": n_expected, "batch_values_compared": nvals, "batch_value_differences": nd,
        "keys": GATE_KEYS + ["freeze_plain_k3 vs L6a freeze_plain_125 (B=62)"], "order_seeds": "0..7", "missing_files": missing[:20], "n_missing_files": len(missing), "mismatches_first20": mism[:20],
        "in_job_selfcheck_failures": chk_fail[:20], "n_in_job_selfcheck_failures": len(chk_fail), "pre_switch_drift_vs_control_z_differences": pre_bad[:20], "n_pre_switch_differences": len(pre_bad)}
if not gate["passed"]:
    json.dump({"verdict": {"outcome": "VOID", "regression_gate_passed": False}, "regression_gate_vs_L6a": gate}, open(os.path.join(rd, "result.json"), "w"), indent=1)
    print("VOID: regression gate / self-checks did not pass", {k: v for k, v in gate.items() if k not in ("mismatches_first20", "in_job_selfcheck_failures", "pre_switch_drift_vs_control_z_differences", "missing_files")})
    print(json.dumps(mism[:3])); print(chk_fail[:3]); print(pre_bad[:3]); sys.exit(3)

# ---------------- per-cell analysis
def ret(A, k):
    """per-batch ratio to thr_agreement (ratio of means over the 16 order seeds) for batches 7..12; (first post-switch batch with ratio >= 1.03, first from which it stays >= 1.03); 1 = batch 7, None = never."""
    r = [float(A[k][:, b].sum() / A["thr_agreement"][:, b].sum()) for b in range(SW, NB)]
    return r, next((i + 1 for i, x in enumerate(r) if x >= RET_R), None), next((i + 1 for i in range(len(r)) if all(x >= RET_R for x in r[i:])), None)

res = {"protocol": __doc__.split("\n")[1:], "cells": {}}
for (dn, B), C in DATA.items():
    A, Cz, Ce = C["drift"]["A"], C["control_z"]["A"], C["control_exact"]["A"]
    assert all(a in A and a in Cz and a in Ce for a in ARMS) and all(v.shape == (NS, NB) for v in A.values())
    S = lambda arr, k, sl, g=slice(SW, NB): arr[k][sl, g].sum(1)  # per-seed hits over batches g for the seeds in sl
    out = {"direction": dn, "budget": B, "n_order_seeds": NS, "n_batches": NB, "mean_base_errors_per_batch_pre_switch_table_A": float(C["drift"]["base"][:, :SW].mean()), "mean_base_errors_per_batch_post_switch_table_B": float(C["drift"]["base"][:, SW:].mean()),
           "mean_base_errors_per_batch_no_drift_control": float(C["control_z"]["base"].mean())}
    out["per_batch_mean_hits_drift"] = {k: [float(x) for x in A[k].mean(0)] for k in ARMS}; out["per_batch_ratio_vs_thr_drift"] = {k: [float(A[k][:, b].sum() / A["thr_agreement"][:, b].sum()) for b in range(NB)] for k in ARMS}
    out["per_batch_ratio_vs_thr_control_z"] = {k: [float(Cz[k][:, b].sum() / Cz["thr_agreement"][:, b].sum()) for b in range(NB)] for k in ARMS}
    tab = {}
    for k in ARMS:
        t = {"groups": {g: {"mean_errors_per_batch": float(A[k][:, sl].mean()), "over_thr": rat(A[k][:, sl].sum(1), A["thr_agreement"][:, sl].sum(1))} for g, sl in GRP.items()}}
        _, t["first_post_switch_batch_ge_1.03"], t["first_post_switch_batch_stays_ge_1.03"] = ret(A, k)
        t["groups_control_z_over_thr"] = {g: float(Cz[k][:, sl].sum() / Cz["thr_agreement"][:, sl].sum()) for g, sl in GRP.items()}
        t["groups_control_exact_over_thr"] = {g: float(Ce[k][:, sl].sum() / Ce["thr_agreement"][:, sl].sum()) for g, sl in GRP.items()}
        t["own_nodrift_control"] = {g: {"drift_over_control_z": float(A[k][:, sl].sum() / Cz[k][:, sl].sum()), "thr_normalised": float((A[k][:, sl].sum() / A["thr_agreement"][:, sl].sum()) / (Cz[k][:, sl].sum() / Cz["thr_agreement"][:, sl].sum()))} for g, sl in GRP.items()}
        t["total_12_over_thr_control_z_vs_exact"] = {"control_z": float(Cz[k].sum() / Cz["thr_agreement"].sum()), "control_exact": float(Ce[k].sum() / Ce["thr_agreement"].sum())}
        if k in PERSIST + ["freeze_plain_k3", "reset_at_switch"]: t["batches_7_12_over_freeze_plain_k6_16seeds"] = rat(S(A, k, slice(None)), S(A, "freeze_plain_k6", slice(None)))
        tab[k] = t
    out["table"] = tab
    out["fit_sets"] = {k: {"n_fit_mean": float(np.mean([C["drift"]["info"][o][k]["n_fit"] for o in range(NS)])), "fit_err_rate_mean": float(np.mean([C["drift"]["info"][o][k]["fit_err_rate"] for o in range(NS)])),
                           "n_fallback_thr_of_16": int(sum(C["drift"]["info"][o][k]["fallback_thr"] for o in range(NS)))} for k in ("freeze_plain_k3", "freeze_plain_k6")}
    sel = lambda sl: max(PERSIST, key=lambda k: S(A, k, sl).mean())  # ties go to the first arm in PERSIST
    bA, bB = sel(FA), sel(FB); bestW16 = max(WIN, key=lambda k: S(A, k, slice(None)).mean())
    out["selection"] = {"best_persistent_on_seeds0_7": bA, "best_persistent_on_seeds8_15": bB, "mean_hits_batches_7_12_by_persistent_arm_seeds0_7": {k: float(S(A, k, FA).mean()) for k in PERSIST},
                        "mean_hits_batches_7_12_by_persistent_arm_seeds8_15": {k: float(S(A, k, FB).mean()) for k in PERSIST}, "best_forgetting_arm_by_post_switch_hits_all16": bestW16,
                        "best_window_on_seeds0_7": max(WIN, key=lambda k: S(A, k, FA).mean()), "best_window_on_seeds8_15": max(WIN, key=lambda k: S(A, k, FB).mean())}
    def fold_stats(arm_of):  # arm_of: {order seed: persistent arm reported for that seed}
        sd = sorted(arm_of); pb = lambda sl: np.array([A[arm_of[s]][s, sl].sum() for s in sd], float); fz = lambda sl, k: np.array([A[k][s, sl].sum() for s in sd], float)  # pb: reported persistent arm, fz: a fixed arm, per-seed hits over batches sl
        r_ctl = rat(np.array([Cz[arm_of[s]][s, GRP["7-12"]].sum() for s in sd], float), np.array([Cz["freeze_plain_k6"][s, GRP["7-12"]].sum() for s in sd], float))  # same arms, same seeds, no drift (control_z, batches 7-12 all from table A)
        r_bf = rat(pb(GRP["7-12"]), fz(GRP["7-12"], "freeze_plain_k6")); r_bt79 = rat(pb(GRP["7-9 post-switch"]), fz(GRP["7-9 post-switch"], "thr_agreement")); r_bt1012 = rat(pb(GRP["10-12 recovery"]), fz(GRP["10-12 recovery"], "thr_agreement"))
        return {"best_arm": f"{arm_of[0]} (seeds 0-7), {arm_of[8]} (seeds 8-15)" if len(set(arm_of.values())) > 1 else arm_of[sd[0]], "best_over_freeze_plain_k6": r_bf, "freeze_plain_k6_over_best": inv(r_bf), "best_over_freeze_plain_k3": rat(pb(GRP["7-12"]), fz(GRP["7-12"], "freeze_plain_k3")),
                "best_over_thr_7_9": r_bt79, "best_over_thr_10_12": r_bt1012, "best_over_thr_7_12": rat(pb(GRP["7-12"]), fz(GRP["7-12"], "thr_agreement")), "best_over_carry_plain_7_12": rat(pb(GRP["7-12"]), fz(GRP["7-12"], "carry_plain")),
                "reset_at_switch_over_freeze_plain_k6_UPPER_BOUND": rat(fz(GRP["7-12"], "reset_at_switch"), fz(GRP["7-12"], "freeze_plain_k6")), "reset_at_switch_over_best_UPPER_BOUND": rat(fz(GRP["7-12"], "reset_at_switch"), pb(GRP["7-12"])),
                "freeze_plain_k6_over_thr_7_12": rat(fz(GRP["7-12"], "freeze_plain_k6"), fz(GRP["7-12"], "thr_agreement")),
                "control_z_same_arms_best_over_freeze_plain_k6_7_12": r_ctl, "drift_excess_best_over_freeze_vs_no_drift": float(r_bf["ratio"] / r_ctl["ratio"]),
                "pass_cell": bool(r_bf["ratio"] >= PASS_R and r_bf["ci_lo"] > 1.0 and r_bt1012["ratio"] >= 1.0), "kill_i_cell_freeze_ge_0.99_best": bool(1 / r_bf["ratio"] >= KILL_R), "kill_ii_cell_best_below_thr_in_7_9_and_10_12": bool(r_bt79["ratio"] < 1.0 and r_bt1012["ratio"] < 1.0)}
    folds = {"A": ({s: bA for s in range(8, 16)}, "select on order seeds 0-7, report on 8-15 (VERDICT)"), "B": ({s: bB for s in range(8)}, "select on 8-15, report on 0-7"),
             "pooled16": ({**{s: bB for s in range(8)}, **{s: bA for s in range(8, 16)}}, "each seed reported with the arm chosen on the other half")}
    out["verdict_folds"] = {fn: {"desc": d_, **fold_stats(ao)} for fn, (ao, d_) in folds.items()}; res["cells"][f"{dn} B={B}"] = out

# ---------------- verdict
def cell_eval(fold):
    rows = {cn: {"best_arm": c["verdict_folds"][fold]["best_arm"], "best_over_freeze_k6": c["verdict_folds"][fold]["best_over_freeze_plain_k6"]["ratio"], "ci_lo": c["verdict_folds"][fold]["best_over_freeze_plain_k6"]["ci_lo"],
                "best_over_thr_7_9": c["verdict_folds"][fold]["best_over_thr_7_9"]["ratio"], "best_over_thr_10_12": c["verdict_folds"][fold]["best_over_thr_10_12"]["ratio"], "pass_cell": c["verdict_folds"][fold]["pass_cell"],
                "kill_i_cell": c["verdict_folds"][fold]["kill_i_cell_freeze_ge_0.99_best"], "kill_ii_cell": c["verdict_folds"][fold]["kill_ii_cell_best_below_thr_in_7_9_and_10_12"]} for cn, c in res["cells"].items()}
    nP, nKi, nKii = (sum(r[k] for r in rows.values()) for k in ("pass_cell", "kill_i_cell", "kill_ii_cell"))
    return {"cells": rows, "n_pass_cells_of_4": int(nP), "n_kill_i_cells_of_4": int(nKi), "n_kill_ii_cells_of_4": int(nKii), "PASS": bool(nP >= 3), "KILL_i": bool(nKi >= 3), "KILL_ii": bool(nKii >= 3), "KILL": bool(nKi >= 3 or nKii >= 3)}
vA, vB, vP = cell_eval("A"), cell_eval("B"), cell_eval("pooled16")
outcome = "PASS" if vA["PASS"] else "KILL" if vA["KILL"] else "INCONCLUSIVE"
nbest_cp = sum(r["best_arm"] == "carry_plain" for r in vA["cells"].values())
res["verdict"] = {"outcome": outcome, "regression_gate_passed": True, "verdict_fold": "best persistent arm chosen per cell on order seeds 0-7 by batches 7-12 hits, reported on order seeds 8-15 (8 order seeds)", "foldA": vA, "crosscheck_swapped_fold": vB, "crosscheck_pooled16": vP,
                  "kill_conditions_met": {"i_freeze_plain_k6_ge_0.99_best_in_3_of_4": vA["KILL_i"], "ii_best_persistent_below_thr_in_7_9_and_10_12_in_3_of_4": vA["KILL_ii"]},
                  "null_case_unbounded_carry_plain_is_best_in_cells_of_4": int(nbest_cp), "null_case_followup": "unbounded carry_plain is the verdict-fold best persistent arm in >= 3 of 4 cells: the drift is too mild to separate forgetting from carrying; the next step is a harsher drift (e.g. flip the sign of one feature's relation to the label) before concluding persistence needs no forgetting horizon" if nbest_cp >= 3 else None}
res["regression_gate_vs_L6a"] = {k: v for k, v in gate.items() if k not in ("mismatches_first20",)}
res["headline_fold_A"] = {cn: {"best_persistent_arm": r["best_arm"], "best_over_freeze_plain_k6": r["best_over_freeze_k6"], "ci_lo": r["ci_lo"], "best_over_thr_10_12": r["best_over_thr_10_12"]} for cn, r in vA["cells"].items()}  # best_over_freeze_plain_k6 is the ratio
res["notes"] = ["Direction confound: 7B-AWQ judge labels are cleaner than 3B labels (judge_error rate 0.438 vs 0.473, label correlation 0.51, judge feature means differ), so direction 1 moves to an easier table and direction 2 to a noisier one; the two directions are reported separately and never pooled into one claim.",
                "Standardization departure from L6a: in the drift run and in control_z each batch is standardized on its own rows (label-free) and carried rows keep the z they had when collected; control_exact keeps the L6a form (raw features with the StandardScaler fitted on the labels, global z for carry_offset_lam1) and is the regression gate. control_z is the like-for-like no-drift reference.",
                "Selection departs from the 8-seed spec exactly as the L6a run did: best persistent arm chosen on order seeds 0-7, reported on 8-15, swapped fold and pooled 16 as cross-checks. The 16 order seeds permute the same 3,000 prompts, so order-seed CIs are optimistic (order and tie-break variation only, not prompt-sampling variation).",
                "reset_at_switch knows the switch and is an upper bound, never an arm in the verdict. freeze_plain_k3 / k6 are frozen before the switch (k <= 6) and never enter the persistent selection.",
                "The drift is a single step at batch 7 between two real judge tables of one population and one task (GSM8K judge_error labels from E04); the 3B judge barely passed the E04 validity gate; offline replay."]
json.dump(res, open(os.path.join(rd, "result.json"), "w"), indent=1)

# ---------------- ledger markdown
L = []; P = L.append
f3 = lambda r: f"{r['ratio']:.3f} ({r['ci_lo']:.3f} to {r['ci_hi']:.3f})"; fb = lambda x: "never" if x is None else str(x)
P(f"# L7a ledger table (CPU replay, regression gate vs runs/L6a: {gate['batch_value_differences']} differences in {gate['batch_values_compared']} per-batch values ({gate['vector_comparisons']} vectors), in-job self-check failures {gate['n_in_job_selfcheck_failures']}, pre-switch drift vs control_z differences {gate['n_pre_switch_differences']})")
P(f"\nVerdict: **{outcome}** (verdict fold: best persistent arm chosen on order seeds 0-7 by batches 7-12, reported on 8-15; PASS cells {vA['n_pass_cells_of_4']} of 4, kill (i) cells {vA['n_kill_i_cells_of_4']} of 4, kill (ii) cells {vA['n_kill_ii_cells_of_4']} of 4). "
  f"Swapped fold: PASS {vB['n_pass_cells_of_4']}/4, kill (i) {vB['n_kill_i_cells_of_4']}/4, kill (ii) {vB['n_kill_ii_cells_of_4']}/4. Pooled 16: PASS {vP['n_pass_cells_of_4']}/4, kill (i) {vP['n_kill_i_cells_of_4']}/4, kill (ii) {vP['n_kill_ii_cells_of_4']}/4.")
P("\nCaveats: a single step drift between two real judge tables; the 16 order seeds permute the same 3,000 prompts, so CIs are optimistic; direction confound (7B-AWQ labels are cleaner); per-batch z departure from L6a (gate runs the L6a form separately); one population, offline replay on weak-judge labels.")
P("\n## Headline (verdict fold A, 8 order seeds 8-15)\n"); P("| cell | best persistent (chosen on 0-7) | best / freeze_plain_k6, batches 7-12 (95% CI) | best / thr 7-9 | best / thr 10-12 | best / carry_plain | same arm / freeze_plain_k6 with no drift (control_z) | drift excess | reset_at_switch / freeze_plain_k6 (upper bound) | pass | kill (i) | kill (ii) |"); P("|---|---|---|---|---|---|---|---|---|---|---|---|")
for cn, c in res["cells"].items():
    f = c["verdict_folds"]["A"]; P(f"| {cn} | {f['best_arm']} | {f3(f['best_over_freeze_plain_k6'])} | {f['best_over_thr_7_9']['ratio']:.3f} | {f['best_over_thr_10_12']['ratio']:.3f} | {f['best_over_carry_plain_7_12']['ratio']:.3f} | {f3(f['control_z_same_arms_best_over_freeze_plain_k6_7_12'])} | {f['drift_excess_best_over_freeze_vs_no_drift']:.3f} | {f3(f['reset_at_switch_over_freeze_plain_k6_UPPER_BOUND'])} | {f['pass_cell']} | {f['kill_i_cell_freeze_ge_0.99_best']} | {f['kill_ii_cell_best_below_thr_in_7_9_and_10_12']} |")
for cn, c in res["cells"].items():
    tb = c["table"]; P(f"\n## {cn} (per-step budget {c['budget']} of 250; mean errors per batch: table A {c['mean_base_errors_per_batch_pre_switch_table_A']:.1f}, table B {c['mean_base_errors_per_batch_post_switch_table_B']:.1f}, no-drift control {c['mean_base_errors_per_batch_no_drift_control']:.1f}; 16 order seeds)")
    P("\nDrift run, all 16 order seeds. Errors per batch and ratio to thr_agreement in the batch group (CI over 16 order seeds); first post-switch batch with per-batch ratio >= 1.03 (1 = batch 7) / first from which it stays >= 1.03; ratio to own no-drift control (control_z, same batches) over batches 7-12 raw / thr-normalised; batches 7-12 / freeze_plain_k6.\n")
    P("| arm | err/batch 4-6 | 7-9 | 10-12 | / thr 4-6 | / thr 7-9 | / thr 10-12 | first >= 1.03 / stays | / own control 7-12 | thr-normalised | 7-12 / freeze_k6 |"); P("|---|---|---|---|---|---|---|---|---|---|---|")
    for k in ARMS:
        t = tb[k]; g = t["groups"]; fz = t.get("batches_7_12_over_freeze_plain_k6_16seeds"); oc = t["own_nodrift_control"]["7-12"]
        P(f"| {k} | " + " | ".join(f"{g[x]['mean_errors_per_batch']:.1f}" for x in list(GRP)[:3]) + " | " + " | ".join(f"{g[x]['over_thr']['ratio']:.3f}" for x in list(GRP)[:3]) + f" | {fb(t['first_post_switch_batch_ge_1.03'])} / {fb(t['first_post_switch_batch_stays_ge_1.03'])} | {oc['drift_over_control_z']:.3f} | {oc['thr_normalised']:.3f} | " + ("n/a" if fz is None else f3(fz)) + " |")
    P("\nNo-drift references (all batches from table A), ratio to thr_agreement: control_z batches 4-6 / 7-9 / 10-12 | control_exact 7-12 | 12-batch total control_z vs control_exact:\n"); P("| arm | control_z 4-6 | 7-9 | 10-12 | control_exact 7-12 | total 12 control_z | total 12 control_exact |"); P("|---|---|---|---|---|---|---|")
    for k in ARMS:
        t = tb[k]; z_ = t["groups_control_z_over_thr"]; P(f"| {k} | {z_['4-6 pre-switch']:.3f} | {z_['7-9 post-switch']:.3f} | {z_['10-12 recovery']:.3f} | {t['groups_control_exact_over_thr']['7-12']:.3f} | {t['total_12_over_thr_control_z_vs_exact']['control_z']:.3f} | {t['total_12_over_thr_control_z_vs_exact']['control_exact']:.3f} |")
    s = c["selection"]; P(f"\nSelection (mean hits over batches 7-12): best persistent on seeds 0-7 = {s['best_persistent_on_seeds0_7']}, on 8-15 = {s['best_persistent_on_seeds8_15']}; best forgetting arm on all 16 = {s['best_forgetting_arm_by_post_switch_hits_all16']}; "
      f"best window on 0-7 = {s['best_window_on_seeds0_7']}, on 8-15 = {s['best_window_on_seeds8_15']}.")
    P("\nFrozen fits (drift run, mean over 16 order seeds): " + "; ".join(f"{k}: n_fit {v['n_fit_mean']:.0f}, label error rate {v['fit_err_rate_mean']:.2f}" + (f", {v['n_fallback_thr_of_16']} one-class fallbacks" if v["n_fallback_thr_of_16"] else "") for k, v in c["fit_sets"].items()))
    P("\nArm / thr_agreement by batch index, drift run (ratio of means over 16 order seeds; the switch is between batch 6 and 7):\n"); P("| arm | " + " | ".join(str(i + 1) for i in range(NB)) + " |"); P("|---|" + "---|" * NB)
    for k in ARMS[1:]: P(f"| {k} | " + " | ".join(f"{x:.3f}" for x in c["per_batch_ratio_vs_thr_drift"][k]) + " |")
    P("\nVerdict folds (batches 7-12; CI over 8 order seeds, 16 for pooled):\n"); P("| fold | best persistent | best / freeze_k6 (95% CI) | freeze_k6 / best | best / freeze_k3 | best / thr 7-9 | best / thr 10-12 | best / carry_plain | reset / best (upper bound) |"); P("|---|---|---|---|---|---|---|---|---|")
    for fn, f in c["verdict_folds"].items():
        P(f"| {fn} | {f['best_arm']} | {f3(f['best_over_freeze_plain_k6'])} | {f['freeze_plain_k6_over_best']['ratio']:.3f} | {f['best_over_freeze_plain_k3']['ratio']:.3f} | {f['best_over_thr_7_9']['ratio']:.3f} | {f['best_over_thr_10_12']['ratio']:.3f} | {f['best_over_carry_plain_7_12']['ratio']:.3f} | {f['reset_at_switch_over_best_UPPER_BOUND']['ratio']:.3f} |")
open(os.path.join(rd, "ledger_table.md"), "w").write("\n".join(L) + "\n"); print("\n".join(L))
