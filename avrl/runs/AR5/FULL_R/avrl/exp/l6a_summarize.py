"""L6a summary (CPU). Usage: l6a_summarize.py <runs/L6a> [<runs/L5a>]
Reads amort_<judge>_B<B>_o<order seed>.json (judges 3B and 7B-AWQ, per-step B in 62, 125, order seeds 0..15; 12 disjoint batches of 250 each; hits = errors caught per batch, calibration labels charged and
counted as hits) and writes result.json and ledger_table.md. Ratios are ratios of means over order seeds (sum of hits over the 12 batches), CI = percentile bootstrap (10,000 resamples) over order seeds.

Regression gate (order seeds 0..7 only): per-batch hits of thr_agreement, uniform, restart_offset, carry_plain, carry_offset_lam1, static_out (and batch_base_errors) must equal runs/L5a/carry_<judge>_B<B>_o<seed>.json
exactly; any difference, a missing file or a failed in-job self-check voids the run (result.json outcome VOID, exit code 3).

Selection protocol (as L5a / L2a): the best amortized fixed arm of the 9 pre-registered ones (amort_{seed,rand,split}_{125,250,500}) is chosen per cell by cumulative 12-batch hits (mean over the selection seeds, ties
to the first in that order) on order seeds 0..7 and reported on order seeds 8..15 (VERDICT fold). Swapped fold (select on 8..15, report on 0..7) and pooled 16 (each seed reported with the arm chosen on the other half) are
cross-checks. freeze_plain_{125,250,500} are diagnostics and never enter the selection or the verdict.

Pre-registered verdict (L6a spec), per cell = judge x per-step B (4 cells), verdict fold, all 12 batches:
  VOID          if the regression gate did not pass
  PASS          if carry_plain / best amortized fixed >= 1.03 with the bootstrap CI lower bound > 1 in at least 3 of the 4 cells (claim 2 survives the strongest persistent fixed baseline)
  KILL          if best amortized fixed >= 0.99 x carry_plain in at least 3 of the 4 cells (the persistent edge is a label-count effect)
  INCONCLUSIVE  otherwise (report the per-cell table and the freeze_plain decomposition)"""
import sys, os, json
import numpy as np

rd = sys.argv[1]; r5 = sys.argv[2] if len(sys.argv) > 2 else os.path.join(os.path.dirname(os.path.abspath(rd)), "L5a")
JUDGES = {"3B": "Qwen2.5-3B-Instruct", "7B-AWQ": "Qwen2.5-7B-Instruct-AWQ"}; BUDS = (62, 125); NS, NB, NBOOT, SS0 = 16, 12, 10000, 6  # steady state = batches 7..12 (index 6..11)
CAL_N, FAMS = (125, 250, 500), ("seed", "rand", "split"); PASS_R, KILL_R = 1.03, 0.99
AM = [f"amort_{f}_{n}" for f in FAMS for n in CAL_N]; FZ = [f"freeze_plain_{n}" for n in CAL_N]
REF6 = ["thr_agreement", "uniform", "restart_offset", "carry_plain", "carry_offset_lam1", "static_out"]; GATE_KEYS = REF6 + ["batch_base_errors"]
TABLE_ARMS = AM + FZ + ["carry_plain", "carry_offset_lam1", "thr_agreement", "static_out", "restart_offset", "uniform"]
FA, FB = slice(0, 8), slice(8, 16)
rng = np.random.default_rng(0); IDX = {n: rng.integers(0, n, (NBOOT, n)) for n in (8, 16)}

def rat(num, den):
    """ratio of means of per-seed vectors with a percentile bootstrap CI over the order seeds (len 8 or 16)."""
    idx = IDX[len(num)]; rb = num[idx].mean(1) / den[idx].mean(1)
    return {"ratio": float(num.mean() / den.mean()), "ci_lo": float(np.percentile(rb, 2.5)), "ci_hi": float(np.percentile(rb, 97.5)), "n_order_seeds": int(len(num))}

# ---------------- load + regression gate + self-checks
CELLS, missing, mism, ncomp, nvals, chk_fail = {}, [], [], 0, 0, []
for lab, j in JUDGES.items():
    for B in BUDS:
        A, base, info = {}, [], {}
        for o in range(NS):
            f = os.path.join(rd, f"amort_{j}_B{B}_o{o}.json")
            if not os.path.exists(f): missing.append(f); continue
            d = json.load(open(f)); assert d["budget"] == B and d["order_seed"] == o and d["n_batches"] == NB and d["m"] == 250
            for k, v in d["arms"].items(): A.setdefault(k, []).append(v)
            base.append(d["batch_base_errors"]); info[o] = d["arm_info"]
            bad = [k for k, v in d["checks"].items() if not v]
            if bad: chk_fail.append({"cell": f"{lab} B={B}", "order_seed": o, "failed_selfchecks": bad})
            if o < 8:
                g = os.path.join(r5, f"carry_{j}_B{B}_o{o}.json")
                if not os.path.exists(g): mism.append({"cell": f"{lab} B={B}", "order_seed": o, "arm": "L5a file missing", "file": g}); continue
                ref = json.load(open(g))
                for k in GATE_KEYS:
                    a_ = d["batch_base_errors"] if k == "batch_base_errors" else d["arms"][k]; r_ = ref["batch_base_errors"] if k == "batch_base_errors" else ref["arms"][k]
                    ncomp += 1; nvals += len(r_)
                    diffs = [i for i in range(len(r_)) if a_[i] != r_[i]] if len(a_) == len(r_) else list(range(max(len(a_), len(r_))))
                    if diffs: mism.append({"cell": f"{lab} B={B}", "order_seed": o, "arm": k, "batches_0idx": diffs, "l6a": a_, "l5a": r_})
        CELLS[(lab, B)] = {"A": {k: np.array(v, float) for k, v in A.items()}, "base": np.array(base), "info": info}
nd = sum(len(m.get("batches_0idx", [1])) for m in mism)
gate = {"passed": bool(not missing and not mism and not chk_fail and ncomp == 4 * 8 * len(GATE_KEYS)), "vector_comparisons": ncomp, "batch_values_compared": nvals, "batch_value_differences": nd,
        "keys": GATE_KEYS, "order_seeds": "0..7", "missing_l6a_files": missing, "mismatches_first20": mism[:20], "in_job_selfcheck_failures": chk_fail[:20], "n_in_job_selfcheck_failures": len(chk_fail)}
if not gate["passed"]:
    json.dump({"verdict": {"outcome": "VOID", "regression_gate_passed": False}, "regression_gate_vs_L5a": gate}, open(os.path.join(rd, "result.json"), "w"), indent=1)
    print("VOID: regression gate / self-checks did not pass", {k: v for k, v in gate.items() if k not in ("mismatches_first20", "in_job_selfcheck_failures")}); print(json.dumps(mism[:3])); print(chk_fail[:3]); sys.exit(3)

# ---------------- per-cell analysis
def first_gt1(cum):
    """(first batch with cumulative ratio > 1.00, first batch from which it stays > 1.00 through batch 12); 1-indexed, None if never."""
    gt = [c > 1.0 + 1e-9 for c in cum]; first = next((i + 1 for i, g in enumerate(gt) if g), None)
    stay = next((i + 1 for i in range(NB) if all(gt[i:])), None); return first, stay

res = {"protocol": __doc__.split("\n")[1:], "cells": {}}
for (lab, B), C in CELLS.items():
    A = C["A"]; assert all(a in A for a in TABLE_ARMS + ["restart_plain"]) and all(v.shape == (NS, NB) for v in A.values())
    T = lambda k, sl=slice(None): A[k][sl].sum(1); S = lambda k, sl=slice(None): A[k][sl, SS0:].sum(1)
    out = {"budget": B, "n_order_seeds": NS, "n_batches": NB, "mean_base_errors_per_batch": float(C["base"].mean()), "population_error_rate": float(C["base"][0].sum() / (NB * 250))}
    out["per_batch_mean_hits"] = {k: [float(x) for x in A[k].mean(0)] for k in TABLE_ARMS}
    out["per_batch_ratio_vs_thr"] = {k: [float(A[k][:, b].sum() / A["thr_agreement"][:, b].sum()) for b in range(NB)] for k in TABLE_ARMS}
    cum_thr = A["thr_agreement"].sum(0).cumsum(); tab = {}
    for k in TABLE_ARMS:
        cum = (A[k].sum(0).cumsum() / cum_thr).tolist(); f1, st = first_gt1(cum)
        t = {"mean_total_hits_12": float(T(k).mean()), "total_over_thr": rat(T(k), T("thr_agreement")), "arm_over_carry_plain": rat(T(k), T("carry_plain")), "steady_7_12_over_thr": rat(S(k), S("thr_agreement")),
             "steady_7_12_arm_over_carry_plain": rat(S(k), S("carry_plain")), "cum_ratio_vs_thr": cum, "first_batch_cum_gt_1.00": f1, "first_batch_stays_gt_1.00": st}
        if k not in ("thr_agreement", "carry_plain", "uniform"):
            t["gain_captured_of_carry_plain_total"] = rat(T(k) - T("thr_agreement"), T("carry_plain") - T("thr_agreement")); t["gain_captured_of_carry_plain_steady_7_12"] = rat(S(k) - S("thr_agreement"), S("carry_plain") - S("thr_agreement"))
        tab[k] = t
    out["table"] = tab
    out["fit_sets"] = {k: {"n_fit_mean": float(np.mean([C["info"][o][k]["n_fit"] for o in range(NS)])), "fit_err_rate_mean": float(np.mean([C["info"][o][k].get("cal_err_rate", C["info"][o][k].get("fit_err_rate")) for o in range(NS)])),
                           "n_fallback_thr_of_16": int(sum(C["info"][o][k]["fallback_thr"] for o in range(NS)))} for k in AM + FZ}
    sel = lambda sl: max(AM, key=lambda k: A[k][sl].sum(1).mean())  # ties go to the first arm in AM
    bA, bB = sel(FA), sel(FB)
    out["selection"] = {"best_on_seeds0_7": bA, "best_on_seeds8_15": bB, "mean_total_hits_by_amort_arm_seeds0_7": {k: float(T(k, FA).mean()) for k in AM}, "mean_total_hits_by_amort_arm_seeds8_15": {k: float(T(k, FB).mean()) for k in AM}}
    cp = lambda sl: T("carry_plain", sl)
    folds = {"A": (bA, FB, "select on order seeds 0-7, report on 8-15 (VERDICT)"), "B": (bB, FA, "select on 8-15, report on 0-7")}
    vf = {}
    for fn, (bk, sl, desc) in folds.items():
        r = rat(cp(sl), T(bk, sl)); vf[fn] = {"desc": desc, "best_arm": bk, "carry_plain_over_best": r, "best_over_carry_plain": {"ratio": 1 / r["ratio"], "ci_lo": 1 / r["ci_hi"], "ci_hi": 1 / r["ci_lo"]},
                                             "carry_offset_lam1_over_best": rat(T("carry_offset_lam1", sl), T(bk, sl)), "carry_plain_over_thr": rat(cp(sl), T("thr_agreement", sl)), "best_over_thr": rat(T(bk, sl), T("thr_agreement", sl)),
                                             "thr_over_best": rat(T("thr_agreement", sl), T(bk, sl))}
    numP = np.concatenate([cp(FA), cp(FB)]); denP = np.concatenate([T(bB, FA), T(bA, FB)]); r = rat(numP, denP)
    vf["pooled16"] = {"desc": "each seed reported with the arm chosen on the other half", "best_arm": f"{bB} (seeds 0-7), {bA} (seeds 8-15)", "carry_plain_over_best": r, "best_over_carry_plain": {"ratio": 1 / r["ratio"], "ci_lo": 1 / r["ci_hi"], "ci_hi": 1 / r["ci_lo"]},
                      "carry_offset_lam1_over_best": rat(np.concatenate([T("carry_offset_lam1", FA), T("carry_offset_lam1", FB)]), denP)}
    out["verdict_folds"] = vf; res["cells"][f"{lab} B={B}"] = out

# ---------------- verdict
def cell_eval(fold):
    rows = {}
    for cn, c in res["cells"].items():
        f = c["verdict_folds"][fold]; r, br = f["carry_plain_over_best"], f["best_over_carry_plain"]
        rows[cn] = {"best_arm": f["best_arm"], "carry_plain_over_best": r["ratio"], "ci_lo": r["ci_lo"], "ci_hi": r["ci_hi"], "best_over_carry_plain": br["ratio"], "pass_cell_ge_1.03_and_ci_lo_gt_1": bool(r["ratio"] >= PASS_R and r["ci_lo"] > 1.0),
                    "kill_cell_best_ge_0.99_carry": bool(br["ratio"] >= KILL_R), "best_at_least_3pct_below_carry": bool(br["ratio"] <= 1 / PASS_R)}
    n_pass = sum(r["pass_cell_ge_1.03_and_ci_lo_gt_1"] for r in rows.values()); n_kill = sum(r["kill_cell_best_ge_0.99_carry"] for r in rows.values())
    return {"cells": rows, "n_pass_cells_of_4": int(n_pass), "n_kill_cells_of_4": int(n_kill), "PASS": bool(n_pass >= 3), "KILL": bool(n_kill >= 3),
            "n_best_at_least_3pct_below_of_4": int(sum(r["best_at_least_3pct_below_carry"] for r in rows.values()))}
vA, vB, vP = cell_eval("A"), cell_eval("B"), cell_eval("pooled16")
outcome = "PASS" if vA["PASS"] else "KILL" if vA["KILL"] else "INCONCLUSIVE"
res["verdict"] = {"outcome": outcome, "regression_gate_passed": True, "verdict_fold": "best amortized arm chosen per cell on order seeds 0-7, reported on order seeds 8-15 (8 order seeds, all 12 batches)", "foldA": vA,
                  "crosscheck_swapped_fold": vB, "crosscheck_pooled16": vP}
res["regression_gate_vs_L5a"] = {k: v for k, v in gate.items() if k not in ("mismatches_first20",)}
res["notes"] = ["Batches are exchangeable draws of one fixed policy, so there is no drift by construction; unbounded carry is an upper bound on what persistence can deliver and a once-fitted rule loses nothing to staleness.",
                "The 16 order seeds permute the same 3,000 prompts, so order-seed CIs are optimistic (order and tie-break variation only, not prompt-sampling variation); each fold has 8 order seeds.",
                "n_fit: amort_seed_n uses every label paid in its ceil(n/B) calibration batches (k*B >= n: 186/310/558 at B = 62, exactly 125/250/500 at B = 125), amort_rand_n and amort_split_n fit on exactly n random labels; freeze_plain_n refits on the k*B labels in hand at the end of batch k.",
                "Calibration labels are charged inside the per-batch budget and counted as hits in every arm (amort_rand / amort_split lose hits while calibrating because random picks catch errors at about the base rate).",
                "static_out is a logistic rule fit with free labels on the 2,750 prompts outside the batch: an optimistic reference and never a policy.",
                "One population, one task (GSM8K judge_error labels from E04), offline replay on weak-judge labels; the 3B judge barely passed the E04 validity gate."]
json.dump(res, open(os.path.join(rd, "result.json"), "w"), indent=1)

# ---------------- ledger markdown
L = []; P = L.append
f3 = lambda r: f"{r['ratio']:.3f} ({r['ci_lo']:.3f} to {r['ci_hi']:.3f})"; fb = lambda x: "never" if x is None else str(x)
pc = lambda r: f"{r['ratio']:.0%} ({r['ci_lo']:.0%} to {r['ci_hi']:.0%})"
P(f"# L6a ledger table (CPU replay, regression gate vs runs/L5a: {gate['batch_value_differences']} differences in {gate['batch_values_compared']} per-batch values ({gate['vector_comparisons']} vectors), in-job self-check failures {gate['n_in_job_selfcheck_failures']})")
P(f"\nVerdict: **{outcome}** (verdict fold: best amortized arm chosen on order seeds 0-7, reported on 8-15; all 12 batches; PASS cells {vA['n_pass_cells_of_4']} of 4, KILL cells {vA['n_kill_cells_of_4']} of 4). "
  f"Swapped fold: PASS {vB['n_pass_cells_of_4']}/4, KILL {vB['n_kill_cells_of_4']}/4. Pooled 16: PASS {vP['n_pass_cells_of_4']}/4, KILL {vP['n_kill_cells_of_4']}/4.")
P("\nCaveats: no drift by construction; the 16 order seeds permute the same 3,000 prompts, so CIs are optimistic; one population, offline replay on weak-judge labels; the 3B judge barely passed the E04 validity gate.")
for cn, c in res["cells"].items():
    tb = c["table"]; P(f"\n## {cn} (per-step budget {c['budget']} of 250; mean errors per batch {c['mean_base_errors_per_batch']:.1f}; 16 order seeds x 12 batches)")
    P("\nAll 16 order seeds, all 12 batches (calibration labels charged and counted as hits). Gain captured = (arm - thr) / (carry_plain - thr). First batch = first batch whose cumulative ratio to thr_agreement is > 1.00 / first batch from which it stays > 1.00.\n")
    P("| arm | mean errors caught (12 batches) | / thr_agreement (95% CI) | / carry_plain (95% CI) | batches 7-12 / thr | gain captured, 12 batches (95% CI) | first batch cum > 1.00 / stays |"); P("|---|---|---|---|---|---|---|")
    for k in TABLE_ARMS:
        t = tb[k]; g = t.get("gain_captured_of_carry_plain_total")
        P(f"| {k} | {t['mean_total_hits_12']:.1f} | {f3(t['total_over_thr'])} | {f3(t['arm_over_carry_plain'])} | {t['steady_7_12_over_thr']['ratio']:.3f} | " + ("n/a" if g is None else pc(g)) + f" | {fb(t['first_batch_cum_gt_1.00'])} / {fb(t['first_batch_stays_gt_1.00'])} |")
    s = c["selection"]; P(f"\nSelection: best amortized arm on order seeds 0-7 = {s['best_on_seeds0_7']} (mean total hits {s['mean_total_hits_by_amort_arm_seeds0_7'][s['best_on_seeds0_7']]:.1f}), on 8-15 = {s['best_on_seeds8_15']} ({s['mean_total_hits_by_amort_arm_seeds8_15'][s['best_on_seeds8_15']]:.1f}).")
    P("\nFit sets (mean over 16 order seeds): " + "; ".join(f"{k}: n_fit {v['n_fit_mean']:.0f}, label error rate {v['fit_err_rate_mean']:.2f}" + (f", {v['n_fallback_thr_of_16']} one-class fallbacks" if v["n_fallback_thr_of_16"] else "") for k, v in c["fit_sets"].items()))
    P("\nArm / thr_agreement by batch index (ratio of means over 16 order seeds):\n"); P("| arm | " + " | ".join(str(i + 1) for i in range(NB)) + " |"); P("|---|" + "---|" * NB)
    for k in AM + FZ + ["carry_plain", "carry_offset_lam1", "static_out"]: P(f"| {k} | " + " | ".join(f"{x:.3f}" for x in c["per_batch_ratio_vs_thr"][k]) + " |")
    P("\nVerdict folds (all 12 batches; CI over 8 order seeds, 16 for pooled):\n"); P("| fold | best amortized arm | carry_plain / best (95% CI) | best / carry_plain | carry_offset_lam1 / best | carry_plain / thr | best / thr |"); P("|---|---|---|---|---|---|---|")
    for fn, f in c["verdict_folds"].items():
        P(f"| {fn} | {f['best_arm']} | {f3(f['carry_plain_over_best'])} | {f['best_over_carry_plain']['ratio']:.3f} | {f3(f['carry_offset_lam1_over_best'])} | " + (f3(f["carry_plain_over_thr"]) if "carry_plain_over_thr" in f else "") + " | " + (f3(f["best_over_thr"]) if "best_over_thr" in f else "") + " |")
P("\n## Verdict detail (fold A = verdict)\n"); P("| cell | best arm (chosen on 0-7) | carry_plain / best (CI lo) | best / carry_plain | pass cell | kill cell | best >= 3% below |"); P("|---|---|---|---|---|---|---|")
for cn, r in vA["cells"].items(): P(f"| {cn} | {r['best_arm']} | {r['carry_plain_over_best']:.3f} ({r['ci_lo']:.3f}) | {r['best_over_carry_plain']:.3f} | {r['pass_cell_ge_1.03_and_ci_lo_gt_1']} | {r['kill_cell_best_ge_0.99_carry']} | {r['best_at_least_3pct_below_carry']} |")
open(os.path.join(rd, "ledger_table.md"), "w").write("\n".join(L) + "\n"); print("\n".join(L))
