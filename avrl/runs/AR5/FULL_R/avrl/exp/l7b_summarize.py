"""L7b summary (CPU). Usage: l7b_summarize.py <runs/L7b> [<runs/L6a>]
Reads design_<judge>_B<B>_o<order seed>.json (judges 3B and 7B-AWQ, per-step B in 62, 125, order seeds 0..15; 12 disjoint batches of 250 each; hits = errors caught per batch, calibration labels charged and
counted as hits) and writes result.json and ledger_table.md. Ratios are ratios of means over order seeds (sum of hits over the batches named), CI = percentile bootstrap (10,000 resamples) over order seeds.
The two judges and the two B values are separate cells and are never pooled.

Regression gate (order seeds 0..15, all 4 cells): per-batch hits of thr_agreement, carry_plain, amort_{seed,rand,split}_{125,250,500}, freeze_plain_{125,250,500} (and batch_base_errors) must equal
runs/L6a/amort_<judge>_B<B>_o<seed>.json exactly; any difference, a missing file or a failed in-job self-check voids the run (result.json outcome VOID, exit code 3). The other L6a reference arms (uniform,
restart_offset, restart_plain, static_out, carry_offset_lam1) are compared too and reported, but only the listed arms enter the gate.

Selection protocol (exactly as L6a): the best designed arm of the 6 pre-registered ones (amort_{strat,bnd}_{125,250,500}) is chosen per cell by cumulative 12-batch hits (mean over the selection seeds, ties to
the first in that order) on order seeds 0..7 and reported on order seeds 8..15 (VERDICT fold). Swapped fold (select on 8..15, report on 0..7) and pooled 16 (each seed reported with the arm chosen on the other half)
are cross-checks. The L6a arms (amort_seed/rand/split, freeze_plain) are references and never enter the selection or the verdict.

Pre-registered verdict (L7b spec), per cell = judge x per-step B (4 cells), verdict fold, all 12 batches:
  VOID          if the regression gate did not pass
  PASS          if carry_plain / best designed >= 1.03 with the bootstrap CI lower bound > 1 in at least 3 of the 4 cells (the learner's own label stream beats any static covering design; claim 2 survives)
  KILL          if best designed >= 0.99 x carry_plain in at least 3 of the 4 cells (the edge is available to a one-shot designed calibration)
  INCONCLUSIVE  otherwise; if additionally the post-calibration (batches k+1..12) ratio best designed / carry_plain is >= 0.99 in 4 of 4 cells the result is recorded as calibration cost only (feeds L7c)"""
import sys, os, json
import numpy as np, pandas as pd

rd = sys.argv[1]; r6 = sys.argv[2] if len(sys.argv) > 2 else os.path.join(os.path.dirname(os.path.abspath(rd)), "L6a")
JUDGES = {"3B": "Qwen2.5-3B-Instruct", "7B-AWQ": "Qwen2.5-7B-Instruct-AWQ"}; BUDS = (62, 125); NS, NB, NBOOT, SS0 = 16, 12, 10000, 6  # SS0: post-calibration window for arms without an n (batches 7..12)
CAL_N = (125, 250, 500); PASS_R, KILL_R = 1.03, 0.99
DES = [f"amort_{f}_{n}" for f in ("strat", "bnd") for n in CAL_N]; AM = [f"amort_{f}_{n}" for f in ("seed", "rand", "split") for n in CAL_N]; FZ = [f"freeze_plain_{n}" for n in CAL_N]
GATE_KEYS = ["thr_agreement", "carry_plain"] + AM + FZ + ["batch_base_errors"]; EXTRA_KEYS = ["uniform", "restart_offset", "restart_plain", "static_out", "carry_offset_lam1"]
TABLE_ARMS = DES + AM + FZ + ["carry_plain", "carry_offset_lam1", "thr_agreement", "static_out", "restart_offset", "uniform"]
COV_ARMS = [f"{f}_{n}" for n in CAL_N for f in ("amort_strat", "amort_bnd", "freeze_plain", "amort_rand")]
FA, FB = slice(0, 8), slice(8, 16)
rng = np.random.default_rng(0); IDX = {n: rng.integers(0, n, (NBOOT, n)) for n in (8, 16)}

def rat(num, den):
    """ratio of means of per-seed vectors with a percentile bootstrap CI over the order seeds (len 8 or 16)."""
    idx = IDX[len(num)]; rb = num[idx].mean(1) / den[idx].mean(1)
    return {"ratio": float(num.mean() / den.mean()), "ci_lo": float(np.percentile(rb, 2.5)), "ci_hi": float(np.percentile(rb, 97.5)), "n_order_seeds": int(len(num))}
inv = lambda r: {"ratio": 1 / r["ratio"], "ci_lo": 1 / r["ci_hi"], "ci_hi": 1 / r["ci_lo"], "n_order_seeds": r["n_order_seeds"]}
def nk(arm, B):
    """first post-calibration batch index (0-indexed) = k = ceil(n / B) for the arms that carry an n; batches 7..12 for the others."""
    t = arm.rpartition("_")[2]; return -(-int(t) // B) if t.isdigit() and arm.startswith(("amort_", "freeze_plain_")) else SS0

# ---------------- load + regression gate + self-checks
CELLS, missing, mism, ncomp, nvals, chk_fail, xmism, xcomp = {}, [], [], 0, 0, [], [], 0
for lab, j in JUDGES.items():
    for B in BUDS:
        A, base, info = {}, [], {}
        for o in range(NS):
            f = os.path.join(rd, f"design_{j}_B{B}_o{o}.json")
            if not os.path.exists(f): missing.append(f); continue
            d = json.load(open(f)); assert d["budget"] == B and d["order_seed"] == o and d["n_batches"] == NB and d["m"] == 250
            for k, v in d["arms"].items(): A.setdefault(k, []).append(v)
            base.append(d["batch_base_errors"]); info[o] = d["arm_info"]
            bad = [k for k, v in d["checks"].items() if not v]
            if bad: chk_fail.append({"cell": f"{lab} B={B}", "order_seed": o, "failed_selfchecks": bad})
            g = os.path.join(r6, f"amort_{j}_B{B}_o{o}.json")
            if not os.path.exists(g): mism.append({"cell": f"{lab} B={B}", "order_seed": o, "arm": "L6a file missing", "file": g}); continue
            ref = json.load(open(g))
            for k in GATE_KEYS + EXTRA_KEYS:
                a_ = d["batch_base_errors"] if k == "batch_base_errors" else d["arms"][k]; r_ = ref["batch_base_errors"] if k == "batch_base_errors" else ref["arms"][k]
                diffs = [i for i in range(len(r_)) if a_[i] != r_[i]] if len(a_) == len(r_) else list(range(max(len(a_), len(r_))))
                if k in GATE_KEYS:
                    ncomp += 1; nvals += len(r_)
                    if diffs: mism.append({"cell": f"{lab} B={B}", "order_seed": o, "arm": k, "batches_0idx": diffs, "l7b": a_, "l6a": r_})
                else:
                    xcomp += 1
                    if diffs: xmism.append({"cell": f"{lab} B={B}", "order_seed": o, "arm": k, "batches_0idx": diffs})
        CELLS[(lab, B)] = {"A": {k: np.array(v, float) for k, v in A.items()}, "base": np.array(base), "info": info}
nd = sum(len(m.get("batches_0idx", [1])) for m in mism)
gate = {"passed": bool(not missing and not mism and not chk_fail and ncomp == 4 * NS * len(GATE_KEYS)), "vector_comparisons": ncomp, "batch_values_compared": nvals, "batch_value_differences": nd,
        "keys": GATE_KEYS, "order_seeds": "0..15", "missing_l7b_files": missing, "mismatches_first20": mism[:20], "in_job_selfcheck_failures": chk_fail[:20], "n_in_job_selfcheck_failures": len(chk_fail),
        "extra_l6a_arms_not_in_gate": {"keys": EXTRA_KEYS, "vector_comparisons": xcomp, "n_vectors_differing": len(xmism), "first5": xmism[:5]}}
if not gate["passed"]:
    json.dump({"verdict": {"outcome": "VOID", "regression_gate_passed": False}, "regression_gate_vs_L6a": gate}, open(os.path.join(rd, "result.json"), "w"), indent=1)
    print("VOID: regression gate / self-checks did not pass", {k: v for k, v in gate.items() if k not in ("mismatches_first20", "in_job_selfcheck_failures")}); print(json.dumps(mism[:3])); print(chk_fail[:3]); sys.exit(3)

# ---------------- population reference for the coverage table (label-free z-agreement of the whole pool, same transform as the job's global z)
POP = {}
for lab, j in JUDGES.items():
    df = pd.read_parquet(f"runs/E04/feats_{j}.parquet"); a = df["agreement"].values.astype(float); z = (a - a.mean()) / a.std()
    POP[lab] = {"z_min": float(z.min()), "z_max": float(z.max()), "z_sd": float(z.std()), "err_rate": float(df["judge_error"].mean()), "nd_gt1_share": float((df["n_distinct"] > 1).mean())}

# ---------------- per-cell analysis
res = {"protocol": __doc__.split("\n")[1:], "cells": {}}
for (lab, B), C in CELLS.items():
    A = C["A"]; assert all(a in A for a in TABLE_ARMS + ["restart_plain"]) and all(v.shape == (NS, NB) for v in A.values())
    T = lambda k, sl=slice(None): A[k][sl].sum(1)                       # 12-batch totals per order seed
    P = lambda k, kk, sl=slice(None): A[k][sl, kk:].sum(1)               # batches kk+1..12 (post-calibration)
    Cb = lambda k, kk, sl=slice(None): A[k][sl, :kk].sum(1)              # batches 1..kk (calibration period)
    out = {"budget": B, "n_order_seeds": NS, "n_batches": NB, "mean_base_errors_per_batch": float(C["base"].mean()), "population_error_rate": float(C["base"][0].sum() / (NB * 250))}
    out["per_batch_mean_hits"] = {k: [float(x) for x in A[k].mean(0)] for k in TABLE_ARMS}
    out["per_batch_ratio_vs_carry_plain"] = {k: [float(A[k][:, b].sum() / A["carry_plain"][:, b].sum()) for b in range(NB)] for k in TABLE_ARMS}
    tab = {}
    for k in TABLE_ARMS:
        kk = nk(k, B)
        t = {"post_cal_first_batch_1idx": kk + 1, "mean_total_hits_12": float(T(k).mean()), "total_over_carry_plain": rat(T(k), T("carry_plain")), "total_over_thr": rat(T(k), T("thr_agreement")),
             "post_cal_over_carry_plain": rat(P(k, kk), P("carry_plain", kk)), "post_cal_over_thr": rat(P(k, kk), P("thr_agreement", kk)),
             "deficit_vs_carry_plain_calibration_frac_of_carry_total": rat(Cb("carry_plain", kk) - Cb(k, kk), T("carry_plain")), "deficit_vs_carry_plain_post_cal_frac_of_carry_total": rat(P("carry_plain", kk) - P(k, kk), T("carry_plain")),
             "mean_deficit_hits_calibration": float((Cb("carry_plain", kk) - Cb(k, kk)).mean()), "mean_deficit_hits_post_cal": float((P("carry_plain", kk) - P(k, kk)).mean())}
        tab[k] = t
    out["table"] = tab
    out["fit_sets"] = {k: {"n_fit_mean": float(np.mean([C["info"][o][k]["n_fit"] for o in range(NS)])), "z_min_mean": float(np.mean([C["info"][o][k]["fit_z_min"] for o in range(NS)])),
                           "z_max_mean": float(np.mean([C["info"][o][k]["fit_z_max"] for o in range(NS)])), "z_range_mean": float(np.mean([C["info"][o][k]["fit_z_max"] - C["info"][o][k]["fit_z_min"] for o in range(NS)])),
                           "z_sd_mean": float(np.mean([C["info"][o][k]["fit_z_sd"] for o in range(NS)])), "label_err_rate_mean": float(np.mean([C["info"][o][k]["fit_err_rate"] for o in range(NS)])),
                           "nd_gt1_share_mean": float(np.mean([C["info"][o][k]["fit_nd_gt1_share"] for o in range(NS)])), "n_fallback_thr_of_16": int(sum(C["info"][o][k]["fallback_thr"] for o in range(NS)))}
                       for k in COV_ARMS}
    out["population"] = POP[lab]
    sel = lambda sl: max(DES, key=lambda k: A[k][sl].sum(1).mean())  # ties go to the first arm in DES
    bA, bB = sel(FA), sel(FB)
    out["selection"] = {"best_on_seeds0_7": bA, "best_on_seeds8_15": bB, "mean_total_hits_by_designed_arm_seeds0_7": {k: float(T(k, FA).mean()) for k in DES}, "mean_total_hits_by_designed_arm_seeds8_15": {k: float(T(k, FB).mean()) for k in DES}}
    def metrics(blocks):
        """blocks = [(arm, seed slice), ...]: the per-seed vectors of the chosen designed arm and of carry_plain / thr_agreement are concatenated over the blocks (one block for a fold, two for pooled 16);
        the post-calibration window of every block is that block's arm's own batches k+1..12."""
        cat = lambda f: np.concatenate([f(bk, sl) for bk, sl in blocks])
        Tb, Tc, Tt = cat(lambda bk, sl: T(bk, sl)), cat(lambda bk, sl: T("carry_plain", sl)), cat(lambda bk, sl: T("thr_agreement", sl))
        Pb, Pc = cat(lambda bk, sl: P(bk, nk(bk, B), sl)), cat(lambda bk, sl: P("carry_plain", nk(bk, B), sl))
        Db_c, Db_p = cat(lambda bk, sl: Cb("carry_plain", nk(bk, B), sl) - Cb(bk, nk(bk, B), sl)), cat(lambda bk, sl: P("carry_plain", nk(bk, B), sl) - P(bk, nk(bk, B), sl))
        r = rat(Tc, Tb)
        return {"best_arm": ", ".join(f"{bk} (seeds {'0-7' if sl == FA else '8-15'})" for bk, sl in blocks), "carry_plain_over_best": r, "best_over_carry_plain": inv(r), "best_over_thr": rat(Tb, Tt), "carry_plain_over_thr": rat(Tc, Tt),
                "thr_over_best": rat(Tt, Tb), "post_cal_best_over_carry_plain": rat(Pb, Pc), "deficit_calibration_frac_of_carry_total": rat(Db_c, Tc), "deficit_post_cal_frac_of_carry_total": rat(Db_p, Tc)}
    vf = {"A": {"desc": "select on order seeds 0-7, report on 8-15 (VERDICT)", **metrics([(bA, FB)])}, "B": {"desc": "select on 8-15, report on 0-7", **metrics([(bB, FA)])},
          "pooled16": {"desc": "each seed reported with the arm chosen on the other half", **metrics([(bB, FA), (bA, FB)])}}
    vf["A"]["all_six_designed_total_over_carry_plain_seeds8_15"] = {k: rat(T(k, FB), T("carry_plain", FB)) for k in DES}
    vf["B"]["all_six_designed_total_over_carry_plain_seeds0_7"] = {k: rat(T(k, FA), T("carry_plain", FA)) for k in DES}
    out["verdict_folds"] = vf; res["cells"][f"{lab} B={B}"] = out

# ---------------- verdict
def cell_eval(fold):
    rows = {}
    for cn, c in res["cells"].items():
        f = c["verdict_folds"][fold]; r, br, ps = f["carry_plain_over_best"], f["best_over_carry_plain"], f["post_cal_best_over_carry_plain"]
        rows[cn] = {"best_arm": f["best_arm"], "carry_plain_over_best": r["ratio"], "ci_lo": r["ci_lo"], "ci_hi": r["ci_hi"], "best_over_carry_plain": br["ratio"], "best_over_carry_plain_ci": [br["ci_lo"], br["ci_hi"]],
                    "best_over_thr": f["best_over_thr"]["ratio"], "post_cal_best_over_carry_plain": ps["ratio"], "post_cal_ci": [ps["ci_lo"], ps["ci_hi"]],
                    "pass_cell_ge_1.03_and_ci_lo_gt_1": bool(r["ratio"] >= PASS_R and r["ci_lo"] > 1.0), "kill_cell_best_ge_0.99_carry": bool(br["ratio"] >= KILL_R),
                    "best_at_least_3pct_below_carry": bool(br["ratio"] <= 1 / PASS_R), "post_cal_ge_0.99": bool(ps["ratio"] >= KILL_R)}
    n_pass = sum(r["pass_cell_ge_1.03_and_ci_lo_gt_1"] for r in rows.values()); n_kill = sum(r["kill_cell_best_ge_0.99_carry"] for r in rows.values()); n_post = sum(r["post_cal_ge_0.99"] for r in rows.values())
    return {"cells": rows, "n_pass_cells_of_4": int(n_pass), "n_kill_cells_of_4": int(n_kill), "n_post_cal_ge_0.99_cells_of_4": int(n_post), "PASS": bool(n_pass >= 3), "KILL": bool(n_kill >= 3),
            "n_best_at_least_3pct_below_of_4": int(sum(r["best_at_least_3pct_below_carry"] for r in rows.values()))}
vA, vB, vP = cell_eval("A"), cell_eval("B"), cell_eval("pooled16")
outcome = "PASS" if vA["PASS"] else "KILL" if vA["KILL"] else "INCONCLUSIVE"
calcost = bool(outcome == "INCONCLUSIVE" and vA["n_post_cal_ge_0.99_cells_of_4"] == 4)
res["verdict"] = {"outcome": outcome, "calibration_cost_only_feeds_L7c": calcost, "regression_gate_passed": True, "verdict_fold": "best designed arm chosen per cell on order seeds 0-7, reported on order seeds 8-15 (8 order seeds, all 12 batches)",
                  "foldA": vA, "crosscheck_swapped_fold": vB, "crosscheck_pooled16": vP}
res["regression_gate_vs_L6a"] = {k: v for k, v in gate.items() if k not in ("mismatches_first20",)}
res["notes"] = ["Batches are exchangeable draws of one fixed policy, so there is no drift by construction; unbounded carry is an upper bound on what persistence can deliver and a once-fitted rule loses nothing to staleness.",
                "The 16 order seeds permute the same 3,000 prompts, so order-seed CIs are optimistic (order and tie-break variation only, not prompt-sampling variation); each fold has 8 order seeds.",
                "amort_strat_n / amort_bnd_n fit on exactly n labels; freeze_plain_n refits on the k*B >= n labels in hand at the end of batch k (186/310/558 at B = 62, exactly 125/250/500 at B = 125), so at B = 62 the freeze arm has up to 12 percent more labels than a designed arm of the same n.",
                "Calibration labels are charged inside the per-batch budget and counted as hits in every arm. amort_strat_n cuts the thr_agreement order into equal-size rank tenths and takes equal numbers from each: a proportional (equal-population) stratification, so it matches uniform sampling in expectation and differs from amort_rand_n only by lower coverage variance; equal-width range stratification was not run.",
                "Agreement takes 8 distinct values on 3,000 prompts, so the 'agreement ranks' of amort_bnd_n are ties broken by the thr_agreement random stream, not by label.",
                "One population, one task (GSM8K judge_error labels from E04), offline replay on weak-judge labels; the 3B judge barely passed the E04 validity gate."]
json.dump(res, open(os.path.join(rd, "result.json"), "w"), indent=1)

# ---------------- ledger markdown
L = []; P_ = L.append
f3 = lambda r: f"{r['ratio']:.3f} ({r['ci_lo']:.3f} to {r['ci_hi']:.3f})"; pc = lambda r: f"{r['ratio']:.1%} ({r['ci_lo']:.1%} to {r['ci_hi']:.1%})"
P_(f"# L7b ledger table (CPU replay, regression gate vs runs/L6a: {gate['batch_value_differences']} differences in {gate['batch_values_compared']} per-batch values ({gate['vector_comparisons']} vectors, order seeds 0-15, 4 cells), in-job self-check failures {gate['n_in_job_selfcheck_failures']}; extra L6a arms differing: {gate['extra_l6a_arms_not_in_gate']['n_vectors_differing']} of {gate['extra_l6a_arms_not_in_gate']['vector_comparisons']} vectors)")
P_(f"\nVerdict: **{outcome}**{' (calibration cost only, feeds L7c)' if calcost else ''} (verdict fold: best designed arm chosen on order seeds 0-7, reported on 8-15; all 12 batches; PASS cells {vA['n_pass_cells_of_4']} of 4, KILL cells {vA['n_kill_cells_of_4']} of 4, post-calibration best/carry_plain >= 0.99 in {vA['n_post_cal_ge_0.99_cells_of_4']} of 4). "
   f"Swapped fold: PASS {vB['n_pass_cells_of_4']}/4, KILL {vB['n_kill_cells_of_4']}/4. Pooled 16: PASS {vP['n_pass_cells_of_4']}/4, KILL {vP['n_kill_cells_of_4']}/4.")
P_("\nCaveats: no drift by construction; the 16 order seeds permute the same 3,000 prompts, so CIs are optimistic; one population, offline replay on weak-judge labels; the 3B judge barely passed the E04 validity gate; amort_strat_n is an equal-population (proportional) stratification.")
for cn, c in res["cells"].items():
    tb = c["table"]; P_(f"\n## {cn} (per-step budget {c['budget']} of 250; mean errors per batch {c['mean_base_errors_per_batch']:.1f}; 16 order seeds x 12 batches)")
    P_("\nAll 16 order seeds (calibration labels charged and counted as hits). Post-cal = batches k+1..12 with k = ceil(n / B) (batches 7-12 for arms without an n). Deficits = (carry_plain - arm) summed over the calibration batches / over the post-calibration batches, as a share of carry_plain's 12-batch total (positive = arm behind).\n")
    P_("| arm | mean errors caught (12 batches) | / carry_plain (95% CI) | / thr_agreement (95% CI) | post-cal / carry_plain (95% CI) | post-cal / thr | deficit during calibration | deficit after calibration |"); P_("|---|---|---|---|---|---|---|---|")
    for k in TABLE_ARMS:
        t = tb[k]; P_(f"| {k} | {t['mean_total_hits_12']:.1f} | {f3(t['total_over_carry_plain'])} | {f3(t['total_over_thr'])} | {f3(t['post_cal_over_carry_plain'])} | {t['post_cal_over_thr']['ratio']:.3f} | "
                      f"{t['deficit_vs_carry_plain_calibration_frac_of_carry_total']['ratio']:.1%} | {t['deficit_vs_carry_plain_post_cal_frac_of_carry_total']['ratio']:.1%} |")
    pp = c["population"]; P_("\nFit-set coverage (mean over 16 order seeds; z-agreement is the global label-free z, population SD 1; label error rate and n_distinct > 1 share of the fit labels):\n")
    P_("| fit set | n_fit | z-agreement min to max (range) | z SD | label error rate | share n_distinct > 1 | one-class fallbacks |"); P_("|---|---|---|---|---|---|---|")
    P_(f"| population (3,000 prompts) | 3000 | {pp['z_min']:.2f} to {pp['z_max']:.2f} ({pp['z_max'] - pp['z_min']:.2f}) | {pp['z_sd']:.2f} | {pp['err_rate']:.3f} | {pp['nd_gt1_share']:.3f} | |")
    for k, v in c["fit_sets"].items():
        P_(f"| {k} | {v['n_fit_mean']:.0f} | {v['z_min_mean']:.2f} to {v['z_max_mean']:.2f} ({v['z_range_mean']:.2f}) | {v['z_sd_mean']:.2f} | {v['label_err_rate_mean']:.3f} | {v['nd_gt1_share_mean']:.3f} | {v['n_fallback_thr_of_16']} |")
    s = c["selection"]; P_(f"\nSelection: best designed arm on order seeds 0-7 = {s['best_on_seeds0_7']} (mean total hits {s['mean_total_hits_by_designed_arm_seeds0_7'][s['best_on_seeds0_7']]:.1f}), on 8-15 = {s['best_on_seeds8_15']} ({s['mean_total_hits_by_designed_arm_seeds8_15'][s['best_on_seeds8_15']]:.1f}).")
    P_("\nArm / carry_plain by batch index (ratio of means over 16 order seeds):\n"); P_("| arm | " + " | ".join(str(i + 1) for i in range(NB)) + " |"); P_("|---|" + "---|" * NB)
    for k in DES + FZ + ["thr_agreement"]: P_(f"| {k} | " + " | ".join(f"{x:.3f}" for x in c["per_batch_ratio_vs_carry_plain"][k]) + " |")
    P_("\nVerdict folds (all 12 batches; CI over 8 order seeds, 16 for pooled):\n"); P_("| fold | best designed arm | carry_plain / best (95% CI) | best / carry_plain | best / thr (95% CI) | post-cal best / carry_plain (95% CI) | deficit during cal | deficit after cal |"); P_("|---|---|---|---|---|---|---|---|")
    for fn, f in c["verdict_folds"].items():
        P_(f"| {fn} | {f['best_arm']} | {f3(f['carry_plain_over_best'])} | {f['best_over_carry_plain']['ratio']:.3f} | {f3(f['best_over_thr'])} | {f3(f['post_cal_best_over_carry_plain'])} | {f['deficit_calibration_frac_of_carry_total']['ratio']:.1%} | {f['deficit_post_cal_frac_of_carry_total']['ratio']:.1%} |")
P_("\n## Verdict detail (fold A = verdict)\n"); P_("| cell | best designed arm (chosen on 0-7) | carry_plain / best (CI lo) | best / carry_plain | best / thr | post-cal best / carry_plain | pass cell | kill cell | best >= 3% below |"); P_("|---|---|---|---|---|---|---|---|---|")
for cn, r in vA["cells"].items(): P_(f"| {cn} | {r['best_arm']} | {r['carry_plain_over_best']:.3f} ({r['ci_lo']:.3f}) | {r['best_over_carry_plain']:.3f} | {r['best_over_thr']:.3f} | {r['post_cal_best_over_carry_plain']:.3f} | {r['pass_cell_ge_1.03_and_ci_lo_gt_1']} | {r['kill_cell_best_ge_0.99_carry']} | {r['best_at_least_3pct_below_carry']} |")
open(os.path.join(rd, "ledger_table.md"), "w").write("\n".join(L) + "\n"); print("\n".join(L))
