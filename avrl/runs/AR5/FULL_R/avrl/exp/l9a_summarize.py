"""L9a summary (CPU). Usage: l9a_summarize.py <runs/L9a> [<runs/L7b>]
Copied in structure from exp/l7b_summarize.py (functions copied, never imported). Reads
  hz_<judge>_B<B>_s<stream seed>.json      iid-with-replacement streams, H = 60 batches of m = 250 (2 judges x B in 62, 125 x stream seeds 0..15), hits = errors caught per batch, calibration charged
  gate_<judge>_B<B>_o<order seed>.json     mode disjoint12 (the L7b layout), compared exactly with runs/L7b/design_<judge>_B<B>_o<order seed>.json
  nr_<judge>_B<B>_s<seed>.json             optional: the memorisation remedy (iid stream with carry_plain's paid prompts removed from later candidate pools, ends when < 250 unpaid prompts remain)
  hz120_<judge>_B<B>_s<seed>.json          optional: the same iid streams to H = 120 (extension only, never enters the verdict)
and writes result.json and ledger_table.md. Ratios are ratios of means over stream seeds (sum of hits over the window named), CI = percentile bootstrap (10,000 resamples) over stream seeds. The two judges and the
two B values are separate cells and are never pooled.

Fixed arms (the verdict set, 12): amort_{strat,split,seed}_{125,250,500,1000}. References and diagnostics (never selected): carry_plain, thr_agreement, uniform, freeze_plain_{125,250,500,1000}.
Selection (as L7b): per cell and per window the best fixed arm is the one with the highest mean cumulative hits over the selection window on stream seeds 0..7 (ties to the first in the order above) and is reported
on stream seeds 8..15 (VERDICT fold A). Swapped fold B (select on 8..15, report on 0..7) and pooled 16 (each seed reported with the arm chosen on the other half) are cross-checks. For the prefix H the selection
window is batches 1..H; for the steady state it is batches 49..60; the block table reuses the arm chosen for H = 60.

Pre-registered verdict (L9a spec), per cell, fold A, H = 60:
  VOID          if the regression gate (0 differences against runs/L7b on mode disjoint12, order seeds 0..15, 4 cells, every arm in the L7b files) or any in-job self-check fails, or if the memorisation control cannot be
                satisfied (control = after the last batch, the final carry_plain fit's AUC on the prompts it labelled minus its AUC on the prompts it never labelled, mean over the 16 stream seeds of a cell; > 0.02 in any cell
                triggers the remedy "rerun with already-paid prompts removed from later candidate pools"; the remedy cannot reach H = 60 because the table has 3,000 prompts, so a triggered control stays unsatisfied at H = 60)
  PASS          if carry_plain / best fixed >= 1.03 with the bootstrap CI lower bound > 1 in at least 3 of 4 cells
  KILL          if best fixed >= 0.99 x carry_plain in at least 3 of 4 cells
  INCONCLUSIVE  otherwise (the ratio-versus-H slope is the result)
The result carries both the pre-registered outcome and outcome_if_control_waived (the PASS / KILL / INCONCLUSIVE rule applied with the control ignored), so a VOID from the control is never mistaken for a missing analysis.
Exit codes: 0 verdict computed and control satisfied, 3 regression gate or self-check void, 4 memorisation control void (verdict still computed and written)."""
import sys, os, json
import numpy as np

rd = sys.argv[1]; r7 = sys.argv[2] if len(sys.argv) > 2 else os.path.join(os.path.dirname(os.path.abspath(rd)), "L7b")
JUDGES = {"3B": "Qwen2.5-3B-Instruct", "7B-AWQ": "Qwen2.5-7B-Instruct-AWQ"}; BUDS = (62, 125); NS, H0, NBOOT = 16, 60, 10000
HS = (12, 24, 36, 60); CAL_N = (125, 250, 500, 1000); KINDS = ("strat", "split", "seed"); PASS_R, KILL_R, CTRL_GAP = 1.03, 0.99, 0.02
FIXED = [f"amort_{f}_{n}" for f in KINDS for n in CAL_N]; FZ = [f"freeze_plain_{n}" for n in CAL_N]; REFS = ["carry_plain", "thr_agreement", "uniform"]
TABLE_ARMS = FIXED + FZ + REFS
FA, FB = slice(0, 8), slice(8, 16)
rng = np.random.default_rng(0); IDX = {n: rng.integers(0, n, (NBOOT, n)) for n in (8, 16)}

def rat(num, den):
    idx = IDX[len(num)]; rb = num[idx].mean(1) / den[idx].mean(1)
    return {"ratio": float(num.mean() / den.mean()), "ci_lo": float(np.percentile(rb, 2.5)), "ci_hi": float(np.percentile(rb, 97.5)), "n_stream_seeds": int(len(num))}
inv = lambda r: {"ratio": 1 / r["ratio"], "ci_lo": 1 / r["ci_hi"], "ci_hi": 1 / r["ci_lo"], "n_stream_seeds": r["n_stream_seeds"]}
def kcal(arm, B):  # calibration batches ceil(n / B) of an arm named ..._n
    return -(-int(arm.rpartition("_")[2]) // B)
fnum = lambda x: None if x is None else float(x)

def load(prefix, cells=True):
    """-> {(label, B): {"A": {arm: array (NS, H)}, "base": ..., "docs": [json...]}} or None if no file with this prefix exists; raises on partial sets."""
    out, found, miss = {}, 0, []
    for lab, j in JUDGES.items():
        for B in BUDS:
            docs = []
            for s in range(NS):
                f = os.path.join(rd, f"{prefix}_{j}_B{B}_{'o' if prefix == 'gate' else 's'}{s}.json")
                if os.path.exists(f): docs.append(json.load(open(f))); found += 1
                else: miss.append(f)
            out[(lab, B)] = docs
    if found == 0: return None
    return out, miss

def arrays(docs, Hcap=None):
    Hc = min(len(d["batch_base_errors"]) for d in docs) if Hcap is None else Hcap
    A = {}
    for d in docs:
        for k, v in d["arms"].items(): A.setdefault(k, []).append(v[:Hc])
    return {k: np.array(v, float) for k, v in A.items()}, Hc

# ---------------- iid main data + regression gate + self-checks
main = load("hz"); assert main is not None, "no hz_*.json files"
HZ, miss_hz = main; GATE = load("gate"); NR = load("nr"); X120 = load("hz120")
chk_fail = [{"file": f"{lab} B={B} s={i}", "failed": [k for k, v in d["checks"].items() if not v]} for (lab, B), docs in HZ.items() for i, d in enumerate(docs) if not all(d["checks"].values())]
if GATE is not None:
    for (lab, B), docs in GATE[0].items():
        chk_fail += [{"file": f"gate {lab} B={B} o={i}", "failed": [k for k, v in d["checks"].items() if not v]} for i, d in enumerate(docs) if not all(d["checks"].values())]
missing, mism, ncomp, nvals = list(miss_hz), [], 0, 0
if GATE is None: missing.append("no gate_*.json files")
else:
    missing += GATE[1]
    for (lab, B), docs in GATE[0].items():
        j = JUDGES[lab]
        for o, d in enumerate(docs):
            f = os.path.join(r7, f"design_{j}_B{B}_o{o}.json")
            if not os.path.exists(f): missing.append(f); continue
            ref = json.load(open(f)); assert d["order_seed"] == o and d["budget"] == B and d["mode"] == "disjoint12"
            keys = list(ref["arms"]) + ["batch_base_errors"]
            for k in keys:
                a_ = d["batch_base_errors"] if k == "batch_base_errors" else d["arms"].get(k); r_ = ref["batch_base_errors"] if k == "batch_base_errors" else ref["arms"][k]
                ncomp += 1; nvals += len(r_)
                diffs = list(range(len(r_))) if a_ is None or len(a_) != len(r_) else [i for i in range(len(r_)) if a_[i] != r_[i]]
                if diffs: mism.append({"cell": f"{lab} B={B}", "order_seed": o, "arm": k, "batches_0idx": diffs})
nd = sum(len(m["batches_0idx"]) for m in mism)
gate = {"passed": bool(not missing and not mism and not chk_fail and ncomp > 0), "vector_comparisons": ncomp, "batch_values_compared": nvals, "batch_value_differences": nd, "arms_compared": "every arm in runs/L7b/design_*.json (25 arms) plus batch_base_errors",
        "order_seeds": "0..15", "missing_files": missing[:20], "n_missing": len(missing), "mismatches_first20": mism[:20], "in_job_selfcheck_failures": chk_fail[:20], "n_in_job_selfcheck_failures": len(chk_fail)}
if not gate["passed"]:
    json.dump({"verdict": {"outcome": "VOID", "reason": "regression gate or self-check", "regression_gate_passed": False}, "regression_gate_vs_L7b": gate}, open(os.path.join(rd, "result.json"), "w"), indent=1)
    print("VOID: regression gate / self-checks did not pass", {k: v for k, v in gate.items() if k not in ("mismatches_first20", "in_job_selfcheck_failures")}); print(json.dumps(mism[:3])); print(chk_fail[:3]); sys.exit(3)

# ---------------- analysis helpers
def W(A, k, sl, lo, hi): return A[k][sl, lo:hi].sum(1)
def fold_eval(A, B, lo, hi, slo, shi, fixed=FIXED):
    """best fixed arm chosen on batches slo..shi of one half of the stream seeds, ratios reported on batches lo..hi of the other half."""
    sel = lambda sl: max(fixed, key=lambda k: W(A, k, sl, slo, shi).mean())
    bA, bB = sel(FA), sel(FB)
    def metrics(blocks):
        cat = lambda f: np.concatenate([f(bk, sl) for bk, sl in blocks]); plo = lambda bk: max(lo, kcal(bk, B))
        Tb, Tc, Tt = cat(lambda bk, sl: W(A, bk, sl, lo, hi)), cat(lambda bk, sl: W(A, "carry_plain", sl, lo, hi)), cat(lambda bk, sl: W(A, "thr_agreement", sl, lo, hi))
        r = rat(Tc, Tb); o = {"best_arm": ", ".join(f"{bk} (seeds {'0-7' if sl == FA else '8-15'})" for bk, sl in blocks), "carry_plain_over_best": r, "best_over_carry_plain": inv(r), "best_over_thr": rat(Tb, Tt), "carry_plain_over_thr": rat(Tc, Tt)}
        if all(plo(bk) < hi for bk, _ in blocks):
            Pb, Pc = cat(lambda bk, sl: W(A, bk, sl, plo(bk), hi)), cat(lambda bk, sl: W(A, "carry_plain", sl, plo(bk), hi)); o["post_cal_best_over_carry_plain"] = rat(Pb, Pc)
        else: o["post_cal_best_over_carry_plain"] = None
        return o
    return {"A": metrics([(bA, FB)]), "B": metrics([(bB, FA)]), "pooled16": metrics([(bB, FA), (bA, FB)]), "chosen": {"seeds0_7": bA, "seeds8_15": bB}}

def extrap(Hs, qs):
    """descriptive only: q(H) = a + c / H least squares on q = best / carry_plain (ratio of means), and the H at which q reaches 1/1.03 and 0.99 under that fit (None = never, asymptote a is not above the level)."""
    X = np.c_[np.ones(len(Hs)), 1.0 / np.array(Hs, float)]; a, c = np.linalg.lstsq(X, np.array(qs, float), rcond=None)[0]
    cross = lambda lvl: (float(c / (lvl - a)) if (lvl < a and c < 0) else None)
    return {"a_asymptote": float(a), "c": float(c), "H_values": list(Hs), "q_values": [float(q) for q in qs], "fit_at_H": {str(h): float(a + c / h) for h in Hs + (120,)}, "H_cross_carry_over_best_1.03": cross(1 / PASS_R), "H_cross_best_over_carry_0.99": cross(KILL_R),
            "asymptote_best_over_carry": float(a), "asymptote_carry_over_best": float(1 / a) if a > 0 else None}

def memo_summary(docs):
    g = lambda key: np.array([d["memorisation"][key] if d["memorisation"][key] is not None else np.nan for d in docs], float)
    gp, gs = g("gap_labelled_minus_never"), g("gap_split_in_minus_held_out"); st = lambda key: float(np.mean([d["stream"][key] for d in docs]))
    return {"primary_gap_labelled_minus_never_mean": float(np.nanmean(gp)), "primary_gap_min": float(np.nanmin(gp)), "primary_gap_max": float(np.nanmax(gp)), "n_seeds_primary_gap_gt_0.02": int(np.sum(gp > CTRL_GAP)),
            "auc_labelled_mean": float(np.nanmean(g("auc_labelled"))), "auc_never_labelled_mean": float(np.nanmean(g("auc_never_labelled"))), "err_rate_labelled_mean": float(np.mean(g("err_rate_labelled"))), "err_rate_never_labelled_mean": float(np.nanmean(g("err_rate_never_labelled"))),
            "split_half_gap_in_minus_held_out_mean": float(np.nanmean(gs)), "split_half_gap_min": float(np.nanmin(gs)), "split_half_gap_max": float(np.nanmax(gs)), "n_seeds_split_gap_gt_0.02": int(np.sum(gs > CTRL_GAP)),
            "n_labels_mean": float(np.mean(g("n_labels"))), "n_distinct_labelled_mean": float(np.mean(g("n_distinct_labelled"))), "n_never_labelled_mean": float(np.mean(g("n_never_labelled"))),
            "carry_plain_repeat_pick_share_mean": st("carry_plain_repeat_pick_share"), "carry_plain_hit_share_on_repeat_picks_mean": float(np.mean([d["stream"]["carry_plain_hits_on_repeat_picks"] / d["stream"]["carry_plain_hits_total"] for d in docs])),
            "n_distinct_prompts_drawn_mean": st("n_distinct_prompts_drawn"), "mean_distinct_share_per_batch": st("mean_distinct_share_per_batch")}

# ---------------- per-cell analysis
res = {"protocol": __doc__.split("\n")[1:], "cells": {}}
for (lab, B), docs in HZ.items():
    A, Hc = arrays(docs); assert Hc == H0 and all(v.shape == (NS, H0) for v in A.values()) and all(a in A for a in TABLE_ARMS), (lab, B, Hc)
    cn = f"{lab} B={B}"; out = {"budget": B, "n_stream_seeds": NS, "H": H0, "mean_base_errors_per_batch": float(np.mean([d["batch_base_errors"] for d in docs]))}
    out["per_batch_mean_hits"] = {k: [float(x) for x in A[k].mean(0)] for k in REFS + ["amort_split_500", "amort_split_250"]}
    # ratio versus H (verdict structure at each prefix)
    out["by_H"] = {str(H): fold_eval(A, B, 0, H, 0, H) for H in HS}
    out["last12"] = {"select_on_last12": fold_eval(A, B, H0 - 12, H0, H0 - 12, H0)}
    bA60 = out["by_H"]["60"]["chosen"]["seeds0_7"]; bB60 = out["by_H"]["60"]["chosen"]["seeds8_15"]
    rl = lambda bk, sl, lo, hi: rat(W(A, "carry_plain", sl, lo, hi), W(A, bk, sl, lo, hi))
    out["last12"]["arm_chosen_for_H60_fold_A"] = {"best_arm": bA60, "carry_plain_over_best": rl(bA60, FB, 48, 60)}
    out["blocks_arm_chosen_for_H60"] = {f"{lo + 1}-{lo + 12}": {"A": {"best_arm": bA60, "carry_plain_over_best": rl(bA60, FB, lo, lo + 12)}, "B": {"best_arm": bB60, "carry_plain_over_best": rl(bB60, FA, lo, lo + 12)},
                                                               "pooled16": {"carry_plain_over_best": rat(np.concatenate([W(A, "carry_plain", FA, lo, lo + 12), W(A, "carry_plain", FB, lo, lo + 12)]),
                                                                                                          np.concatenate([W(A, bB60, FA, lo, lo + 12), W(A, bA60, FB, lo, lo + 12)]))}} for lo in range(0, 60, 12)}
    qA = [out["by_H"][str(H)]["A"]["best_over_carry_plain"]["ratio"] for H in HS]; qP = [out["by_H"][str(H)]["pooled16"]["best_over_carry_plain"]["ratio"] for H in HS]
    out["extrapolation_in_1_over_H"] = {"descriptive_only": True, "fold_A": extrap(HS, qA), "pooled16": extrap(HS, qP)}
    # all-arm table at H = 60 (all 16 seeds, no selection)
    T = lambda k, lo=0, hi=H0: A[k][:, lo:hi].sum(1)
    tab = {}
    for k in TABLE_ARMS:
        isn = k.rpartition("_")[2].isdigit() and k.startswith(("amort_", "freeze_plain_")); kk = kcal(k, B) if isn else 0
        t = {"mean_total_hits_60": float(T(k).mean()), "total_over_carry_plain": rat(T(k), T("carry_plain")), "total_over_thr": rat(T(k), T("thr_agreement")), "last12_over_carry_plain": rat(T(k, 48, 60), T("carry_plain", 48, 60))}
        if isn: t["first_post_cal_batch_1idx"] = kk + 1; t["post_cal_over_carry_plain"] = rat(T(k, kk, H0), T("carry_plain", kk, H0)); t["post_cal_over_thr"] = rat(T(k, kk, H0), T("thr_agreement", kk, H0))
        tab[k] = t
    out["table_H60"] = tab
    out["fixed_arm_mean_total_hits_by_fold"] = {"seeds0_7": {k: float(T(k)[FA].mean()) for k in FIXED}, "seeds8_15": {k: float(T(k)[FB].mean()) for k in FIXED}}
    out["fit_sets"] = {k: {"n_fit_mean": float(np.mean([d["arm_info"][k]["n_fit"] for d in docs])), "label_err_rate_mean": float(np.mean([d["arm_info"][k]["fit_err_rate"] for d in docs])), "n_fallback_thr_of_16": int(sum(d["arm_info"][k]["fallback_thr"] for d in docs))}
                       for k in [f"amort_{f}_{n}" for f in ("strat", "split", "seed") for n in CAL_N] + FZ}
    out["memorisation_control"] = memo_summary(docs)
    # no-repeat remedy (optional)
    if NR is not None:
        ndocs = NR[0][(lab, B)]
        if len(ndocs) == NS:
            An, Hn = arrays(ndocs); mem = memo_summary(ndocs); rows = {}
            for H in sorted(set([h for h in (12, 24, 36) if h <= Hn] + [Hn])):
                fn = fold_eval(An, B, 0, H, 0, H); fi = fold_eval(A, B, 0, H, 0, H)
                rows[str(H)] = {"nr_foldA": fn["A"]["carry_plain_over_best"], "nr_pooled16": fn["pooled16"]["carry_plain_over_best"], "nr_best_arm_foldA": fn["A"]["best_arm"], "iid_foldA": fi["A"]["carry_plain_over_best"], "iid_pooled16": fi["pooled16"]["carry_plain_over_best"],
                                "iid_best_arm_foldA": fi["A"]["best_arm"], "nr_minus_iid_pooled16_ratio": float(fn["pooled16"]["carry_plain_over_best"]["ratio"] - fi["pooled16"]["carry_plain_over_best"]["ratio"])}
            out["no_repeat_remedy"] = {"H_reached_min_over_seeds": int(Hn), "H_reached_per_seed": [len(d["batch_base_errors"]) for d in ndocs], "memorisation_control_after_remedy": mem, "by_H": rows,
                                       "mean_unpaid_prompts_at_end": float(np.mean([d["stream"]["final_unpaid_prompts"] for d in ndocs]))}
    # H = 120 extension (optional)
    if X120 is not None:
        xdocs = X120[0][(lab, B)]
        if len(xdocs) == NS:
            Ax, Hx = arrays(xdocs); assert Hx == 120
            prefix_ok = bool(all(np.array_equal(Ax[k][:, :H0], A[k]) for k in A))
            ext = {str(H): fold_eval(Ax, B, 0, H, 0, H) for H in (60, 120)}
            q120 = ext["120"]["pooled16"]["best_over_carry_plain"]["ratio"]; fit = out["extrapolation_in_1_over_H"]["pooled16"]["fit_at_H"]["120"]
            out["extension_H120"] = {"prefix_60_equals_main_run_exactly": prefix_ok, "by_H": {H: {"A": e["A"]["carry_plain_over_best"], "pooled16": e["pooled16"]["carry_plain_over_best"], "best_arm_foldA": e["A"]["best_arm"], "best_arm_pooled": e["pooled16"]["best_arm"]} for H, e in ext.items()},
                                     "last12_of_120_select_on_last12_foldA": fold_eval(Ax, B, 108, 120, 108, 120)["A"]["carry_plain_over_best"], "measured_best_over_carry_pooled16_at_120": q120, "extrapolated_from_H_le_60_best_over_carry_at_120": fit,
                                     "memorisation_control_at_120": memo_summary(xdocs)}
    res["cells"][cn] = out

# ---------------- verdict
def cell_eval(fold):
    rows = {}
    for cn, c in res["cells"].items():
        f = c["by_H"]["60"][fold]; r, br = f["carry_plain_over_best"], f["best_over_carry_plain"]
        rows[cn] = {"best_arm": f["best_arm"], "carry_plain_over_best": r["ratio"], "ci_lo": r["ci_lo"], "ci_hi": r["ci_hi"], "best_over_carry_plain": br["ratio"], "best_over_carry_plain_ci": [br["ci_lo"], br["ci_hi"]], "best_over_thr": f["best_over_thr"]["ratio"],
                    "pass_cell_ge_1.03_and_ci_lo_gt_1": bool(r["ratio"] >= PASS_R and r["ci_lo"] > 1.0), "kill_cell_best_ge_0.99_carry": bool(br["ratio"] >= KILL_R)}
    n_pass = sum(r["pass_cell_ge_1.03_and_ci_lo_gt_1"] for r in rows.values()); n_kill = sum(r["kill_cell_best_ge_0.99_carry"] for r in rows.values())
    return {"cells": rows, "n_pass_cells_of_4": int(n_pass), "n_kill_cells_of_4": int(n_kill), "PASS": bool(n_pass >= 3), "KILL": bool(n_kill >= 3)}
vA, vB, vP = cell_eval("A"), cell_eval("B"), cell_eval("pooled16")
rule_outcome = "PASS" if vA["PASS"] else "KILL" if vA["KILL"] else "INCONCLUSIVE"
trig = {cn: c["memorisation_control"]["primary_gap_labelled_minus_never_mean"] for cn, c in res["cells"].items() if c["memorisation_control"]["primary_gap_labelled_minus_never_mean"] > CTRL_GAP}
control = {"statistic": "mean over 16 stream seeds of AUC(final carry_plain fit; prompts it labelled) - AUC(same fit; prompts it never labelled), after batch 60", "threshold": CTRL_GAP, "triggered_cells": trig, "n_triggered_of_4": len(trig),
           "remedy": "rerun with already-paid prompts removed from later candidate pools (mode iid_nr); the table has 3,000 prompts and carry_plain pays B per batch, so the remedy stream ends after about 3000/B batches and cannot reach H = 60 (see no_repeat_remedy per cell)",
           "satisfied_at_H60": bool(len(trig) == 0),
           "caveat": "the labelled prompts are the learner's own high-score picks, so the primary statistic mixes memorisation with range restriction (the never-labelled set is the low-score region, where AUC is near or below 0.5); the split-half statistic (per cell, memorisation_control) separates the two"}
outcome = "VOID" if not control["satisfied_at_H60"] else rule_outcome
res["verdict"] = {"outcome": outcome, "outcome_if_control_waived": rule_outcome, "void_reason": None if outcome != "VOID" else "memorisation control triggered (gap > 0.02) and its remedy cannot reach H = 60", "regression_gate_passed": True, "memorisation_control": control,
                  "verdict_fold": "best fixed arm chosen per cell on stream seeds 0-7 (cumulative hits over the 60 batches), reported on stream seeds 8-15", "foldA": vA, "crosscheck_swapped_fold": vB, "crosscheck_pooled16": vP}
res["regression_gate_vs_L7b"] = {k: v for k, v in gate.items() if k != "mismatches_first20"}
res["notes"] = ["Streams are iid with replacement from one fixed 3,000-prompt table: batches are exchangeable, there is no drift, and a once-fitted rule loses nothing to staleness (most favourable to the fixed arms). Prompts repeat across and within batches with identical weak-judge labels; a pick of a known prompt is paid again and counted again in every arm.",
                "Stream-seed CIs cover stream and tie-break variation only, not prompt-sampling variation (every stream draws from the same 3,000 prompts), so they are optimistic. Each fold has 8 stream seeds.",
                "The selection set is the 12 fixed arms; freeze_plain_n (which uses the learner's own labels) is a diagnostic and never selected. Calibration labels are charged inside the per-batch budget and counted as hits in every arm.",
                "The extrapolation in 1/H is a two-parameter least-squares line through four prefix points and is descriptive only.",
                "One population, one task (GSM8K judge_error labels from E04), offline replay on weak-judge labels; the 3B judge barely passed the E04 validity gate."]
json.dump(res, open(os.path.join(rd, "result.json"), "w"), indent=1)

# ---------------- ledger markdown
L = []; P_ = L.append
f3 = lambda r: f"{r['ratio']:.3f} ({r['ci_lo']:.3f} to {r['ci_hi']:.3f})"; f3n = lambda r: "n/a" if r is None else f3(r)
P_(f"# L9a ledger table (CPU replay; regression gate vs runs/L7b: {gate['batch_value_differences']} differences in {gate['batch_values_compared']} per-batch values, {gate['vector_comparisons']} vectors; in-job self-check failures {gate['n_in_job_selfcheck_failures']})")
P_(f"\nPre-registered outcome: **{outcome}**; the rule applied with the memorisation control waived: **{rule_outcome}** (fold A, H = 60: PASS cells {vA['n_pass_cells_of_4']} of 4, KILL cells {vA['n_kill_cells_of_4']} of 4; swapped fold {vB['n_pass_cells_of_4']}/{vB['n_kill_cells_of_4']}; pooled 16 {vP['n_pass_cells_of_4']}/{vP['n_kill_cells_of_4']}).")
P_(f"\nMemorisation control: primary gap > {CTRL_GAP} in {len(trig)} of 4 cells ({', '.join(f'{k}: {v:.3f}' for k, v in trig.items()) or 'none'}).")
P_("\n## Carry_plain / best fixed arm by horizon (fold A: best chosen on stream seeds 0-7, reported on 8-15; 95% CI over 8 stream seeds; swapped fold; pooled 16)\n")
P_("| cell | H | best fixed (seeds 0-7) | carry/best fold A (CI) | best/carry fold A | swapped fold B | pooled 16 | best/thr (fold A) | post-cal best/carry |"); P_("|---|---|---|---|---|---|---|---|---|")
for cn, c in res["cells"].items():
    for H in HS:
        e = c["by_H"][str(H)]; P_(f"| {cn} | {H} | {e['chosen']['seeds0_7']} | {f3(e['A']['carry_plain_over_best'])} | {e['A']['best_over_carry_plain']['ratio']:.3f} | {e['B']['carry_plain_over_best']['ratio']:.3f} | {e['pooled16']['carry_plain_over_best']['ratio']:.3f} | {f3(e['A']['best_over_thr'])} | {f3n(e['A']['post_cal_best_over_carry_plain'])} |")
P_("\n## Steady state and blocks (carry_plain / best fixed, fold A, arm chosen for H = 60 unless stated)\n")
P_("| cell | last 12 (arm chosen on last 12) | last 12 (H=60 arm) | " + " | ".join(f"batches {b}" for b in next(iter(res['cells'].values()))["blocks_arm_chosen_for_H60"]) + " |"); P_("|---|---|---|" + "---|" * 5)
for cn, c in res["cells"].items():
    P_(f"| {cn} | {f3(c['last12']['select_on_last12']['A']['carry_plain_over_best'])} ({c['last12']['select_on_last12']['A']['best_arm']}) | {f3(c['last12']['arm_chosen_for_H60_fold_A']['carry_plain_over_best'])} | " + " | ".join(f"{b['A']['carry_plain_over_best']['ratio']:.3f}" for b in c["blocks_arm_chosen_for_H60"].values()) + " |")
P_("\n## Extrapolation in 1/H (descriptive only; q = best / carry_plain, a + c/H fit through H = 12, 24, 36, 60)\n")
P_("| cell | basis | asymptote best/carry (carry/best) | H where carry/best = 1.03 | H where best/carry = 0.99 | fit at 120 |"); P_("|---|---|---|---|---|---|")
for cn, c in res["cells"].items():
    for b in ("fold_A", "pooled16"):
        e = c["extrapolation_in_1_over_H"][b]; P_(f"| {cn} | {b} | {e['asymptote_best_over_carry']:.3f} ({e['asymptote_carry_over_best']:.3f}) | {e['H_cross_carry_over_best_1.03'] and round(e['H_cross_carry_over_best_1.03'])} | {e['H_cross_best_over_carry_0.99'] and round(e['H_cross_best_over_carry_0.99'])} | {e['fit_at_H']['120']:.3f} |")
for cn, c in res["cells"].items():
    P_(f"\n## {cn} (per-step budget {c['budget']} of 250; mean errors per batch {c['mean_base_errors_per_batch']:.1f}; 16 stream seeds x 60 batches)\n")
    P_("All 16 stream seeds, no selection. Post-cal = batches k+1..60 with k = ceil(n / B).\n")
    P_("| arm | mean errors caught (60 batches) | / carry_plain (95% CI) | / thr_agreement (95% CI) | post-cal / carry_plain (95% CI) | last 12 / carry_plain |"); P_("|---|---|---|---|---|---|")
    for k in TABLE_ARMS:
        t = c["table_H60"][k]; P_(f"| {k} | {t['mean_total_hits_60']:.1f} | {f3(t['total_over_carry_plain'])} | {f3(t['total_over_thr'])} | {f3n(t.get('post_cal_over_carry_plain'))} | {t['last12_over_carry_plain']['ratio']:.3f} |")
    m = c["memorisation_control"]; P_(f"\nMemorisation control (final carry_plain fit, mean over 16 seeds): labels {m['n_labels_mean']:.0f}, distinct labelled {m['n_distinct_labelled_mean']:.0f}, never labelled {m['n_never_labelled_mean']:.0f}; AUC labelled {m['auc_labelled_mean']:.3f}, never labelled {m['auc_never_labelled_mean']:.3f}; "
       f"primary gap {m['primary_gap_labelled_minus_never_mean']:.3f} (min {m['primary_gap_min']:.3f}, max {m['primary_gap_max']:.3f}; {m['n_seeds_primary_gap_gt_0.02']} of 16 seeds above 0.02); split-half in-sample minus held-out gap {m['split_half_gap_in_minus_held_out_mean']:.3f} "
       f"(min {m['split_half_gap_min']:.3f}, max {m['split_half_gap_max']:.3f}; {m['n_seeds_split_gap_gt_0.02']} of 16 above 0.02); carry_plain re-pays a known prompt on {m['carry_plain_repeat_pick_share_mean']:.1%} of picks ({m['carry_plain_hit_share_on_repeat_picks_mean']:.1%} of its hits); distinct prompts drawn in 60 batches {m['n_distinct_prompts_drawn_mean']:.0f} of 3000.")
    if "no_repeat_remedy" in c:
        r = c["no_repeat_remedy"]; mm = r["memorisation_control_after_remedy"]
        P_(f"\nNo-repeat remedy: stream reached H = {r['H_reached_min_over_seeds']} batches (min over seeds; per seed {min(r['H_reached_per_seed'])} to {max(r['H_reached_per_seed'])}), then fewer than 250 unpaid prompts were left (mean {r['mean_unpaid_prompts_at_end']:.0f} unpaid at the end). Control after remedy: primary gap {mm['primary_gap_labelled_minus_never_mean']:.3f}, split-half gap {mm['split_half_gap_in_minus_held_out_mean']:.3f}, repeat-pick share {mm['carry_plain_repeat_pick_share_mean']:.1%}.\n")
        P_("| H | carry/best no-repeat fold A (CI) | carry/best iid fold A (CI) | no-repeat pooled 16 | iid pooled 16 | no-repeat minus iid (pooled 16) |"); P_("|---|---|---|---|---|---|")
        for H, e in r["by_H"].items(): P_(f"| {H} | {f3(e['nr_foldA'])} | {f3(e['iid_foldA'])} | {e['nr_pooled16']['ratio']:.3f} | {e['iid_pooled16']['ratio']:.3f} | {e['nr_minus_iid_pooled16_ratio']:+.3f} |")
    if "extension_H120" in c:
        x = c["extension_H120"]; P_(f"\nH = 120 extension (prefix 60 identical to the main run: {x['prefix_60_equals_main_run_exactly']}): carry/best fold A at 60 {f3(x['by_H']['60']['A'])}, at 120 {f3(x['by_H']['120']['A'])} (best arm {x['by_H']['120']['best_arm_foldA']}); pooled 16 at 120 {x['by_H']['120']['pooled16']['ratio']:.3f}; "
           f"last 12 of 120 {f3(x['last12_of_120_select_on_last12_foldA'])}; best/carry measured (pooled 16) at 120 {x['measured_best_over_carry_pooled16_at_120']:.3f} vs extrapolated from H <= 60 {x['extrapolated_from_H_le_60_best_over_carry_at_120']:.3f}.")
open(os.path.join(rd, "ledger_table.md"), "w").write("\n".join(L) + "\n"); print("\n".join(L))
sys.exit(0 if outcome != "VOID" else 4)
