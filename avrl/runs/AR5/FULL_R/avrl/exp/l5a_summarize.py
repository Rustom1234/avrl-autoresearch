"""L5a summary (CPU). Usage: l5a_summarize.py <runs/L5a>
Reads carry_<judge>_B<B>_o<order seed>.json (judges 3B and 7B-AWQ, per-step B in 62, 125, order seeds 0..7; 12 disjoint batches of 250 each; hits = errors caught per batch) and writes result.json,
ledger_table.md and carry_ratio_vs_batch.png. Ratios are ratios of means (sum of hits over the pooled batches and order seeds), CI = percentile bootstrap (10,000 resamples) over order seeds.
Pooled window = batches 4 to 12 (at least about 250 labels in hand at B = 62 from batch 5, at B = 125 from batch 3).
Carry_offset choice (lam 10 vs 1): per cell, the lam with the higher pooled batches 4-12 mean hits on order seeds 0..3, reported on order seeds 4..7 (the VERDICT fold). Swapped fold (select on 4..7,
report on 0..3) and pooled 8 (each seed reported with the lam chosen on the other half) are cross-checks. carry_window{W} uses the same lam as the chosen carry_offset arm.

Pre-registered verdict (L5a spec), per cell = judge x per-step B (4 cells):
  VOID          if the regression gate against runs/L4a (m = 250, subset seed 0) did not pass
  PASS          if the chosen carry_offset / thr_agreement >= 1.03 pooled over batches 4-12 with the bootstrap CI lower bound > 1 in ALL FOUR cells, AND carry_window250 is within 2% of the chosen carry arm
                (|carry_window250 / carry_offset - 1| <= 0.02, read two-sided and required in all four cells)
  KILL          if carry_plain / thr_agreement <= 1.00 AND the chosen carry_offset / thr_agreement <= 1.00 (pooled batches 4-12) in at least 3 of the 4 cells
  INCONCLUSIVE  otherwise (report the per-cell table and the batch-index curves)"""
import sys, os, json
import numpy as np

rd = sys.argv[1]
JUDGES = {"3B": "Qwen2.5-3B-Instruct", "7B-AWQ": "Qwen2.5-7B-Instruct-AWQ"}; BUDS = (62, 125); NS, NB, NBOOT, P0 = 8, 12, 10000, 3
WINS, LAMS = (125, 250, 500), (10, 1); PASS_R, WIN_TOL, KILL_R = 1.03, 0.02, 1.00
FA, FB = slice(0, 4), slice(4, 8)
rng = np.random.default_rng(0); IDX = {n: rng.integers(0, n, (NBOOT, n)) for n in (4, 8)}
gate_path = os.path.join(rd, "gate", "gate_check.json"); gate = json.load(open(gate_path)) if os.path.exists(gate_path) else None
if not (gate and gate.get("passed")):
    json.dump({"verdict": {"outcome": "VOID", "regression_gate_passed": False}, "regression_gate_vs_L4a": gate}, open(os.path.join(rd, "result.json"), "w"), indent=1)
    print("VOID: regression gate did not pass", gate and {k: v for k, v in gate.items() if "diffs" not in k}); sys.exit(0)

def rat(num, den):
    """ratio of means of per-seed vectors with a percentile bootstrap CI over the seeds (len 4 or 8)."""
    idx = IDX[len(num)]; rb = num[idx].mean(1) / den[idx].mean(1)
    return {"ratio": float(num.mean() / den.mean()), "ci_lo": float(np.percentile(rb, 2.5)), "ci_hi": float(np.percentile(rb, 97.5)), "n_order_seeds": int(len(num))}

ARMS = ["uniform", "restart_offset", "restart_plain", "carry_plain", "carry_offset_lam10", "carry_offset_lam1"] + [f"carry_window{W}_lam{l}" for l in LAMS for W in WINS] + ["calibrate_once", "static_out"]
CELLS = {}
for lab, j in JUDGES.items():
    for B in BUDS:
        A, base, chk, dg = {}, [], [], {}
        for o in range(NS):
            d = json.load(open(os.path.join(rd, f"carry_{j}_B{B}_o{o}.json"))); assert d["budget"] == B and d["order_seed"] == o and d["n_batches"] == NB and d["m"] == 250
            for k, v in d["arms"].items(): A.setdefault(k, []).append(v)
            base.append(d["batch_base_errors"]); chk.append(d["checks"])
            for k, v in d["diag"].items(): dg.setdefault(k, []).append(v)
        A = {k: np.array(v, float) for k, v in A.items()}; assert "thr_agreement" in A and all(a in A for a in ARMS)
        assert all(c["batch1_carry_plain_eq_restart_plain"] and c["batch1_windows_eq_carry_offset"] for c in chk), "in-job batch-1 equality checks failed"
        diag = {k: {"err_rate": np.array([[b["err_rate"] for b in s] for s in v]), "mean_z_agree": np.array([[b["mean_z_agree"] for b in s] for s in v]), "n": np.array([[b["n"] for b in s] for s in v])} for k, v in dg.items()}
        CELLS[(lab, B)] = {"A": A, "base": np.array(base), "diag": diag}
cell_names = list(CELLS)

def first_cross(vals, thr, start=0):
    for i in range(start, len(vals)):
        if vals[i] >= thr: return i + 1
    return None

res = {"protocol": __doc__.split("\n")[1:], "cells": {}}
for (lab, B), C in CELLS.items():
    A = C["A"]; thr = A["thr_agreement"]; out = {"budget": B, "n_order_seeds": NS, "n_batches": NB, "mean_base_errors_per_batch": float(C["base"].mean()), "batch_base_errors_mean": [float(x) for x in C["base"].mean(0)]}
    out["per_batch_mean_hits"] = {k: [float(x) for x in A[k].mean(0)] for k in ["thr_agreement"] + ARMS}
    pb = {}
    for k in ARMS:
        r = [rat(A[k][:, b], thr[:, b]) for b in range(NB)]
        tr = [float(A[k][:, max(0, b - 2):b + 1].sum() / thr[:, max(0, b - 2):b + 1].sum()) for b in range(NB)]  # trailing 3-batch pooled ratio (smoother)
        pb[k] = {"ratio": [x["ratio"] for x in r], "ci_lo": [x["ci_lo"] for x in r], "ci_hi": [x["ci_hi"] for x in r], "trailing3_ratio": tr,
                 "first_batch_ge_1.00": first_cross([x["ratio"] for x in r], 1.00), "first_batch_ge_1.03": first_cross([x["ratio"] for x in r], 1.03),
                 "first_batch_trailing3_ge_1.00": first_cross(tr, 1.00, 2), "first_batch_trailing3_ge_1.03": first_cross(tr, 1.03, 2)}
    out["per_batch_ratio_vs_thr"] = pb
    ps = lambda k, sl=slice(None), b0=P0: A[k][sl, b0:].sum(1)
    gapd = lambda k, sl=slice(None): ps(k, sl) - ps("thr_agreement", sl)
    pool = {}
    for k in ARMS:
        pool[k] = {"all8": rat(ps(k), ps("thr_agreement")), "seeds4_7": rat(ps(k, FB), ps("thr_agreement", FB)), "seeds0_3": rat(ps(k, FA), ps("thr_agreement", FA)),
                   "all8_batches1_12": rat(ps(k, slice(None), 0), ps("thr_agreement", slice(None), 0)), "all8_batches2_12": rat(ps(k, slice(None), 1), ps("thr_agreement", slice(None), 1)),
                   "mean_hits_per_batch_4_12": float(A[k][:, P0:].mean())}
        if k != "static_out": pool[k]["gap_captured_all8"] = rat(gapd(k), gapd("static_out")) if gapd("static_out").mean() > 0 else None
    out["pooled_batches_4_12"] = pool; out["thr_agreement_mean_hits_per_batch_4_12"] = float(thr[:, P0:].mean()); out["static_out_over_thr_all8"] = pool["static_out"]["all8"]
    sel = lambda sl: max(LAMS, key=lambda l: A[f"carry_offset_lam{l}"][sl, P0:].sum(1).mean())  # ties go to lam 10 (first in LAMS)
    lamA, lamB = sel(FA), sel(FB)
    out["lam_selection"] = {"selected_on_seeds0_3": lamA, "selected_on_seeds4_7": lamB, "mean_hits_4_12_by_lam_seeds0_3": {str(l): float(A[f"carry_offset_lam{l}"][FA, P0:].sum(1).mean()) for l in LAMS},
                            "mean_hits_4_12_by_lam_seeds4_7": {str(l): float(A[f"carry_offset_lam{l}"][FB, P0:].sum(1).mean()) for l in LAMS}}
    def fv(fn, fold):
        pa = lambda l, sl: A[fn(l)][sl, P0:].sum(1)
        return pa(lamA, FB) if fold == "A" else pa(lamB, FA) if fold == "B" else np.concatenate([pa(lamB, FA), pa(lamA, FB)])
    folds = {}
    for fold in ("A", "B", "pooled8"):
        thv = fv(lambda l: "thr_agreement", fold); car = fv(lambda l: f"carry_offset_lam{l}", fold); cp = fv(lambda l: "carry_plain", fold)
        w = {W: fv(lambda l, W=W: f"carry_window{W}_lam{l}", fold) for W in WINS}
        folds[fold] = {"carry_offset_over_thr": rat(car, thv), "carry_plain_over_thr": rat(cp, thv), "window_over_carry_offset": {str(W): rat(w[W], car) for W in WINS},
                       "window250_over_thr": rat(w[250], thv), "calibrate_once_over_thr": rat(fv(lambda l: "calibrate_once", fold), thv), "restart_offset_over_thr": rat(fv(lambda l: "restart_offset", fold), thv)}
    out["verdict_folds"] = folds  # keys: A = select lam on seeds 0-3, report 4-7 (verdict); B = select on 4-7, report 0-3; pooled8 = each seed reported with the lam chosen on the other half
    out["window_curve_both_lams_all8"] = {str(l): {str(W): rat(ps(f"carry_window{W}_lam{l}"), ps(f"carry_offset_lam{l}")) for W in WINS} for l in LAMS}
    bd = lambda k: {"err_rate": [float(x) for x in C["diag"][k]["err_rate"].mean(0)], "mean_z_agree": [float(x) for x in C["diag"][k]["mean_z_agree"].mean(0)], "n_labels": [int(x) for x in C["diag"][k]["n"].mean(0)]}
    out["carried_label_bias"] = {k: bd(k) for k in ("carry_plain", "carry_offset_lam10", "carry_offset_lam1")}; out["population_error_rate"] = float(C["base"][0].sum() / (NB * 250))
    res["cells"][f"{lab} B={B}"] = out

# ---------------- verdict
def cell_eval(fold):
    rows = {}
    for cn, c in res["cells"].items():
        f = c["verdict_folds"][fold]; ro, rp, rw = f["carry_offset_over_thr"], f["carry_plain_over_thr"], f["window_over_carry_offset"]["250"]
        rows[cn] = {"carry_offset_over_thr": ro["ratio"], "ci_lo": ro["ci_lo"], "carry_plain_over_thr": rp["ratio"], "window250_over_carry_offset": rw["ratio"],
                    "pass_cell_ge_1.03_and_ci_lo_gt_1": bool(ro["ratio"] >= PASS_R and ro["ci_lo"] > 1.0), "window250_within_2pct_two_sided": bool(abs(rw["ratio"] - 1) <= WIN_TOL),
                    "window250_not_worse_than_2pct": bool(rw["ratio"] >= 1 - WIN_TOL), "kill_cell_both_le_1.00": bool(rp["ratio"] <= KILL_R and ro["ratio"] <= KILL_R)}
    n_pass = sum(r["pass_cell_ge_1.03_and_ci_lo_gt_1"] for r in rows.values()); n_win = sum(r["window250_within_2pct_two_sided"] for r in rows.values()); n_kill = sum(r["kill_cell_both_le_1.00"] for r in rows.values())
    return {"cells": rows, "n_pass_cells_of_4": int(n_pass), "n_window250_within_2pct_of_4": int(n_win), "n_window250_not_worse_of_4": int(sum(r["window250_not_worse_than_2pct"] for r in rows.values())),
            "n_kill_cells_of_4": int(n_kill), "PASS": bool(n_pass == 4 and n_win == 4), "KILL": bool(n_kill >= 3)}
vA, vB, vP = cell_eval("A"), cell_eval("B"), cell_eval("pooled8")
outcome = "PASS" if vA["PASS"] else "KILL" if vA["KILL"] else "INCONCLUSIVE"
res["verdict"] = {"outcome": outcome, "regression_gate_passed": True, "verdict_fold": "carry_offset lam chosen per cell on order seeds 0-3, reported on 4-7 (4 order seeds, batches 4-12)", "foldA": vA,
                  "crosscheck_swapped_fold": vB, "crosscheck_pooled8": vP}
res["regression_gate_vs_L4a"] = {k: v for k, v in gate.items() if "diffs" not in k}
res["notes"] = ["Batches are exchangeable draws of one fixed policy, so there is no drift by construction; unbounded carry is an upper bound on what persistence can deliver.",
                "The 8 order seeds permute the same 3,000 prompts, so order-seed CIs are optimistic (they cover order and tie-break variation only, not prompt-sampling variation); the verdict fold has only 4 order seeds.",
                "Carried labels come from the learner's own selection-biased stream (see carried_label_bias: mean global z-agreement of the labels in hand is far below 0), which is the risk this experiment tests.",
                "Standardization: restart arms standardize on the current 250-prompt batch exactly as L4a; carry_offset and carry_window use one global label-free z-score over all 3,000 prompts (offset = -2 * global z agreement); carry_plain keeps L4a's rule (StandardScaler fit on the labels in hand), which is invariant to any extra affine transform.",
                "static_out is a logistic rule fit with free labels on the 2,750 prompts outside the batch: an optimistic ceiling and a no-budget reference, never a policy. calibrate_once charges 50 random labels inside the batch-1 budget only.",
                "One population, one task (GSM8K judge_error labels from E04), offline replay."]
json.dump(res, open(os.path.join(rd, "result.json"), "w"), indent=1)

# ---------------- ledger markdown
L = []; P = L.append
f3 = lambda r: f"{r['ratio']:.3f} ({r['ci_lo']:.3f} to {r['ci_hi']:.3f})"; fb = lambda x: "never" if x is None else str(x)
P(f"# L5a ledger table (CPU replay, regression gate vs runs/L4a: {res['regression_gate_vs_L4a']['differences']} differences in {res['regression_gate_vs_L4a']['checks_compared']} per-seed comparisons, self-check differences {res['regression_gate_vs_L4a']['selfcheck_differences']} in {res['regression_gate_vs_L4a']['selfcheck_compared']})")
P(f"\nVerdict: **{outcome}** (verdict fold: carry_offset lam chosen on order seeds 0-3, reported on 4-7; pooled over batches 4-12; PASS cells {vA['n_pass_cells_of_4']} of 4, window250 within 2% in {vA['n_window250_within_2pct_of_4']} of 4, KILL cells {vA['n_kill_cells_of_4']} of 4). Swapped fold: PASS cells {vB['n_pass_cells_of_4']}/4, KILL cells {vB['n_kill_cells_of_4']}/4. Pooled 8: PASS cells {vP['n_pass_cells_of_4']}/4, KILL cells {vP['n_kill_cells_of_4']}/4.")
P("\nStandardization (stated per the spec): restart arms standardize on the current 250-prompt batch exactly as L4a; carry_offset and carry_window use ONE global label-free z-score over all 3,000 prompts so carried and current rows are comparable; carry_plain keeps L4a's rule (StandardScaler fit on the labels in hand), invariant to any further affine transform.")
P("\nCaveats: batches are exchangeable draws from one fixed policy, so there is no drift by construction and unbounded carry is an upper bound; the 8 order seeds permute the same 3,000 prompts, so order-seed CIs are optimistic (the verdict fold has 4 order seeds); carried labels come from the learner's own selection-biased stream, which is the risk being tested; one population, one task, offline replay.")
for cn, c in res["cells"].items():
    B = c["budget"]; P(f"\n## {cn} (per-step budget {B} of 250; mean errors per batch {c['mean_base_errors_per_batch']:.1f}; 8 order seeds x 12 batches)")
    P("\nErrors caught per batch index (mean over 8 order seeds):\n"); P("| arm | " + " | ".join(str(i + 1) for i in range(NB)) + " | mean 4-12 |"); P("|---|" + "---|" * (NB + 1))
    for k in ["thr_agreement"] + ARMS: P(f"| {k} | " + " | ".join(f"{x:.1f}" for x in c["per_batch_mean_hits"][k]) + f" | {np.mean(c['per_batch_mean_hits'][k][P0:]):.1f} |")
    P("\nArm / thr_agreement by batch index (mean over 8 order seeds; per-batch 95% CIs are in result.json and the plot):\n"); P("| arm | " + " | ".join(str(i + 1) for i in range(NB)) + " |"); P("|---|" + "---|" * NB)
    for k in ["restart_plain", "restart_offset", "carry_plain", "carry_offset_lam10", "carry_offset_lam1", "carry_window250_lam10", "carry_window250_lam1", "calibrate_once", "static_out"]:
        P(f"| {k} | " + " | ".join(f"{x:.3f}" for x in c["per_batch_ratio_vs_thr"][k]["ratio"]) + " |")
    P(f"\nPooled over batches 4-12 (thr_agreement mean {c['thr_agreement_mean_hits_per_batch_4_12']:.1f} errors per batch; static_out / thr = {f3(c['static_out_over_thr_all8'])}). Gap captured = (arm - thr) / (static_out - thr).\n")
    P("| arm | / thr, all 8 seeds (95% CI) | seeds 4-7 | seeds 0-3 | first batch >= 1.00 (raw / trailing 3) | first batch >= 1.03 (raw / trailing 3) | gap captured, all 8 (95% CI) |"); P("|---|---|---|---|---|---|---|")
    for k in ARMS:
        p = c["pooled_batches_4_12"][k]; b = c["per_batch_ratio_vs_thr"][k]; g = p.get("gap_captured_all8")
        P(f"| {k} | {f3(p['all8'])} | {p['seeds4_7']['ratio']:.3f} | {p['seeds0_3']['ratio']:.3f} | {fb(b['first_batch_ge_1.00'])} / {fb(b['first_batch_trailing3_ge_1.00'])} | {fb(b['first_batch_ge_1.03'])} / {fb(b['first_batch_trailing3_ge_1.03'])} | " + ("n/a" if g is None else f"{g['ratio']:.0%} ({g['ci_lo']:.0%} to {g['ci_hi']:.0%})") + " |")
    ls = c["lam_selection"]; P(f"\nlam choice: selected on order seeds 0-3 = {ls['selected_on_seeds0_3']} (mean hits 4-12 by lam {ls['mean_hits_4_12_by_lam_seeds0_3']}); selected on 4-7 = {ls['selected_on_seeds4_7']}.")
    P("\nVerdict-fold ratios (pooled batches 4-12; CI over order seeds):\n"); P("| fold | carry_offset / thr | carry_plain / thr | window250 / carry_offset | window125 / carry_offset | window500 / carry_offset | calibrate_once / thr | restart_offset / thr |"); P("|---|---|---|---|---|---|---|---|")
    for fn, f in c["verdict_folds"].items():
        w = f["window_over_carry_offset"]; P(f"| {fn} | {f3(f['carry_offset_over_thr'])} | {f3(f['carry_plain_over_thr'])} | {f3(w['250'])} | {f3(w['125'])} | {f3(w['500'])} | {f3(f['calibrate_once_over_thr'])} | {f3(f['restart_offset_over_thr'])} |")
    P("\nLabel staleness (window W / unbounded carry_offset, pooled batches 4-12, all 8 order seeds): " + "; ".join(f"lam {l}: " + ", ".join(f"W{W} {c['window_curve_both_lams_all8'][str(l)][str(W)]['ratio']:.3f}" for W in WINS) for l in LAMS))
    cb = c["carried_label_bias"]; P(f"\nCarried-label selection bias (mean over order seeds; population error rate {c['population_error_rate']:.3f}, global z agreement 0): " + "; ".join(f"{k}: after batch 1/4/12 n = {v['n_labels'][0]}/{v['n_labels'][3]}/{v['n_labels'][11]}, error rate {v['err_rate'][0]:.3f}/{v['err_rate'][3]:.3f}/{v['err_rate'][11]:.3f}, mean z agreement {v['mean_z_agree'][0]:.2f}/{v['mean_z_agree'][3]:.2f}/{v['mean_z_agree'][11]:.2f}" for k, v in cb.items()))
P("\n## Verdict detail (fold A = verdict)\n"); P("| cell | carry_offset / thr (CI lo) | carry_plain / thr | window250 / carry | pass cell | window within 2% | kill cell |"); P("|---|---|---|---|---|---|---|")
for cn, r in vA["cells"].items(): P(f"| {cn} | {r['carry_offset_over_thr']:.3f} ({r['ci_lo']:.3f}) | {r['carry_plain_over_thr']:.3f} | {r['window250_over_carry_offset']:.3f} | {r['pass_cell_ge_1.03_and_ci_lo_gt_1']} | {r['window250_within_2pct_two_sided']} | {r['kill_cell_both_le_1.00']} |")
open(os.path.join(rd, "ledger_table.md"), "w").write("\n".join(L) + "\n"); print("\n".join(L))

# ---------------- figure
try:
    import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
    INK, MUTE = "#222222", "#6b6b6b"; plt.rcParams.update({"text.color": INK, "axes.labelcolor": INK, "xtick.color": MUTE, "ytick.color": MUTE, "axes.edgecolor": "#bbbbbb", "font.size": 9})
    fig, axs = plt.subplots(2, 2, figsize=(11, 7.4), sharex=True); x = np.arange(1, NB + 1)
    for ax, (cn, c) in zip(axs.ravel(), res["cells"].items()):
        lam = c["lam_selection"]["selected_on_seeds0_3"]; pbr = c["per_batch_ratio_vs_thr"]
        ax.axhline(1.0, color="#999999", lw=0.8); ax.axhline(1.03, color="#cccccc", lw=0.8, ls=":"); ax.grid(axis="y", color="#eeeeee", lw=0.6); ax.set_axisbelow(True)
        for k, nm, col, ls, mk in ((f"carry_offset_lam{lam}", f"carry_offset (lam {lam}, chosen on seeds 0-3)", "#D55E00", "-", "o"), ("carry_plain", "carry_plain (greedy n3 R1)", "#0072B2", "-", "s"),
                                    (f"carry_window250_lam{lam}", "carry_window250 (same lam)", "#009E73", "-", "^"), ("calibrate_once", "calibrate_once (50 labels, batch 1 only)", "#CC79A7", "--", "D"),
                                    ("restart_offset", "restart_offset (no carry)", "#666666", "--", "x"), ("static_out", "static_out (free-label ceiling)", "#111111", ":", None)):
            ax.plot(x, pbr[k]["ratio"], color=col, ls=ls, marker=mk, ms=4, lw=1.3, label=nm)
            if k == f"carry_offset_lam{lam}": ax.fill_between(x, pbr[k]["ci_lo"], pbr[k]["ci_hi"], color=col, alpha=0.12, lw=0)
        ax.set_title(cn, fontsize=10, loc="left"); ax.set_xticks(x)
        if cn.startswith("7B"): ax.set_xlabel("batch index (labels in hand grow with it)")
        ax.set_ylabel("arm / thr_agreement" if cn.endswith("B=62") else "")
    h, l = axs[0, 0].get_legend_handles_labels(); fig.legend(h, l, loc="lower center", ncol=3, fontsize=8, frameon=False); fig.tight_layout(rect=(0, 0.08, 1, 0.95))
    fig.suptitle("L5a: errors caught relative to thr_agreement by batch index (mean over 8 order seeds; band = 95% CI of the chosen carry_offset)", fontsize=9.5, x=0.01, ha="left", y=0.995)
    fig.savefig(os.path.join(rd, "carry_ratio_vs_batch.png"), dpi=130)
except Exception as ex: print("figure skipped:", repr(ex))
