"""L4a summary (CPU). Usage: l4a_summarize.py <runs/L4a>
Reads sim_<name>_m<m>_s<subset>.json (m in 250, 500, 1000; subsets 0..15; 8 policy seeds each; name = 3B and 7B-AWQ judge_error labels) and writes result.json, ratio_vs_B.png and markdown to stdout.

Selection protocol (cloned from L2b): per (label set, m, budget) the best member of the charged fixed family {thr_agreement, recal/seeded/split_charged_n with n in the L2b list}
and the best learned variant are chosen on subsets 0..7 (mean over subsets and the 8 policy seeds) and reported on subsets 8..15 (fold A, the verdict fold). Fold B swaps the roles
(select on 8..15, report on 0..7). Pooled-16 = the 16 held-out subset values (8 from each fold). Ratio = mean learned hits / mean fixed hits.
CI = percentile bootstrap (10000 resamples) over the held-out subsets (subset-cluster; per-subset value = mean over the 8 policy seeds). static is an upper bound and never a fixed member.
Learned families: plain = greedy_online, greedy_n<n0>_R<R> (L2b grid plus six small-end points), greedy_seeded_n25_R25; offset = offset_n<n0>_R<R>_lam<lam> (12 arms);
pooled = plain + offset (the "best learned family" of the verdict: selected on subsets 0..7, reported on 8..15).
Sensitivity only (never in the verdict): the fixed family extended with recal/seeded/split_charged at n = budget // 2 (removes the L2b cap truncation).
Subsets are draws from the same 3,000 prompts (pairwise overlap about m/3000 = 8%, 17%, 33%): a scale and subset-variance test, not a fresh population.

Pre-registered verdict (L4a spec), B = absolute budget = floor(budget fraction x m):
  VOID          if the regression gate against runs/L2b (m = 500, subset 0) did not pass (or gate_check.json is missing)
  PASS          if the pooled selected learned ratio is >= 1.02 with CI lower bound > 1 in at least 3 of the 4 cells with B = 125 (m = 250 at 50%, m = 500 at 25%, both judges)
                AND the ratio is >= 0.97 in all 12 cells with B <= 125 (m = 250 at 10/25/50%, m = 500 at 10/25%, m = 1000 at 10%, both judges)
  KILL          if no learner variant (plain or offset) reaches ratio 1.00 in any of the four B = 125 cells (variant held-out mean / selected-fixed held-out mean, max over all variants)
                AND the selected offset arm is below 0.97 in at least 3 of the 12 cells with B <= 125
  INCONCLUSIVE  otherwise"""
import sys, os, json, re, glob
import numpy as np

rd = sys.argv[1]
SETS = {"3B judge_error": "Qwen2.5-3B-Instruct", "7B-AWQ judge_error": "Qwen2.5-7B-Instruct-AWQ"}
MS = [250, 500, 1000]; BUD = ["0.1", "0.25", "0.5"]; NSUB = 16; FA, FB = slice(0, 8), slice(8, 16); B = 10000
NCALS = (10, 15, 25, 50, 75, 100, 150, 200, 250)
PASS_R, PASS_FLOOR, KILL_R, KILL_FLOOR = 1.02, 0.97, 1.00, 0.97
rng = np.random.default_rng(0)

gate_path = os.path.join(rd, "gate", "gate_check.json"); gate = json.load(open(gate_path)) if os.path.exists(gate_path) else None
gate_ok = bool(gate and gate.get("passed"))
san_files = sorted(glob.glob(os.path.join(rd, "sanity", "san_*.json")))
sanity = None
if san_files:
    sd = [json.load(open(f)) for f in san_files]
    sanity = {"cells": [{"file": os.path.basename(f), "m": d["m"], "subset_seed": d["subset_seed"], "max_abs_diff_hits": d["max_abs_diff_overall"], "n_comparisons": d["n_comparisons"]} for f, d in zip(san_files, sd)],
              "max_abs_diff_hits_overall": max(d["max_abs_diff_overall"] for d in sd), "n_comparisons_total": sum(d["n_comparisons"] for d in sd)}
if not gate_ok:
    res = {"verdict": {"outcome": "VOID", "regression_gate_passed": False}, "regression_gate_m500_s0_vs_L2b": gate, "sanity_lam_inf": sanity}
    json.dump(res, open(os.path.join(rd, "result.json"), "w"), indent=1); print("VOID: regression gate did not pass", gate and {k: v for k, v in gate.items() if k not in ("diffs", "extra_diffs")}); sys.exit(0)

def load(name, m):
    H, meta = {}, []
    for s in range(NSUB):
        d = json.load(open(os.path.join(rd, f"sim_{name}_m{m}_s{s}.json"))); meta.append((d["n_sim"], d["hits"]))
        assert d["m"] == m and d["subset_seed"] == s
        for b in BUD:
            if s > 0: assert set(d[f"budget_{b}"]) == set(H[b]), f"policy set differs at subset {s} budget {b}"
            for p, v in d[f"budget_{b}"].items():
                assert len(v["per_seed"]) == 8
                H.setdefault(b, {}).setdefault(p, np.full((NSUB, 8), np.nan))[s] = v["per_seed"]
    return H, meta

def boot_ratio(L, F):
    """L, F: per-subset mean hits (n subsets). Ratio of means, subset-cluster percentile CI, SD of per-subset ratios."""
    n = len(L); idx = rng.integers(0, n, (B, n)); rb = L[idx].mean(1) / F[idx].mean(1); sr = L / F
    return {"ratio": float(L.mean() / F.mean()), "ci_lo": float(np.percentile(rb, 2.5)), "ci_hi": float(np.percentile(rb, 97.5)), "subset_ratios": [float(x) for x in sr],
            "subset_sd": float(sr.std(ddof=1)), "mean_learned": float(L.mean()), "mean_fixed": float(F.mean()), "n_subsets": int(n)}

nkey = lambda p: int(p.rsplit("_", 1)[1])
def fam_of(p):
    if p == "thr_agreement": return "thr"
    mm = re.fullmatch(r"(recal|seeded|split)_charged_(\d+)", p)
    if mm: return "fixed" if int(mm.group(2)) in NCALS else "fixed_extra"
    if p == "greedy_online" or re.fullmatch(r"greedy_(seeded_)?n\d+_R\d+", p): return "plain"
    if re.fullmatch(r"offset_n\d+_R\d+_lam\d+", p): return "offset"
    return None  # uniform, static, oracle
def arm(p):
    mm = re.fullmatch(r"(?:greedy|offset)_n(\d+)_R(\d+)(?:_lam(\d+))?", p)
    return None if mm is None else tuple(int(x) if x is not None else None for x in mm.groups())

result = {"protocol": __doc__.split("\n")[1:], "label_sets": {}}
for lab, name in SETS.items():
    R = {}
    for m in MS:
        H, meta = load(name, m); cellsM = {"n_sim_per_subset_mean": float(np.mean([x[0] for x in meta])), "errors_per_subset_mean": float(np.mean([x[1] for x in meta])), "budgets": {}}
        for b in BUD:
            h = H[b]; F = {k: sorted([p for p in h if fam_of(p) == k]) for k in ("thr", "fixed", "fixed_extra", "plain", "offset")}
            fixed_c = F["thr"] + F["fixed"]; fixed_ext = fixed_c + F["fixed_extra"]; fams = {"plain": F["plain"], "offset": F["offset"], "pooled": F["plain"] + F["offset"]}
            mean_on = lambda ps, sl: {p: float(h[p][sl].mean()) for p in ps}
            def pick(cands, sl): d = mean_on(cands, sl); return max(d, key=d.get)
            cell = {"n_sim_mean": cellsM["n_sim_per_subset_mean"], "budget_abs": int(float(b) * m), "fixed_selected_on_0_7": pick(fixed_c, FA), "fixed_selected_on_8_15": pick(fixed_c, FB),
                    "fixed_ext_selected_on_0_7": pick(fixed_ext, FA), "fixed_ext_selected_on_8_15": pick(fixed_ext, FB),
                    "heldout_8_15_mean_all_policies": mean_on(list(h), FB), "all16_mean_all_policies": {p: float(h[p].mean()) for p in h}}
            bfA, bfB, bxA, bxB = cell["fixed_selected_on_0_7"], cell["fixed_selected_on_8_15"], cell["fixed_ext_selected_on_0_7"], cell["fixed_ext_selected_on_8_15"]
            FA_, FB_ = h[bfA][FB].mean(1), h[bfB][FA].mean(1)  # selected fixed, held-out per-subset means (fold A, fold B)
            XA_, XB_ = h[bxA][FB].mean(1), h[bxB][FA].mean(1)
            for fk, cands in fams.items():
                blA, blB = pick(cands, FA), pick(cands, FB); LA, LB = h[blA][FB].mean(1), h[blB][FA].mean(1)
                cell[fk] = {"selected_on_0_7": blA, "selected_on_8_15": blB, "vs_selected_fixed": boot_ratio(LA, FA_), "swapped_fold": boot_ratio(LB, FB_),
                            "pooled_16_crossfit": boot_ratio(np.concatenate([LA, LB]), np.concatenate([FA_, FB_])),
                            "vs_thr_agreement": boot_ratio(LA, h["thr_agreement"][FB].mean(1)), "vs_selected_fixed_extended_family": boot_ratio(LA, XA_),
                            "swapped_fold_vs_extended_family": boot_ratio(LB, XB_),
                            "heldout_max_sensitivity": {"learned": pick(cands, FB), **boot_ratio(h[pick(cands, FB)][FB].mean(1), FA_)}}
            # every variant against the selected fixed rule (held-out means over subsets 8..15; swapped: subsets 0..7 against the fixed rule selected on 8..15)
            vr = {p: float(h[p][FB].mean() / FA_.mean()) for p in fams["pooled"]}; vr_sw = {p: float(h[p][FA].mean() / FB_.mean()) for p in fams["pooled"]}
            cell["variant_ratio_heldout"] = vr; cell["variant_ratio_swapped"] = vr_sw
            cell["max_variant"] = {"name": max(vr, key=vr.get), "ratio": max(vr.values()), "swapped_name": max(vr_sw, key=vr_sw.get), "swapped_ratio": max(vr_sw.values()),
                                   "n_variants_ge_1": int(sum(v >= 1.0 for v in vr.values())), "n_variants": len(vr),
                                   "plain_max": max(vr[p] for p in fams["plain"]), "offset_max": max(vr[p] for p in fams["offset"])}
            # bracketing of the plain grid: is the best random-start (n0, R) arm off the lower boundary of the cell's grid?
            grid = [p for p in F["plain"] if re.fullmatch(r"greedy_n\d+_R\d+", p)]; n0s = [arm(p)[0] for p in grid]; Rs = [arm(p)[1] for p in grid]
            def brk(p):
                a = arm(p)
                if a is None: return {"arm": p, "in_grid": False}
                return {"arm": p, "in_grid": True, "n0": a[0], "R": a[1], "at_smallest_corner": bool(a[0] == min(n0s) and a[1] == min(Rs)), "n0_at_lower_edge": bool(a[0] == min(n0s)),
                        "R_at_lower_edge": bool(a[1] == min(Rs)), "bracketed_below": bool(a[0] > min(n0s) and a[1] > min(Rs))}
            best_grid16 = max(grid, key=lambda p: h[p].mean()); l2b_grid = [p for p in grid if arm(p)[:2] in ((10, 5), (10, 10), (15, 15), (25, 10), (25, 25))]; best_l2b16 = max(l2b_grid, key=lambda p: h[p].mean())
            cell["plain_bracketing"] = {"selected_on_0_7": brk(cell["plain"]["selected_on_0_7"]), "best_grid_arm_all16": brk(best_grid16), "best_l2b_grid_arm_all16": best_l2b16,
                                        "grid_n0_values": sorted(set(n0s)), "grid_R_values": sorted(set(Rs))}
            cell["n_sweep_all16_mean"] = {f: {str(nkey(p)): float(h[p].mean()) for p in F["fixed"] + F["fixed_extra"] if p.startswith(f + "_charged_")} for f in ("recal", "seeded", "split")}
            cell["plain_sweep_all16_mean"] = {p: float(h[p].mean()) for p in F["plain"]}; cell["offset_sweep_all16_mean"] = {p: float(h[p].mean()) for p in F["offset"]}
            cellsM["budgets"][b] = cell
        R[str(m)] = cellsM
    result["label_sets"][lab] = R

# ---- L2b reference curve (fold A selected learned / selected fixed from runs/L2b/result.json)
l2b_path = os.path.join(os.path.dirname(os.path.abspath(rd)), "L2b", "result.json"); l2b_ref = {}
if os.path.exists(l2b_path):
    l2b = json.load(open(l2b_path))
    for lab in SETS:
        for m in MS:
            for b in BUD:
                x = l2b["headline"][lab][str(m)][b]["learned_over_fixed"]; l2b_ref[(lab, m, b)] = {"ratio": x["ratio"], "ci95": x["ci95"], "selected_learned": l2b["headline"][lab][str(m)][b]["selected_learned"]}
get = lambda lab, m, b: result["label_sets"][lab][str(m)]["budgets"][b]
cells_all = [(lab, m, b) for lab in SETS for m in MS for b in BUD]
B125 = [c for c in cells_all if get(*c)["budget_abs"] == 125]; SMALL = [c for c in cells_all if get(*c)["budget_abs"] <= 125]
assert len(B125) == 4 and len(SMALL) == 12, (len(B125), len(SMALL))
key = lambda c: f"{c[0]} m{c[1]} {float(c[2]):.0%} (B={get(*c)['budget_abs']})"

def pass_kill(fold):
    """fold: 'vs_selected_fixed' (verdict), 'swapped_fold', or 'pooled_16_crossfit' (cross-checks)."""
    p125 = {key(c): bool(get(*c)["pooled"][fold]["ratio"] >= PASS_R and get(*c)["pooled"][fold]["ci_lo"] > 1.0) for c in B125}
    pfloor = {key(c): bool(get(*c)["pooled"][fold]["ratio"] >= PASS_FLOOR) for c in SMALL}
    off_low = {key(c): bool(get(*c)["offset"][fold]["ratio"] < KILL_FLOOR) for c in SMALL}
    return {"pass_cells_B125_ratio_ge_1.02_and_ci_lo_gt_1": p125, "n_pass_cells_B125_of_4": int(sum(p125.values())), "pooled_ratio_ge_0.97_cells_B_le_125": pfloor, "n_floor_cells_of_12": int(sum(pfloor.values())),
            "offset_ratio_lt_0.97_cells_B_le_125": off_low, "n_offset_below_0.97_of_12": int(sum(off_low.values())),
            "PASS": bool(sum(p125.values()) >= 3 and all(pfloor.values()))}
vA, vS, vP = pass_kill("vs_selected_fixed"), pass_kill("swapped_fold"), pass_kill("pooled_16_crossfit")
mv = {key(c): {"name": get(*c)["max_variant"]["name"], "ratio": round(get(*c)["max_variant"]["ratio"], 4), "swapped_ratio": round(get(*c)["max_variant"]["swapped_ratio"], 4),
               "n_variants_ge_1": get(*c)["max_variant"]["n_variants_ge_1"], "of": get(*c)["max_variant"]["n_variants"]} for c in B125}
no_variant_ge_1 = all(get(*c)["max_variant"]["ratio"] < KILL_R for c in B125); no_variant_ge_1_sw = all(get(*c)["max_variant"]["swapped_ratio"] < KILL_R for c in B125)
kill_A = bool(no_variant_ge_1 and vA["n_offset_below_0.97_of_12"] >= 3); kill_S = bool(no_variant_ge_1_sw and vS["n_offset_below_0.97_of_12"] >= 3)
outcome = "PASS" if vA["PASS"] else "KILL" if kill_A else "INCONCLUSIVE"
result["verdict"] = {"outcome": outcome, "regression_gate_passed": True, "verdict_fold": "pooled learned family selected on subsets 0-7, reported on 8-15 (fold A)",
                     "PASS_conditions_foldA": vA, "KILL_conditions_foldA": {"no_variant_reaches_1.00_in_any_B125_cell": no_variant_ge_1, "max_variant_per_B125_cell": mv,
                                                                          "n_offset_below_0.97_of_12": vA["n_offset_below_0.97_of_12"], "KILL": kill_A},
                     "crosscheck_swapped_fold": {"PASS": vS["PASS"], "n_pass_cells_B125_of_4": vS["n_pass_cells_B125_of_4"], "n_floor_cells_of_12": vS["n_floor_cells_of_12"],
                                                 "no_variant_reaches_1.00_in_any_B125_cell": no_variant_ge_1_sw, "n_offset_below_0.97_of_12": vS["n_offset_below_0.97_of_12"], "KILL": kill_S},
                     "crosscheck_pooled16": {"PASS": vP["PASS"], "n_pass_cells_B125_of_4": vP["n_pass_cells_B125_of_4"], "n_floor_cells_of_12": vP["n_floor_cells_of_12"],
                                             "n_offset_below_0.97_of_12": vP["n_offset_below_0.97_of_12"]}}
result["regression_gate_m500_s0_vs_L2b"] = gate; result["sanity_lam_inf"] = sanity
result["notes"] = {"protocol_change_item": "Out-of-fold cell-encoding delta AUC is not applicable to L4a: no prompt-space features are used (only the ten L2b cheap+judge columns, standardized on the m sim prompts without labels), so the Phase 1 protocol change (out-of-fold cell-encoding delta AUC) is neither triggered nor tested here.",
                   "offset_feature_convention": "w is a 9-vector on the standardized non-agreement features; the agreement slope is fixed at -2 (standardized units); intercept unpenalized; no class weights; refit skipped while the labels are one class."}

# ---- curve: ratio vs absolute B, new arms overlaid, L2b reference
rd3 = lambda x: round(x, 3)
def cv(c):
    cell = get(*c); r = lambda k, f="vs_selected_fixed": cell[k][f]
    return {"label": c[0], "m": c[1], "budget_frac": float(c[2]), "B": cell["budget_abs"], "L2b_selected_learned_over_fixed": None if c not in l2b_ref else rd3(l2b_ref[c]["ratio"]),
            "L4a_plain": rd3(r("plain")["ratio"]), "L4a_offset": rd3(r("offset")["ratio"]), "L4a_pooled": rd3(r("pooled")["ratio"]), "L4a_pooled_ci95": [rd3(r("pooled")["ci_lo"]), rd3(r("pooled")["ci_hi"])],
            "L4a_pooled_swapped": rd3(r("pooled", "swapped_fold")["ratio"]), "L4a_pooled_pooled16": rd3(r("pooled", "pooled_16_crossfit")["ratio"]),
            "offset_over_thr_agreement": rd3(r("offset", "vs_thr_agreement")["ratio"]), "offset_over_thr_ci95": [rd3(r("offset", "vs_thr_agreement")["ci_lo"]), rd3(r("offset", "vs_thr_agreement")["ci_hi"])],
            "pooled_over_extended_fixed_family": rd3(r("pooled", "vs_selected_fixed_extended_family")["ratio"]), "max_variant_ratio": rd3(cell["max_variant"]["ratio"]), "max_variant": cell["max_variant"]["name"]}
curve = sorted([cv(c) for c in cells_all], key=lambda d: (d["B"], d["label"], d["m"])); result["curve_ratio_vs_B"] = curve
smallest_B = {lab: {k: min([d["B"] for d in curve if d["label"] == lab and d[k] is not None and d[k] >= 1.0], default=None) for k in ("L2b_selected_learned_over_fixed", "L4a_pooled", "L4a_plain", "L4a_offset")} for lab in SETS}
result["smallest_B_with_ratio_ge_1"] = smallest_B

rd1 = lambda r: {"ratio": round(r["ratio"], 3), "ci95": [round(r["ci_lo"], 3), round(r["ci_hi"], 3)], "subset_sd": round(r["subset_sd"], 3)}
def head(c):
    cell = get(*c); e = cell["heldout_8_15_mean_all_policies"]; fx = cell["fixed_selected_on_0_7"]
    out = {"budget_abs": cell["budget_abs"], "n_subsets_heldout": 8, "n_policy_seeds": 8, "selected_fixed": fx, "mean_errors_caught": {"thr_agreement": round(e["thr_agreement"], 1), "selected_fixed": round(e[fx], 1),
           "uniform": round(e["uniform"], 1), "static_out_free_label_ceiling": round(e["static"], 1), "oracle": round(e["oracle"], 1)}}
    for fk in ("plain", "offset", "pooled"):
        x = cell[fk]; sel = x["selected_on_0_7"]; out["mean_errors_caught"][f"selected_{fk}"] = round(e[sel], 1)
        out[fk] = {"selected": sel, "learned_over_fixed": rd1(x["vs_selected_fixed"]), "swapped_fold": rd1(x["swapped_fold"]), "pooled16": rd1(x["pooled_16_crossfit"]), "over_thr_agreement": rd1(x["vs_thr_agreement"]),
                   "over_extended_fixed_family": rd1(x["vs_selected_fixed_extended_family"]), "heldout_max_sensitivity": rd1(x["heldout_max_sensitivity"])}
    out["max_variant"] = {"name": cell["max_variant"]["name"], "ratio": round(cell["max_variant"]["ratio"], 3)}; out["plain_bracketing"] = cell["plain_bracketing"]["selected_on_0_7"]
    return out
result["headline"] = {lab: {str(m): {b: head((lab, m, b)) for b in BUD} for m in MS} for lab in SETS}
n_cells = len(cells_all); pb = [get(*c)["plain_bracketing"] for c in cells_all]
result["interior_check"] = {"n_cells": n_cells,
    "plain_selected_on_0_7_at_smallest_corner": int(sum(x["selected_on_0_7"].get("at_smallest_corner", False) for x in pb)),
    "plain_selected_on_0_7_bracketed_below": int(sum(x["selected_on_0_7"].get("bracketed_below", False) for x in pb)),
    "plain_selected_on_0_7_in_grid": int(sum(x["selected_on_0_7"]["in_grid"] for x in pb)),
    "best_grid_arm_all16_at_smallest_corner": int(sum(x["best_grid_arm_all16"].get("at_smallest_corner", False) for x in pb)),
    "best_grid_arm_all16_bracketed_below": int(sum(x["best_grid_arm_all16"].get("bracketed_below", False) for x in pb)),
    "best_L2b_grid_arm_all16_is_n10_R5": int(sum(x["best_l2b_grid_arm_all16"] == "greedy_n10_R5" for x in pb)),
    "selected_offset_on_0_7": {key(c): get(*c)["offset"]["selected_on_0_7"] for c in cells_all}, "selected_plain_on_0_7": {key(c): get(*c)["plain"]["selected_on_0_7"] for c in cells_all},
    "selected_plain_best_grid_arm_all16": {key(c): get(*c)["plain_bracketing"]["best_grid_arm_all16"]["arm"] for c in cells_all}}
json.dump(result, open(os.path.join(rd, "result.json"), "w"), indent=1)

# ---- figure
try:
    import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
    fig, axs = plt.subplots(1, 2, figsize=(11, 4.2), sharey=True)
    for ax, lab in zip(axs, SETS):
        rows = [d for d in curve if d["label"] == lab]
        ax.axhline(1.0, color="0.4", lw=0.8); ax.axhline(PASS_FLOOR, color="0.7", lw=0.8, ls=":"); ax.axhline(PASS_R, color="0.7", lw=0.8, ls=":"); ax.axvline(125, color="0.85", lw=0.8)
        for k, nm, col, mk, ls in (("L2b_selected_learned_over_fixed", "L2b (reference)", "#555555", "s", "--"), ("L4a_plain", "L4a plain (extended grid)", "#0072B2", "o", "-"),
                                   ("L4a_offset", "L4a offset (warm start)", "#D55E00", "^", "-"), ("L4a_pooled", "L4a pooled selected", "#009E73", "D", "-")):
            pts = sorted([(d["B"], d[k], d["m"]) for d in rows if d[k] is not None]); ax.plot([p[0] for p in pts], [p[1] for p in pts], color=col, marker=mk, ls=ls, lw=1.2, ms=4.5, label=nm)
        ax.set_xscale("log"); ax.set_xticks([25, 50, 100, 250, 500]); ax.set_xticklabels(["25", "50", "100", "250", "500"]); ax.set_xlabel("absolute budget B (verifications)"); ax.set_title(lab, fontsize=10)
    axs[0].set_ylabel("selected learned / selected fixed (held-out subsets 8-15)"); axs[0].legend(fontsize=8, loc="lower right"); fig.tight_layout(); fig.savefig(os.path.join(rd, "ratio_vs_B.png"), dpi=130)
except Exception as ex: print("figure skipped:", repr(ex))

# ---- markdown output
f1 = lambda x: f"{x:.1f}"
print("regression gate (m = 500, subset 0 vs runs/L2b):", {k: v for k, v in gate.items() if k not in ("diffs", "extra_diffs")})
print("lam = inf sanity (run_offset with w = 0 vs thr_agreement per-seed hits):", None if sanity is None else {k: v for k, v in sanity.items() if k != "cells"})
print("\nCaveat: subsets are draws from the same 3,000 prompts (pairwise overlap about 8%, 17%, 33% for m = 250, 500, 1000): a scale and subset-variance test, not a fresh population.")
for lab, Rl in result["label_sets"].items():
    print(f"\n## {lab}")
    for m in MS:
        print(f"\n### m = {m} (mean sim size {Rl[str(m)]['n_sim_per_subset_mean']:.0f}, mean errors per subset {Rl[str(m)]['errors_per_subset_mean']:.1f}); held-out subsets 8..15 x 8 policy seeds")
        print("| budget | uniform | thr_agreement | selected fixed (on 0-7) | selected plain (on 0-7) | selected offset (on 0-7) | selected pooled | static_out (upper bound) | oracle |"); print("|---|---|---|---|---|---|---|---|---|")
        for b, c in Rl[str(m)]["budgets"].items():
            e = c["heldout_8_15_mean_all_policies"]; fx = c["fixed_selected_on_0_7"]; sp, so, sg = (c[k]["selected_on_0_7"] for k in ("plain", "offset", "pooled"))
            print(f"| {float(b):.0%} ({c['budget_abs']}) | {f1(e['uniform'])} | {f1(e['thr_agreement'])} | {fx} {f1(e[fx])} | {sp} {f1(e[sp])} | {so} {f1(e[so])} | {sg} {f1(e[sg])} | {f1(e['static'])} | {f1(e['oracle'])} |")
        for b, c in Rl[str(m)]["budgets"].items():
            def line(nm, r, extra=""): print(f"  - {float(b):.0%} {nm}: {r['ratio']:.3f} (95% CI {r['ci_lo']:.3f} to {r['ci_hi']:.3f}; SD of per-subset ratios {r['subset_sd']:.3f}; means {r['mean_learned']:.1f} vs {r['mean_fixed']:.1f}; n subsets {r['n_subsets']}){extra}")
            line("pooled selected learned / selected fixed (VERDICT RATIO)", c["pooled"]["vs_selected_fixed"])
            line("plain selected / selected fixed", c["plain"]["vs_selected_fixed"]); line("offset selected / selected fixed", c["offset"]["vs_selected_fixed"])
            line("pooled swapped fold (select 8-15, report 0-7)", c["pooled"]["swapped_fold"], f" [fixed {c['fixed_selected_on_8_15']}, learned {c['pooled']['selected_on_8_15']}]")
            line("pooled 16-subset cross-fit", c["pooled"]["pooled_16_crossfit"])
            line("offset selected / thr_agreement", c["offset"]["vs_thr_agreement"]); line("pooled selected / thr_agreement", c["pooled"]["vs_thr_agreement"])
            line("SENSITIVITY pooled selected / extended fixed family (n = budget // 2 added)", c["pooled"]["vs_selected_fixed_extended_family"], f" [fixed {c['fixed_ext_selected_on_0_7']}]")
            mx = c["max_variant"]; print(f"  - {float(b):.0%} max over all {mx['n_variants']} learner variants of (variant held-out mean / selected fixed): {mx['ratio']:.3f} ({mx['name']}); plain max {mx['plain_max']:.3f}, offset max {mx['offset_max']:.3f}; variants >= 1.00: {mx['n_variants_ge_1']}")
            br = c["plain_bracketing"]["selected_on_0_7"]; print(f"  - {float(b):.0%} selected plain (n0, R) on 0-7: {br}; best random-start grid arm on all 16: {c['plain_bracketing']['best_grid_arm_all16']['arm']}; best L2b-grid arm on all 16: {c['plain_bracketing']['best_l2b_grid_arm_all16']}")
        print("  plain (n0, R) sweep, mean errors caught over all 16 subsets:")
        for b, c in Rl[str(m)]["budgets"].items(): print(f"  - {float(b):.0%}: " + ", ".join(f"{k.replace('greedy_', '')}={v:.1f}" for k, v in c["plain_sweep_all16_mean"].items()))
        print("  offset arm sweep, mean errors caught over all 16 subsets (thr_agreement for reference):")
        for b, c in Rl[str(m)]["budgets"].items(): print(f"  - {float(b):.0%}: thr={c['all16_mean_all_policies']['thr_agreement']:.1f}; " + ", ".join(f"{k.replace('offset_', '')}={v:.1f}" for k, v in c["offset_sweep_all16_mean"].items()))
print("\n## Ratio of selected learned over selected fixed vs absolute B (fold A held-out; L2b reference = L2b plain grid, same protocol)")
print("| B | label | m | frac | L2b | L4a plain | L4a offset | L4a pooled [95% CI] | pooled swapped / pooled-16 | offset / thr_agreement [95% CI] | pooled / ext. fixed | max variant |"); print("|---|---|---|---|---|---|---|---|---|---|---|---|")
for d in curve:
    print(f"| {d['B']} | {d['label'].split()[0]} | {d['m']} | {d['budget_frac']:.0%} | {d['L2b_selected_learned_over_fixed']} | {d['L4a_plain']} | {d['L4a_offset']} | {d['L4a_pooled']} [{d['L4a_pooled_ci95'][0]}, {d['L4a_pooled_ci95'][1]}] | "
          f"{d['L4a_pooled_swapped']} / {d['L4a_pooled_pooled16']} | {d['offset_over_thr_agreement']} [{d['offset_over_thr_ci95'][0]}, {d['offset_over_thr_ci95'][1]}] | {d['pooled_over_extended_fixed_family']} | {d['max_variant_ratio']} |")
print("\nSmallest B in the grid with ratio >= 1.00 (sparse grid: B in 25, 50, 62, 100, 125, 250, 500):", json.dumps(smallest_B))
print("\n## Interior check (is the plain optimum still the grid corner?)"); print(json.dumps({k: v for k, v in result["interior_check"].items() if not k.startswith("selected_")}, indent=1))
print("\n## VERDICT (PASS: >= 3 of 4 B = 125 cells at ratio >= 1.02 with CI lower bound > 1 AND ratio >= 0.97 in all 12 cells with B <= 125; KILL: no variant at 1.00 in any B = 125 cell AND selected offset below 0.97 in >= 3 of 12 cells)")
print(json.dumps(result["verdict"], indent=1))
