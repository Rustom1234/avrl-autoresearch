"""L2b summary (CPU). Usage: l2b_summarize.py <runs/L2b>
Reads sim_<name>_m<m>_s<subset>.json (m in 250, 500, 1000; subsets 0..15; 8 policy seeds each; name = 3B and 7B-AWQ judge_error labels) and writes result.json plus markdown to stdout.

Selection protocol (cloned from L2a, with subsets in place of splits): per (label set, m, budget) the best member of the charged fixed family {thr_agreement, recal/seeded/split_charged_n}
and the best learned variant {greedy_online, greedy_n<n0>_R<R>, greedy_seeded_n25_R25} are chosen on subsets 0..7 (mean over subsets and the 8 policy seeds) and reported on subsets 8..15
(fold A, the verdict fold). Fold B swaps the roles (select on 8..15, report on 0..7). Pooled = the 16 held-out subset values (8 from each fold). Ratio = mean learned hits / mean fixed hits.
CI = percentile bootstrap (10000 resamples) over the held-out subsets (subset-cluster; per-subset value = mean over the 8 policy seeds). static is an upper bound and never a fixed member.
Subsets are draws from the same 3,000 prompts (pairwise overlap about m/3000 = 8%, 17%, 33%): a scale and subset-variance test, not a fresh population.

Pre-registered verdict (L2b spec), cells = {3B, 7B-AWQ} x {25%, 50%} budget:
  VOID          if the m = 0 regression gate against runs/L2a did not pass (or gate_check.json is missing)
  PASS          if at m = 500 the ratio >= 1.03 with subset-cluster CI lower bound > 1 in all four cells (fold A)
  KILL          if at m = 500 and at m = 250 the ratio <= 1.00 in all four cells each (8 cells, fold A)
  INCONCLUSIVE  otherwise"""
import sys, os, json, re
import numpy as np

rd = sys.argv[1]
SETS = {"3B judge_error": "Qwen2.5-3B-Instruct", "7B-AWQ judge_error": "Qwen2.5-7B-Instruct-AWQ"}
MS = [250, 500, 1000]; BUD = ["0.1", "0.25", "0.5"]; VBUD = ["0.25", "0.5"]; NSUB = 16; FA, FB = slice(0, 8), slice(8, 16); B = 10000
PASS_R, KILL_R = 1.03, 1.00
L2A_REF = {"3B judge_error": {"0.25": 1.050, "0.5": 1.060}, "7B-AWQ judge_error": {"0.25": 1.139, "0.5": 1.061}}  # from the spec; overwritten from runs/L2a/result.json when present
rng = np.random.default_rng(0)

gate_path = os.path.join(rd, "gate", "gate_check.json"); gate = json.load(open(gate_path)) if os.path.exists(gate_path) else None
gate_ok = bool(gate and gate.get("passed"))
if not gate_ok:
    res = {"verdict": {"outcome": "VOID", "regression_gate_passed": False}, "regression_gate_m0_split0_vs_L2a": gate}
    json.dump(res, open(os.path.join(rd, "result.json"), "w"), indent=1); print("VOID: regression gate did not pass", gate and {k: v for k, v in gate.items() if k != "diffs"}); sys.exit(0)

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
def gkey(p): m = re.search(r"n(\d+)_R(\d+)$", p); return (int(m.group(1)), int(m.group(2)))
def families(h):
    return {"thr_agreement": ["thr_agreement"], "recal": sorted([p for p in h if p.startswith("recal_charged_")], key=nkey), "seeded": sorted([p for p in h if p.startswith("seeded_charged_")], key=nkey),
            "split": sorted([p for p in h if p.startswith("split_charged_")], key=nkey), "greedy": ["greedy_online"] + sorted([p for p in h if re.fullmatch(r"greedy_(seeded_)?n\d+_R\d+", p)], key=gkey)}
CORNERS = ("greedy_n10_R5", "greedy_n25_R25")  # smallest and largest (n0, R) of the 5-point grid

result = {"protocol": __doc__.split("\n")[2:9], "label_sets": {}}
for lab, name in SETS.items():
    R = {}
    for m in MS:
        H, meta = load(name, m); cellsM = {"n_sim_per_subset_mean": float(np.mean([x[0] for x in meta])), "errors_per_subset_mean": float(np.mean([x[1] for x in meta])), "budgets": {}}
        for b in BUD:
            h = H[b]; fam = families(h); fixed_c = fam["thr_agreement"] + fam["recal"] + fam["seeded"] + fam["split"]; learn_c = fam["greedy"]
            mean_on = lambda ps, sl: {p: float(h[p][sl].mean()) for p in ps}
            def fold(sel, ev):
                sf, sl = mean_on(fixed_c, sel), mean_on(learn_c, sel); bf, bl = max(sf, key=sf.get), max(sl, key=sl.get)
                return bf, bl, h[bl][ev].mean(1), h[bf][ev].mean(1)
            bfA, blA, LA, FA_ = fold(FA, FB); bfB, blB, LB, FB_ = fold(FB, FA)
            evA = mean_on(list(h), FB)  # held-out means (subsets 8..15) of every policy
            eA, eFa = mean_on(fixed_c, FB), mean_on(learn_c, FB); mxF, mxL = max(eA, key=eA.get), max(eFa, key=eFa.get)
            a16 = {p: float(h[p].mean()) for p in h}; fx16, lr16 = max(fixed_c, key=a16.get), max(learn_c, key=a16.get)
            cell = {"n_sim_mean": cellsM["n_sim_per_subset_mean"], "budget_abs": int(float(b) * m),
                    "fixed_selected_on_0_7": bfA, "learned_selected_on_0_7": blA, "fixed_selected_on_8_15": bfB, "learned_selected_on_8_15": blB,
                    "heldout_8_15_mean_all_policies": evA, "all16_mean_all_policies": a16,
                    "selected_learned_vs_selected_fixed": boot_ratio(LA, FA_), "swapped_fold": boot_ratio(LB, FB_),
                    "pooled_16_crossfit": boot_ratio(np.concatenate([LA, LB]), np.concatenate([FA_, FB_])),
                    "greedy_online_vs_selected_fixed": boot_ratio(h["greedy_online"][FB].mean(1), FA_),
                    "selected_learned_vs_thr_agreement": boot_ratio(LA, h["thr_agreement"][FB].mean(1)),
                    "sensitivity_max_on_heldout": {"fixed": mxF, "learned": mxL, **boot_ratio(h[mxL][FB].mean(1), h[mxF][FB].mean(1))},
                    "sensitivity_max_on_all16": {"fixed": fx16, "learned": lr16, **boot_ratio(h[lr16].mean(1), h[fx16].mean(1))},
                    "learned_selected_at_grid_corner": {"fold_A": blA in CORNERS, "fold_B": blB in CORNERS},
                    "fixed_selected_n_at_edge_of_allowed_list": {"fold_A": bool(bfA != "thr_agreement" and nkey(bfA) in (min(nkey(p) for p in fam["recal"]), max(nkey(p) for p in fam["recal"]))),
                                                                  "fold_B": bool(bfB != "thr_agreement" and nkey(bfB) in (min(nkey(p) for p in fam["recal"]), max(nkey(p) for p in fam["recal"])))} if fam["recal"] else None,
                    "n_sweep_all16_mean": {f: {str(nkey(p)): float(h[p].mean()) for p in fam[k]} for f, k in (("recal_charged", "recal"), ("seeded_charged", "seeded"), ("split_charged", "split"))},
                    "n0_R_sweep_all16_mean": {p: float(h[p].mean()) for p in fam["greedy"]}}
            cellsM["budgets"][b] = cell
        R[str(m)] = cellsM
    result["label_sets"][lab] = R

# ---- L2a full-pool reference for the ratio-vs-m curve
l2a_path = os.path.join(os.path.dirname(os.path.abspath(rd)), "L2a", "result.json"); l2a_ref = {}
if os.path.exists(l2a_path):
    l2a = json.load(open(l2a_path))
    for lab in SETS:
        l2a_ref[lab] = {b: float(l2a["headline"][lab][b]["ratio_selected_learned_over_selected_fixed"]) for b in BUD}
        l2a_ref[lab]["n_sim_per_split_mean"] = float(np.mean(l2a["label_sets"][lab]["n_sim_per_split"]))
else: l2a_ref = {lab: dict(v) for lab, v in L2A_REF.items()}
get = lambda lab, m, b, k="selected_learned_vs_selected_fixed": result["label_sets"][lab][str(m)]["budgets"][b][k]
flag = lambda lab, m, b, k: bool(get(lab, m, b, k)["ratio"] >= PASS_R and get(lab, m, b, k)["ci_lo"] > 1.0)
pass_c = {f"{lab} {float(b):.0%}": flag(lab, 500, b, "selected_learned_vs_selected_fixed") for lab in SETS for b in VBUD}
kill_c = {f"{lab} m{m} {float(b):.0%}": bool(get(lab, m, b)["ratio"] <= KILL_R) for lab in SETS for m in (250, 500) for b in VBUD}
pass_swapped = {f"{lab} {float(b):.0%}": flag(lab, 500, b, "swapped_fold") for lab in SETS for b in VBUD}
pass_pooled = {f"{lab} {float(b):.0%}": flag(lab, 500, b, "pooled_16_crossfit") for lab in SETS for b in VBUD}
kill_swapped = {f"{lab} m{m} {float(b):.0%}": bool(get(lab, m, b, "swapped_fold")["ratio"] <= KILL_R) for lab in SETS for m in (250, 500) for b in VBUD}
kill_sens = {f"{lab} m{m} {float(b):.0%}": bool(get(lab, m, b, "sensitivity_max_on_heldout")["ratio"] <= KILL_R) for lab in SETS for m in (250, 500) for b in VBUD}
if all(pass_c.values()): outcome = "PASS"
elif all(kill_c.values()): outcome = "KILL"
else: outcome = "INCONCLUSIVE"
result["verdict"] = {"outcome": outcome, "regression_gate_passed": True, "pass_cells_m500_foldA": pass_c, "n_pass_cells_of_4": int(sum(pass_c.values())), "kill_cells_m250_m500_foldA": kill_c, "n_kill_cells_of_8": int(sum(kill_c.values())),
                     "crosscheck_pass_cells_swapped_fold": pass_swapped, "crosscheck_pass_cells_pooled16": pass_pooled, "crosscheck_kill_cells_swapped_fold": kill_swapped,
                     "crosscheck_n_kill_cells_of_8_swapped": int(sum(kill_swapped.values())), "crosscheck_n_kill_cells_of_8_heldout_max_sensitivity": int(sum(kill_sens.values()))}
result["regression_gate_m0_split0_vs_L2a"] = gate
result["l2a_full_pool_reference_ratios"] = l2a_ref
rd1 = lambda r: {"ratio": round(r["ratio"], 3), "ci95": [round(r["ci_lo"], 3), round(r["ci_hi"], 3)], "subset_sd": round(r["subset_sd"], 3)}
result["headline"] = {lab: {str(m): {b: {"learned_over_fixed": rd1(c["selected_learned_vs_selected_fixed"]), "swapped_fold": rd1(c["swapped_fold"]), "pooled16": rd1(c["pooled_16_crossfit"]),
                                         "heldout_max_sensitivity": rd1(c["sensitivity_max_on_heldout"]), "selected_learned": c["learned_selected_on_0_7"], "selected_fixed": c["fixed_selected_on_0_7"],
                                         "mean_learned": round(c["selected_learned_vs_selected_fixed"]["mean_learned"], 1), "mean_fixed": round(c["selected_learned_vs_selected_fixed"]["mean_fixed"], 1),
                                         "budget_abs": c["budget_abs"], "n_subsets_heldout": 8, "n_policy_seeds": 8}
                                     for b, c in R[str(m)]["budgets"].items()} for m in MS} for lab, R in result["label_sets"].items()}
json.dump(result, open(os.path.join(rd, "result.json"), "w"), indent=1)

# ---- markdown output
f1 = lambda x: f"{x:.1f}"
print("regression gate (m = 0, split 0 vs runs/L2a):", {k: v for k, v in gate.items() if k != "diffs"})
print("\nCaveat: subsets are draws from the same 3,000 prompts (pairwise overlap about 8%, 17%, 33% for m = 250, 500, 1000): a scale and subset-variance test, not a fresh population.")
for lab, R in result["label_sets"].items():
    print(f"\n## {lab}")
    for m in MS:
        print(f"\n### m = {m} (mean sim size {R[str(m)]['n_sim_per_subset_mean']:.0f}, mean errors per subset {R[str(m)]['errors_per_subset_mean']:.1f}); held-out subsets 8..15 x 8 policy seeds")
        print("| budget | uniform | thr_agreement | selected fixed (on 0-7) | selected learned (on 0-7) | greedy_online | static_out (upper bound) | oracle |"); print("|---|---|---|---|---|---|---|---|")
        for b, c in R[str(m)]["budgets"].items():
            e = c["heldout_8_15_mean_all_policies"]
            print(f"| {float(b):.0%} ({c['budget_abs']}) | {f1(e['uniform'])} | {f1(e['thr_agreement'])} | {c['fixed_selected_on_0_7']} {f1(e[c['fixed_selected_on_0_7']])} | {c['learned_selected_on_0_7']} {f1(e[c['learned_selected_on_0_7']])} | {f1(e['greedy_online'])} | {f1(e['static'])} | {f1(e['oracle'])} |")
        for b, c in R[str(m)]["budgets"].items():
            def line(nm, r, extra=""): print(f"  - {float(b):.0%} {nm}: {r['ratio']:.3f} (95% CI {r['ci_lo']:.3f} to {r['ci_hi']:.3f}; SD of per-subset ratios {r['subset_sd']:.3f}; means {r['mean_learned']:.1f} vs {r['mean_fixed']:.1f}; n subsets {r['n_subsets']}){extra}")
            line("selected learned / selected fixed (PRIMARY)", c["selected_learned_vs_selected_fixed"], "" if b in VBUD else " [10%: reported, not in the verdict]")
            line("swapped fold (select 8-15, report 0-7)", c["swapped_fold"], f" [fixed {c['fixed_selected_on_8_15']}, learned {c['learned_selected_on_8_15']}]")
            line("pooled 16-subset cross-fit", c["pooled_16_crossfit"])
            line("greedy_online / selected fixed", c["greedy_online_vs_selected_fixed"])
            line("held-out-max sensitivity", c["sensitivity_max_on_heldout"], f" [fixed {c['sensitivity_max_on_heldout']['fixed']}, learned {c['sensitivity_max_on_heldout']['learned']}]")
            print(f"  - {float(b):.0%} selected learned at (10,5)/(25,25) grid corner: {c['learned_selected_at_grid_corner']}; fixed n at edge of allowed list: {c['fixed_selected_n_at_edge_of_allowed_list']}")
        print("  (n0, R) sweep, mean errors caught over all 16 subsets:")
        for b, c in R[str(m)]["budgets"].items(): print(f"  - {float(b):.0%}: " + ", ".join(f"{k.replace('greedy_', '')}={v:.1f}" for k, v in c["n0_R_sweep_all16_mean"].items()))
print("\n## Ratio of selected learned over selected fixed vs m (fold A held-out; pooled 16-subset cross-fit in brackets; L2a full-pool ratio last)")
print("| label | budget | m=250 | m=500 | m=1000 | L2a full pool (n_sim about 2200) |"); print("|---|---|---|---|---|---|")
for lab in SETS:
    for b in BUD:
        cs = [result["label_sets"][lab][str(m)]["budgets"][b] for m in MS]
        print(f"| {lab} | {float(b):.0%} | " + " | ".join(f"{c['selected_learned_vs_selected_fixed']['ratio']:.3f} [{c['pooled_16_crossfit']['ratio']:.3f}]" for c in cs) + f" | {l2a_ref[lab].get(b, float('nan')):.3f} |")
print("\n## VERDICT (pass cells: m = 500 x {3B, 7B-AWQ} x {25%, 50%}; kill cells: m in 250, 500 x the same four)")
print(json.dumps(result["verdict"], indent=1))
