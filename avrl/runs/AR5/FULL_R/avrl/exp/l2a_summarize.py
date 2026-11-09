"""L2a summary (CPU). Usage: l2a_summarize.py <runs/L2a>
Reads sim_<name>_split<s>.json (s = 0..3, 8 policy seeds each, name = 3B and 7B-AWQ judge_error labels) and writes result.json plus markdown tables to stdout.

Selection protocol (same as L1b): the best member of the extended charged fixed family {thr_agreement, recal_charged_n, seeded_charged_n, split_charged_n} and the
best learned variant {greedy_online (L1b reference setting), greedy_n<n0>_R<R>, greedy_seeded_n<n0>_R<R>} are chosen per (label set, budget) on policy seeds 0..3
averaged over the 4 splits. All verdict numbers use policy seeds 4..7 on the same 4 splits = 16 paired runs. Ratio R = mean(learned errors caught) / mean(fixed errors caught)
over the 16 runs; CIs are percentile bootstraps (10000 resamples): over the 16 paired runs, and a cluster bootstrap over the 4 splits (the pass criterion); also reported
is the SD of the 4 per-split ratios. static (free labels) is an upper bound and is never a member of the fixed family.

Pre-registered verdict (L2a spec), cells = {3B, 7B-AWQ} x {25%, 50%} budget (4 cells):
  VOID          if the split-0 regression gate against runs/L1b did not pass (or gate_check.json is missing)
  PASS          if R >= 1.03 with split-cluster CI lower bound > 1 in all four cells (repeated refitting on its own labels is the active ingredient)
  KILL          if R <= 1.02 in all four cells (calibration-design effect: a one-shot fixed rule matches the greedy edge)
  INCONCLUSIVE  otherwise (a cell between 1.02 and 1.03, a cluster CI that includes 1, or a mix of passing and killing cells)"""
import sys, os, json, re
import numpy as np

rd = sys.argv[1]
SETS = {"3B judge_error": "Qwen2.5-3B-Instruct", "7B-AWQ judge_error": "Qwen2.5-7B-Instruct-AWQ"}
BUD = ["0.1", "0.25", "0.5"]; VBUD = ["0.25", "0.5"]; SPLITS = [0, 1, 2, 3]; SEL = slice(0, 4); EV = slice(4, 8); B = 10000
PASS_R, KILL_R = 1.03, 1.02
rng = np.random.default_rng(0)

def load(name):
    H, meta = {}, []
    for s in SPLITS:
        d = json.load(open(os.path.join(rd, f"sim_{name}_split{s}.json"))); meta.append((d["n_sim"], d["hits"], d["n_buckets"]))
        assert d["split_seed"] == s
        for b in BUD:
            if s > 0: assert set(d[f"budget_{b}"]) == set(H[b]), f"policy set differs at split {s} budget {b}"
            for p, v in d[f"budget_{b}"].items():
                assert len(v["per_seed"]) == 8
                H.setdefault(b, {}).setdefault(p, np.full((len(SPLITS), 8), np.nan))[s] = v["per_seed"]
    return H, meta

def boot_ratio(L, F):
    """L, F: (4 splits, 4 seeds) paired hits. Returns ratio of means, paired-run CI, split SD, split-cluster CI."""
    l, f = L.ravel(), F.ravel(); n = len(l)
    idx = rng.integers(0, n, (B, n)); rb = l[idx].mean(1) / f[idx].mean(1)
    sr = L.mean(1) / F.mean(1); ci = rng.integers(0, L.shape[0], (B, L.shape[0])); cb = L[ci].mean((1, 2)) / F[ci].mean((1, 2))
    return {"ratio": float(l.mean() / f.mean()), "ci_lo": float(np.percentile(rb, 2.5)), "ci_hi": float(np.percentile(rb, 97.5)),
            "split_ratios": [float(x) for x in sr], "split_sd": float(sr.std(ddof=1)), "cluster_ci_lo": float(np.percentile(cb, 2.5)), "cluster_ci_hi": float(np.percentile(cb, 97.5)),
            "mean_learned": float(l.mean()), "mean_fixed": float(f.mean()), "n_paired_runs": int(n)}

nkey = lambda p: int(p.rsplit("_", 1)[1])
def gkey(p): m = re.search(r"n(\d+)_R(\d+)$", p); return (int(m.group(1)), int(m.group(2)))
def families(h):
    fam = {"thr_agreement": ["thr_agreement"], "recal": sorted([p for p in h if p.startswith("recal_charged_")], key=nkey),
           "seeded": sorted([p for p in h if p.startswith("seeded_charged_")], key=nkey), "split": sorted([p for p in h if p.startswith("split_charged_")], key=nkey),
           "greedy_random": ["greedy_online"] + sorted([p for p in h if re.fullmatch(r"greedy_n\d+_R\d+", p)], key=gkey),
           "greedy_seeded": sorted([p for p in h if p.startswith("greedy_seeded_")], key=gkey)}
    return fam

result = {"protocol": __doc__.split("\n")[2:9], "label_sets": {}}
for lab, name in SETS.items():
    H, meta = load(name); R = {"n_sim_per_split": [m[0] for m in meta], "errors_in_sim_per_split": [m[1] for m in meta], "n_buckets_per_split": [m[2] for m in meta], "budgets": {}}
    for b in BUD:
        h = H[b]; fam = families(h); fixed_c = fam["thr_agreement"] + fam["recal"] + fam["seeded"] + fam["split"]; learn_c = fam["greedy_random"] + fam["greedy_seeded"]
        sel = lambda ps: {p: float(h[p][:, SEL].mean()) for p in ps}; ev = lambda ps: {p: float(h[p][:, EV].mean()) for p in ps}
        sel_fixed, sel_learn = sel(fixed_c), sel(learn_c); best_fixed = max(sel_fixed, key=sel_fixed.get); best_learned = max(sel_learn, key=sel_learn.get)
        best_recal = max(fam["recal"], key=sel_fixed.get) if fam["recal"] else None
        F = h[best_fixed][:, EV]
        cell = {"best_fixed_selected_on_seeds_0_3": best_fixed, "best_recal_selected_on_seeds_0_3": best_recal, "learned_selected_on_seeds_0_3": best_learned,
                "eval_mean_all_policies": ev(list(h)), "eval_sd_across_16_runs": {p: float(h[p][:, EV].std(ddof=1)) for p in h},
                "sel_mean_fixed_candidates": sel_fixed, "sel_mean_learned_candidates": sel_learn,
                "selected_learned_vs_selected_fixed": boot_ratio(h[best_learned][:, EV], F),
                "greedy_online_vs_selected_fixed": boot_ratio(h["greedy_online"][:, EV], F),
                "selected_learned_vs_thr_agreement": boot_ratio(h[best_learned][:, EV], h["thr_agreement"][:, EV])}
        # 2x2 decomposition: first-phase design (random vs agreement-seeded) x fitting schedule (one-shot vs repeated refit); each family's member is chosen on seeds 0..3
        d2 = {}
        for nm, ps in (("random_oneshot (recal_charged)", fam["recal"]), ("seeded_oneshot (seeded_charged)", fam["seeded"]), ("random_repeated (greedy_online + greedy_n*_R*)", fam["greedy_random"]),
                       ("seeded_repeated (greedy_seeded_n*_R*)", fam["greedy_seeded"]), ("split_oneshot (split_charged, range-restriction guard)", fam["split"]), ("thr_agreement (no labels)", fam["thr_agreement"])):
            sm = sel(ps); pick = max(sm, key=sm.get); d2[nm] = {"selected_member": pick, "eval_mean_errors_caught": float(h[pick][:, EV].mean()), "vs_selected_best_fixed": boot_ratio(h[pick][:, EV], F)}
        cell["decomposition_2x2"] = d2
        # sweeps (held-out eval means): n sweep for the three charged families, (n0, R) sweep for the two greedy families
        cell["n_sweep_eval_mean"] = {f: {str(nkey(p)): float(h[p][:, EV].mean()) for p in fam[k]} for f, k in (("recal_charged", "recal"), ("seeded_charged", "seeded"), ("split_charged", "split"))}
        cell["n0_R_sweep_eval_mean"] = {f: {f"n{gkey(p)[0]}_R{gkey(p)[1]}": float(h[p][:, EV].mean()) for p in fam[f] if p != "greedy_online"} for f in ("greedy_random", "greedy_seeded")}
        cell["n0_R_sweep_eval_mean"]["greedy_random"]["reference_greedy_online"] = float(h["greedy_online"][:, EV].mean())
        # sensitivity, biased against the learned arm: fixed rule and learned arm both picked as the max over the held-out seeds
        fe, le = ev(fixed_c), ev(learn_c); bfe, ble = max(fe, key=fe.get), max(le, key=le.get)
        cell["sensitivity_max_on_heldout"] = {"fixed": bfe, "learned": ble, **boot_ratio(h[ble][:, EV], h[bfe][:, EV])}
        R["budgets"][b] = cell
    result["label_sets"][lab] = R

# ---- verdict, coded as pre-registered
gate_path = os.path.join(rd, "gate", "gate_check.json"); gate = json.load(open(gate_path)) if os.path.exists(gate_path) else None
gate_ok = bool(gate and gate.get("passed"))
cells = [(lab, b) for lab in result["label_sets"] for b in VBUD]
get = lambda lab, b, k="selected_learned_vs_selected_fixed": result["label_sets"][lab]["budgets"][b][k]
pass_c = {f"{lab} {float(b):.0%}": bool(get(lab, b)["ratio"] >= PASS_R and get(lab, b)["cluster_ci_lo"] > 1.0) for lab, b in cells}
kill_c = {f"{lab} {float(b):.0%}": bool(get(lab, b)["ratio"] <= KILL_R) for lab, b in cells}
kill_sens = {f"{lab} {float(b):.0%}": bool(get(lab, b, "sensitivity_max_on_heldout")["ratio"] <= KILL_R) for lab, b in cells}
pass_ref = {f"{lab} {float(b):.0%}": bool(get(lab, b, "greedy_online_vs_selected_fixed")["ratio"] >= PASS_R and get(lab, b, "greedy_online_vs_selected_fixed")["cluster_ci_lo"] > 1.0) for lab, b in cells}
if not gate_ok: outcome = "VOID"
elif all(pass_c.values()): outcome = "PASS"
elif all(kill_c.values()): outcome = "KILL"
else: outcome = "INCONCLUSIVE"
result["verdict"] = {"outcome": outcome, "regression_gate_passed": gate_ok, "pass_cells": pass_c, "kill_cells": kill_c, "n_pass_cells_of_4": int(sum(pass_c.values())), "n_kill_cells_of_4": int(sum(kill_c.values())),
                     "n_kill_cells_of_4_under_heldout_max_sensitivity": int(sum(kill_sens.values())), "reference_greedy_online_pass_cells": pass_ref}
if gate is not None: result["regression_gate_split0_vs_L1b"] = {k: v for k, v in gate.items()}
result["headline"] = {lab: {b: {"ratio_selected_learned_over_selected_fixed": round(c["selected_learned_vs_selected_fixed"]["ratio"], 3),
                               "ci95_paired_runs": [round(c["selected_learned_vs_selected_fixed"]["ci_lo"], 3), round(c["selected_learned_vs_selected_fixed"]["ci_hi"], 3)],
                               "ci95_split_cluster": [round(c["selected_learned_vs_selected_fixed"]["cluster_ci_lo"], 3), round(c["selected_learned_vs_selected_fixed"]["cluster_ci_hi"], 3)],
                               "sd_of_4_split_ratios": round(c["selected_learned_vs_selected_fixed"]["split_sd"], 3), "selected_learned": c["learned_selected_on_seeds_0_3"], "selected_fixed": c["best_fixed_selected_on_seeds_0_3"],
                               "mean_learned": round(c["selected_learned_vs_selected_fixed"]["mean_learned"], 1), "mean_fixed": round(c["selected_learned_vs_selected_fixed"]["mean_fixed"], 1),
                               "ratio_greedy_online_over_selected_fixed": round(c["greedy_online_vs_selected_fixed"]["ratio"], 3), "n_paired_runs": c["selected_learned_vs_selected_fixed"]["n_paired_runs"]}
                          for b, c in R["budgets"].items()} for lab, R in result["label_sets"].items()}
json.dump(result, open(os.path.join(rd, "result.json"), "w"), indent=1)

# ---- markdown output
f1 = lambda x: f"{x:.1f}"
def table(rows, cols, title):
    print(f"\n{title}"); print("| | " + " | ".join(cols) + " |"); print("|---|" + "---|" * len(cols))
    for r, d in rows.items(): print(f"| {r} | " + " | ".join(f1(d[c]) if c in d else "-" for c in cols) + " |")
def line(nm, r, extra=""):
    print(f"- {nm}: {r['ratio']:.3f} (95% CI {r['ci_lo']:.3f} to {r['ci_hi']:.3f}; split-cluster CI {r['cluster_ci_lo']:.3f} to {r['cluster_ci_hi']:.3f}; SD of 4 split ratios {r['split_sd']:.3f}; per-split {[round(x, 3) for x in r['split_ratios']]}; means {r['mean_learned']:.1f} vs {r['mean_fixed']:.1f}){extra}")
print(f"regression gate (split 0 vs runs/L1b): {gate if gate is None else {k: v for k, v in gate.items() if k != 'diffs'}}")
for lab, R in result["label_sets"].items():
    print(f"\n## {lab} | n_sim per split {R['n_sim_per_split']} | errors per split {R['errors_in_sim_per_split']}")
    for b, c in R["budgets"].items():
        e = c["eval_mean_all_policies"]
        print(f"\n### budget {float(b):.0%} (held-out policy seeds 4..7 x 4 splits = 16 runs; mean errors caught)")
        print("| uniform | oracle | static free-label (upper bound) | thr_agreement | best recal | best seeded | best split | greedy_online | selected fixed | selected learned |"); print("|" + "---|" * 10)
        d2 = c["decomposition_2x2"]; bests = [d2[k]["selected_member"] for k in d2 if k.startswith(("random_oneshot", "seeded_oneshot", "split_oneshot"))]
        print("| " + " | ".join(f1(e[p]) for p in ["uniform", "oracle", "static", "thr_agreement"]) + " | " + " | ".join(f"{p} {f1(e[p])}" for p in bests) + f" | {f1(e['greedy_online'])} | {c['best_fixed_selected_on_seeds_0_3']} {f1(e[c['best_fixed_selected_on_seeds_0_3']])} | {c['learned_selected_on_seeds_0_3']} {f1(e[c['learned_selected_on_seeds_0_3']])} |")
        print(f"selected on seeds 0-3: fixed = {c['best_fixed_selected_on_seeds_0_3']}, learned = {c['learned_selected_on_seeds_0_3']}")
        line("selected learned / selected best extended fixed (PRIMARY)", c["selected_learned_vs_selected_fixed"])
        line("reference greedy_online / selected best extended fixed", c["greedy_online_vs_selected_fixed"])
        line("selected learned / thr_agreement", c["selected_learned_vs_thr_agreement"])
        line("SENSITIVITY max-on-heldout learned / max-on-heldout fixed", c["sensitivity_max_on_heldout"], f" [fixed {c['sensitivity_max_on_heldout']['fixed']} vs learned {c['sensitivity_max_on_heldout']['learned']}]")
        table(c["n_sweep_eval_mean"], ["25", "50", "100", "200", "400"], "n sweep (charged fixed families, held-out mean errors caught)")
        r0 = c["n0_R_sweep_eval_mean"]; cols = sorted({k for d in r0.values() for k in d}, key=lambda k: (k.startswith("ref"), gkey(k) if k[0] == "n" else (0, 0)))
        table(r0, cols, "(n0, R) sweep (greedy families, held-out mean errors caught)")
        print("\n2x2 decomposition (member chosen on seeds 0-3 within each family; held-out mean errors caught; ratio to the selected best fixed rule):")
        print("| family | selected member | mean errors caught | ratio to selected best fixed | split-cluster CI |"); print("|---|---|---|---|---|")
        for nm, d in c["decomposition_2x2"].items(): r = d["vs_selected_best_fixed"]; print(f"| {nm} | {d['selected_member']} | {f1(d['eval_mean_errors_caught'])} | {r['ratio']:.3f} | {r['cluster_ci_lo']:.3f} to {r['cluster_ci_hi']:.3f} |")
print("\n## VERDICT (cells: 3B and 7B-AWQ at 25% and 50%)")
print(json.dumps(result["verdict"], indent=1))
