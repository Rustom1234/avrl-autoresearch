"""L1b summary (CPU). Usage: l1b_summarize.py <runs/L1b>
Reads sim_<name>_split<s>.json (s = 0..3, 8 policy seeds each) and writes result.json plus markdown tables to stdout.

Selection protocol (to avoid selection bias toward the fixed rule): the best charged fixed rule (thr_agreement or recal_charged_<n>) and the better
learned arm (greedy_online or ts_online) are chosen per (label set, budget) on policy seeds 0..3 averaged over the 4 splits. All verdict numbers use
policy seeds 4..7 on the same 4 splits = 16 paired runs. Ratio = mean(learned hits) / mean(fixed hits) over the 16 runs; CI = percentile bootstrap
over the 16 paired runs (10000 resamples); also reported are the SD of the 4 per-split ratios and a cluster bootstrap over the 4 splits.
Verdict rules are the pre-registered ones in the L1b spec (pass: ratio >= 1.05 with CI lower bound > 1 at 25% and 50% on >= 2 of 3 label sets;
kill: ratio <= 1.02 at 25% and 50% on all 3 label sets; otherwise inconclusive)."""
import sys, os, json, glob
import numpy as np

rd = sys.argv[1]
SETS = {"E01 proxy_error": "E01", "3B judge_error": "Qwen2.5-3B-Instruct", "7B-AWQ judge_error": "Qwen2.5-7B-Instruct-AWQ"}
BUD = ["0.1", "0.25", "0.5"]; SPLITS = [0, 1, 2, 3]; SEL = slice(0, 4); EV = slice(4, 8); B = 10000
rng = np.random.default_rng(0)

def load(name):
    H, meta = {}, []
    for s in SPLITS:
        d = json.load(open(os.path.join(rd, f"sim_{name}_split{s}.json"))); meta.append((d["n_sim"], d["hits"], d["n_buckets"]))
        assert d["split_seed"] == s
        for b in BUD:
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

result = {"protocol": __doc__.split("\n")[2:7], "label_sets": {}}
for lab, name in SETS.items():
    H, meta = load(name); R = {"n_sim_per_split": [m[0] for m in meta], "errors_in_sim_per_split": [m[1] for m in meta], "n_buckets_per_split": [m[2] for m in meta], "budgets": {}}
    for b in BUD:
        h = H[b]; fixed_c = ["thr_agreement"] + sorted([p for p in h if p.startswith("recal_charged_")], key=lambda p: int(p.rsplit("_", 1)[1]))
        sel_mean = {p: float(h[p][:, SEL].mean()) for p in fixed_c}; best_fixed = max(sel_mean, key=sel_mean.get)
        recal_c = [p for p in fixed_c if p != "thr_agreement"]; best_recal = max(recal_c, key=sel_mean.get) if recal_c else None
        lsel = {p: float(h[p][:, SEL].mean()) for p in ("greedy_online", "ts_online")}; best_learned = max(lsel, key=lsel.get)
        F = h[best_fixed][:, EV]
        cell = {"best_fixed_selected_on_seeds_0_3": best_fixed, "best_recal_selected_on_seeds_0_3": best_recal, "learned_selected_on_seeds_0_3": best_learned,
                "eval_mean_all_policies": {p: float(h[p][:, EV].mean()) for p in h}, "eval_sd_across_16_runs": {p: float(h[p][:, EV].std(ddof=1)) for p in h},
                "sel_mean_fixed_candidates": sel_mean,
                "selected_learned_vs_selected_fixed": boot_ratio(h[best_learned][:, EV], F),
                "greedy_vs_selected_fixed": boot_ratio(h["greedy_online"][:, EV], F), "ts_vs_selected_fixed": boot_ratio(h["ts_online"][:, EV], F),
                "greedy_vs_thr_agreement": boot_ratio(h["greedy_online"][:, EV], h["thr_agreement"][:, EV]), "ts_vs_thr_agreement": boot_ratio(h["ts_online"][:, EV], h["thr_agreement"][:, EV]),
                "ts_vs_greedy": boot_ratio(h["ts_online"][:, EV], h["greedy_online"][:, EV])}
        # sensitivity, biased against the learned arm: fixed rule and learned arm both picked as the max over the held-out seeds
        fe = {p: float(h[p][:, EV].mean()) for p in fixed_c}; bfe = max(fe, key=fe.get); le = {p: float(h[p][:, EV].mean()) for p in ("greedy_online", "ts_online")}; ble = max(le, key=le.get)
        cell["sensitivity_max_on_heldout"] = {"fixed": bfe, "learned": ble, **boot_ratio(h[ble][:, EV], h[bfe][:, EV])}
        # sensitivity: thr_agreement is the only fixed rule
        R["budgets"][b] = cell
    result["label_sets"][lab] = R

def passes(c, key="selected_learned_vs_selected_fixed"):
    r = c[key]; return r["ratio"] >= 1.05 and r["ci_lo"] > 1.0
pass_labels, pass_labels_greedy, pass_labels_ts, pass_labels_any = [], [], [], []
kill_cells, kill_cells_sens = [], []
for lab, R in result["label_sets"].items():
    c25, c50 = R["budgets"]["0.25"], R["budgets"]["0.5"]
    if passes(c25) and passes(c50): pass_labels.append(lab)
    if passes(c25, "greedy_vs_selected_fixed") and passes(c50, "greedy_vs_selected_fixed"): pass_labels_greedy.append(lab)
    if passes(c25, "ts_vs_selected_fixed") and passes(c50, "ts_vs_selected_fixed"): pass_labels_ts.append(lab)
    if (passes(c25, "greedy_vs_selected_fixed") or passes(c25, "ts_vs_selected_fixed")) and (passes(c50, "greedy_vs_selected_fixed") or passes(c50, "ts_vs_selected_fixed")): pass_labels_any.append(lab)
    for c in (c25, c50):
        kill_cells.append(c["selected_learned_vs_selected_fixed"]["ratio"] <= 1.02); kill_cells_sens.append(c["sensitivity_max_on_heldout"]["ratio"] <= 1.02)
passed = len(pass_labels) >= 2; killed = all(kill_cells)
result["verdict"] = {"PASS": passed, "KILL": killed and not passed, "label_sets_passing_selected_learned_arm": pass_labels, "label_sets_passing_greedy_online": pass_labels_greedy,
                     "label_sets_passing_ts_online": pass_labels_ts, "label_sets_passing_either_arm": pass_labels_any, "kill_cells_true_of_6": int(sum(kill_cells)),
                     "kill_cells_true_of_6_under_heldout_max_sensitivity": int(sum(kill_cells_sens)),
                     "outcome": "PASS" if passed else ("KILL" if killed else "INCONCLUSIVE")}
gc = os.path.join(rd, "gate", "gate_check.json")
if os.path.exists(gc): result["regression_gate_split0_vs_E04"] = json.load(open(gc))
result["headline"] = {lab: {b: {k: round(R["budgets"][b][k]["ratio"], 3) if k != "arm" else R["budgets"][b]["learned_selected_on_seeds_0_3"] for k in ("selected_learned_vs_selected_fixed", "ts_vs_selected_fixed", "arm")}
                            | {"ci95_selected_learned": [round(R["budgets"][b]["selected_learned_vs_selected_fixed"]["ci_lo"], 3), round(R["budgets"][b]["selected_learned_vs_selected_fixed"]["ci_hi"], 3)], "best_fixed": R["budgets"][b]["best_fixed_selected_on_seeds_0_3"],
                               "mean_learned": round(R["budgets"][b]["selected_learned_vs_selected_fixed"]["mean_learned"], 1), "mean_fixed": round(R["budgets"][b]["selected_learned_vs_selected_fixed"]["mean_fixed"], 1)}
                            for b in BUD} for lab, R in result["label_sets"].items()}
json.dump(result, open(os.path.join(rd, "result.json"), "w"), indent=1)

# ---- markdown output
f2 = lambda x: f"{x:.1f}"
for lab, R in result["label_sets"].items():
    print(f"\n## {lab} | n_sim per split {R['n_sim_per_split']} | errors per split {R['errors_in_sim_per_split']}")
    for b, c in R["budgets"].items():
        e = c["eval_mean_all_policies"]
        print(f"\n### budget {float(b):.0%} (held-out policy seeds 4..7 x 4 splits = 16 runs; mean errors caught)")
        print("| uniform | oracle | static free-label (upper bound) | static_matched (half budget charged) | thr_agreement | " + " | ".join(p for p in e if p.startswith("recal_charged_")) + " | greedy_online | ts_online |")
        print("|" + "---|" * (7 + sum(p.startswith("recal_charged_") for p in e)))
        print("| " + " | ".join(f2(e[p]) for p in ["uniform", "oracle", "static", "static_matched", "thr_agreement"] + [p for p in e if p.startswith("recal_charged_")] + ["greedy_online", "ts_online"]) + " |")
        print(f"selected on seeds 0-3: fixed = {c['best_fixed_selected_on_seeds_0_3']}, best recal = {c['best_recal_selected_on_seeds_0_3']}, learned = {c['learned_selected_on_seeds_0_3']}")
        for key, nm in (("selected_learned_vs_selected_fixed", "selected learned / best charged fixed"), ("greedy_vs_selected_fixed", "greedy_online / best charged fixed"), ("ts_vs_selected_fixed", "ts_online / best charged fixed"),
                        ("greedy_vs_thr_agreement", "greedy_online / thr_agreement"), ("ts_vs_thr_agreement", "ts_online / thr_agreement"), ("ts_vs_greedy", "ts_online / greedy_online"), ("sensitivity_max_on_heldout", "SENSITIVITY max-on-heldout learned / max-on-heldout fixed")):
            r = c[key]; print(f"- {nm}: {r['ratio']:.3f} (95% CI {r['ci_lo']:.3f} to {r['ci_hi']:.3f}; split-cluster CI {r['cluster_ci_lo']:.3f} to {r['cluster_ci_hi']:.3f}; SD of 4 split ratios {r['split_sd']:.3f}; per-split {[round(x, 3) for x in r['split_ratios']]}; means {r['mean_learned']:.1f} vs {r['mean_fixed']:.1f})" + (f" [{r['fixed']} vs {r['learned']}]" if key == "sensitivity_max_on_heldout" else ""))
print("\n## VERDICT\n" + json.dumps(result["verdict"], indent=1))
