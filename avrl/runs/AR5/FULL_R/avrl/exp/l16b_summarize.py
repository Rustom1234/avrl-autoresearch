"""L16b summary (CPU). Usage: l16b_summarize.py <runs/L16b>. Writes result.json and summary.md, prints the markdown.

Reads gate/gate_check.json (gate 1, regression vs L15a flip25), gate/gate_check_ident.json (gate 2, warm == recal_free with refits off under every STALE variant, plus early gate 3), gate/gate_check_post.json
(192 files, flip25 == L15a, uniform/static/greedy_online == L13a, pooled gate 3) and sim_<tag>_<stale>_n2250_sp<split>_sub<sub>.json (corrupted label, cell30, cheap_r, K = 1, 8 policy seeds; tag = {3B,7B}_q_corrupted;
STALE in jlflip50 / jlflip50_hi / jlrev100 / fn80 / fp35 / flip25; n_sim 2250 x split 0..3 x subsample 0..3 = 16 clusters per rung and variant). The clean warm arm is warm_online_200 of runs/L13a (same n2250 files, paired).
  free rule  = recal_free_200: 200 free fit-half labels (stale in the variant), logistic fitted once, whole budget B spent on the top-ranked sim rows
  warm arm   = warm_online_200: greedy online learner warm-started with that same logistic, refit every max(B // 8, 40) labels on the 200 free (stale) labels pooled with the seen sim labels, unweighted
  w25 arm    = warm_online_200_w25: same, but the stale free labels carry sample_weight 0.25 in every refit (stream labels weight 1)
  w0 arm     = warm_online_200_w0: same, but the stale free labels carry sample_weight 0 in every refit (weight 1 in the initial fit and in the picks before the first refit)
  cold       = greedy_online (no free labels at all)
Variants (STALE corrupts only the 200 free labels, never the features): jlflip50 = PRIMARY, free labels whose judge_logit is strictly below the fit-half median flipped with p 0.5; jlflip50_hi = strictly above the median
(descriptive control); jlrev100 = strictly below the median flipped with p 1.0 (descriptive); fn80 = positives flipped with p 0.8; fp35 = negatives flipped with p 0.35; flip25 = all labels p 0.25 (control, must equal L15a).
Ratios are ratios of means over clusters x 8 seeds (all 8 policy seeds, nothing is selected); CI = percentile bootstrap over the 16 clusters (10000 resamples, one shared resample index). Honest unit: the 4 splits
(the 4 subsamples of a split share one fit set); reported as per-split ratios, the split-level paired t on a minus b at df 3, a t interval on the 4 per-split ratios (mean +- 3.182 se) and the exact bootstrap
over the 4 splits (all 256 resamples, percentile).

Pre-registered verdict (L16b spec), cell = 25% budget, STALE = jlflip50, corrupted label, n_sim 2250 (B about 562), rungs 3B and 7B-AWQ:
  VOID          if gate 1, gate 2, gate 3 or the post gate fails
  PASS          if on BOTH rungs warm_stale_200 / greedy_online >= 1.00 AND warm_stale_200 / recal_free_200 (same stale labels) >= 1.06 with the bootstrap CI lower end above 1
  KILL          if on EITHER rung warm_stale_200 / greedy_online <= 0.97 with the bootstrap CI upper end below 1
  INCONCLUSIVE  otherwise (the fn80, fp35 and jlrev100 rows, which are descriptive, decide the weight)
If KILL: Phase 1 starts cold, or warm with label-age weighting if warm_online_200_w25 / cold or warm_online_200_w0 / cold is >= 0.99 on both rungs."""
import sys, os, json, itertools
import numpy as np
from scipy import stats

rd = sys.argv[1]
RUNG = {"3B": "3B_q", "7B-AWQ": "7B_q"}; STALES = ["jlflip50", "jlflip50_hi", "jlrev100", "fn80", "fp35", "flip25"]; BUD = ["0.1", "0.25", "0.5"]; NB = 10000; T3 = float(stats.t.ppf(0.975, 3))
jl = lambda p: json.load(open(p))
_idx = {}
SPL = np.array(list(itertools.product(range(4), repeat=4)))  # 256 exact resamples of the 4 splits


def boot_idx(C):
    if C not in _idx: _idx[C] = np.random.default_rng(0).integers(0, C, (NB, C))
    return _idx[C]


def ci(x): return float(np.percentile(x, 2.5)), float(np.percentile(x, 97.5))


def cmp(a, b):
    """a, b: (16 clusters split-major, 8 seeds) paired arrays. Ratio of means a/b, cluster-bootstrap CI, split-level honest-unit statistics."""
    C = a.shape[0]; idx = boot_idx(C); rb = a[idx].mean((1, 2)) / b[idx].mean((1, 2)); lo, hi = ci(rb)
    sa, sb = a.reshape(4, C // 4, -1).mean((1, 2)), b.reshape(4, C // 4, -1).mean((1, 2)); spr = sa / sb; d = sa - sb; s = d.std(ddof=1)
    t = float(d.mean() / (s / 2)) if s > 0 else None; se = spr.std(ddof=1) / 2; eb = ci(sa[SPL].mean(1) / sb[SPL].mean(1))
    return {"ratio": float(a.mean() / b.mean()), "ci_lo": lo, "ci_hi": hi, "mean_a": float(a.mean()), "mean_b": float(b.mean()), "diff_hits": float(a.mean() - b.mean()), "n_runs": int(a.size), "n_clusters": int(C),
            "per_split_ratio": [float(v) for v in spr], "n_splits_ratio_gt_1": int((spr > 1.0).sum()), "frac_clusters_ratio_gt_1": float((a.mean(1) / b.mean(1) > 1).mean()),
            "split_t_df3": t, "split_p_df3": float(2 * stats.t.sf(abs(t), 3)) if t is not None else None, "split_mean_ratio": float(spr.mean()),
            "split_t_ci_lo": float(spr.mean() - T3 * se), "split_t_ci_hi": float(spr.mean() + T3 * se), "split_boot_lo": eb[0], "split_boot_hi": eb[1]}


def load(tag, st):
    ds = [jl(os.path.join(rd, f"sim_{tag}_{st}_n2250_sp{sp}_sub{sub}.json")) for sp in range(4) for sub in range(4)]
    H = {b: {p: np.array([d[f"budget_{b}"][p]["per_seed"] for d in ds], float) for p in ds[0][f"budget_{b}"]} for b in BUD}
    return ds, H


def load_clean(tag):
    ds = [jl(f"runs/L13a/sim_{tag}_n2250_sp{sp}_sub{sub}.json") for sp in range(4) for sub in range(4)]
    return {b: np.array([d[f"budget_{b}"]["warm_online_200"]["per_seed"] for d in ds], float) for b in BUD}


res = {"protocol": __doc__.split("\n")[1:], "cells": {}, "changed_labels": {}}
opt = lambda p: jl(os.path.join(rd, p)) if os.path.exists(os.path.join(rd, p)) else None
g1, gi, g2 = opt("gate/gate_check.json"), opt("gate/gate_check_ident.json"), opt("gate/gate_check_post.json")
res["gate1_regression_vs_L15a_flip25"] = g1; res["gate2_identity_warm_eq_free_norefit"] = {k: v for k, v in gi.items() if k != "gate3"} if gi else None
res["gate_post"] = {k: v for k, v in g2.items() if k != "gate3"} if g2 else None
for rg, pre in RUNG.items():
    tag = f"{pre}_corrupted"; clean = load_clean(tag)
    for st in STALES:
        ds, H = load(tag, st); key = f"{rg}|{st}"
        ss = [v for d in ds for v in d["stale_stats"].values()]; n = sum(v["n"] for v in ss); ch = sum(v["n_changed"] for v in ss); el = sum(v["n_eligible"] for v in ss)
        res["changed_labels"][key] = {"n_labels": n, "n_changed": ch, "frac_changed": ch / n, "frac_eligible": el / n, "frac_changed_of_eligible": ch / max(el, 1), "frac_pos_before": sum(v["n_pos_before"] for v in ss) / n,
                                      "frac_pos_after": sum(v["n_pos_after"] for v in ss) / n, "changed_not_eligible": sum(v["n_changed_not_eligible"] for v in ss),
                                      "changed_below_median": sum(v["n_changed_below_median"] for v in ss), "changed_above_median": sum(v["n_changed_above_median"] for v in ss), "changed_at_median": sum(v["n_changed_at_median"] for v in ss)}
        c = {"clusters": len(ds), "n_sim": float(np.mean([d["n_sim"] for d in ds])), "hits": float(np.mean([d["hits"] for d in ds])), "oneclass_free_draw_files": int(sum(bool(d["warm_oneclass_draws"]) for d in ds)), "budgets": {}}
        for b in BUD:
            h = H[b]; w, f, w25, w0, g, u, s = h["warm_online_200"], h["recal_free_200"], h["warm_online_200_w25"], h["warm_online_200_w0"], h["greedy_online"], h["uniform"], h["static"]
            c["budgets"][b] = {"B": float(np.mean([int(float(b) * d["n_sim"]) for d in ds])),
                               "warm_over_cold": cmp(w, g), "warm_over_free_stale": cmp(w, f), "warm_over_warm_clean": cmp(w, clean[b]), "w25_over_cold": cmp(w25, g), "w0_over_cold": cmp(w0, g),
                               "w25_over_warm": cmp(w25, w), "w0_over_warm": cmp(w0, w), "w25_over_free_stale": cmp(w25, f), "w0_over_free_stale": cmp(w0, f), "free_stale_over_cold": cmp(f, g), "free_stale_over_static": cmp(f, s),
                               "abs_mean_errors_caught_all8": {"uniform": float(u.mean()), "static": float(s.mean()), "greedy_online": float(g.mean()), "recal_free_200": float(f.mean()),
                                                               "warm_online_200": float(w.mean()), "warm_online_200_w25": float(w25.mean()), "warm_online_200_w0": float(w0.mean()), "warm_online_200_clean_L13a": float(clean[b].mean())}}
        res["cells"][key] = c

# ---- verdict (pre-registered primary: jlflip50, 25%, n_sim 2250)
gate_ok = bool(g1 and g1.get("passed") and gi and gi.get("passed") and g2 and g2.get("passed")); V = {}
for rg in RUNG:
    cell = res["cells"][f"{rg}|jlflip50"]["budgets"]["0.25"]; G, F, Wc, X5, X0 = cell["warm_over_cold"], cell["warm_over_free_stale"], cell["warm_over_warm_clean"], cell["w25_over_cold"], cell["w0_over_cold"]
    V[rg] = {"warm_stale_over_cold": G["ratio"], "warm_stale_over_cold_ci": [G["ci_lo"], G["ci_hi"]], "warm_stale_over_cold_per_split": G["per_split_ratio"], "warm_stale_over_cold_splits_gt_1": G["n_splits_ratio_gt_1"],
             "warm_stale_over_cold_split_t_ci": [G["split_t_ci_lo"], G["split_t_ci_hi"]], "warm_stale_over_cold_split_boot": [G["split_boot_lo"], G["split_boot_hi"]],
             "warm_stale_over_cold_split_t_df3": G["split_t_df3"], "warm_stale_over_cold_split_p_df3": G["split_p_df3"],
             "warm_stale_over_free_stale": F["ratio"], "warm_stale_over_free_stale_ci": [F["ci_lo"], F["ci_hi"]], "warm_stale_over_free_stale_per_split": F["per_split_ratio"], "warm_stale_over_free_stale_splits_gt_1": F["n_splits_ratio_gt_1"],
             "warm_stale_over_free_stale_split_t_ci": [F["split_t_ci_lo"], F["split_t_ci_hi"]], "warm_stale_over_free_stale_split_boot": [F["split_boot_lo"], F["split_boot_hi"]],
             "warm_stale_over_free_stale_split_t_df3": F["split_t_df3"], "warm_stale_over_free_stale_split_p_df3": F["split_p_df3"],
             "warm_stale_over_warm_clean": Wc["ratio"], "warm_stale_over_warm_clean_ci": [Wc["ci_lo"], Wc["ci_hi"]],
             "w25_over_cold": X5["ratio"], "w25_over_cold_ci": [X5["ci_lo"], X5["ci_hi"]], "w25_over_cold_split_t_ci": [X5["split_t_ci_lo"], X5["split_t_ci_hi"]], "w25_over_cold_splits_gt_1": X5["n_splits_ratio_gt_1"],
             "w0_over_cold": X0["ratio"], "w0_over_cold_ci": [X0["ci_lo"], X0["ci_hi"]], "w0_over_cold_split_t_ci": [X0["split_t_ci_lo"], X0["split_t_ci_hi"]], "w0_over_cold_splits_gt_1": X0["n_splits_ratio_gt_1"],
             "mean_warm_stale": G["mean_a"], "mean_cold": G["mean_b"], "mean_free_stale": F["mean_b"], "mean_warm_clean": Wc["mean_b"], "mean_w25": X5["mean_a"], "mean_w0": X0["mean_a"],
             "n_clusters": G["n_clusters"], "n_runs": G["n_runs"], "B": cell["B"],
             "pass": bool(G["ratio"] >= 1.00 and F["ratio"] >= 1.06 and F["ci_lo"] > 1.0), "kill": bool(G["ratio"] <= 0.97 and G["ci_hi"] < 1.0),
             "w25_restores_099_of_cold": bool(X5["ratio"] >= 0.99), "w0_restores_099_of_cold": bool(X0["ratio"] >= 0.99)}
outcome = "VOID" if not gate_ok else ("PASS" if all(v["pass"] for v in V.values()) else ("KILL" if any(v["kill"] for v in V.values()) else "INCONCLUSIVE"))
READING = {"PASS": "the Phase 1 warm-start default holds beyond rank-preserving noise, for staleness that scrambles labels in the ranked region.",
           "KILL": "warm start by default is withdrawn for ranking-changing staleness and claim 2's surviving form is restricted to benign symmetric noise.",
           "INCONCLUSIVE": "anything else is inconclusive and the fn80, fp35 and jlrev100 rows decide the weight.",
           "VOID": "a gate failed; no reading is drawn."}
w25_ok, w0_ok = all(v["w25_restores_099_of_cold"] for v in V.values()), all(v["w0_restores_099_of_cold"] for v in V.values())
kill_detail = (("Phase 1 default = warm start with label-age weighting (w25 / cold >= 0.99 on both rungs: %s; w0 / cold >= 0.99 on both rungs: %s)" % (w25_ok, w0_ok)) if (w25_ok or w0_ok) else "neither w25 nor w0 restores 0.99 of cold on both rungs: Phase 1 default = cold start") if outcome == "KILL" else None
res["verdict"] = {"outcome": outcome, "reading": READING[outcome], "kill_followup": kill_detail, "gate_ok": gate_ok, "rungs": V,
                  "n_sim_actual": {rg: res["cells"][f"{rg}|jlflip50"]["n_sim"] for rg in RUNG}, "B_25pct": {rg: V[rg]["B"] for rg in RUNG}}
res["metric"] = {"cell": "25% budget, n_sim 2250, corrupted label, jlflip50 primary (below-median judge_logit free labels flipped with p 0.5), n = 200, 8 policy seeds, 16 clusters", "rungs": V, "outcome": outcome}
json.dump(res, open(os.path.join(rd, "result.json"), "w"), indent=1)

# ---------------- markdown
f2 = lambda x: "n/a" if x is None else f"{x:.2f}"; f3 = lambda x: "n/a" if x is None else f"{x:.3f}"
ct = lambda d: f"{f3(d['ratio'])} [{f3(d['ci_lo'])}, {f3(d['ci_hi'])}]"
L = []; P_ = L.append
P_(f"## Verdict: {outcome}")
P_(f"- reading: {READING[outcome]}" + (f" Follow-up: {kill_detail}." if kill_detail else ""))
P_(f"- gate 1 (flip25 gate files vs runs/L15a, n2250 split 0 sub 0, both rungs, six arms): {g1}")
P_(f"- gate 2 (identity, refits off, warm_online_200 and its w25 and w0 twins == recal_free_200 per seed under all six STALE variants, both rungs) and early gate 3: {res['gate2_identity_warm_eq_free_norefit']}")
P_(f"- post gate (192 files; flip25 == L15a on six arms; uniform/static/greedy_online == L13a in every variant; pooled gate 3): {res['gate_post']}")
for rg, v in V.items():
    P_(f"- {rg} (jlflip50, 25%, n_sim 2250, B {v['B']:.0f}, {v['n_clusters']} clusters x 8 seeds = {v['n_runs']} runs): warm_stale_200 {v['mean_warm_stale']:.1f}, cold greedy {v['mean_cold']:.1f}, recal_free_200 (stale) {v['mean_free_stale']:.1f}, warm clean (L13a) {v['mean_warm_clean']:.1f}, w25 {v['mean_w25']:.1f}, w0 {v['mean_w0']:.1f}")
    P_(f"  - warm_stale / cold {f3(v['warm_stale_over_cold'])} [{f3(v['warm_stale_over_cold_ci'][0])}, {f3(v['warm_stale_over_cold_ci'][1])}] (split t interval [{f3(v['warm_stale_over_cold_split_t_ci'][0])}, {f3(v['warm_stale_over_cold_split_t_ci'][1])}], exact split bootstrap [{f3(v['warm_stale_over_cold_split_boot'][0])}, {f3(v['warm_stale_over_cold_split_boot'][1])}], per-split {[round(x, 3) for x in v['warm_stale_over_cold_per_split']]}, {v['warm_stale_over_cold_splits_gt_1']} of 4 splits above 1.00, split t(df 3) {f2(v['warm_stale_over_cold_split_t_df3'])} p {f3(v['warm_stale_over_cold_split_p_df3'])})")
    P_(f"  - warm_stale / recal_free_200 (stale) {f3(v['warm_stale_over_free_stale'])} [{f3(v['warm_stale_over_free_stale_ci'][0])}, {f3(v['warm_stale_over_free_stale_ci'][1])}] (split t interval [{f3(v['warm_stale_over_free_stale_split_t_ci'][0])}, {f3(v['warm_stale_over_free_stale_split_t_ci'][1])}], exact split bootstrap [{f3(v['warm_stale_over_free_stale_split_boot'][0])}, {f3(v['warm_stale_over_free_stale_split_boot'][1])}], per-split {[round(x, 3) for x in v['warm_stale_over_free_stale_per_split']]}, {v['warm_stale_over_free_stale_splits_gt_1']} of 4 splits above 1.00, split t(df 3) {f2(v['warm_stale_over_free_stale_split_t_df3'])} p {f3(v['warm_stale_over_free_stale_split_p_df3'])})")
    P_(f"  - warm_stale / warm clean (L13a) {f3(v['warm_stale_over_warm_clean'])} [{f3(v['warm_stale_over_warm_clean_ci'][0])}, {f3(v['warm_stale_over_warm_clean_ci'][1])}]; w25 / cold {f3(v['w25_over_cold'])} [{f3(v['w25_over_cold_ci'][0])}, {f3(v['w25_over_cold_ci'][1])}] ({v['w25_over_cold_splits_gt_1']} of 4 splits above 1.00); w0 / cold {f3(v['w0_over_cold'])} [{f3(v['w0_over_cold_ci'][0])}, {f3(v['w0_over_cold_ci'][1])}] ({v['w0_over_cold_splits_gt_1']} of 4 splits above 1.00); pass {v['pass']}, kill {v['kill']}, w25 restores 0.99 of cold {v['w25_restores_099_of_cold']}, w0 restores 0.99 of cold {v['w0_restores_099_of_cold']}")
P_("\n## Realised changed free labels (pooled over 16 files x 8 policy seeds x 200 labels per rung and variant)")
P_("| rung | stale | labels | changed | fraction changed | fraction eligible | changed / eligible | positives before | positives after | changed below / above / at median | changed on non-eligible rows |"); P_("|" + "---|" * 11)
for key, r in res["changed_labels"].items():
    rg, st = key.split("|")
    P_(f"| {rg} | {st} | {r['n_labels']} | {r['n_changed']} | {r['frac_changed']:.4f} | {r['frac_eligible']:.4f} | {r['frac_changed_of_eligible']:.4f} | {r['frac_pos_before']:.4f} | {r['frac_pos_after']:.4f} | {r['changed_below_median']} / {r['changed_above_median']} / {r['changed_at_median']} | {r['changed_not_eligible']} |")
P_("\n## Main table: ratios of means [16-cluster percentile bootstrap CI] (all 8 policy seeds). Primary rows: jlflip50 at 25%; the other variants are descriptive; flip25 is the L15a-reproducing control")
P_("| rung | stale | budget | B | warm_stale | cold | free_stale | warm clean | w25 | w0 | warm/cold [CI] | warm/free_stale [CI] | warm/warm_clean [CI] | w25/cold [CI] | w0/cold [CI] | w25/warm [CI] | w0/warm [CI] | free_stale/cold |"); P_("|" + "---|" * 18)
for key, c in res["cells"].items():
    rg, st = key.split("|")
    for b, cell in c["budgets"].items():
        a = cell["abs_mean_errors_caught_all8"]
        P_(f"| {rg} | {st} | {float(b):.0%} | {cell['B']:.0f} | {a['warm_online_200']:.1f} | {a['greedy_online']:.1f} | {a['recal_free_200']:.1f} | {a['warm_online_200_clean_L13a']:.1f} | {a['warm_online_200_w25']:.1f} | {a['warm_online_200_w0']:.1f} | {ct(cell['warm_over_cold'])} | {ct(cell['warm_over_free_stale'])} | {ct(cell['warm_over_warm_clean'])} | {ct(cell['w25_over_cold'])} | {ct(cell['w0_over_cold'])} | {ct(cell['w25_over_warm'])} | {ct(cell['w0_over_warm'])} | {f3(cell['free_stale_over_cold']['ratio'])} |")
P_("\n## Honest unit: 4 splits. Per-split ratios, split-level paired t (df 3) on a minus b, t interval of the 4 split ratios (mean +- 3.182 se), exact 256-resample split bootstrap")
P_("| rung | stale | budget | metric | ratio | per-split ratios | splits > 1 | split t (p) | split t interval | exact split bootstrap | clusters > 1 |"); P_("|" + "---|" * 11)
for key, c in res["cells"].items():
    rg, st = key.split("|")
    for b, cell in c["budgets"].items():
        for mk, mn in (("warm_over_cold", "warm_stale / cold"), ("warm_over_free_stale", "warm_stale / free_stale"), ("warm_over_warm_clean", "warm_stale / warm_clean"), ("w25_over_cold", "w25 / cold"), ("w0_over_cold", "w0 / cold")):
            d = cell[mk]
            P_(f"| {rg} | {st} | {float(b):.0%} | {mn} | {f3(d['ratio'])} | {', '.join(f'{x:.3f}' for x in d['per_split_ratio'])} | {d['n_splits_ratio_gt_1']} | {f2(d['split_t_df3'])} ({f3(d['split_p_df3'])}) | [{f3(d['split_t_ci_lo'])}, {f3(d['split_t_ci_hi'])}] | [{f3(d['split_boot_lo'])}, {f3(d['split_boot_hi'])}] | {d['frac_clusters_ratio_gt_1']:.2f} |")
P_("\n## Absolute mean errors caught (all 8 seeds, mean over 16 clusters)")
P_("| rung | stale | budget | n_sim | errors in sim | B | uniform | static | cold greedy | free_stale | warm_stale | w25 | w0 | warm clean (L13a) |"); P_("|" + "---|" * 14)
for key, c in res["cells"].items():
    rg, st = key.split("|")
    for b, cell in c["budgets"].items():
        a = cell["abs_mean_errors_caught_all8"]
        P_(f"| {rg} | {st} | {float(b):.0%} | {c['n_sim']:.0f} | {c['hits']:.0f} | {cell['B']:.0f} | {a['uniform']:.1f} | {a['static']:.1f} | {a['greedy_online']:.1f} | {a['recal_free_200']:.1f} | {a['warm_online_200']:.1f} | {a['warm_online_200_w25']:.1f} | {a['warm_online_200_w0']:.1f} | {a['warm_online_200_clean_L13a']:.1f} |")
P_(f"\nOne-class free draws (files with any): " + ", ".join(f"{k} {c['oneclass_free_draw_files']}" for k, c in res["cells"].items()))
open(os.path.join(rd, "summary.md"), "w").write("\n".join(L) + "\n"); print("\n".join(L))
