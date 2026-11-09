"""L15a summary (CPU). Usage: l15a_summarize.py <runs/L15a>. Writes result.json and summary.md, prints the markdown.

Reads gate/gate_check.json (gate 1, regression vs L13a), gate/gate_check_ident.json (gate 2, warm == recal_free with refits off under every STALE variant), gate/gate_check_post.json and
sim_<tag>_<stale>_n2250_sp<split>_sub<sub>.json (corrupted label, cell30, cheap_r, K = 1, 8 policy seeds; tag = {3B,7B}_q_corrupted; STALE in none / flip25 / flip50 / lowlen; n_sim 2250 x split 0..3 x
subsample 0..3 = 16 clusters per rung and variant). The clean warm arm is warm_online_200 of runs/L13a (same n2250 files, same fit draws, so it is paired per cluster and policy seed).
  free rule  = recal_free_200: 200 free fit-half labels (stale in the variant), logistic fitted once, whole budget B spent on the top-ranked sim rows
  warm arm   = warm_online_200: greedy online learner warm-started with that same logistic, refit every max(B // 8, 40) labels on the 200 free (stale) labels pooled with the seen sim labels, unweighted
  w25 arm    = warm_online_200_w25: same, but the stale free labels carry sample_weight 0.25 in every refit (stream labels weight 1)
  cold       = greedy_online (no free labels at all)
Ratios are ratios of means over clusters x 8 seeds (all 8 policy seeds, nothing is selected); CI = percentile bootstrap over the 16 clusters (10000 resamples, one shared resample index). Honest unit: the 4 splits
(the 4 subsamples of a split share one fit set); reported as per-split ratios, the split-level paired t on a minus b at df 3, a t interval on the 4 per-split ratios (mean +- 3.182 se) and the exact bootstrap
over the 4 splits (all 256 resamples, percentile).

Pre-registered verdict (L15a spec), cell = 25% budget, STALE = flip25, corrupted label, n_sim 2250 (B about 562), rungs 3B and 7B-AWQ:
  VOID          if gate 1, gate 2 or the post gate fails
  PASS          if on BOTH rungs warm_stale_200 / greedy_online >= 1.00 AND warm_stale_200 / recal_free_200 (same stale labels) >= 1.06 with the bootstrap CI lower end above 1
  KILL          if on EITHER rung warm_stale_200 / greedy_online <= 0.97 with the bootstrap CI upper end below 1
  INCONCLUSIVE  otherwise (weight shifts to the flip50 and lowlen rows, which are descriptive)
If KILL: the Phase 1 default becomes cold start, or warm start with label-age weighting if warm_online_200_w25 / greedy_online >= 0.99 (reported on both rungs)."""
import sys, os, json, itertools
import numpy as np
from scipy import stats

rd = sys.argv[1]
RUNG = {"3B": "3B_q", "7B-AWQ": "7B_q"}; STALES = ["flip25", "flip50", "lowlen", "none"]; BUD = ["0.1", "0.25", "0.5"]; NB = 10000; T3 = float(stats.t.ppf(0.975, 3))
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


res = {"protocol": __doc__.split("\n")[1:], "cells": {}}
opt = lambda p: jl(os.path.join(rd, p)) if os.path.exists(os.path.join(rd, p)) else None
g1, gi, g2 = opt("gate/gate_check.json"), opt("gate/gate_check_ident.json"), opt("gate/gate_check_post.json")
res["gate1_regression_vs_L13a"] = g1; res["gate2_identity_warm_eq_free_norefit"] = gi; res["gate_post"] = g2
for rg, pre in RUNG.items():
    tag = f"{pre}_corrupted"; clean = load_clean(tag)
    for st in STALES:
        ds, H = load(tag, st); key = f"{rg}|{st}"
        c = {"clusters": len(ds), "n_sim": float(np.mean([d["n_sim"] for d in ds])), "hits": float(np.mean([d["hits"] for d in ds])), "oneclass_free_draw_files": int(sum(bool(d["warm_oneclass_draws"]) for d in ds)), "budgets": {}}
        for b in BUD:
            h = H[b]; w, f, w25, g, u, s = h["warm_online_200"], h["recal_free_200"], h["warm_online_200_w25"], h["greedy_online"], h["uniform"], h["static"]
            c["budgets"][b] = {"B": float(np.mean([int(float(b) * d["n_sim"]) for d in ds])),
                               "warm_over_cold": cmp(w, g), "warm_over_free_stale": cmp(w, f), "warm_over_warm_clean": cmp(w, clean[b]), "w25_over_cold": cmp(w25, g),
                               "w25_over_warm": cmp(w25, w), "w25_over_free_stale": cmp(w25, f), "free_stale_over_cold": cmp(f, g), "free_stale_over_static": cmp(f, s),
                               "abs_mean_errors_caught_all8": {"uniform": float(u.mean()), "static": float(s.mean()), "greedy_online": float(g.mean()), "recal_free_200": float(f.mean()),
                                                               "warm_online_200": float(w.mean()), "warm_online_200_w25": float(w25.mean()), "warm_online_200_clean_L13a": float(clean[b].mean())}}
        res["cells"][key] = c

# ---- verdict (pre-registered primary: flip25, 25%, n_sim 2250)
gate_ok = bool(g1 and g1.get("passed") and gi and gi.get("passed") and g2 and g2.get("passed")); V = {}
for rg in RUNG:
    cell = res["cells"][f"{rg}|flip25"]["budgets"]["0.25"]; G, F, Wc, X5 = cell["warm_over_cold"], cell["warm_over_free_stale"], cell["warm_over_warm_clean"], cell["w25_over_cold"]
    V[rg] = {"warm_stale_over_cold": G["ratio"], "warm_stale_over_cold_ci": [G["ci_lo"], G["ci_hi"]], "warm_stale_over_cold_per_split": G["per_split_ratio"],
             "warm_stale_over_cold_split_t_ci": [G["split_t_ci_lo"], G["split_t_ci_hi"]], "warm_stale_over_cold_split_boot": [G["split_boot_lo"], G["split_boot_hi"]],
             "warm_stale_over_cold_split_t_df3": G["split_t_df3"], "warm_stale_over_cold_split_p_df3": G["split_p_df3"],
             "warm_stale_over_free_stale": F["ratio"], "warm_stale_over_free_stale_ci": [F["ci_lo"], F["ci_hi"]], "warm_stale_over_free_stale_per_split": F["per_split_ratio"],
             "warm_stale_over_free_stale_split_t_ci": [F["split_t_ci_lo"], F["split_t_ci_hi"]], "warm_stale_over_free_stale_split_boot": [F["split_boot_lo"], F["split_boot_hi"]],
             "warm_stale_over_free_stale_split_t_df3": F["split_t_df3"], "warm_stale_over_free_stale_split_p_df3": F["split_p_df3"],
             "warm_stale_over_warm_clean": Wc["ratio"], "warm_stale_over_warm_clean_ci": [Wc["ci_lo"], Wc["ci_hi"]],
             "w25_over_cold": X5["ratio"], "w25_over_cold_ci": [X5["ci_lo"], X5["ci_hi"]], "w25_over_cold_split_t_ci": [X5["split_t_ci_lo"], X5["split_t_ci_hi"]],
             "mean_warm_stale": G["mean_a"], "mean_cold": G["mean_b"], "mean_free_stale": F["mean_b"], "mean_warm_clean": Wc["mean_b"], "mean_w25": X5["mean_a"],
             "n_clusters": G["n_clusters"], "n_runs": G["n_runs"], "B": cell["B"],
             "pass": bool(G["ratio"] >= 1.00 and F["ratio"] >= 1.06 and F["ci_lo"] > 1.0), "kill": bool(G["ratio"] <= 0.97 and G["ci_hi"] < 1.0), "w25_restores_099_of_cold": bool(X5["ratio"] >= 0.99)}
outcome = "VOID" if not gate_ok else ("PASS" if all(v["pass"] for v in V.values()) else ("KILL" if any(v["kill"] for v in V.values()) else "INCONCLUSIVE"))
READING = {"PASS": "a learning component that is not a label-count effect exists (adaptation to stale labels), Phase 1 warm-starts from earlier verified labels by default and keeps refitting, and claim 2's surviving form gains a drift-robustness clause.",
           "KILL": "the L13a 1.04 is a same-distribution artifact. The Phase 1 default becomes cold start, or warm start with label-age weighting if the w25 arm restores at least 0.99 of cold; claim 2 is restated as 'the learner is the cheapest label source when none exist'.",
           "INCONCLUSIVE": "anything else is inconclusive and shifts the weight to the flip50 and lowlen rows.",
           "VOID": "a gate failed; no reading is drawn."}
kill_detail = ("w25 restores >= 0.99 of cold on both rungs: Phase 1 default = warm start with label-age weighting" if all(v["w25_restores_099_of_cold"] for v in V.values()) else "w25 does not restore 0.99 of cold on both rungs: Phase 1 default = cold start") if outcome == "KILL" else None
res["verdict"] = {"outcome": outcome, "reading": READING[outcome], "kill_followup": kill_detail, "gate_ok": gate_ok, "rungs": V,
                  "n_sim_actual": {rg: res["cells"][f"{rg}|flip25"]["n_sim"] for rg in RUNG}, "B_25pct": {rg: V[rg]["B"] for rg in RUNG}}
res["metric"] = {"cell": "25% budget, n_sim 2250, corrupted label, flip25 primary, n = 200, 8 policy seeds, 16 clusters", "rungs": V, "outcome": outcome}
json.dump(res, open(os.path.join(rd, "result.json"), "w"), indent=1)

# ---------------- markdown
f2 = lambda x: "n/a" if x is None else f"{x:.2f}"; f3 = lambda x: "n/a" if x is None else f"{x:.3f}"
ct = lambda d: f"{f3(d['ratio'])} [{f3(d['ci_lo'])}, {f3(d['ci_hi'])}]"
L = []; P_ = L.append
P_(f"## Verdict: {outcome}")
P_(f"- reading: {READING[outcome]}" + (f" Follow-up: {kill_detail}." if kill_detail else ""))
P_(f"- gate 1 (STALE=none files vs runs/L13a, n2250 split 0 sub 0, both rungs, five arms): {g1}")
P_(f"- gate 2 (identity, refits off, warm_online_200 == recal_free_200 per seed under all four STALE variants, both rungs): {gi}")
P_(f"- post gate (128 files; none == L13a on five arms; uniform/static/greedy_online == L13a in every variant): {g2}")
for rg, v in V.items():
    P_(f"- {rg} (flip25, 25%, n_sim 2250, B {v['B']:.0f}, {v['n_clusters']} clusters x 8 seeds = {v['n_runs']} runs): warm_stale_200 {v['mean_warm_stale']:.1f}, cold greedy {v['mean_cold']:.1f}, recal_free_200 (stale) {v['mean_free_stale']:.1f}, warm clean (L13a) {v['mean_warm_clean']:.1f}, w25 {v['mean_w25']:.1f}")
    P_(f"  - warm_stale / cold {f3(v['warm_stale_over_cold'])} [{f3(v['warm_stale_over_cold_ci'][0])}, {f3(v['warm_stale_over_cold_ci'][1])}] (split t interval [{f3(v['warm_stale_over_cold_split_t_ci'][0])}, {f3(v['warm_stale_over_cold_split_t_ci'][1])}], exact split bootstrap [{f3(v['warm_stale_over_cold_split_boot'][0])}, {f3(v['warm_stale_over_cold_split_boot'][1])}], per-split {[round(x, 3) for x in v['warm_stale_over_cold_per_split']]}, split t(df 3) {f2(v['warm_stale_over_cold_split_t_df3'])} p {f3(v['warm_stale_over_cold_split_p_df3'])})")
    P_(f"  - warm_stale / recal_free_200 (stale) {f3(v['warm_stale_over_free_stale'])} [{f3(v['warm_stale_over_free_stale_ci'][0])}, {f3(v['warm_stale_over_free_stale_ci'][1])}] (split t interval [{f3(v['warm_stale_over_free_stale_split_t_ci'][0])}, {f3(v['warm_stale_over_free_stale_split_t_ci'][1])}], exact split bootstrap [{f3(v['warm_stale_over_free_stale_split_boot'][0])}, {f3(v['warm_stale_over_free_stale_split_boot'][1])}], per-split {[round(x, 3) for x in v['warm_stale_over_free_stale_per_split']]}, split t(df 3) {f2(v['warm_stale_over_free_stale_split_t_df3'])} p {f3(v['warm_stale_over_free_stale_split_p_df3'])})")
    P_(f"  - warm_stale / warm clean (L13a) {f3(v['warm_stale_over_warm_clean'])} [{f3(v['warm_stale_over_warm_clean_ci'][0])}, {f3(v['warm_stale_over_warm_clean_ci'][1])}]; w25 / cold {f3(v['w25_over_cold'])} [{f3(v['w25_over_cold_ci'][0])}, {f3(v['w25_over_cold_ci'][1])}] (split t interval [{f3(v['w25_over_cold_split_t_ci'][0])}, {f3(v['w25_over_cold_split_t_ci'][1])}]); pass {v['pass']}, kill {v['kill']}, w25 restores 0.99 of cold {v['w25_restores_099_of_cold']}")
P_("\n## Main table: ratios of means [16-cluster percentile bootstrap CI] (all 8 policy seeds). Primary row: flip25 at 25%; flip50 and lowlen are descriptive; none is the L13a-reproducing control")
P_("| rung | stale | budget | B | warm_stale | cold | free_stale | warm clean | w25 | warm/cold [CI] | warm/free_stale [CI] | warm/warm_clean [CI] | w25/cold [CI] | w25/warm [CI] | free_stale/cold |"); P_("|" + "---|" * 15)
for key, c in res["cells"].items():
    rg, st = key.split("|")
    for b, cell in c["budgets"].items():
        a = cell["abs_mean_errors_caught_all8"]
        P_(f"| {rg} | {st} | {float(b):.0%} | {cell['B']:.0f} | {a['warm_online_200']:.1f} | {a['greedy_online']:.1f} | {a['recal_free_200']:.1f} | {a['warm_online_200_clean_L13a']:.1f} | {a['warm_online_200_w25']:.1f} | {ct(cell['warm_over_cold'])} | {ct(cell['warm_over_free_stale'])} | {ct(cell['warm_over_warm_clean'])} | {ct(cell['w25_over_cold'])} | {ct(cell['w25_over_warm'])} | {f3(cell['free_stale_over_cold']['ratio'])} |")
P_("\n## Honest unit: 4 splits. Per-split ratios, split-level paired t (df 3) on a minus b, t interval of the 4 split ratios (mean +- 3.182 se), exact 256-resample split bootstrap")
P_("| rung | stale | budget | metric | ratio | per-split ratios | splits > 1 | split t (p) | split t interval | exact split bootstrap | clusters > 1 |"); P_("|" + "---|" * 11)
for key, c in res["cells"].items():
    rg, st = key.split("|")
    for b, cell in c["budgets"].items():
        for mk, mn in (("warm_over_cold", "warm_stale / cold"), ("warm_over_free_stale", "warm_stale / free_stale"), ("warm_over_warm_clean", "warm_stale / warm_clean"), ("w25_over_cold", "w25 / cold")):
            d = cell[mk]
            P_(f"| {rg} | {st} | {float(b):.0%} | {mn} | {f3(d['ratio'])} | {', '.join(f'{x:.3f}' for x in d['per_split_ratio'])} | {d['n_splits_ratio_gt_1']} | {f2(d['split_t_df3'])} ({f3(d['split_p_df3'])}) | [{f3(d['split_t_ci_lo'])}, {f3(d['split_t_ci_hi'])}] | [{f3(d['split_boot_lo'])}, {f3(d['split_boot_hi'])}] | {d['frac_clusters_ratio_gt_1']:.2f} |")
P_("\n## Absolute mean errors caught (all 8 seeds, mean over 16 clusters)")
P_("| rung | stale | budget | n_sim | errors in sim | B | uniform | static | cold greedy | free_stale | warm_stale | w25 | warm clean (L13a) |"); P_("|" + "---|" * 13)
for key, c in res["cells"].items():
    rg, st = key.split("|")
    for b, cell in c["budgets"].items():
        a = cell["abs_mean_errors_caught_all8"]
        P_(f"| {rg} | {st} | {float(b):.0%} | {c['n_sim']:.0f} | {c['hits']:.0f} | {cell['B']:.0f} | {a['uniform']:.1f} | {a['static']:.1f} | {a['greedy_online']:.1f} | {a['recal_free_200']:.1f} | {a['warm_online_200']:.1f} | {a['warm_online_200_w25']:.1f} | {a['warm_online_200_clean_L13a']:.1f} |")
P_(f"\nOne-class free draws (files with any): " + ", ".join(f"{k} {c['oneclass_free_draw_files']}" for k, c in res["cells"].items()))
open(os.path.join(rd, "summary.md"), "w").write("\n".join(L) + "\n"); print("\n".join(L))
