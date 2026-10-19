"""L12c summary (CPU). Usage: l12c_summarize.py <runs/L12c>. Writes result.json and summary.md, prints the markdown.

Reads gate/gate_check.json, gate/gate_check_post.json and sim_<tag>_n<target>_sp<split>_sub<sub>.json (corrupted label, cell30, cheap_r, 8 policy seeds; tag = {3B,7B}_q_corrupted; target 4500 / 2250 / 1125 rows with
split 0..3 x subsample 0..3 = 16 clusters, plus the full-scale reference target 0 = split 0..3, 4 clusters). Cell = (rung, scale, budget fraction).
Rules are selected on policy seeds 0..3 pooled over the clusters of the cell and reported on policy seeds 4..7 (the L2a/L11a protocol); hindsight-best n shown on the eval seeds and on all 8 seeds.
  free rule     = recal_free_<n>: n labels drawn from the fit half and NOT charged, whole budget B spent on the top-ranked picks (uncharged, size-matched logistic)
  charged rule  = recal_charged_<n> (L11a): n random sim rows charged to B
Metrics per cell: greedy_online / best free (ratio of means over clusters x 4 eval seeds, percentile bootstrap CI over the clusters, 10000 resamples, one shared resample index per cluster count), greedy / best charged
(must reproduce L11a), charge cost = free_n minus charged_n at equal n (hits and percent, all 8 seeds), share explained = (free - charged) / (greedy - charged) reported only where greedy - charged > 5 hits.
HEADLINE share = equal-n: free and charged both at the n selected for the charged rule (pure charge accounting: same logistic, same n, only the charge differs). Variant: best free vs best charged (each at its own selected n).
Split-level paired t: the cluster differences are averaged to the 4 splits (the 4 subsamples of a split share a fit set and a sim population) and tested at the honest df 3 (two-sided p).

Pre-registered verdict (L12c spec), cell = 25% budget, corrupted label, target n_sim 2250 (B about 562), rungs 3B (primary) and 7B-AWQ (confirmation):
  VOID          if a regression gate failed (L11a arms not reproduced with 0 per-seed differences, the recal_free_400 <= static + 2 SE sanity, or the charged ratios not reproducing the L11a result.json)
  PASS          if on BOTH rungs greedy/best free <= 1.00 with CI upper end <= 1.02 AND the headline share explained >= 0.80 (the edge is calibration-cost accounting)
  KILL          if on BOTH rungs greedy/best free >= 1.03 with CI lower end > 1 (a learning component beyond cost accounting)
  INCONCLUSIVE  otherwise"""
import sys, os, json
import numpy as np
from scipy import stats

rd = sys.argv[1]
RUNG = {"3B": "3B_q", "7B-AWQ": "7B_q"}
SCALES = [1125, 2250, 4500, 0]; BUD = ["0.1", "0.25", "0.5"]
SEL, EV, ALL = slice(0, 4), slice(4, 8), slice(0, 8); NB = 10000
jl = lambda p: json.load(open(os.path.join(rd, p)))
_idx = {}


def boot_idx(C):
    if C not in _idx: _idx[C] = np.random.default_rng(0).integers(0, C, (NB, C))
    return _idx[C]


def ci(x): return float(np.percentile(x, 2.5)), float(np.percentile(x, 97.5))


def split_t(d):
    """d: (clusters, seeds) paired difference; clusters ordered split-major (4 splits). Paired t over the 4 split means, df 3."""
    C = d.shape[0]; sp = d.reshape(4, C // 4, -1).mean((1, 2)); s = sp.std(ddof=1)
    t = float(sp.mean() / (s / 2)) if s > 0 else None
    return {"split_mean_diffs": [float(v) for v in sp], "split_t_df3": t, "split_p_df3": float(2 * stats.t.sf(abs(t), 3)) if t is not None else None}


def cmp(a, b):
    """a, b: (clusters, seeds) paired arrays; ratio of means, cluster-bootstrap CI, split-level t of a - b."""
    C = a.shape[0]; idx = boot_idx(C); rb = a[idx].mean((1, 2)) / b[idx].mean((1, 2)); lo, hi = ci(rb)
    return {"ratio": float(a.mean() / b.mean()), "ci_lo": lo, "ci_hi": hi, "mean_a": float(a.mean()), "mean_b": float(b.mean()), "diff_hits": float(a.mean() - b.mean()), "n_runs": int(a.size), "n_clusters": int(C),
            "frac_clusters_ratio_gt_1": float((a.mean(1) / b.mean(1) > 1).mean()), **split_t(a - b)}


def share(g, f, c):
    """(free - charged) / (greedy - charged) on the same runs; None unless greedy - charged > 5 hits."""
    edge = float(g.mean() - c.mean()); o = {"edge_hits": edge, "cost_hits": float(f.mean() - c.mean()), "share": None, "ci_lo": None, "ci_hi": None}
    if edge > 5:
        idx = boot_idx(g.shape[0]); gm, fm, cm = g[idx].mean((1, 2)), f[idx].mean((1, 2)), c[idx].mean((1, 2)); lo, hi = ci((fm - cm) / (gm - cm))
        o.update(share=float((f.mean() - c.mean()) / edge), ci_lo=lo, ci_hi=hi)
    return o


def pick(h, names, sl):
    m = {p: float(h[p][:, sl].mean()) for p in names}; return max(m, key=m.get)


def load(tag, tgt):
    ds = [jl(f"sim_{tag}_n{tgt}_sp{sp}_sub{sub}.json") for sp in range(4) for sub in (range(4) if tgt else [0])]
    H = {}
    for b in BUD:
        names = set.intersection(*[set(d[f"budget_{b}"]) for d in ds])
        H[b] = {p: np.array([d[f"budget_{b}"][p]["per_seed"] for d in ds], float) for p in names}
    return ds, H, {"n_sim": [d["n_sim"] for d in ds], "hits": [d["hits"] for d in ds], "n_buckets": [d["n_buckets"] for d in ds]}


nn = lambda p: int(p.rsplit("_", 1)[1])
res = {"protocol": __doc__.split("\n")[1:], "cells": {}}
g1 = jl("gate/gate_check.json") if os.path.exists(os.path.join(rd, "gate", "gate_check.json")) else None
g2 = jl("gate/gate_check_post.json") if os.path.exists(os.path.join(rd, "gate", "gate_check_post.json")) else None
res["gate_pre_vs_L11a"] = g1; res["gate_post"] = {k: v for k, v in (g2 or {}).items() if k != "sanity_cells"}
L11 = json.load(open(os.path.join(rd, "..", "L11a", "result.json")))
repro = []
for rg, pre in RUNG.items():
    tag = f"{pre}_corrupted"
    for tgt in SCALES:
        ds, H, meta = load(tag, tgt); key = f"{rg}|corrupted|{tgt or 'full'}"
        c = {"clusters": len(ds), **{k: float(np.mean(v)) for k, v in meta.items()}, "per_cluster_n_sim": meta["n_sim"], "budgets": {}}
        for b in BUD:
            h = H[b]; rec = sorted([p for p in h if p.startswith("recal_charged_")], key=nn); fre = sorted([p for p in h if p.startswith("recal_free_")], key=nn); g = h["greedy_online"]
            cs, fs = pick(h, rec, SEL), pick(h, fre, SEL); fat = f"recal_free_{nn(cs)}"
            hef, haf = pick(h, fre, EV), pick(h, fre, ALL); hec, hac = pick(h, rec, EV), pick(h, rec, ALL)
            both = [n for n in (nn(p) for p in rec) if f"recal_free_{n}" in h]
            cost_by_n = {str(n): {"charged": float(h[f"recal_charged_{n}"].mean()), "free": float(h[f"recal_free_{n}"].mean()), "cost_hits": float(h[f"recal_free_{n}"].mean() - h[f"recal_charged_{n}"].mean()),
                                  "cost_pct": float(100 * (h[f"recal_free_{n}"].mean() / h[f"recal_charged_{n}"].mean() - 1)), "greedy_over_free": float(g.mean() / h[f"recal_free_{n}"].mean())} for n in both}
            cell = {"labels_used": float(np.mean([int(float(b) * n) for n in meta["n_sim"]])), "n_charged_candidates": [nn(p) for p in rec], "n_free_candidates": [nn(p) for p in fre],
                    "n_charged_selected_seeds_0_3": nn(cs), "n_free_selected_seeds_0_3": nn(fs),
                    "greedy_vs_selected_free": cmp(g[:, EV], h[fs][:, EV]),
                    "greedy_vs_selected_charged": cmp(g[:, EV], h[cs][:, EV]),
                    "greedy_vs_free_at_charged_n": cmp(g[:, EV], h[fat][:, EV]),
                    "free_selected_vs_charged_selected": cmp(h[fs][:, EV], h[cs][:, EV]),
                    "hind_eval_greedy_vs_best_free": {"n": nn(hef), **cmp(g[:, EV], h[hef][:, EV])}, "hind_all8_greedy_vs_best_free": {"n": nn(haf), **cmp(g, h[haf])},
                    "hind_eval_greedy_vs_best_charged": {"n": nn(hec), **cmp(g[:, EV], h[hec][:, EV])}, "hind_all8_greedy_vs_best_charged": {"n": nn(hac), **cmp(g, h[hac])},
                    "share_equal_n_headline": {"n": nn(cs), **share(g[:, EV], h[fat][:, EV], h[cs][:, EV])},
                    "share_best_vs_best": {"n_free": nn(fs), "n_charged": nn(cs), **share(g[:, EV], h[fs][:, EV], h[cs][:, EV])},
                    "charge_cost_by_n_all8": cost_by_n,
                    "greedy_vs_static_free": cmp(g, h["static"]), "greedy_vs_static_matched": cmp(g, h["static_matched"]), "greedy_vs_ts_online": cmp(g, h["ts_online"]), "greedy_vs_uniform": cmp(g, h["uniform"]),
                    "abs_mean_errors_caught_all8": {p: float(h[p].mean()) for p in h}, "abs_eval_mean_errors_caught_seeds_4_7": {p: float(h[p][:, EV].mean()) for p in h},
                    "free_n_sweep_eval_mean": {str(nn(p)): float(h[p][:, EV].mean()) for p in fre}, "charged_n_sweep_eval_mean": {str(nn(p)): float(h[p][:, EV].mean()) for p in rec}}
            c["budgets"][b] = cell
            l11 = L11["cells"][key]["budgets"][b]["primary_greedy_vs_selected_recal"]["ratio"]; l11n = L11["cells"][key]["budgets"][b]["n_cal_selected_seeds_0_3"]
            repro.append({"cell": key, "budget": b, "ratio_L12c": cell["greedy_vs_selected_charged"]["ratio"], "ratio_L11a": l11, "abs_diff": abs(cell["greedy_vs_selected_charged"]["ratio"] - l11), "n_cal_L12c": nn(cs), "n_cal_L11a": l11n})
        res["cells"][key] = c
primary_repro = [r for r in repro if r["cell"].endswith("|2250") and r["budget"] == "0.25"]
res["reproduces_L11a_charged_ratio"] = {"primary_cells": primary_repro, "max_abs_diff_all_cells": max(r["abs_diff"] for r in repro), "n_cal_mismatches": sum(r["n_cal_L12c"] != r["n_cal_L11a"] for r in repro),
                                        "ok_primary": bool(all(r["abs_diff"] < 1e-9 and r["n_cal_L12c"] == r["n_cal_L11a"] for r in primary_repro)), "all_cells": repro}

# ---- verdict
gate_ok = bool(g1 and g1.get("passed") and g2 and g2.get("passed")); V = {}
for rg in RUNG:
    cell = res["cells"][f"{rg}|corrupted|2250"]["budgets"]["0.25"]; F = cell["greedy_vs_selected_free"]; S = cell["share_equal_n_headline"]; Sb = cell["share_best_vs_best"]; Ch = cell["greedy_vs_selected_charged"]
    sh = S["share"]
    V[rg] = {"greedy_over_best_free": F["ratio"], "free_ci": [F["ci_lo"], F["ci_hi"]], "n_free_selected": cell["n_free_selected_seeds_0_3"], "split_t_df3": F["split_t_df3"], "split_p_df3": F["split_p_df3"],
             "greedy_over_best_charged_L11a_repro": Ch["ratio"], "n_charged_selected": cell["n_charged_selected_seeds_0_3"], "charged_split_t_df3": Ch["split_t_df3"], "charged_split_p_df3": Ch["split_p_df3"],
             "share_equal_n": sh, "share_equal_n_ci": [S["ci_lo"], S["ci_hi"]], "share_best_vs_best": Sb["share"], "share_best_vs_best_ci": [Sb["ci_lo"], Sb["ci_hi"]], "edge_over_charged_hits": S["edge_hits"], "cost_hits_equal_n": S["cost_hits"],
             "hindsight_eval_ratio_free": cell["hind_eval_greedy_vs_best_free"]["ratio"], "hindsight_all8_ratio_free": cell["hind_all8_greedy_vs_best_free"]["ratio"],
             "pass": bool(F["ratio"] <= 1.00 and F["ci_hi"] <= 1.02 and sh is not None and sh >= 0.80), "kill": bool(F["ratio"] >= 1.03 and F["ci_lo"] > 1.0)}
repro_ok = res["reproduces_L11a_charged_ratio"]["ok_primary"]
outcome = "VOID" if not (gate_ok and repro_ok) else ("PASS" if all(v["pass"] for v in V.values()) else ("KILL" if all(v["kill"] for v in V.values()) else "INCONCLUSIVE"))
res["verdict"] = {"outcome": outcome, "gate_ok": gate_ok, "charged_ratio_reproduces_L11a": repro_ok, "rungs": V,
                  "n_sim_actual_2250": {rg: res["cells"][f"{rg}|corrupted|2250"]["n_sim"] for rg in RUNG}, "B_25pct_2250": {rg: res["cells"][f"{rg}|corrupted|2250"]["budgets"]["0.25"]["labels_used"] for rg in RUNG}}
# ---- prediction check (spec: recal_free_100 between the charged rule 250.8 and the static ceiling 280.1 against greedy 265.7 on 3B, ratio 0.95 to 1.06)
pc = {}
for rg in RUNG:
    a = res["cells"][f"{rg}|corrupted|2250"]["budgets"]["0.25"]["abs_mean_errors_caught_all8"]; pc[rg] = {k: a[k] for k in ("recal_charged_100", "recal_free_100", "static", "greedy_online")}
    pc[rg]["greedy_over_free100"] = a["greedy_online"] / a["recal_free_100"]; pc[rg]["free100_between_charged100_and_static"] = bool(a["recal_charged_100"] <= a["recal_free_100"] <= a["static"])
res["prediction_check_all8_seeds"] = pc
# ---- dose-response: free ratio against absolute B
dr = []
for key, c in res["cells"].items():
    rg, lab, sc = key.split("|")
    for b, cell in c["budgets"].items():
        F, Ch, S = cell["greedy_vs_selected_free"], cell["greedy_vs_selected_charged"], cell["share_equal_n_headline"]
        dr.append({"rung": rg, "scale": sc, "frac": b, "B": cell["labels_used"], "clusters": c["clusters"], "ratio_free": F["ratio"], "free_ci": [F["ci_lo"], F["ci_hi"]], "n_free": cell["n_free_selected_seeds_0_3"], "ratio_charged": Ch["ratio"],
                   "charged_ci": [Ch["ci_lo"], Ch["ci_hi"]], "n_charged": cell["n_charged_selected_seeds_0_3"], "share_equal_n": S["share"], "edge_hits": S["edge_hits"], "cost_hits": S["cost_hits"],
                   "hind_eval_ratio_free": cell["hind_eval_greedy_vs_best_free"]["ratio"], "split_t_df3_free": F["split_t_df3"], "split_p_df3_free": F["split_p_df3"]})
res["dose_response_by_B"] = sorted(dr, key=lambda r: (r["rung"], r["B"]))
cross = {}
for rg in RUNG:
    rows = [r for r in res["dose_response_by_B"] if r["rung"] == rg]; cross[rg] = None
    for a, b_ in zip(rows, rows[1:]):
        if a["ratio_free"] > 1.0 >= b_["ratio_free"]:
            w = (a["ratio_free"] - 1.0) / (a["ratio_free"] - b_["ratio_free"]); cross[rg] = float(np.exp(np.log(a["B"]) + w * (np.log(b_["B"]) - np.log(a["B"])))); break
res["crossing_B_ratio_free_falls_to_1_descriptive"] = cross
res["metric"] = {"cell": "25% budget, n_sim 2250, corrupted label", "rungs": V, "outcome": outcome}
json.dump(res, open(os.path.join(rd, "result.json"), "w"), indent=1)

# ---------------- markdown
f2 = lambda x: "n/a" if x is None else f"{x:.2f}"; f3 = lambda x: "n/a" if x is None else f"{x:.3f}"; f4 = lambda x: "n/a" if x is None else f"{x:.4f}"
L = []; P_ = L.append
P_(f"## Verdict: {outcome}"); P_(f"- gate pre (L12c arms with recal_free skipped vs runs/L11a, n2250 and full, split 0): {g1}"); P_(f"- gate post (all 104 grid files, existing arms vs L11a; recal_free_400 <= static + 2 SE): {res['gate_post']}")
P_(f"- charged ratios reproduce the L11a result.json: primary cells ok {repro_ok}, max abs diff over all 24 cells {res['reproduces_L11a_charged_ratio']['max_abs_diff_all_cells']:.2e}, n_cal mismatches {res['reproduces_L11a_charged_ratio']['n_cal_mismatches']}")
for rg, v in V.items():
    P_(f"- {rg} (corrupted, n_sim 2250, 25%, B {res['verdict']['B_25pct_2250'][rg]:.0f}): greedy / best free (n {v['n_free_selected']}) = {f3(v['greedy_over_best_free'])} [{f3(v['free_ci'][0])}, {f3(v['free_ci'][1])}], split-level t(df 3) {f2(v['split_t_df3'])} p {f3(v['split_p_df3'])}; "
         f"greedy / best charged (n {v['n_charged_selected']}) = {f3(v['greedy_over_best_charged_L11a_repro'])}, t {f2(v['charged_split_t_df3'])} p {f3(v['charged_split_p_df3'])}; edge over charged {v['edge_over_charged_hits']:.1f} hits, charge cost at equal n {v['cost_hits_equal_n']:.1f} hits; "
         f"share explained (equal n) {f3(v['share_equal_n'])} [{f3(v['share_equal_n_ci'][0])}, {f3(v['share_equal_n_ci'][1])}], best-vs-best {f3(v['share_best_vs_best'])}; hindsight free ratio eval {f3(v['hindsight_eval_ratio_free'])} / all8 {f3(v['hindsight_all8_ratio_free'])}; pass {v['pass']}, kill {v['kill']}")
P_(f"- actual n_sim at target 2250 (mean over 16 clusters): {res['verdict']['n_sim_actual_2250']}")
P_(f"- prediction check (all 8 seeds, 25%, n_sim 2250): {json.dumps(pc)}")
P_(f"- descriptive crossing B where greedy / best free falls to 1.0: {cross}")
P_("\n## Main table: greedy_online vs best uncharged (free) and best charged rule (n selected on seeds 0-3, reported on 4-7; cluster-bootstrap 95% CI; split-level paired t at df 3 on greedy minus rule)")
P_("| rung | scale | budget | clusters | n_sim | B | greedy | n_free | free | n_cal | charged | greedy/free [CI] | split t (p) | greedy/charged [CI] | split t (p) | cost at equal n (hits, %) | edge (hits) | share equal-n [CI] | share best-vs-best | hind_eval greedy/free (n) | hind_all8 greedy/free (n) |"); P_("|" + "---|" * 21)
for key, c in res["cells"].items():
    rg, lab, sc = key.split("|")
    for b, cell in c["budgets"].items():
        F, Ch, S, Sb = cell["greedy_vs_selected_free"], cell["greedy_vs_selected_charged"], cell["share_equal_n_headline"], cell["share_best_vs_best"]; he, ha = cell["hind_eval_greedy_vs_best_free"], cell["hind_all8_greedy_vs_best_free"]
        cc = cell["charge_cost_by_n_all8"][str(cell["n_charged_selected_seeds_0_3"])]
        sh = f"{f3(S['share'])} [{f3(S['ci_lo'])}, {f3(S['ci_hi'])}]" if S["share"] is not None else "n/a (edge <= 5 hits)"
        P_(f"| {rg} | {sc} | {float(b):.0%} | {c['clusters']} | {c['n_sim']:.0f} | {cell['labels_used']:.0f} | {F['mean_a']:.1f} | {cell['n_free_selected_seeds_0_3']} | {F['mean_b']:.1f} | {cell['n_charged_selected_seeds_0_3']} | {Ch['mean_b']:.1f} | "
           f"{f3(F['ratio'])} [{f3(F['ci_lo'])}, {f3(F['ci_hi'])}] | {f2(F['split_t_df3'])} ({f3(F['split_p_df3'])}) | {f3(Ch['ratio'])} [{f3(Ch['ci_lo'])}, {f3(Ch['ci_hi'])}] | {f2(Ch['split_t_df3'])} ({f3(Ch['split_p_df3'])}) | {cc['cost_hits']:.1f}, {cc['cost_pct']:.1f}% | {S['edge_hits']:.1f} | {sh} | {f3(Sb['share'])} | {f3(he['ratio'])} ({he['n']}) | {f3(ha['ratio'])} ({ha['n']}) |")
P_("\n## Charge cost by n (free_n minus charged_n at equal n; all 8 policy seeds, mean over clusters) and greedy over free_n")
P_("| rung | scale | budget | n | charged_n | free_n | cost (hits) | cost (%) | greedy/free_n |"); P_("|" + "---|" * 9)
for key, c in res["cells"].items():
    rg, lab, sc = key.split("|")
    if sc not in ("2250", "full"): continue
    for b, cell in c["budgets"].items():
        for n, v in cell["charge_cost_by_n_all8"].items(): P_(f"| {rg} | {sc} | {float(b):.0%} | {n} | {v['charged']:.1f} | {v['free']:.1f} | {v['cost_hits']:.1f} | {v['cost_pct']:.1f}% | {f3(v['greedy_over_free'])} |")
P_("\n## Absolute mean errors caught (all 8 seeds, mean over clusters)")
P_("| rung | scale | budget | n_sim | errors in sim | B | uniform | static (free labels, all fit rows) | static_matched | greedy_online | ts_online | best free (sel) | best charged (sel) | oracle | free_25/50/100/200/400 |"); P_("|" + "---|" * 15)
for key, c in res["cells"].items():
    rg, lab, sc = key.split("|")
    for b, cell in c["budgets"].items():
        a = cell["abs_mean_errors_caught_all8"]; fr = "/".join(f"{a[f'recal_free_{n}']:.1f}" for n in (25, 50, 100, 200, 400))
        P_(f"| {rg} | {sc} | {float(b):.0%} | {c['n_sim']:.0f} | {c['hits']:.0f} | {cell['labels_used']:.0f} | {a['uniform']:.1f} | {a['static']:.1f} | {a['static_matched']:.1f} | {a['greedy_online']:.1f} | {a['ts_online']:.1f} | {a['recal_free_%d' % cell['n_free_selected_seeds_0_3']]:.1f} | {a['recal_charged_%d' % cell['n_charged_selected_seeds_0_3']]:.1f} | {a['oracle']:.1f} | {fr} |")
P_("\n## Dose-response: greedy / best free and greedy / best charged against absolute B (sorted by B within rung)")
P_("| rung | B | scale | budget | greedy/free [CI] | n_free | greedy/charged [CI] | n_cal | share equal-n | edge (hits) | cost (hits) | split t df3 (free) | p |"); P_("|" + "---|" * 13)
for r in res["dose_response_by_B"]: P_(f"| {r['rung']} | {r['B']:.0f} | {r['scale']} | {float(r['frac']):.0%} | {f3(r['ratio_free'])} [{f3(r['free_ci'][0])}, {f3(r['free_ci'][1])}] | {r['n_free']} | {f3(r['ratio_charged'])} [{f3(r['charged_ci'][0])}, {f3(r['charged_ci'][1])}] | {r['n_charged']} | {f3(r['share_equal_n'])} | {r['edge_hits']:.1f} | {r['cost_hits']:.1f} | {f2(r['split_t_df3_free'])} | {f3(r['split_p_df3_free'])} |")
open(os.path.join(rd, "summary.md"), "w").write("\n".join(L) + "\n"); print("\n".join(L))
