"""L13a summary (CPU). Usage: l13a_summarize.py <runs/L13a>. Writes result.json and summary.md, prints the markdown.

Reads gate/gate_check.json (gate 1, regression vs L12c), gate/gate_check_ident.json (gate 2, warm == recal_free with refits off), gate/gate_check_post.json and
sim_<tag>_n<target>_sp<split>_sub<sub>.json (corrupted label, cell30, cheap_r, 8 policy seeds; tag = {3B,7B}_q_corrupted; target 4500 / 2250 / 1125 rows with split 0..3 x subsample 0..3 = 16 clusters,
plus the full-scale reference target 0 = split 0..3, 4 clusters). Cell = (rung, scale, budget fraction). Nothing is tuned: every ratio uses all 8 policy seeds (no selection step).
  free rule   = recal_free_<n> (L12c): n labels drawn from the fit half and NOT charged, logistic fitted once, whole budget B spent on the top-ranked sim rows
  warm arm    = warm_online_<n>: the greedy online learner (no layout) warm-started with that same logistic (same fit-half draw per policy seed), refit every max(B // 8, 40) labels on the n free labels pooled with
                the seen sim labels; hits on picks only, the n free labels count nothing
Metrics per cell: warm_online_n / recal_free_n (ratio of means over clusters x 8 seeds, percentile bootstrap CI over the clusters, 10000 resamples, one shared resample index per cluster count), the per-split
ratios (the 4 splits are the honest independent unit because the 4 subsamples of a split share a fit set and sim prompts), the split-level paired t on warm minus free at df 3 (two-sided p),
warm_online_200 / recal_free_400, warm_online_n / greedy_online (cold), warm_online_n / static, and the share of the free-to-static gap closed, (warm_n - free_n) / (static - free_n) (reported only where static - free_n > 5 hits).

Pre-registered verdict (L13a spec), cell = 25% budget, corrupted label, target n_sim 2250 (B about 562), n = 200, rungs 3B and 7B-AWQ:
  VOID          if gate 1 (L12c arms not reproduced with 0 per-seed differences), gate 2 (warm_online_n != recal_free_n with refits off) or the post gate (existing arms of the grid != L12c) fails
  PASS          if on BOTH rungs warm_online_200 / recal_free_200 >= 1.03 with the bootstrap CI lower end above 1 AND the ratio is above 1.00 in all 4 splits on both rungs
  KILL          if on BOTH rungs warm_online_200 / recal_free_200 <= 1.01 with the bootstrap CI upper end <= 1.02
  INCONCLUSIVE  otherwise"""
import sys, os, json
import numpy as np
from scipy import stats

rd = sys.argv[1]
RUNG = {"3B": "3B_q", "7B-AWQ": "7B_q"}
SCALES = [1125, 2250, 4500, 0]; BUD = ["0.1", "0.25", "0.5"]; NS = (100, 200, 400); NB = 10000
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
    """a, b: (clusters, seeds) paired arrays; ratio of means, cluster-bootstrap CI, per-split ratios, split-level t of a - b."""
    C = a.shape[0]; idx = boot_idx(C); rb = a[idx].mean((1, 2)) / b[idx].mean((1, 2)); lo, hi = ci(rb)
    spr = a.reshape(4, C // 4, -1).mean((1, 2)) / b.reshape(4, C // 4, -1).mean((1, 2))
    return {"ratio": float(a.mean() / b.mean()), "ci_lo": lo, "ci_hi": hi, "mean_a": float(a.mean()), "mean_b": float(b.mean()), "diff_hits": float(a.mean() - b.mean()), "n_runs": int(a.size), "n_clusters": int(C),
            "per_split_ratio": [float(v) for v in spr], "min_split_ratio": float(spr.min()), "n_splits_ratio_gt_1": int((spr > 1.0).sum()),
            "frac_clusters_ratio_gt_1": float((a.mean(1) / b.mean(1) > 1).mean()), **split_t(a - b)}


def share(w, f, s):
    """(warm - free) / (static - free) on the same runs, bootstrap CI over clusters; None unless static - free > 5 hits."""
    gap = float(s.mean() - f.mean()); o = {"gap_hits": gap, "gain_hits": float(w.mean() - f.mean()), "share": None, "ci_lo": None, "ci_hi": None}
    if gap > 5:
        idx = boot_idx(w.shape[0]); wm, fm, sm = w[idx].mean((1, 2)), f[idx].mean((1, 2)), s[idx].mean((1, 2)); lo, hi = ci((wm - fm) / (sm - fm))
        o.update(share=float((w.mean() - f.mean()) / gap), ci_lo=lo, ci_hi=hi)
    return o


def load(tag, tgt):
    ds = [jl(f"sim_{tag}_n{tgt}_sp{sp}_sub{sub}.json") for sp in range(4) for sub in (range(4) if tgt else [0])]
    H = {}
    for b in BUD:
        names = set.intersection(*[set(d[f"budget_{b}"]) for d in ds])
        H[b] = {p: np.array([d[f"budget_{b}"][p]["per_seed"] for d in ds], float) for p in names}
    return ds, H, {"n_sim": [d["n_sim"] for d in ds], "hits": [d["hits"] for d in ds], "n_buckets": [d["n_buckets"] for d in ds]}


res = {"protocol": __doc__.split("\n")[1:], "cells": {}}
opt = lambda p: jl(p) if os.path.exists(os.path.join(rd, p)) else None
g1, gi, g2 = opt("gate/gate_check.json"), opt("gate/gate_check_ident.json"), opt("gate/gate_check_post.json")
res["gate1_regression_vs_L12c"] = g1; res["gate2_identity_warm_eq_free_norefit"] = gi; res["gate_post"] = g2
for rg, pre in RUNG.items():
    tag = f"{pre}_corrupted"
    for tgt in SCALES:
        ds, H, meta = load(tag, tgt); key = f"{rg}|corrupted|{tgt or 'full'}"
        c = {"clusters": len(ds), **{k: float(np.mean(v)) for k, v in meta.items()}, "per_cluster_n_sim": meta["n_sim"], "oneclass_free_draws_files": int(sum(bool(d["warm_oneclass_draws"]) for d in ds)), "budgets": {}}
        for b in BUD:
            h = H[b]; w = {n: h[f"warm_online_{n}"] for n in NS}; f = {n: h[f"recal_free_{n}"] for n in NS}; g, st = h["greedy_online"], h["static"]
            cell = {"labels_used": float(np.mean([int(float(b) * n) for n in meta["n_sim"]])),
                    "warm_over_free_same_n": {str(n): cmp(w[n], f[n]) for n in NS},
                    "warm200_over_free400": cmp(w[200], f[400]),
                    "warm_over_cold_greedy": {str(n): cmp(w[n], g) for n in NS},
                    "warm_over_static": {str(n): cmp(w[n], st) for n in NS},
                    "share_gap_closed": {str(n): share(w[n], f[n], st) for n in NS},
                    "headroom": {"static_over_free200": cmp(st, f[200]), "cold_greedy_over_free200": cmp(g, f[200]), "cold_greedy_over_static": cmp(g, st)},
                    "abs_mean_errors_caught_all8": {p: float(h[p].mean()) for p in h}}
            c["budgets"][b] = cell
        res["cells"][key] = c

# ---- verdict
gate_ok = bool(g1 and g1.get("passed") and gi and gi.get("passed") and g2 and g2.get("passed")); V = {}
for rg in RUNG:
    cell = res["cells"][f"{rg}|corrupted|2250"]["budgets"]["0.25"]; W = cell["warm_over_free_same_n"]["200"]; S = cell["share_gap_closed"]["200"]; Hd = cell["headroom"]
    V[rg] = {"warm200_over_free200": W["ratio"], "ci": [W["ci_lo"], W["ci_hi"]], "per_split_ratio": W["per_split_ratio"], "n_splits_ratio_gt_1": W["n_splits_ratio_gt_1"],
             "split_t_df3": W["split_t_df3"], "split_p_df3": W["split_p_df3"], "mean_warm200": W["mean_a"], "mean_free200": W["mean_b"], "n_clusters": W["n_clusters"], "n_runs": W["n_runs"],
             "warm200_over_free400": cell["warm200_over_free400"]["ratio"], "warm200_over_free400_ci": [cell["warm200_over_free400"]["ci_lo"], cell["warm200_over_free400"]["ci_hi"]],
             "warm200_over_cold_greedy": cell["warm_over_cold_greedy"]["200"]["ratio"], "warm200_over_static": cell["warm_over_static"]["200"]["ratio"],
             "share_gap_closed_200": S["share"], "share_gap_closed_200_ci": [S["ci_lo"], S["ci_hi"]],
             "headroom_static_over_free200": Hd["static_over_free200"]["ratio"], "headroom_cold_greedy_over_free200": Hd["cold_greedy_over_free200"]["ratio"],
             "pass": bool(W["ratio"] >= 1.03 and W["ci_lo"] > 1.0 and W["n_splits_ratio_gt_1"] == 4), "kill": bool(W["ratio"] <= 1.01 and W["ci_hi"] <= 1.02)}
outcome = "VOID" if not gate_ok else ("PASS" if all(v["pass"] for v in V.values()) else ("KILL" if all(v["kill"] for v in V.values()) else "INCONCLUSIVE"))
res["verdict"] = {"outcome": outcome, "gate_ok": gate_ok, "rungs": V, "n_sim_actual_2250": {rg: res["cells"][f"{rg}|corrupted|2250"]["n_sim"] for rg in RUNG},
                  "B_25pct_2250": {rg: res["cells"][f"{rg}|corrupted|2250"]["budgets"]["0.25"]["labels_used"] for rg in RUNG}}
res["metric"] = {"cell": "25% budget, n_sim 2250, corrupted label, n = 200, 8 policy seeds, 16 clusters", "rungs": V, "outcome": outcome}
json.dump(res, open(os.path.join(rd, "result.json"), "w"), indent=1)

# ---------------- markdown
f2 = lambda x: "n/a" if x is None else f"{x:.2f}"; f3 = lambda x: "n/a" if x is None else f"{x:.3f}"
ct = lambda d: f"{f3(d['ratio'])} [{f3(d['ci_lo'])}, {f3(d['ci_hi'])}]"
L = []; P_ = L.append
P_(f"## Verdict: {outcome}")
P_(f"- gate 1 (nowarm files vs runs/L12c, n2250 and full, split 0, both rungs, every arm): {g1}")
P_(f"- gate 2 (identity, refits off, warm_online_n == recal_free_n per seed, n 100/200/400, 3 budgets, both rungs): {gi}")
P_(f"- post gate (all 104 grid files, pre-existing arms vs L12c; warm arms present): {g2}")
for rg, v in V.items():
    P_(f"- {rg} (corrupted, n_sim 2250, 25%, B {res['verdict']['B_25pct_2250'][rg]:.0f}, {v['n_clusters']} clusters x 8 seeds): warm_online_200 {v['mean_warm200']:.1f} vs recal_free_200 {v['mean_free200']:.1f}: ratio {f3(v['warm200_over_free200'])} "
         f"[{f3(v['ci'][0])}, {f3(v['ci'][1])}]; per-split ratios {[round(x, 3) for x in v['per_split_ratio']]} ({v['n_splits_ratio_gt_1']} of 4 above 1.00); split t(df 3) {f2(v['split_t_df3'])} p {f3(v['split_p_df3'])}; "
         f"warm200 / free400 {f3(v['warm200_over_free400'])} [{f3(v['warm200_over_free400_ci'][0])}, {f3(v['warm200_over_free400_ci'][1])}]; warm200 / cold greedy {f3(v['warm200_over_cold_greedy'])}; warm200 / static {f3(v['warm200_over_static'])}; "
         f"share of the free200-to-static gap closed {f3(v['share_gap_closed_200'])} [{f3(v['share_gap_closed_200_ci'][0])}, {f3(v['share_gap_closed_200_ci'][1])}]; headroom static / free200 {f3(v['headroom_static_over_free200'])}, cold greedy / free200 {f3(v['headroom_cold_greedy_over_free200'])}; pass {v['pass']}, kill {v['kill']}")
P_(f"- actual n_sim at target 2250 (mean over 16 clusters): {res['verdict']['n_sim_actual_2250']}")
P_("\n## Main table: warm_online_200 vs recal_free_200 (all 8 policy seeds; cluster-bootstrap 95% CI; per-split ratios; split-level paired t at df 3 on warm minus free)")
P_("| rung | scale | budget | clusters | n_sim | B | warm200 | free200 | warm200/free200 [CI] | per-split ratios | splits > 1 | split t (p) | warm200/free400 [CI] | warm200/cold greedy [CI] | warm200/static [CI] | gap closed [CI] | static/free200 | cold/free200 |"); P_("|" + "---|" * 18)
for key, c in res["cells"].items():
    rg, lab, sc = key.split("|")
    for b, cell in c["budgets"].items():
        W = cell["warm_over_free_same_n"]["200"]; F4 = cell["warm200_over_free400"]; Gc = cell["warm_over_cold_greedy"]["200"]; St = cell["warm_over_static"]["200"]; S = cell["share_gap_closed"]["200"]; Hd = cell["headroom"]
        sh = f"{f3(S['share'])} [{f3(S['ci_lo'])}, {f3(S['ci_hi'])}]" if S["share"] is not None else "n/a (gap <= 5 hits)"
        P_(f"| {rg} | {sc} | {float(b):.0%} | {c['clusters']} | {c['n_sim']:.0f} | {cell['labels_used']:.0f} | {W['mean_a']:.1f} | {W['mean_b']:.1f} | {ct(W)} | {', '.join(f'{x:.3f}' for x in W['per_split_ratio'])} | {W['n_splits_ratio_gt_1']} | "
           f"{f2(W['split_t_df3'])} ({f3(W['split_p_df3'])}) | {ct(F4)} | {ct(Gc)} | {ct(St)} | {sh} | {f3(Hd['static_over_free200']['ratio'])} | {f3(Hd['cold_greedy_over_free200']['ratio'])} |")
P_("\n## n sweep: warm_online_n / recal_free_n [CI] (n = 100, 400 descriptive) and share of gap closed")
P_("| rung | scale | budget | warm100/free100 [CI] | warm400/free400 [CI] | warm100/cold | warm200/cold | warm400/cold | warm100/static | warm200/static | warm400/static | gap closed 100 | 200 | 400 |"); P_("|" + "---|" * 14)
for key, c in res["cells"].items():
    rg, lab, sc = key.split("|")
    for b, cell in c["budgets"].items():
        a, o = cell["warm_over_free_same_n"], cell["warm_over_cold_greedy"]; s_, sh = cell["warm_over_static"], cell["share_gap_closed"]
        P_(f"| {rg} | {sc} | {float(b):.0%} | {ct(a['100'])} | {ct(a['400'])} | " + " | ".join(f3(o[n]["ratio"]) for n in ("100", "200", "400")) + " | " + " | ".join(f3(s_[n]["ratio"]) for n in ("100", "200", "400")) + " | " + " | ".join(f3(sh[n]["share"]) for n in ("100", "200", "400")) + " |")
P_("\n## Absolute mean errors caught (all 8 seeds, mean over clusters)")
P_("| rung | scale | budget | n_sim | errors in sim | B | uniform | static (free labels, all fit rows) | greedy_online (cold) | ts_online | free_100/200/400 | warm_100/200/400 | oracle |"); P_("|" + "---|" * 13)
for key, c in res["cells"].items():
    rg, lab, sc = key.split("|")
    for b, cell in c["budgets"].items():
        a = cell["abs_mean_errors_caught_all8"]
        P_(f"| {rg} | {sc} | {float(b):.0%} | {c['n_sim']:.0f} | {c['hits']:.0f} | {cell['labels_used']:.0f} | {a['uniform']:.1f} | {a['static']:.1f} | {a['greedy_online']:.1f} | {a['ts_online']:.1f} | "
           f"{'/'.join(f'{a[f'recal_free_{n}']:.1f}' for n in NS)} | {'/'.join(f'{a[f'warm_online_{n}']:.1f}' for n in NS)} | {a['oracle']:.1f} |")
open(os.path.join(rd, "summary.md"), "w").write("\n".join(L) + "\n"); print("\n".join(L))
