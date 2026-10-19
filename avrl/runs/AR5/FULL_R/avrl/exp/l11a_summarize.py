"""L11a summary (CPU). Usage: l11a_summarize.py <runs/L11a>. Writes result.json and summary.md, prints the markdown.

Reads gate/gate_check.json and sim_<tag>_n<target>_sp<split>_sub<sub>.json (cell30, cheap_r, 8 policy seeds each; tag = {3B,7B}_q_{corrupted,judgewrong}; target 4500 / 2250 / 1125 rows with split 0..3 x subsample 0..3
= 16 clusters, plus the full-scale reference target 0 = split 0..3, 4 clusters). Cell = (rung, label, scale, budget fraction).
Primary ratio = greedy_online errors caught / best charged recal rule (recal_charged_<n_cal>, n_cal chosen on policy seeds 0..3 pooled over all clusters, ratio reported on policy seeds 4..7: the L2a protocol),
mean over (clusters x 4 seeds) runs. Pooled-gap z = (mean A - mean B) / sqrt(sd_A^2 + sd_B^2) with np.std (ddof 0) over all runs of the evaluation seeds (cluster spread is inside the SD, so z is conservative: it is shared
between arms). CI = percentile bootstrap (10000 resamples) over the clusters (a resampled cluster keeps its policy seeds). Conservative checks: n_cal picked in hindsight on the evaluation seeds 4..7 themselves (hind_eval)
and on all 8 seeds (hind_all8). Secondary ratios: greedy_online / ts_online, / static (free labels, upper bound), / static_matched, / uniform on all 8 seeds; / best label-free threshold (thr_judge_low, thr_boundary,
thr_tokens_long, thr_trunc_first) with the same select-on-0..3 / report-on-4..7 rule, plus hindsight on all 8 seeds.

Pre-registered verdict (L11a spec), cell = 25% budget, corrupted label, target n_sim 2250 (B about 560), rungs 3B (primary) and 7B-AWQ (confirmation):
  VOID          if the regression gate (0 per-seed differences against runs/L1a at target 0, split 0, cheap_r, cell100) did not pass
  PASS          if the primary ratio >= 1.04 with the pooled gap z > 2 on BOTH rungs (budget scale, not feature regime, explains the L1a vs L1b gap)
  KILL          if the primary ratio <= 1.02 on BOTH rungs (the learner's edge over a fair charged multi-feature rule needs group-level features)
  INCONCLUSIVE  otherwise (one rung passing and one failing, or 1.02 to 1.04); the dose-response table then decides what to report"""
import sys, os, json
import numpy as np

rd = sys.argv[1]
RUNG = {"3B": "3B_q", "7B-AWQ": "7B_q"}; LABEL = {"corrupted": "corrupted", "judge_wrong": "judgewrong"}
SCALES = [1125, 2250, 4500, 0]; BUD = ["0.1", "0.25", "0.5"]; THR = ["thr_judge_low", "thr_boundary", "thr_tokens_long", "thr_trunc_first"]
SEL, EV = slice(0, 4), slice(4, 8); NB = 10000
rng = np.random.default_rng(0)
jl = lambda p: json.load(open(os.path.join(rd, p)))

def load(tag, tgt):
    ds = [jl(f"sim_{tag}_n{tgt}_sp{sp}_sub{sub}.json") for sp in range(4) for sub in (range(4) if tgt else [0])]
    H = {}
    for b in BUD:
        names = set.intersection(*[set(d[f"budget_{b}"]) for d in ds])  # a policy that exists in every cluster (n_cal filter n <= budget // 2 can differ at the edge)
        H[b] = {p: np.array([d[f"budget_{b}"][p]["per_seed"] for d in ds], float) for p in names}
    meta = {"n_sim": [d["n_sim"] for d in ds], "hits": [d["hits"] for d in ds], "n_buckets": [d["n_buckets"] for d in ds], "n_sim_pids": [d["n_sim_pids"] for d in ds]}
    return ds, H, meta

def cmp(a, b):
    """a, b: (clusters, seeds) paired arrays."""
    ma, mb, sa, sb = a.mean(), b.mean(), a.std(), b.std(); C = a.shape[0]
    idx = rng.integers(0, C, (NB, C)); rb = a[idx].mean((1, 2)) / b[idx].mean((1, 2)); cr = a.mean(1) / b.mean(1)
    return {"ratio": float(ma / mb), "z": float((ma - mb) / np.sqrt(sa ** 2 + sb ** 2)) if sa ** 2 + sb ** 2 > 0 else None, "ci_lo": float(np.percentile(rb, 2.5)), "ci_hi": float(np.percentile(rb, 97.5)),
            "mean_a": float(ma), "mean_b": float(mb), "sd_a": float(sa), "sd_b": float(sb), "n_runs": int(a.size), "n_clusters": int(C), "cluster_ratio_sd": float(cr.std(ddof=1)) if C > 1 else None,
            "frac_clusters_ratio_gt_1": float((cr > 1).mean()),
            # post-hoc diagnostics, NOT part of the pre-registered rule: z with the SD taken within clusters (policy-seed noise only, the subsample/split spread removed) and the paired t over cluster differences
            "z_within_cluster_sd": float((ma - mb) / np.sqrt(a.var(1).mean() + b.var(1).mean())) if a.var(1).mean() + b.var(1).mean() > 0 else None,
            "paired_t_over_clusters": float((a.mean(1) - b.mean(1)).mean() / ((a.mean(1) - b.mean(1)).std(ddof=1) / np.sqrt(C))) if C > 1 and (a.mean(1) - b.mean(1)).std(ddof=1) > 0 else None}

def pick(h, names, sl):
    m = {p: float(h[p][:, sl].mean()) for p in names}; return max(m, key=m.get), m

res = {"protocol": __doc__.split("\n")[1:], "cells": {}}
gate = jl("gate/gate_check.json") if os.path.exists(os.path.join(rd, "gate", "gate_check.json")) else None
gate_ok = bool(gate and gate.get("passed")); res["regression_gate_vs_L1a"] = gate
for lab, ltag in LABEL.items():
    for rg, pre in RUNG.items():
        tag = f"{pre}_{ltag}"
        for tgt in SCALES:
            ds, H, meta = load(tag, tgt); key = f"{rg}|{lab}|{tgt or 'full'}"
            c = {"clusters": len(ds), **{k: float(np.mean(v)) for k, v in meta.items()}, "per_cluster_n_sim": meta["n_sim"], "budgets": {}}
            for b in BUD:
                h = H[b]; rec = sorted([p for p in h if p.startswith("recal_charged_")], key=lambda p: int(p.rsplit("_", 1)[1]))
                g = h["greedy_online"]; bsel, msel = pick(h, rec, SEL); tsel, tm = pick(h, THR, SEL); hev, _ = pick(h, rec, EV); hall, _ = pick(h, rec, slice(0, 8)); thall, _ = pick(h, THR, slice(0, 8))
                labels_used = float(np.mean([int(float(b) * n) for n in meta["n_sim"]]))
                cell = {"labels_used": labels_used, "n_cal_candidates": [int(p.rsplit("_", 1)[1]) for p in rec], "n_cal_selected_seeds_0_3": int(bsel.rsplit("_", 1)[1]), "sel_means": msel,
                        "primary_greedy_vs_selected_recal": cmp(g[:, EV], h[bsel][:, EV]),
                        "hind_eval_greedy_vs_best_recal": {"n_cal": int(hev.rsplit("_", 1)[1]), **cmp(g[:, EV], h[hev][:, EV])},
                        "hind_all8_greedy_vs_best_recal": {"n_cal": int(hall.rsplit("_", 1)[1]), **cmp(g, h[hall])},
                        "greedy_vs_ts_online": cmp(g, h["ts_online"]), "greedy_vs_static_free": cmp(g, h["static"]), "greedy_vs_static_matched": cmp(g, h["static_matched"]), "greedy_vs_uniform": cmp(g, h["uniform"]),
                        "greedy_vs_selected_thr": {"thr": tsel, **cmp(g[:, EV], h[tsel][:, EV])}, "greedy_vs_hind_all8_thr": {"thr": thall, **cmp(g, h[thall])},
                        "recal_selected_vs_selected_thr": cmp(h[bsel][:, EV], h[tsel][:, EV]),
                        "abs_mean_errors_caught_all8": {p: float(h[p].mean()) for p in h}, "abs_eval_mean_errors_caught_seeds_4_7": {p: float(h[p][:, EV].mean()) for p in h},
                        "recal_n_cal_sweep_eval_mean": {str(int(p.rsplit("_", 1)[1])): float(h[p][:, EV].mean()) for p in rec}}
                c["budgets"][b] = cell
            res["cells"][key] = c
# ---- full-scale (target 0, cell30, split 0) must reproduce the L1a cell30 s0 file
rep = {}
for lab, ltag in LABEL.items():
    for rg, pre in RUNG.items():
        tag = f"{pre}_{ltag}"; A = json.load(open(f"runs/L1a/sim_{tag}_cheap_cell30_s0.json")); Bj = jl(f"sim_{tag}_n0_sp0_sub0.json")
        bad = [(b, p) for b in BUD for p, v in A[f"budget_{b}"].items() if v["per_seed"] != Bj[f"budget_{b}"][p]["per_seed"]]
        rep[tag] = {"policy_budget_cells_compared": sum(len(A[f"budget_{b}"]) for b in BUD), "differences": len(bad)}
res["full_scale_split0_vs_L1a_cell30"] = rep
# ---- verdict
V = {}
for rg in RUNG:
    P = res["cells"][f"{rg}|corrupted|2250"]["budgets"]["0.25"]["primary_greedy_vs_selected_recal"]; H = res["cells"][f"{rg}|corrupted|2250"]["budgets"]["0.25"]["hind_eval_greedy_vs_best_recal"]
    V[rg] = {"ratio": P["ratio"], "z": P["z"], "ci": [P["ci_lo"], P["ci_hi"]], "pass": bool(P["ratio"] >= 1.04 and P["z"] is not None and P["z"] > 2), "kill": bool(P["ratio"] <= 1.02),
             "posthoc_z_within_cluster_sd": P["z_within_cluster_sd"], "posthoc_paired_t_over_clusters": P["paired_t_over_clusters"], "hindsight_eval_ratio": H["ratio"], "hindsight_eval_z": H["z"], "hindsight_pass": bool(H["ratio"] >= 1.04 and H["z"] is not None and H["z"] > 2), "hindsight_kill": bool(H["ratio"] <= 1.02)}
outcome = "VOID" if not gate_ok else ("PASS" if all(v["pass"] for v in V.values()) else ("KILL" if all(v["kill"] for v in V.values()) else "INCONCLUSIVE"))
res["verdict"] = {"outcome": outcome, "gate_ok": gate_ok, "rungs": V, "n_sim_actual_2250": {rg: res["cells"][f"{rg}|corrupted|2250"]["n_sim"] for rg in RUNG},
                  "B_25pct_2250": {rg: res["cells"][f"{rg}|corrupted|2250"]["budgets"]["0.25"]["labels_used"] for rg in RUNG}}
# ---- dose-response: ratio vs absolute budget B (all cells of all scales)
dr = []
for key, c in res["cells"].items():
    rg, lab, sc = key.split("|")
    for b, cell in c["budgets"].items():
        p = cell["primary_greedy_vs_selected_recal"]; dr.append({"rung": rg, "label": lab, "scale": sc, "frac": b, "B": cell["labels_used"], "ratio": p["ratio"], "z": p["z"], "ci": [p["ci_lo"], p["ci_hi"]], "hind_eval_ratio": cell["hind_eval_greedy_vs_best_recal"]["ratio"]})
res["dose_response_by_B"] = sorted(dr, key=lambda r: r["B"])
json.dump(res, open(os.path.join(rd, "result.json"), "w"), indent=1)

# ---------------- markdown
f2 = lambda x: "n/a" if x is None else f"{x:.2f}"; f3 = lambda x: "n/a" if x is None else f"{x:.3f}"
L = []; P_ = L.append
P_(f"## Verdict: {outcome}"); P_(f"- regression gate vs runs/L1a: {gate}"); P_(f"- full scale (target 0, cell30, split 0) vs L1a cell30 s0 per-seed differences: {json.dumps(rep)}")
for rg, v in V.items():
    P_(f"- {rg} (corrupted, n_sim 2250, 25%): greedy/best charged recal = {f3(v['ratio'])} [{f3(v['ci'][0])}, {f3(v['ci'][1])}], z {f2(v['z'])} (post-hoc, not the rule: z within-cluster SD {f2(v['posthoc_z_within_cluster_sd'])}, paired t over 16 clusters {f2(v['posthoc_paired_t_over_clusters'])}), pass {v['pass']}, kill {v['kill']}; hindsight-on-eval-seeds ratio {f3(v['hindsight_eval_ratio'])} (z {f2(v['hindsight_eval_z'])})")
P_(f"- actual n_sim at target 2250 (mean over 16 clusters): {res['verdict']['n_sim_actual_2250']}, B at 25%: {res['verdict']['B_25pct_2250']}")
P_("\n## Primary ratio: greedy_online / best charged recal rule (n_cal selected on policy seeds 0-3, reported on 4-7; cluster-bootstrap 95% CI; z = pooled-SD gap)")
P_("| rung | label | scale | budget | n_sim | B | n_cal | greedy | recal | ratio [CI] | z | hind_eval ratio (n_cal) | hind_all8 ratio (n_cal) | clusters ratio>1 | posthoc z_within / paired t |"); P_("|" + "---|" * 15)
for key, c in res["cells"].items():
    rg, lab, sc = key.split("|")
    for b, cell in c["budgets"].items():
        p = cell["primary_greedy_vs_selected_recal"]; he = cell["hind_eval_greedy_vs_best_recal"]; ha = cell["hind_all8_greedy_vs_best_recal"]
        P_(f"| {rg} | {lab} | {sc} | {float(b):.0%} | {c['n_sim']:.0f} | {cell['labels_used']:.0f} | {cell['n_cal_selected_seeds_0_3']} | {p['mean_a']:.1f} | {p['mean_b']:.1f} | {f3(p['ratio'])} [{f3(p['ci_lo'])}, {f3(p['ci_hi'])}] | {f2(p['z'])} | {f3(he['ratio'])} ({he['n_cal']}) | {f3(ha['ratio'])} ({ha['n_cal']}) | {p['frac_clusters_ratio_gt_1']:.2f} | {f2(p['z_within_cluster_sd'])} / {f2(p['paired_t_over_clusters'])} |")
P_("\n## Secondary ratios (greedy_online over; all 8 policy seeds unless stated; z in brackets)")
P_("| rung | label | scale | budget | / ts_online | / static (free labels) | / static_matched | / uniform | / best thr (sel 0-3, eval 4-7) | thr | / best thr (hindsight all8) |"); P_("|" + "---|" * 11)
for key, c in res["cells"].items():
    rg, lab, sc = key.split("|")
    for b, cell in c["budgets"].items():
        x = lambda k: f"{f3(cell[k]['ratio'])} ({f2(cell[k]['z'])})"
        P_(f"| {rg} | {lab} | {sc} | {float(b):.0%} | {x('greedy_vs_ts_online')} | {x('greedy_vs_static_free')} | {x('greedy_vs_static_matched')} | {x('greedy_vs_uniform')} | {x('greedy_vs_selected_thr')} | {cell['greedy_vs_selected_thr']['thr']} | {x('greedy_vs_hind_all8_thr')} |")
P_("\n## Absolute mean errors caught (all 8 seeds, mean over clusters)")
P_("| rung | label | scale | budget | n_sim | errors in sim | B | uniform | static | static_matched | greedy_online | ts_online | best thr | best recal (sel) | oracle |"); P_("|" + "---|" * 15)
for key, c in res["cells"].items():
    rg, lab, sc = key.split("|")
    for b, cell in c["budgets"].items():
        a = cell["abs_mean_errors_caught_all8"]; bt = max(a[t] for t in THR); br = a[f"recal_charged_{cell['n_cal_selected_seeds_0_3']}"]
        P_(f"| {rg} | {lab} | {sc} | {float(b):.0%} | {c['n_sim']:.0f} | {c['hits']:.0f} | {cell['labels_used']:.0f} | {a['uniform']:.1f} | {a['static']:.1f} | {a['static_matched']:.1f} | {a['greedy_online']:.1f} | {a['ts_online']:.1f} | {bt:.1f} | {br:.1f} | {a['oracle']:.1f} |")
P_("\n## Dose-response: primary ratio against absolute B (all cells, sorted by B; L2b curve for comparison: 0.94 at B=100, ~1.0 at 125, 1.03 to 1.09 at 250, 1.07 to 1.08 at 500)")
P_("| B | rung | label | scale | budget | ratio [CI] | z | hind_eval ratio |"); P_("|" + "---|" * 8)
for r in res["dose_response_by_B"]: P_(f"| {r['B']:.0f} | {r['rung']} | {r['label']} | {r['scale']} | {float(r['frac']):.0%} | {f3(r['ratio'])} [{f3(r['ci'][0])}, {f3(r['ci'][1])}] | {f2(r['z'])} | {f3(r['hind_eval_ratio'])} |")
open(os.path.join(rd, "summary.md"), "w").write("\n".join(L) + "\n"); print("\n".join(L))
