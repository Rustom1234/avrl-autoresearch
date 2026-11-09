"""L1a driver: read the 12 sim_warm JSONs (+ optional pool-bootstrap JSON), check sanity against E21, write table.md and result.json.
Usage: l1a_table.py <run_dir> [e21_dir]
Pre-registered rules (from the L1a spec):
  learned_warm = better (by mean) of greedy_warm and ts_warm per cell; ratio = learned_warm / agree_thr; gap = learned_warm - agree_thr;
  gap in SDs = gap / SD(paired per-seed difference, ddof=1) of the chosen policy; headline variants = cell100 x {cheap, cheap_emb}.
  PASS  : headline cell, T in {0.25, 0.5}, 25% budget, ratio >= 1.05 and gap > 2 paired SD, AND pool-bootstrap CI of the ratio excludes 1.0.
  KILL  : ratio <= 1.02 at every budget (10/25/50%) and every T in the headline variants.
  Ratios in (1.02, 1.05) are marginal, not a pass. Sanity failure invalidates the run.
Bootstrap trigger: some 25% cell at T in {0.25, 0.5} (any of the 4 variants) with ratio >= 1.03; bootstrap cell = best headline cell among those (else best cell)."""
import os, sys, json
import numpy as np
run_dir = sys.argv[1]; e21_dir = sys.argv[2] if len(sys.argv) > 2 else "/homes/rdubash/avrl_live/FULL/avrl/runs/E21-lowT-confident-errors"
TS, CELLS, FEATS, BUDS = ("0.25", "0.5", "1.0"), ("cell30", "cell100"), ("cheap", "cheap_emb"), (0.1, 0.25, 0.5)
HEAD = [("cell100", "cheap"), ("cell100", "cheap_emb")]
fmt = lambda x, d=1: "n/a" if x is None else f"{x:.{d}f}"
res = {"cells": {}, "sanity": {"ok": True, "failures": [], "n_cells": 0, "n_seed_vector_comparisons": 0}}

def best(r, a, b): return a if r[a]["mean"] >= r[b]["mean"] else b
for T in TS:
    for c in CELLS:
        for f in FEATS:
            key = f"T{T}_{c}_{f}"; s = json.load(open(os.path.join(run_dir, f"sim_{key}.json"))); e = json.load(open(os.path.join(e21_dir, f"sim_{key}.json")))
            res["sanity"]["n_cells"] += 1
            cell = {"n_sim": s["n_sim"], "n_buckets": s["n_buckets"], "hits_in_sim": s["hits"], "budgets": {}}
            for bf in BUDS:
                b = f"budget_{bf}"; r, er = s[b], e[b]
                for p in ("uniform", "agree_thr", "oracle", "greedy_online", "ts_online"):  # reproduction of E21 (seed-level, stronger than to the integer)
                    if r[p]["seeds"] != er[p]["seeds"]: res["sanity"]["ok"] = False; res["sanity"]["failures"].append(f"{key} {b} {p}: {r[p]['seeds']} vs E21 {er[p]['seeds']}")
                if r["greedy_warm_norefit"]["seeds"] != r["agree_thr"]["seeds"]:
                    res["sanity"]["ok"] = False; res["sanity"]["failures"].append(f"{key} {b} greedy_warm_norefit {r['greedy_warm_norefit']['seeds']} != agree_thr {r['agree_thr']['seeds']}")
                res["sanity"]["n_seed_vector_comparisons"] += 6
                thr, orc = r["agree_thr"]["mean"], r["oracle"]["mean"]
                lp = best(r, "greedy_warm", "ts_warm"); cp = best(r, "greedy_online", "ts_online"); pr = s["paired"][b][f"{lp}_minus_agree_thr"]
                gap = r[lp]["mean"] - thr; sdp = pr["sd"]
                cell["budgets"][str(bf)] = {"budget_n": s["budget_n"][str(bf)], "uniform": r["uniform"]["mean"], "agree_thr": thr, "agree_thr_sd": r["agree_thr"]["sd"],
                    "greedy_warm": r["greedy_warm"]["mean"], "greedy_warm_sd": r["greedy_warm"]["sd"], "ts_warm": r["ts_warm"]["mean"], "ts_warm_sd": r["ts_warm"]["sd"],
                    "learned_warm_policy": lp, "learned_warm": r[lp]["mean"], "ratio": r[lp]["mean"] / thr, "ratio_greedy_warm": r["greedy_warm"]["mean"] / thr, "ratio_ts_warm": r["ts_warm"]["mean"] / thr,
                    "gap": gap, "gap_paired_sd": sdp, "gap_in_sd": (gap / sdp) if sdp > 0 else None, "gap_gt_2sd": bool(gap > 2 * sdp),
                    "paired_seed_diffs": pr["seeds"], "learned_cold_policy": cp, "learned_cold": r[cp]["mean"], "ratio_cold": r[cp]["mean"] / thr, "warm_minus_cold": r[lp]["mean"] - r[cp]["mean"],
                    "greedy_online": r["greedy_online"]["mean"], "ts_online": r["ts_online"]["mean"], "oracle": orc, "H": (orc - thr) / orc}
            res["cells"][key] = cell

# bootstrap trigger and selection
trig = [(T, c, f) for T in ("0.25", "0.5") for c in CELLS for f in FEATS if res["cells"][f"T{T}_{c}_{f}"]["budgets"]["0.25"]["ratio"] >= 1.03]
hd = [t for t in trig if (t[1], t[2]) in HEAD]; pool = hd or trig
boot_cell = max(pool, key=lambda t: res["cells"][f"T{t[0]}_{t[1]}_{t[2]}"]["budgets"]["0.25"]["ratio"]) if pool else None
nm = lambda t: f"T{t[0]}_{t[1]}_{t[2]}"
best_head = max([(T, c, f) for T in ("0.25", "0.5") for (c, f) in HEAD], key=lambda t: res["cells"][nm(t)]["budgets"]["0.25"]["ratio"])  # best headline 25% cell, bootstrapped too once triggered
res["bootstrap_trigger"] = {"cells_ge_1.03_at_25pct": [nm(t) for t in trig], "bootstrap_cell": (nm(boot_cell) if boot_cell else None),
                            "bootstrap_cells": ([nm(t) for t in dict.fromkeys([boot_cell, best_head])] if boot_cell else [])}

# bootstrap CI (any boot_*.json present)
res["bootstrap"] = {}
for T in ("0.25", "0.5"):
    for c in CELLS:
        for f in FEATS:
            key = f"T{T}_{c}_{f}"; p = os.path.join(run_dir, f"boot_{key}.json")
            if not os.path.exists(p): continue
            B = json.load(open(p))["resamples"]; lp = res["cells"][key]["budgets"]["0.25"]["learned_warm_policy"]
            thr = np.array([np.mean(r["agree_thr"]) for r in B]); lw = np.array([np.mean(r[lp]) for r in B])
            bo = np.array([max(np.mean(r["greedy_warm"]), np.mean(r["ts_warm"])) for r in B]); rat = lw / thr; rat_bo = bo / thr
            res["bootstrap"][key] = {"n_resamples": len(B), "seeds_per_resample": len(B[0]["agree_thr"]), "policy": lp,
                "ratio_mean": float(rat.mean()), "ratio_ci95": [float(np.percentile(rat, 2.5)), float(np.percentile(rat, 97.5))], "frac_ratio_gt_1": float((rat > 1).mean()),
                "gap_mean": float((lw - thr).mean()), "gap_ci95": [float(np.percentile(lw - thr, 2.5)), float(np.percentile(lw - thr, 97.5))],
                "best_of_ratio_mean": float(rat_bo.mean()), "best_of_ratio_ci95": [float(np.percentile(rat_bo, 2.5)), float(np.percentile(rat_bo, 97.5))],
                "mean_pool_errors": float(np.mean([r["n_err"] for r in B])), "ci_excludes_1": bool(np.percentile(rat, 2.5) > 1.0)}

# verdict
hv = {(T, c, f, bf): res["cells"][f"T{T}_{c}_{f}"]["budgets"][str(bf)] for T in TS for (c, f) in HEAD for bf in BUDS}
max_ratio_head = max(v["ratio"] for v in hv.values()); argmax_head = max(hv, key=lambda k: hv[k]["ratio"])
cands = [(T, c, f) for T in ("0.25", "0.5") for (c, f) in HEAD if hv[(T, c, f, 0.25)]["ratio"] >= 1.05 and hv[(T, c, f, 0.25)]["gap_gt_2sd"]]
boot_ok = [t for t in cands if res["bootstrap"].get(f"T{t[0]}_{t[1]}_{t[2]}", {}).get("ci_excludes_1")]
if not res["sanity"]["ok"]: verdict, why = "INVALID", "sanity failure: " + "; ".join(res["sanity"]["failures"][:5])
elif max_ratio_head <= 1.02: verdict, why = "KILL", f"max headline ratio {max_ratio_head:.3f} <= 1.02 at every budget and temperature"
elif boot_ok: verdict, why = "PASS", f"headline 25% cell(s) {['T%s_%s_%s' % t for t in boot_ok]} ratio >= 1.05, gap > 2 paired SD, bootstrap CI excludes 1.0"
elif cands: verdict, why = "NOT_A_PASS_PENDING_BOOTSTRAP", f"ratio >= 1.05 and gap > 2 SD at {['T%s_%s_%s' % t for t in cands]} but the bootstrap CI does not (or did not yet) exclude 1.0"
else:
    m25 = max(hv[(T, c, f, 0.25)]["ratio"] for T in ("0.25", "0.5") for (c, f) in HEAD)
    verdict = "MARGINAL" if m25 > 1.02 else "NEITHER_PASS_NOR_KILL"
    why = (f"no headline 25% cell at T=0.25/0.5 reaches ratio 1.05 (best {m25:.3f}); max headline ratio over all 18 headline cells is {max_ratio_head:.3f} at "
           f"T{argmax_head[0]} {argmax_head[1]} {argmax_head[2]} {int(argmax_head[3]*100)}%, so the kill (<= 1.02 everywhere) also does not hold")
res["verdict"] = {"label": verdict, "reason": why, "max_headline_ratio": max_ratio_head, "max_headline_ratio_cell": [argmax_head[0], argmax_head[1], argmax_head[2], argmax_head[3]],
                  "pass_candidates_before_bootstrap": [f"T{t[0]}_{t[1]}_{t[2]}" for t in cands]}
# sensitivity: best ratio over all 36 cells, per budget over headline
res["summary"] = {"max_ratio_all_36_cells": max(v["ratio"] for c in res["cells"].values() for v in c["budgets"].values()),
                  "headline_ratio_range_by_budget": {str(bf): [min(hv[(T, c, f, bf)]["ratio"] for T in TS for (c, f) in HEAD), max(hv[(T, c, f, bf)]["ratio"] for T in TS for (c, f) in HEAD)] for bf in BUDS},
                  "warm_minus_cold_headline": {str(bf): {f"T{T}_{c}_{f}": hv[(T, c, f, bf)]["warm_minus_cold"] for T in TS for (c, f) in HEAD} for bf in (0.1, 0.25)}}
json.dump(res, open(os.path.join(run_dir, "result.json"), "w"), indent=1)

L = ["# L1a warm-start replay: table", "",
     f"Pool: {res['cells']['T0.25_cell100_cheap']['n_sim']} groups (cell100) / {res['cells']['T0.25_cell30_cheap']['n_sim']} (cell30), 8 seeds, K=1. Counts are errors caught (mean over seeds). "
     "learned_warm = better of greedy_warm and ts_warm by mean; ratio = learned_warm / agree_thr; gap SD = SD (ddof=1) of the paired per-seed difference of the chosen policy.", "",
     f"## Verdict: {verdict}", "", why, "",
     f"Sanity: {'OK' if res['sanity']['ok'] else 'FAILED'} ({res['sanity']['n_seed_vector_comparisons']} per-seed vector comparisons over {res['sanity']['n_cells']} cells x 3 budgets). In all 36 cell-budget combinations greedy_warm_norefit equals agree_thr per seed, and uniform, agree_thr, oracle, greedy_online, ts_online equal the E21 sim JSONs per seed." if res["sanity"]["ok"] else "Sanity FAILED: " + "; ".join(res["sanity"]["failures"]), "",
     "## Main table (budget 10/25/50%)", "",
     "| T | cell | feats | budget | audits | uniform | threshold | greedy_warm | ts_warm | learned_warm (policy) | ratio | gap | paired SD | gap/SD | cold learned | cold ratio | warm minus cold | oracle | H |", "|" + "---|" * 19]
for T in TS:
    for c in CELLS:
        for f in FEATS:
            for bf in BUDS:
                v = res["cells"][f"T{T}_{c}_{f}"]["budgets"][str(bf)]
                L.append(f"| {T} | {c} | {f} | {bf:.0%} | {v['budget_n']} | {v['uniform']:.0f} | {v['agree_thr']:.1f} | {v['greedy_warm']:.1f} | {v['ts_warm']:.1f} | {v['learned_warm']:.1f} ({v['learned_warm_policy'][:-5]}) | "
                         f"{v['ratio']:.3f} | {v['gap']:+.1f} | {v['gap_paired_sd']:.2f} | {fmt(v['gap_in_sd'])} | {v['learned_cold']:.1f} ({v['learned_cold_policy'][:-7]}) | {v['ratio_cold']:.3f} | {v['warm_minus_cold']:+.1f} | {v['oracle']:.0f} | {v['H']:.2f} |")
L += ["", "## Cold-start decomposition, headline variants (cell100): learned ratio cold (E21) vs warm", "", "| T | feats | 10% cold | 10% warm | 10% warm minus cold (errors) | 25% cold | 25% warm | 25% warm minus cold (errors) | 50% warm |", "|" + "---|" * 9]
for T in TS:
    for (c, f) in HEAD:
        a, b, d = (hv[(T, c, f, bf)] for bf in BUDS)
        L.append(f"| {T} | {f} | {a['ratio_cold']:.3f} | {a['ratio']:.3f} | {a['warm_minus_cold']:+.1f} | {b['ratio_cold']:.3f} | {b['ratio']:.3f} | {b['warm_minus_cold']:+.1f} | {d['ratio']:.3f} |")
L += ["", f"Max ratio over the 36 cell-budget combinations: {res['summary']['max_ratio_all_36_cells']:.3f}. Headline ratio range by budget: " + "; ".join(f"{int(float(k)*100)}%: {v[0]:.3f} to {v[1]:.3f}" for k, v in res['summary']['headline_ratio_range_by_budget'].items()) + ".", ""]
L += [f"Bootstrap trigger (25% cell at T=0.25/0.5 with ratio >= 1.03): {res['bootstrap_trigger']['cells_ge_1.03_at_25pct'] or 'none'}; bootstrap cells: {res['bootstrap_trigger']['bootstrap_cells']}.", ""]
if res["bootstrap"]:
    L += ["## Pool bootstrap (resample the simulated pool with replacement; 25% budget)", "", "| cell | policy | resamples x seeds | mean pool errors | ratio mean | ratio 95% CI | frac ratio > 1 | gap mean | gap 95% CI | best-of ratio mean | best-of 95% CI | CI excludes 1.0 |", "|" + "---|" * 12]
    for k, b in res["bootstrap"].items():
        L.append(f"| {k} | {b['policy']} | {b['n_resamples']} x {b['seeds_per_resample']} | {b['mean_pool_errors']:.0f} | {b['ratio_mean']:.3f} | [{b['ratio_ci95'][0]:.3f}, {b['ratio_ci95'][1]:.3f}] | {b['frac_ratio_gt_1']:.2f} | {b['gap_mean']:+.1f} | [{b['gap_ci95'][0]:+.1f}, {b['gap_ci95'][1]:+.1f}] | {b['best_of_ratio_mean']:.3f} | [{b['best_of_ratio_ci95'][0]:.3f}, {b['best_of_ratio_ci95'][1]:.3f}] | {b['ci_excludes_1']} |")
open(os.path.join(run_dir, "table.md"), "w").write("\n".join(L) + "\n")
print("\n".join(L[:12])); print("verdict:", verdict, "|", why); print("bootstrap cells:", res["bootstrap_trigger"]["bootstrap_cells"])
