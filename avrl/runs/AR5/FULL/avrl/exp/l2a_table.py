"""L2a driver: read the 8 pool-bootstrap JSONs from sim_warm_boot.py, write table.md and result.json.
Usage: l2a_table.py <run_dir>
Per cell (T, cell, budget fraction): per resample, ratio = mean over seeds of policy hits / mean over seeds of agree_thr hits on the SAME resampled pool (paired);
percentile 95% CI over resamples; fraction of resamples with ratio > 1; paired gap in errors (policy minus threshold, mean over seeds per resample) with percentile CI.
Pre-registered verdict (L2a spec), primary cells = cell100 cheap, 10% budget, T=0.25 and T=0.5, policy greedy_warm:
  class of a cell: above (CI lower end > 1.0), below (CI upper end < 1.0), includes (otherwise).
  PASS  : both primary cells are 'above' and both sanity gates held in every job.
  KILL  : neither primary cell is 'above' (CI includes 1.0 at both, or lies below 1.0): the 10% edge is pool noise, L1a converts to a kill for claim 2 at T<=0.5.
  MIXED : exactly one primary cell is 'above' (logged inconclusive, temperature specific).
  Extra kill-side read at T=1.0, 10%: greedy_warm CI upper end < 1.0 and greedy_blend does not fix it (blend CI lower end <= 1.0 or blend mean ratio < 1.0)."""
import os, sys, json
import numpy as np
run_dir = sys.argv[1]
SPECS = [("0.25", "cell100", "0.1"), ("0.5", "cell100", "0.1"), ("1.0", "cell100", "0.1"), ("0.25", "cell100", "0.25"), ("0.5", "cell100", "0.25"), ("1.0", "cell100", "0.25"),
         ("0.25", "cell30", "0.1"), ("0.5", "cell30", "0.1")]
PRIMARY = [("0.25", "cell100", "0.1"), ("0.5", "cell100", "0.1")]
POLS = ("greedy_warm", "greedy_blend")
key = lambda t: f"T{t[0]}_{t[1]}_f{t[2]}"
ci = lambda x: [float(np.percentile(x, 2.5)), float(np.percentile(x, 97.5))]
res = {"cells": {}, "sanity": {"ok": True, "failures": []}}
for t in SPECS:
    d = json.load(open(os.path.join(run_dir, f"boot_{key(t)}.json"))); B = d["resamples"]
    san = d["sanity"]
    if not (san["gate_norefit_equals_threshold_first20"] and san["gate_fullpool_reproduces_L1a"]): res["sanity"]["ok"] = False; res["sanity"]["failures"].append(f"{key(t)}: {san}")
    thr = np.array([np.mean(r["agree_thr"]) for r in B]); c = {"n_resamples": len(B), "seeds_per_resample": d["seeds_per_resample"], "n_sim": d["n_sim"], "budget_n": B[0]["budget"],
        "mean_pool_errors": float(np.mean([r["n_err"] for r in B])), "agree_thr_mean": float(thr.mean()),
        "full_pool": {"agree_thr": float(np.mean(d["full_pool"]["agree_thr"])), "greedy_warm": float(np.mean(d["full_pool"]["greedy_warm"])), "greedy_blend": float(np.mean(d["full_pool"]["greedy_blend"])),
                      "ratio_greedy_warm": d["full_pool"]["ratio_greedy_warm"], "ratio_greedy_blend": d["full_pool"]["ratio_greedy_blend"], "L1a_ratio_greedy_warm": d["full_pool"]["ref_ratio_greedy_warm"]},
        "gates": san}
    for p in POLS:
        v = np.array([np.mean(r[p]) for r in B]); rat = v / thr; gap = v - thr; lo, hi = ci(rat)
        c[p] = {"ratio_mean": float(rat.mean()), "ratio_ci95": [lo, hi], "frac_ratio_gt_1": float((rat > 1).mean()), "gap_mean": float(gap.mean()), "gap_ci95": ci(gap),
                "frac_gap_gt_0": float((gap > 0).mean()), "mean_hits": float(v.mean()), "cls": "above" if lo > 1.0 else ("below" if hi < 1.0 else "includes")}
    bw = np.array([np.mean(r["greedy_blend"]) - np.mean(r["greedy_warm"]) for r in B]); c["blend_minus_warm_gap"] = {"mean": float(bw.mean()), "ci95": ci(bw)}
    res["cells"][key(t)] = c

prim = {key(t): res["cells"][key(t)]["greedy_warm"] for t in PRIMARY}; n_above = sum(v["cls"] == "above" for v in prim.values())
if not res["sanity"]["ok"]: label, why = "INVALID", "sanity gate failure: " + "; ".join(res["sanity"]["failures"][:3])
elif n_above == 2: label, why = "PASS", "greedy_warm / agree_thr CI lower end > 1.0 at the 10% budget in cell100 cheap at both T=0.25 and T=0.5, both gates intact: the edge survives pool resampling"
elif n_above == 0:
    below = [k for k, v in prim.items() if v["cls"] == "below"]
    label, why = "KILL", ("the 95% CI of greedy_warm / agree_thr includes 1.0 at the 10% budget in cell100 cheap at both T=0.25 and T=0.5" if not below else
                          f"neither primary CI has its lower end above 1.0 and {below} lies entirely below 1.0 (stronger than the pre-registered kill)") + ": the 10% edge is pool noise"
else:
    above = [k for k, v in prim.items() if v["cls"] == "above"]; label, why = "INCONCLUSIVE_MIXED", f"exactly one primary CI excludes 1.0 from above ({above}): temperature specific, not a pass and not a kill"
res["verdict"] = {"label": label, "reason": why, "primary": {k: {"ratio_mean": v["ratio_mean"], "ratio_ci95": v["ratio_ci95"], "cls": v["cls"]} for k, v in prim.items()}}

t1 = res["cells"][key(("1.0", "cell100", "0.1"))]; w, b = t1["greedy_warm"], t1["greedy_blend"]
fixes = bool(b["ratio_ci95"][0] > 1.0 and b["ratio_mean"] >= 1.0)
res["t1_10pct_read"] = {"warm_ci95": w["ratio_ci95"], "warm_ratio_mean": w["ratio_mean"], "blend_ci95": b["ratio_ci95"], "blend_ratio_mean": b["ratio_mean"],
    "warm_ci_upper_below_1": bool(w["ratio_ci95"][1] < 1.0), "blend_fixes": fixes,
    "kill_side_claim2_false_at_high_T": bool(w["ratio_ci95"][1] < 1.0 and not fixes),
    "blend_removes_deficit_loose": bool(b["ratio_ci95"][1] >= 1.0 and b["ratio_mean"] >= 1.0)}
json.dump(res, open(os.path.join(run_dir, "result.json"), "w"), indent=1)

f3 = lambda x: f"{x:.3f}"; f1 = lambda x: f"{x:+.1f}"
L = ["# L2a pool bootstrap of the 10% budget: table", "",
     f"Pool bootstrap: resample the simulated pool (rows) with replacement, {res['cells'][key(SPECS[0])]['n_resamples']} resamples x {res['cells'][key(SPECS[0])]['seeds_per_resample']} seeds each, budget = int(frac * pool size), "
     "K=1. ratio = (mean over seeds of policy errors caught) / (mean over seeds of agree_thr errors caught) on the same resampled pool; CI = percentile 2.5/97.5 over resamples; "
     "gap = paired difference in errors caught (policy minus threshold). greedy_blend = greedy_warm with the refit score replaced by 0.5 * rank(warm score) + 0.5 * rank(refit probability).", "",
     f"## Verdict: {label}", "", why, "",
     f"Sanity gates: {'both held in all 8 jobs' if res['sanity']['ok'] else 'FAILED'} (refit-disabled greedy_warm equals agree_thr per seed on the first 20 resamples; unresampled full-pool 8-seed agree_thr and greedy_warm equal the L1a seed vectors exactly).", "",
     "## Primary cells (greedy_warm / agree_thr, cell100 cheap, 10% budget)", "", "| T | pool errors (mean) | audits | threshold hits | greedy_warm hits | ratio mean | ratio 95% CI | frac ratio > 1 | gap mean | gap 95% CI | full-pool ratio (8 seeds) | class |", "|" + "---|" * 12]
for t in PRIMARY:
    c = res["cells"][key(t)]; w = c["greedy_warm"]
    L.append(f"| {t[0]} | {c['mean_pool_errors']:.0f} | {c['budget_n']} | {c['agree_thr_mean']:.1f} | {w['mean_hits']:.1f} | {f3(w['ratio_mean'])} | [{f3(w['ratio_ci95'][0])}, {f3(w['ratio_ci95'][1])}] | {w['frac_ratio_gt_1']:.2f} | "
             f"{f1(w['gap_mean'])} | [{f1(w['gap_ci95'][0])}, {f1(w['gap_ci95'][1])}] | {f3(c['full_pool']['ratio_greedy_warm'])} | {w['cls']} |")
L += ["", "## All 8 cells, both policies", "", "| T | cell | budget | audits | pool errors | policy | threshold hits | policy hits | ratio mean | ratio 95% CI | frac ratio > 1 | gap mean | gap 95% CI | frac gap > 0 | full-pool ratio | class |", "|" + "---|" * 16]
for t in SPECS:
    c = res["cells"][key(t)]
    for p in POLS:
        x = c[p]
        L.append(f"| {t[0]} | {t[1]} | {float(t[2]):.0%} | {c['budget_n']} | {c['mean_pool_errors']:.0f} | {p} | {c['agree_thr_mean']:.1f} | {x['mean_hits']:.1f} | {f3(x['ratio_mean'])} | [{f3(x['ratio_ci95'][0])}, {f3(x['ratio_ci95'][1])}] | "
                 f"{x['frac_ratio_gt_1']:.2f} | {f1(x['gap_mean'])} | [{f1(x['gap_ci95'][0])}, {f1(x['gap_ci95'][1])}] | {x['frac_gap_gt_0']:.2f} | {f3(c['full_pool']['ratio_'+p])} | {x['cls']} |")
r = res["t1_10pct_read"]
L += ["", "## Blend minus warm (paired gap in errors, greedy_blend minus greedy_warm)", "", "| T | cell | budget | gap mean | gap 95% CI |", "|---|---|---|---|---|"]
for t in SPECS:
    g = res["cells"][key(t)]["blend_minus_warm_gap"]; L.append(f"| {t[0]} | {t[1]} | {float(t[2]):.0%} | {f1(g['mean'])} | [{f1(g['ci95'][0])}, {f1(g['ci95'][1])}] |")
L += ["", "## Secondary read, T=1.0, 10%, cell100 cheap", "",
      f"greedy_warm ratio mean {f3(r['warm_ratio_mean'])}, CI [{f3(r['warm_ci95'][0])}, {f3(r['warm_ci95'][1])}] (upper end below 1.0: {r['warm_ci_upper_below_1']}); "
      f"greedy_blend ratio mean {f3(r['blend_ratio_mean'])}, CI [{f3(r['blend_ci95'][0])}, {f3(r['blend_ci95'][1])}]; blend fixes it (CI lower end > 1.0 and mean >= 1.0): {r['blend_fixes']}. "
      f"Kill-side flag (claim 2 as worded false at high temperature): {r['kill_side_claim2_false_at_high_T']}.", ""]
open(os.path.join(run_dir, "table.md"), "w").write("\n".join(L) + "\n")
print("\n".join(L[:20])); print("verdict:", label, "|", why)
