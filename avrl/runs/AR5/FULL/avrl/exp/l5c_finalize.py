"""L5c stage 3 (CPU): the experiment verdict and the claim 2 replay table. PRE-REGISTERED with l5c_analyze.py (bins are quoted there and in l5c_proxy_score.py).
Reads <run_dir>/score_status.json, <run_dir>/{rm,judge3b}/result_analyze.json (per-proxy `lead`, or status failed), feats_J.parquet of both proxies and of L3b (Qwen3-4B, pairwise Jaccard of the J-error sets under each
proxy's PRIMARY accept rule), and <run_dir>/{rm,judge3b,qwen3_4b_ref}/boot_f0.25.json (sim_warm_boot_j2.py; the qwen3_4b_ref pool is the L3b feats_J.parquet, reference only).
Verdict at a cut (top50 cell100 is THE registered cut; top25, top75 and all are recomputed with the same rule): PASS = both proxies passed gates, n_err in ACC >= 100 for both, both REPLICATES (ratio >= 1.5 and > own
null p95) and delta(JA_emb - JA) >= 0.03 with bootstrap CI lower end > 0 for at least one; KILL = both passed gates with n_err in ACC >= 100 and both KILL-SIDE (ratio < 1.3 or <= own null p95); a proxy that failed to
load or failed a gate = partial, inconclusive; anything else inconclusive (split when exactly one replicates). The verdict line says so when the class (PASS / KILL / INCONCLUSIVE) at the registered cut differs from the
class at two or more of the other three cuts. Claim 2 (secondary, never changes the verdict): per pool the percentile 95% CI (verdict interval) and the basic CI of greedy_warm / best fixed and greedy_blend / best fixed
over the bootstrap resamples, headroom H = (oracle - best fixed) / oracle, errors caught at the 25% budget for every policy. Writes <run_dir>/result.json and replay_table.txt."""
import os, sys, json, argparse
import numpy as np, pandas as pd
ap = argparse.ArgumentParser(); ap.add_argument("--run_dir", required=True); ap.add_argument("--l3b", default=None); a = ap.parse_args(); Q = a.run_dir
ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))); L3B = a.l3b or os.path.join(ROOT, "avrl", "runs", "L3b-judge-proxy")
KINDS = ("rm", "judge3b"); CUTS = ("top50", "top25", "top75", "all"); NAMES = {"rm": "Skywork-Reward-V2-Qwen3-0.6B (reward model)", "judge3b": "Qwen2.5-3B-Instruct Yes/No judge", "qwen3_4b_ref": "Qwen3-4B Yes/No judge (L3b, reference)"}
FIXED = ("agree_thr", "judge_thr", "blend50", "agree_acc", "blend50_acc"); ALL = ("uniform",) + FIXED + ("greedy_warm", "greedy_blend", "oracle")
def load(p): return json.load(open(p)) if os.path.exists(p) else None
status = load(os.path.join(Q, "score_status.json")); A = {k: load(os.path.join(Q, k, "result_analyze.json")) for k in KINDS}

def klass(v): return "PASS" if v.startswith("PASS") else ("KILL" if v.startswith("KILL") else "INCONCLUSIVE")

def verdict(cut):
    L = {k: (A[k] or {}).get("lead") for k in KINDS}; bad = [k for k in KINDS if A[k] is None or A[k].get("status") != "ok"]
    if bad: return "INCONCLUSIVE (partial: " + ", ".join(f"{k} failed gates or did not load" for k in bad) + ")"
    cc = {k: L[k]["by_cut"][cut] if "by_cut" in L[k] else {"replicates": False, "kill_side": False} for k in KINDS}; pw = all(L[k]["n_err_ge_100"] for k in KINDS)
    rep, kill = [cc[k]["replicates"] for k in KINDS], [cc[k]["kill_side"] for k in KINDS]; dok = any(L[k]["delta_ok"] for k in KINDS)
    if pw and all(rep) and dok: return "PASS (claim 3 lead replicates in both proxies, strong form)"
    if pw and all(kill): return "KILL (claim 3 closed: cell100 ratio below 1.3 or at/below own null p95 for both proxies at this cut)"
    if not pw: return "INCONCLUSIVE (underpowered: n_err in ACC < 100 for " + ", ".join(k for k in KINDS if not L[k]["n_err_ge_100"]) + ")"
    if sum(rep) == 1: return "INCONCLUSIVE (split: only " + [k for k, r in zip(KINDS, rep) if r][0] + " replicates)"
    if all(rep): return "INCONCLUSIVE (ratio replicates in both proxies, delta AUC route not met)"
    return "INCONCLUSIVE (not both kill-side, not both replicating)"

by_cut = {c: verdict(c) for c in CUTS}; main = by_cut["top50"]; diff = [c for c in CUTS[1:] if klass(by_cut[c]) != klass(main)]
line = f"L5c-proxies claim 3: {main}"
if len(diff) >= 2: line += f" [the verdict flips with the cut: the registered top-half cut gives {klass(main)}, but {', '.join(f'{c} gives {klass(by_cut[c])}' for c in diff)}]"
elif diff: line += f" [one other cut differs: {diff[0]} gives {klass(by_cut[diff[0]])}]"
else: line += " [same class at all four cuts]"
strict = {}  # strict kill reading (ratio < 1.3 only) at the registered cut, reported beside the literal one
if all(A[k] and A[k].get("status") == "ok" and "by_cut" in A[k]["lead"] for k in KINDS): strict = {k: A[k]["lead"]["by_cut"]["top50"]["kill_side_strict"] for k in KINDS}

# pairwise Jaccard of J-error sets (primary rule of each proxy)
sets = {}
for k, p in (("qwen3_4b", os.path.join(L3B, "feats_J.parquet")), ("rm", os.path.join(Q, "rm", "feats_J.parquet")), ("judge3b", os.path.join(Q, "judge3b", "feats_J.parquet"))):
    if os.path.exists(p): f = pd.read_parquet(p); sets[k] = set(f.pid.values[f.judge_error.values == 1])
pj = {"sizes": {k: len(v) for k, v in sets.items()}, "pairs": {}}
ks = list(sets)
for i in range(len(ks)):
    for j in range(i + 1, len(ks)): x, y = sets[ks[i]], sets[ks[j]]; pj["pairs"][f"{ks[i]}_vs_{ks[j]}"] = {"intersection": len(x & y), "union": len(x | y), "jaccard": len(x & y) / max(len(x | y), 1)}
if len(ks) == 3: pj["triple_intersection"] = len(sets[ks[0]] & sets[ks[1]] & sets[ks[2]]); pj["triple_union"] = len(sets[ks[0]] | sets[ks[1]] | sets[ks[2]])

def replay(path):
    bj = json.load(open(path)); fp = bj["full_pool"]; best = fp["best_fixed"]; rs = bj["resamples"]; mean = {p: np.array([np.mean(r[p]) for r in rs]) for p in ALL}
    full = {p: float(np.mean(fp[p])) for p in ALL}; out = {"file": path, "pool_mode": bj["pool_mode"], "n_pool": bj["n_pool"], "n_pool_errors": fp["n_err"], "budget_frac": bj["budget_frac"], "budget": fp["budget"], "n_boot": bj["n_boot"],
                                                       "seeds_per_resample": bj["seeds_per_resample"], "sanity": bj["sanity"], "best_fixed": best, "fixed_mean_at_25_full_pool": fp["fixed_mean_at_25"], "caught_full_pool_8seed": full,
                                                       "caught_boot_mean": {p: float(mean[p].mean()) for p in ALL}, "caught_boot_sd": {p: float(mean[p].std(ddof=1)) for p in ALL}}
    def ci(num, den):
        r = mean[num] / np.maximum(mean[den], 1e-9); plug = full[num] / max(full[den], 1e-9); q = np.percentile(r, [2.5, 97.5])
        return {"plug_in": float(plug), "boot_mean": float(r.mean()), "boot_sd": float(r.std(ddof=1)), "percentile_ci95": [float(q[0]), float(q[1])], "basic_ci95": [float(2 * plug - q[1]), float(2 * plug - q[0])], "frac_above_1": float((r > 1).mean())}
    out["ratio_greedy_warm_over_best_fixed"] = ci("greedy_warm", best); out["ratio_greedy_blend_over_best_fixed"] = ci("greedy_blend", best); out["ratio_greedy_warm_over"] = {p: ci("greedy_warm", p) for p in FIXED}
    out["H_full_pool"] = float((full["oracle"] - full[best]) / full["oracle"]); out["H_boot_mean"] = float(((mean["oracle"] - mean[best]) / mean["oracle"]).mean())
    return out

R2 = {}
for k in KINDS + ("qwen3_4b_ref",):
    p = os.path.join(Q, k, "boot_f0.25.json")
    if os.path.exists(p): R2[k] = replay(p)
l3b_res = load(os.path.join(L3B, "result_analyze.json")); qref = None
if l3b_res and "primary_tau0" in l3b_res:
    pt = l3b_res["primary_tau0"]; qref = {"n_err": pt["n_err"], "nhr_top50_cell100": pt["nhr_top_half"]["cell100"], "delta_JA": pt["deltas"]["JA_emb_minus_JA"], "jaccard": pt["jaccard_vs_majority_vote_wrong"]}
res = {"experiment": "L5c-proxies", "verdict_line": line, "verdict_registered_cut_top50": main, "verdict_by_cut": by_cut, "strict_kill_reading_ratio_lt_1.3_at_top50": strict, "score_status": status,
       "claim3": {k: (A[k]["lead"] if A[k] and A[k].get("status") == "ok" else (A[k] or {"status": "no result_analyze.json"})) for k in KINDS}, "qwen3_4b_reference_L3b_nhr_cut_values_from_critic2": {"top50": 2.093, "top25": 3.382, "top75": 1.411, "all": 1.066},
       "qwen3_4b_reference_L3b": qref, "pairwise_jaccard_J_errors": pj, "claim2_secondary_replay": R2,
       "full_analysis": {k: (A[k] if A[k] and A[k].get("status") == "ok" else None) for k in KINDS}}
json.dump(res, open(os.path.join(Q, "result.json"), "w"), indent=1, default=float)

L = [line, ""]
for k in KINDS:
    x = res["claim3"][k]
    if "by_cut" not in x: L += [f"## {k}: {x.get('message', x.get('status', 'no neighbor-ratio result'))[:600]}", ""]; continue
    L += [f"## {NAMES[k]}: sanity AUC {x['sanity_auc']:.4f}, ACC {x['n_acc']}, n_err {x['n_err_acc']} (top half {x['n_err_top_half']}), Jaccard vs majority-vote-wrong {x['jaccard_vs_majority_vote_wrong']:.3f}, vs Qwen3-4B J {x['jaccard_J_vs_qwen3_4b_J']:.3f}",
          f"top-half cell100 ratio {x['nhr_top_half_cell100']:.3f}, null mean {x['null_mean']:.3f}, null p95 {x['null_p95']:.3f}, cell-bootstrap 95% CI [{x['boot_ci95'][0]:.2f}, {x['boot_ci95'][1]:.2f}]; replicates {x['replicates']}, kill-side {x['kill_side']}",
          f"delta AUC (JA + MiniLM - JA) {x['delta_auc']:+.4f}, bootstrap 95% CI [{x['delta_boot_ci95'][0]:+.4f}, {x['delta_boot_ci95'][1]:+.4f}]", "", "| cut | n_err | cell100 ratio | null p95 | replicates | kill-side |", "|---|---|---|---|---|---|"]
    L += [f"| {c} | {v['n_err']} | {v['ratio']:.3f} | {v['null_p95']:.3f} | {v['replicates']} | {v['kill_side']} |" for c, v in x["by_cut"].items()] + [""]
L += ["verdict by cut: " + json.dumps(by_cut), "pairwise Jaccard of J-error sets: " + json.dumps(pj), ""]
for k, r in R2.items():
    bf = r["best_fixed"]; L += [f"## claim 2 replay, {NAMES[k]}: pool {r['n_pool']}, {r['n_pool_errors']} J errors, budget {r['budget']}, {r['n_boot']} resamples x {r['seeds_per_resample']} seeds, best fixed = {bf}, fixed means at 25% {json.dumps({q: round(v, 1) for q, v in r['fixed_mean_at_25_full_pool'].items()})}", "",
                                "| policy | caught full pool (8 seeds) | caught bootstrap mean (SD) |", "|---|---|---|"]
    L += [f"| {p} | {r['caught_full_pool_8seed'][p]:.1f} | {r['caught_boot_mean'][p]:.1f} ({r['caught_boot_sd'][p]:.1f}) |" for p in ALL]
    for nm in ("ratio_greedy_warm_over_best_fixed", "ratio_greedy_blend_over_best_fixed"):
        x = r[nm]; L.append(f"{nm}: plug-in {x['plug_in']:.3f}, boot mean {x['boot_mean']:.3f} (SD {x['boot_sd']:.3f}), percentile CI [{x['percentile_ci95'][0]:.3f}, {x['percentile_ci95'][1]:.3f}] (verdict interval), basic CI [{x['basic_ci95'][0]:.3f}, {x['basic_ci95'][1]:.3f}], frac above 1 {x['frac_above_1']:.2f}")
    L += [f"H = (oracle - best fixed) / oracle: full pool {r['H_full_pool']:.3f}, bootstrap mean {r['H_boot_mean']:.3f}", ""]
open(os.path.join(Q, "replay_table.txt"), "w").write("\n".join(L)); print("\n".join(L))
