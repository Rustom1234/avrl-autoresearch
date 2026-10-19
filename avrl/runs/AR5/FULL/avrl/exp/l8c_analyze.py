"""L8c-A-fresh-tiebreak, CPU analysis (.venv python). PRE-REGISTERED here, before any fresh gold-derived column (pass_rate, majority_correct, proxy_error, judge_error) was read for this experiment.
Question. Does the label-free dissenting-score tie-break (inside the agreement 0.875 boundary tier of the accepted groups: rm by s_oth DESCENDING, Qwen3-4B by s_diff ASCENDING) transfer to 2,000 FRESH GSM8K prompts
(orig pids 2000..3999 of sample(n=4000, random_state=0), disjoint from pids 0..1999) with every rule frozen from the pids 0..1999 results? Null: the in-sample edge (within-tier AUC 0.70 to 0.75, +6 to +9% over agree_acc) is a
winner's-curse artifact of a post hoc scan of 9 features on the same 2,000 prompts.
Stages. --stage build: scores -> feats_J.parquet per proxy (this file). Replay: exp/l8c_boot_forced.py (copy of sim_warm_boot_j4.py, BEST7 forced by env FORCE_BEST7) run by exp/L8c-A-fresh-tiebreak.sh. --stage final: gates, tier AUCs, ratios, verdict.
FROZEN RULES (nothing is refit on the fresh pool).
  Features: s_maj, s_oth, s_diff = s_maj - s_oth, mean_tokens, std_tokens, any_truncated, log_qlen = log1p(q_len), agreement, n_distinct. s_maj / s_oth = proxy score of the majority completion / of the label-free other completion (l5c_proxy_score.py / l3b_judge_score.py rows, unchanged).
  Accept: Qwen3-4B accept = s_maj > 0. rm accept = the K = round(0.8575 x n) = 1,715 groups with the highest s_maj in the fresh pool (exact top-K, stable on ties; the quantile 0.8575 is the frozen pids 0..1999 value, never recomputed from fresh majority-vote accuracy).
  Label J = judge_error = ACCEPTED AND the majority completion is gold-wrong (checked equal to majority_correct == 0 of the scored 'maj' row).
  Boundary tier = accepted groups with agreement == 0.875 (frozen from pids 0..1999, where it is the agreement of the last audited slot of the agree_acc order at 25%). The OPERATIVE tier of the fresh pool (agreement of the last audited slot of the agree_acc key order at budget 500, as sim_warm_boot_j4.pick_tb)
  is computed from label-free columns and reported; if it is not 0.875 the verdict line says so (the verdict AUC still uses 0.875).
  Verdict arms: rm = s_oth (higher = audit earlier) = replay arm agree_acc_soth; Qwen3-4B = minus s_diff (lower s_diff = audit earlier) = agree_acc_sdiff. The other sign / feature and the learner arms (greedy_warm7, greedy_blend7, forced BEST7 = the verdict arm) are context only.
METRICS (fresh pool, per proxy). (1) within-tier AUC of the verdict-arm score for J, point plus 95% percentile CI from 2,000 group-level bootstrap resamples (tier rows resampled with replacement, default_rng(0)), tier group and error counts.
  (2) caught errors at the 25% budget (500 audits, 8 seeds full pool; B = 200 x 4 seeds, default_rng(12345), per-resample seed means) for agree_acc, agree_acc_soth, agree_acc_sdiff, greedy_warm7, greedy_blend7, oracle; ratio verdict-arm / agree_acc, point = ratio of 8-seed full-pool means, 95% percentile CI over the 200 resamples.
  (3) the same at 10%, secondary, never in the verdict. (4) descriptive: greedy_blend7 / tie-break and greedy_blend7 / best fixed (in-sample best of the seven fixed rules on the fresh pool at 25%), percentile CI; power is about 0.04 ratio SD, so a 4.5% learner edge cannot be resolved here (stated up front).
  (5) descriptive, outside the verdict: within-tier AUC of all 9 features (what the post hoc scan would pick on fresh data), and claim 3 out of sample: the frozen pids 0..1999 cell100 shrunk error-rate prior (cell = nearest frozen KMeans centroid of the MiniLM question embedding) as a model-free score on the fresh confident half (accepted, s_maj >= median of accepted).
GATES (any failed gate = FAILED experiment, no verdict). (i) forced-BEST7 replay reproduces runs/L7a-tiebreak/<pool>/boot_tb_f0.25.json exactly (full pool and every resample, all keys of the old file) when run on the pids 0..1999 feats_J of the same proxy;
  (ii) scorer gates all passed incl. sanity AUC of s vs rollout correctness >= 0.70 (gates.json); (iii) at least 25 boundary-tier errors per proxy, else the verdict is UNDERPOWERED (not a failure); (iv) null control: the mean AUC of 1,000 within-tier permutations of the verdict score is within 0.02 of 0.5;
  plus structural: 2,000 scored fresh groups, local pid == arange, majority-row correctness equals majority_correct, replay-script norefit gates (exit codes).
REGISTERED BINS (verdict at 25%).
  PASS (the tie-break replicates out of sample): on BOTH proxies, verdict-arm within-tier AUC point >= 0.62 with bootstrap CI lower end > 0.50, AND verdict-arm tie-break / agree_acc point ratio >= 1.04, all gates passing, >= 25 boundary-tier errors per proxy.
  KILL (the tie-break does not transfer): on BOTH proxies, verdict-arm within-tier AUC point <= 0.57 AND tie-break / agree_acc point ratio <= 1.02, all gates passing, >= 25 boundary-tier errors per proxy.
  UNDERPOWERED: fewer than 25 boundary-tier errors on a proxy. Anything else: INCONCLUSIVE. Prior (judgment): 30% pass, 25% kill, 45% inconclusive.
CAVEATS carried: the two proxies share about 70% of their errors on pids 0..1999 (two correlated tests, not two independent ones); the boundary tier holds only about 30 errors per proxy (AUC SE about 0.05, ratio SD about 0.03 to 0.04); all proxies and the policy are Qwen-family, one policy (1.5B), GSM8K train.
Usage: l8c_analyze.py --run_dir <runs/L8c-A-fresh-tiebreak> --stage build|final"""
import os, sys, json, argparse, time, subprocess
import numpy as np, pandas as pd
from sklearn.metrics import roc_auc_score
ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))); RUNS = os.path.join(ROOT, "avrl", "runs")
ap = argparse.ArgumentParser(); ap.add_argument("--run_dir", required=True); ap.add_argument("--stage", required=True, choices=["build", "final"])
a = ap.parse_args(); R = a.run_dir; t0 = time.time()
FEATS = ["s_maj", "s_oth", "s_diff", "mean_tokens", "std_tokens", "any_truncated", "log_qlen", "agreement", "n_distinct"]
RM_Q = 0.8575; N = 2000; TIER = 0.875; NBOOT_AUC = 2000; NPERM = 1000; SHRINK = 5.0
P = {"rm": dict(name="Skywork rm (top 85.75% of s_maj)", arm="agree_acc_soth", scoref="s_oth", sign=1.0, score_dir=os.path.join(R, "rm", "rm"), old_pool=os.path.join(RUNS, "L5c-proxies", "rm"), old_boot=os.path.join(RUNS, "L7a-tiebreak", "rm")),
     "q3_4b": dict(name="Qwen3-4B Yes/No judge (s_maj > 0)", arm="agree_acc_sdiff", scoref="s_diff", sign=-1.0, score_dir=os.path.join(R, "q3_4b"), old_pool=os.path.join(RUNS, "L3b-judge-proxy"), old_boot=os.path.join(RUNS, "L7a-tiebreak", "L3b-judge-proxy"))}
POOL = lambda k: os.path.join(R, "pool", k); BOOT = lambda k, f: os.path.join(R, "boot", k, f"boot_tb_f{f}.json")
auc = lambda y, s: float(roc_auc_score(y, s)) if 0 < int(np.sum(y)) < len(y) and np.ptp(s) > 0 else float("nan")


def topk_accept(s, K):
    m = np.zeros(len(s), bool); m[np.argsort(-s, kind="stable")[:K]] = True; return m


def build(k):
    p = P[k]; sd = p["score_dir"]; os.makedirs(POOL(k), exist_ok=True)
    if os.path.exists(os.path.join(sd, "failed.txt")): raise SystemExit(f"[{k}] scorer raised:\n" + open(os.path.join(sd, "failed.txt")).read())
    gates = json.load(open(os.path.join(sd, "gates.json")))
    G = pd.read_parquet(os.path.join(R, "rollouts", "T0.25", "groups.parquet")).sort_values("pid").reset_index(drop=True); assert len(G) == N and (G.pid.values == np.arange(N)).all()
    q4 = pd.read_parquet(os.path.join(ROOT, "avrl", "data", "gsm8k_train.parquet")).sample(n=4000, random_state=0).reset_index(drop=True).question.values[2000:4000]; assert (G.question.values == q4).all() and (G.orig_pid.values == 2000 + np.arange(N)).all()
    S = pd.read_parquet(os.path.join(sd, "scores.parquet")); M = S[S.role == "maj"].set_index("pid"); O = S[S.role == "oth"].set_index("pid"); pids = np.arange(N)
    assert len(M) == N and len(O) == N and (np.sort(M.index.values) == pids).all(), "all 2,000 fresh groups must be scored"
    F = G.copy(); F["log_qlen"] = np.log1p(F.q_len); F["split"] = (F.agreement < 1).astype(int); F["cell100"] = np.load(os.path.join(R, "cells", "cell100_fresh.npy"))
    F["s_maj"] = M.loc[pids, "s"].values; F["s_oth"] = O.loc[pids, "s"].values; F["s_diff"] = F.s_maj - F.s_oth
    assert (M.loc[pids, "correct"].values == F.majority_correct.values).all(), "majority completion correctness must equal majority_correct"; assert np.isfinite(F[["s_maj", "s_oth"]].values).all()
    acc = (F.s_maj.values > 0) if k == "q3_4b" else topk_accept(F.s_maj.values, int(round(RM_Q * N)))
    F["accept"] = acc.astype(int); F["judge_error"] = (acc & (F.majority_correct.values == 0)).astype(int)
    F.to_parquet(os.path.join(POOL(k), "feats_J.parquet"), index=False)
    ac = F.accept.values == 1; tier = ac & np.isclose(F.agreement.values, TIER)
    print(f"[{k}] fresh pool {N}, accepted {int(ac.sum())}, tier(0.875) groups {int(tier.sum())}, J errors {int(F.judge_error.sum())} (majority-vote wrong {int(F.proxy_error.sum())}), tier errors {int(F.judge_error.values[tier].sum())}, scorer all_pass {gates['all_pass']}, sanity AUC {gates['4_sanity_auc']['auc_s_vs_correct']:.4f}", flush=True)


def operative_tier(F, frac=0.25):
    """Same as sim_warm_boot_j4.pick_tb tier: agreement of the group in the last audited slot of the agree_acc key order (accepted by agreement asc, n_distinct desc) at budget int(frac * n)."""
    ac, ag, nd = F.accept.values.astype(bool), F.agreement.values, F.n_distinct.values; idx = np.flatnonzero(ac); o = idx[np.lexsort((-nd[idx], ag[idx]))]; b = int(frac * len(F)); last = o[min(b, len(o)) - 1]
    return float(ag[last]), ac & (ag == ag[last])


def tier_stats(F, k, seed=0):
    """Within-tier AUCs on the frozen tier (accepted & agreement == 0.875): verdict score with group-bootstrap CI, other score, 9-feature table, permutation null."""
    p = P[k]; ac = F.accept.values == 1; m = ac & np.isclose(F.agreement.values, TIER); y = F.judge_error.values[m].astype(int); sc = p["sign"] * F[p["scoref"]].values[m]
    pt = auc(y, sc); rng = np.random.default_rng(seed); n = len(y); bs = []
    for _ in range(NBOOT_AUC):
        i = rng.integers(0, n, n)
        if 0 < y[i].sum() < n: bs.append(auc(y[i], sc[i]))
    bs = np.array(bs); other = {"s_oth_desc": auc(y, F.s_oth.values[m]), "minus_s_diff": auc(y, -F.s_diff.values[m])}
    nl = np.array([auc(y, rng.permutation(sc)) for _ in range(NPERM)]); op_ag, opm = operative_tier(F)
    return {"tier_agreement": TIER, "n_tier": int(m.sum()), "n_err_tier": int(y.sum()), "verdict_score": ("s_oth" if p["sign"] > 0 else "minus s_diff"), "auc": pt, "auc_ci95_boot": [float(np.percentile(bs, 2.5)), float(np.percentile(bs, 97.5))], "n_boot_used": int(len(bs)), "auc_boot_sd": float(bs.std(ddof=1)),
            "auc_other_scores": other, "null_perm_auc_mean": float(nl.mean()), "null_perm_auc_sd": float(nl.std(ddof=1)), "null_perm_auc_p975": float(np.percentile(nl, 97.5)),
            "feature_aucs_higher_is_error": {f: auc(y, F[f].values[m]) for f in FEATS}, "operative_tier_agreement": op_ag, "operative_tier_n": int(opm.sum()), "operative_tier_n_err": int(F.judge_error.values[opm].sum()), "operative_tier_is_0.875": bool(np.isclose(op_ag, TIER)),
            "operative_tier_auc_verdict_score": auc(F.judge_error.values[opm].astype(int), p["sign"] * F[p["scoref"]].values[opm]) if not np.isclose(op_ag, TIER) else pt,
            "audit_slots_in_0.875_tier_at_25pct": int(min(max(int(0.25 * len(F)) - int((ac & (F.agreement.values < TIER - 1e-9)).sum()), 0), int(m.sum()))), "audit_slots_in_operative_tier_at_25pct": int(min(max(int(0.25 * len(F)) - int((ac & (F.agreement.values < op_ag - 1e-9)).sum()), 0), int(opm.sum()))),
            "n_acc": int(ac.sum()), "n_J_total": int(F.judge_error.sum()), "n_majority_vote_wrong": int(F.proxy_error.sum()), "n_tier_all_agreement_0.875": int(np.isclose(F.agreement.values, TIER).sum())}


def ci_of(mean, full, num, den):
    r = mean[num] / np.maximum(mean[den], 1e-9); plug = full[num] / max(full[den], 1e-9); q = np.percentile(r, [2.5, 97.5])
    return {"plug_in": float(plug), "boot_mean": float(r.mean()), "boot_sd": float(r.std(ddof=1)), "percentile_ci95": [float(q[0]), float(q[1])], "frac_above_1": float((r > 1).mean())}


FIX7 = ("agree_thr", "judge_thr", "blend50", "agree_acc", "blend50_acc", "agree_acc_soth", "agree_acc_sdiff")
ARMS = ("uniform",) + FIX7[:5] + ("cell_rerank", "greedy_warm", "greedy_blend", "oracle", "agree_acc_soth", "agree_acc_sdiff", "greedy_warm7", "greedy_blend7")


def boot_stats(path, arm):
    bj = json.load(open(path)); fp, rs = bj["full_pool"], bj["resamples"]; mean = {q: np.array([np.mean(r[q]) for r in rs]) for q in ARMS}; full = {q: float(np.mean(fp[q])) for q in ARMS}; ci = lambda n, d: ci_of(mean, full, n, d)
    nat = fp["best_fixed7_natural"] if "best_fixed7_natural" in fp else fp["best_fixed7"]
    out = {"file": path, "n_pool": bj["n_pool"], "n_pool_errors": fp["n_err"], "budget": fp["budget"], "n_boot": bj["n_boot"], "seeds_per_resample": bj["seeds_per_resample"], "sanity": bj["sanity"], "best_fixed5": fp["best_fixed"], "best_fixed7_used_for_learner": fp["best_fixed7"], "best_fixed7_natural_fresh": nat,
           "fixed_mean_at_25_7": fp["fixed_mean_at_25_7"], "caught_full_pool_8seed": full, "caught_boot_mean": {q: float(mean[q].mean()) for q in ARMS}, "caught_boot_sd": {q: float(mean[q].std(ddof=1)) for q in ARMS},
           "ratios": {"arm_over_agree_acc": ci(arm, "agree_acc"), "soth_over_agree_acc": ci("agree_acc_soth", "agree_acc"), "sdiff_over_agree_acc": ci("agree_acc_sdiff", "agree_acc"), "greedy_blend7_over_arm": ci("greedy_blend7", arm), "greedy_blend7_over_best_fixed_natural": ci("greedy_blend7", nat),
                      "greedy_blend7_over_agree_acc": ci("greedy_blend7", "agree_acc"), "greedy_warm7_over_arm": ci("greedy_warm7", arm), "best_fixed_natural_over_agree_acc": ci(nat, "agree_acc"), "arm_over_oracle": ci(arm, "oracle")}}
    out["gate_flags_ok"] = bool(all(v is True for kk, v in bj["sanity"].items() if kk.startswith("gate_")))
    return out


def repro(new_path, old_path):
    """Gate (i): every key of the old (L7a) file's full_pool and of every old resample equals the forced-BEST7 run on the same pids 0..1999 pool."""
    new, old = json.load(open(new_path)), json.load(open(old_path)); top = ("n_pool", "accept_col", "label", "features", "budget_frac", "seeds_per_resample", "n_boot")
    ok_top = all(new[q] == old[q] for q in top); ok_full = all(new["full_pool"].get(q) == v for q, v in old["full_pool"].items()); ok_res = len(new["resamples"]) == len(old["resamples"]) and all(rn.get(q) == v for rn, ro in zip(new["resamples"], old["resamples"]) for q, v in ro.items())
    return {"pass": bool(ok_top and ok_full and ok_res), "top_equal": bool(ok_top), "full_pool_equal": bool(ok_full), "resamples_equal": bool(ok_res), "n_resamples": len(old["resamples"]), "reference": old_path}


def prior_oos(Fo, Ff):
    """Claim 3 out of sample (descriptive): frozen pids 0..1999 cell100 shrunk error rate (errors among old accepted members of the cell + 5 x base) / (old accepted members + 5) as a model-free score on the fresh confident half (accepted, s_maj >= median of fresh accepted)."""
    A = Fo[Fo.accept == 1]; base = float(A.judge_error.mean()); cnt = A.groupby("cell100").size(); err = A.groupby("cell100").judge_error.sum(); rate = ((err.reindex(range(100)).fillna(0) + SHRINK * base) / (cnt.reindex(range(100)).fillna(0) + SHRINK)).values
    acc = Ff.accept.values == 1; conf = acc & (Ff.s_maj.values >= np.median(Ff.s_maj.values[acc])); y = Ff.judge_error.values[conf].astype(int); sc = rate[Ff.cell100.values[conf]]; pt = auc(y, sc); rng = np.random.default_rng(0); bs = []
    for _ in range(1000):
        i = rng.integers(0, len(y), len(y))
        if 0 < y[i].sum() < len(y) and np.ptp(sc[i]) > 0: bs.append(auc(y[i], sc[i]))
    # old-pool in-sample counterpart (leave-one-out within the cell, same shrinkage), confident half of the old pool
    ao = Fo.accept.values == 1; confo = ao & (Fo.s_maj.values >= np.median(Fo.s_maj.values[ao])); cell = Fo.cell100.values; yo = Fo.judge_error.values; ce = np.bincount(cell[ao], weights=yo[ao], minlength=100); cn = np.bincount(cell[ao], minlength=100); bo = float(yo[ao].mean())
    loo = (ce[cell] - yo + SHRINK * bo) / (cn[cell] - 1 + SHRINK)
    return {"n_conf_fresh": int(conf.sum()), "n_err_conf_fresh": int(y.sum()), "auc_fresh_frozen_prior": pt, "auc_ci95_boot": [float(np.percentile(bs, 2.5)), float(np.percentile(bs, 97.5))] if bs else None, "old_pool_conf_half_loo_auc_in_sample": auc(yo[confo].astype(int), loo[confo]),
            "n_conf_old": int(confo.sum()), "n_err_conf_old": int(yo[confo].sum()), "base_rate_old_acc": base}


if a.stage == "build":
    for k in P: build(k)
    print(f"build done in {time.time()-t0:.0f}s", flush=True); sys.exit(0)

# ---- final stage ----
res = {"experiment": "L8c-A-fresh-tiebreak", "pools": {}}; gate_msgs = []
for k, p in P.items():
    Ff = pd.read_parquet(os.path.join(POOL(k), "feats_J.parquet")); Fo = pd.read_parquet(os.path.join(p["old_pool"], "feats_J.parquet")); x = {"name": p["name"], "verdict_arm": p["arm"]}
    sg = json.load(open(os.path.join(p["score_dir"], "gates.json"))); x["scorer_gates"] = {"all_pass": sg["all_pass"], "sanity_auc_s_vs_correct": sg["4_sanity_auc"]["auc_s_vs_correct"], "sanity_threshold": 0.70, "pass_2": sg["2_unbatched"]["pass"], "pass_3": sg["3_finite_mass"]["pass"], "subsampled": sg["1_timing"].get("subsampled")}
    x["gate_ii_scorer"] = bool(sg["all_pass"] and sg["4_sanity_auc"]["auc_s_vs_correct"] >= 0.70 and not sg["1_timing"].get("subsampled"))
    if not x["gate_ii_scorer"]: gate_msgs.append(f"{k}: gate (ii) scorer gates failed or subsampled {json.dumps(x['scorer_gates'])}")
    x["fresh"] = tier_stats(Ff, k); x["old"] = tier_stats(Fo, k); x["fresh"]["n_J_per_accepted"] = float(Ff.judge_error.sum() / max(Ff.accept.sum(), 1)); x["old"]["n_J_per_accepted"] = float(Fo.judge_error.sum() / max(Fo.accept.sum(), 1))
    x["gate_iii_tier_errors_ge_25"] = bool(x["fresh"]["n_err_tier"] >= 25); x["gate_iv_null_perm"] = bool(abs(x["fresh"]["null_perm_auc_mean"] - 0.5) <= 0.02)
    if not x["gate_iv_null_perm"]: gate_msgs.append(f"{k}: gate (iv) null control mean AUC {x['fresh']['null_perm_auc_mean']:.4f}")
    gi = repro(os.path.join(R, "gate_i", k, "boot_f0.25.json"), os.path.join(p["old_boot"], "boot_tb_f0.25.json")) if os.path.exists(os.path.join(R, "gate_i", k, "boot_f0.25.json")) else {"pass": False, "note": "missing gate (i) run"}
    x["gate_i_repro"] = gi
    if not gi["pass"]: gate_msgs.append(f"{k}: gate (i) forced-BEST7 reproduction of L7a failed {json.dumps(gi)}")
    x["main"] = {}
    for f in ("0.25", "0.1"):
        if os.path.exists(BOOT(k, f)):
            x["main"][f] = boot_stats(BOOT(k, f), p["arm"])
            if not x["main"][f]["gate_flags_ok"]: gate_msgs.append(f"{k} f{f}: replay sanity gate flag false {json.dumps(x['main'][f]['sanity'])}")
            if x["main"][f]["n_pool"] != N: gate_msgs.append(f"{k} f{f}: replay pool {x['main'][f]['n_pool']} != {N} (cell filter dropped groups)")
        else: gate_msgs.append(f"{k} f{f}: no replay file")
    # old pool replay numbers computed with the same function (the L7a file stores no best_fixed7_natural; best_fixed7 is natural there)
    x["old_main"] = {"0.25": boot_stats(os.path.join(p["old_boot"], "boot_tb_f0.25.json"), p["arm"]), "0.1": boot_stats(os.path.join(p["old_boot"], "boot_tb_f0.1.json"), p["arm"])}
    x["prior_oos"] = prior_oos(Fo, Ff); res["pools"][k] = x

# verdict
v = {}
for k, x in res["pools"].items():
    m = x["main"].get("0.25"); r = m["ratios"]["arm_over_agree_acc"] if m else None; fr = x["fresh"]
    v[k] = {"auc": fr["auc"], "auc_ci_lo": fr["auc_ci95_boot"][0], "ratio": r["plug_in"] if r else None, "n_err_tier": fr["n_err_tier"],
            "pass_arm": bool(r is not None and fr["auc"] >= 0.62 and fr["auc_ci95_boot"][0] > 0.50 and r["plug_in"] >= 1.04), "kill_arm": bool(r is not None and fr["auc"] <= 0.57 and r["plug_in"] <= 1.02)}
under = [k for k in v if v[k]["n_err_tier"] < 25]; flags = [f"operative tier of {k} is agreement {res['pools'][k]['fresh']['operative_tier_agreement']}" for k in v if not res["pools"][k]["fresh"]["operative_tier_is_0.875"]]
if gate_msgs: verdict = "FAILED (gate): " + " | ".join(gate_msgs)
elif under: verdict = f"UNDERPOWERED (fewer than 25 boundary-tier errors on {under}: " + ", ".join(f"{k} {v[k]['n_err_tier']}" for k in under) + ")"
elif all(v[k]["pass_arm"] for k in v): verdict = "PASS (the tie-break replicates out of sample on both proxies)"
elif all(v[k]["kill_arm"] for k in v): verdict = "KILL (the tie-break does not transfer on either proxy)"
else: verdict = "INCONCLUSIVE"
if flags and not gate_msgs: verdict += "; NOTE: " + "; ".join(flags)
# context only, NOT the verdict: what the registered PASS / KILL bins give if the >= 25 boundary-tier-error requirement (gate iii) were waived (gates i, ii, iv, pool size and the replay flags must still hold); and the first fresh replay run that dropped 2 groups through the min-3 cell filter
other_gates = [m for m in gate_msgs]
res["reading_if_gate_iii_waived"] = ("FAILED (other gate)" if other_gates else "PASS" if all(v[k]["pass_arm"] for k in v) else "KILL" if all(v[k]["kill_arm"] for k in v) else "INCONCLUSIVE")
res["first_run_1998_groups_sensitivity"] = {}
for k in P:
    fp_ = os.path.join(R, "boot_cellfilter1998_first_run", k, "boot_tb_f0.25.json")
    if os.path.exists(fp_): bs_ = boot_stats(fp_, P[k]["arm"]); res["first_run_1998_groups_sensitivity"][k] = {"n_pool": bs_["n_pool"], "budget": bs_["budget"], "arm_over_agree_acc": bs_["ratios"]["arm_over_agree_acc"], "caught_full_pool_8seed": {q: bs_["caught_full_pool_8seed"][q] for q in ("agree_acc", "agree_acc_soth", "agree_acc_sdiff", "greedy_blend7", "oracle")}}
res["verdict"] = verdict; res["verdict_inputs"] = v; res["gate_messages"] = gate_msgs
# compact key numbers
res["key_numbers"] = {k: {"tier_n": x["fresh"]["n_tier"], "tier_err": x["fresh"]["n_err_tier"], "auc": x["fresh"]["auc"], "auc_ci95": x["fresh"]["auc_ci95_boot"], "old_auc": x["old"]["auc"], "old_auc_ci95": x["old"]["auc_ci95_boot"],
                           "ratio_25": x["main"]["0.25"]["ratios"]["arm_over_agree_acc"]["plug_in"] if "0.25" in x["main"] else None, "ratio_25_ci95": x["main"]["0.25"]["ratios"]["arm_over_agree_acc"]["percentile_ci95"] if "0.25" in x["main"] else None,
                           "old_ratio_25": x["old_main"]["0.25"]["ratios"]["arm_over_agree_acc"]["plug_in"], "caught_25": {q: x["main"]["0.25"]["caught_full_pool_8seed"][q] for q in ("agree_acc", "agree_acc_soth", "agree_acc_sdiff", "greedy_warm7", "greedy_blend7", "oracle")} if "0.25" in x["main"] else None,
                           "n_J_fresh": x["fresh"]["n_J_total"], "n_J_old": x["old"]["n_J_total"]} for k, x in res["pools"].items()}
res["wall_s"] = time.time() - t0; json.dump(res, open(os.path.join(R, "result.json"), "w"), indent=1, default=float)

f3 = lambda z: "nan" if z is None or (isinstance(z, float) and np.isnan(z)) else f"{z:.3f}"; L = [verdict, f"[context only, not the verdict] registered bins with gate (iii) waived: {res['reading_if_gate_iii_waived']}", "[sensitivity] first fresh replay on 1998 groups (min-3 cell filter dropped 2): " + "; ".join(f"{k} n_pool {z['n_pool']} budget {z['budget']} tie-break/agree_acc {z['arm_over_agree_acc']['plug_in']:.3f} CI {z['arm_over_agree_acc']['percentile_ci95']}" for k, z in res["first_run_1998_groups_sensitivity"].items()), ""]
for k, x in res["pools"].items():
    fr, od = x["fresh"], x["old"]; L += [f"## {x['name']} (verdict arm {x['verdict_arm']})", f"scorer gates {json.dumps(x['scorer_gates'])}; gate (i) repro {x['gate_i_repro']['pass']}; gate (iii) {x['gate_iii_tier_errors_ge_25']}; gate (iv) null mean AUC {fr['null_perm_auc_mean']:.4f} (SD {fr['null_perm_auc_sd']:.3f})"]
    for tag, t in (("fresh 2000..3999", fr), ("old 0..1999", od)):
        L.append(f"[{tag}] accepted {t['n_acc']}, J {t['n_J_total']} (mv-wrong {t['n_majority_vote_wrong']}), tier(0.875 accepted) {t['n_tier']} groups / {t['n_err_tier']} errors; verdict AUC {t['auc']:.3f} CI [{t['auc_ci95_boot'][0]:.3f}, {t['auc_ci95_boot'][1]:.3f}] (boot SD {t['auc_boot_sd']:.3f}); audit slots at 25% inside the 0.875 tier {t['audit_slots_in_0.875_tier_at_25pct']} (operative tier {t['audit_slots_in_operative_tier_at_25pct']}); s_oth AUC {f3(t['auc_other_scores']['s_oth_desc'])}, -s_diff AUC {f3(t['auc_other_scores']['minus_s_diff'])}; operative tier {t['operative_tier_agreement']} ({t['operative_tier_n']} groups, {t['operative_tier_n_err']} errors, verdict-score AUC {f3(t['operative_tier_auc_verdict_score'])})")
        L.append("   9-feature within-tier AUC (higher = error): " + ", ".join(f"{q} {f3(z)}" for q, z in t["feature_aucs_higher_is_error"].items()))
    for f in ("0.25", "0.1"):
        for tag, mm in (("fresh", x["main"].get(f)), ("old", x["old_main"][f])):
            if not mm: continue
            c, sd = mm["caught_full_pool_8seed"], mm["caught_boot_sd"]; L.append(f"[{tag}, {int(float(f)*100)}% budget {mm['budget']}] pool errors {mm['n_pool_errors']}; best fixed5 {mm['best_fixed5']}, natural best of 7 {mm['best_fixed7_natural_fresh']}, learner warm-start {mm['best_fixed7_used_for_learner']}; caught 8 seeds (boot SD): " + "; ".join(f"{q} {c[q]:.1f} ({sd[q]:.1f})" for q in ("uniform", "agree_acc", "agree_acc_soth", "agree_acc_sdiff", "greedy_warm7", "greedy_blend7", "oracle")))
            L.append("   fixed means at 25% (7 rules): " + ", ".join(f"{q} {z:.1f}" for q, z in mm["fixed_mean_at_25_7"].items()))
            for nm, r in mm["ratios"].items(): L.append(f"   {nm}: plug-in {r['plug_in']:.3f}, boot mean {r['boot_mean']:.3f} (SD {r['boot_sd']:.3f}), percentile CI [{r['percentile_ci95'][0]:.3f}, {r['percentile_ci95'][1]:.3f}]")
    po = x["prior_oos"]; L += [f"[claim 3 out of sample, descriptive] frozen cell100 shrunk-rate prior on the fresh confident half: AUC {f3(po['auc_fresh_frozen_prior'])} CI {po['auc_ci95_boot']} ({po['n_err_conf_fresh']} errors in {po['n_conf_fresh']}); old pool confident half leave-one-out in-sample AUC {f3(po['old_pool_conf_half_loo_auc_in_sample'])} ({po['n_err_conf_old']} errors in {po['n_conf_old']})", ""]
open(os.path.join(R, "table.txt"), "w").write("\n".join(L)); print("\n".join(L), flush=True)
