"""L9c-weak-policy-tiebreak, CPU analysis (.venv python). Copy of exp/l8c_analyze.py with the design change below. PRE-REGISTERED here, before any weak-policy gold-derived column (pass_rate, majority_correct, proxy_error, judge_error, rollout correct)
or any rollout summary.json was read for this experiment (GPU job A was running when this text was written; the only files looked at were the L8c-A scripts, results and logs of the 1.5B policy).
Question. Is the label-free dissenting-score tie-break (inside the partially audited cell of the agree_acc order: Qwen3-4B Yes/No judge by minus s_diff, Skywork rm by s_oth) a property of the proxy-minority-completion mechanism, or specific to the strong policy
(Qwen2.5-1.5B-Instruct, few and mostly arithmetic-slip errors)? Weak policy here: Qwen2.5-0.5B-Instruct on the SAME 2,000 fresh GSM8K prompts as L8c-A (orig pids 2000..3999 of sample(n=4000, random_state=0), local pid 0..1999), T=0.25, G=8, max_tokens 512, seed 0.
Null: the L8c-A signal (within-tier AUC 0.75 to 0.78, +2.6% to +3.5% gain over agree_acc) vanishes when the error mix changes. Every rule is frozen from the 1.5B results; nothing is refit on this pool.
Stages. --stage build: scores -> pool/<proxy>/feats_J.parquet (also computes the frozen cell300 assignment of the fresh prompts for the descriptive claim 3 read). Replay: exp/l8c_boot_forced.py, byte-identical to the L8c-A copy (sha256 recorded in gate_i/l8c_boot_forced.sha256), run by exp/L9c-weak-policy-tiebreak.sh. --stage final: gates, operative-cell AUCs, ratios, verdict.
FROZEN RULES.
  Features: s_maj, s_oth, s_diff = s_maj - s_oth, mean_tokens, std_tokens, any_truncated, log_qlen = log1p(q_len), agreement, n_distinct (as L8c-A). Frozen arms (replay, env FORCE_BEST7, MIN_CELL=1, no reselection, no cross-fit): Qwen3-4B -> agree_acc_sdiff (score = minus s_diff, lower s_diff = audit earlier);
  rm -> agree_acc_soth (score = s_oth, higher = audit earlier). The learner arms (greedy_warm7, greedy_blend7) are warm-started from the frozen arm and are context only.
  ACCEPT RULES (both label-free). Qwen3-4B: accept = s_maj > 0. rm: accept = the top-K groups by s_maj on this pool (stable on ties) with K = the number of groups the Qwen3-4B rule accepts on the SAME pool, i.e. the same accepted FRACTION; it is NOT the 85.75% quantile of the 1.5B pool.
  Label J = judge_error = ACCEPTED AND the majority completion is gold-wrong (checked equal to majority_correct == 0 of the scored 'maj' row).
DESIGN CHANGE FROM L8c-A (registered because L8c-A's recorder found the fixed 0.875 tier was the wrong object, finding (a) and (d) in the L8c-A log entry). The OPERATIVE CELL of a proxy and budget is the (agreement, n_distinct) cell of the group in audit slot number b = int(frac x 2000) (the 500th slot at 25%) of the agree_acc key order
  (accepted groups by agreement ascending, then n_distinct descending; rejected groups last). Confirmed in sim_warm_boot_j4.py / l8c_boot_forced.py orders(): tb(score) = np.lexsort((u6, np.where(acc, score, u4), np.where(acc, -ndist, 0), np.where(acc, agree, 0), (~acc).astype(int))), so the dissenting-score order reorders groups only inside a block of equal
  (agreement, n_distinct) among the accepted; every block fully audited or never reached is order-invariant, so the gain at a budget comes from the one partially audited block, which is exactly the operative cell. It is computed from label-free columns only (accept, agreement, n_distinct). The within-cell AUC, the error-count gate and the ratio all refer to it.
  The old 0.875 agreement tier is dropped; the AUC over the whole agreement tier of the operative cell (accepted groups with the same agreement, any n_distinct) is reported beside it, descriptive. A structural check (order_check) re-implements the lexsort above and verifies on this pool that the first b slots of the tie-break order and of the agree_acc order are the same blocks up to the operative cell.
METRICS (per proxy, 2,000-group weak-policy pool). (1) operative-cell size, groups before it, audit slots inside it, judge-error count, within-cell AUC of the verdict score for J with 95% percentile CI from 2,000 group-level bootstrap resamples of the cell rows (default_rng(0)); the same at 10% (descriptive).
  (2) caught errors at 25% (500 audits; 8 seeds on the full pool; B = 200 resamples x 4 seeds, default_rng(12345); per-resample seed means; percentile CI) for uniform, agree_acc, agree_acc_soth, agree_acc_sdiff, greedy_warm7, greedy_blend7, oracle; ratio verdict arm / agree_acc, point = ratio of 8-seed full-pool means (the verdict statistic) with its percentile CI; the same at 10% (secondary).
  (3) context against L8c-A: weak-policy majority-vote error rate (group_proxy_error_rate) and rollout accuracy against 0.144 and 0.779, accepted count, truncated rollout fraction, J against 186 (rm) and 192 (Qwen3-4B).
  (4) DESCRIPTIVE, OUTSIDE THE VERDICT (E06 regime-variation branch): greedy_blend7 / best fixed rule (natural in-sample best of the seven fixed rules at 25%, claim 2) and H25 = (oracle - best fixed) / oracle with errors caught per audit (the oracle may be budget-saturated when J exceeds 500); the claim 3 cell-rate AUC gain on the confident half
  from exp/l6a_cell_rate.py run UNCHANGED on pool/q3_4b/feats_J.parquet (CLI: --e21 pointing to a missing path so its pids 0..1999 E21 reproduction gate is skipped, --consensus the two pools of this experiment); and the 9-feature within-operative-cell AUC table.
GATES (any failed gate = FAILED experiment, no verdict). (i) forced-BEST7 replay on the pids 0..1999 feats_J of each proxy reproduces runs/L7a-tiebreak/<pool>/boot_tb_f0.25.json (all keys of the old file, full pool and every resample); also compared with the L8c-A gate_i copy.
  (ii) scorer gates.json all_pass and sanity AUC of s vs rollout correctness >= 0.70 for each proxy, no subsampling. (iii) REGIME gate: weak-policy majority-vote error rate >= 1.5 x 0.144 = 0.216 AND truncated rollout fraction < 0.25 (else the weak policy is degenerate and the experiment is FAILED, not a verdict).
  (iv) null control: the mean AUC of 1,000 within-operative-cell permutations of the verdict score is within 0.02 of 0.5, per proxy. Structural: 2,000 scored groups, local pid == arange, questions equal the L8c-A questions (same prompts, so the frozen-centroid cells apply), majority-row correctness equals majority_correct, replay n_pool == 2,000 and the replay sanity flags true, order_check true, cell reassignment check true.
  Error-count gate (not a failure): the Qwen3-4B operative cell must hold >= 60 judge errors, else UNDERPOWERED.
REGISTERED BINS (verdict at 25%, on Qwen3-4B; rm is secondary).
  PASS (the tie-break transfers across error regimes): Qwen3-4B operative-cell AUC point >= 0.62 AND bootstrap CI lower end > 0.50 AND tie-break / agree_acc point ratio >= 1.04 at 25% AND >= 60 judge errors in the operative cell, all gates passing. rm must show a positive plug-in gain at 25% (ratio > 1): reported as a flag, NOT a pass condition.
  KILL (the mechanism does not transfer to a weak policy): Qwen3-4B operative-cell AUC point <= 0.57 AND ratio <= 1.02 at 25%, with >= 60 judge errors in the operative cell and all gates passing. UNDERPOWERED: fewer than 60 operative-cell judge errors on Qwen3-4B (not a kill). Anything else: INCONCLUSIVE.
  Prior (judgment): 35% pass, 20% kill, 45% inconclusive.
CAVEATS carried: both proxies are Qwen-family and the policy is Qwen2.5-0.5B (a different size, same family and the same prompts); the two proxies share most of their errors (two correlated tests); the ratio SD is about 0.02 to 0.04 at a single 25% budget and the gain lives in one cell, so a 4% effect is about 1 to 2 SD;
the operative cell is a different object from L8c-A's 0.875 tier (it is the cell of the 500th slot of this pool and proxy), so L8c-A numbers are context, not a paired comparison; the weak policy may change what the proxy accepts, so the accepted count and J are reported before any ratio is read.
Usage: l9c_analyze.py --run_dir <runs/L9c-weak-policy-tiebreak> --stage build|final"""
import os, sys, json, argparse, time
import numpy as np, pandas as pd
from sklearn.metrics import roc_auc_score
ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))); RUNS = os.path.join(ROOT, "avrl", "runs")
ap = argparse.ArgumentParser(); ap.add_argument("--run_dir", required=True); ap.add_argument("--stage", required=True, choices=["build", "final"])
a = ap.parse_args(); R = a.run_dir; t0 = time.time()
L8C = os.path.join(RUNS, "L8c-A-fresh-tiebreak")
FEATS = ["s_maj", "s_oth", "s_diff", "mean_tokens", "std_tokens", "any_truncated", "log_qlen", "agreement", "n_distinct"]
N = 2000; NBOOT_AUC = 2000; NPERM = 1000; MV_ERR_L8C = 0.144; MIN_CELL_ERR = 60; REGIME_MULT = 1.5; TRUNC_MAX = 0.25
P = {"q3_4b": dict(name="Qwen3-4B Yes/No judge (s_maj > 0)", arm="agree_acc_sdiff", scoref="s_diff", sign=-1.0, score_dir=os.path.join(R, "q3_4b"), old_pool=os.path.join(RUNS, "L3b-judge-proxy"), old_boot=os.path.join(RUNS, "L7a-tiebreak", "L3b-judge-proxy")),
     "rm": dict(name="Skywork rm (top fraction of s_maj = Qwen3-4B accepted fraction)", arm="agree_acc_soth", scoref="s_oth", sign=1.0, score_dir=os.path.join(R, "rm", "rm"), old_pool=os.path.join(RUNS, "L5c-proxies", "rm"), old_boot=os.path.join(RUNS, "L7a-tiebreak", "rm"))}
POOL = lambda k: os.path.join(R, "pool", k); BOOT = lambda k, f: os.path.join(R, "boot", k, f"boot_tb_f{f}.json")
auc = lambda y, s: float(roc_auc_score(y, s)) if 0 < int(np.sum(y)) < len(y) and np.ptp(s) > 0 else float("nan")


def topk_accept(s, K):
    m = np.zeros(len(s), bool); m[np.argsort(-s, kind="stable")[:K]] = True; return m


def ensure_cells():
    """Frozen KMeans(k, n_init=3, random_state=0) refit on the stored pids 0..1999 MiniLM embeddings (as p0_analyze.py / e21_lowT_analyze.py), fresh prompts assigned by predict. Checks: refit reproduces the stored old cell100 / cell300 exactly, predict reproduces L8c-A's cell100_fresh.npy exactly. Same prompts as L8c-A, so no new embedding."""
    out = os.path.join(R, "cells"); os.makedirs(out, exist_ok=True); p3 = os.path.join(out, "cell300_fresh.npy")
    if os.path.exists(p3): return np.load(os.path.join(L8C, "cells", "cell100_fresh.npy")), np.load(p3)
    from sklearn.cluster import KMeans
    eo = np.load(os.path.join(RUNS, "E21-lowT-confident-errors", "q_emb_minilm.npy")); ef = np.load(os.path.join(L8C, "cells", "q_emb_minilm_fresh.npy")); Fo = pd.read_parquet(os.path.join(P["q3_4b"]["old_pool"], "feats_J.parquet")).sort_values("pid").reset_index(drop=True)
    res = {}; c100f = np.load(os.path.join(L8C, "cells", "cell100_fresh.npy"))
    for k, old, ref in ((100, Fo.cell100.values, c100f), (300, Fo.cell300.values, None)):
        km = KMeans(n_clusters=k, n_init=3, random_state=0).fit(eo); fresh = km.predict(ef); res[f"cell{k}"] = {"refit_equals_stored_old": bool((km.labels_ == old).all()), "predict_old_equals_labels": bool((km.predict(eo) == km.labels_).all())}
        if ref is not None: res[f"cell{k}"]["predict_fresh_equals_L8c_A_file"] = bool((fresh == ref).all())
        else: np.save(p3, fresh)
    json.dump(res, open(os.path.join(out, "cells_check.json"), "w"), indent=1); assert all(all(v.values()) for v in res.values()), f"cell reassignment check failed {res}"
    return c100f, np.load(p3)


def build(k):
    p = P[k]; sd = p["score_dir"]; os.makedirs(POOL(k), exist_ok=True)
    if os.path.exists(os.path.join(sd, "failed.txt")): raise SystemExit(f"[{k}] scorer raised:\n" + open(os.path.join(sd, "failed.txt")).read())
    gates = json.load(open(os.path.join(sd, "gates.json")))
    G = pd.read_parquet(os.path.join(R, "rollouts", "T0.25", "groups.parquet")).sort_values("pid").reset_index(drop=True); assert len(G) == N and (G.pid.values == np.arange(N)).all()
    q4 = pd.read_parquet(os.path.join(ROOT, "avrl", "data", "gsm8k_train.parquet")).sample(n=4000, random_state=0).reset_index(drop=True).question.values[2000:4000]; assert (G.question.values == q4).all() and (G.orig_pid.values == 2000 + np.arange(N)).all()
    G8 = pd.read_parquet(os.path.join(L8C, "rollouts", "T0.25", "groups.parquet")).sort_values("pid").reset_index(drop=True); assert (G.question.values == G8.question.values).all(), "same prompts as L8c-A (frozen-centroid cells valid)"
    S = pd.read_parquet(os.path.join(sd, "scores.parquet")); M = S[S.role == "maj"].set_index("pid"); O = S[S.role == "oth"].set_index("pid"); pids = np.arange(N)
    assert len(M) == N and len(O) == N and (np.sort(M.index.values) == pids).all(), "all 2,000 groups must be scored"
    c100, c300 = ensure_cells(); F = G.copy(); F["log_qlen"] = np.log1p(F.q_len); F["split"] = (F.agreement < 1).astype(int); F["cell100"] = c100; F["cell300"] = c300
    F["s_maj"] = M.loc[pids, "s"].values; F["s_oth"] = O.loc[pids, "s"].values; F["s_diff"] = F.s_maj - F.s_oth
    assert (M.loc[pids, "correct"].values == F.majority_correct.values).all(), "majority completion correctness must equal majority_correct"; assert np.isfinite(F[["s_maj", "s_oth"]].values).all()
    if k == "q3_4b": acc = F.s_maj.values > 0
    else: Kq = int(pd.read_parquet(os.path.join(POOL("q3_4b"), "feats_J.parquet")).accept.sum()); acc = topk_accept(F.s_maj.values, Kq)
    F["accept"] = acc.astype(int); F["judge_error"] = (acc & (F.majority_correct.values == 0)).astype(int); F.to_parquet(os.path.join(POOL(k), "feats_J.parquet"), index=False)
    print(f"[{k}] weak pool {N}, accepted {int(acc.sum())}, J errors {int(F.judge_error.sum())} (majority-vote wrong {int(F.proxy_error.sum())}), scorer all_pass {gates['all_pass']}, sanity AUC {gates['4_sanity_auc']['auc_s_vs_correct']:.4f}", flush=True)


def operative_cell(F, frac):
    """(agreement, n_distinct) block of the group in slot b = int(frac * n) of the agree_acc key order, from label-free columns. Returns dict with masks."""
    ac, ag, nd = F.accept.values.astype(bool), F.agreement.values, F.n_distinct.values; idx = np.flatnonzero(ac); o = idx[np.lexsort((-nd[idx], ag[idx]))]; b = int(frac * len(F)); last = o[min(b, len(o)) - 1]
    cell = ac & np.isclose(ag, ag[last]) & (nd == nd[last]); before = ac & ((ag < ag[last] - 1e-9) | (np.isclose(ag, ag[last]) & (nd > nd[last])))
    return {"agreement": float(ag[last]), "n_distinct": int(nd[last]), "cell": cell, "before": before, "budget": b, "slots": int(min(max(b - int(before.sum()), 0), int(cell.sum()))), "agreement_tier": ac & np.isclose(ag, ag[last])}


def cell_stats(F, k, frac, seed=0):
    """Within-operative-cell AUCs: verdict score with group-bootstrap CI, other score, 9 features, permutation null, tie-break found versus expected errors in the cell."""
    p = P[k]; oc = operative_cell(F, frac); m = oc["cell"]; y = F.judge_error.values[m].astype(int); sc = p["sign"] * F[p["scoref"]].values[m]; pt = auc(y, sc); rng = np.random.default_rng(seed); n = len(y); bs = []
    for _ in range(NBOOT_AUC):
        i = rng.integers(0, n, n)
        if 0 < y[i].sum() < n: bs.append(auc(y[i], sc[i]))
    bs = np.array(bs) if bs else np.array([np.nan]); nl = np.array([auc(y, rng.permutation(sc)) for _ in range(NPERM)]); ta = oc["agreement_tier"]; yt = F.judge_error.values[ta].astype(int)
    o = np.argsort(-sc, kind="stable"); s_ = oc["slots"]; found = int(y[o[:s_]].sum()); expected = float(s_ * y.sum() / max(n, 1))
    return {"frac": frac, "budget": oc["budget"], "cell_agreement": oc["agreement"], "cell_n_distinct": oc["n_distinct"], "n_cell": int(m.sum()), "n_err_cell": int(y.sum()), "n_before_cell": int(oc["before"].sum()), "audit_slots_in_cell": oc["slots"], "cell_fully_audited": bool(oc["slots"] >= int(m.sum())),
            "verdict_score": ("s_oth" if p["sign"] > 0 else "minus s_diff"), "auc": pt, "auc_ci95_boot": [float(np.nanpercentile(bs, 2.5)), float(np.nanpercentile(bs, 97.5))], "n_boot_used": int(len(bs)), "auc_boot_sd": float(np.nanstd(bs, ddof=1)),
            "auc_other_scores": {"s_oth_desc": auc(y, F.s_oth.values[m]), "minus_s_diff": auc(y, -F.s_diff.values[m])}, "null_perm_auc_mean": float(nl.mean()), "null_perm_auc_sd": float(nl.std(ddof=1)), "feature_aucs_higher_is_error": {f: auc(y, F[f].values[m]) for f in FEATS},
            "agreement_tier_n": int(ta.sum()), "agreement_tier_n_err": int(yt.sum()), "agreement_tier_auc_verdict_score": auc(yt, p["sign"] * F[p["scoref"]].values[ta]),
            "cell_tiebreak_errors_found_in_slots": found, "cell_random_expected_errors_in_slots": expected, "cell_gain_errors": found - expected, "n_acc": int(F.accept.sum()), "n_J_total": int(F.judge_error.sum()), "n_majority_vote_wrong": int(F.proxy_error.sum())}


def order_check(F, k, seed=0):
    """Re-implementation of the agree_acc and tie-break lexsorts of l8c_boot_forced.orders(): the first b slots of both orders must be the same blocks up to the operative cell, and slot b must sit inside the cell."""
    ac, ag, nd = F.accept.values.astype(bool), F.agreement.values, F.n_distinct.values; rng = np.random.default_rng(seed); u4 = rng.random(len(F)); u6 = rng.random(len(F)); p = P[k]; score = -F.s_oth.values if p["sign"] > 0 else F.s_diff.values
    o_base = np.lexsort((u4, np.where(ac, -nd, 0), np.where(ac, ag, 0), (~ac).astype(int))); o_tb = np.lexsort((u6, np.where(ac, score, u4), np.where(ac, -nd, 0), np.where(ac, ag, 0), (~ac).astype(int)))
    oc = operative_cell(F, 0.25); nb, nc, b = int(oc["before"].sum()), int(oc["cell"].sum()), oc["budget"]; S = lambda o, lo, hi: set(o[lo:hi].tolist())
    ok = S(o_base, 0, nb) == S(o_tb, 0, nb) == set(np.flatnonzero(oc["before"]).tolist()) and S(o_base, nb, nb + nc) == S(o_tb, nb, nb + nc) == set(np.flatnonzero(oc["cell"]).tolist()) and nb < b <= nb + nc
    return bool(ok)


def ci_of(mean, full, num, den):
    r = mean[num] / np.maximum(mean[den], 1e-9); plug = full[num] / max(full[den], 1e-9); q = np.percentile(r, [2.5, 97.5])
    return {"plug_in": float(plug), "boot_mean": float(r.mean()), "boot_sd": float(r.std(ddof=1)), "percentile_ci95": [float(q[0]), float(q[1])], "frac_above_1": float((r > 1).mean())}


FIX7 = ("agree_thr", "judge_thr", "blend50", "agree_acc", "blend50_acc", "agree_acc_soth", "agree_acc_sdiff")
ARMS = ("uniform",) + FIX7[:5] + ("cell_rerank", "greedy_warm", "greedy_blend", "oracle", "agree_acc_soth", "agree_acc_sdiff", "greedy_warm7", "greedy_blend7")


def boot_stats(path, arm):
    bj = json.load(open(path)); fp, rs = bj["full_pool"], bj["resamples"]; mean = {q: np.array([np.mean(r[q]) for r in rs]) for q in ARMS}; full = {q: float(np.mean(fp[q])) for q in ARMS}; ci = lambda n, d: ci_of(mean, full, n, d)
    nat = fp["best_fixed7_natural"]; fm7 = fp["fixed_mean_at_25_7"]; budget = fp["budget"]; best_fixed_caught = full[nat] if budget == fp["budget_25"] else None
    out = {"file": path, "n_pool": bj["n_pool"], "n_pool_errors": fp["n_err"], "budget": budget, "n_boot": bj["n_boot"], "seeds_per_resample": bj["seeds_per_resample"], "sanity": bj["sanity"], "best_fixed5": fp["best_fixed"], "best_fixed7_used_for_learner": fp["best_fixed7"], "best_fixed7_natural": nat,
           "fixed_mean_at_25_7": fm7, "caught_full_pool_8seed": full, "caught_boot_mean": {q: float(mean[q].mean()) for q in ARMS}, "caught_boot_sd": {q: float(mean[q].std(ddof=1)) for q in ARMS},
           "oracle_per_audit": full["oracle"] / budget, "oracle_budget_saturated": bool(fp["n_err"] >= budget),
           "H_oracle_minus_best_fixed_over_oracle": (full["oracle"] - full[nat]) / max(full["oracle"], 1e-9), "H_oracle_minus_agree_acc_over_oracle": (full["oracle"] - full["agree_acc"]) / max(full["oracle"], 1e-9),
           "ratios": {"arm_over_agree_acc": ci(arm, "agree_acc"), "soth_over_agree_acc": ci("agree_acc_soth", "agree_acc"), "sdiff_over_agree_acc": ci("agree_acc_sdiff", "agree_acc"), "greedy_blend7_over_arm": ci("greedy_blend7", arm), "greedy_blend7_over_best_fixed_natural": ci("greedy_blend7", nat),
                      "greedy_blend7_over_agree_acc": ci("greedy_blend7", "agree_acc"), "greedy_warm7_over_arm": ci("greedy_warm7", arm), "best_fixed_natural_over_agree_acc": ci(nat, "agree_acc"), "arm_over_oracle": ci(arm, "oracle")}}
    out["gate_flags_ok"] = bool(all(v is True for kk, v in bj["sanity"].items() if kk.startswith("gate_")))
    return out


def repro(new_path, old_path):
    """Gate (i): every key of the old (L7a) file's full_pool and of every old resample equals the forced-BEST7 run on the same pids 0..1999 pool."""
    new, old = json.load(open(new_path)), json.load(open(old_path)); top = ("n_pool", "accept_col", "label", "features", "budget_frac", "seeds_per_resample", "n_boot")
    ok_top = all(new[q] == old[q] for q in top); ok_full = all(new["full_pool"].get(q) == v for q, v in old["full_pool"].items()); ok_res = len(new["resamples"]) == len(old["resamples"]) and all(rn.get(q) == v for rn, ro in zip(new["resamples"], old["resamples"]) for q, v in ro.items())
    return {"pass": bool(ok_top and ok_full and ok_res), "top_equal": bool(ok_top), "full_pool_equal": bool(ok_full), "resamples_equal": bool(ok_res), "n_resamples": len(old["resamples"]), "reference": old_path}


if a.stage == "build":
    for k in P: build(k)  # q3_4b first: its accepted count sets the rm K
    print(f"build done in {time.time()-t0:.0f}s", flush=True); sys.exit(0)

# ---- final stage ----
res = {"experiment": "L9c-weak-policy-tiebreak", "pools": {}}; gate_msgs = []
rs_ = json.load(open(os.path.join(R, "rollouts", "T0.25", "summary.json"))); rs8 = json.load(open(os.path.join(L8C, "rollouts", "T0.25", "summary.json")))
mv_err = rs_["group_proxy_error_rate"]; trunc = rs_["frac_truncated_rollouts"]
res["regime"] = {"weak_policy": {q: rs_[q] for q in ("rollout_accuracy", "majority_accuracy", "group_proxy_error_rate", "frac_truncated_rollouts", "frac_unanimous", "proxy_error_when_unanimous", "n_confident", "n_conf_err", "wall_s")},
                 "l8c_a_1p5b": {q: rs8[q] for q in ("rollout_accuracy", "majority_accuracy", "group_proxy_error_rate", "frac_truncated_rollouts", "frac_unanimous", "n_confident", "n_conf_err")}, "mv_err_threshold": REGIME_MULT * MV_ERR_L8C, "trunc_threshold": TRUNC_MAX,
                 "mv_err_ratio_to_l8c_a": mv_err / MV_ERR_L8C}
res["regime"]["gate_iii_pass"] = bool(mv_err >= REGIME_MULT * MV_ERR_L8C and trunc < TRUNC_MAX)
if not res["regime"]["gate_iii_pass"]: gate_msgs.append(f"gate (iii) regime: majority-vote error {mv_err:.4f} (needs >= {REGIME_MULT * MV_ERR_L8C:.4f}), truncated fraction {trunc:.4f} (needs < {TRUNC_MAX}): the weak policy is degenerate or not weak enough")
l8 = json.load(open(os.path.join(L8C, "result.json")))["key_numbers"]
for k, p in P.items():
    Ff = pd.read_parquet(os.path.join(POOL(k), "feats_J.parquet")); x = {"name": p["name"], "verdict_arm": p["arm"]}
    sg = json.load(open(os.path.join(p["score_dir"], "gates.json"))); x["scorer_gates"] = {"all_pass": sg["all_pass"], "sanity_auc_s_vs_correct": sg["4_sanity_auc"]["auc_s_vs_correct"], "sanity_threshold": 0.70, "pass_2": sg["2_unbatched"]["pass"], "pass_3": sg["3_finite_mass"]["pass"], "subsampled": sg["1_timing"].get("subsampled")}
    x["gate_ii_scorer"] = bool(sg["all_pass"] and sg["4_sanity_auc"]["auc_s_vs_correct"] >= 0.70 and not sg["1_timing"].get("subsampled"))
    if not x["gate_ii_scorer"]: gate_msgs.append(f"{k}: gate (ii) scorer gates failed or subsampled {json.dumps(x['scorer_gates'])}")
    x["structural"] = {"n_pool": int(len(Ff)), "pid_arange": bool((np.sort(Ff.pid.values) == np.arange(N)).all()), "order_check": order_check(Ff, k)}
    if len(Ff) != N or not x["structural"]["pid_arange"] or not x["structural"]["order_check"]: gate_msgs.append(f"{k}: structural check failed {json.dumps(x['structural'])}")
    x["op25"] = cell_stats(Ff, k, 0.25); x["op10"] = cell_stats(Ff, k, 0.10); fr = x["op25"]
    x["gate_iv_null_perm"] = bool(abs(fr["null_perm_auc_mean"] - 0.5) <= 0.02)
    if not x["gate_iv_null_perm"]: gate_msgs.append(f"{k}: gate (iv) null control mean AUC {fr['null_perm_auc_mean']:.4f}")
    new_gi = os.path.join(R, "gate_i", k, "boot_f0.25.json"); gi = repro(new_gi, os.path.join(p["old_boot"], "boot_tb_f0.25.json")) if os.path.exists(new_gi) else {"pass": False, "note": "missing gate (i) run"}
    l8gi = os.path.join(L8C, "gate_i", k, "boot_f0.25.json")
    if os.path.exists(new_gi) and os.path.exists(l8gi): gi["equals_L8c_A_gate_i_json"] = bool(json.load(open(new_gi)) == json.load(open(l8gi)))
    x["gate_i_repro"] = gi
    if not gi["pass"]: gate_msgs.append(f"{k}: gate (i) forced-BEST7 reproduction of L7a failed {json.dumps(gi)}")
    x["main"] = {}
    for f in ("0.25", "0.1"):
        if os.path.exists(BOOT(k, f)):
            x["main"][f] = boot_stats(BOOT(k, f), p["arm"])
            if not x["main"][f]["gate_flags_ok"]: gate_msgs.append(f"{k} f{f}: replay sanity gate flag false {json.dumps(x['main'][f]['sanity'])}")
            if x["main"][f]["n_pool"] != N: gate_msgs.append(f"{k} f{f}: replay pool {x['main'][f]['n_pool']} != {N} (cell filter dropped groups)")
            if x["main"][f]["n_pool_errors"] != fr["n_J_total"]: gate_msgs.append(f"{k} f{f}: replay pool errors {x['main'][f]['n_pool_errors']} != J {fr['n_J_total']}")
        else: gate_msgs.append(f"{k} f{f}: no replay file")
    # exact expected agree_acc at 25% under a random tie-break inside blocks (descriptive cross-check of the 8-seed baseline)
    ac = Ff.accept.values.astype(bool); yy = Ff.judge_error.values; ag, nd = Ff.agreement.values, Ff.n_distinct.values; exp_ = 0.0; b = int(0.25 * N)
    if b <= int(ac.sum()):
        keys = sorted(set(zip(ag[ac].round(6), nd[ac]))); keys.sort(key=lambda t: (t[0], -t[1])); left = b
        for g_, n_ in keys:
            mm = ac & np.isclose(ag, g_) & (nd == n_); s_ = min(left, int(mm.sum())); exp_ += s_ * yy[mm].mean(); left -= s_
            if left <= 0: break
    x["exact_expected_agree_acc_25"] = float(exp_) if b <= int(ac.sum()) else None
    # POST HOC, descriptive, added after the first final run (first-run files kept as final_first_run.log / result_first_run.json; bins and verdict code unchanged): ratio a PERFECT order inside the operative cell would reach at 25% (every cell error found in the cell's slots)
    x["perfect_order_ceiling_ratio_25"] = float((exp_ + min(fr["n_err_cell"], fr["audit_slots_in_cell"]) - fr["cell_random_expected_errors_in_slots"]) / exp_) if b <= int(ac.sum()) and exp_ > 0 else None
    res["pools"][k] = x

# verdict (Qwen3-4B only; rm secondary)
q, r_ = res["pools"]["q3_4b"], res["pools"]["rm"]; m3 = q["main"].get("0.25"); rr3 = m3["ratios"]["arm_over_agree_acc"] if m3 else None; mr = r_["main"].get("0.25"); rrr = mr["ratios"]["arm_over_agree_acc"] if mr else None; fq = q["op25"]
v = {"q3_4b": {"auc": fq["auc"], "auc_ci_lo": fq["auc_ci95_boot"][0], "ratio": rr3["plug_in"] if rr3 else None, "n_err_cell": fq["n_err_cell"], "n_cell": fq["n_cell"], "cell": [fq["cell_agreement"], fq["cell_n_distinct"]], "slots_in_cell": fq["audit_slots_in_cell"]}}
v["q3_4b"]["pass_arm"] = bool(rr3 is not None and fq["auc"] >= 0.62 and fq["auc_ci95_boot"][0] > 0.50 and rr3["plug_in"] >= 1.04); v["q3_4b"]["kill_arm"] = bool(rr3 is not None and fq["auc"] <= 0.57 and rr3["plug_in"] <= 1.02)
v["rm_secondary"] = {"auc": r_["op25"]["auc"], "ratio": rrr["plug_in"] if rrr else None, "n_err_cell": r_["op25"]["n_err_cell"], "positive_plug_in_gain_25": bool(rrr is not None and rrr["plug_in"] > 1.0)}
under = fq["n_err_cell"] < MIN_CELL_ERR
if gate_msgs: verdict = "FAILED (gate): " + " | ".join(gate_msgs)
elif under: verdict = f"UNDERPOWERED (Qwen3-4B operative cell holds {fq['n_err_cell']} judge errors, fewer than {MIN_CELL_ERR})"
elif v["q3_4b"]["pass_arm"]: verdict = "PASS (the tie-break transfers across error regimes" + ("" if v["rm_secondary"]["positive_plug_in_gain_25"] else "; NOTE rm shows no positive plug-in gain at 25%") + ")"
elif v["q3_4b"]["kill_arm"]: verdict = "KILL (the mechanism does not transfer to the weak policy)"
else: verdict = "INCONCLUSIVE"
res["verdict"] = verdict; res["verdict_inputs"] = v; res["gate_messages"] = gate_msgs
# POST HOC context (added after the first final run, never the verdict): what the registered PASS / KILL bins would read if the >= 60 operative-cell-error requirement were waived (all other gates must still hold)
res["reading_if_error_gate_waived"] = "FAILED (other gate)" if gate_msgs else "PASS" if v["q3_4b"]["pass_arm"] else "KILL" if v["q3_4b"]["kill_arm"] else "INCONCLUSIVE"
# claim 3 (descriptive): l6a_cell_rate.py output on pool/q3_4b, if present
cvp = os.path.join(R, "l6a", "q3_4b", "result_cv.json")
if os.path.exists(cvp):
    cv = json.load(open(cvp)); pr = cv["primary"]; th = pr.get("top_half", {})
    res["claim3_l6a_descriptive"] = {"gates": cv["gates"], "n_acc": pr["n_acc"], "n_err": pr["n_err"], "n_top": pr["n_top"], "n_err_top": pr["n_err_top"], "JA_auc_acc": pr["sets"]["JA"]["auc"], "JA_auc_top": th["sets"]["JA"]["auc_fold_mean"] if th else None,
                                    "deltas_acc": {kk: {"delta": d["delta"], "ci95": d["boot_ci95"]} for kk, d in pr["deltas"].items()}, "deltas_top_half": {kk: {"delta": d["delta"], "ci95": d["boot_ci95"]} for kk, d in th["deltas"].items()} if th else None,
                                    "cell_rate_alone_top": th["cell_rate_alone"] if th else None, "best_cell_set": pr["best_cell_set"], "A_pass": pr["A_pass"], "K_kill_side": pr["K_kill_side"]}
else: res["claim3_l6a_descriptive"] = {"note": "result_cv.json missing"}
res["key_numbers"] = {k: {"cell": [x["op25"]["cell_agreement"], x["op25"]["cell_n_distinct"]], "n_cell": x["op25"]["n_cell"], "n_err_cell": x["op25"]["n_err_cell"], "slots_in_cell_25": x["op25"]["audit_slots_in_cell"], "auc": x["op25"]["auc"], "auc_ci95": x["op25"]["auc_ci95_boot"], "agreement_tier_auc": x["op25"]["agreement_tier_auc_verdict_score"],
                           "ratio_25": x["main"]["0.25"]["ratios"]["arm_over_agree_acc"]["plug_in"] if "0.25" in x["main"] else None, "ratio_25_ci95": x["main"]["0.25"]["ratios"]["arm_over_agree_acc"]["percentile_ci95"] if "0.25" in x["main"] else None,
                           "ratio_10": x["main"]["0.1"]["ratios"]["arm_over_agree_acc"]["plug_in"] if "0.1" in x["main"] else None,
                           "caught_25": {qq: x["main"]["0.25"]["caught_full_pool_8seed"][qq] for qq in ("uniform", "agree_acc", "agree_acc_soth", "agree_acc_sdiff", "greedy_warm7", "greedy_blend7", "oracle")} if "0.25" in x["main"] else None,
                           "n_J": x["op25"]["n_J_total"], "n_acc": x["op25"]["n_acc"], "l8c_a_n_J": l8[k]["n_J_fresh"], "l8c_a_auc": l8[k]["auc"], "l8c_a_ratio_25": l8[k]["ratio_25"]} for k, x in res["pools"].items()}
res["wall_s"] = time.time() - t0; json.dump(res, open(os.path.join(R, "result.json"), "w"), indent=1, default=float)

f3 = lambda z: "nan" if z is None or (isinstance(z, float) and np.isnan(z)) else f"{z:.3f}"
L = [verdict, f"[post hoc context, not the verdict] registered bins with the >= 60 operative-cell-error requirement waived: {res['reading_if_error_gate_waived']}", "", f"[regime] weak 0.5B: majority-vote error {mv_err:.4f} ({mv_err / MV_ERR_L8C:.2f}x the 1.5B {rs8['group_proxy_error_rate']:.4f}), rollout accuracy {rs_['rollout_accuracy']:.4f} (1.5B {rs8['rollout_accuracy']:.4f}), truncated fraction {trunc:.4f}, unanimous {rs_['frac_unanimous']:.3f} (1.5B {rs8['frac_unanimous']:.3f}); gate (iii) {res['regime']['gate_iii_pass']}", ""]
for k, x in res["pools"].items():
    L += [f"## {x['name']} (verdict arm {x['verdict_arm']})", f"scorer gates {json.dumps(x['scorer_gates'])}; gate (i) repro {x['gate_i_repro']}; structural {x['structural']}; gate (iv) null mean AUC {x['op25']['null_perm_auc_mean']:.4f} (SD {x['op25']['null_perm_auc_sd']:.3f})"]
    for tag, t in (("25%", x["op25"]), ("10% descriptive", x["op10"])):
        L.append(f"[operative cell at {tag}, budget {t['budget']}] accepted {t['n_acc']}, J {t['n_J_total']} (mv-wrong {t['n_majority_vote_wrong']}); cell agreement {t['cell_agreement']} n_distinct {t['cell_n_distinct']}: {t['n_cell']} groups / {t['n_err_cell']} errors, {t['n_before_cell']} accepted groups before it, {t['audit_slots_in_cell']} audit slots inside (fully audited {t['cell_fully_audited']}); "
                 f"verdict AUC {t['auc']:.3f} CI [{t['auc_ci95_boot'][0]:.3f}, {t['auc_ci95_boot'][1]:.3f}] (boot SD {t['auc_boot_sd']:.3f}); s_oth AUC {f3(t['auc_other_scores']['s_oth_desc'])}, -s_diff AUC {f3(t['auc_other_scores']['minus_s_diff'])}; agreement-tier ({t['agreement_tier_n']} groups, {t['agreement_tier_n_err']} errors) AUC {f3(t['agreement_tier_auc_verdict_score'])}; "
                 f"tie-break found {t['cell_tiebreak_errors_found_in_slots']} errors in its {t['audit_slots_in_cell']} slots against {t['cell_random_expected_errors_in_slots']:.1f} expected at random (+{t['cell_gain_errors']:.1f})")
        L.append("   9-feature within-cell AUC (higher = error): " + ", ".join(f"{qq} {f3(z)}" for qq, z in t["feature_aucs_higher_is_error"].items()))
    for f in ("0.25", "0.1"):
        mm = x["main"].get(f)
        if not mm: continue
        c, sd = mm["caught_full_pool_8seed"], mm["caught_boot_sd"]; L.append(f"[{int(float(f)*100)}% budget {mm['budget']}] pool errors {mm['n_pool_errors']} (oracle per audit {mm['oracle_per_audit']:.3f}, saturated {mm['oracle_budget_saturated']}); best fixed5 {mm['best_fixed5']}, natural best of 7 {mm['best_fixed7_natural']}; caught 8 seeds (boot SD): " + "; ".join(f"{qq} {c[qq]:.1f} ({sd[qq]:.1f})" for qq in ("uniform", "agree_acc", "agree_acc_soth", "agree_acc_sdiff", "greedy_warm7", "greedy_blend7", "oracle")))
        L.append("   fixed means at 25% (7 rules): " + ", ".join(f"{qq} {z:.1f}" for qq, z in mm["fixed_mean_at_25_7"].items()) + f"; H = (oracle - natural best fixed) / oracle {mm['H_oracle_minus_best_fixed_over_oracle']:.3f}, (oracle - agree_acc) / oracle {mm['H_oracle_minus_agree_acc_over_oracle']:.3f}")
        for nm, r in mm["ratios"].items(): L.append(f"   {nm}: plug-in {r['plug_in']:.3f}, boot mean {r['boot_mean']:.3f} (SD {r['boot_sd']:.3f}), percentile CI [{r['percentile_ci95'][0]:.3f}, {r['percentile_ci95'][1]:.3f}]")
    L += [f"exact expected agree_acc at 25% under random in-block order: {x['exact_expected_agree_acc_25']}; [post hoc] ratio a perfect order inside the operative cell would reach at 25%: {x['perfect_order_ceiling_ratio_25']}", ""]
L.append("[claim 3, l6a_cell_rate.py unchanged on pool/q3_4b, descriptive] " + json.dumps(res["claim3_l6a_descriptive"])[:3000])
open(os.path.join(R, "table.txt"), "w").write("\n".join(L)); print("\n".join(L), flush=True)
