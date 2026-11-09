"""L9a-shardB-tiebreak, CPU analysis (.venv python). PRE-REGISTERED here, before any shard-B gold-derived column (pass_rate, majority_correct, proxy_error, judge_error, gold, majority, rollouts correct / proxy columns) was read for this experiment.
Written and frozen BEFORE the shard-B rollout job was launched. Shard-A labels (pids 2000..3999) were already known (L8c-A-fresh-tiebreak); nothing about shard B (pids 4000..5999) had been generated or read when this text was written.
Question. Does the label-free dissenting-score tie-break (inside each agreement tier of the accepted groups: rm by s_oth DESCENDING, Qwen3-4B by s_diff ASCENDING, i.e. replay arms agree_acc_soth / agree_acc_sdiff) catch more proxy errors than plain agree_acc when averaged over the
budget curve, on 4,000 FRESH GSM8K prompts (shard A = orig pids 2000..3999 of sample(n=6000, random_state=0), shard B = orig pids 4000..5999, both disjoint from pids 0..1999 and from each other), with every rule frozen from pids 0..1999 (no reselection)?
Null: the shard-A gain (12 of 14 budget cells positive, budget-mean ratio about 1.017 rm and 1.034 Qwen3-4B) is noise or a post hoc budget pick.
STAGES. --stage cells (label-free: MiniLM embeddings of the shard-B questions, frozen KMeans cell100 assignment), --stage build (scores -> feats_J.parquet), --stage power (design-only power check, shard A alone), --stage chA (comparator CH_A per budget from shard A), --stage merge (joins chunked replay files boot_tb_f<f>.c<j>.json, resamples [50j, 50j+50) of the same default_rng(12345) stream, into boot_tb_f<f>.json), --stage final (gates, statistics, verdict).
Replays (exp/l9a_boot_forced.py, copy of l8c_boot_forced.py with the optional env CH_RULE arm "ch_a") are run by exp/L9a-shardB-tiebreak.sh.
POOL AND FROZEN RULES (nothing is refit).
  Features: s_maj, s_oth, s_diff = s_maj - s_oth, mean_tokens, std_tokens, any_truncated, log_qlen = log1p(q_len), agreement, n_distinct (replay features, as L8c). s_maj / s_oth = proxy score of the majority completion / of the label-free other completion (l5c_proxy_score.py, l3b_judge_score.py, unchanged).
  Pooled pool = concatenation of shard A (2,000 groups, orig pids 2000..3999, rows in orig_pid order) and shard B (2,000 groups, orig pids 4000..5999), 4,000 groups, row order A then B, replay column pid = orig_pid.
  Accept (label-free, defined ONCE on the pooled pool and restricted to the shards): Qwen3-4B accept = s_maj > 0; rm accept = the K = round(0.8575 x 4000) = 3,430 groups with the highest s_maj in the pooled pool (exact top-K, stable on ties; 0.8575 is the frozen pids 0..1999 quantile).
  Label J = judge_error = ACCEPTED AND the majority completion is gold-wrong. Verdict arms: rm agree_acc_soth, Qwen3-4B agree_acc_sdiff (FORCE_BEST7), MIN_CELL=1 (all groups kept), B = 200 resamples x 4 seeds, resample generator default_rng(12345) (same resample indices at every budget, checked via per-resample n_err).
METRICS (per proxy). (1) S_tb = mean over the six budgets 10 / 15 / 20 / 25 / 30 / 40% (budget = int(frac x pool size) audits) of log(tie-break / agree_acc), where per resample the ratio is mean over 4 seeds of the verdict arm over mean over 4 seeds of agree_acc; plug-in = same on the unresampled pool with 8-seed means;
  percentile 95% CI over the 200 resamples, bootstrap mean and SD. Measured pooled SD = sample SD (ddof 1) of S_tb over the 200 pooled resamples. Also the per-budget ratio with percentile CI.
  (2) tier-stratified AUC of the verdict score for J (rm s_oth, Qwen3-4B minus s_diff; higher = audit earlier) over ALL audited agreement tiers: accepted groups with agreement < 1.0, strata = agreement value (k / 8), U statistic summed over strata (average ranks for ties) divided by the total number of
  within-stratum (error, non-error) pairs; strata lacking either class contribute no pairs; 95% percentile CI from 2,000 group-level bootstrap resamples (rows of the accepted agreement < 1 set resampled with replacement, default_rng(0), strata recomputed). Reported alongside: per-stratum counts / errors / AUC, the 0.875 tier (L8c definition) counts and AUC, AUC of the other sign / feature, and the
  (agreement, n_distinct) cell-stratified AUC (the cell the replay actually breaks ties in).
  (3) shard-B-alone sign check: S_tb on shard B alone (2,000 groups, the POOLED accept flags restricted to B, replay B = 200 x 4, default_rng(12345)); only its plug-in sign enters the bins. Shard-A-alone S_tb (L8c-A pool, its own accept rule) comes from the design-only power check.
  (4) DESCRIPTIVE SECONDARY, no bin: on the pooled pool greedy_blend7 (warm-started from the verdict arm) against the best fixed rule re-chosen at each budget (argmax of the 8-seed full-pool mean of the seven fixed rules agree_thr, judge_thr, blend50, agree_acc, blend50_acc, agree_acc_soth, agree_acc_sdiff at that budget, first listed wins ties):
  ratio with percentile CI at 10% and 25%, and the curve statistic mean over the six budgets of log(greedy_blend7 / best fixed) per resample. Known instrument bias (L10a): the bootstrap mean sits about 0.01 below the plug-in for learner-against-fixed contrasts; stated, not corrected.
  (5) REGISTERED SECONDARY, descriptive, no bin (L11b statistic for the two 1.5B cells, so the shard-B labels are read once under one registration): comparator CH_A per budget = argmax of the 8-seed full-pool mean caught of the seven fixed rules on SHARD A in full (2,000 groups, pooled accept flags), first listed wins ties, chosen from shard-A files only
  (stage chA, before any shard-B replay); held pool = shard B in full (2,000 groups, pooled accept flags); learner greedy_blend7 warm-started from the verdict arm; C_B = mean over budgets 15 / 20 / 25% of log(greedy_blend7 / CH_A) per resample on the 200 x 4 shard-B resamples, percentile CI, SD; also the six-budget C_B, the 10 / 30 / 40% window, C_B against agree_acc and against the frozen
  verdict arm, the CH_A identity per budget, and the per-cell plug-in signs. The L11b bins (pooled plug-in >= +0.025 with 4 cells) are NOT applied: only two cells exist here, so no L11b verdict is issued from L9a.
GATES (any failed gate = FAILED experiment, no verdict). (i) forced-BEST7 replay reproduces runs/L7a-tiebreak/<pool>/boot_tb_f0.25.json exactly (all keys, full pool and every resample) on the pids 0..1999 feats_J of the same proxy; (ii) scorer gates all passed for shard A and shard B including sanity AUC of s vs rollout correctness >= 0.70 and no subsampling;
  (iv) null control: mean stratified AUC of 1,000 within-stratum permutations of the verdict score within 0.02 of 0.5; structural: 2,000 scored groups per shard, local pid == arange, orig_pid / questions equal sample(n=6000, random_state=0), majority-row correctness equals majority_correct, all replay pools have the expected size (4,000 / 2,000),
  replay sanity flags true, the same resample indices at all six budgets. (Gate (iii) of L8c, 25 boundary-tier errors, is replaced by the measured-power gate below.)
MEASURED-POWER GATE. Measured pooled SD of S_tb <= 0.015 on BOTH proxies, else the experiment is UNDERPOWERED (neither pass nor kill can be declared). Design-only power check run BEFORE any GPU job: SD of S_tb over the 200 resamples of shard A alone, expected pooled SD = that divided by about 1.4 (stage power, needs only shard-A files).
REGISTERED BINS.
  PASS: on BOTH rm and Qwen3-4B, pooled A + B: S_tb percentile CI lower end > 0 AND tier-stratified AUC point >= 0.62 with bootstrap CI lower end > 0.50 AND S_tb on shard B alone has a positive plug-in; all gates passing and the measured pooled SD of S_tb <= 0.015 on both. A pass makes the label-free order a finding; the deliverable becomes
  agreement threshold plus dissenting-score tie-break with a gain of about 2% to 4% of errors caught.
  KILL: S_tb plug-in <= +0.005 on BOTH proxies on pooled A + B, gates passing and the SD gate met (the shard-A budget-mean ratios 1.017 and 1.034 did not hold on fresh data). A kill retires the tie-break pivot.
  UNDERPOWERED: measured pooled SD > 0.015 on either proxy (and gates passing). Anything else: INCONCLUSIVE. Gate failure: FAILED with the exact message.
CAVEATS carried: the two proxies share about 70% of their errors (two correlated tests, not two independent ones); both proxies and the policy are Qwen-family, one policy (1.5B), one task (GSM8K train); the tie-break features and signs were chosen post hoc on pids 0..1999 (frozen since); the six budgets are fixed in advance and include budgets where
the tie-break cell is lumpy; the rm accept fraction is a convenience top-K with no natural zero; the resample instrument penalises learner arms (descriptive items only).
Usage: l9a_analyze.py --run_dir <runs/L9a-shardB-tiebreak> --stage cells|build|power|chA|merge|final [--dirs d1,d2 for merge] [--proxies rm,q3_4b for build / chA]"""
import os, sys, json, argparse, time
import numpy as np, pandas as pd
from scipy.stats import rankdata
from sklearn.metrics import roc_auc_score
ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))); RUNS = os.path.join(ROOT, "avrl", "runs"); L8C = os.path.join(RUNS, "L8c-A-fresh-tiebreak")
ap = argparse.ArgumentParser(); ap.add_argument("--run_dir", required=True); ap.add_argument("--stage", required=True, choices=["cells", "build", "power", "chA", "merge", "final"]); ap.add_argument("--dirs", default=""); ap.add_argument("--proxies", default="rm,q3_4b")
a = ap.parse_args(); R = a.run_dir; t0 = time.time()
FEATS = ["s_maj", "s_oth", "s_diff", "mean_tokens", "std_tokens", "any_truncated", "log_qlen", "agreement", "n_distinct"]
FRACS = (0.1, 0.15, 0.2, 0.25, 0.3, 0.4); FS = lambda f: f"{f:g}"; RM_Q = 0.8575; NA = NB = 2000; N = NA + NB; K_POOL = int(round(RM_Q * N)); TIER = 0.875; NBOOT_AUC = 2000; NPERM = 1000; SD_GATE = 0.015
FIX7 = ("agree_thr", "judge_thr", "blend50", "agree_acc", "blend50_acc", "agree_acc_soth", "agree_acc_sdiff")
P = {"rm": dict(name="Skywork rm (top 85.75% of pooled s_maj)", arm="agree_acc_soth", scoref="s_oth", sign=1.0, a_dir=os.path.join(L8C, "rm", "rm"), b_dir=os.path.join(R, "rm", "rm"), old_pool=os.path.join(RUNS, "L5c-proxies", "rm"), old_boot=os.path.join(RUNS, "L7a-tiebreak", "rm")),
     "q3_4b": dict(name="Qwen3-4B Yes/No judge (s_maj > 0)", arm="agree_acc_sdiff", scoref="s_diff", sign=-1.0, a_dir=os.path.join(L8C, "q3_4b"), b_dir=os.path.join(R, "q3_4b"), old_pool=os.path.join(RUNS, "L3b-judge-proxy"), old_boot=os.path.join(RUNS, "L7a-tiebreak", "L3b-judge-proxy"))}
POOL = lambda k, w: os.path.join(R, {"AB": "pool", "A": "pool_A", "B": "pool_B"}[w], k, "feats_J.parquet")
BOOT_AB = lambda k, f: os.path.join(R, "boot", k, f"boot_tb_f{FS(f)}.json"); BOOT_B = lambda k, f: os.path.join(R, "boot_B", k, f"boot_tb_f{FS(f)}.json"); BOOT_A0 = lambda k, f: os.path.join(R, "boot_A0", k, f"boot_tb_f{FS(f)}.json")
BOOT_A_L8C = lambda k, f: os.path.join(L8C, "boot", k, f"boot_tb_f{FS(f)}.json") if f in (0.1, 0.25) else os.path.join(R, "power_check", k, f"boot_tb_f{FS(f)}.json")  # L8c-A pool replays (its own accept rule): 10% and 25% from L8c, the other four from the power check
auc = lambda y, s: float(roc_auc_score(y, s)) if 0 < int(np.sum(y)) < len(y) and np.ptp(s) > 0 else float("nan")
q6000 = lambda: pd.read_parquet(os.path.join(ROOT, "avrl", "data", "gsm8k_train.parquet")).sample(n=6000, random_state=0).reset_index(drop=True).question.values


def topk_accept(s, K):
    m = np.zeros(len(s), bool); m[np.argsort(-s, kind="stable")[:K]] = True; return m


# ---------------------------------------------------------------- stage cells (label-free)
def stage_cells():
    """Frozen KMeans(100, n_init=3, random_state=0) refit on the stored pids 0..1999 MiniLM embeddings (reproduces the stored old cell100 and L8c-A's cell100_fresh.npy exactly, checked); shard-B questions embedded with all-MiniLM-L6-v2 (cpu, normalized, batch 64, as e21_lowT_analyze.py) and assigned by predict."""
    out = os.path.join(R, "cells"); os.makedirs(out, exist_ok=True)
    from sklearn.cluster import KMeans
    eo = np.load(os.path.join(RUNS, "E21-lowT-confident-errors", "q_emb_minilm.npy")); ea = np.load(os.path.join(L8C, "cells", "q_emb_minilm_fresh.npy")); ca = np.load(os.path.join(L8C, "cells", "cell100_fresh.npy"))
    Fo = pd.read_parquet(os.path.join(P["q3_4b"]["old_pool"], "feats_J.parquet")).sort_values("pid").reset_index(drop=True)
    km = KMeans(n_clusters=100, n_init=3, random_state=0).fit(eo); chk = {"refit_equals_stored_old_cell100": bool((km.labels_ == Fo.cell100.values).all()), "predict_A_equals_L8c_file": bool((km.predict(ea) == ca).all())}
    qb = q6000()[4000:6000]
    from sentence_transformers import SentenceTransformer
    eb = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2", device="cpu").encode(list(qb), batch_size=64, normalize_embeddings=True, show_progress_bar=False)
    cb = km.predict(eb); np.save(os.path.join(out, "q_emb_minilm_B.npy"), eb); np.save(os.path.join(out, "cell100_B.npy"), cb)
    chk.update(n_B=int(len(qb)), n_cells_used_B=int(len(np.unique(cb))), embedding_shape=list(eb.shape)); json.dump(chk, open(os.path.join(out, "cells_check.json"), "w"), indent=1); print(chk, flush=True)
    assert chk["refit_equals_stored_old_cell100"] and chk["predict_A_equals_L8c_file"], chk


# ---------------------------------------------------------------- stage build
def shard_frame(G, M, O, cell):
    F = G.copy(); F["log_qlen"] = np.log1p(F.q_len); F["split"] = (F.agreement < 1).astype(int); F["cell100"] = cell; pids = np.arange(len(F))
    F["s_maj"] = M.loc[pids, "s"].values; F["s_oth"] = O.loc[pids, "s"].values; F["s_diff"] = F.s_maj - F.s_oth
    assert (M.loc[pids, "correct"].values == F.majority_correct.values).all(), "majority completion correctness must equal majority_correct"; assert np.isfinite(F[["s_maj", "s_oth"]].values).all(); return F


def build(k):
    p = P[k]
    for d in (p["a_dir"], p["b_dir"]):
        if os.path.exists(os.path.join(d, "failed.txt")): raise SystemExit(f"[{k}] scorer raised:\n" + open(os.path.join(d, "failed.txt")).read())
    # shard A: the L8c-A feats_J rows (same scores), accept and judge_error recomputed under the pooled rule below
    Fa = pd.read_parquet(os.path.join(L8C, "pool", k, "feats_J.parquet")).sort_values("pid").reset_index(drop=True); qa = q6000()
    assert len(Fa) == NA and (Fa.orig_pid.values == 2000 + np.arange(NA)).all() and (Fa.question.values == qa[2000:4000]).all()
    # shard B
    G = pd.read_parquet(os.path.join(R, "rollouts", "T0.25", "groups.parquet")).sort_values("pid").reset_index(drop=True); assert len(G) == NB and (G.pid.values == np.arange(NB)).all()
    assert (G.question.values == qa[4000:6000]).all() and (G.orig_pid.values == 4000 + np.arange(NB)).all(), "shard B must be orig pids 4000..5999 of sample(n=6000, random_state=0)"
    S = pd.read_parquet(os.path.join(p["b_dir"], "scores.parquet")); M = S[S.role == "maj"].set_index("pid"); O = S[S.role == "oth"].set_index("pid")
    assert len(M) == NB and len(O) == NB and (np.sort(M.index.values) == np.arange(NB)).all(), "all 2,000 shard-B groups must be scored"
    Fb = shard_frame(G, M, O, np.load(os.path.join(R, "cells", "cell100_B.npy")))
    cols = [c for c in Fa.columns if c not in ("accept", "judge_error")]; Fa = Fa[cols].copy(); Fb = Fb[cols].copy(); Fa["shard"] = "A"; Fb["shard"] = "B"
    F = pd.concat([Fa, Fb], ignore_index=True); F["pid"] = F.orig_pid.values; assert len(F) == N and F.pid.is_unique
    acc = (F.s_maj.values > 0) if k == "q3_4b" else topk_accept(F.s_maj.values, K_POOL)
    F["accept"] = acc.astype(int); F["judge_error"] = (acc & (F.majority_correct.values == 0)).astype(int)
    for w, d in (("AB", F), ("A", F[F.shard == "A"]), ("B", F[F.shard == "B"])):
        os.makedirs(os.path.dirname(POOL(k, w)), exist_ok=True); d.reset_index(drop=True).to_parquet(POOL(k, w), index=False)
    ga = json.load(open(os.path.join(p["a_dir"], "gates.json"))); gb = json.load(open(os.path.join(p["b_dir"], "gates.json")))
    print(f"[{k}] pooled {N}: accepted {int(F.accept.sum())} (A {int(F.accept[F.shard=='A'].sum())}, B {int(F.accept[F.shard=='B'].sum())}), J errors {int(F.judge_error.sum())} (A {int(F.judge_error[F.shard=='A'].sum())}, B {int(F.judge_error[F.shard=='B'].sum())}), "
          f"scorer all_pass A {ga['all_pass']} B {gb['all_pass']}, sanity AUC A {ga['4_sanity_auc']['auc_s_vs_correct']:.4f} B {gb['4_sanity_auc']['auc_s_vs_correct']:.4f}", flush=True)


# ---------------------------------------------------------------- replay-json statistics
def rsmean(bj, arm): return np.array([np.mean(r[arm]) for r in bj["resamples"]])
def fpmean(bj, arm): return float(np.mean(bj["full_pool"][arm]))


def summ(S, plug):
    q = np.percentile(S, [2.5, 97.5]); return {"plug_in": float(plug), "boot_mean": float(S.mean()), "boot_sd": float(S.std(ddof=1)), "percentile_ci95": [float(q[0]), float(q[1])], "frac_above_0": float((S > 0).mean())}


def series(path_of, fracs, num, den, need_boot=True):
    """Per budget: log(num / den) per resample (mean over seeds first) and plug-in (8-seed full-pool means); curve statistic = mean over the budgets. num / den = arm name or function(bj) -> arm name. Returns summary dict plus the raw per-resample curve array."""
    lr, pl, per, nerr, ids = [], [], {}, [], {}
    for f in fracs:
        bj = json.load(open(path_of(f))); nm = num(bj) if callable(num) else num; dn = den(bj) if callable(den) else den
        ids[f] = {"num": nm, "den": dn, "n_pool": bj["n_pool"], "n_boot": bj["n_boot"], "seeds": bj["seeds_per_resample"], "budget": bj["full_pool"]["budget"], "sanity_ok": bool(all(v is True for kk, v in bj["sanity"].items() if kk.startswith("gate_")))}
        pl.append(np.log(max(fpmean(bj, nm), 1e-9) / max(fpmean(bj, dn), 1e-9)))
        if need_boot and bj["n_boot"] > 0:
            l = np.log(np.maximum(rsmean(bj, nm), 1e-9) / np.maximum(rsmean(bj, dn), 1e-9)); lr.append(l); nerr.append([r["n_err"] for r in bj["resamples"]])
            q = np.percentile(np.exp(l), [2.5, 97.5]); per[FS(f)] = {"ratio_plug_in": float(np.exp(pl[-1])), "ratio_boot_mean": float(np.exp(l).mean()), "ratio_sd": float(np.exp(l).std(ddof=1)), "ratio_ci95": [float(q[0]), float(q[1])], "arms": [nm, dn], "num_full_8seed": fpmean(bj, nm), "den_full_8seed": fpmean(bj, dn)}
        else: per[FS(f)] = {"ratio_plug_in": float(np.exp(pl[-1])), "arms": [nm, dn], "num_full_8seed": fpmean(bj, nm), "den_full_8seed": fpmean(bj, dn)}
    out = {"fracs": list(fracs), "per_budget": per, "ids": {FS(f): v for f, v in ids.items()}, "plug_in_curve": float(np.mean(pl)), "n_positive_budget_plugins": int(sum(x > 0 for x in pl))}
    S = None
    if lr:
        assert all(n == nerr[0] for n in nerr), "resample indices must be identical at every budget (per-resample n_err differs)"; S = np.mean(lr, axis=0); out["curve"] = summ(S, np.mean(pl)); out["resample_indices_identical_across_budgets"] = True
    return out, S


# ---------------------------------------------------------------- stratified AUC
def sauc(y, s, st):
    U = T = 0.0
    for g in np.unique(st):
        i = st == g; yy = y[i]; n1 = int(yy.sum()); n0 = len(yy) - n1
        if n1 == 0 or n0 == 0: continue
        r = rankdata(s[i]); U += r[yy == 1].sum() - n1 * (n1 + 1) / 2; T += n1 * n0
    return float(U / T) if T > 0 else float("nan")


def strat_stats(F, k, seed=0):
    p = P[k]; ac = F.accept.values == 1; m = ac & (F.agreement.values < 1 - 1e-9); y = F.judge_error.values[m].astype(int); sc = p["sign"] * F[p["scoref"]].values[m]; st = np.round(F.agreement.values[m] * 8).astype(int)
    cellst = st * 100 + F.n_distinct.values[m].astype(int); pt = sauc(y, sc, st); rng = np.random.default_rng(seed); n = len(y); bs = []
    for _ in range(NBOOT_AUC):
        i = rng.integers(0, n, n)
        if 0 < y[i].sum() < n: bs.append(sauc(y[i], sc[i], st[i]))
    bs = np.array(bs)
    nl = []
    for _ in range(NPERM):
        sp = sc.copy()
        for g in np.unique(st): j = np.flatnonzero(st == g); sp[j] = rng.permutation(sc[j])
        nl.append(sauc(y, sp, st))
    nl = np.array(nl); tab = {}
    for g in np.unique(st):
        j = st == g; tab[f"{g}/8"] = {"n": int(j.sum()), "n_err": int(y[j].sum()), "auc": auc(y[j], sc[j])}
    t = ac & np.isclose(F.agreement.values, TIER); other_sc = {"s_oth_desc": F.s_oth.values[m], "minus_s_diff": -F.s_diff.values[m]}
    return {"n_groups_audited_tiers": int(m.sum()), "n_err_audited_tiers": int(y.sum()), "verdict_score": ("s_oth" if p["sign"] > 0 else "minus s_diff"), "strat_auc": pt, "auc_ci95_boot": [float(np.percentile(bs, 2.5)), float(np.percentile(bs, 97.5))], "n_boot_used": int(len(bs)), "auc_boot_sd": float(bs.std(ddof=1)),
            "per_stratum": tab, "other_scores_strat_auc": {q: sauc(y, s_, st) for q, s_ in other_sc.items()}, "cell_stratified_auc_agreement_x_ndistinct": sauc(y, sc, cellst), "null_perm_auc_mean": float(nl.mean()), "null_perm_auc_sd": float(nl.std(ddof=1)), "null_perm_auc_p975": float(np.percentile(nl, 97.5)),
            "tier_0.875": {"n": int(t.sum()), "n_err": int(F.judge_error.values[t].sum()), "auc": auc(F.judge_error.values[t].astype(int), p["sign"] * F[p["scoref"]].values[t])},
            "n_acc": int(ac.sum()), "n_J_total": int(F.judge_error.sum()), "n_majority_vote_wrong": int(F.proxy_error.sum()), "n_J_in_agreement_lt_1": int(y.sum()), "n_J_unanimous_accepted": int(F.judge_error.values[ac & (F.agreement.values >= 1 - 1e-9)].sum())}


def repro(new_path, old_path):
    """Gate (i): every key of the old (L7a) file's full_pool and of every old resample equals the forced-BEST7 run on the same pids 0..1999 pool."""
    new, old = json.load(open(new_path)), json.load(open(old_path)); top = ("n_pool", "accept_col", "label", "features", "budget_frac", "seeds_per_resample", "n_boot")
    ok_top = all(new[q] == old[q] for q in top); ok_full = all(new["full_pool"].get(q) == v for q, v in old["full_pool"].items()); ok_res = len(new["resamples"]) == len(old["resamples"]) and all(rn.get(q) == v for rn, ro in zip(new["resamples"], old["resamples"]) for q, v in ro.items())
    return {"pass": bool(ok_top and ok_full and ok_res), "top_equal": bool(ok_top), "full_pool_equal": bool(ok_full), "resamples_equal": bool(ok_res), "n_resamples": len(old["resamples"]), "reference": old_path}


# ---------------------------------------------------------------- stage power (shard A only)
if a.stage == "cells": stage_cells(); sys.exit(0)
if a.stage == "build":
    for k in a.proxies.split(","): build(k)
    print(f"build done in {time.time()-t0:.0f}s", flush=True); sys.exit(0)
if a.stage == "power":
    res = {"note": "design-only power check, shard A alone (L8c-A pool, its own accept rule, labels already known), nothing about shard B read", "proxies": {}}
    for k, p in P.items():
        s, S = series(lambda f: BOOT_A_L8C(k, f), FRACS, p["arm"], "agree_acc"); sd = s["curve"]["boot_sd"]
        res["proxies"][k] = {"S_tb_A_alone": s["curve"], "per_budget": s["per_budget"], "n_positive_budget_plugins": s["n_positive_budget_plugins"], "sd_S_tb_shardA": sd, "expected_pooled_sd_div_1.4": sd / 1.4, "expected_pooled_sd_div_sqrt2": sd / np.sqrt(2), "design_gate_expected_sd_le_0.015": bool(sd / 1.4 <= SD_GATE)}
        print(f"[{k}] shard A alone: S_tb plug-in {s['curve']['plug_in']:+.4f}, boot mean {s['curve']['boot_mean']:+.4f}, SD {sd:.4f}, CI {s['curve']['percentile_ci95']}; expected pooled SD {sd/1.4:.4f} (/1.4) {sd/np.sqrt(2):.4f} (/sqrt2); budgets positive {s['n_positive_budget_plugins']}/6; "
              + ", ".join(f"{b} {v['ratio_plug_in']:.3f}" for b, v in s["per_budget"].items()), flush=True)
    os.makedirs(os.path.join(R, "power_check"), exist_ok=True); json.dump(res, open(os.path.join(R, "power_check", "power.json"), "w"), indent=1, default=float); sys.exit(0)
if a.stage == "merge":
    import glob
    dirs = a.dirs.split(",") if a.dirs else [os.path.join(R, w, k) for w in ("boot", "boot_B") for k in P]
    for d in dirs:
        for c0 in sorted(glob.glob(os.path.join(d, "boot_tb_f*.c0.json"))):
            stem = c0[:-len(".c0.json")]; parts = sorted(glob.glob(stem + ".c*.json"), key=lambda q: int(q[len(stem) + 2:-5])); js = [json.load(open(q)) for q in parts]
            for j in js[1:]: assert all(j[q] == js[0][q] for q in js[0] if q not in ("resamples", "chunk")), f"chunk header / full pool differs in {stem}"
            rs = sorted([r for j in js for r in j["resamples"]], key=lambda r: r["boot"]); assert [r["boot"] for r in rs] == list(range(js[0]["n_boot"])), f"resample indices incomplete in {stem}: {[r['boot'] for r in rs][:5]}"
            m = {q: v for q, v in js[0].items() if q != "chunk"}; m["resamples"] = rs; json.dump(m, open(stem + ".json", "w"), indent=1); print(f"merged {len(parts)} chunks -> {stem}.json ({len(rs)} resamples)", flush=True)
    sys.exit(0)
if a.stage == "chA":
    out = {}
    for k in a.proxies.split(","):
        out[k] = {}
        for f in FRACS:
            bj = json.load(open(BOOT_A0(k, f))); assert bj["n_pool"] == NA, (k, f, bj["n_pool"]); fm = bj["full_pool"]["fixed_mean_at_frac7"]; ch = max(FIX7, key=lambda q: fm[q]); out[k][FS(f)] = {"rule": ch, "fixed_means_8seed": fm}
    prev = json.load(open(os.path.join(R, "chA.json"))) if os.path.exists(os.path.join(R, "chA.json")) else {}; prev.update(out); out = prev
    json.dump(out, open(os.path.join(R, "chA.json"), "w"), indent=1); print({k: {f: v["rule"] for f, v in d.items()} for k, d in out.items()}, flush=True); sys.exit(0)

# ---------------------------------------------------------------- stage final
res = {"experiment": "L9a-shardB-tiebreak", "proxies": {}}; gate_msgs = []; samples = {}
for k, p in P.items():
    F = pd.read_parquet(POOL(k, "AB")); x = {"name": p["name"], "verdict_arm": p["arm"], "n_pool": int(len(F))}
    for sh, d in (("A", p["a_dir"]), ("B", p["b_dir"])):
        sg = json.load(open(os.path.join(d, "gates.json"))); ok = bool(sg["all_pass"] and sg["4_sanity_auc"]["auc_s_vs_correct"] >= 0.70 and not sg["1_timing"].get("subsampled"))
        x[f"scorer_gates_{sh}"] = {"all_pass": sg["all_pass"], "sanity_auc_s_vs_correct": sg["4_sanity_auc"]["auc_s_vs_correct"], "sanity_threshold": 0.70, "pass_2": sg["2_unbatched"]["pass"], "pass_3": sg["3_finite_mass"]["pass"], "subsampled": sg["1_timing"].get("subsampled"), "gate_ii": ok}
        if not ok: gate_msgs.append(f"{k}: gate (ii) scorer gates failed or subsampled for shard {sh} {json.dumps(x[f'scorer_gates_{sh}'])}")
    x["strat"] = strat_stats(F, k); x["gate_iv_null_perm"] = bool(abs(x["strat"]["null_perm_auc_mean"] - 0.5) <= 0.02)
    if not x["gate_iv_null_perm"]: gate_msgs.append(f"{k}: gate (iv) null control mean stratified AUC {x['strat']['null_perm_auc_mean']:.4f}")
    gp = os.path.join(R, "gate_i", k, "boot_f0.25.json"); x["gate_i_repro"] = repro(gp, os.path.join(p["old_boot"], "boot_tb_f0.25.json")) if os.path.exists(gp) else {"pass": False, "note": "missing gate (i) run"}
    if not x["gate_i_repro"]["pass"]: gate_msgs.append(f"{k}: gate (i) forced-BEST7 reproduction of L7a failed {json.dumps(x['gate_i_repro'])}")
    # pooled A + B curve statistic (verdict)
    s_pool, S_pool = series(lambda f: BOOT_AB(k, f), FRACS, p["arm"], "agree_acc"); x["S_tb_pooled"] = s_pool; samples[(k, "S_tb_pooled")] = S_pool
    for f, v in s_pool["ids"].items():
        if v["n_pool"] != N or not v["sanity_ok"] or v["n_boot"] != 200 or v["seeds"] != 4: gate_msgs.append(f"{k} pooled f{f}: replay pool / sanity / B mismatch {json.dumps(v)}")
    # shard B alone (sign check) and shard A alone (L8c-A pool, from the power check)
    s_b, S_b = series(lambda f: BOOT_B(k, f), FRACS, p["arm"], "agree_acc"); x["S_tb_shardB"] = s_b; samples[(k, "S_tb_B")] = S_b
    for f, v in s_b["ids"].items():
        if v["n_pool"] != NB or not v["sanity_ok"]: gate_msgs.append(f"{k} shard B f{f}: replay pool / sanity mismatch {json.dumps(v)}")
    s_a, S_a = series(lambda f: BOOT_A_L8C(k, f), FRACS, p["arm"], "agree_acc"); x["S_tb_shardA_L8c_pool"] = s_a
    # descriptive secondary (4): pooled learner against the best fixed rule re-chosen at each budget, and against agree_acc / the frozen arm
    bestfix = lambda bj: max(FIX7, key=lambda q: bj["full_pool"]["fixed_mean_at_frac7"][q])
    s_l, S_l = series(lambda f: BOOT_AB(k, f), FRACS, "greedy_blend7", bestfix); x["learner_vs_best_fixed_per_budget_pooled"] = s_l; samples[(k, "learner_vs_bestfixed")] = S_l
    x["learner_vs_agree_acc_pooled"], _ = series(lambda f: BOOT_AB(k, f), FRACS, "greedy_blend7", "agree_acc"); x["learner_vs_frozen_arm_pooled"], _ = series(lambda f: BOOT_AB(k, f), FRACS, "greedy_blend7", p["arm"])
    x["best_fixed_rule_per_budget_pooled"] = {FS(f): s_l["ids"][FS(f)]["den"] for f in FRACS}
    x["caught_pooled_8seed"] = {}
    for f in (0.1, 0.25):
        bj = json.load(open(BOOT_AB(k, f))); x["caught_pooled_8seed"][FS(f)] = {"budget": bj["full_pool"]["budget"], "n_pool_errors": bj["full_pool"]["n_err"], **{q: fpmean(bj, q) for q in ("uniform", "agree_acc", "agree_acc_soth", "agree_acc_sdiff", "greedy_warm7", "greedy_blend7", "oracle")}, "fixed_means_8seed": bj["full_pool"]["fixed_mean_at_frac7"]}
    # registered secondary (5): L11b statistic, comparator from shard A, scored on shard B
    chA = json.load(open(os.path.join(R, "chA.json")))[k]; w3 = (0.15, 0.2, 0.25)
    s_cb, S_cb = series(lambda f: BOOT_B(k, f), w3, "greedy_blend7", "ch_a"); s_cb6, S_cb6 = series(lambda f: BOOT_B(k, f), FRACS, "greedy_blend7", "ch_a"); s_cbo, S_cbo = series(lambda f: BOOT_B(k, f), (0.1, 0.3, 0.4), "greedy_blend7", "ch_a")
    s_cba, _ = series(lambda f: BOOT_B(k, f), w3, "greedy_blend7", "agree_acc"); s_cbf, _ = series(lambda f: BOOT_B(k, f), w3, "greedy_blend7", p["arm"])
    x["L11b_C_B"] = {"window_15_20_25": s_cb, "six_budgets": s_cb6, "window_10_30_40": s_cbo, "vs_agree_acc_15_20_25": s_cba, "vs_frozen_arm_15_20_25": s_cbf, "CH_A_rule_per_budget": {f: v["rule"] for f, v in chA.items()}}
    for f in FRACS:
        rb = json.load(open(BOOT_B(k, f)))["full_pool"]
        if rb.get("ch_a_rule") != chA[FS(f)]["rule"]: gate_msgs.append(f"{k} f{f}: shard-B replay used CH rule {rb.get('ch_a_rule')} != chA.json {chA[FS(f)]['rule']}")
        x["L11b_C_B"].setdefault("CH_A_on_B_vs_best_fixed_on_B_8seed", {})[FS(f)] = {"CH_A": rb["ch_a_rule"], "CH_A_caught_on_B": float(np.mean(rb["ch_a"])), "best_fixed_on_B": max(FIX7, key=lambda q: rb["fixed_mean_at_frac7"][q]), "best_fixed_caught_on_B": max(rb["fixed_mean_at_frac7"].values())}
    samples[(k, "C_B")] = S_cb; samples[(k, "C_B6")] = S_cb6; res["proxies"][k] = x
# two-cell pooled C_B (same resample indices of shard B in both proxies' pools, checked via identical per-resample n_err? the groups are the same rows, so the indices coincide)
Sc = (samples[("rm", "C_B")] + samples[("q3_4b", "C_B")]) / 2; plug2 = (res["proxies"]["rm"]["L11b_C_B"]["window_15_20_25"]["curve"]["plug_in"] + res["proxies"]["q3_4b"]["L11b_C_B"]["window_15_20_25"]["curve"]["plug_in"]) / 2
res["L11b_C_B_two_cell_pooled_15_20_25"] = {**summ(Sc, plug2), "n_cells_positive_plug_in": int(sum(res["proxies"][k]["L11b_C_B"]["window_15_20_25"]["curve"]["plug_in"] > 0 for k in P)), "note": "descriptive; L11b bins need four cells (L10c) and are not applied"}
# verdict
v = {}
for k, x in res["proxies"].items():
    S = x["S_tb_pooled"]["curve"]; sb = x["S_tb_shardB"]["curve"] if "curve" in x["S_tb_shardB"] else None; st = x["strat"]
    v[k] = {"S_tb_plug_in": S["plug_in"], "S_tb_ci_lo": S["percentile_ci95"][0], "S_tb_ci_hi": S["percentile_ci95"][1], "S_tb_measured_sd": S["boot_sd"], "strat_auc": st["strat_auc"], "strat_auc_ci_lo": st["auc_ci95_boot"][0], "S_tb_shardB_plug_in": x["S_tb_shardB"]["plug_in_curve"],
            "sd_gate_ok": bool(S["boot_sd"] <= SD_GATE), "pass_k": bool(S["percentile_ci95"][0] > 0 and st["strat_auc"] >= 0.62 and st["auc_ci95_boot"][0] > 0.50 and x["S_tb_shardB"]["plug_in_curve"] > 0), "kill_k": bool(S["plug_in"] <= 0.005)}
if gate_msgs: verdict = "FAILED (gate): " + " | ".join(gate_msgs)
elif not all(v[k]["sd_gate_ok"] for k in v): verdict = "UNDERPOWERED (measured pooled SD of S_tb above 0.015: " + ", ".join(f"{k} {v[k]['S_tb_measured_sd']:.4f}" for k in v) + ")"
elif all(v[k]["pass_k"] for k in v): verdict = "PASS (the label-free dissenting-score tie-break replicates on fresh prompts on both proxies)"
elif all(v[k]["kill_k"] for k in v): verdict = "KILL (S_tb plug-in at most +0.005 on both proxies)"
else: verdict = "INCONCLUSIVE"
res["verdict"] = verdict; res["verdict_inputs"] = v; res["gate_messages"] = gate_msgs
res["key_numbers"] = {k: {"S_tb_pooled_plug_in": v[k]["S_tb_plug_in"], "S_tb_pooled_ci95": [v[k]["S_tb_ci_lo"], v[k]["S_tb_ci_hi"]], "S_tb_pooled_sd": v[k]["S_tb_measured_sd"], "S_tb_shardB_plug_in": v[k]["S_tb_shardB_plug_in"], "strat_auc": v[k]["strat_auc"], "strat_auc_ci95": res["proxies"][k]["strat"]["auc_ci95_boot"],
                         "ratios_pooled_per_budget": {b: z["ratio_plug_in"] for b, z in res["proxies"][k]["S_tb_pooled"]["per_budget"].items()}, "n_J_pooled": res["proxies"][k]["strat"]["n_J_total"], "n_J_agreement_lt_1": res["proxies"][k]["strat"]["n_J_in_agreement_lt_1"],
                         "C_B_15_20_25": res["proxies"][k]["L11b_C_B"]["window_15_20_25"]["curve"]["plug_in"], "learner_vs_best_fixed_curve": res["proxies"][k]["learner_vs_best_fixed_per_budget_pooled"]["curve"]["plug_in"]} for k in P}
res["wall_s"] = time.time() - t0; json.dump(res, open(os.path.join(R, "result.json"), "w"), indent=1, default=float)

f3 = lambda z: "nan" if z is None or (isinstance(z, float) and np.isnan(z)) else f"{z:.3f}"; f4 = lambda z: f"{z:+.4f}"
L = [verdict, ""]
for k, x in res["proxies"].items():
    st = x["strat"]; S = x["S_tb_pooled"]["curve"]; Sb = x["S_tb_shardB"]; Sa = x["S_tb_shardA_L8c_pool"]
    L += [f"## {x['name']} (verdict arm {x['verdict_arm']})", f"gates: scorer A {x['scorer_gates_A']['gate_ii']} (sanity {x['scorer_gates_A']['sanity_auc_s_vs_correct']:.3f}), scorer B {x['scorer_gates_B']['gate_ii']} (sanity {x['scorer_gates_B']['sanity_auc_s_vs_correct']:.3f}), gate (i) repro {x['gate_i_repro']['pass']}, gate (iv) null perm mean AUC {st['null_perm_auc_mean']:.4f} (SD {st['null_perm_auc_sd']:.3f})",
          f"pooled A + B: accepted {st['n_acc']}, J {st['n_J_total']} (agreement<1: {st['n_J_in_agreement_lt_1']}, unanimous accepted: {st['n_J_unanimous_accepted']}), majority-vote-wrong {st['n_majority_vote_wrong']}",
          f"S_tb pooled: plug-in {f4(S['plug_in'])}, boot mean {f4(S['boot_mean'])}, SD {S['boot_sd']:.4f} (gate <= {SD_GATE}), CI [{S['percentile_ci95'][0]:+.4f}, {S['percentile_ci95'][1]:+.4f}], {x['S_tb_pooled']['n_positive_budget_plugins']}/6 budget plug-ins positive",
          "  per budget tie-break / agree_acc (plug-in [CI]): " + "; ".join(f"{b} {z['ratio_plug_in']:.3f} [{z['ratio_ci95'][0]:.3f}, {z['ratio_ci95'][1]:.3f}]" for b, z in x["S_tb_pooled"]["per_budget"].items()),
          f"S_tb shard B alone: plug-in {f4(Sb['plug_in_curve'])}" + (f", boot mean {f4(Sb['curve']['boot_mean'])}, SD {Sb['curve']['boot_sd']:.4f}, CI [{Sb['curve']['percentile_ci95'][0]:+.4f}, {Sb['curve']['percentile_ci95'][1]:+.4f}]" if "curve" in Sb else "") + "; per budget " + ", ".join(f"{b} {z['ratio_plug_in']:.3f}" for b, z in Sb["per_budget"].items()),
          f"S_tb shard A alone (L8c-A pool, own accept rule): plug-in {f4(Sa['plug_in_curve'])}, SD {Sa['curve']['boot_sd']:.4f}; per budget " + ", ".join(f"{b} {z['ratio_plug_in']:.3f}" for b, z in Sa["per_budget"].items()),
          f"tier-stratified AUC (agreement<1, {st['n_groups_audited_tiers']} groups / {st['n_err_audited_tiers']} errors): {st['strat_auc']:.3f} CI [{st['auc_ci95_boot'][0]:.3f}, {st['auc_ci95_boot'][1]:.3f}] (boot SD {st['auc_boot_sd']:.3f}); other score {json.dumps({q: round(z, 3) for q, z in st['other_scores_strat_auc'].items()})}; cell-stratified (agreement x n_distinct) {st['cell_stratified_auc_agreement_x_ndistinct']:.3f}",
          "  per stratum (n / errors / AUC): " + "; ".join(f"{g} {z['n']}/{z['n_err']}/{f3(z['auc'])}" for g, z in st["per_stratum"].items()) + f"; 0.875 tier {st['tier_0.875']['n']} groups / {st['tier_0.875']['n_err']} errors, AUC {f3(st['tier_0.875']['auc'])}"]
    for f in ("0.1", "0.25"):
        c = x["caught_pooled_8seed"][f]; L.append(f"  caught pooled {int(float(f)*100)}% (budget {c['budget']}, pool errors {c['n_pool_errors']}, 8 seeds): " + ", ".join(f"{q} {c[q]:.1f}" for q in ("uniform", "agree_acc", "agree_acc_soth", "agree_acc_sdiff", "greedy_warm7", "greedy_blend7", "oracle")))
    sl = x["learner_vs_best_fixed_per_budget_pooled"]; L += [f"[descriptive] greedy_blend7 / best fixed re-chosen per budget (pooled): curve plug-in {f4(sl['plug_in_curve'])}, boot mean {f4(sl['curve']['boot_mean'])}, SD {sl['curve']['boot_sd']:.4f}, CI [{sl['curve']['percentile_ci95'][0]:+.4f}, {sl['curve']['percentile_ci95'][1]:+.4f}]; per budget (rule, ratio [CI]): "
                                                  + "; ".join(f"{b} {sl['ids'][b]['den']} {z['ratio_plug_in']:.3f} [{z['ratio_ci95'][0]:.3f}, {z['ratio_ci95'][1]:.3f}]" for b, z in sl["per_budget"].items()),
          f"   greedy_blend7 / agree_acc curve plug-in {f4(x['learner_vs_agree_acc_pooled']['plug_in_curve'])} (boot mean {f4(x['learner_vs_agree_acc_pooled']['curve']['boot_mean'])}); greedy_blend7 / frozen arm curve plug-in {f4(x['learner_vs_frozen_arm_pooled']['plug_in_curve'])} (boot mean {f4(x['learner_vs_frozen_arm_pooled']['curve']['boot_mean'])})"]
    cb = x["L11b_C_B"]; w = cb["window_15_20_25"]["curve"]; L += [f"[registered secondary, L11b statistic, comparator from shard A, scored on shard B] CH_A per budget {json.dumps(cb['CH_A_rule_per_budget'])}",
          f"   C_B 15/20/25%: plug-in {f4(w['plug_in'])}, boot mean {f4(w['boot_mean'])}, SD {w['boot_sd']:.4f}, CI [{w['percentile_ci95'][0]:+.4f}, {w['percentile_ci95'][1]:+.4f}]; six budgets plug-in {f4(cb['six_budgets']['plug_in_curve'])}; 10/30/40% window plug-in {f4(cb['window_10_30_40']['plug_in_curve'])}; vs agree_acc {f4(cb['vs_agree_acc_15_20_25']['plug_in_curve'])}; vs frozen arm {f4(cb['vs_frozen_arm_15_20_25']['plug_in_curve'])}",
          "   per budget greedy_blend7 / CH_A: " + "; ".join(f"{b} {z['ratio_plug_in']:.3f}" for b, z in cb["six_budgets"]["per_budget"].items()) + "; CH_A caught on B vs best fixed on B (8 seeds): " + "; ".join(f"{b} {z['CH_A_caught_on_B']:.1f} vs {z['best_fixed_caught_on_B']:.1f} ({z['best_fixed_on_B']})" for b, z in cb["CH_A_on_B_vs_best_fixed_on_B_8seed"].items()), ""]
t2 = res["L11b_C_B_two_cell_pooled_15_20_25"]; L.append(f"[registered secondary] two-cell pooled C_B 15/20/25%: plug-in {f4(t2['plug_in'])}, boot mean {f4(t2['boot_mean'])}, SD {t2['boot_sd']:.4f}, CI [{t2['percentile_ci95'][0]:+.4f}, {t2['percentile_ci95'][1]:+.4f}], cells positive {t2['n_cells_positive_plug_in']}/2 (descriptive, no L11b verdict)")
L.append("verdict inputs: " + json.dumps({k: {q: (round(z, 4) if isinstance(z, float) else z) for q, z in d.items()} for k, d in v.items()}))
open(os.path.join(R, "table.txt"), "w").write("\n".join(L)); print("\n".join(L), flush=True)
