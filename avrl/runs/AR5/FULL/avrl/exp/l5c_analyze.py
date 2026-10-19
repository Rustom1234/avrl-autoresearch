"""L5c stage 1 (CPU, .venv python): claim 3 analysis of ONE new proxy (--kind rm | judge3b) on the saved E21 T=0.25 rollouts. PRE-REGISTERED here before any real score was read; copy of exp/l3b_analyze.py
(already generic in the score column) with these changes only: --kind, the accept rule, 1,000-draw permutation nulls, a cell bootstrap CI of the neighbor ratio, the neighbor ratio at four registered cuts,
and the L5c reading. Inputs: <run_dir>/scores.parquet, gates.json (l5c_proxy_score.py) and <e21>/feats_T0.25.parquet (+ result.json). Writes <out>/feats_J.parquet (E21 feats + s_maj, s_oth, s_diff, accept,
judge_error, plus the sensitivity accept/judge_error_sens; this is the replay pool), <out>/result_analyze.json, <out>/analyze_table.txt. Exit 5 (and a result_analyze.json saying so) if the proxy failed to load or a gate failed.
Definitions. s_maj / s_oth = proxy score of the majority completion / of the label-free other completion. Label J = ACCEPTED AND the majority completion is gold-wrong (no gold in any proxy input; its correctness equals
majority_correct, checked). ACC = accepted groups; within ACC the label J is simply gold-wrong. n_err = J count in ACC.
Accept rule, PRIMARY: judge3b accept = s_maj > 0 (as L3b). rm has no natural zero: accept = the K = round(majority-vote accuracy x 2,000) = 1,715 groups with the highest s_maj (accept rate equal to the
majority-vote accuracy 85.75%, exact top-K by s_maj, stable on ties). SENSITIVITY (reported, never used for the verdict): judge3b = the same exact top-K rule (tau-matched, as L3b's sensitivity); rm = top-K with K equal
to the Qwen3-4B accept count (1,840 of 2,000, 92.0%, from L3b feats_J.parquet), i.e. the same acceptance volume as the lead.
Feature sets: J0 = s_maj, s_oth, s_diff = s_maj - s_oth, mean_tokens, std_tokens, any_truncated, log_qlen (no agreement); JA = J0 + agreement + n_distinct (strongest cheap baseline, PRIMARY); each + MiniLM pc0..15.
CV = E21 code path on ACC (RepeatedStratifiedKFold 5x3 random_state 0, StandardScaler + balanced LogisticRegression, identical folds for every set), 15-fold mean AUC and fold SD, delta = fold mean of
(set - base); paired bootstrap (1,000 ACC resamples, rng 0) of the delta of the 3-repeat-averaged out-of-fold scores. Jaccard = |J-error groups ∩ majority-vote-wrong groups| / |union| (J is a subset of the
majority-vote-wrong groups, so it equals n_err / n_majority_vote_wrong); also reported against the Qwen3-4B J set from L3b feats_J.parquet.
Neighbor ratio (NHR) = D.neighbor_hit_ratio of J (leave-one-out within cell, P(hit | a same-cell neighbor is a hit) / P(hit)), computed by an exactly equivalent vectorized routine (self-checked against
D.neighbor_hit_ratio on the observed labels and 5 permutations) at cell30 / cell100 / cell300 on four registered cuts of ACC ranked by s_maj (higher = more confidently accepted):
  top50 = s_maj >= median of ACC (THE VERDICT CUT, registered in L3b), top25 = s_maj >= 75th percentile, top75 = s_maj >= 25th percentile, all = all of ACC.
Null = 1,000 draws permuting J within the cut (mean and p95); cell bootstrap CI = 1,000 resamples of whole cells with replacement within the cut (percentile 95%). L3b saw 2.09 / 3.38 / 1.41 / 1.07 at these cuts for Qwen3-4B.
REGISTERED BINS (L5c-proxies; the experiment verdict is formed in l5c_finalize.py from both kinds):
  Per proxy, at a cut: REPLICATES = ratio >= 1.5 and ratio > own null p95. KILL-SIDE = ratio < 1.3 or ratio <= own null p95 (the strict reading, ratio < 1.3 only, is reported beside it).
  PASS (claim 3 lead replicates, strong form): BOTH proxies pass their gates, have n_err in ACC >= 100 and REPLICATES at top50 cell100, AND delta(JA_emb - JA) >= 0.03 with bootstrap CI lower end > 0 for at least one.
  weaker reading, NOT a pass: exactly one proxy REPLICATES (split, inconclusive; the AUC route is reported separately).
  KILL (claim 3 closed for every proxy type tried): BOTH proxies pass gates with n_err in ACC >= 100 and KILL-SIDE at top50 cell100.
  A proxy that fails a gate or cannot load is a failed experiment for that proxy and cannot support a kill (partial, inconclusive). n_err in ACC < 100: underpowered, not a kill. Anything else: inconclusive.
  If the verdict at the top50 cut differs from the verdict recomputed at two or more of the other three cuts, the verdict line says so.
Reproduction gate: the same CV code on the E21 confident subset reproduces E21 result.json cheap / cheap+emb AUCs to 1e-6 (exit 4)."""
import os, sys, json, argparse, time
import numpy as np, pandas as pd
ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))); sys.path.insert(0, ROOT)
from aa import data as D
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import RepeatedStratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.metrics import roc_auc_score
ap = argparse.ArgumentParser(); ap.add_argument("--kind", required=True, choices=["rm", "judge3b"]); ap.add_argument("--run_dir", required=True); ap.add_argument("--e21", required=True); ap.add_argument("--out", required=True)
ap.add_argument("--l3b", default=os.path.join(ROOT, "avrl", "runs", "L3b-judge-proxy"))
a = ap.parse_args(); t0 = time.time(); os.makedirs(a.out, exist_ok=True); kind = a.kind
cheap = ["agreement", "n_distinct", "mean_tokens", "std_tokens", "any_truncated", "log_qlen"]; embc = [f"pc{i}" for i in range(16)]
J0 = ["s_maj", "s_oth", "s_diff", "mean_tokens", "std_tokens", "any_truncated", "log_qlen"]; JA = J0 + ["agreement", "n_distinct"]
SETS = {"J0": J0, "J0_emb": J0 + embc, "JA": JA, "JA_emb": JA + embc}; PAIRS = [("JA_emb", "JA"), ("J0_emb", "J0")]
B, NPERM, NBOOT_NHR, CELLS, CUTS = 1000, 1000, 1000, (30, 100, 300), ("top50", "top25", "top75", "all")

def bail(msg, extra=None):
    json.dump({"kind": kind, "status": "failed", "message": msg, **(extra or {})}, open(os.path.join(a.out, "result_analyze.json"), "w"), indent=1, default=float); print(msg, flush=True); sys.exit(5)

fx = os.path.join(a.run_dir, "failed.txt")
if os.path.exists(fx): bail(f"[{kind}] proxy failed to load or raised in l5c_proxy_score.py, experiment failed for this proxy, no verdict:\n" + open(fx).read(), {"failed_txt": open(fx).read()})
if not os.path.exists(os.path.join(a.run_dir, "gates.json")): bail(f"[{kind}] no gates.json in {a.run_dir}, experiment failed for this proxy, no verdict")
gates = json.load(open(os.path.join(a.run_dir, "gates.json")))
if not gates.get("all_pass"): bail(f"[{kind}] GATES FAILED in l5c_proxy_score.py, experiment failed for this proxy, no verdict: " + json.dumps(gates), {"gates_score": gates})
F = pd.read_parquet(os.path.join(a.e21, "feats_T0.25.parquet")); S = pd.read_parquet(os.path.join(a.run_dir, "scores.parquet"))
M = S[S.role == "maj"].set_index("pid"); O = S[S.role == "oth"].set_index("pid"); pids = np.sort(np.intersect1d(M.index.values, O.index.values))
F = F[F.pid.isin(pids)].sort_values("pid").reset_index(drop=True); assert len(F) == len(pids) and (F.pid.values == pids).all()
F["s_maj"] = M.loc[pids, "s"].values; F["s_oth"] = O.loc[pids, "s"].values; F["s_diff"] = F.s_maj - F.s_oth
assert (M.loc[pids, "correct"].values == F.majority_correct.values).all(), "majority completion correctness must equal majority_correct"
assert np.isfinite(F[["s_maj", "s_oth"]].values).all()
n = len(F); print(f"[{kind}] scored groups {n}, errors (majority-vote wrong) {int(F.proxy_error.sum())}", flush=True)

def cv(G, label, sets, free):
    """E21 cv_aucs plus out-of-fold predictions: fold AUCs [15] per set / free score, OOF [3 repeats, n] per set; identical folds for every set."""
    y = G[label].values.astype(int); folds = list(RepeatedStratifiedKFold(n_splits=5, n_repeats=3, random_state=0).split(np.zeros(len(y)), y))
    auc = {k: [] for k in list(sets) + list(free)}; oof = {k: np.zeros((3, len(y))) for k in sets}
    for f, (tr, te) in enumerate(folds):
        for k, sc in free.items(): auc[k].append(roc_auc_score(y[te], sc[te]))
        for k, cols in sets.items():
            m = make_pipeline(StandardScaler(), LogisticRegression(max_iter=1000, class_weight="balanced")).fit(G[cols].values[tr], y[tr])
            p = m.predict_proba(G[cols].values[te])[:, 1]; auc[k].append(roc_auc_score(y[te], p)); oof[k][f // 5, te] = p
    return y, {k: np.array(v) for k, v in auc.items()}, oof

# reproduction gate: E21 confident subset, cheap and cheap+emb
if n == 2000:
    conf = F[F.agreement >= 0.75].reset_index(drop=True); _, au, _ = cv(conf, "proxy_error", {"cheap": cheap, "cheap_emb": cheap + embc}, {"agree": -conf.agreement.values})
    e = json.load(open(os.path.join(a.e21, "result.json")))["datasets"]["T0.25"]["confident"]
    rep = {"e21_cheap": e["auc_cheap"], "here_cheap": float(au["cheap"].mean()), "e21_cheap_emb": e["auc_cheap_emb"], "here_cheap_emb": float(au["cheap_emb"].mean()), "e21_agree": e["auc_agreement_only"], "here_agree": float(au["agree"].mean())}
    rep["max_abs_diff"] = max(abs(rep["e21_cheap"] - rep["here_cheap"]), abs(rep["e21_cheap_emb"] - rep["here_cheap_emb"]), abs(rep["e21_agree"] - rep["here_agree"])); rep["pass"] = bool(rep["max_abs_diff"] < 1e-6)
    print(f"[{kind}] E21 reproduction gate:", json.dumps(rep), flush=True)
    if not rep["pass"]: print("GATE FAILED: E21 confident-subset AUCs not reproduced", flush=True); sys.exit(4)
else: rep = {"skipped": "not all 2000 pids scored"}

def auc_direct(y, s): return float(roc_auc_score(y, s)) if 0 < y.sum() < len(y) else float("nan")

def nhr_fast(y, codes):
    """Same quantity as D.neighbor_hit_ratio(y, cells)['ratio'] on integer cell codes: leave-one-out hit rate of the other docs in the cell, P(hit | loo > 0) / P(hit)."""
    y = y.astype(float); cnt = np.bincount(codes); sm = np.bincount(codes, weights=y); base = y.mean(); cn = cnt[codes]; multi = cn > 1
    loo = (sm[codes] - y)[multi] / (cn[multi] - 1); sel = loo > 0
    return float(y[multi][sel].mean() / base) if (base > 0 and sel.any()) else float("nan")

def topk_accept(s, K):
    m = np.zeros(len(s), bool); m[np.argsort(-s, kind="stable")[:K]] = True; return m

def cut_subsets(ACC):
    s = ACC.s_maj
    return {"top50": ACC[s >= s.median()], "top25": ACC[s >= s.quantile(0.75)], "top75": ACC[s >= s.quantile(0.25)], "all": ACC}

def nhr_report(sub, k, pr, br):
    """Observed ratio, 1,000-draw permutation null (labels permuted within the cut) and 1,000-draw cell bootstrap (whole cells resampled with replacement) for one cut and cell size."""
    y = sub.J.values.astype(int); codes, _ = pd.factorize(sub[f"cell{k}"].values); obs = nhr_fast(y, codes)
    chk = D.neighbor_hit_ratio(y, sub[f"cell{k}"].values)["ratio"]; assert (np.isnan(obs) and np.isnan(chk)) or abs(obs - chk) < 1e-9, (obs, chk)
    for _ in range(5):
        yp = pr.permutation(y); v1, v2 = nhr_fast(yp, codes), D.neighbor_hit_ratio(yp, sub[f"cell{k}"].values)["ratio"]; assert (np.isnan(v1) and np.isnan(v2)) or abs(v1 - v2) < 1e-9, (v1, v2)
    nl = np.array([nhr_fast(pr.permutation(y), codes) for _ in range(NPERM)])
    rows = [np.where(codes == c)[0] for c in range(codes.max() + 1)]; sizes = np.array([len(r) for r in rows]); bs = []
    for _ in range(NBOOT_NHR):
        ch = br.integers(0, len(rows), len(rows)); idx = np.concatenate([rows[c] for c in ch]); bs.append(nhr_fast(y[idx], np.repeat(np.arange(len(ch)), sizes[ch])))
    bs = np.array(bs); p95 = float(np.nanpercentile(nl, 95)) if np.isfinite(nl).any() else float("nan")
    r = {"n": int(len(sub)), "n_err": int(y.sum()), "base_rate": float(y.mean()), "n_cells": int(len(rows)), "ratio": obs, "null_mean": float(np.nanmean(nl)), "null_p95": p95, "frac_null_ge_obs": float(np.nanmean(nl >= obs)),
         "n_null_finite": int(np.isfinite(nl).sum()), "boot_mean": float(np.nanmean(bs)), "boot_ci95": [float(np.nanpercentile(bs, 2.5)), float(np.nanpercentile(bs, 97.5))], "n_boot_finite": int(np.isfinite(bs).sum())}
    ok = np.isfinite(obs) and np.isfinite(p95)
    r["replicates"] = bool(ok and obs >= 1.5 and obs > p95); r["kill_side"] = bool(ok and (obs < 1.3 or obs <= p95)); r["kill_side_strict_lt_1.3"] = bool(ok and obs < 1.3)
    return r

def subset_report(F, accept, tag, rule):
    F = F.copy(); F["accept"] = accept.astype(int); F["judge_error"] = (accept & (F.majority_correct.values == 0)).astype(int); ACC = F[accept].reset_index(drop=True); ACC["J"] = ACC.judge_error
    n_err = int(ACC.J.sum()); mv = F.proxy_error.values == 1; Jset = F.judge_error.values == 1; inter = int((Jset & mv).sum()); union = int((Jset | mv).sum())
    conf_err = (F.agreement.values >= 0.75) & mv
    r = {"tag": tag, "rule": rule, "tau_min_accepted_s_maj": float(F.s_maj.values[accept].min()) if accept.any() else None, "n_groups": int(len(F)), "n_acc": int(len(ACC)), "accept_rate": float(len(ACC) / len(F)), "n_err": n_err,
         "judge_error_rate_overall": float(Jset.mean()), "judge_error_rate_in_acc": float(n_err / max(len(ACC), 1)), "n_majority_vote_wrong": int(mv.sum()), "jaccard_vs_majority_vote_wrong": float(inter / max(union, 1)),
         "J_subset_of_mv_wrong": bool(inter == int(Jset.sum())), "frac_of_mv_wrong_accepted": float(Jset.sum() / max(mv.sum(), 1)), "n_confident_mv_errors_agree_ge_0.75": int(conf_err.sum()),
         "frac_of_J_in_confident_mv_errors": float((Jset & conf_err).sum() / max(Jset.sum(), 1)), "jaccard_vs_confident_mv_errors": float((Jset & conf_err).sum() / max((Jset | conf_err).sum(), 1)),
         "judge_rejects_right_majority_frac": float(((~accept) & (F.majority_correct.values == 1)).sum() / max((F.majority_correct.values == 1).sum(), 1))}
    if os.path.exists(os.path.join(a.l3b, "feats_J.parquet")):
        Q = pd.read_parquet(os.path.join(a.l3b, "feats_J.parquet")).set_index("pid").loc[F.pid.values]; q = Q.judge_error.values == 1
        r["n_qwen3_4b_J"] = int(q.sum()); r["jaccard_J_vs_qwen3_4b_J"] = float((Jset & q).sum() / max((Jset | q).sum(), 1)); r["n_J_and_qwen3_4b_J"] = int((Jset & q).sum())
    if n_err < 10 or len(ACC) - n_err < 10: r["note"] = "too few errors for CV"; return r, F
    y, au, oof = cv(ACC, "J", SETS, {"smaj_alone": -ACC.s_maj.values, "agree_alone": -ACC.agreement.values})
    rng = np.random.default_rng(0); idx = [rng.integers(0, len(y), len(y)) for _ in range(B)]; idx = [i for i in idx if 0 < y[i].sum() < len(i)]
    avg = {k: v.mean(0) for k, v in oof.items()}; r.update(n_boot_used=len(idx), auc_smaj_alone_direct=auc_direct(y, -ACC.s_maj.values), auc_agree_alone_direct=auc_direct(y, -ACC.agreement.values), sets={})
    r["auc_smaj_alone"] = float(au["smaj_alone"].mean()); r["auc_smaj_alone_fold_sd"] = float(au["smaj_alone"].std()); r["auc_agree_alone"] = float(au["agree_alone"].mean())
    for k in SETS: r["sets"][k] = {"auc": float(au[k].mean()), "auc_fold_sd": float(au[k].std()), "oof_avg_auc": auc_direct(y, avg[k])}
    r["deltas"] = {}
    for k, b in PAIRS:
        d = au[k] - au[b]; db = np.array([roc_auc_score(y[i], avg[k][i]) for i in idx]) - np.array([roc_auc_score(y[i], avg[b][i]) for i in idx])
        r["deltas"][f"{k}_minus_{b}"] = {"delta": float(d.mean()), "delta_fold_sd": float(d.std()), "oof_avg_delta": float(auc_direct(y, avg[k]) - auc_direct(y, avg[b])),
                                         "boot_ci95": [float(np.percentile(db, 2.5)), float(np.percentile(db, 97.5))], "boot_sd": float(db.std()), "boot_mean": float(db.mean())}
    pr, br = np.random.default_rng(0), np.random.default_rng(1); r["nhr"] = {}
    for cut, sub in cut_subsets(ACC).items(): r["nhr"][cut] = {f"cell{k}": nhr_report(sub.reset_index(drop=True), k, pr, br) for k in CELLS}
    r["nhr_top_half"] = r["nhr"]["top50"]  # L3b-compatible alias
    return r, F

mv_acc = float(F.majority_correct.mean()); K_mv = int(round(mv_acc * n)); acc_topk = topk_accept(F.s_maj.values, K_mv)
Q3 = os.path.join(a.l3b, "feats_J.parquet"); K_q3 = int(pd.read_parquet(Q3).accept.sum()) if os.path.exists(Q3) else None
if kind == "judge3b":
    acc_p, rule_p = (F.s_maj.values > 0), "s_maj > 0 (label-free, as L3b)"; acc_s, rule_s = acc_topk, f"top-K by s_maj, K = round(majority-vote accuracy x n) = {K_mv} (tau-matched)"
else:
    acc_p, rule_p = acc_topk, f"top-K by s_maj, K = round(majority-vote accuracy {mv_acc:.4f} x n) = {K_mv}"
    acc_s, rule_s = (topk_accept(F.s_maj.values, K_q3), f"top-K by s_maj, K = Qwen3-4B accept count = {K_q3}") if K_q3 else (None, None)
res = {"kind": kind, "status": "ok", "n_groups": n, "gates_score": gates, "e21_reproduction_gate": rep}
res["overall"] = {"auc_neg_smaj_for_majority_vote_error": auc_direct(F.proxy_error.values, -F.s_maj.values), "auc_neg_agreement_for_majority_vote_error": auc_direct(F.proxy_error.values, -F.agreement.values),
                  "s_maj_mean": float(F.s_maj.mean()), "s_maj_sd": float(F.s_maj.std()), "s_oth_mean": float(F.s_oth.mean()), "mv_accuracy": mv_acc, "K_majority_vote_matched": K_mv, "accept_rate_primary": float(acc_p.mean()),
                  "accept_rate_sensitivity": float(acc_s.mean()) if acc_s is not None else None, "sanity_auc_s_vs_correct_all_rows": gates["4_sanity_auc"]["auc_s_vs_correct"]}
res["primary"], FJ = subset_report(F, acc_p, "primary", rule_p)
if acc_s is not None: res["sensitivity"], FS = subset_report(F, acc_s, "sensitivity", rule_s); FJ["accept_sens"] = FS.accept.values; FJ["judge_error_sens"] = FS.judge_error.values
FJ.to_parquet(os.path.join(a.out, "feats_J.parquet"), index=False)

# per-proxy claim 3 reading (primary rule)
p = res["primary"]; lead = {"kind": kind, "gates_pass": True, "sanity_auc": gates["4_sanity_auc"]["auc_s_vs_correct"], "n_err_acc": p["n_err"], "n_err_ge_100": bool(p["n_err"] >= 100), "n_acc": p["n_acc"],
                            "jaccard_vs_majority_vote_wrong": p["jaccard_vs_majority_vote_wrong"], "jaccard_J_vs_qwen3_4b_J": p.get("jaccard_J_vs_qwen3_4b_J")}
if "sets" in p:
    d = p["deltas"]["JA_emb_minus_JA"]; h = p["nhr"]["top50"]["cell100"]
    lead.update(n_err_top_half=h["n_err"], nhr_top_half_cell100=h["ratio"], null_mean=h["null_mean"], null_p95=h["null_p95"], boot_ci95=h["boot_ci95"], replicates=h["replicates"], kill_side=h["kill_side"], kill_side_strict=h["kill_side_strict_lt_1.3"],
                delta_auc=d["delta"], delta_boot_ci95=d["boot_ci95"], delta_ok=bool(d["delta"] >= 0.03 and d["boot_ci95"][0] > 0),
                by_cut={c: {"n_err": p["nhr"][c]["cell100"]["n_err"], "ratio": p["nhr"][c]["cell100"]["ratio"], "null_p95": p["nhr"][c]["cell100"]["null_p95"], "replicates": p["nhr"][c]["cell100"]["replicates"], "kill_side": p["nhr"][c]["cell100"]["kill_side"],
                            "kill_side_strict": p["nhr"][c]["cell100"]["kill_side_strict_lt_1.3"]} for c in CUTS})
else: lead.update(replicates=False, kill_side=False, kill_side_strict=False, delta_ok=False, note="too few errors for CV and neighbor ratio")
res["lead"] = lead; res["wall_s"] = time.time() - t0
json.dump(res, open(os.path.join(a.out, "result_analyze.json"), "w"), indent=1, default=float)

f4 = lambda x: f"{x:.4f}"; L = []
for tag in ("primary", "sensitivity"):
    if tag not in res: continue
    r = res[tag]; L += [f"## [{kind}] {tag}: {r['rule']}", f"ACC {r['n_acc']} of {r['n_groups']} (accept rate {r['accept_rate']:.3f}), n_err {r['n_err']} (rate in ACC {r['judge_error_rate_in_acc']:.4f}, overall {r['judge_error_rate_overall']:.4f}), "
                        f"Jaccard vs majority-vote-wrong {r['jaccard_vs_majority_vote_wrong']:.3f} ({r['n_majority_vote_wrong']} mv-wrong, {r['frac_of_mv_wrong_accepted']:.3f} accepted), Jaccard vs Qwen3-4B J {r.get('jaccard_J_vs_qwen3_4b_J', float('nan')):.3f}", ""]
    if "sets" not in r: continue
    L += [f"AUC of s_maj alone: fold mean {f4(r['auc_smaj_alone'])} (SD {f4(r['auc_smaj_alone_fold_sd'])}), direct {f4(r['auc_smaj_alone_direct'])}; agreement alone fold mean {f4(r['auc_agree_alone'])}", "", "| set | AUC (fold SD) |", "|---|---|"]
    L += [f"| {k} | {f4(v['auc'])} ({f4(v['auc_fold_sd'])}) |" for k, v in r["sets"].items()] + ["", "| delta | fold mean (SD) | OOF-avg delta | bootstrap 95% CI | boot SD |", "|---|---|---|---|---|"]
    L += [f"| {k} | {d['delta']:+.4f} ({f4(d['delta_fold_sd'])}) | {d['oof_avg_delta']:+.4f} | [{d['boot_ci95'][0]:+.4f}, {d['boot_ci95'][1]:+.4f}] | {f4(d['boot_sd'])} |" for k, d in r["deltas"].items()]
    L += ["", "| cut | cell | n | n_err | ratio | null mean | null p95 | P(null>=obs) | cell-boot 95% CI | replicates | kill-side |", "|---|---|---|---|---|---|---|---|---|---|---|"]
    for cut in CUTS:
        for k in CELLS:
            h = r["nhr"][cut][f"cell{k}"]; L.append(f"| {cut} | {k} | {h['n']} | {h['n_err']} | {h['ratio']:.3f} | {h['null_mean']:.3f} | {h['null_p95']:.3f} | {h['frac_null_ge_obs']:.3f} | [{h['boot_ci95'][0]:.2f}, {h['boot_ci95'][1]:.2f}] | {h['replicates']} | {h['kill_side']} |")
    L.append("")
L += ["## overall", json.dumps(res["overall"], indent=1), "", "## lead (per-proxy claim 3 reading, primary rule)", json.dumps(res["lead"], indent=1, default=float)]
open(os.path.join(a.out, "analyze_table.txt"), "w").write("\n".join(L)); print("\n".join(L), flush=True); print(f"[{kind}] analysis done in {time.time()-t0:.0f}s", flush=True)
