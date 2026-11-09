"""L3b stage 1 (CPU, .venv python): claim 3 analysis of the Qwen3-4B judge proxy on the saved E21 T=0.25 rollouts. PRE-REGISTERED here before any score was read.
Inputs: <run_dir>/scores.parquet, gates.json (l3b_judge_score.py) and <e21>/feats_T0.25.parquet (+ result.json). Writes <out>/feats_J.parquet (E21 feats + s_maj, s_oth, s_diff,
accept, judge_error and the tau-matched variants; this is the replay pool), <out>/result_analyze.json, <out>/analyze_table.txt.
Definitions. s_maj / s_oth = judge score of the majority completion / of the label-free other completion. accept = s_maj > 0 (tau = 0, label-free, PRIMARY); sensitivity
tau_match = the quantile of s_maj that gives an accept rate equal to the majority-vote-answer accuracy. judge_error (label J) = accept AND the majority completion is gold-wrong
(its correctness equals majority_correct, checked). ACC = accepted groups; within ACC the label J is simply gold-wrong. n_err = J count in ACC; any kill needs n_err >= 100.
Feature sets: J0 = s_maj, s_oth, s_diff = s_maj - s_oth, mean_tokens, std_tokens, any_truncated, log_qlen (no agreement); JA = J0 + agreement + n_distinct (strongest cheap baseline,
PRIMARY for claim 3); each + MiniLM pc0..15. CV = E21 code path on ACC (RepeatedStratifiedKFold 5x3 random_state 0, StandardScaler + balanced LogisticRegression, identical folds
for every set), 15-fold mean AUC and fold SD, delta = fold mean of (set - base); paired bootstrap (1,000 ACC resamples, rng 0) of the delta of the 3-repeat-averaged out-of-fold scores.
Jaccard = |J-error groups ∩ majority-vote-wrong groups| / |union| (J is a subset of the majority-vote-wrong groups by construction, so this equals n_err / n_majority_vote_wrong).
Confident neighbor ratio = D.neighbor_hit_ratio of J on the top half of ACC by s_maj (ranked by s_maj, higher = more confidently accepted) at cell30/100/300 with a 300-draw
permutation null (labels permuted within that subset; mean and p95).
Claim 3 PASS (all): n_err >= 100 AND delta(JA_emb - JA) >= 0.05 with bootstrap CI lower end > 0 AND Jaccard < 0.5 AND NHR cell100 >= 1.5 and > null p95.
Claim 3 KILL: n_err >= 100 AND (Jaccard > 0.8 OR [delta < 0.03 AND bootstrap CI upper end < 0.06 AND NHR cell100 < 1.3]). n_err < 100 -> 'underpowered, not a kill'.
Anything else -> inconclusive. A failed gate (judge sanity AUC < 0.70 etc.) -> failed experiment, no verdict. The sensitivity (tau_match) block is reported, never used for the verdict.
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
ap = argparse.ArgumentParser(); ap.add_argument("--run_dir", required=True); ap.add_argument("--e21", required=True); ap.add_argument("--out", required=True)
a = ap.parse_args(); t0 = time.time(); os.makedirs(a.out, exist_ok=True)
cheap = ["agreement", "n_distinct", "mean_tokens", "std_tokens", "any_truncated", "log_qlen"]; embc = [f"pc{i}" for i in range(16)]
J0 = ["s_maj", "s_oth", "s_diff", "mean_tokens", "std_tokens", "any_truncated", "log_qlen"]; JA = J0 + ["agreement", "n_distinct"]
SETS = {"J0": J0, "J0_emb": J0 + embc, "JA": JA, "JA_emb": JA + embc}; PAIRS = [("JA_emb", "JA"), ("J0_emb", "J0")]
B, NPERM, CELLS = 1000, 300, (30, 100, 300)

gates = json.load(open(os.path.join(a.run_dir, "gates.json")))
if not gates.get("all_pass"): print("GATES FAILED in l3b_judge_score.py, experiment failed, no verdict:", json.dumps(gates), flush=True); sys.exit(5)
F = pd.read_parquet(os.path.join(a.e21, "feats_T0.25.parquet")); S = pd.read_parquet(os.path.join(a.run_dir, "scores.parquet"))
M = S[S.role == "maj"].set_index("pid"); O = S[S.role == "oth"].set_index("pid"); pids = np.sort(np.intersect1d(M.index.values, O.index.values))
F = F[F.pid.isin(pids)].sort_values("pid").reset_index(drop=True); assert len(F) == len(pids) and (F.pid.values == pids).all()
F["s_maj"] = M.loc[pids, "s"].values; F["s_oth"] = O.loc[pids, "s"].values; F["s_diff"] = F.s_maj - F.s_oth
assert (M.loc[pids, "correct"].values == F.majority_correct.values).all(), "majority completion correctness must equal majority_correct"
n = len(F); print(f"scored groups {n}, errors (majority-vote wrong) {int(F.proxy_error.sum())}", flush=True)

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
    print("E21 reproduction gate:", json.dumps(rep), flush=True)
    if not rep["pass"]: print("GATE FAILED: E21 confident-subset AUCs not reproduced", flush=True); sys.exit(4)
else: rep = {"skipped": "not all 2000 pids scored"}

def auc_direct(y, s): return float(roc_auc_score(y, s)) if 0 < y.sum() < len(y) else float("nan")

def subset_report(F, accept, tag):
    F = F.copy(); F["accept"] = accept.astype(int); F["judge_error"] = (accept & (F.majority_correct.values == 0)).astype(int); ACC = F[accept].reset_index(drop=True); ACC["J"] = ACC.judge_error
    n_err = int(ACC.J.sum()); mv = F.proxy_error.values == 1; Jset = F.judge_error.values == 1; inter = int((Jset & mv).sum()); union = int((Jset | mv).sum())
    conf_err = (F.agreement.values >= 0.75) & mv
    r = {"tag": tag, "n_groups": int(len(F)), "n_acc": int(len(ACC)), "accept_rate": float(len(ACC) / len(F)), "n_err": n_err, "judge_error_rate_overall": float(Jset.mean()),
         "judge_error_rate_in_acc": float(n_err / max(len(ACC), 1)), "n_majority_vote_wrong": int(mv.sum()), "jaccard_vs_majority_vote_wrong": float(inter / max(union, 1)), "J_subset_of_mv_wrong": bool(inter == int(Jset.sum())),
         "frac_of_mv_wrong_accepted": float(Jset.sum() / max(mv.sum(), 1)), "n_confident_mv_errors_agree_ge_0.75": int(conf_err.sum()), "frac_of_J_in_confident_mv_errors": float((Jset & conf_err).sum() / max(Jset.sum(), 1)),
         "jaccard_vs_confident_mv_errors": float((Jset & conf_err).sum() / max((Jset | conf_err).sum(), 1)), "judge_rejects_right_majority_frac": float(((~accept) & (F.majority_correct.values == 1)).sum() / max((F.majority_correct.values == 1).sum(), 1))}
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
    top = ACC[ACC.s_maj >= ACC.s_maj.median()].reset_index(drop=True); yt = top.J.values; r["nhr_top_half"] = {"n": int(len(top)), "n_err": int(yt.sum()), "base_rate": float(yt.mean())}
    pr = np.random.default_rng(0)
    for k in CELLS:
        cells = top[f"cell{k}"].values; obs = D.neighbor_hit_ratio(yt, cells)["ratio"]; nl = np.array([D.neighbor_hit_ratio(pr.permutation(yt), cells)["ratio"] for _ in range(NPERM)])
        r["nhr_top_half"][f"cell{k}"] = {"ratio": float(obs), "null_mean": float(np.nanmean(nl)), "null_p95": float(np.nanpercentile(nl, 95)), "frac_null_ge_obs": float(np.nanmean(nl >= obs)), "n_null": int(np.isfinite(nl).sum())}
    return r, F

acc0 = (F.s_maj.values > 0); res = {"n_groups": n, "gates_score": gates, "e21_reproduction_gate": rep}
acc_rate_target = float(F.majority_correct.mean()); tau_m = float(np.quantile(F.s_maj.values, 1 - acc_rate_target)); acc1 = (F.s_maj.values > tau_m)
res["judge_overall"] = {"auc_neg_smaj_for_majority_vote_error": auc_direct(F.proxy_error.values, -F.s_maj.values), "auc_neg_agreement_for_majority_vote_error": auc_direct(F.proxy_error.values, -F.agreement.values),
                        "s_maj_mean": float(F.s_maj.mean()), "s_maj_sd": float(F.s_maj.std()), "s_oth_mean": float(F.s_oth.mean()), "mv_accuracy": acc_rate_target, "tau_match": tau_m, "accept_rate_tau0": float(acc0.mean()),
                        "accept_rate_tau_match": float(acc1.mean())}
res["primary_tau0"], FJ = subset_report(F, acc0, "tau0"); res["sensitivity_tau_match"], FJ1 = subset_report(F, acc1, "tau_match")
FJ["accept_tau_match"] = FJ1.accept.values; FJ["judge_error_tau_match"] = FJ1.judge_error.values; FJ.to_parquet(os.path.join(a.out, "feats_J.parquet"), index=False)

# claim 3 reading (primary tau = 0)
p = res["primary_tau0"]; c3 = {"n_err_ge_100": bool(p["n_err"] >= 100), "n_err": p["n_err"]}
if "sets" in p:
    d = p["deltas"]["JA_emb_minus_JA"]; nh = p["nhr_top_half"]["cell100"]
    c3.update(delta=d["delta"], boot_ci95=d["boot_ci95"], jaccard=p["jaccard_vs_majority_vote_wrong"], nhr_cell100=nh["ratio"], nhr_cell100_null_p95=nh["null_p95"],
              pass_conditions={"delta_ge_0.05": bool(d["delta"] >= 0.05), "ci_lower_gt_0": bool(d["boot_ci95"][0] > 0), "jaccard_lt_0.5": bool(p["jaccard_vs_majority_vote_wrong"] < 0.5), "nhr100_ge_1.5": bool(nh["ratio"] >= 1.5),
                               "nhr100_gt_null_p95": bool(nh["ratio"] > nh["null_p95"]), "n_err_ge_100": c3["n_err_ge_100"]},
              kill_conditions={"jaccard_gt_0.8": bool(p["jaccard_vs_majority_vote_wrong"] > 0.8), "delta_lt_0.03": bool(d["delta"] < 0.03), "ci_upper_lt_0.06": bool(d["boot_ci95"][1] < 0.06), "nhr100_lt_1.3": bool(nh["ratio"] < 1.3)})
    k = c3["kill_conditions"]; c3["pass"] = bool(all(c3["pass_conditions"].values())); c3["kill"] = bool(c3["n_err_ge_100"] and (k["jaccard_gt_0.8"] or (k["delta_lt_0.03"] and k["ci_upper_lt_0.06"] and k["nhr100_lt_1.3"])))
    c3["verdict"] = "claim 3 PASS" if c3["pass"] else ("claim 3 KILL" if c3["kill"] else ("underpowered, not a kill (n_err < 100)" if not c3["n_err_ge_100"] else "inconclusive"))
else: c3["verdict"] = "underpowered, not a kill (n_err < 100 or too few correct)"
res["claim3"] = c3; res["wall_s"] = time.time() - t0
json.dump(res, open(os.path.join(a.out, "result_analyze.json"), "w"), indent=1, default=float)

f4 = lambda x: f"{x:.4f}"; L = []
for tag in ("primary_tau0", "sensitivity_tau_match"):
    r = res[tag]; L += [f"## {tag}: ACC {r['n_acc']} of {r['n_groups']} (accept rate {r['accept_rate']:.3f}), n_err {r['n_err']} (rate in ACC {r['judge_error_rate_in_acc']:.4f}, overall {r['judge_error_rate_overall']:.4f}), "
                        f"Jaccard vs majority-vote-wrong {r['jaccard_vs_majority_vote_wrong']:.3f} ({r['n_majority_vote_wrong']} mv-wrong, {r['frac_of_mv_wrong_accepted']:.3f} accepted), Jaccard vs confident mv errors {r['jaccard_vs_confident_mv_errors']:.3f}", ""]
    if "sets" not in r: continue
    L += [f"AUC of s_maj alone: fold mean {f4(r['auc_smaj_alone'])} (SD {f4(r['auc_smaj_alone_fold_sd'])}), direct {f4(r['auc_smaj_alone_direct'])}; agreement alone fold mean {f4(r['auc_agree_alone'])}", "", "| set | AUC (fold SD) |", "|---|---|"]
    L += [f"| {k} | {f4(v['auc'])} ({f4(v['auc_fold_sd'])}) |" for k, v in r["sets"].items()] + ["", "| delta | fold mean (SD) | OOF-avg delta | bootstrap 95% CI | boot SD |", "|---|---|---|---|---|"]
    L += [f"| {k} | {d['delta']:+.4f} ({f4(d['delta_fold_sd'])}) | {d['oof_avg_delta']:+.4f} | [{d['boot_ci95'][0]:+.4f}, {d['boot_ci95'][1]:+.4f}] | {f4(d['boot_sd'])} |" for k, d in r["deltas"].items()]
    h = r["nhr_top_half"]; L += ["", f"NHR on top half of ACC by s_maj: n {h['n']}, n_err {h['n_err']}, base rate {h['base_rate']:.4f}"] + [f"  cell{k}: ratio {h[f'cell{k}']['ratio']:.3f}, null mean {h[f'cell{k}']['null_mean']:.3f}, p95 {h[f'cell{k}']['null_p95']:.3f}" for k in CELLS] + [""]
L += ["## judge overall", json.dumps(res["judge_overall"], indent=1), "", "## claim 3", json.dumps(res["claim3"], indent=1, default=float)]
open(os.path.join(a.out, "analyze_table.txt"), "w").write("\n".join(L)); print("\n".join(L), flush=True); print(f"analysis done in {time.time()-t0:.0f}s", flush=True)
