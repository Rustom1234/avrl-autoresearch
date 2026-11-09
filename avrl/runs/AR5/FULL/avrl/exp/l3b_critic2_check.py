"""Second adversarial check of L3b-judge-proxy (CPU). Independent recompute from scores.parquet, rollouts.parquet, groups.parquet, boot JSONs. Writes runs/L3b-judge-proxy/critic2_check.json."""
import sys, json, numpy as np, pandas as pd
sys.path.insert(0, "/homes/rdubash/avrl_live/FULL")
from aa import data as D
from scipy.stats import spearmanr
from sklearn.metrics import roc_auc_score
R = "runs/L3b-judge-proxy"; E = "runs/E21-lowT-confident-errors/T0.25"; out = {}
S = pd.read_parquet(f"{R}/scores.parquet"); G = pd.read_parquet(f"{E}/groups.parquet").sort_values("pid").reset_index(drop=True); RO = pd.read_parquet(f"{E}/rollouts.parquet")
S1 = pd.read_parquet(f"{R}/scores_run1_batched_promptv1.parquet")
M = S[S.role == "maj"].set_index("pid").sort_index(); O = S[S.role == "oth"].set_index("pid").sort_index()
# independent selection of maj / oth from proxy column only
sel_ok_m = sel_ok_o = 0
for pid, g in RO.sort_values(["pid", "k"]).groupby("pid"):
    maj = g[g.proxy == 1].iloc[0]; d = g[g.proxy == 0]; oth = d.iloc[0] if len(d) else g[g.k != maj.k].iloc[0]
    sel_ok_m += int(M.loc[pid, "k"] == maj.k); sel_ok_o += int(O.loc[pid, "k"] == oth.k)
out["selection_match_maj_oth"] = [sel_ok_m, sel_ok_o]
# text actually scored: score script reads text from rollouts, so check correctness columns agree with rollouts
chk = RO.set_index(["pid", "k"])
out["correct_col_match"] = int(sum(chk.loc[(p, k), "correct"] == c for p, k, c in zip(S.pid, S.k, S.correct)))
acc = M.s.values > 0; mvwrong = G.proxy_error.values == 1; J = acc & mvwrong
out.update(n_acc=int(acc.sum()), n_err=int(J.sum()), n_mv_wrong=int(mvwrong.sum()), jaccard=float(J.sum() / (J | mvwrong).sum()))
conf = (G.agreement.values >= 0.75) & mvwrong; out["n_conf_err"] = int(conf.sum()); out["conf_err_judge_accepted_frac"] = float((conf & acc).sum() / conf.sum())
out["sanity_auc_rollout"] = float(roc_auc_score(S.correct, S.s)); out["acc_rate_run1_run2"] = [float((S1[S1.role == 'maj'].set_index('pid').sort_index().s.values > 0).mean()), float(acc.mean())]
s1m = S1[S1.role == "maj"].set_index("pid").sort_index().s.values
out["spearman_run1_run2_smaj"] = float(spearmanr(s1m, M.s.values)[0]); out["accept_agree_run1_run2"] = float(((s1m > 0) == acc).mean())
out["judge_ge_mass"] = float(S.mass.min())
# accuracy of judge-accepted errors by whether wrong majority is right/wrong judged
out["judge_rejects_right_majority"] = int(((~acc) & (G.majority_correct.values == 1)).sum()); out["n_right_majority"] = int((G.majority_correct.values == 1).sum())
# bootstrapped ratios straight from JSON
for tag in ("f0.25", "f0.1", "held_f0.25"):
    fn = f"{R}/boot_J_{tag}.json".replace("boot_J_f0.25", "boot_J_f0.25"); fn = f"{R}/boot_J_{tag}.json"
    j = json.load(open(fn)); rs = j["resamples"]; bf = j["full_pool"]["best_fixed"]
    gw = np.array([np.mean(r["greedy_warm"]) for r in rs]); bx = np.array([np.mean(r[bf]) for r in rs]); gb = np.array([np.mean(r["greedy_blend"]) for r in rs])
    # alt: ratio of resample means vs mean of per-seed ratios
    rat = gw / bx; rat_seed = np.array([np.mean(np.array(r["greedy_warm"]) / np.maximum(np.array(r[bf]), 1)) for r in rs])
    out[tag] = dict(best_fixed=bf, n_boot=len(rs), n_pool=j["n_pool"], ratio_pct=[float(np.percentile(rat, 2.5)), float(np.percentile(rat, 97.5))], ratio_per_seed_pct=[float(np.percentile(rat_seed, 2.5)), float(np.percentile(rat_seed, 97.5))],
                    fp=j["full_pool"]["fixed_mean_at_25"], gw_mean=float(np.mean(j["full_pool"]["greedy_warm"])), sanity=j["sanity"], budgets=sorted(set(r["budget"] for r in rs)),
                    n_err_range=[min(r["n_err"] for r in rs), max(r["n_err"] for r in rs)])
json.dump(out, open(f"{R}/critic2_check.json", "w"), indent=1); print(json.dumps(out, indent=1))
