"""L9a critic: is the split-half in-sample optimism caused by repeated prompts, or by the small, class-imbalanced distinct labelled set? Re-runs carry_plain on the iid stream (functions exec'd from exp/l9a_horizon.py's
definitions, main section not executed), then refits the split-half rule (a) with the label multiplicities of the learner's list L (as memo() does) and (b) once per distinct prompt (no repeats in the fit).
Usage: l9a_critic_split.py <judge> <B> <seed>   -> prints one JSON line"""
import sys, json
import numpy as np
j, B, seed = sys.argv[1], int(sys.argv[2]), int(sys.argv[3])
CJ = "agreement,n_distinct,mean_tokens,std_tokens,any_truncated,log_qlen,judge_top,judge_margin,judge_mean,judge_std"
src = open("/homes/rdubash/avrl_live/FULL_R/avrl/exp/l9a_horizon.py").read()
cut = src.index('if MODE == "disjoint12":\n    perm')
sys.argv = ["x", f"runs/E04/feats_{j}.parquet", "judge_error", CJ, str(B), str(seed), "iid", "60", "/dev/null"]
g = {"__name__": "critic"}; exec(compile(src[:cut], "l9a_horizon_prefix", "exec"), g)
N, M, yall, Xall, rule, roc_auc_score, carry_plain, Zg, IA = (g[k] for k in ("N", "M", "yall", "Xall", "rule", "roc_auc_score", "carry_plain", "Zg", "IA"))
srng = np.random.default_rng(30000 + seed); batches = [srng.integers(0, N, M) for _ in range(60)]
_, _, L = carry_plain(Xall, yall, batches, B, seed, Zg[:, IA]); L = np.asarray(L); S = np.unique(L)
auc = lambda sc, ix: float(roc_auc_score(yall[ix], sc[ix]))
res = {"judge": j, "B": B, "seed": seed, "n_labels": len(L), "n_distinct": len(S), "err_rate_S": float(yall[S].mean()), "n_neg_S": int((yall[S] == 0).sum())}
gm, gu = [], []
for r in range(20):
    h = np.random.default_rng(50000 + 100 * seed + r).permutation(S); A, Bh = h[: len(S) // 2], h[len(S) // 2:]
    inA = np.isin(L, A); pm = rule(Xall[L[inA]], yall[L[inA]], 500).predict_proba(Xall)[:, 1]; gm.append(auc(pm, A) - auc(pm, Bh))
    pu = rule(Xall[A], yall[A], 500).predict_proba(Xall)[:, 1]; gu.append(auc(pu, A) - auc(pu, Bh))
res.update(gap_multiplicity_mean20=float(np.mean(gm)), gap_unit_weight_mean20=float(np.mean(gu)))
print(json.dumps(res))
