"""L3b critic check 5 (CPU): POST HOC, non-gating replication of the claim 3 statistics on the run 1 scores (prompt 'Reply Yes or No.', batched; run 1 failed the mass and batch gates and was not analysed by the run).
Question: do the neighbor ratio and the AUC delta move when the judge prompt changes (Spearman 0.936 between run 1 and run 2 s_maj)? Same definitions as l3b_analyze.py primary (tau = 0)."""
import numpy as np, pandas as pd, json
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import RepeatedStratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.metrics import roc_auc_score
L = "/homes/rdubash/avrl_live/FULL/avrl/runs/L3b-judge-proxy"; R = "/homes/rdubash/avrl_live/FULL/avrl/runs"
F = pd.read_parquet(f"{R}/E21-lowT-confident-errors/feats_T0.25.parquet").sort_values("pid").reset_index(drop=True)
out = {}
for tag, fn in (("run1_promptv1_batched", "scores_run1_batched_promptv1.parquet"), ("run2_primary", "scores.parquet")):
    S = pd.read_parquet(f"{L}/{fn}"); M = S[S.role == "maj"].set_index("pid").sort_index(); O = S[S.role == "oth"].set_index("pid").sort_index()
    D = F.copy(); D["s_maj"] = M.s.values; D["s_oth"] = O.s.values; D["s_diff"] = D.s_maj - D.s_oth; acc = D.s_maj.values > 0; D["J"] = (acc & (D.majority_correct.values == 0)).astype(int)
    A = D[acc].reset_index(drop=True); y = A.J.values
    J0 = ["s_maj", "s_oth", "s_diff", "mean_tokens", "std_tokens", "any_truncated", "log_qlen"]; JA = J0 + ["agreement", "n_distinct"]; EMB = [f"pc{i}" for i in range(16)]
    mk = lambda: make_pipeline(StandardScaler(), LogisticRegression(max_iter=1000, class_weight="balanced"))
    folds = list(RepeatedStratifiedKFold(n_splits=5, n_repeats=3, random_state=0).split(np.zeros(len(y)), y)); au = {k: [] for k in ("JA", "JA_emb")}
    for tr, te in folds:
        for k, c in (("JA", JA), ("JA_emb", JA + EMB)): au[k].append(roc_auc_score(y[te], mk().fit(A[c].values[tr], y[tr]).predict_proba(A[c].values[te])[:, 1]))
    def nhr(lab, g):
        df = pd.DataFrame({"y": lab.astype(float), "g": g}); base = df.y.mean(); s = df.groupby("g").y.agg(["sum", "count"]); df = df.join(s, on="g"); m = df[df["count"] > 1]
        loo = (m["sum"] - m["y"]) / (m["count"] - 1); return float(m.loc[loo > 0, "y"].mean() / base) if (loo > 0).any() and base > 0 else float("nan")
    top = A[A.s_maj >= A.s_maj.median()]; rr = np.random.default_rng(0); o = {"n_acc": int(len(A)), "n_err": int(y.sum()), "jaccard": float(y.sum() / (D.proxy_error.sum())), "JA": float(np.mean(au["JA"])), "JA_emb": float(np.mean(au["JA_emb"])),
                                                                         "delta": float(np.mean(np.array(au["JA_emb"]) - np.array(au["JA"]))), "top_half_n_err": int(top.J.sum())}
    for ck in (30, 100, 300):
        ob = nhr(top.J.values, top[f"cell{ck}"].values); nl = np.array([nhr(rr.permutation(top.J.values), top[f"cell{ck}"].values) for _ in range(600)]); nl = nl[np.isfinite(nl)]
        o[f"nhr_cell{ck}"] = {"obs": ob, "null_p95": float(np.percentile(nl, 95)), "frac_null_ge": float((nl >= ob).mean())}
    out[tag] = o
json.dump(out, open(f"{L}/critic_run1_replication.json", "w"), indent=1); print(json.dumps(out, indent=1))
