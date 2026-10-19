"""L5c critic check 2 (CPU): recompute the AUC route (JA vs JA + MiniLM pc0..15, 5x3 CV, balanced logistic) from feats_J.parquet with separately written code."""
import numpy as np, pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import RepeatedStratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.metrics import roc_auc_score
Q = "/homes/rdubash/avrl_live/FULL/avrl/runs/"
emb = [f"pc{i}" for i in range(16)]; J0 = ["s_maj", "s_oth", "s_diff", "mean_tokens", "std_tokens", "any_truncated", "log_qlen"]; JA = J0 + ["agreement", "n_distinct"]
for nm, p in [("rm", "L5c-proxies/rm"), ("judge3b", "L5c-proxies/judge3b"), ("qwen3_4b", "L3b-judge-proxy")]:
    d = pd.read_parquet(Q + p + "/feats_J.parquet"); A = d[d.accept == 1].reset_index(drop=True); y = A.judge_error.values.astype(int)
    folds = list(RepeatedStratifiedKFold(n_splits=5, n_repeats=3, random_state=0).split(np.zeros(len(y)), y)); res = {}
    for k, cols in {"J0": J0, "J0_emb": J0 + emb, "JA": JA, "JA_emb": JA + emb}.items():
        a = []
        for tr, te in folds:
            m = make_pipeline(StandardScaler(), LogisticRegression(max_iter=1000, class_weight="balanced")).fit(A[cols].values[tr], y[tr]); a.append(roc_auc_score(y[te], m.predict_proba(A[cols].values[te])[:, 1]))
        res[k] = np.mean(a)
    print(nm, len(A), y.sum(), {k: round(v, 4) for k, v in res.items()}, "delta JA", round(res["JA_emb"] - res["JA"], 4), "delta J0", round(res["J0_emb"] - res["J0"], 4))
