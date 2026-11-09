import numpy as np, pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.metrics import roc_auc_score
cj="agreement,n_distinct,mean_tokens,std_tokens,any_truncated,log_qlen,judge_top,judge_margin,judge_mean,judge_std".split(",")
P="/homes/rdubash/avrl_live/FULL_R/avrl/runs/E04/feats_%s.parquet"
a=pd.read_parquet(P%"Qwen2.5-3B-Instruct"); b=pd.read_parquet(P%"Qwen2.5-7B-Instruct-AWQ")
Xa=a[cj].fillna(0).values; Xb=b[cj].fillna(0).values; ya=a.judge_error.values; yb=b.judge_error.values
print("feature mean/sd 3B vs 7B (judge cols):")
for c in cj[6:]: print(c, round(a[c].mean(),3), round(a[c].std(),3), "|", round(b[c].mean(),3), round(b[c].std(),3))
ma=make_pipeline(StandardScaler(),LogisticRegression(max_iter=2000,class_weight="balanced")).fit(Xa,ya)
mb=make_pipeline(StandardScaler(),LogisticRegression(max_iter=2000,class_weight="balanced")).fit(Xb,yb)
print("RAW (no re-standardization) transfer 3B->7B", roc_auc_score(yb,ma.decision_function(Xb)), "7B->3B", roc_auc_score(ya,mb.decision_function(Xa)))
# Per-batch-style: within a 250-batch top-62/125 hit rate using transferred rule: raw vs per-table-z
