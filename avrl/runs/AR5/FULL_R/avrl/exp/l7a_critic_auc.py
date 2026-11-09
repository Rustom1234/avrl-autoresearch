import numpy as np, pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
cj="agreement,n_distinct,mean_tokens,std_tokens,any_truncated,log_qlen,judge_top,judge_margin,judge_mean,judge_std".split(",")
P="/homes/rdubash/avrl_live/FULL_R/avrl/runs/E04/feats_%s.parquet"
a=pd.read_parquet(P%"Qwen2.5-3B-Instruct"); b=pd.read_parquet(P%"Qwen2.5-7B-Instruct-AWQ")
print(np.array_equal(a.pid.values,b.pid.values))
Xa=a[cj].fillna(0).values; Xb=b[cj].fillna(0).values; ya=a.judge_error.values; yb=b.judge_error.values
def gz(X): return (X-X.mean(0))/np.where(X.std(0)>0,X.std(0),1)
Za,Zb=gz(Xa),gz(Xb)
for cw in (None,"balanced"):
    ma=LogisticRegression(max_iter=2000,class_weight=cw).fit(Za,ya); mb=LogisticRegression(max_iter=2000,class_weight=cw).fit(Zb,yb)
    print(cw,"3B->7B transfer",roc_auc_score(yb,ma.decision_function(Zb)),"in-sample 7B",roc_auc_score(yb,mb.decision_function(Zb)),
          "7B->3B transfer",roc_auc_score(ya,mb.decision_function(Za)),"in-sample 3B",roc_auc_score(ya,ma.decision_function(Za)))
    print("coef 3B",dict(zip(cj,np.round(ma.coef_[0],2)))); print("coef 7B",dict(zip(cj,np.round(mb.coef_[0],2))))
    # sign flip
    ij=[cj.index("judge_top"),cj.index("judge_mean")]; ik=ij+[cj.index("judge_std"),cj.index("n_distinct")]
    for idx,name in ((ij,"top+mean"),(ik,"top,mean,std,ndist")):
        Zb2=Zb.copy(); Zb2[:,idx]*=-1; Za2=Za.copy(); Za2[:,idx]*=-1
        print(name,"3B->7B flipped target",roc_auc_score(yb,ma.decision_function(Zb2)),"7B->3B flipped target",roc_auc_score(ya,mb.decision_function(Za2)))
print("corr labels",np.corrcoef(ya,yb)[0,1],"err",ya.mean(),yb.mean(),"judge_top means",a.judge_top.mean(),b.judge_top.mean())
