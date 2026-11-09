import os, sys, json
import numpy as np, pandas as pd
ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, ROOT)
from aa import data as D
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.model_selection import cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
c = pd.read_parquet(os.path.join(ROOT, "runs", "infl", "influence.parquet"))
c["log_tc"] = np.log1p(c.token_count)
cheap = ["token_count", "language_score", "email", "phone", "ipv4", "street", "any_pii", "log_tc"]
rich = cheap + ["edu_score", "nll"]
def icc(x, g):
    df = pd.DataFrame({"x": x, "g": g}); gm = df.groupby("g")["x"].transform("mean")
    return float(1 - ((df.x - gm) ** 2).mean() / df.x.var())
out = {"n": int(len(c)), "n_domains": int(c.domain.nunique())}
rows = []
for col in ["infl_general", "infl_edu", "cos_general", "cos_edu", "grad_norm", "nll_06b"]:
    x = c[col].values
    top = (x >= np.quantile(x, 0.9)).astype(int)   # top-10% most influential (or highest grad norm / loss)
    r = {"metric": col, "icc_domain": icc(x, c.domain.values), "icc_host": icc(x, c.host.values),
         "ratio_top10_domain": D.neighbor_hit_ratio(top, c.domain.values)["ratio"],
         "cheap_auc_top10": float(cross_val_score(make_pipeline(StandardScaler(), LogisticRegression(max_iter=500)), c[cheap].fillna(0).values, top, cv=5, scoring="roc_auc").mean()),
         "rich_auc_top10": float(cross_val_score(make_pipeline(StandardScaler(), LogisticRegression(max_iter=500)), c[rich].fillna(0).values, top, cv=5, scoring="roc_auc").mean()),
         "r2_cheap": float(cross_val_score(make_pipeline(StandardScaler(), Ridge()), c[cheap].fillna(0).values, x, cv=5, scoring="r2").mean()),
         "r2_rich": float(cross_val_score(make_pipeline(StandardScaler(), Ridge()), c[rich].fillna(0).values, x, cv=5, scoring="r2").mean()),
         "corr_edu": float(np.corrcoef(x, c.edu_score)[0, 1]), "corr_nll4b": float(np.corrcoef(x, c.nll)[0, 1])}
    out[col] = r; rows.append(r); print(col, json.dumps(r), flush=True)
json.dump(out, open(os.path.join(ROOT, "runs", "infl", "result.json"), "w"), indent=1)
print("\n| metric | ICC domain | ICC host | top-10% neighbor ratio (domain) | cheap AUC | cheap+edu+nll AUC | R2 cheap | R2 rich | corr edu | corr nll |\n|---|---|---|---|---|---|---|---|---|---|")
for r in rows:
    print(f"| {r['metric']} | {r['icc_domain']:.2f} | {r['icc_host']:.2f} | {r['ratio_top10_domain']:.2f}x | {r['cheap_auc_top10']:.2f} | {r['rich_auc_top10']:.2f} | {r['r2_cheap']:.2f} | {r['r2_rich']:.2f} | {r['corr_edu']:.2f} | {r['corr_nll4b']:.2f} |")
