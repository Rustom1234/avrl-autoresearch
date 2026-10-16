"""For each candidate label, test SOFisher's two data-side conditions:
(C3) autocorrelation within domain/host (neighbor ratio, oracle concentration, ICC),
(C2) weakness of cheap features (cross-validated AUC of logistic regression on cheap features).
Writes runs/hyp/result.json and prints a markdown table."""
import os, sys, json
import numpy as np, pandas as pd
ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, ROOT)
from aa import data as D
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline

s = pd.read_parquet(os.path.join(ROOT, "runs", "hyp", "scores.parquet"))
dom = s[s.stratum == "domain"].copy()
uni = s[s.stratum == "uniform"].copy()
q = lambda col, p: float(np.quantile(uni[col], p))
labels = {
    "strong_pii": lambda d: d.strong_pii.values,
    "edu_low(<=1)": lambda d: (d.edu_score.round().clip(0, 5) <= 1).astype(int).values,
    "edu_high(>=3)": lambda d: (d.edu_score.round().clip(0, 5) >= 3).astype(int).values,
    "nll_top10(hard)": lambda d: (d.nll >= q("nll", 0.9)).astype(int).values,
    "nll_bot10(easy)": lambda d: (d.nll <= q("nll", 0.1)).astype(int).values,
}
cheap = ["token_count", "language_score", "email", "phone", "ipv4", "street", "any_pii"]
dom["log_tc"] = np.log1p(dom.token_count); cheap.append("log_tc")

def icc(x, g):
    df = pd.DataFrame({"x": x, "g": g}); gm = df.groupby("g")["x"].transform("mean")
    return float(1 - ((df.x - gm) ** 2).mean() / df.x.var())

def concentration(y, g):
    df = pd.DataFrame({"y": y, "g": g}); agg = df.groupby("g")["y"].agg(["sum", "count"])
    agg = agg.sort_values("sum", ascending=False)  # rank domains by number of hits (oracle)
    cum_docs = agg["count"].cumsum() / agg["count"].sum(); cum_hits = agg["sum"].cumsum() / max(agg["sum"].sum(), 1)
    return {f"top{p}pct_docs": float(cum_hits[cum_docs >= p / 100].iloc[0]) for p in (1, 3, 10)}

out = {"n_domain_stratum": int(len(dom)), "n_uniform": int(len(uni)), "n_domains": int(dom.domain.nunique()),
       "icc_domain": {"edu_score": icc(dom.edu_score.values, dom.domain.values), "nll": icc(dom.nll.values, dom.domain.values)}}
rows = []
for name, f in labels.items():
    y = f(dom); yu = f(uni)
    r_dom = D.neighbor_hit_ratio(y, dom.domain.values); r_host = D.neighbor_hit_ratio(y, dom.host.values)
    X = dom[cheap].fillna(0).values
    auc = float(cross_val_score(make_pipeline(StandardScaler(), LogisticRegression(max_iter=500)), X, y, cv=5, scoring="roc_auc").mean()) if 0 < y.mean() < 1 else float("nan")
    conc = concentration(y, dom.domain.values)
    rec = {"label": name, "base_uniform": float(yu.mean()), "base_stratified": float(y.mean()),
           "ratio_domain": r_dom["ratio"], "ratio_host": r_host["ratio"], "cheap_auc": auc, **conc}
    out[name] = rec; rows.append(rec)
    print(name, json.dumps(rec), flush=True)
json.dump(out, open(os.path.join(ROOT, "runs", "hyp", "result.json"), "w"), indent=1)
print("\n| label | base rate | neighbor ratio (domain) | neighbor ratio (host) | cheap-feature AUC | top 10% domains hold |")
print("|---|---|---|---|---|---|")
for r in rows:
    print(f"| {r['label']} | {r['base_uniform']:.3f} | {r['ratio_domain']:.2f}x | {r['ratio_host']:.2f}x | {r['cheap_auc']:.2f} | {r['top10pct_docs']*100:.0f}% |")
print("ICC by domain:", json.dumps(out["icc_domain"]))
