"""Unified analysis on the 73k domain-stratified sample once judge labels exist: for every label
(judge, edu_low, edu_high, nll_top10, strong_pii) report neighbor ratio by domain / host / embedding cell,
concentration, cheap AUC, cheap+embedding AUC, and run the budgeted-audit simulator by domain and by cell
with an extra 'static_emb' policy (distill-then-scan competitor). Writes runs/hyp/unified.json and unified.md."""
import os, sys, json, subprocess
import numpy as np, pandas as pd
ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, ROOT)
from aa import data as D
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA

s = pd.read_parquet(os.path.join(ROOT, "runs", "hyp", "scores.parquet"))
j = pd.read_parquet(os.path.join(ROOT, "runs", "judge", "labels_hyp.parquet"), columns=["id", "judge_sensitive", "judge_category"])
s = s.merge(j, on="id", how="inner"); print("docs with all labels", len(s), flush=True)
full = D.load_fineweb(columns=("id", "text")); text = dict(zip(full.id, full.text)); del full
from sentence_transformers import SentenceTransformer
emb = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2", device="cuda").encode([str(text[i])[:2000] for i in s.id], batch_size=256, show_progress_bar=False, normalize_embeddings=True)
del text
for k in (300, 1000):
    s[f"cell{k}"] = KMeans(n_clusters=k, n_init=2, random_state=0).fit_predict(emb)
pcs = PCA(n_components=32, random_state=0).fit_transform(emb)
for i in range(32): s[f"pc{i}"] = pcs[:, i]
s["log_tc"] = np.log1p(s.token_count)
s["judge"] = s.judge_sensitive.astype(int)
s["edu_low"] = (s.edu_score.round().clip(0, 5) <= 1).astype(int); s["edu_high"] = (s.edu_score.round().clip(0, 5) >= 3).astype(int)
s["nll_top10"] = (s.nll >= np.quantile(s[s.stratum == "uniform"].nll, 0.9)).astype(int)
uni = s[s.stratum == "uniform"]
cheap = ["token_count", "language_score", "email", "phone", "ipv4", "street", "any_pii", "log_tc"]
embc = [f"pc{i}" for i in range(32)]
auc = lambda d, cols, y: float(cross_val_score(make_pipeline(StandardScaler(), LogisticRegression(max_iter=1000, class_weight="balanced")), d[cols].fillna(0).values, y, cv=5, scoring="roc_auc").mean()) if 0 < y.mean() < 1 else float("nan")
def conc(y, g):
    agg = pd.DataFrame({"y": y, "g": g}).groupby("g")["y"].agg(["sum", "count"]).sort_values("sum", ascending=False)
    cd = agg["count"].cumsum() / agg["count"].sum(); ch = agg["sum"].cumsum() / max(agg["sum"].sum(), 1)
    return float(ch[cd >= 0.10].iloc[0])
dom = s[s.stratum == "domain"]
def multi(y, g):
    h = pd.Series(y).groupby(g).sum(); return {"n_groups_with_hits": int((h > 0).sum()), "frac_hits_in_multi_hit_groups": float(h[h >= 2].sum() / max(h.sum(), 1))}
out = {"n": int(len(s)), "n_domain_stratum": int(len(dom)), "labels": {}}
md = ["| label | base (uniform) | ratio domain | ratio host | ratio cell300 | ratio cell1000 | top10% domains hold | top10% cells hold | AUC cheap | AUC cheap+emb |", "|---|---|---|---|---|---|---|---|---|---|"]
for lab in ["judge", "strong_pii", "edu_low", "edu_high", "nll_top10"]:
    y = dom[lab].values.astype(int)
    r = {"base_uniform": float(uni[lab].mean()), "ratio_domain": D.neighbor_hit_ratio(y, dom.domain.values)["ratio"], "ratio_host": D.neighbor_hit_ratio(y, dom.host.values)["ratio"],
         "ratio_cell300": D.neighbor_hit_ratio(s[lab].values, s.cell300.values)["ratio"], "ratio_cell1000": D.neighbor_hit_ratio(s[lab].values, s.cell1000.values)["ratio"],
         "top10_domains": conc(y, dom.domain.values), "top10_cells": conc(s[lab].values, s.cell300.values), "auc_cheap": auc(s, cheap, s[lab].values), "auc_cheap_emb": auc(s, cheap + embc, s[lab].values), "domain_multi": multi(y, dom.domain.values), "host_multi": multi(y, dom.host.values)}
    out["labels"][lab] = r
    md.append(f"| {lab} | {r['base_uniform']:.4f} | {r['ratio_domain']:.2f}x | {r['ratio_host']:.2f}x | {r['ratio_cell300']:.2f}x | {r['ratio_cell1000']:.2f}x | {r['top10_domains']*100:.0f}% | {r['top10_cells']*100:.0f}% | {r['auc_cheap']:.2f} | {r['auc_cheap_emb']:.2f} |")
    print(md[-1], "| domain hits:", r["domain_multi"], "| host hits:", r["host_multi"], flush=True)
s.to_parquet(os.path.join(ROOT, "runs", "hyp", "unified_feats.parquet"), index=False)
json.dump(out, open(os.path.join(ROOT, "runs", "hyp", "unified.json"), "w"), indent=1)
open(os.path.join(ROOT, "runs", "hyp", "unified.md"), "w").write("\n".join(md) + "\n")
# simulator: cheap-feature static vs adaptive, and distill-then-scan (cheap+emb) static, by domain and by cell
for lab in ["judge", "edu_low", "strong_pii"]:
    for g in ["domain", "cell300"]:
        for name, cols in [("cheap", cheap), ("cheap+emb", cheap + embc)]:
            print(f"\n=== sim {lab} by {g}, static features = {name}", flush=True)
            subprocess.run([sys.executable, os.path.join(ROOT, "runs", "sim_generic.py"), os.path.join(ROOT, "runs", "hyp", "unified_feats.parquet"), lab, g, ",".join(cols), os.path.join(ROOT, "runs", "hyp", f"sim_{lab}_{g}_{name}.json")])
