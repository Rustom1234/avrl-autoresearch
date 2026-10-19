"""Phase 0 analysis: does cheap-proxy error (majority answer wrong) have exploitable structure, and which
allocation policy catches the most proxy errors per expensive verifier call at a fixed budget?
Writes <run_dir>/feats.parquet, result.json, and prints the tables.
E10 copy of avrl/p0_analyze.py. Changed: run dir from argv[1], the ROOT depth for avrl/exp/, and the allocation replay uses the existing p0 sim_alloc.py."""
import os, sys, json, subprocess
import numpy as np, pandas as pd
ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))); sys.path.insert(0, ROOT)  # one level deeper than p0 (avrl/exp/)
from aa import data as D
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.metrics import roc_auc_score
P = os.path.abspath(sys.argv[1]); os.makedirs(P, exist_ok=True)
SIM = os.path.join(ROOT, "avrl", "runs", "p0", "sim_alloc.py")
G = pd.read_parquet(os.path.join(P, "groups.parquet")); R = pd.read_parquet(os.path.join(P, "rollouts.parquet"))
from sentence_transformers import SentenceTransformer
emb = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2", device="cuda").encode(G.question.tolist(), batch_size=256, normalize_embeddings=True, show_progress_bar=False)
for k in (30, 100, 300): G[f"cell{k}"] = KMeans(n_clusters=k, n_init=3, random_state=0).fit_predict(emb)
pcs = PCA(n_components=16, random_state=0).fit_transform(emb)
for i in range(16): G[f"pc{i}"] = pcs[:, i]
G["log_qlen"] = np.log1p(G.q_len); G["split"] = (G.agreement < 1).astype(int)
cheap = ["agreement", "n_distinct", "mean_tokens", "std_tokens", "any_truncated", "log_qlen"]
embc = [f"pc{i}" for i in range(16)]
y = G.proxy_error.values
auc = lambda cols: float(cross_val_score(make_pipeline(StandardScaler(), LogisticRegression(max_iter=1000, class_weight="balanced")), G[cols].values, y, cv=5, scoring="roc_auc").mean())
def conc(yv, g):
    agg = pd.DataFrame({"y": yv, "g": g}).groupby("g")["y"].agg(["sum", "count"]).sort_values("sum", ascending=False)
    cd = agg["count"].cumsum() / agg["count"].sum(); ch = agg["sum"].cumsum() / max(agg["sum"].sum(), 1)
    return float(ch[cd >= 0.10].iloc[0])
out = {"n_groups": int(len(G)), "proxy_error_rate": float(y.mean()), "unanimous_frac": float((G.agreement == 1).mean()),
       "proxy_error_unanimous": float(y[G.agreement == 1].mean()), "proxy_error_split": float(y[G.agreement < 1].mean()),
       "auc_agreement_only": float(roc_auc_score(y, -G.agreement.values)), "auc_cheap": auc(cheap), "auc_cheap_emb": auc(cheap + embc), "auc_emb_only": auc(embc),
       "neighbor_ratio": {f"cell{k}": D.neighbor_hit_ratio(y, G[f"cell{k}"].values)["ratio"] for k in (30, 100, 300)},
       "top10pct_cells_hold": {f"cell{k}": conc(y, G[f"cell{k}"].values) for k in (30, 100, 300)}}
# structure of proxy error among CONFIDENT groups only (the ones a threshold rule never verifies): this is what a learned allocator must exploit
conf = G[G.agreement >= 0.75]; yc = conf.proxy_error.values
out["confident_subset"] = {"n": int(len(conf)), "proxy_error_rate": float(yc.mean()), "auc_cheap": auc.__call__(cheap) if False else float(cross_val_score(make_pipeline(StandardScaler(), LogisticRegression(max_iter=1000, class_weight="balanced")), conf[cheap].values, yc, cv=5, scoring="roc_auc").mean()),
                          "auc_cheap_emb": float(cross_val_score(make_pipeline(StandardScaler(), LogisticRegression(max_iter=1000, class_weight="balanced")), conf[cheap + embc].values, yc, cv=5, scoring="roc_auc").mean()),
                          "neighbor_ratio_cell100": D.neighbor_hit_ratio(yc, conf.cell100.values)["ratio"]}
G.to_parquet(os.path.join(P, "feats.parquet"), index=False)
json.dump(out, open(os.path.join(P, "result.json"), "w"), indent=1); print(json.dumps(out, indent=1), flush=True)
# budgeted allocation replay: hits = proxy-error groups caught by verification at 1/3/10% ... use 10/25/50% budgets
for g in ("cell100", "cell30"):
    for name, cols in (("cheap", cheap), ("cheap+emb", cheap + embc)):
        print(f"\n=== allocation replay: label proxy_error, buckets {g}, features {name} (K=1 group per visit)", flush=True)
        subprocess.run([sys.executable, SIM, os.path.join(P, "feats.parquet"), "proxy_error", g, ",".join(cols), os.path.join(P, f"sim_{g}_{name}.json")])
