"""E08 analysis: does MBPP proxy error (output-signature majority wrong) have exploitable structure,
mirroring avrl/p0_analyze.py's measures exactly (same AUC/neighbor-ratio/concentration/allocation-replay
machinery) so the two tasks are directly comparable. The only addition is two code-specific cheap
features (frac_syntax_error, frac_exception) that have no GSM8K analogue.
Writes avrl/runs/E08/feats.parquet, result.json; prints the tables."""
import os, sys, json, subprocess
import numpy as np, pandas as pd
ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))); sys.path.insert(0, ROOT)
from aa import data as D
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.metrics import roc_auc_score
P = os.path.join(ROOT, "avrl", "runs", "E08")
G = pd.read_parquet(os.path.join(P, "groups.parquet")); R = pd.read_parquet(os.path.join(P, "rollouts.parquet"))
from sentence_transformers import SentenceTransformer
emb = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2", device="cuda").encode(G.question.tolist(), batch_size=256, normalize_embeddings=True, show_progress_bar=False)
for k in (30, 100, 300): G[f"cell{k}"] = KMeans(n_clusters=k, n_init=3, random_state=0).fit_predict(emb)
pcs = PCA(n_components=16, random_state=0).fit_transform(emb)
for i in range(16): G[f"pc{i}"] = pcs[:, i]
G["log_qlen"] = np.log1p(G.q_len); G["split"] = (G.agreement < 1).astype(int)
cheap = ["agreement", "n_distinct", "mean_tokens", "std_tokens", "any_truncated", "log_qlen", "frac_syntax_error", "frac_exception"]
embc = [f"pc{i}" for i in range(16)]
y = G.proxy_error.values
auc = lambda cols: float(cross_val_score(make_pipeline(StandardScaler(), LogisticRegression(max_iter=1000, class_weight="balanced")), G[cols].values, y, cv=5, scoring="roc_auc").mean())
def conc(yv, g):
    agg = pd.DataFrame({"y": yv, "g": g}).groupby("g")["y"].agg(["sum", "count"]).sort_values("sum", ascending=False)
    cd = agg["count"].cumsum() / agg["count"].sum(); ch = agg["sum"].cumsum() / max(agg["sum"].sum(), 1)
    return float(ch[cd >= 0.10].iloc[0])
out = {"n_groups": int(len(G)), "n_proxy_error_groups": int(y.sum()), "proxy_error_rate": float(y.mean()), "unanimous_frac": float((G.agreement == 1).mean()),
       "proxy_error_unanimous": float(y[G.agreement == 1].mean()), "proxy_error_split": float(y[G.agreement < 1].mean()),
       "auc_agreement_only": float(roc_auc_score(y, -G.agreement.values)), "auc_cheap": auc(cheap), "auc_cheap_emb": auc(cheap + embc), "auc_emb_only": auc(embc),
       "neighbor_ratio": {f"cell{k}": D.neighbor_hit_ratio(y, G[f"cell{k}"].values)["ratio"] for k in (30, 100, 300)},
       "top10pct_cells_hold": {f"cell{k}": conc(y, G[f"cell{k}"].values) for k in (30, 100, 300)}}
# structure of proxy error among CONFIDENT groups only (the ones a threshold rule never verifies): this is what a learned allocator must exploit
conf = G[G.agreement >= 0.75]; yc = conf.proxy_error.values
out["confident_subset"] = {"n": int(len(conf)), "proxy_error_rate": float(yc.mean()),
                          "auc_cheap": float(cross_val_score(make_pipeline(StandardScaler(), LogisticRegression(max_iter=1000, class_weight="balanced")), conf[cheap].values, yc, cv=5, scoring="roc_auc").mean()),
                          "auc_cheap_emb": float(cross_val_score(make_pipeline(StandardScaler(), LogisticRegression(max_iter=1000, class_weight="balanced")), conf[cheap + embc].values, yc, cv=5, scoring="roc_auc").mean()),
                          "neighbor_ratio_cell100": D.neighbor_hit_ratio(yc, conf.cell100.values)["ratio"]}
# pre-registered verdict (PROJECT.md / task spec): pass if embeddings add >= 0.05 AUC over cheap, or neighbor ratio at k=100 >= 1.5;
# kill if both fail; inconclusive if fewer than 60 proxy-error groups (too few to read AUC/neighbor-ratio honestly).
auc_gap = out["auc_cheap_emb"] - out["auc_cheap"]; nr100 = out["neighbor_ratio"]["cell100"]
n_err = out["n_proxy_error_groups"]
if n_err < 60:
    verdict = "inconclusive"
elif auc_gap >= 0.05 or nr100 >= 1.5:
    verdict = "pass"
else:
    verdict = "kill"
out["verdict"] = {"rule": "pass if auc_cheap_emb-auc_cheap>=0.05 or neighbor_ratio_cell100>=1.5; kill if both fail; inconclusive if n_proxy_error_groups<60",
                   "auc_gap": auc_gap, "neighbor_ratio_cell100": nr100, "n_proxy_error_groups": n_err, "label": verdict}
G.to_parquet(os.path.join(P, "feats.parquet"), index=False)
json.dump(out, open(os.path.join(P, "result.json"), "w"), indent=1); print(json.dumps(out, indent=1), flush=True)
# budgeted allocation replay: hits = proxy-error groups caught by verification at 10/25/50% budgets, same as p0_analyze.py
sim = open(os.path.join(ROOT, "runs", "sim_generic.py")).read().replace("MIN, K, SEEDS, BUDGET_FRACS = 3, 4, 8, (0.01, 0.03, 0.10)", "MIN, K, SEEDS, BUDGET_FRACS = 3, 1, 8, (0.10, 0.25, 0.50)")
open(os.path.join(P, "sim_alloc.py"), "w").write(sim)
for g in ("cell100", "cell30"):
    for name, cols in (("cheap", cheap), ("cheap+emb", cheap + embc)):
        print(f"\n=== allocation replay: label proxy_error, buckets {g}, features {name} (K=1 group per visit)", flush=True)
        subprocess.run([sys.executable, os.path.join(P, "sim_alloc.py"), os.path.join(P, "feats.parquet"), "proxy_error", g, ",".join(cols), os.path.join(P, f"sim_{g}_{name}.json")])
