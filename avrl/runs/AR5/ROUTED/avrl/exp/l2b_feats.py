"""L2b step 2: feature columns for the fresh pool.

Copied from the feature-writing lines of avrl/p0_analyze.py: all-MiniLM-L6-v2 normalized question embeddings,
KMeans cells (30, 100, 300; n_init 3; random_state 0), log_qlen, split. The six cheap features (agreement,
n_distinct, mean_tokens, std_tokens, any_truncated, log_qlen) are already in groups.parquet, with log_qlen computed
here exactly as in p0. Structure statistics, the PCA columns, the neighbor ratios and the sim_alloc replay are not
copied. Writes <run_dir>/feats.parquet.
"""
import argparse, os
import numpy as np, pandas as pd
from sklearn.cluster import KMeans
from sentence_transformers import SentenceTransformer

ap = argparse.ArgumentParser(); ap.add_argument("--run_dir", required=True); a = ap.parse_args()
G = pd.read_parquet(os.path.join(a.run_dir, "groups.parquet"))
emb = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2", device="cuda").encode(G.question.tolist(), batch_size=256, normalize_embeddings=True, show_progress_bar=False)
for k in (30, 100, 300): G[f"cell{k}"] = KMeans(n_clusters=k, n_init=3, random_state=0).fit_predict(emb)
G["log_qlen"] = np.log1p(G.q_len); G["split"] = (G.agreement < 1).astype(int)
G.to_parquet(os.path.join(a.run_dir, "feats.parquet"), index=False)
cnt = G.cell100.value_counts()
print(f"wrote {os.path.join(a.run_dir, 'feats.parquet')}: {len(G)} groups, proxy_error rate {G.proxy_error.mean():.4f}, "
      f"cell100 buckets with >= 3 docs {int((cnt >= 3).sum())}, embedding dim {emb.shape[1]}", flush=True)
