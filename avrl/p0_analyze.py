"""Phase 0 of verification-aware RL, step 2 of 2: analyze the rollouts written by p0_rollouts.py.

Two questions, both about proxy error (a rollout group's majority answer is wrong, so the cheap
majority-vote reward would mislead training):
  1. Does proxy error have exploitable structure? Measured by how well cheap, label-free features
     predict it (cross-validated AUC) and by how strongly it clusters among similar prompts
     (neighbor ratio, and the share of errors held by the worst 10% of prompt cells).
  2. Which allocation policy catches the most proxy errors per expensive verifier call at a fixed
     budget? Measured by replaying allocation policies offline on the saved groups.

Inputs: avrl/runs/p0/groups.parquet and rollouts.parquet.
Outputs in avrl/runs/p0: feats.parquet (groups plus features), result.json (question 1), sim_alloc.py
and sim_<buckets>_<features>.json (question 2). The tables are also printed.

Dependencies outside avrl/: aa.data.neighbor_hit_ratio and the allocation replay simulator
sim_generic.py, both part of the earlier project kept under archive/active-audit/. This script looks
for them at <repo>/aa and <repo>/runs/sim_generic.py, so those two paths must be present at the
repository root to rerun it.

Run (after p0_rollouts.py; needs a GPU for the sentence embedder): python avrl/p0_analyze.py
"""
import os, sys, json, subprocess
import numpy as np, pandas as pd
# repository root, put on sys.path so the `aa` package resolves (see the module docstring)
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__))); sys.path.insert(0, ROOT)
from aa import data as D
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.metrics import roc_auc_score
P = os.path.join(ROOT, "avrl", "runs", "p0")
# G: one row per prompt (rollout group). R: one row per rollout (not needed by the group-level analysis below).
G = pd.read_parquet(os.path.join(P, "groups.parquet")); R = pd.read_parquet(os.path.join(P, "rollouts.parquet"))
from sentence_transformers import SentenceTransformer
# --- features ---
# unit-normalized MiniLM sentence embedding of each question
emb = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2", device="cuda").encode(G.question.tolist(), batch_size=256, normalize_embeddings=True, show_progress_bar=False)
# cell{k}: KMeans cluster id of the question embedding at three granularities. A cell is a neighbourhood
# of similar prompts, used to test whether proxy errors cluster and as buckets for the allocation replay.
for k in (30, 100, 300): G[f"cell{k}"] = KMeans(n_clusters=k, n_init=3, random_state=0).fit_predict(emb)
# 16 PCA components of the embedding: a low-dimensional description of the prompt
pcs = PCA(n_components=16, random_state=0).fit_transform(emb)
for i in range(16): G[f"pc{i}"] = pcs[:, i]
G["log_qlen"] = np.log1p(G.q_len); G["split"] = (G.agreement < 1).astype(int)
# cheap features: everything here is available without the answer key (split = the G rollouts disagree)
cheap = ["agreement", "n_distinct", "mean_tokens", "std_tokens", "any_truncated", "log_qlen"]
embc = [f"pc{i}" for i in range(16)]
y = G.proxy_error.values  # target: 1 if the group's majority answer is wrong
# --- question 1: is proxy error predictable, and does it cluster? ---
# 5-fold cross-validated ROC AUC of a class-balanced logistic regression predicting proxy error from `cols`
auc = lambda cols: float(cross_val_score(make_pipeline(StandardScaler(), LogisticRegression(max_iter=1000, class_weight="balanced")), G[cols].values, y, cv=5, scoring="roc_auc").mean())
def conc(yv, g):
    """Concentration of errors: sort cells g by error count, take the top cells until they cover 10% of
    all groups, and return the share of all errors yv they hold (0.10 would mean no clustering)."""
    agg = pd.DataFrame({"y": yv, "g": g}).groupby("g")["y"].agg(["sum", "count"]).sort_values("sum", ascending=False)
    cd = agg["count"].cumsum() / agg["count"].sum(); ch = agg["sum"].cumsum() / max(agg["sum"].sum(), 1)
    return float(ch[cd >= 0.10].iloc[0])
# auc_agreement_only scores groups by -agreement directly (no fitted model): low agreement should predict
# error. neighbor_ratio = P(error | another group in the same cell is an error) / P(error); 1.0 = no clustering.
out = {"n_groups": int(len(G)), "proxy_error_rate": float(y.mean()), "unanimous_frac": float((G.agreement == 1).mean()),
       "proxy_error_unanimous": float(y[G.agreement == 1].mean()), "proxy_error_split": float(y[G.agreement < 1].mean()),
       "auc_agreement_only": float(roc_auc_score(y, -G.agreement.values)), "auc_cheap": auc(cheap), "auc_cheap_emb": auc(cheap + embc), "auc_emb_only": auc(embc),
       "neighbor_ratio": {f"cell{k}": D.neighbor_hit_ratio(y, G[f"cell{k}"].values)["ratio"] for k in (30, 100, 300)},
       "top10pct_cells_hold": {f"cell{k}": conc(y, G[f"cell{k}"].values) for k in (30, 100, 300)}}
# structure of proxy error among CONFIDENT groups only (the ones a threshold rule never verifies): this is what a learned allocator must exploit
# (in "auc_cheap" below the `if False` arm is never evaluated; the value is the cross-validated AUC on the confident subset)
conf = G[G.agreement >= 0.75]; yc = conf.proxy_error.values
out["confident_subset"] = {"n": int(len(conf)), "proxy_error_rate": float(yc.mean()), "auc_cheap": auc.__call__(cheap) if False else float(cross_val_score(make_pipeline(StandardScaler(), LogisticRegression(max_iter=1000, class_weight="balanced")), conf[cheap].values, yc, cv=5, scoring="roc_auc").mean()),
                          "auc_cheap_emb": float(cross_val_score(make_pipeline(StandardScaler(), LogisticRegression(max_iter=1000, class_weight="balanced")), conf[cheap + embc].values, yc, cv=5, scoring="roc_auc").mean()),
                          "neighbor_ratio_cell100": D.neighbor_hit_ratio(yc, conf.cell100.values)["ratio"]}
G.to_parquet(os.path.join(P, "feats.parquet"), index=False)
json.dump(out, open(os.path.join(P, "result.json"), "w"), indent=1); print(json.dumps(out, indent=1), flush=True)
# --- question 2: budgeted allocation replay ---
# Reuses the generic replay simulator from the earlier project. A copy of its source is written to
# sim_alloc.py with one constants line patched: budgets 10/25/50% of groups instead of its default 1/3/10%,
# and K=1 group verified per visit instead of 4. A hit = a proxy-error group caught by verification.
sim = open(os.path.join(ROOT, "runs", "sim_generic.py")).read().replace("BUDGET_FRACS = 3, 4, 8, (0.01, 0.03, 0.10)", "BUDGET_FRACS = 3, 1, 8, (0.10, 0.25, 0.50)")
open(os.path.join(P, "sim_alloc.py"), "w").write(sim)
# one replay per (bucket granularity, feature set); each run prints its table and writes sim_<g>_<name>.json
for g in ("cell100", "cell30"):
    for name, cols in (("cheap", cheap), ("cheap+emb", cheap + embc)):
        print(f"\n=== allocation replay: label proxy_error, buckets {g}, features {name} (K=1 group per visit)", flush=True)
        subprocess.run([sys.executable, os.path.join(P, "sim_alloc.py"), os.path.join(P, "feats.parquet"), "proxy_error", g, ",".join(cols), os.path.join(P, f"sim_{g}_{name}.json")])
