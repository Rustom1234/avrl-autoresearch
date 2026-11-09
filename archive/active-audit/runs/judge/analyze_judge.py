"""D4: geometry of the LLM-judge contextual-sensitivity label on 20k FineWeb docs.
Neighbor ratios by domain / host / embedding cell, oracle concentration, cheap-feature AUC with and without
leave-one-out domain context from the 10-shard regex rates, and judge-vs-regex agreement."""
import os, sys, json, glob
import numpy as np, pandas as pd
ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, ROOT)
from aa import data as D
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.cluster import KMeans
from sklearn.metrics import roc_auc_score

J = pd.read_parquet(os.path.join(ROOT, "runs", "judge", "labels_20k.parquet"))
df = D.load_fineweb(n=len(J), seed=0, columns=("id", "url", "text", "token_count", "language_score"))
assert (df.id.values == J.id.values).all(), "sample mismatch"
df["judge"] = J.judge_sensitive.values; df["cat"] = J.judge_category.values
df = D.add_regex_features(df); df = D.add_domains(df)
df["host"] = df.url.str.extract(r"^https?://([^/]+)")[0].str.lower()
y = df.judge.values.astype(int)
print("n", len(df), "judge base", y.mean(), "regex strong_pii base", df.strong_pii.mean(), flush=True)

# leave-one-out domain / host context from the 10-shard regex features
F = pd.concat([pd.read_parquet(f, columns=["domain", "host", "strong_pii"]) for f in glob.glob(os.path.join(ROOT, "runs", "density", "feats_*.parquet"))], ignore_index=True)
for g in ("domain", "host"):
    agg = F.groupby(g)["strong_pii"].agg(["sum", "count"])
    s = df[g].map(agg["sum"]).fillna(0).values; c = df[g].map(agg["count"]).fillna(0).values
    df[f"{g}_ctx_rate"] = np.where(c > 1, (s - df.strong_pii.values) / np.maximum(c - 1, 1), np.nan)
    df[f"{g}_ctx_n"] = c
del F

# embeddings -> cells
from sentence_transformers import SentenceTransformer
emb = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2", device="cuda").encode([t[:2000] for t in df.text.astype(str)], batch_size=256, show_progress_bar=False, normalize_embeddings=True)
for k in (100, 400):
    df[f"cell{k}"] = KMeans(n_clusters=k, n_init=3, random_state=0).fit_predict(emb)

def concentration(yv, g):
    agg = pd.DataFrame({"y": yv, "g": g}).groupby("g")["y"].agg(["sum", "count"]).sort_values("sum", ascending=False)
    cd = agg["count"].cumsum() / agg["count"].sum(); ch = agg["sum"].cumsum() / max(agg["sum"].sum(), 1)
    return {f"top{p}pct": float(ch[cd >= p / 100].iloc[0]) for p in (1, 3, 10)}

cheap = ["token_count", "language_score", "email", "phone", "ipv4", "street", "ssn", "credit", "secret", "any_pii"]
df["log_tc"] = np.log1p(df.token_count); cheap.append("log_tc")
ctx = cheap + ["domain_ctx_rate", "host_ctx_rate", "domain_ctx_n", "host_ctx_n"]
auc = lambda cols: float(cross_val_score(make_pipeline(StandardScaler(), LogisticRegression(max_iter=1000, class_weight="balanced")), df[cols].fillna(df[cols].median()).values, y, cv=5, scoring="roc_auc").mean())
out = {"n": int(len(df)), "judge_base": float(y.mean()), "regex_base": float(df.strong_pii.mean()),
       "categories": df.cat.value_counts().to_dict(),
       "neighbor_ratio": {g: D.neighbor_hit_ratio(y, df[g].values)["ratio"] for g in ("domain", "host", "cell100", "cell400")},
       "neighbor_ratio_regex_same_sample": {g: D.neighbor_hit_ratio(df.strong_pii.values, df[g].values)["ratio"] for g in ("domain", "host", "cell100", "cell400")},
       "concentration_domain": concentration(y, df.domain.values), "concentration_cell400": concentration(y, df.cell400.values),
       "auc_cheap": auc(cheap), "auc_cheap_plus_domain_context": auc(ctx),
       "auc_regex_email_count_only": float(roc_auc_score(y, df.email.values)),
       "regex_vs_judge": {"precision_of_regex": float(y[df.strong_pii.values == 1].mean()), "recall_of_regex": float(df.strong_pii.values[y == 1].mean())},
       "frac_docs_with_domain_context": float(np.isfinite(df.domain_ctx_rate).mean())}
json.dump(out, open(os.path.join(ROOT, "runs", "judge", "result.json"), "w"), indent=1)
df.drop(columns=["text"]).to_parquet(os.path.join(ROOT, "runs", "judge", "feats_20k.parquet"), index=False)
print(json.dumps(out, indent=1))
