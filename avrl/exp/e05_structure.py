"""E05 analysis: is the self-verification proxy's error pattern (self_verify_error) measurably distinct
from majority-vote's (proxy_error, E01), or a redundant copy of it?

Copied from p0_analyze.py / e04_structure.py's structure block (cheap-feature + embedding AUC,
neighbor-hit ratio), not an edit of either. No re-embedding: reuses the cell30/cell100/cell300 and
pc0-15 columns already computed in avrl/runs/p0/feats.parquet. Adds the Pearson correlation between
self_verify_error and proxy_error at the group level (the pre-registered pass/kill statistic) and the
Jaccard overlap between the two proxies' error sets (extra context, mirrors E04's table).

Writes <out> (result.json).

Usage:
  e05_structure.py --groups avrl/runs/p0/groups.parquet --selfverify avrl/runs/E05/selfverify.parquet \
      --feats avrl/runs/p0/feats.parquet --out avrl/runs/E05/result.json
"""
import argparse, json, os, sys, time
import numpy as np, pandas as pd
from scipy.stats import pearsonr
ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))); sys.path.insert(0, ROOT)
from aa import data as D
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline

ap = argparse.ArgumentParser()
ap.add_argument("--groups", required=True)
ap.add_argument("--selfverify", required=True)
ap.add_argument("--feats", required=True)
ap.add_argument("--out", required=True)
a = ap.parse_args(); t0 = time.time()
os.makedirs(os.path.dirname(a.out), exist_ok=True)

Gd = pd.read_parquet(a.groups)
F = pd.read_parquet(a.feats)  # cheap + cell30/100/300 + pc0-15 columns, already computed by p0_analyze.py (includes proxy_error)
SV = pd.read_parquet(a.selfverify)  # pid, sv_majority, sv_agreement, sv_n_distinct, self_verify_error, sv_gold_correct

assert set(F.pid) == set(Gd.pid), "feats.parquet and groups.parquet cover different pids"
df = F.merge(SV[["pid", "self_verify_error", "sv_agreement", "sv_gold_correct"]], on="pid", how="inner")
assert len(df) == len(F), f"selfverify did not cover all {len(F)} feats groups (got {len(df)})"

cheap = ["agreement", "n_distinct", "mean_tokens", "std_tokens", "any_truncated", "log_qlen"]
embc = [f"pc{i}" for i in range(16)]
y = df.self_verify_error.values
yp = df.proxy_error.values
auc = lambda cols: float(cross_val_score(make_pipeline(StandardScaler(), LogisticRegression(max_iter=1000, class_weight="balanced")), df[cols].values, y, cv=5, scoring="roc_auc").mean())
a_cheap, a_cheap_emb, a_emb = auc(cheap), auc(cheap + embc), auc(embc)

conf = df[df.agreement >= 0.75]  # same confident-subset cut as E01/E04, on the ORIGINAL rollouts' agreement
yc = conf.self_verify_error.values

r, r_p = pearsonr(y, yp)

A = set(df.pid[df.proxy_error == 1]); B = set(df.pid[df.self_verify_error == 1])
inter, union = len(A & B), len(A | B)
jaccard = float(inter / union) if union else float("nan")

sv_rate, proxy_rate = float(y.mean()), float(yp.mean())
diff_pp = (sv_rate - proxy_rate) * 100

out = {
    "id": "E05",
    "n_groups": int(len(df)),
    "self_verify_error_rate": sv_rate,
    "proxy_error_rate": proxy_rate,
    "diff_pp": diff_pp,
    "pearson_r": float(r),
    "pearson_p": float(r_p),
    "auc_cheap": a_cheap,
    "auc_cheap_emb": a_cheap_emb,
    "auc_cheap_emb_minus_cheap": a_cheap_emb - a_cheap,
    "auc_emb_only": a_emb,
    "confident_subset": {"n": int(len(conf)), "min_agreement": 0.75, "self_verify_error_rate": float(yc.mean()) if len(conf) else None},
    "neighbor_ratio": {f"cell{k}": D.neighbor_hit_ratio(y, df[f"cell{k}"].values)["ratio"] for k in (30, 100, 300)},
    "majority_vote_overlap": {"n_proxy_error": len(A), "n_self_verify_error": len(B), "n_intersection": inter, "n_union": union, "jaccard": jaccard},
    "sv_gold_accuracy": float(df.sv_gold_correct.mean()),
}
pass_ = bool(abs(diff_pp) >= 3.0)
kill = bool(r >= 0.9)
out["verdict"] = "KILL" if kill else ("PASS" if pass_ else "INCONCLUSIVE")
out["pass"] = pass_
out["kill"] = kill
out["runtime_sec"] = time.time() - t0

json.dump(out, open(a.out, "w"), indent=1)
print(json.dumps(out, indent=1), flush=True)
