"""L5a analysis: identical structure pipeline to e04_structure.py (cheap-feature + embedding AUC,
neighbor-hit ratio, Jaccard overlap vs majority-vote proxy_error), pointed at the consistency-free
judge_scores.parquet produced by e05a_judge_score_notie.py --tie_break random instead of E04's
consistency-weighted judge_scores.parquet.

Copied from e04_structure.py (not an edit of it). No re-embedding: reuses the cell30/cell100/cell300
and pc0-15 columns already computed in avrl/runs/p0/feats.parquet.

Writes avrl/runs/L5a/result.json.

Usage:
  e05a_structure.py --groups avrl/runs/p0/groups.parquet --feats avrl/runs/p0/feats.parquet \
      --judge_scores avrl/runs/L5a/judge_scores.parquet --out avrl/runs/L5a/result.json
"""
import argparse, json, os, sys, time
import numpy as np, pandas as pd
ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))); sys.path.insert(0, ROOT)
from aa import data as D
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline

ap = argparse.ArgumentParser()
ap.add_argument("--groups", default="avrl/runs/p0/groups.parquet")
ap.add_argument("--feats", required=True)
ap.add_argument("--judge_scores", required=True)
ap.add_argument("--out", required=True)
a = ap.parse_args(); t0 = time.time()
os.makedirs(os.path.dirname(a.out), exist_ok=True)

Gd = pd.read_parquet(a.groups)  # majority-vote's own groups table, used to cross-check pid coverage
F = pd.read_parquet(a.feats)  # cheap + cell30/100/300 + pc0-15 columns, already computed by p0_analyze.py (includes proxy_error)
J = pd.read_parquet(a.judge_scores)  # pid, judge_pick, judge_error, judge_pick_share, judge_all_no

assert set(F.pid) == set(Gd.pid), "feats.parquet and groups.parquet cover different pids"
df = F.merge(J[["pid", "judge_error", "judge_pick_share"]], on="pid", how="inner")
assert len(df) == len(F), f"judge_scores did not cover all {len(F)} feats groups (got {len(df)})"

cheap = ["agreement", "n_distinct", "mean_tokens", "std_tokens", "any_truncated", "log_qlen"]
embc = [f"pc{i}" for i in range(16)]
y = df.judge_error.values
auc = lambda cols: float(cross_val_score(make_pipeline(StandardScaler(), LogisticRegression(max_iter=1000, class_weight="balanced")), df[cols].values, y, cv=5, scoring="roc_auc").mean())
a_cheap, a_cheap_emb, a_emb = auc(cheap), auc(cheap + embc), auc(embc)

# judge-confident subset: judge_pick's own self-consistency share >= 0.75, analogous to E01's agreement>=0.75 cut
conf = df[df.judge_pick_share >= 0.75]
yc = conf.judge_error.values

# Jaccard overlap between {majority-vote wrong} and {judge wrong}
A = set(df.pid[df.proxy_error == 1]); B = set(df.pid[df.judge_error == 1])
inter, union = len(A & B), len(A | B)
jaccard = float(inter / union) if union else float("nan")

auc_gap = a_cheap_emb - a_cheap
out = {
    "id": "L5a",
    "n_groups": int(len(df)),
    "judge_error_rate": float(y.mean()),
    "n_judge_error": int(y.sum()),
    "confident_subset": {"n": int(len(conf)), "min_share": 0.75, "judge_error_rate": float(yc.mean()) if len(conf) else None},
    "auc_cheap": a_cheap,
    "auc_cheap_emb": a_cheap_emb,
    "auc_cheap_emb_minus_cheap": auc_gap,
    "auc_emb_only": a_emb,
    "neighbor_ratio": {f"cell{k}": D.neighbor_hit_ratio(y, df[f"cell{k}"].values)["ratio"] for k in (30, 100, 300)},
    "majority_vote_overlap": {
        "n_majority_error": len(A), "n_judge_error": len(B), "n_intersection": inter, "n_union": union,
        "jaccard": jaccard,
    },
}
# pre-registered thresholds (backlog L5a section): pass = Jaccard < 0.8 (tally-artifact explanation confirmed,
# claim 3 reopened for judge proxies); kill = Jaccard >= 0.8 (judge and majority genuinely fail on the same
# prompts regardless of aggregation -- a stronger kill than E04's own).
pass_ = bool(jaccard < 0.8)
kill = bool(jaccard >= 0.8)
out["verdict"] = "KILL" if kill else "PASS"
out["pass"] = pass_
out["kill"] = kill
out["runtime_sec"] = time.time() - t0

json.dump(out, open(a.out, "w"), indent=1)
print(json.dumps(out, indent=1), flush=True)
