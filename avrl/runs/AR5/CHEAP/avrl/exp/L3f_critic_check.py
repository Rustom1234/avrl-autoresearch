#!/usr/bin/env python3
"""L3f critic check (CPU, read-only on runs/). Recomputes every L3f ledger number from the raw files:
pool stats from groups/rollouts parquet, leakage and overlap checks, claim-3 AUCs and neighbor ratios from the
stored features, the E03 main arms (8 seeds) by re-running the unchanged allocator code, two bootstrap resamples
against the logged values, the E01 references, and the three-way verdicts. Writes nothing to disk."""
import os, sys, time
import numpy as np, pandas as pd
ROOT = "/homes/rdubash/avrl_live/CHEAP"
AVRL = os.path.join(ROOT, "avrl")
L3F = os.path.join(AVRL, "runs", "L3f")
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(AVRL, "exp"))
from aa import data as D
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.metrics import roc_auc_score
import e03_real_allocators as E

G = pd.read_parquet(os.path.join(L3F, "pool", "groups.parquet"))
R = pd.read_parquet(os.path.join(L3F, "pool", "rollouts.parquet"))
F = pd.read_parquet(os.path.join(L3F, "pool", "feats.parquet"))
P0 = pd.read_parquet(os.path.join(AVRL, "runs", "p0", "groups.parquet"))
print("== pool stats (groups/rollouts)")
print("groups", len(G), "unique questions", G.question.nunique(), "proxy_error sum", int(G.proxy_error.sum()))
print("unanimous groups", int((G.agreement == 1).sum()), "proxy err when unanimous", float(G.proxy_error[G.agreement == 1].mean()),
      "split groups", int((G.agreement < 1).sum()), "proxy err when split", float(G.proxy_error[G.agreement < 1].mean()))
print("rollout acc", round(R.correct.mean(), 6), "majority acc", round(G.majority_correct.mean(), 6),
      "rollout proxy err", round((R.proxy != R.correct).mean(), 6), "group proxy err", round(G.proxy_error.mean(), 6))
rec_agree = R.groupby("pid").answer.agg(lambda s: s.value_counts().iloc[0] / len(s))
print("agreement recomputed from rollouts == stored:", bool(np.allclose(rec_agree.values, G.agreement.values)))
print("empty-answer rollouts", int((R.answer == "").sum()), "truncated rollouts", int(R.truncated.sum()),
      "frac truncated", round(R.truncated.mean(), 4))
print("unanimous groups with empty answer", int(((G.agreement == 1) & (G.majority == "")).sum()))
print("== leakage / overlap")
print("feature cols in feats:", [c for c in F.columns if c in ("pass_rate", "correct", "majority_correct", "gold", "proxy_error")][:5], "(gold-derived cols present in table but not features)")
print("E03 FEAT_COLS contain gold-derived names:", any(c in E.FEAT_COLS for c in ("pass_rate", "correct", "majority_correct", "gold", "proxy_error")))
print("question overlap with p0:", len(set(G.question) & set(P0.question)), "p0 unique", P0.question.nunique())
print("feats agreement == groups agreement:", bool(np.allclose(F.agreement.values, G.agreement.values)))

print("== claim 3 recomputed from stored features")
cheap = ["agreement", "n_distinct", "mean_tokens", "std_tokens", "any_truncated", "log_qlen"]
embc = [f"pc{i}" for i in range(16)]
y = F.proxy_error.values
def auc(X, yy):
    return float(cross_val_score(make_pipeline(StandardScaler(), LogisticRegression(max_iter=1000, class_weight="balanced")), X, yy, cv=5, scoring="roc_auc").mean())
ac, ace = auc(F[cheap].values, y), auc(F[cheap + embc].values, y)
print("auc_cheap", round(ac, 6), "auc_cheap_emb", round(ace, 6), "gain", round(ace - ac, 6))
print("cell100 ratio", round(D.neighbor_hit_ratio(y, F.cell100.values)["ratio"], 6), "cell30", round(D.neighbor_hit_ratio(y, F.cell30.values)["ratio"], 6),
      "cell300", round(D.neighbor_hit_ratio(y, F.cell300.values)["ratio"], 6))
C = F[F.agreement >= 0.75]; yc = C.proxy_error.values
print("confident n", len(C), "err", round(yc.mean(), 6), "gain", round(auc(C[cheap + embc].values, yc) - auc(C[cheap].values, yc), 6),
      "cell100", round(D.neighbor_hit_ratio(yc, C.cell100.values)["ratio"], 6))
print("n cell100 clusters", F.cell100.nunique())

print("== E03 main arms re-run (8 seeds, budget 0.25, unchanged allocator code)")
pool = E.load_pool(os.path.join(L3F, "pool", "feats.parquet"))
t0 = time.time()
rows = {arm: [E.run_arm(pool, arm, 0.25, s) for s in range(8)] for arm in E.ARMS}
print("wall", round(time.time() - t0, 1), "s")
calls = {arm: sorted(set(r["calls"] for r in rows[arm])) for arm in rows}
print("calls per run by arm:", calls)
for arm in rows:
    print(arm, "caught", [r["caught"] for r in rows[arm]], "mean", np.mean([r["caught"] for r in rows[arm]]),
          "admitted mean", round(np.mean([r["admitted"] for r in rows[arm]]), 6))
l4 = [a["admitted"] - t["admitted"] for a, t in zip(rows["learned"][:4], rows["threshold"][:4])]
g4 = [a["admitted"] - t["admitted"] for a, t in zip(rows["greedy"][:4], rows["threshold"][:4])]
l8 = [a["admitted"] - t["admitted"] for a, t in zip(rows["learned"], rows["threshold"])]
g8 = [a["admitted"] - t["admitted"] for a, t in zip(rows["greedy"], rows["threshold"])]
print("learned-thr seeds0-3", round(np.mean(l4), 6), "seeds0-7", round(np.mean(l8), 6), "per seed", [round(x, 4) for x in l8])
print("greedy-thr seeds0-3", round(np.mean(g4), 6), "seeds0-7", round(np.mean(g8), 6), "per seed", [round(x, 4) for x in g8])

print("== bootstrap spot checks (logged resample values)")
n = len(pool["y"])
logged = {9: (0.4209, 0.0301), 19: (0.4569, 0.0306), 99: (0.4453, 0.0204)}
for b, (lg_l, lg_g) in logged.items():
    ids = np.random.default_rng(10000 + b).choice(n, size=n, replace=True)
    sub = E.subset(pool, ids)
    adm = {arm: [] for arm in E.BOOT_ARMS}
    for s in range(4):
        rs = {arm: E.run_arm(sub, arm, 0.25, s) for arm in E.BOOT_ARMS}
        for arm in E.BOOT_ARMS:
            adm[arm].append(rs[arm]["admitted"])
    dl = float(np.mean([a - t for a, t in zip(adm["learned"], adm["threshold"])]))
    dg = float(np.mean([a - t for a, t in zip(adm["greedy"], adm["threshold"])]))
    print(f"b={b+1}: recomputed learned-thr {dl:+.4f} (logged {lg_l:+.4f}), greedy-thr {dg:+.4f} (logged {lg_g:+.4f})")

print("== verdicts from stored CIs")
def verdict(lo, hi):
    return "KILL" if lo > 0 else ("PASS_CANDIDATE" if hi < 0 else "INCONCLUSIVE")
import json
res = json.load(open(os.path.join(L3F, "result.json")))
ci_l = res["claim2"]["learned_minus_threshold_admitted"]["ci95"]
ci_g = res["claim2"]["greedy_minus_threshold_admitted_diagnostic"]["ci95"]
print("learned", ci_l, verdict(*ci_l), "greedy", ci_g, verdict(*ci_g))
print("claim3 gain<0.05 and ratio<1.5:", (ace - ac) < 0.05 and D.neighbor_hit_ratio(y, F.cell100.values)["ratio"] < 1.5)
print("E01 learned CI", json.load(open(os.path.join(AVRL, "runs", "E03", "result.json")))["metric"]["bootstrap"]["learned_minus_threshold_admitted"]["ci95"])
print("E01 greedy CI", json.load(open(os.path.join(AVRL, "runs", "E03", "result.json")))["metric"]["bootstrap"]["greedy_minus_threshold_admitted"]["ci95"])
