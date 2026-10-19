"""L7a critic: independent replay. CPU only. Imports nothing from the experiment modules.

Own code for: the sim split, the 200-label frozen config B table, plain fixed, and greedy_prior (StandardScaler + balanced
LogisticRegression on the six cheap features, fitted on the subset's 200 p0 labels, refit every R = max(b // 8, 40) verified
labels on p0 labels + verified eval labels). Checks
  (1) the 20 registered subsets x splits 0-7 x 8 tie seeds against runs/L7a/result.json (subset means, to 1e-6 pp),
  (2) 48 fresh eval cuts (bucket-shuffle seeds 100-147) of each full pool, mean over the 20 tables (greedy: tie seed 0, the
      greedy hits do not depend on the tie seed; frozen and fixed: tie seeds 0-2),
  (3) the L4b registered test rows (pid <= 1318, as in L6a): 20 subsets x splits 0-7 x 8 tie seeds.
Usage: python -I l7a_critic_indep.py   (prints; the log is runs/L7a_critic/indep.log)
"""
import sys, json, numpy as np, pandas as pd
sys.dont_write_bytecode = True
from joblib import Parallel, delayed
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline

R = "/homes/rdubash/avrl_live/ROUTED/avrl/runs"
C = ["agreement", "n_distinct", "mean_tokens", "std_tokens", "any_truncated", "log_qlen"]; ST = ["agreement", "n_distinct", "any_truncated"]
p0 = pd.read_parquet(f"{R}/p0/feats.parquet")


def sim_split(df, split):
    cnt = df["cell100"].value_counts(); df = df[df["cell100"].isin(cnt[cnt >= 3].index)].reset_index(drop=True)
    g = df["cell100"].unique(); np.random.default_rng(split).shuffle(g); fit = set(g[: len(g) // 4])
    return df[~df["cell100"].isin(fit)].reset_index(drop=True)


def scorer(): return make_pipeline(StandardScaler(), LogisticRegression(max_iter=1000, class_weight="balanced"))


def greedy(y, X, Xp, yp, b, seed):
    rnd = np.random.default_rng(seed).random(len(y)); Rr = max(b // 8, 40)
    seen = np.zeros(len(y), bool); sc = scorer().fit(Xp, yp).predict_proba(X)[:, 1]; n = 0; nf = Rr; hits = 0
    while n < b:
        mu = np.flatnonzero(~seen); c = mu[np.lexsort((rnd[mu], -sc[mu]))[0]]; seen[c] = True; hits += y[c]; n += 1
        if n >= nf:
            sc = scorer().fit(np.vstack([Xp, X[seen]]), np.concatenate([yp, y[seen]])).predict_proba(X)[:, 1]; nf += Rr
    return int(hits)


subs = []
for i in range(20):
    idx = np.random.RandomState(100 + i).choice(3000, 200, replace=False); sub = p0.iloc[idx]
    g = sub.groupby(ST)["proxy_error"].agg(["sum", "size"]).reset_index(); g["post"] = (g["sum"] + 50 * (1 - g["agreement"])) / (g["size"] + 50)
    subs.append((g, sub[C].fillna(0).to_numpy(float), sub["proxy_error"].to_numpy(int)))


def gaps_one_cut(sim, n_tie):
    """Per subset: (greedy - frozen, frozen - fixed, greedy - fixed) in pp for one eval cut, tie seeds 0..n_tie-1."""
    y = sim["proxy_error"].to_numpy(int); X = sim[C].fillna(0).to_numpy(float); b = int(0.1 * len(y))
    ag = sim["agreement"].to_numpy(); nd = sim["n_distinct"].to_numpy()
    rnds = [np.random.default_rng(t).random(len(y)) for t in range(n_tie)]; out = []
    for g, Xp, yp in subs:
        m = sim[ST].merge(g[ST + ["post"]], how="left", on=ST)
        score = np.where(m["post"].isna(), 1 - sim["agreement"], m["post"]).astype(float)
        zh = np.mean([y[np.lexsort((r, -nd, ag, -score))[:b]].sum() for r in rnds])
        fh = np.mean([y[np.lexsort((r, -nd, ag))[:b]].sum() for r in rnds]); gh = greedy(y, X, Xp, yp, b, 0)
        out.append((100 * (gh - zh) / b, 100 * (zh - fh) / b, 100 * (gh - fh) / b))
    return np.array(out)  # subset x 3


def registered(ev, ref=None):
    A = np.stack(Parallel(n_jobs=8)(delayed(lambda s: gaps_one_cut(sim_split(ev, s), 8))(s) for s in range(8)))  # split x subset x 3
    return A.mean(axis=0)  # subset x 3


def fresh(ev, cuts):
    res = Parallel(n_jobs=16)(delayed(lambda c: gaps_one_cut(sim_split(ev, c), 3).mean(axis=0))(c) for c in cuts)
    return np.array(res)  # cut x 3


L7 = json.load(open(f"{R}/L7a/result.json"))
for pool in ["L2b", "L4b"]:
    ev = pd.read_parquet(f"{R}/{pool}/feats.parquet")
    M = registered(ev); ref = np.array([d[f"{pool}_greedy_minus_frozen_pp"] for d in L7["subset_detail"]])
    print(f"(1) {pool} registered replay: greedy-frozen mean {M[:,0].mean():+.4f} sd {M[:,0].std(ddof=1):.4f} subsets>+1.0 {(M[:,0] > 1).sum()} "
          f"| frozen-fixed {M[:,1].mean():+.4f} greedy-fixed {M[:,2].mean():+.4f} | max abs diff to L7a subset means {np.abs(M[:,0] - ref).max():.2e} pp "
          f"(ledger {L7['pools'][pool]['mean_pp']:+.4f})", flush=True)
    F = fresh(ev, range(100, 148))
    print(f"(2) {pool} 48 fresh cuts: greedy-frozen mean {F[:,0].mean():+.2f} sd {F[:,0].std(ddof=1):.2f} min {F[:,0].min():+.2f} max {F[:,0].max():+.2f} "
          f"cuts above +1.0: {(F[:,0] > 1).sum()} of 48", flush=True)
ev = pd.read_parquet(f"{R}/L4b/feats.parquet"); et = ev[ev["pid"] <= 1318]
M = registered(et)
print(f"(3) L4b test rows (pid<=1318, {len(et)} rows): greedy-frozen mean {M[:,0].mean():+.2f} sd over subsets {M[:,0].std(ddof=1):.2f} "
      f"subsets>+1.0 {(M[:,0] > 1).sum()} of 20 min {M[:,0].min():+.2f} max {M[:,0].max():+.2f} | frozen-fixed {M[:,1].mean():+.2f} (L6a +1.22) "
      f"greedy-fixed {M[:,2].mean():+.2f}", flush=True)
F = fresh(et, range(100, 148))
print(f"(3b) L4b test rows, 48 fresh cuts: greedy-frozen mean {F[:,0].mean():+.2f} sd {F[:,0].std(ddof=1):.2f} min {F[:,0].min():+.2f} max {F[:,0].max():+.2f} "
      f"cuts above +1.0: {(F[:,0] > 1).sum()} of 48", flush=True)
