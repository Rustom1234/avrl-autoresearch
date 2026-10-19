"""L9a critic: independent replay. CPU only. Imports nothing from the experiment modules.

Own code for: the sim split, the 200-label frozen config B table, plain fixed, the 5-fold out-of-fold table posterior and the
table-start greedy learner (StandardScaler + balanced LogisticRegression on six cheap features + table posterior, refit every
R = max(b // 8, 40) verified labels on the 200 p0 rows + verified eval rows). Checks
  (1) the 20 registered subsets x splits 0-7 x 8 tie seeds against runs/L9a/result.json (subset means, to 1e-6 pp),
      plus n_sim / budget / refit counts per split,
  (2) 48 fresh eval cuts (bucket-shuffle seeds 100-147) of each full pool, mean over the 20 tables, tie seeds 0-2,
  (3) the L4b registered test rows (pid <= 1318, as in L6a).
Usage: python -I l9a_critic_indep.py   (prints; the log is runs/L9a_critic/indep.log)
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


def table(sub, k0=50.0):
    g = sub.groupby(ST)["proxy_error"].agg(["sum", "size"]).reset_index()
    g["post"] = (g["sum"] + k0 * (1 - g["agreement"])) / (g["size"] + k0)
    return g


def score_with(df, g):
    m = df[ST].merge(g[ST + ["post"]], how="left", on=ST)
    return np.where(m["post"].isna(), 1 - df["agreement"].to_numpy(float), m["post"]).astype(float)


def greedy(y, ag, nd, X, init, Xp, yp, b, seed, refit):
    rnd = np.random.default_rng(seed).random(len(y)); Rr = max(b // 8, 40)
    seen = np.zeros(len(y), bool); sc = init.copy(); n = 0; nf = Rr; hits = 0; nref = 0
    while n < b:
        mu = np.flatnonzero(~seen); c = mu[np.lexsort((rnd[mu], -nd[mu], ag[mu], -sc[mu]))[0]]; seen[c] = True; hits += y[c]; n += 1
        if refit and n >= nf and n < b:
            sc = scorer().fit(np.vstack([Xp, X[seen]]), np.concatenate([yp, y[seen]])).predict_proba(X)[:, 1]; nf += Rr; nref += 1
    return int(hits), nref


subs = []
for i in range(20):
    idx = np.random.RandomState(100 + i).choice(3000, 200, replace=False); sub = p0.iloc[idx].reset_index(drop=True)
    g = table(sub)
    fold = np.empty(200, int); fold[np.random.RandomState(300 + i).permutation(200)] = np.arange(200) % 5
    oof = np.empty(200)
    for f in range(5):
        oof[fold == f] = score_with(sub[fold == f], table(sub[fold != f]))
    Xp = np.column_stack([sub[C].fillna(0).to_numpy(float), oof]); yp = sub["proxy_error"].to_numpy(int)
    subs.append((g, Xp, yp))


def gaps_one_cut(sim, tie_seeds):
    """Per subset: (refit - frozen, frozen - fixed, refit - fixed) in pp, mean over tie seeds, plus refit counts."""
    y = sim["proxy_error"].to_numpy(int); b = int(0.1 * len(y))
    ag = sim["agreement"].to_numpy(); nd = sim["n_distinct"].to_numpy(); Xc = sim[C].fillna(0).to_numpy(float)
    out = []; nrefs = set()
    for g, Xp, yp in subs:
        init = score_with(sim, g); X = np.column_stack([Xc, init])
        zh, fh, lh = [], [], []
        for t in tie_seeds:
            r = np.random.default_rng(t).random(len(y))
            zh.append(y[np.lexsort((r, -nd, ag, -init))[:b]].sum()); fh.append(y[np.lexsort((r, -nd, ag))[:b]].sum())
            h0, _ = greedy(y, ag, nd, X, init, Xp, yp, b, t, False); assert h0 == zh[-1], "zero-refit greedy != frozen"
            h, nr = greedy(y, ag, nd, X, init, Xp, yp, b, t, True); lh.append(h); nrefs.add(nr)
        zh, fh, lh = np.mean(zh), np.mean(fh), np.mean(lh)
        out.append((100 * (lh - zh) / b, 100 * (zh - fh) / b, 100 * (lh - fh) / b))
    return np.array(out), len(y), b, nrefs  # subset x 3


def registered(ev):
    res = Parallel(n_jobs=8)(delayed(lambda s: gaps_one_cut(sim_split(ev, s), range(8)))(s) for s in range(8))
    A = np.stack([r[0] for r in res]); ns = [r[1] for r in res]; bs = [r[2] for r in res]; nr = sorted(set().union(*[r[3] for r in res]))
    return A, ns, bs, nr  # split x subset x 3


def fresh(ev, cuts):
    res = Parallel(n_jobs=24)(delayed(lambda c: gaps_one_cut(sim_split(ev, c), range(3))[0].mean(axis=0))(c) for c in cuts)
    return np.array(res)  # cut x 3


if __name__ == "__main__":
    L9 = json.load(open(f"{R}/L9a/result.json"))
    for pool in ["L2b", "L4b"]:
        ev = pd.read_parquet(f"{R}/{pool}/feats.parquet")
        A, ns, bs, nr = registered(ev); M = A.mean(axis=0)
        ref = np.array([d[f"{pool}_table_refit_minus_frozen_pp"] for d in L9["subset_detail"]])
        refs = np.array([d[f"{pool}_table_refit_minus_frozen_pp_per_split"] for d in L9["subset_detail"]])  # subset x split
        print(f"(1) {pool} registered replay: refit-frozen mean {M[:,0].mean():+.4f} sd {M[:,0].std(ddof=1):.4f} subsets>+1.0 {(M[:,0] > 1).sum()} "
              f"| frozen-fixed {M[:,1].mean():+.4f} refit-fixed {M[:,2].mean():+.4f} | max abs diff to L9a subset means {np.abs(M[:,0] - ref).max():.2e} pp, "
              f"to per-split gaps {np.abs(A[:,:,0].T - refs).max():.2e} pp (ledger {L9['pools'][pool]['mean_pp']:+.4f})", flush=True)
        print(f"    n_sim per split {min(ns)}..{max(ns)} (list {ns}); budget {min(bs)}..{max(bs)}; refits per run {nr}", flush=True)
        F = fresh(ev, range(100, 148))
        print(f"(2) {pool} 48 fresh cuts: refit-frozen mean {F[:,0].mean():+.2f} sd {F[:,0].std(ddof=1):.2f} min {F[:,0].min():+.2f} max {F[:,0].max():+.2f} "
              f"cuts above +1.0: {(F[:,0] > 1).sum()} of 48 | frozen-fixed {F[:,1].mean():+.2f} refit-fixed {F[:,2].mean():+.2f}", flush=True)
    ev = pd.read_parquet(f"{R}/L4b/feats.parquet"); et = ev[ev["pid"] <= 1318]
    A, ns, bs, nr = registered(et); M = A.mean(axis=0)
    print(f"(3) L4b test rows (pid<=1318, {len(et)} rows): refit-frozen mean {M[:,0].mean():+.2f} sd over subsets {M[:,0].std(ddof=1):.2f} "
          f"subsets>+1.0 {(M[:,0] > 1).sum()} of 20 min {M[:,0].min():+.2f} max {M[:,0].max():+.2f} | frozen-fixed {M[:,1].mean():+.2f} (L6a +1.22) "
          f"refit-fixed {M[:,2].mean():+.2f}; budgets {min(bs)}..{max(bs)}, refits {nr}", flush=True)
    F = fresh(et, range(100, 148))
    print(f"(3b) L4b test rows, 48 fresh cuts: refit-frozen mean {F[:,0].mean():+.2f} sd {F[:,0].std(ddof=1):.2f} min {F[:,0].min():+.2f} max {F[:,0].max():+.2f} "
          f"cuts above +1.0: {(F[:,0] > 1).sum()} of 48", flush=True)
