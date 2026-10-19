"""L10b critic: independent replay. CPU only. Imports nothing from the experiment modules.

Own code for: the sim split, the 200-label frozen config B table, plain fixed, the five 160-label fold tables, the 5-fold
out-of-fold posterior for the 200 p0 rows, the matched eval-row posterior (one fold draw per eval row) and the table-start greedy
learner (StandardScaler + balanced LogisticRegression on six cheap features + posterior, refit every R = max(b // 8, 40) verified
labels on the 200 p0 rows + verified eval rows). Checks
  (1) the 20 registered subsets x splits 0-7 x 8 tie seeds against runs/L10b/result.json (per-split gaps, to 1e-6 pp), draw seed 400 + i,
  (2) sensitivity to the one assumed eval fold draw: draw seeds 1400 + i, 2400 + i, 3400 + i (splits 0-7, tie seeds 0-2),
  (3) 48 fresh eval cuts (bucket-shuffle seeds 100-147) of each full pool, mean over the 20 tables, tie seeds 0-2, draw seed 400 + i,
  (4) leakage probe: a refit that is given the gold-free features only is what the experiment does; here the probe checks that the
      matched score of every eval row is identical when that row's label is flipped (it must not depend on eval labels).
Usage: python -I l10b_critic_indep.py [part ...]   (parts: 1 2 3 4; prints; log runs/L10b_critic/indep.log)
"""
import sys, json, numpy as np, pandas as pd
sys.dont_write_bytecode = True
from joblib import Parallel, delayed
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline

R = "/homes/rdubash/avrl_live/ROUTED/avrl/runs"
C = ["agreement", "n_distinct", "mean_tokens", "std_tokens", "any_truncated", "log_qlen"]; ST = ["agreement", "n_distinct", "any_truncated"]
NJ = 8
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


subs = []  # per subset: (200-label table, five 160-label fold tables, Xp, yp)
for i in range(20):
    idx = np.random.RandomState(100 + i).choice(3000, 200, replace=False); sub = p0.iloc[idx].reset_index(drop=True)
    g = table(sub)
    fold = np.empty(200, int); fold[np.random.RandomState(300 + i).permutation(200)] = np.arange(200) % 5
    tabs = [table(sub[fold != f]) for f in range(5)]
    oof = np.empty(200)
    for f in range(5):
        oof[fold == f] = score_with(sub[fold == f], tabs[f])
    Xp = np.column_stack([sub[C].fillna(0).to_numpy(float), oof]); yp = sub["proxy_error"].to_numpy(int)
    subs.append((g, tabs, Xp, yp))


def matched_all(ev, draw_off):
    """Row-aligned matched posterior per subset: M[i, r] is the posterior of the 160-label fold table drawn for eval row r."""
    M = np.empty((20, len(ev)))
    for i, (g, tabs, Xp, yp) in enumerate(subs):
        d = np.random.RandomState(draw_off + i).randint(0, 5, size=len(ev))
        for f in range(5):
            m = d == f
            M[i, m] = score_with(ev[m], tabs[f])
    return M


def gaps_one_cut(sim, M, tie_seeds):
    """Per subset: (refit - frozen, matched zero - frozen, refit - fixed, frozen - fixed) in pp, mean over tie seeds."""
    y = sim["proxy_error"].to_numpy(int); b = int(0.1 * len(y)); rows = sim["_row"].to_numpy()
    ag = sim["agreement"].to_numpy(); nd = sim["n_distinct"].to_numpy(); Xc = sim[C].fillna(0).to_numpy(float)
    out = []; nrefs = set()
    for i, (g, tabs, Xp, yp) in enumerate(subs):
        fro = score_with(sim, g); mat = M[i, rows]; X = np.column_stack([Xc, mat])
        zh, fh, lh, frh = [], [], [], []
        for t in tie_seeds:
            r = np.random.default_rng(t).random(len(y))
            frh.append(y[np.lexsort((r, -nd, ag, -fro))[:b]].sum()); fh.append(y[np.lexsort((r, -nd, ag))[:b]].sum())
            h0, _ = greedy(y, ag, nd, X, mat, Xp, yp, b, t, False)
            assert h0 == y[np.lexsort((r, -nd, ag, -mat))[:b]].sum(), "zero-refit greedy != frozen_pick on the matched score"
            zh.append(h0)
            h, nr = greedy(y, ag, nd, X, mat, Xp, yp, b, t, True); lh.append(h); nrefs.add(nr)
        zh, fh, lh, frh = np.mean(zh), np.mean(fh), np.mean(lh), np.mean(frh)
        out.append((100 * (lh - frh) / b, 100 * (zh - frh) / b, 100 * (lh - fh) / b, 100 * (frh - fh) / b))
    return np.array(out), len(y), b, nrefs  # subset x 4


def registered(ev, M, tie_seeds):
    ev = ev.assign(_row=np.arange(len(ev)))
    res = Parallel(n_jobs=NJ)(delayed(lambda s: gaps_one_cut(sim_split(ev, s), M, tie_seeds))(s) for s in range(8))
    A = np.stack([r[0] for r in res]); ns = [r[1] for r in res]; bs = [r[2] for r in res]; nr = sorted(set().union(*[r[3] for r in res]))
    return A, ns, bs, nr  # split x subset x 4


def fresh(ev, M, cuts):
    ev = ev.assign(_row=np.arange(len(ev)))
    res = Parallel(n_jobs=NJ)(delayed(lambda c: gaps_one_cut(sim_split(ev, c), M, range(3))[0].mean(axis=0))(c) for c in cuts)
    return np.array(res)  # cut x 4


if __name__ == "__main__":
    args = sys.argv[1:]; pools = [a for a in args if a in ("L2b", "L4b")] or ["L2b", "L4b"]
    parts = {a for a in args if a not in ("L2b", "L4b")} or {"1", "2", "3", "4"}
    L10 = json.load(open(f"{R}/L10b/result.json")); L9 = json.load(open(f"{R}/L9a/result.json"))
    for pool in pools:
        ev = pd.read_parquet(f"{R}/{pool}/feats.parquet"); M = matched_all(ev, 400)
        if "1" in parts:
            A, ns, bs, nr = registered(ev, M, range(8)); S = A.mean(axis=0)  # subset x 4
            refs = np.array([d[f"{pool}_table_refit_minus_frozen_pp_per_split"] for d in L10["subset_detail"]])  # subset x split
            print(f"(1) {pool} registered replay: refit-frozen mean {S[:,0].mean():+.4f} sd {S[:,0].std(ddof=1):.4f} subsets>+1.0 {(S[:,0] > 1).sum()} "
                  f"| zero-frozen {S[:,1].mean():+.4f} | refit-fixed {S[:,2].mean():+.4f} frozen-fixed {S[:,3].mean():+.4f} | max abs diff to L10b per-split gaps "
                  f"{np.abs(A[:,:,0].T - refs).max():.2e} pp (ledger {L10['pools'][pool]['mean_pp']:+.4f}); refit below its zero start in {(S[:,0] < S[:,1]).sum()} of 20 subsets", flush=True)
            print(f"    n_sim per split {min(ns)}..{max(ns)}; budget {min(bs)}..{max(bs)}; refits per run {nr}", flush=True)
        if "2" in parts:
            for off in (1400, 2400, 3400):
                A, ns, bs, nr = registered(ev, matched_all(ev, off), range(3)); S = A.mean(axis=0)
                print(f"(2) {pool} draw seed {off}+i: refit-frozen mean {S[:,0].mean():+.2f} sd {S[:,0].std(ddof=1):.2f} subsets>+1.0 {(S[:,0] > 1).sum()} "
                      f"| zero-frozen {S[:,1].mean():+.2f} | refit-fixed {S[:,2].mean():+.2f}", flush=True)
        if "3" in parts:
            F = fresh(ev, M, range(100, 148))
            print(f"(3) {pool} 48 fresh cuts (draw 400+i): refit-frozen mean {F[:,0].mean():+.2f} sd {F[:,0].std(ddof=1):.2f} min {F[:,0].min():+.2f} max {F[:,0].max():+.2f} "
                  f"cuts above +1.0: {(F[:,0] > 1).sum()} of 48 | zero-frozen {F[:,1].mean():+.2f} | refit-fixed {F[:,2].mean():+.2f} frozen-fixed {F[:,3].mean():+.2f}", flush=True)
        if "4" in parts:
            # matched score never reads an eval label: recompute with every eval label flipped and compare
            ev2 = ev.assign(proxy_error=1 - ev["proxy_error"].to_numpy(int)); M2 = matched_all(ev2, 400)
            print(f"(4) {pool} matched score unchanged when all eval labels are flipped: {bool(np.array_equal(M, M2))}; "
                  f"eval columns used by the learner: {C} (no gold, majority, pass_rate or proxy_error column); range [{M.min():.4f}, {M.max():.4f}]", flush=True)
