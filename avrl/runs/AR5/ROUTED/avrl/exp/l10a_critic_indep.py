"""L10a critic: independent replay. CPU only. Imports nothing from the experiment modules.

Own code for: the sim split, the 200-label frozen config B table, plain fixed, and the table-pooled greedy learner
(per-stratum score (e + 50 * table_post) / (v + 50), recomputed every R = max(b // 8, 40) verified labels). Checks
  (1) the 20 registered subsets x splits 0-7 x 8 tie seeds against runs/L10a/result.json (subset means and per-split gaps),
  (2) 48 fresh eval cuts (bucket-shuffle seeds 100-147) of each full pool, mean over the 20 tables, tie seeds 0-2,
  (3) the L4b registered test rows (pid <= 1318) as a stand-in replay, registered cut and 48 fresh cuts,
  (4) a leakage probe: scores computed with the eval labels of unverified docs shuffled must give identical picks.
Usage: python -I l10a_critic_indep.py   (prints; the log is runs/L10a_critic/indep.log)
"""
import sys, json, numpy as np, pandas as pd
sys.dont_write_bytecode = True
from joblib import Parallel, delayed

R = "/homes/rdubash/avrl_live/ROUTED/avrl/runs"
ST = ["agreement", "n_distinct", "any_truncated"]
p0 = pd.read_parquet(f"{R}/p0/feats.parquet")


def sim_split(df, split):
    cnt = df["cell100"].value_counts(); df = df[df["cell100"].isin(cnt[cnt >= 3].index)].reset_index(drop=True)
    g = df["cell100"].unique(); np.random.default_rng(split).shuffle(g); fit = set(g[: len(g) // 4])
    return df[~df["cell100"].isin(fit)].reset_index(drop=True)


def table(sub, k0=50.0):
    g = sub.groupby(ST)["proxy_error"].agg(["sum", "size"]).reset_index()
    g["post"] = (g["sum"] + k0 * (1 - g["agreement"])) / (g["size"] + k0)
    return g


def score_with(df, g):
    m = df[ST].merge(g[ST + ["post"]], how="left", on=ST)
    return np.where(m["post"].isna(), 1 - df["agreement"].to_numpy(float), m["post"]).astype(float)


def greedy(y, ag, nd, code, init, b, seed, update, k0=50.0, y_for_update=None):
    """Own loop. y_for_update lets the leakage probe feed a different label vector to the update; hits always use y."""
    yu = y if y_for_update is None else y_for_update
    rnd = np.random.default_rng(seed).random(len(y)); Rr = max(b // 8, 40)
    seen = np.zeros(len(y), bool); sc = init.copy(); n = 0; nf = Rr; hits = 0; nup = 0; picks = []
    K = code.max() + 1
    while n < b:
        mu = np.flatnonzero(~seen); c = mu[np.lexsort((rnd[mu], -nd[mu], ag[mu], -sc[mu]))[0]]
        seen[c] = True; hits += y[c]; n += 1; picks.append(c)
        if update and n >= nf and n < b:
            e = np.zeros(K); v = np.zeros(K); ix = np.flatnonzero(seen)
            np.add.at(e, code[ix], yu[ix].astype(float)); np.add.at(v, code[ix], 1.0)
            sc = (e[code] + k0 * init) / (v[code] + k0); nf += Rr; nup += 1
    return int(hits), nup, picks


subs = []
for i in range(20):
    idx = np.random.RandomState(100 + i).choice(3000, 200, replace=False)
    subs.append(table(p0.iloc[idx].reset_index(drop=True)))


def gaps_one_cut(sim, tie_seeds):
    """Per subset: (pooled - frozen, frozen - fixed, pooled - fixed) in pp, mean over tie seeds."""
    y = sim["proxy_error"].to_numpy(int); b = int(0.1 * len(y))
    ag = sim["agreement"].to_numpy(); nd = sim["n_distinct"].to_numpy()
    code = sim.groupby(ST).ngroup().to_numpy(int)
    out = []; nups = set()
    for g in subs:
        init = score_with(sim, g)
        zh, fh, ph = [], [], []
        for t in tie_seeds:
            r = np.random.default_rng(t).random(len(y))
            zh.append(y[np.lexsort((r, -nd, ag, -init))[:b]].sum()); fh.append(y[np.lexsort((r, -nd, ag))[:b]].sum())
            h0, _, _ = greedy(y, ag, nd, code, init, b, t, False); assert h0 == zh[-1], "zero-update greedy != frozen"
            h, nu, _ = greedy(y, ag, nd, code, init, b, t, True); ph.append(h); nups.add(nu)
        zh, fh, ph = np.mean(zh), np.mean(fh), np.mean(ph)
        out.append((100 * (ph - zh) / b, 100 * (zh - fh) / b, 100 * (ph - fh) / b))
    return np.array(out), len(y), b, nups


def registered(ev):
    res = Parallel(n_jobs=8)(delayed(lambda s: gaps_one_cut(sim_split(ev, s), range(8)))(s) for s in range(8))
    A = np.stack([r[0] for r in res]); ns = [r[1] for r in res]; bs = [r[2] for r in res]; nu = sorted(set().union(*[r[3] for r in res]))
    return A, ns, bs, nu  # split x subset x 3


def fresh(ev, cuts):
    res = Parallel(n_jobs=24)(delayed(lambda c: gaps_one_cut(sim_split(ev, c), range(3))[0].mean(axis=0))(c) for c in cuts)
    return np.array(res)  # cut x 3


def leak_probe(ev):
    """Pick sequence must not depend on the labels of unverified docs: permute y among docs not picked, picks must be equal."""
    sim = sim_split(ev, 0); y = sim["proxy_error"].to_numpy(int); b = int(0.1 * len(y))
    ag = sim["agreement"].to_numpy(); nd = sim["n_distinct"].to_numpy(); code = sim.groupby(ST).ngroup().to_numpy(int)
    init = score_with(sim, subs[0]); bad = 0
    for t in range(4):
        h, _, p = greedy(y, ag, nd, code, init, b, t, True)
        rng = np.random.default_rng(900 + t); y2 = y.copy(); rest = np.setdiff1d(np.arange(len(y)), p); y2[rest] = rng.permutation(y[rest])
        h2, _, p2 = greedy(y2, ag, nd, code, init, b, t, True)
        bad += int(p != p2)
    return bad


if __name__ == "__main__":
    L10 = json.load(open(f"{R}/L10a/result.json"))
    for pool in ["L2b", "L4b"]:
        ev = pd.read_parquet(f"{R}/{pool}/feats.parquet")
        A, ns, bs, nu = registered(ev); M = A.mean(axis=0)
        ref = np.array([d[f"{pool}_table_pooled_minus_frozen_pp"] for d in L10["subset_detail"]])
        refs = np.array([d[f"{pool}_table_pooled_minus_frozen_pp_per_split"] for d in L10["subset_detail"]])  # subset x split
        print(f"(1) {pool} registered replay: pooled-frozen mean {M[:,0].mean():+.4f} sd {M[:,0].std(ddof=1):.4f} subsets>+1.0 {(M[:,0] > 1).sum()} "
              f"| frozen-fixed {M[:,1].mean():+.4f} pooled-fixed {M[:,2].mean():+.4f} | max abs diff to L10a subset means {np.abs(M[:,0] - ref).max():.2e} pp, "
              f"to per-split gaps {np.abs(A[:,:,0].T - refs).max():.2e} pp (ledger {L10['pools'][pool]['mean_pp']:+.4f})", flush=True)
        print(f"    n_sim per split {min(ns)}..{max(ns)}; budget {min(bs)}..{max(bs)}; updates per run {nu}", flush=True)
        F = fresh(ev, range(100, 148))
        print(f"(2) {pool} 48 fresh cuts: pooled-frozen mean {F[:,0].mean():+.2f} sd {F[:,0].std(ddof=1):.2f} min {F[:,0].min():+.2f} max {F[:,0].max():+.2f} "
              f"cuts above +1.0: {(F[:,0] > 1).sum()} of 48 | frozen-fixed {F[:,1].mean():+.2f} pooled-fixed {F[:,2].mean():+.2f}", flush=True)
        print(f"(4) {pool} leakage probe, runs whose picks changed when unverified labels were permuted: {leak_probe(ev)} of 4", flush=True)
    ev = pd.read_parquet(f"{R}/L4b/feats.parquet"); et = ev[ev["pid"] <= 1318]
    A, ns, bs, nu = registered(et); M = A.mean(axis=0)
    print(f"(3) L4b test rows (pid<=1318, {len(et)} rows): pooled-frozen mean {M[:,0].mean():+.2f} sd over subsets {M[:,0].std(ddof=1):.2f} "
          f"subsets>+1.0 {(M[:,0] > 1).sum()} of 20 min {M[:,0].min():+.2f} max {M[:,0].max():+.2f} | frozen-fixed {M[:,1].mean():+.2f} (L6a +1.22) "
          f"pooled-fixed {M[:,2].mean():+.2f}; budgets {min(bs)}..{max(bs)}, updates {nu}", flush=True)
    F = fresh(et, range(100, 148))
    print(f"(3b) L4b test rows, 48 fresh cuts: pooled-frozen mean {F[:,0].mean():+.2f} sd {F[:,0].std(ddof=1):.2f} min {F[:,0].min():+.2f} max {F[:,0].max():+.2f} "
          f"cuts above +1.0: {(F[:,0] > 1).sum()} of 48", flush=True)
