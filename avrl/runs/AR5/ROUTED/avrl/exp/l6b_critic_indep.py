"""L6b critic: independent checks. CPU only. Does NOT import l1b_tuned_threshold, l5a_frozen_table or l6b_frozen_label_budget.
Own code for the table fit, the split build and the picks (numpy lexsort, same ordering as the sorted() tuple key in l5a_critic_indep.py).
(1) Rebuild the 20 subset draws (RandomState(100 + i).choice(3000, 200, replace=False) over the p0 feats row order), fit each 200-label table,
    replay splits 0 to 7 x 8 tie seeds at 10% on L2b and L4b, compare every per-split gap, fixed and frozen hit rate with runs/L6b/result.json.
(2) Distinct tables and distinct selected sets behind the 20 subsets (non-independence).
(3) Fresh label draws: 15 disjoint 200-prompt subsets of p0 (one permutation) and 100 new RandomState(1000 + i) subsets. Same replay.
(4) Eval-pool noise: bucket bootstrap (75 cell100 buckets with replacement, 3 tie seeds) of the mean-over-20-subsets gap.
(5) Top strata of the 3,000-label table (for reference).
(6) Composition stand-in: L4b test rows (pid < 1319) and remaining-train rows, 200-label tables, stand-in buckets from the joint cell100 clustering.
(7) Budget accounting: calls per split for fixed and frozen, and the refit label counts from the reference files.
Usage: /homes/rdubash/active-audit/.venv/bin/python -I l6b_critic_indep.py
"""
import json
import numpy as np, pandas as pd

RUN = "/homes/rdubash/avrl_live/ROUTED/avrl/runs"
S = ["agreement", "n_distinct", "any_truncated"]
K0, FRAC = 50.0, 0.10


def fit_table(df):
    g = df.assign(_y=df["proxy_error"].astype(int)).groupby(S)["_y"].agg(["sum", "size"]).reset_index()
    g["prior"] = 1 - g["agreement"]; g["post"] = (g["sum"] + K0 * g["prior"]) / (g["size"] + K0)
    return {(float(r.agreement), int(r.n_distinct), int(r.any_truncated)): float(r.post) for r in g.itertuples()}


def score_of(d, tab):
    return np.array([tab.get((float(a), int(n), int(t)), 1 - a) for a, n, t in zip(d.agreement, d.n_distinct, d.any_truncated)])


def build_idx(df, split, G="cell100", MIN=3):
    """Row positions (into df) of the sim for one split: drop buckets with fewer than MIN docs, then drop a quarter of the buckets."""
    cnt = df[G].value_counts(); keep = df[G].isin(cnt[cnt >= MIN].index).to_numpy()
    d = df[keep]; pos = np.flatnonzero(keep)
    g = d[G].unique().copy(); np.random.default_rng(split).shuffle(g)
    drop = set(g[: len(g) // 4].tolist())
    return pos[~d[G].isin(drop).to_numpy()]


class Pool:
    def __init__(self, name, df, splits=range(8)):
        self.name, self.df = name, df.reset_index(drop=True)
        self.y = self.df["proxy_error"].to_numpy(int); self.ag = self.df.agreement.to_numpy(float); self.nd = self.df.n_distinct.to_numpy(int)
        self.idx = {s: build_idx(self.df, s) for s in splits}
        self.rnd = {}  # (split, seed) -> random tie-break per sim doc, same draw order as the reference: default_rng(seed).random(n_sim)
        for s in splits:
            for sd in range(8): self.rnd[(s, sd)] = np.random.default_rng(sd).random(len(self.idx[s]))

    def fixed_sel(self, s, sd):
        i = self.idx[s]; return np.lexsort((self.rnd[(s, sd)], -self.nd[i], self.ag[i]))[: int(FRAC * len(i))]

    def frozen_sel(self, s, sd, sc):
        i = self.idx[s]; return np.lexsort((self.rnd[(s, sd)], -self.nd[i], self.ag[i], -sc[i]))[: int(FRAC * len(i))]

    def replay(self, sc, splits=range(8), seeds=range(8)):
        out = []; bad = 0
        for s in splits:
            i = self.idx[s]; B = int(FRAC * len(i)); fh, zh = [], []
            for sd in seeds:
                f, z = self.fixed_sel(s, sd), self.frozen_sel(s, sd, sc)
                if not (len(f) == B == len(np.unique(f)) and len(z) == B == len(np.unique(z))): bad += 1
                fh.append(100.0 * self.y[i][f].sum() / B); zh.append(100.0 * self.y[i][z].sum() / B)
            out.append((float(np.mean(zh) - np.mean(fh)), float(np.mean(fh)), float(np.mean(zh)), B))
        return out, bad


def stats(g):
    g = np.asarray(g, float); return f"mean {g.mean():+.2f} sd {g.std(ddof=1):.2f} above2 {int((g > 2).sum())} of {len(g)} min {g.min():+.2f} max {g.max():+.2f}"


def run_tables(pools, tables, label):
    means = {n: [] for n in pools}; bad = 0
    for t in tables:
        for n, P in pools.items():
            r, b = P.replay(P_score(P, t)); bad += b; means[n].append(float(np.mean([x[0] for x in r])))
    print(f"{label}: spend failures {bad}")
    for n in pools:
        m = np.array(means[n]); print(f"  {n}: tables {len(m)} | subset means {stats(m)} | grand mean {m.mean():+.3f} se {m.std(ddof=1) / np.sqrt(len(m)):.2f}")
    return means


_cache = {}
def P_score(P, tab):
    k = (P.name, id(tab))
    if k not in _cache: _cache[k] = score_of(P.df, tab)
    return _cache[k]


def bootstrap(P, tabs, nb=200, seeds=3):
    """Resample 75 of the cell100 buckets with replacement; gap = mean over tables and tie seeds (fixed hits computed once)."""
    df = P.df; cnt = df.cell100.value_counts(); df = df[df.cell100.isin(cnt[cnt >= 3].index)]
    scs = [score_of(df, t) for t in tabs]; ag = df.agreement.to_numpy(float); nd = df.n_distinct.to_numpy(int); y = df.proxy_error.to_numpy(int)
    buck = {b: np.flatnonzero((df.cell100 == b).to_numpy()) for b in df.cell100.unique()}; keys = list(buck); out = []
    for i in range(nb):
        rng = np.random.default_rng(9000 + i); pick = rng.choice(len(keys), size=75, replace=True)
        ii = np.concatenate([buck[keys[j]] for j in pick]); B = int(FRAC * len(ii)); g = []
        for sd in range(seeds):
            r = np.random.default_rng(sd).random(len(ii)); fh = y[ii][np.lexsort((r, -nd[ii], ag[ii]))[:B]].sum()
            for sc in scs: g.append(100.0 * (y[ii][np.lexsort((r, -nd[ii], ag[ii], -sc[ii]))[:B]].sum() - fh) / B)
        out.append(float(np.mean(g)))
    return np.array(out)


if __name__ == "__main__":
    res = json.load(open(f"{RUN}/L6b/result.json"))
    p0 = pd.read_parquet(f"{RUN}/p0/feats.parquet")
    pools = {n: Pool(n, pd.read_parquet(f"{RUN}/{n}/feats.parquet")) for n in ("L2b", "L4b")}
    print("== (1) independent replay of the 20 registered subsets vs runs/L6b/result.json")
    tabs, idxs = [], []
    for i in range(20):
        idx = np.random.RandomState(100 + i).choice(len(p0), size=200, replace=False); idxs.append(idx)
        sub = p0.iloc[idx]; assert sub.pid.nunique() == 200
        tabs.append(fit_table(sub))
        assert list(idx) == res["subset_detail"][i]["row_positions"], f"row positions differ for subset {i}"
    print("row positions equal result.json for all 20 subsets; errors per 200-label subset:",
          [int(p0.iloc[ix].proxy_error.sum()) for ix in idxs])
    dg = dfx = dzx = 0.0; means = {n: [] for n in pools}; bad = 0
    for n, P in pools.items():
        rj = res["pools"][n]
        for i, t in enumerate(tabs):
            r, b = P.replay(P_score(P, t)); bad += b
            dg = max(dg, max(abs(a[0] - c) for a, c in zip(r, rj["gap_pp_per_subset"][i])))
            means[n].append(float(np.mean([a[0] for a in r])))
        print(n, "max abs diff of the 160 per-split gaps vs result.json:", dg, "| my subset means", stats(means[n]), "| result.json mean", round(rj["mean_pp"], 4))
        dg = 0.0
    print("spend failures (any policy not taking exactly int(0.10 x n_sim) distinct docs):", bad)
    print("calls per split L2b", [x[3] for x in pools["L2b"].replay(np.zeros(len(pools['L2b'].df)))[0]],
          "L4b", [x[3] for x in pools["L4b"].replay(np.zeros(len(pools['L4b'].df)))[0]])
    print("== (2) non-independence: distinct 200-label tables and distinct selected sets")
    for n, P in pools.items():
        sig = []
        for t in tabs:
            sc = P_score(P, t); sig.append(tuple(tuple(sorted(P.frozen_sel(s, 0, sc).tolist())) for s in range(8)))
        print(n, "distinct selected-set signatures (splits 0 to 7, tie seed 0) among 20 subsets:", len(set(sig)),
              "| distinct subset means (2 d.p.):", len(set(np.round(means[n], 2))))
    rank_sig = []
    for t in tabs:
        ks = sorted(t, key=lambda k: -t[k]); rank_sig.append(tuple(ks[:6]))
    print("distinct top-6 stratum orderings among 20 tables:", len(set(rank_sig)))
    ov = [len(set(idxs[a]) & set(idxs[b])) for a in range(20) for b in range(a + 1, 20)]
    print("pairwise overlap of the 20 label subsets (prompts): mean", round(float(np.mean(ov)), 1), "max", max(ov), "(random expectation 13.3)")
    print("== (3) fresh label draws")
    perm = np.random.RandomState(7).permutation(len(p0)); disj = [fit_table(p0.iloc[perm[k * 200:(k + 1) * 200]]) for k in range(15)]
    run_tables(pools, disj, "15 disjoint 200-prompt subsets (a partition of p0)")
    fresh = [fit_table(p0.iloc[np.random.RandomState(1000 + i).choice(len(p0), size=200, replace=False)]) for i in range(100)]
    fm = run_tables(pools, fresh, "100 fresh RandomState(1000 + i) 200-prompt subsets")
    for n in pools: print(f"  {n}: fraction of 100 fresh tables with subset mean above +2.0 = {np.mean(np.array(fm[n]) > 2):.2f}")
    print("== (4) eval-pool noise: bucket bootstrap of the mean-over-20-subsets gap (200 resamples, 3 tie seeds)")
    for n, P in pools.items():
        b = bootstrap(P, tabs)
        print(n, f"mean {b.mean():+.2f} sd {b.std(ddof=1):.2f} 95% [{np.percentile(b, 2.5):+.2f}, {np.percentile(b, 97.5):+.2f}] above +2.0: {(b > 2).sum()} of {len(b)}")
    print("== (5) reference")
    t3000 = fit_table(p0)
    top = sorted(t3000.items(), key=lambda kv: -kv[1])[:6]
    print("top 6 strata of the 3,000-label table (agreement, n_distinct, any_truncated): posterior", [(k, round(v, 3)) for k, v in top])
    print("== (6) composition stand-in (L4b test rows pid < 1319, remaining train rows pid >= 1319), 20 registered 200-label tables")
    L4 = pools["L4b"].df
    for name, sub in (("L4b test rows", L4[L4.pid < 1319]), ("L4b remaining train", L4[L4.pid >= 1319])):
        P = Pool(name, sub); m = []
        for t in tabs:
            r, _ = P.replay(score_of(P.df, t)); m.append(float(np.mean([x[0] for x in r])))
        r3, _ = P.replay(score_of(P.df, t3000))
        print(name, "n", len(sub), "err", round(float(sub.proxy_error.mean()), 3), "| 200-label tables:", stats(m), "| 3,000-label table:", f"{np.mean([x[0] for x in r3]):+.2f}")
    print("== (7) refit label counts (calls per split) in the reference files")
    a = json.load(open(f"{RUN}/L3a/fresh/result.json"))["configs"]["B"]; b = json.load(open(f"{RUN}/L4b/result.json"))["configs"]["B40"]
    print("L2b refit B calls", a["calls_per_split"], "mean", np.mean(a["calls_per_split"]), "| L4b refit B40 calls", b["calls_per_split"], "mean", np.mean(b["calls_per_split"]))
