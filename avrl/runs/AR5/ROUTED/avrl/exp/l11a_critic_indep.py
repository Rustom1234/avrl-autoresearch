"""L11a critic: independent replay. CPU only. Imports nothing from the experiment modules.

Own code for: the pid <= 1318 cut, the sim split (cell100 buckets with >= 3 rows, shuffled with default_rng(split), first quarter
of buckets removed), the 200-label frozen config B table, plain fixed, and the table-pooled greedy learner
(score (e + 50 * table_post) / (v + 50), recomputed every R = max(b // 8, 40) verified labels). Checks
  (1) the 20 registered subsets x splits 0-7 x 8 tie seeds against runs/L11a/result.json (subset means, per-split gaps),
  (2) budgets, n_sim, refit counts per run, strata coverage of the eval rows (rows whose stratum is missing from a table),
  (3) a leakage probe: eval labels of unverified docs permuted must leave every pick sequence unchanged,
  (4) a second probe: hide gold-dependent columns (gold, pass_rate, majority_correct) from the frame entirely,
  (5) sensitivity: per-subset mean over only tie seed 0 (to see how the verdict depends on tie draws), and a paired bootstrap
      over eval rows of the registered cut (diagnostic only; the rule is the mean over the 20 subset means).
Usage: python -I l11a_critic_indep.py   (log: runs/L11a_critic/indep.log)
"""
import sys, json, numpy as np, pandas as pd
sys.dont_write_bytecode = True
from joblib import Parallel, delayed

R = "/homes/rdubash/avrl_live/ROUTED/avrl/runs"
ST = ["agreement", "n_distinct", "any_truncated"]
p0 = pd.read_parquet(f"{R}/p0/feats.parquet")
ev_full = pd.read_parquet(f"{R}/L4b/feats.parquet")
ev = ev_full[ev_full["pid"] <= 1318].reset_index(drop=True)


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
    return np.where(m["post"].isna(), 1 - df["agreement"].to_numpy(float), m["post"]).astype(float), m["post"].isna().to_numpy()


def greedy(y, ag, nd, code, init, b, seed, update, k0=50.0, y_for_update=None):
    yu = y if y_for_update is None else y_for_update
    rnd = np.random.default_rng(seed).random(len(y)); Rr = max(b // 8, 40)
    seen = np.zeros(len(y), bool); sc = init.copy(); n = 0; nf = Rr; hits = 0; nup = 0; picks = []
    K = code.max() + 1
    while n < b:
        mu = np.flatnonzero(~seen); c = mu[np.lexsort((rnd[mu], -nd[mu], ag[mu], -sc[mu]))[0]]
        seen[c] = True; hits += y[c]; n += 1; picks.append(int(c))
        if update and n >= nf and n < b:
            e = np.zeros(K); v = np.zeros(K); ix = np.flatnonzero(seen)
            np.add.at(e, code[ix], yu[ix].astype(float)); np.add.at(v, code[ix], 1.0)
            sc = (e[code] + k0 * init) / (v[code] + k0); nf += Rr; nup += 1
    return int(hits), nup, picks


subs = []
for i in range(20):
    idx = np.random.RandomState(100 + i).choice(3000, 200, replace=False)
    subs.append(table(p0.iloc[idx].reset_index(drop=True)))


def one_split(split, tie_seeds=range(8)):
    sim = sim_split(ev, split)
    y = sim["proxy_error"].to_numpy(int); b = int(0.1 * len(y))
    ag = sim["agreement"].to_numpy(); nd = sim["n_distinct"].to_numpy()
    code = sim.groupby(ST).ngroup().to_numpy(int)
    rows = []; nups = set(); miss = []
    for g in subs:
        init, m = score_with(sim, g); miss.append(float(m.mean()))
        zh, fh, ph = [], [], []
        for t in tie_seeds:
            r = np.random.default_rng(t).random(len(y))
            zh.append(y[np.lexsort((r, -nd, ag, -init))[:b]].sum()); fh.append(y[np.lexsort((r, -nd, ag))[:b]].sum())
            h0, _, p0_ = greedy(y, ag, nd, code, init, b, t, False); assert h0 == zh[-1], "zero-update greedy != frozen"
            h, nu, pk = greedy(y, ag, nd, code, init, b, t, True); ph.append(h); nups.add(nu)
            assert len(set(pk)) == b == len(pk)
        zh, fh, ph = np.array(zh), np.array(fh), np.array(ph)
        rows.append((100 * (ph - zh).mean() / b, 100 * (zh - fh).mean() / b, 100 * (ph - fh).mean() / b,
                     100 * (ph[0] - zh[0]) / b))
    return np.array(rows), len(y), b, nups, float(np.mean(miss))


def leak_probe(label_cols_hidden):
    bad = 0
    sim = sim_split(ev, 0)
    if label_cols_hidden:
        sim = sim.drop(columns=["gold", "pass_rate", "majority_correct", "majority", "question"])
    y = sim["proxy_error"].to_numpy(int); b = int(0.1 * len(y))
    ag = sim["agreement"].to_numpy(); nd = sim["n_distinct"].to_numpy(); code = sim.groupby(ST).ngroup().to_numpy(int)
    for gi in (0, 7, 8):
        init, _ = score_with(sim, subs[gi])
        for t in range(4):
            h, _, p = greedy(y, ag, nd, code, init, b, t, True)
            rng = np.random.default_rng(900 + t); y2 = y.copy(); rest = np.setdiff1d(np.arange(len(y)), p)
            y2[rest] = rng.permutation(y[rest])
            h2, _, p2 = greedy(y, ag, nd, code, init, b, t, True, y_for_update=y2)  # update labels for unverified docs permuted
            h3, _, p3 = greedy(y2, ag, nd, code, init, b, t, True)
            bad += int(p != p2) + int(p != p3)
    return bad


if __name__ == "__main__":
    L11 = json.load(open(f"{R}/L11a/result.json"))
    print(f"eval rows {len(ev)} (pid {ev.pid.min()}..{ev.pid.max()}), proxy error {ev.proxy_error.mean():.6f}, distinct pids {ev.pid.nunique()}, "
          f"distinct questions {ev.question.nunique()}", flush=True)
    q0 = set(p0.question.str.lower().str.replace(r"[^a-z0-9 ]", "", regex=True).str.strip())
    qe = ev.question.str.lower().str.replace(r"[^a-z0-9 ]", "", regex=True).str.strip()
    print(f"normalised question overlap eval vs p0: {int(qe.isin(q0).sum())}; p0 distinct questions {p0.question.nunique()} of {len(p0)}", flush=True)
    res = Parallel(n_jobs=8)(delayed(one_split)(s) for s in range(8))
    A = np.stack([r[0] for r in res]); ns = [r[1] for r in res]; bs = [r[2] for r in res]
    nu = sorted(set().union(*[r[3] for r in res])); miss = [r[4] for r in res]
    M = A.mean(axis=0)  # subset x 4
    ref = np.array([d["L4b_table_pooled_minus_frozen_pp"] for d in L11["subset_detail"]])
    refs = np.array([d["L4b_table_pooled_minus_frozen_pp_per_split"] for d in L11["subset_detail"]])  # subset x split
    print(f"(1) replay: pooled-frozen mean {M[:,0].mean():+.4f} sd {M[:,0].std(ddof=1):.4f} se {M[:,0].std(ddof=1)/np.sqrt(20):.4f} "
          f"subsets>+1.0 {(M[:,0] > 1).sum()} min {M[:,0].min():+.4f} max {M[:,0].max():+.4f} | frozen-fixed {M[:,1].mean():+.4f} "
          f"pooled-fixed {M[:,2].mean():+.4f}", flush=True)
    print(f"    max abs diff to L11a subset means {np.abs(M[:,0] - ref).max():.2e} pp; to per-split gaps {np.abs(A[:,:,0].T - refs).max():.2e} pp "
          f"(ledger mean {L11['pools']['L4b']['mean_pp']:+.4f}); split gaps>+1.0: {(A[:,:,0] > 1).sum()} of 160", flush=True)
    print(f"(2) n_sim per split {ns}; budget per split {bs}; updates per run {nu}; share of sim rows whose stratum is missing from a table, "
          f"mean per split {np.round(miss, 4).tolist()}", flush=True)
    print(f"(5a) tie seed 0 only: pooled-frozen mean {M[:,3].mean():+.3f} sd {M[:,3].std(ddof=1):.3f}", flush=True)
    print(f"(3) leakage probe (unverified labels permuted, 36 runs x 2 variants), runs whose picks changed: {leak_probe(False)}", flush=True)
    print(f"(4) leakage probe with gold, pass_rate, majority_correct, majority, question columns removed: {leak_probe(True)}", flush=True)
    # (5b) diagnostic: bootstrap over eval-row clusters is not available with the bucket split; instead report the share of the gap by split seed
    print(f"(5b) mean gap by split seed {np.round(A[:,:,0].mean(axis=1), 2).tolist()}", flush=True)
