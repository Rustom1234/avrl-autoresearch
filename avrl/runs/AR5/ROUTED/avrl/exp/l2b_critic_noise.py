"""L2b critic: pool-level noise. CPU only. Uses the L1b Sim class for speed (its output on this pool was matched exactly
by the independent reimplementation in l2b_critic_indep.py).
(a) 12 random partitions of the 100 cell100 buckets into two disjoint halves (no fit cut): gap of config A at 5% and B at 10%
    over plain fixed on each half, 8 tie-break seeds. Halves have about 1,500 docs (75 calls at 5%, 150 at 10%).
(b) bucket bootstrap: 200 resamples of 75 buckets drawn with replacement from the 100 buckets, 4 tie-break seeds.
"""
import sys, json
from multiprocessing import Pool
import numpy as np, pandas as pd
sys.path.insert(0, "/homes/rdubash/avrl_live/ROUTED/avrl/exp")
import l1b_tuned_threshold as L
F = pd.read_parquet("/homes/rdubash/avrl_live/ROUTED/avrl/runs/L2b/feats.parquet")
cnt = F.cell100.value_counts(); F = F[F.cell100.isin(cnt[cnt >= 3].index)].reset_index(drop=True)
BUCKETS = np.array(sorted(F.cell100.unique())); SPEC = dict(L.SPECS)
CFG = {"A": ("k02_Rdiv8_agreement_x_nd", 0.05), "B": ("k050_Rdiv8_agreement_x_nd_x_trunc", 0.10)}
def gaps(df, seeds):
    S = L.Sim(df.reset_index(drop=True), "proxy_error"); out = {}
    for t, (name, f) in CFG.items():
        b = int(f * S.n); out[t] = float(np.mean([100 * (L.hits_for(S, SPEC[name], b, s) - L.hits_for(S, SPEC[L.FIXED], b, s)) / b for s in seeds]))
    return out, S.n
def half(i):
    rng = np.random.default_rng(1000 + i); p = rng.permutation(BUCKETS); h = len(p) // 2; r = []
    for part in (p[:h], p[h:]):
        g, n = gaps(F[F.cell100.isin(part)], range(8)); r.append((g, n))
    return r
def boot(i):
    rng = np.random.default_rng(5000 + i); pick = rng.choice(BUCKETS, size=75, replace=True)
    df = pd.concat([F[F.cell100 == b] for b in pick]); g, n = gaps(df, range(4)); return g, n
if __name__ == "__main__":
    with Pool(8) as pool:
        H = pool.map(half, range(12)); Bt = pool.map(boot, range(200))
    flat = [g for r in H for g, n in r]
    for t in "AB":
        v = np.array([g[t] for g in flat])
        print(f"(a) halves {t}: n={len(v)} mean {v.mean():+.2f} sd {v.std(ddof=1):.2f} min {v.min():+.2f} max {v.max():+.2f} above 2.0: {(v>2).sum()} of {len(v)} above 0: {(v>0).sum()}; docs per half {sorted(set(n for r in H for g,n in r))[:1]}..{sorted(set(n for r in H for g,n in r))[-1:]}")
    for t in "AB":
        v = np.array([g[t] for g, n in Bt])
        print(f"(b) bootstrap {t}: n={len(v)} mean {v.mean():+.2f} sd {v.std(ddof=1):.2f} 2.5% {np.percentile(v,2.5):+.2f} 97.5% {np.percentile(v,97.5):+.2f} above 2.0: {(v>2).sum()} of {len(v)} above 0: {(v>0).sum()}")
    json.dump({"halves": flat, "boot": [g for g, n in Bt]}, open("/homes/rdubash/avrl_live/ROUTED/avrl/runs/L2b_critic/noise.json", "w"))
