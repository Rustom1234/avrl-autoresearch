"""L3a critic: pool-level noise of config A at 5% under refit floor 40 vs floor 10, and the paired floor effect. CPU only.
Uses the independent Pool class from l3a_critic_indep.py (it reproduced result.json to 4e-15 pp on both pools).
(a) 64 random cuts that drop a random quarter of the 100 cell100 buckets (same cut shape as the L3a splits), 8 tie-break seeds:
    gap of A at 5% at floor 40 and floor 10, and B at 10% at floor 40; paired floor effect (floor 10 minus floor 40).
(b) 12 random partitions of the buckets into two disjoint halves (24 half-pool runs, about 1,500 docs, 75 calls at 5%).
(c) bucket bootstrap, 200 resamples of 75 buckets with replacement, 4 tie-break seeds (duplicated buckets inflate the spread).
Both pools. Writes nothing but stdout.
"""
import sys
from multiprocessing import Pool as MP
import numpy as np, pandas as pd
sys.path.insert(0, "/homes/rdubash/avrl_live/ROUTED/avrl/exp")
from l3a_critic_indep import Pool, ROOT

POOLS = {"fresh": f"{ROOT}/L2b/feats.parquet", "l1b_pool": f"{ROOT}/p0/feats.parquet"}


def load(name):
    F = pd.read_parquet(POOLS[name]); cnt = F.cell100.value_counts()
    return F[F.cell100.isin(cnt[cnt >= 3].index)].reset_index(drop=True)


def gaps(d, seeds):
    P = Pool(d.reset_index(drop=True)); b5, b10 = int(0.05 * P.n), int(0.10 * P.n); a40, a10, b40 = [], [], []
    for s in seeds:
        f5 = P.y[P.fixed_set(b5, s)].sum(); f10 = P.y[P.fixed_set(b10, s)].sum()
        a40.append(100 * (P.y[P.recal_set(b5, s, 2, max(b5 // 8, 40), "x")[0]].sum() - f5) / b5)
        a10.append(100 * (P.y[P.recal_set(b5, s, 2, max(b5 // 8, 10), "x")[0]].sum() - f5) / b5)
        b40.append(100 * (P.y[P.recal_set(b10, s, 50, max(b10 // 8, 40), "xt")[0]].sum() - f10) / b10)
    return dict(A40=float(np.mean(a40)), A10=float(np.mean(a10)), B40=float(np.mean(b40)), n=P.n)


def cut(args):
    name, i = args; F = load(name); B = np.array(sorted(F.cell100.unique()))
    rng = np.random.default_rng(7000 + i); drop = set(rng.permutation(B)[: len(B) // 4].tolist())
    return gaps(F[~F.cell100.isin(drop)], range(8))


def half(args):
    name, i = args; F = load(name); B = np.array(sorted(F.cell100.unique()))
    p = np.random.default_rng(1000 + i).permutation(B); h = len(p) // 2
    return [gaps(F[F.cell100.isin(part)], range(8)) for part in (p[:h], p[h:])]


def boot(args):
    name, i = args; F = load(name); B = np.array(sorted(F.cell100.unique()))
    pick = np.random.default_rng(5000 + i).choice(B, size=75, replace=True)
    return gaps(pd.concat([F[F.cell100 == b] for b in pick]), range(4))


def summ(label, v, bar=2.0):
    v = np.asarray(v); print(f"  {label}: n={len(v)} mean {v.mean():+.2f} sd {v.std(ddof=1):.2f} min {v.min():+.2f} max {v.max():+.2f} "
                             f"exactly 0: {(np.abs(v) < 1e-9).sum()} above 0: {(v > 0).sum()} above +2.0: {(v > bar).sum()}")


if __name__ == "__main__":
    with MP(16) as mp:
        for name in POOLS:
            print(f"=== pool {name}")
            C = mp.map(cut, [(name, i) for i in range(64)])
            print("(a) 64 random 75-of-100 bucket cuts")
            for k in ("A40", "A10", "B40"): summ(k, [c[k] for c in C])
            d = np.array([c["A10"] - c["A40"] for c in C]); print(f"  paired floor effect A (floor10 - floor40): mean {d.mean():+.2f} sd {d.std(ddof=1):.2f} "
                  f"positive in {(d > 0).sum()} of 64; sd of A40 gives jackknife-scale pool sd ~ sqrt(3) x {np.std([c['A40'] for c in C], ddof=1):.2f} = "
                  f"{np.sqrt(3) * np.std([c['A40'] for c in C], ddof=1):.2f}")
            m = np.mean([c["A40"] for c in C]); sdp = np.sqrt(3) * np.std([c["A40"] for c in C], ddof=1)
            print(f"  A40 mean {m:+.2f}, bar +2.0 is {(m - 2.0) / sdp:+.2f} jackknife-scale sd away")
            H = [g for r in mp.map(half, [(name, i) for i in range(12)]) for g in r]
            print("(b) 12 disjoint-half partitions, 24 half-pool runs")
            for k in ("A40", "A10", "B40"): summ(k, [c[k] for c in H])
            Bt = mp.map(boot, [(name, i) for i in range(200)])
            print("(c) bucket bootstrap, 200 resamples")
            for k in ("A40", "A10", "B40"):
                v = np.array([c[k] for c in Bt]); print(f"  {k}: mean {v.mean():+.2f} sd {v.std(ddof=1):.2f} 2.5% {np.percentile(v, 2.5):+.2f} 97.5% {np.percentile(v, 97.5):+.2f} "
                                                        f"above +2.0: {(v > 2).sum()} of 200 above 0: {(v > 0).sum()}")
