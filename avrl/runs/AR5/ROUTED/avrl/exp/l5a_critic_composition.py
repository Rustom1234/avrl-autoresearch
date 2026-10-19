"""L5a critic: (a) 64 random 75-of-100 bucket cuts (default_rng(7000 + i), as in the L4b critic) for the frozen p0 table vs plain fixed vs refit B
(floor 40) on L4b full, L4b test rows (pid < 1319), L4b remaining-train rows, and L2b full; (b) paired bucket bootstrap of frozen minus refit B on
L2b and L4b. Refit code is the L4b critic's independent Pool (not the L1B module). CPU only.
Usage: /homes/rdubash/active-audit/.venv/bin/python -I l5a_critic_composition.py
"""
import sys
sys.path.insert(0, "/homes/rdubash/avrl_live/ROUTED/avrl/exp")
from multiprocessing import Pool as MP
import numpy as np, pandas as pd
import l4b_critic_indep as C
import l5a_critic_indep as I

RUN = C.RUN
SEEDS = range(8)


def gaps(d, tab, floor=40, seeds=SEEDS):
    d = d.reset_index(drop=True); P = C.Pool(d); sc = I.score_of(d, tab); b10 = int(0.10 * P.n); fz, rf = [], []
    for s in seeds:
        f = P.fixed_set(b10, s); _, _, z = I.picks(d, sc, b10, s); b, _ = P.recal_set(b10, s, 50, max(b10 // 8, floor), "xt")
        fz.append(100 * (P.y[z].sum() - P.y[f].sum()) / b10); rf.append(100 * (P.y[b].sum() - P.y[f].sum()) / b10)
    return float(np.mean(fz)), float(np.mean(rf))


def cut(args):
    F, tab, i = args
    cnt = F.cell100.value_counts(); F = F[F.cell100.isin(cnt[cnt >= 3].index)]
    B = np.array(sorted(F.cell100.unique())); rng = np.random.default_rng(7000 + i)
    drop = set(rng.permutation(B)[: len(B) // 4].tolist())
    return gaps(F[~F.cell100.isin(drop)], tab)


def boot(args):
    F, tab, i = args
    cnt = F.cell100.value_counts(); F = F[F.cell100.isin(cnt[cnt >= 3].index)]
    by = {b: g for b, g in F.groupby("cell100")}; keys = list(by); rng = np.random.default_rng(9000 + i)
    d = pd.concat([by[keys[j]] for j in rng.choice(len(keys), size=75, replace=True)], ignore_index=True)
    return gaps(d, tab, seeds=range(2))


def report(name, r):
    r = np.array(r); fz, rf = r[:, 0], r[:, 1]; df = fz - rf
    print(f"[{name}] frozen {fz.mean():+.2f} (sd {fz.std(ddof=1):.2f}, above +2.0 {int((fz > 2).sum())} of {len(fz)}, min {fz.min():+.2f}) | "
          f"refit B40 {rf.mean():+.2f} (sd {rf.std(ddof=1):.2f}, above +2.0 {int((rf > 2).sum())}) | frozen minus refit {df.mean():+.2f} "
          f"(sd {df.std(ddof=1):.2f}, frozen > refit in {int((df > 0).sum())} of {len(df)}, 95% [{np.percentile(df, 2.5):+.2f}, {np.percentile(df, 97.5):+.2f}])", flush=True)


if __name__ == "__main__":
    p0 = pd.read_parquet(f"{RUN}/p0/feats.parquet"); tab, _ = I.fit_table(p0)
    L4 = pd.read_parquet(f"{RUN}/L4b/feats.parquet"); L2 = pd.read_parquet(f"{RUN}/L2b/feats.parquet")
    n = 8
    with MP(n) as mp:
        print("== 64 random 75-of-100 bucket cuts")
        for name, sub in (("L4b full", L4), ("L4b test only", L4[L4.pid < 1319]), ("L4b remaining train", L4[L4.pid >= 1319]), ("L2b full", L2)):
            sub = sub.reset_index(drop=True); report(name, mp.map(cut, [(sub, tab, i) for i in range(64)]))
        print("== bucket bootstrap, 100 resamples x 2 tie seeds, paired frozen vs refit B40")
        for name, sub in (("L2b", L2), ("L4b", L4)):
            report(name, mp.map(boot, [(sub, tab, i) for i in range(100)]))
