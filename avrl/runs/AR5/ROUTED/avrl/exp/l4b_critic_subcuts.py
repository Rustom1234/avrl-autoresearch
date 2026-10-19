"""L4b critic: 64 random 75-of-100 bucket cuts (default_rng(7000 + i), same as the author's secondary) on the test-only rows and on the
remaining-train rows of the L4b pool, plus the same three group columns (cell30, cell100, cell300) on the 8 canonical splits.
Independent Pool code from l4b_critic_indep.py. CPU only.  Usage: python -I l4b_critic_subcuts.py
"""
import sys
sys.path.insert(0, "/homes/rdubash/avrl_live/ROUTED/avrl/exp")
from multiprocessing import Pool as MP
import numpy as np, pandas as pd
import l4b_critic_indep as C


def gaps(d, floor, seeds=range(8)):
    P = C.Pool(d.reset_index(drop=True)); b5, b10 = int(0.05 * P.n), int(0.10 * P.n); a, b = [], []
    for s in seeds:
        a.append(100 * (P.y[P.recal_set(b5, s, 2, max(b5 // 8, floor), "x")[0]].sum() - P.y[P.fixed_set(b5, s)].sum()) / b5)
        b.append(100 * (P.y[P.recal_set(b10, s, 50, max(b10 // 8, floor), "xt")[0]].sum() - P.y[P.fixed_set(b10, s)].sum()) / b10)
    return float(np.mean(a)), float(np.mean(b))


def cut(args):
    F, i, floor = args
    cnt = F.cell100.value_counts(); F = F[F.cell100.isin(cnt[cnt >= 3].index)]
    B = np.array(sorted(F.cell100.unique())); rng = np.random.default_rng(7000 + i)
    drop = set(rng.permutation(B)[: len(B) // 4].tolist())
    return gaps(F[~F.cell100.isin(drop)], floor)


def main():
    F = pd.read_parquet(f"{C.RUN}/L4b/feats.parquet")
    for name, sub in (("full pool", F), ("test only", F[F.pid < 1319]), ("remaining train", F[F.pid >= 1319])):
        sub = sub.reset_index(drop=True)
        for floor in (40, 10):
            with MP(16) as mp: r = np.array(mp.map(cut, [(sub, i, floor) for i in range(64)]))
            for j, nm in ((0, "A 5%"), (1, "B 10%")):
                v = r[:, j]
                print(f"[{name}] floor {floor} {nm}: 64 cuts mean {v.mean():+.2f} sd {v.std(ddof=1):.2f} min {v.min():+.2f} max {v.max():+.2f} above +2.0: {(v > 2).sum()} of 64", flush=True)
    # alternative bucketisations on the canonical split procedure (seed = split), test-only rows
    for col in ("cell30", "cell100", "cell300"):
        for name, sub in (("full pool", F), ("test only", F[F.pid < 1319])):
            sub = sub.reset_index(drop=True); ga, gb = [], []
            for s in range(8):
                d = C.build(sub, s, G=col); a, b = gaps(d, 40); ga.append(a); gb.append(b)
            print(f"[{name}] group column {col}, floor 40, splits 0 to 7: A 5% mean {np.mean(ga):+.2f}, B 10% mean {np.mean(gb):+.2f} (B per split {[round(x, 2) for x in gb]})", flush=True)


if __name__ == "__main__":
    main()
