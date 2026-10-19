"""L4b critic: sensitivity of the verdict to the pool deviation. CPU only. Replays configs A (5%) and B (10%) at floors 40 and 10 on
(a) the test-only rows of the L4b pool (pid < 1319, the pool the backlog entry registered) and (b) the remaining-train rows (pid >= 1319),
using the same independent Pool code as l4b_critic_indep.py, same cell100 buckets (min 3 docs), 75% kept, splits 0 to 7, 8 seeds.
Usage: python -I l4b_critic_subpools.py
"""
import sys
sys.path.insert(0, "/homes/rdubash/avrl_live/ROUTED/avrl/exp")
import numpy as np, pandas as pd
import l4b_critic_indep as C

F = pd.read_parquet(f"{C.RUN}/L4b/feats.parquet")
for name, sub in (("test only (pid < 1319)", F[F.pid < 1319]), ("remaining train (pid >= 1319)", F[F.pid >= 1319])):
    sub = sub.reset_index(drop=True)
    for floor in (40, 10):
        o = C.replay(sub, floor)
        print(f"[{name}] floor {floor}: n_sim {o['n']} calls(5%,10%) {o['calls']}")
        print(f"   A 5% gaps {[round(x, 2) for x in o['A']]} mean {np.mean(o['A']):+.2f} sd {np.std(o['A'], ddof=1):.2f} | fixed5 {np.mean(o['A_fix']):.2f} A {np.mean(o['A_cfg']):.2f} | refits {sorted({tuple(v) for v in o['refits'].values()})} | spend/unique failures {o['bad']}")
        print(f"   B 10% gaps {[round(x, 2) for x in o['B']]} mean {np.mean(o['B']):+.2f} sd {np.std(o['B'], ddof=1):.2f} | fixed10 {np.mean(o['B_fix']):.2f} B {np.mean(o['B_cfg']):.2f}")
