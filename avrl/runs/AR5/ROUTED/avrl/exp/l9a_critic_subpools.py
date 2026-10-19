"""L9a critic: split the L4b full pool into the registered test rows (pid <= 1318) and the remaining train rows (pid > 1318).
Reuses the independent replay in l9a_critic_indep.py (imports nothing from the experiment modules). CPU only."""
import sys, os
sys.dont_write_bytecode = True
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np, pandas as pd
import l9a_critic_indep as I

ev = pd.read_parquet(f"{I.R}/L4b/feats.parquet")
for name, d in (("remaining-train rows (pid > 1318)", ev[ev["pid"] > 1318]), ("test rows (pid <= 1318)", ev[ev["pid"] <= 1318])):
    A, ns, bs, nr = I.registered(d); M = A.mean(axis=0)
    print(f"L4b {name}, {len(d)} rows, budgets {min(bs)}..{max(bs)}, refits {nr}: refit-frozen mean {M[:,0].mean():+.2f} sd {M[:,0].std(ddof=1):.2f} "
          f"subsets>+1.0 {(M[:,0] > 1).sum()} of 20 min {M[:,0].min():+.2f} max {M[:,0].max():+.2f} | frozen-fixed {M[:,1].mean():+.2f} refit-fixed {M[:,2].mean():+.2f}", flush=True)
    F = I.fresh(d, range(100, 148))
    print(f"   48 fresh cuts: refit-frozen mean {F[:,0].mean():+.2f} sd {F[:,0].std(ddof=1):.2f} cuts above +1.0 {(F[:,0] > 1).sum()} of 48", flush=True)
