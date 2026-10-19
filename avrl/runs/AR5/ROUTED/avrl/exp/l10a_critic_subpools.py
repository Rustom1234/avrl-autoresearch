"""L10a critic: L4b remaining-train rows (pid > 1318) with the same independent replay as l10a_critic_indep.py. CPU only."""
import sys, numpy as np, pandas as pd
sys.dont_write_bytecode = True
sys.path.insert(0, "/homes/rdubash/avrl_live/ROUTED/avrl/exp")
import l10a_critic_indep as C
ev = pd.read_parquet(f"{C.R}/L4b/feats.parquet"); er = ev[ev["pid"] > 1318]
A, ns, bs, nu = C.registered(er); M = A.mean(axis=0)
print(f"L4b remaining-train rows (pid>1318, {len(er)} rows): pooled-frozen mean {M[:,0].mean():+.2f} sd {M[:,0].std(ddof=1):.2f} "
      f"subsets>+1.0 {(M[:,0] > 1).sum()} of 20 min {M[:,0].min():+.2f} max {M[:,0].max():+.2f} | frozen-fixed {M[:,1].mean():+.2f} pooled-fixed {M[:,2].mean():+.2f}; budgets {min(bs)}..{max(bs)}, updates {nu}", flush=True)
F = C.fresh(er, range(100, 148))
print(f"  48 fresh cuts: pooled-frozen mean {F[:,0].mean():+.2f} sd {F[:,0].std(ddof=1):.2f} min {F[:,0].min():+.2f} max {F[:,0].max():+.2f} cuts above +1.0: {(F[:,0] > 1).sum()} of 48", flush=True)
