"""L2b critic: sd of the pre-registered gap over 64 random 75-of-100 bucket cuts (same cut as the replay, split seeds 0 to 63).
Delete-d jackknife scale factor (n-d)/d = 75/25 = 3 turns the cut-to-cut variance into a rough pool-level variance. CPU only."""
import sys, json
from multiprocessing import Pool
import numpy as np, pandas as pd
sys.path.insert(0, "/homes/rdubash/avrl_live/ROUTED/avrl/exp")
import l1b_tuned_threshold as L
FE = "/homes/rdubash/avrl_live/ROUTED/avrl/runs/L2b/feats.parquet"; SPEC = dict(L.SPECS)
CFG = {"A": ("k02_Rdiv8_agreement_x_nd", 0.05), "B": ("k050_Rdiv8_agreement_x_nd_x_trunc", 0.10)}
def one(sp):
    sim, _ = L.build_sim(pd.read_parquet(FE), "cell100", sp); S = L.Sim(sim, "proxy_error"); o = {}
    for t, (name, f) in CFG.items():
        b = int(f * S.n); o[t] = float(np.mean([100 * (L.hits_for(S, SPEC[name], b, s) - L.hits_for(S, SPEC[L.FIXED], b, s)) / b for s in range(8)]))
    return o
if __name__ == "__main__":
    with Pool(16) as p: R = p.map(one, range(64))
    for t in "AB":
        v = np.array([r[t] for r in R]); print(f"{t}: 64 cuts mean {v.mean():+.2f} sd {v.std(ddof=1):.2f} min {v.min():+.2f} max {v.max():+.2f} | rough pool-level sd = sqrt(3) * sd = {np.sqrt(3)*v.std(ddof=1):.2f} | z vs +2.0 = {(v.mean()-2)/(np.sqrt(3)*v.std(ddof=1)):.2f}")
    json.dump(R, open("/homes/rdubash/avrl_live/ROUTED/avrl/runs/L2b_critic/splits64.json", "w"))
