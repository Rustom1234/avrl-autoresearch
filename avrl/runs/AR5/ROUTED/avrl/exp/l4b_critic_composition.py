"""L4b critic: composition of the selected sets on the full L4b pool (test rows vs remaining-train rows), floor 40, splits 0 to 7, 8 seeds,
plus the fixed 5% hit rate on each pool's repro/result files (headroom). CPU only. Independent Pool code. Usage: python -I l4b_critic_composition.py
"""
import sys, json
sys.path.insert(0, "/homes/rdubash/avrl_live/ROUTED/avrl/exp")
import numpy as np, pandas as pd
import l4b_critic_indep as C

F = pd.read_parquet(f"{C.RUN}/L4b/feats.parquet"); rows = []
for s in range(8):
    d = C.build(F, s); P = C.Pool(d); is_test = (d.pid < 1319).to_numpy(); b5, b10 = int(0.05 * P.n), int(0.10 * P.n)
    for sd in range(8):
        f5 = P.fixed_set(b5, sd); f10 = P.fixed_set(b10, sd); b, _ = P.recal_set(b10, sd, 50, max(b10 // 8, 40), "xt")
        rows.append(dict(split=s, seed=sd, sim_test_frac=is_test.mean(), fix5_test=is_test[f5].mean(), fix10_test=is_test[f10].mean(), B10_test=is_test[b].mean(),
                         fix10_hit=P.y[f10].mean(), B10_hit=P.y[b].mean(), fix10_ag_max=P.ag[f10].max(), B10_ag_max=P.ag[b].max(),
                         fix5_ag_max=P.ag[f5].max(), fix5_n_ag125=int((P.ag[f5] == 0.125).sum()), n_ag125_sim=int((P.ag == 0.125).sum()), b5=b5))
R = pd.DataFrame(rows)
print("mean over splits and seeds:"); print(R.drop(columns=["split", "seed"]).mean().round(3).to_string())
print("share of sim docs with agreement 0.125 vs 5% budget (split-mean):", round(R.groupby("split").n_ag125_sim.mean().mean(), 1), "vs", round(R.groupby("split").b5.mean().mean(), 1))
for nm, path in (("L1B pool (p0)", f"{C.RUN}/L3a/l1b_pool/result.json"), ("L2b pool", f"{C.RUN}/L3a/fresh/result.json"), ("L4b pool", f"{C.RUN}/L4b/result.json")):
    c = json.load(open(path))["configs"]; k = "A40" if "A40" in c else "A"
    print(f"{nm}: fixed 5% hit mean {np.mean(c[k]['fixed_hit_pct_per_split']):.2f}, A floor 40 gap {c[k]['all_splits']['mean_pp']:+.2f}")
