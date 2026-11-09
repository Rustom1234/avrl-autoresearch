"""Second critic: sensitivity of the claim 3 bin to the definition of the 'confident subset' for the cell100 neighbor ratio (J label on ACC). 1000-draw permutation null. Writes runs/L3b-judge-proxy/critic2_nhr_defs.json."""
import sys, json, numpy as np, pandas as pd
sys.path.insert(0, "/homes/rdubash/avrl_live/FULL"); from aa import data as D
R = "runs/L3b-judge-proxy"; F = pd.read_parquet(f"{R}/feats_J.parquet"); A = F[F.accept == 1].reset_index(drop=True); rng = np.random.default_rng(1); out = {}
def nhr(sub, k=100):
    y = sub.judge_error.values; c = sub[f"cell{k}"].values; obs = D.neighbor_hit_ratio(y, c)["ratio"]
    nl = np.array([D.neighbor_hit_ratio(rng.permutation(y), c)["ratio"] for _ in range(1000)])
    return dict(n=int(len(sub)), n_err=int(y.sum()), obs=float(obs), null_p95=float(np.nanpercentile(nl, 95)), frac_null_ge=float(np.nanmean(nl >= obs)))
q = lambda f: A[A.s_maj >= A.s_maj.quantile(1 - f)]
defs = {"top25_smaj": q(0.25), "top50_smaj_REGISTERED": A[A.s_maj >= A.s_maj.median()], "top75_smaj": q(0.75), "all_ACC": A, "agreement_ge_0.75_in_ACC": A[A.agreement >= 0.75],
        "agreement_ge_0.75_and_top50_smaj": A[(A.agreement >= 0.75) & (A.s_maj >= A.s_maj.median())]}
for k, s in defs.items(): out[k] = nhr(s)
json.dump(out, open(f"{R}/critic2_nhr_defs.json", "w"), indent=1)
for k, v in out.items(): print(k, v, "-> kill-side (obs<1.3)" if v["obs"] < 1.3 else ("pass-side NHR (>=1.5 and >p95)" if v["obs"] >= 1.5 and v["obs"] > v["null_p95"] else "between 1.3 and 1.5 or not above p95"))
