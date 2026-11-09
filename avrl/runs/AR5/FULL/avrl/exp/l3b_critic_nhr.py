"""L3b critic check 4 (CPU): influence and selection checks on the confident neighbor ratio (cell100, top half of ACC by s_maj). Post hoc, not part of the verdict."""
import numpy as np, pandas as pd, json
L = "/homes/rdubash/avrl_live/FULL/avrl/runs/L3b-judge-proxy"
d = pd.read_parquet(f"{L}/feats_J.parquet"); A = d[d.accept == 1].reset_index(drop=True); A["J"] = A.judge_error
def nhr(lab, g):
    df = pd.DataFrame({"y": lab.astype(float), "g": g}); base = df.y.mean(); s = df.groupby("g").y.agg(["sum", "count"]); df = df.join(s, on="g"); m = df[df["count"] > 1]
    loo = (m["sum"] - m["y"]) / (m["count"] - 1); return float(m.loc[loo > 0, "y"].mean() / base) if (loo > 0).any() and base > 0 else float("nan")
top = A[A.s_maj >= A.s_maj.median()].reset_index(drop=True); out = {"obs_cell100": nhr(top.J.values, top.cell100.values)}
cnt = top.groupby("cell100").J.sum().sort_values(ascending=False)
for k in (1, 2, 3, 5):
    t = top[~top.cell100.isin(cnt.index[:k])]; out[f"drop_top{k}_cells"] = {"n": int(len(t)), "n_err": int(t.J.sum()), "nhr": nhr(t.J.values, t.cell100.values)}
rng = np.random.default_rng(0); half = len(top)
rh = np.array([nhr(A.J.values[i], A.cell100.values[i]) for i in (rng.permutation(len(A))[:half] for _ in range(500))])
out["random_half_of_ACC_cell100"] = {"mean": float(np.nanmean(rh)), "p95": float(np.nanpercentile(rh, 95)), "frac_ge_2.093": float(np.nanmean(rh >= out["obs_cell100"]))}
# cell-level bootstrap CI of the top-half NHR (resample cells with their members)
cells = top.cell100.unique(); br = np.random.default_rng(1); vals = []
for _ in range(500):
    pick = br.choice(cells, len(cells)); t = pd.concat([top[top.cell100 == c].assign(cb=i) for i, c in enumerate(pick)]); vals.append(nhr(t.J.values, t.cb.values))
vals = np.array(vals); out["cell_bootstrap_top_half_cell100"] = {"mean": float(np.nanmean(vals)), "ci95": [float(np.nanpercentile(vals, 2.5)), float(np.nanpercentile(vals, 97.5))]}
# is the top half special because it is judge-selected? same stat on the top half chosen by agreement instead
top_ag = A.sort_values(["agreement", "n_distinct"], ascending=[False, True]).head(half); out["top_half_by_agreement_cell100"] = {"n_err": int(top_ag.J.sum()), "nhr": nhr(top_ag.J.values, top_ag.cell100.values)}
json.dump(out, open(f"{L}/critic_nhr.json", "w"), indent=1); print(json.dumps(out, indent=1))
