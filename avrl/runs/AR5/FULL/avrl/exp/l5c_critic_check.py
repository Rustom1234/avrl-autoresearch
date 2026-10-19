"""L5c critic check (CPU). Recomputes the exploratory diagnostics (a)-(d) quoted in the L5c ledger entry and the sensitivity of the rm agree_acc replay to the gold-derived accept volume K."""
import numpy as np, pandas as pd, json
from scipy.stats import spearmanr
Q = "/homes/rdubash/avrl_live/FULL/avrl/runs/"
F = pd.read_parquet(Q + "E21-lowT-confident-errors/feats_T0.25.parquet").sort_values("pid").reset_index(drop=True)
P = {"rm": pd.read_parquet(Q + "L5c-proxies/rm/feats_J.parquet").sort_values("pid").reset_index(drop=True),
     "j3b": pd.read_parquet(Q + "L5c-proxies/judge3b/feats_J.parquet").sort_values("pid").reset_index(drop=True),
     "q4b": pd.read_parquet(Q + "L3b-judge-proxy/feats_J.parquet").sort_values("pid").reset_index(drop=True)}
def nhr(y, codes):
    y = y.astype(float); cnt = np.bincount(codes); sm = np.bincount(codes, weights=y); base = y.mean(); cn = cnt[codes]; m = cn > 1
    loo = (sm[codes] - y)[m] / (cn[m] - 1); sel = loo > 0
    return y[m][sel].mean() / base
# (a) concentration of top-half J errors in cells (cell100)
for k, d in P.items():
    A = d[d.accept == 1]; top = A[A.s_maj >= A.s_maj.median()]; e = top[top.judge_error == 1]
    vc = e.cell100.value_counts(); print(k, "top-half errors", len(e), "top3 cells share", [round(x / len(e), 3) for x in (vc.iloc[:1].sum(), vc.iloc[:2].sum(), vc.iloc[:3].sum())], "max in one cell", vc.iloc[0], "top3 sum share", round(vc.iloc[:3].sum() / len(e), 3))
# (b) per-cell J rate in top half correlated across proxies (cells with >=1 top-half group in all)
def cellrate(d):
    A = d[d.accept == 1]; top = A[A.s_maj >= A.s_maj.median()]; return top.groupby("cell100").judge_error.mean()
R = pd.concat({k: cellrate(d) for k, d in P.items()}, axis=1).dropna()
print("cells", len(R)); print("spearman", {f"{a}-{b}": round(spearmanr(R[a], R[b])[0], 3) for a, b in [("rm", "j3b"), ("rm", "q4b"), ("j3b", "q4b")]})
# (c) consensus label
cons_acc = (P["rm"].accept.values == 1) & (P["j3b"].accept.values == 1) & (P["q4b"].accept.values == 1)
mv_wrong = F.majority_correct.values == 0; cons_err = cons_acc & mv_wrong
print("consensus accepted", cons_acc.sum(), "consensus errors", cons_err.sum())
ranks = np.mean([pd.Series(P[k].s_maj.values).rank(ascending=False).values for k in P], axis=0)  # lower = more confident
for nm, rk in [("mean rank of s_maj (rank over all 2000 groups)", ranks)]:
    idx = np.where(cons_acc)[0]; r = rk[idx]; order = idx[np.argsort(r, kind="stable")]; half = order[: len(order) // 2]
    y = cons_err[half].astype(int); codes = pd.factorize(F.cell100.values[half])[0]; rng = np.random.default_rng(5)
    nl = np.array([nhr(rng.permutation(y), codes) for _ in range(1000)]); print(nm, "top half n", len(half), "errors", y.sum(), "nhr", round(nhr(y, codes), 3), "p95", round(np.percentile(nl, 95), 3))
    ally = cons_err[idx].astype(int); codes = pd.factorize(F.cell100.values[idx])[0]; nl = np.array([nhr(rng.permutation(ally), codes) for _ in range(1000)]); print(" all consensus-accepted", len(idx), ally.sum(), "nhr", round(nhr(ally, codes), 3), "p95", round(np.percentile(nl, 95), 3))
# (d) label check on consensus errors
def num(x):
    try: return float(str(x).replace(",", "").replace("$", ""))
    except Exception: return np.nan
g = pd.to_numeric(F.gold.map(num), errors="coerce"); m = pd.to_numeric(F.majority.map(num), errors="coerce")
E = np.where(cons_err)[0]; gg, mm = g.values[E], m.values[E]
within10 = np.sum(np.abs(mm - gg) <= 0.1 * np.abs(gg)); fac = 0
for a, b in zip(gg, mm):
    if a and b and not np.isnan(a) and not np.isnan(b) and a != 0 and b != 0:
        r = max(abs(a / b), abs(b / a))
        if any(abs(r - f) < 1e-6 or abs(r - f) / f < 0.01 for f in (2, 10, 100)): fac += 1
print("consensus errors", len(E), "within 10% of gold", int(within10), "factor 2/10/100", fac, "nan", int(np.isnan(mm).sum()))
# sensitivity of agree_acc replay (25%, 500 audits) to accept volume K for rm
rm = P["rm"].copy(); y = rm.judge_error.values  # labels differ with K: recompute
corr = F.majority_correct.values; s = rm.s_maj.values; ag = rm.agreement.values; nd = rm.n_distinct.values
out = {}
for K in (1500, 1600, 1715, 1800, 1840, 1900, 2000):
    acc = np.zeros(2000, bool); acc[np.argsort(-s, kind="stable")[:K]] = True; yy = (acc & (corr == 0)).astype(int); res = []
    for sd in range(300):
        u = np.random.default_rng(1000 + sd).random(2000); o = np.lexsort((u, np.where(acc, -nd, 0), np.where(acc, ag, 0), (~acc).astype(int))); res.append(yy[o[:500]].sum())
    out[K] = (int(yy.sum()), round(float(np.mean(res)), 1))
print("rm agree_acc at 25% by K -> (n_J, mean caught):", out)
