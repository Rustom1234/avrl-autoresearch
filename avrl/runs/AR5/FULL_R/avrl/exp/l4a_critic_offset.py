"""L4a critic: independent re-implementation of the offset learner (scipy L-BFGS instead of IRLS) vs saved per-seed hits. CPU only."""
import json, numpy as np, pandas as pd
from scipy.optimize import minimize
cheap = "agreement,n_distinct,mean_tokens,std_tokens,any_truncated,log_qlen,judge_top,judge_margin,judge_mean,judge_std".split(",")
def check(name, m, s, arms, bf):
    df = pd.read_parquet(f"runs/E04/feats_{name}.parquet"); idx = np.sort(np.random.default_rng(s).choice(len(df), m, replace=False)); sim = df.iloc[idx].reset_index(drop=True)
    X = sim[cheap].fillna(0).values.astype(float); y = sim.judge_error.values.astype(float); sd = X.std(0); Z = (X - X.mean(0)) / np.where(sd > 0, sd, 1)
    ia = cheap.index("agreement"); off = -2.0 * Z[:, ia]; Zo = np.delete(Z, ia, 1); B = int(bf * m)
    saved = json.load(open(f"runs/L4a/sim_{name}_m{m}_s{s}.json"))[f"budget_{bf}"]
    def fit(idxs, lam):
        A = np.hstack([np.ones((len(idxs), 1)), Zo[idxs]]); o = off[idxs]; yy = y[idxs]
        def f(b):
            e = o + A @ b; pen = lam * np.sum(b[1:] ** 2) / 2
            return np.sum(np.logaddexp(0, e) - yy * e) + pen, A.T @ (1 / (1 + np.exp(-e)) - yy) + np.r_[0, lam * b[1:]]
        r = minimize(f, np.zeros(A.shape[1]), jac=True, method="L-BFGS-B", options=dict(maxiter=500, ftol=1e-14, gtol=1e-10)); return r.x[1:]
    out = {}
    for n0, R, lam in arms:
        mine = []
        for seed in range(8):
            rng = np.random.default_rng(seed); tb = rng.random(m); w = np.zeros(Zo.shape[1]); seen = np.zeros(m, bool); hits = 0; nxt = n0 or R; order = []
            for t in range(1, B + 1):
                sc = off + Zo @ w; cand = np.where(~seen)[0]; best = cand[np.lexsort((tb[cand], -sc[cand]))[0]]
                seen[best] = True; hits += int(y[best])
                if t >= nxt and 0 < y[seen].sum() < seen.sum(): w = fit(np.where(seen)[0], lam); nxt = t + R
            mine.append(hits)
        sv = saved[f"offset_n{n0}_R{R}_lam{lam}"]["per_seed"]; out[(n0, R, lam)] = (mine, sv, sum(a != b for a, b in zip(mine, sv)))
        print(name, m, s, bf, (n0, R, lam), "mine", mine, "saved", sv, "ndiff", out[(n0, R, lam)][2], flush=True)
check("Qwen2.5-7B-Instruct-AWQ", 250, 5, [(0, 5, 1), (5, 10, 10), (0, 5, 100)], 0.5)
check("Qwen2.5-3B-Instruct", 250, 2, [(0, 5, 1), (5, 10, 10)], 0.25)
