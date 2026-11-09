"""L4a: copy of exp/l2b_sim_alloc.py. run() and run_online() are kept verbatim (every L2b policy stays bit-identical to L2b at the same m and subset seed).
Usage: l4a_sim_alloc.py <feats.parquet> <label> <cheap_col1,cheap_col2,...> <m> <subset_seed> [out.json]
  m > 0: the sim population is m prompts drawn without replacement (default_rng(subset_seed)) from all rows of the feats file; units = prompts, K = 1.
         static = optional upper-bound reference: logistic rule fit with free labels on the rows OUTSIDE the subset (never a member of the fixed family).
  m = 0: L2a population construction verbatim (kept from L2b, not used by L4a).
Policies (decided in advance):
  fixed family: thr_agreement, and recal_charged_n / seeded_charged_n / split_charged_n for n in NCALS with n <= budget // 2 (the L2b family)
  sensitivity-only fixed extras: the same three charged rules at n = budget // 2 when that n is not in NCALS (removes the L2b cap truncation; not in the verdict)
  plain learned family: random-start greedy_n<n0>_R<R> for (n0, R) in GRID (the L2b grid plus the six new small-end points) with n0 <= budget // 2,
                        plus the L1b reference greedy_online and greedy_seeded_n25_R25 (if 25 <= budget // 2)
  offset learned family: offset_n<n0>_R<R>_lam<lam> (see run_offset), lam in 1, 10, 100; R in 5, 10; n0 in 0, 5 (12 arms, fixed in advance)
  also uniform, static (free labels, upper bound only), oracle. Policy seeds 0..7.
Every policy builds its own RNG from the policy seed, so the draws cannot disturb the reference path.
Env L4A_SANITY=<out.json>: instead of the sweep, check that run_offset with lam = inf (w = 0) reproduces thr_agreement per-seed hits, and write the differences."""
import os, sys, json
import numpy as np, pandas as pd
from scipy.special import expit
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
MIN, K, SEEDS, BUDGET_FRACS = 3, 1, 8, (0.10, 0.25, 0.50)
path, LABEL, cheap = sys.argv[1], sys.argv[2], sys.argv[3].split(",")
M, SUBSET_SEED = int(sys.argv[4]), int(sys.argv[5])
out_path = sys.argv[6] if len(sys.argv) > 6 else None
G = "cell100"
NCALS = (10, 15, 25, 50, 75, 100, 150, 200, 250); GRID = ((10, 5), (10, 10), (15, 15), (25, 10), (25, 25))
NEW_GRID = ((3, 1), (5, 1), (5, 2), (5, 5), (10, 1), (10, 2))  # L4a: plain learner below the L2b grid corner (10, 5)
OFFSET_ARMS = tuple((n0, R, lam) for lam in (1, 10, 100) for R in (5, 10) for n0 in (0, 5))  # fixed in advance, none chosen on held-out subsets
OFFSET_C = 2.0  # score = -OFFSET_C * standardized agreement + w.x (constant fixed in advance)
df = pd.read_parquet(path)
X = lambda d: d[cheap].fillna(0).values
if M == 0:  # L2a population construction, verbatim
    cnt = df[G].value_counts(); df = df[df[G].isin(cnt[cnt >= MIN].index)].reset_index(drop=True)
    rng0 = np.random.default_rng(SUBSET_SEED); groups = df[G].unique(); rng0.shuffle(groups)
    fit_g = set(groups[: len(groups) // 4]); fit = df[df[G].isin(fit_g)]; sim = df[~df[G].isin(fit_g)].reset_index(drop=True)
    codes, uniq = pd.factorize(sim[G].values); n_b = len(uniq)
else:
    idx = np.sort(np.random.default_rng(SUBSET_SEED).choice(len(df), M, replace=False))
    fit = df.drop(df.index[idx]); sim = df.iloc[idx].reset_index(drop=True); n_b = len(sim)
clf = make_pipeline(StandardScaler(), LogisticRegression(max_iter=1000, class_weight="balanced")).fit(X(fit), fit[LABEL])
Xs = X(sim); sim["static"] = clf.predict_proba(Xs)[:, 1]
y = sim[LABEL].values.astype(int)
agree = sim["agreement"].values; ndist = sim["n_distinct"].values
# offset-learner features: standardized on the m sim prompts (unlabeled, so no label leakage); agreement is the fixed-slope offset, the other nine are the ridge-penalized correction
_mu, _sd = Xs.mean(0), Xs.std(0); Zs = (Xs - _mu) / np.where(_sd > 0, _sd, 1.0)
IA = cheap.index("agreement"); Zoff = -OFFSET_C * Zs[:, IA]; Zoth = np.delete(Zs, IA, axis=1)
print(f"m {M} subset_seed {SUBSET_SEED} sim docs {len(sim)} units {n_b} base {y.mean():.4f} hits {int(y.sum())} fit docs {len(fit)}", flush=True)

def rule(Xa, ya, max_iter=1000): return make_pipeline(StandardScaler(), LogisticRegression(max_iter=max_iter, class_weight="balanced")).fit(Xa, ya)

def charged_order(cal, fallback):
    """cal labels are verified first (charged, counted as hits); a rule fit on them ranks the rest. fallback must start with cal."""
    if y[cal].sum() == 0 or y[cal].sum() == len(cal): return fallback
    sc = rule(Xs[cal], y[cal]).predict_proba(Xs)[:, 1]; sc[cal] = -1
    return np.concatenate([cal, np.argsort(-sc)[: len(y) - len(cal)]])

def run(policy, budget, seed):
    rng = np.random.default_rng(seed)
    if policy == "uniform": order = rng.permutation(len(y))
    elif policy.startswith("recal_charged_"):  # n_cal random verified docs
        n_cal = int(policy.rsplit("_", 1)[1]); perm = rng.permutation(len(y)); order = charged_order(perm[:n_cal], perm)
    elif policy.startswith("seeded_charged_"):  # n_cal lowest-agreement docs, random ties
        n_cal = int(policy.rsplit("_", 1)[1]); ag = np.lexsort((rng.random(len(y)), agree)); order = charged_order(ag[:n_cal], ag)
    elif policy.startswith("split_charged_"):  # n_cal random docs among those with n_distinct > 1
        n_cal = int(policy.rsplit("_", 1)[1]); pool = np.where(ndist > 1)[0]; cal = rng.choice(pool, min(n_cal, len(pool)), replace=False)
        ag = np.lexsort((rng.random(len(y)), agree)); order = charged_order(cal, np.concatenate([cal, ag[~np.isin(ag, cal)]]))
    elif policy == "static": order = np.argsort(-sim.static.values)
    elif policy == "oracle": order = np.argsort(-(y + 1e-6 * rng.random(len(y))))
    elif policy == "thr_agreement": order = np.lexsort((rng.random(len(y)), agree))  # low agreement first, random ties, no labels
    else: raise ValueError(policy)
    return int(y[order[:budget]].sum())

def run_online(budget, seed, n0=None, R=None, seeded=False):
    """greedy_online: inspect the top-scored unseen docs anywhere; score is random (or lowest-agreement first when seeded) until the first fit after n0 labels,
    then a classifier is refit on all labels collected every R labels. n0 = R = max(budget // 8, 40) is the L1b reference."""
    rng = np.random.default_rng(seed); R = R or max(budget // 8, 40); n0 = n0 or R
    seen = np.zeros(len(y), bool); hits = spent = 0; next_fit = n0
    if seeded:
        ag = np.lexsort((rng.random(len(y)), agree)); score = np.empty(len(y)); score[ag] = -np.arange(len(y), dtype=float)
    else: score = rng.random(len(y)) * 1e-3
    while spent < budget:
        take = min(K, budget - spent)
        mu = np.where(~seen)[0]; c = mu[np.argsort(-score[mu])][:take]; seen[c] = True; hits += int(y[c].sum()); spent += len(c)
        if spent >= next_fit and 0 < y[seen].sum() < seen.sum():
            score = rule(Xs[seen], y[seen], 500).predict_proba(Xs)[:, 1]; next_fit += R
    return hits

def ridge_logit(Z, yy, off, lam, iters=100):
    """Ridge logistic regression with a fixed offset, by Newton/IRLS with backtracking: eta = off + b + Z w, loss = sum(log(1+e^eta) - y eta) + lam/2 |w|^2,
    intercept b unpenalized, no class weights. lam = inf returns w = 0. Returns w (len Z.shape[1])."""
    p = Z.shape[1]
    if np.isinf(lam): return np.zeros(p)
    A = np.hstack([np.ones((len(yy), 1)), Z]); pen = np.r_[0.0, np.full(p, float(lam))]; beta = np.zeros(p + 1)
    nll = lambda b: float(np.sum(np.logaddexp(0.0, off + A @ b) - yy * (off + A @ b)) + 0.5 * np.sum(pen * b * b))
    f = nll(beta)
    for _ in range(iters):
        pr = expit(off + A @ beta); g = A.T @ (pr - yy) + pen * beta
        H = (A * (pr * (1 - pr))[:, None]).T @ A + np.diag(pen) + 1e-10 * np.eye(p + 1)
        step = np.linalg.solve(H, g); t = 1.0
        while t > 1e-10:
            nb = beta - t * step; fn = nll(nb)
            if fn <= f: break
            t /= 2
        else: break
        done = f - fn < 1e-10 * (1 + abs(f)); beta, f = nb, fn
        if done: break
    return beta[1:]

def run_offset(budget, seed, n0, R, lam):
    """Warm-start learner. score = -OFFSET_C * z_agreement + w.z_other; w = 0 at the start, so the first labels are the lowest-agreement prompts with random ties
    (identical to thr_agreement's prefix). The first refit is attempted after n0 labels (after R labels when n0 = 0), then every R labels: w is refit by ridge-logistic IRLS
    (offset = -OFFSET_C * z_agreement, intercept unpenalized) on all labels so far. A refit is skipped while the labels are all one class (retried at every label until it can run)."""
    rng = np.random.default_rng(seed); tb = rng.random(len(y))  # same first draw as thr_agreement, so ties break identically
    score = Zoff.copy(); seen = np.zeros(len(y), bool); hits = spent = 0; next_fit = n0 or R
    while spent < budget:
        mu = np.where(~seen)[0]; c = mu[np.lexsort((tb[mu], -score[mu]))[0]]
        seen[c] = True; hits += int(y[c]); spent += 1
        if spent >= next_fit and 0 < y[seen].sum() < seen.sum():
            score = Zoff + Zoth @ ridge_logit(Zoth[seen], y[seen].astype(float), Zoff[seen], lam); next_fit = spent + R
    return hits

if os.environ.get("L4A_SANITY"):
    rows = {}
    for bf in BUDGET_FRACS:
        budget = int(bf * len(sim)); thr = [run("thr_agreement", budget, s) for s in range(SEEDS)]
        for n0, R in ((0, 5), (0, 10), (5, 5), (5, 10)):
            off = [run_offset(budget, s, n0, R, np.inf) for s in range(SEEDS)]
            rows[f"budget_{bf}_n{n0}_R{R}"] = {"budget": budget, "thr_agreement": thr, "offset_lam_inf": off, "max_abs_diff": int(np.max(np.abs(np.array(thr) - np.array(off))))}
    res = {"m": M, "subset_seed": SUBSET_SEED, "label": LABEL, "max_abs_diff_overall": max(v["max_abs_diff"] for v in rows.values()), "n_comparisons": len(rows) * SEEDS, "rows": rows}
    json.dump(res, open(os.environ["L4A_SANITY"], "w"), indent=1); print(f"sanity m {M} subset {SUBSET_SEED}: max abs diff in hits {res['max_abs_diff_overall']} over {res['n_comparisons']} per-seed comparisons", flush=True); sys.exit(0)

out = {"n_sim": int(len(sim)), "n_buckets": n_b, "hits": int(y.sum()), "split_seed": SUBSET_SEED, "subset_seed": SUBSET_SEED, "m": M, "label": LABEL, "features": cheap, "n_policy_seeds": SEEDS, "n_fit_docs": int(len(fit)), "offset_arms": [list(a) for a in OFFSET_ARMS], "offset_c": OFFSET_C, "grid": [list(g) for g in GRID + NEW_GRID]}
for bf in BUDGET_FRACS:
    budget = int(bf * len(sim)); row = {}
    ns = [n for n in NCALS if n <= budget // 2]; grid = [(n0, R) for n0, R in GRID + NEW_GRID if n0 <= budget // 2]
    jobs = [(p, None) for p in ("uniform", "static", "oracle", "thr_agreement")] + [("greedy_online", {})]
    jobs += [(f"{f}_charged_{n}", None) for f in ("recal", "seeded", "split") for n in ns]
    jobs += [(f"greedy_n{n0}_R{R}", dict(n0=n0, R=R, seeded=False)) for n0, R in grid]
    if 25 <= budget // 2: jobs += [("greedy_seeded_n25_R25", dict(n0=25, R=25, seeded=True))]
    if budget // 2 not in NCALS: jobs += [(f"{f}_charged_{budget // 2}", None) for f in ("recal", "seeded", "split")]  # sensitivity-only extras (n = budget // 2)
    jobs += [(f"offset_n{n0}_R{R}_lam{lam}", dict(offset=(n0, R, lam))) for n0, R, lam in OFFSET_ARMS if n0 <= budget // 2]
    for pol, kw in jobs:
        r = [run_offset(budget, s, *kw["offset"]) if kw is not None and "offset" in kw else run_online(budget, s, **kw) if kw is not None else run(pol, budget, s) for s in range(SEEDS)]
        row[pol] = {"mean": float(np.mean(r)), "sd": float(np.std(r)), "per_seed": [int(v) for v in r]}
    out[f"budget_{bf}"] = row
    print(f"m {M} subset {SUBSET_SEED} {bf:.0%} budget {budget}: " + " ".join(f"{p}={v['mean']:.1f}" for p, v in row.items()), flush=True)
if out_path: json.dump(out, open(out_path, "w"), indent=1)
