"""L2a: copy of exp/l1b_sim_alloc.py (ts_flat, ts_online, ts_static, neighbor, static_matched and the thr_margin variants dropped) with an extended charged fixed family and a learned grid.
Usage: l2a_sim_alloc.py <feats.parquet> <label> <group_col> <cheap_col1,cheap_col2,...> [out.json] [split_seed]
Units = docs in groups with >= MIN docs, inspected one at a time (K = 1). Policies:
  reference (must reproduce runs/L1b bit for bit): uniform, static (free labels, upper bound only), oracle, thr_agreement, recal_charged_<n>, greedy_online
  new fixed family (n labels charged to the budget and counted as hits, then a balanced logistic rule ranks the rest):
    seeded_charged_<n>: the n lowest-agreement docs (random ties) are the calibration labels; fallback to agreement order if the sample has 0 or all positives
    split_charged_<n>: n random docs drawn only from docs with n_distinct > 1 (guards against range restriction in the seeded sample); fallback: cal first, then agreement order
  new learned grid (greedy, no layout; first fit after n0 labels, then refit every R labels on all labels collected):
    greedy_n<n0>_R<R> (first n0 labels random) and greedy_seeded_n<n0>_R<R> (first n0 labels are the lowest-agreement docs), n0 in 25, 50, 100, R in n0, 4*n0
Every policy builds its own RNG from the policy seed, so the new draws cannot disturb the reference path."""
import os, sys, json
import numpy as np, pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
MIN, K, SEEDS, BUDGET_FRACS = 3, 1, 8, (0.10, 0.25, 0.50)
path, LABEL, G, cheap = sys.argv[1], sys.argv[2], sys.argv[3], sys.argv[4].split(",")
out_path = sys.argv[5] if len(sys.argv) > 5 else None
SPLIT_SEED = int(sys.argv[6]) if len(sys.argv) > 6 else 0
NCALS = (25, 50, 100, 200, 400); N0S = (25, 50, 100)
df = pd.read_parquet(path)
cnt = df[G].value_counts(); df = df[df[G].isin(cnt[cnt >= MIN].index)].reset_index(drop=True)
rng0 = np.random.default_rng(SPLIT_SEED); groups = df[G].unique(); rng0.shuffle(groups)
fit_g = set(groups[: len(groups) // 4]); fit = df[df[G].isin(fit_g)]; sim = df[~df[G].isin(fit_g)].reset_index(drop=True)
X = lambda d: d[cheap].fillna(0).values
clf = make_pipeline(StandardScaler(), LogisticRegression(max_iter=1000, class_weight="balanced")).fit(X(fit), fit[LABEL])
Xs = X(sim); sim["static"] = clf.predict_proba(Xs)[:, 1]
y = sim[LABEL].values.astype(int); codes, uniq = pd.factorize(sim[G].values); n_b = len(uniq)
members = [np.where(codes == b)[0] for b in range(n_b)]
agree = sim["agreement"].values; ndist = sim["n_distinct"].values
print(f"sim docs {len(sim)} buckets {n_b} base {y.mean():.4f} hits {int(y.sum())} median bucket {int(np.median([len(m) for m in members]))}", flush=True)

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

out = {"n_sim": int(len(sim)), "n_buckets": n_b, "hits": int(y.sum()), "split_seed": SPLIT_SEED, "label": LABEL, "group_col": G, "features": cheap, "n_policy_seeds": SEEDS}
for bf in BUDGET_FRACS:
    budget = int(bf * len(sim)); row = {}
    ns = [n for n in NCALS if n <= budget // 2]; n0s = [n for n in N0S if n <= budget // 2]
    jobs = [(p, None) for p in ("uniform", "static", "oracle", "thr_agreement")] + [("greedy_online", {})]
    jobs += [(f"{f}_charged_{n}", None) for f in ("recal", "seeded", "split") for n in ns]
    jobs += [(f"greedy_{'seeded_' if sd else ''}n{n0}_R{R}", dict(n0=n0, R=R, seeded=sd)) for sd in (False, True) for n0 in n0s for R in (n0, 4 * n0)]
    for pol, kw in jobs:
        r = [run_online(budget, s, **kw) if kw is not None else run(pol, budget, s) for s in range(SEEDS)]
        row[pol] = {"mean": float(np.mean(r)), "sd": float(np.std(r)), "per_seed": [int(v) for v in r]}
    out[f"budget_{bf}"] = row
    print(f"split {SPLIT_SEED} {bf:.0%} budget {budget}: " + " ".join(f"{p}={v['mean']:.1f}" for p, v in row.items()), flush=True)
if out_path: json.dump(out, open(out_path, "w"), indent=1)
