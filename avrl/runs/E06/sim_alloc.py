"""Minimal budgeted-audit simulator. Usage: sim_generic.py <feats.parquet> <label> <group_col> <cheap_col1,cheap_col2,...> [out.json]
Buckets = groups with >= MIN docs. Inspect K docs per visit. Policies: uniform, static triage (cheap features, fit on disjoint groups),
Thompson flat, Thompson + static prior, neighbor (stay on a bucket after a hit), oracle. Hits at 1/3/10% budgets, mean over seeds."""
import os, sys, json
import numpy as np, pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
MIN, K, SEEDS, BUDGET_FRACS = 3, 1, 8, (0.10, 0.25, 0.50)
path, LABEL, G, cheap = sys.argv[1], sys.argv[2], sys.argv[3], sys.argv[4].split(",")
out_path = sys.argv[5] if len(sys.argv) > 5 else None
df = pd.read_parquet(path)
cnt = df[G].value_counts(); df = df[df[G].isin(cnt[cnt >= MIN].index)].reset_index(drop=True)
rng0 = np.random.default_rng(0); groups = df[G].unique(); rng0.shuffle(groups)
fit_g = set(groups[: len(groups) // 4]); fit = df[df[G].isin(fit_g)]; sim = df[~df[G].isin(fit_g)].reset_index(drop=True)
X = lambda d: d[cheap].fillna(0).values
clf = make_pipeline(StandardScaler(), LogisticRegression(max_iter=1000, class_weight="balanced")).fit(X(fit), fit[LABEL])
Xs = X(sim); sim["static"] = clf.predict_proba(Xs)[:, 1]
y = sim[LABEL].values.astype(int); codes, uniq = pd.factorize(sim[G].values); n_b = len(uniq)
members = [np.where(codes == b)[0] for b in range(n_b)]
prior = pd.Series(sim.static.values).groupby(codes).mean().values; prior = prior / prior.max()
print(f"sim docs {len(sim)} buckets {n_b} base {y.mean():.4f} hits {int(y.sum())} median bucket {int(np.median([len(m) for m in members]))}", flush=True)

def run(policy, budget, seed):
    rng = np.random.default_rng(seed); hits = spent = 0
    alpha = np.ones(n_b); beta = np.ones(n_b); last = None; last_hit = False
    mem = [rng.permutation(m) for m in members] if policy == "ts_flat" else [m[np.argsort(-sim.static.values[m])] for m in members]
    ptr = np.zeros(n_b, int); remaining = np.array([len(m) for m in mem])
    if policy == "uniform": order = rng.permutation(len(y))
    elif policy == "static_matched":  # fair static: spend half the budget on random labels, fit, rank the rest
        perm = rng.permutation(len(y)); tr = perm[: max(budget // 2, 1)]
        if y[tr].sum() == 0: order = perm
        else:
            sc = make_pipeline(StandardScaler(), LogisticRegression(max_iter=1000, class_weight="balanced")).fit(Xs[tr], y[tr]).predict_proba(Xs)[:, 1]
            sc[tr] = -1; order = np.concatenate([tr, np.argsort(-sc)])
    elif policy == "static": order = np.argsort(-sim.static.values)
    elif policy == "oracle": order = np.argsort(-(y + 1e-6 * rng.random(len(y))))
    while spent < budget:
        take = min(K, budget - spent)
        if policy in ("uniform", "static", "static_matched", "oracle"):
            c = order[spent:spent + take]
        else:
            avail = remaining > 0
            if policy == "ts_static":
                s = rng.beta(alpha + 10 * prior, beta + 10 * (1 - prior))
            elif policy == "ts_flat":
                s = rng.beta(alpha, beta)
            elif policy == "neighbor":
                s = prior + 0.01 * rng.random(n_b)
                if last is not None and avail[last] and last_hit: s[last] = 9
            s[~avail] = -1; b = int(np.argmax(s)); c = mem[b][ptr[b]:ptr[b] + take]; ptr[b] += len(c); remaining[b] -= len(c)
            h = int(y[c].sum()); alpha[b] += h; beta[b] += len(c) - h; last, last_hit = b, h > 0
        hits += int(y[c].sum()); spent += len(c)
    return hits

def run_online(budget, seed, R=None, greedy=False):
    """ts_online: Thompson over buckets, within-bucket ranking and bucket prior from a classifier refit every R labels on the labels collected so far."""
    rng = np.random.default_rng(seed); R = R or max(budget // 8, 40)
    seen = np.zeros(len(y), bool); score = rng.random(len(y)) * 1e-3; bprior = None
    alpha = np.ones(n_b); beta = np.ones(n_b); remaining = np.array([len(m) for m in members]); hits = spent = 0; next_fit = R
    while spent < budget:
        take = min(K, budget - spent); avail = remaining > 0
        if greedy:  # no layout: inspect the top-scored unseen docs anywhere (random until the first fit)
            mu = np.where(~seen)[0]; c = mu[np.argsort(-score[mu])][:take]; seen[c] = True; hits += int(y[c].sum()); spent += len(c)
        else:
            s = rng.beta(alpha + 10 * bprior, beta + 10 * (1 - bprior)) if bprior is not None else rng.beta(alpha, beta)
            s[~avail] = -1; b = int(np.argmax(s)); mu = members[b][~seen[members[b]]]; c = mu[np.argsort(-score[mu])][:take]
            seen[c] = True; remaining[b] -= len(c); h = int(y[c].sum()); alpha[b] += h; beta[b] += len(c) - h; hits += h; spent += len(c)
        if spent >= next_fit and 0 < y[seen].sum() < seen.sum():
            score = make_pipeline(StandardScaler(), LogisticRegression(max_iter=500, class_weight="balanced")).fit(Xs[seen], y[seen]).predict_proba(Xs)[:, 1]
            bp = pd.Series(score).groupby(codes).mean().values; bprior = bp / bp.max(); next_fit += R
    return hits

out = {"n_sim": int(len(sim)), "n_buckets": n_b, "hits": int(y.sum())}
for bf in BUDGET_FRACS:
    budget = int(bf * len(sim)); row = {}
    for pol in ["uniform", "static", "static_matched", "ts_flat", "greedy_online", "ts_online", "ts_static", "neighbor", "oracle"]:
        r = [run_online(budget, s, greedy=(pol == "greedy_online")) if pol in ("ts_online", "greedy_online") else run(pol, budget, s) for s in range(SEEDS)]; row[pol] = {"mean": float(np.mean(r)), "sd": float(np.std(r))}
    out[f"budget_{bf}"] = row
if out_path: json.dump(out, open(out_path, "w"), indent=1)
print(f"\n{LABEL} by {G} | budget | uniform | static (free labels) | static (budget-matched) | TS flat (no labels) | greedy online (self-trained, no layout) | TS online (self-trained) | TS + free static prior | neighbor (free static) | oracle |\n|---|---|---|---|---|---|---|---|---|---|")
for bf in BUDGET_FRACS:
    row = out[f"budget_{bf}"]; u = row["uniform"]["mean"]
    print(f"| {bf:.0%} | {u:.0f} | " + " | ".join(f"{row[p]['mean']:.0f}±{row[p]['sd']:.0f} ({row[p]['mean']/max(u,1):.2f}x)" for p in ["static", "static_matched", "ts_flat", "greedy_online", "ts_online", "ts_static", "neighbor", "oracle"]) + " |")
