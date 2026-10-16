"""Budgeted audit simulator on code: buckets = repos with >= MIN files. Inspect K files per visit.
Policies: uniform, static triage (cheap features, fit on disjoint repos), Thompson bandit over repos
with static prior, neighbor-expansion (revisit a repo after a hit), oracle. Label = any_secret (regex proxy)."""
import os, sys, json
import numpy as np, pandas as pd
ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import make_pipeline
MIN, K, SEEDS, BUDGET_FRACS = 3, 4, 3, (0.01, 0.03, 0.10)
LABEL = sys.argv[1] if len(sys.argv) > 1 else "any_secret"
df = pd.read_parquet(os.path.join(ROOT, "runs", "code", "feats.parquet"))
cnt = df.repo_name.value_counts(); df = df[df.repo_name.isin(cnt[cnt >= MIN].index)].reset_index(drop=True)
rng0 = np.random.default_rng(0); repos = df.repo_name.unique(); rng0.shuffle(repos)
fit_repos = set(repos[: len(repos) // 4]); fit = df[df.repo_name.isin(fit_repos)]; sim = df[~df.repo_name.isin(fit_repos)].reset_index(drop=True)
ct = ColumnTransformer([("cat", OneHotEncoder(handle_unknown="ignore", min_frequency=50), ["language", "ext"]), ("num", StandardScaler(), ["log_size", "depth"])])
clf = make_pipeline(ct, LogisticRegression(max_iter=1000)).fit(fit[["language", "ext", "log_size", "depth"]], fit[LABEL])
sim["static"] = clf.predict_proba(sim[["language", "ext", "log_size", "depth"]])[:, 1]
y = sim[LABEL].values; g = sim.repo_name.values; codes, uniq = pd.factorize(g); n_b = len(uniq)
members = [np.where(codes == b)[0] for b in range(n_b)]
prior = pd.Series(sim.static.values).groupby(codes).mean().values; prior = prior / prior.max()
print(f"sim docs {len(sim)} repos {n_b} base {y.mean():.4f} hits {int(y.sum())}", flush=True)

def run(policy, budget, seed):
    rng = np.random.default_rng(seed); hits = spent = 0
    alpha = np.ones(n_b); beta = np.ones(n_b); last = None; last_hit = False
    mem = [rng.permutation(m) for m in members]; ptr = np.zeros(n_b, int); remaining = np.array([len(m) for m in mem])
    if policy == "uniform": order = rng.permutation(len(y))
    elif policy == "static": order = np.argsort(-sim.static.values)
    elif policy == "oracle": order = np.argsort(-(y + 1e-6 * rng.random(len(y))))
    while spent < budget:
        take = min(K, budget - spent)
        if policy in ("uniform", "static", "oracle"):
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

out = {}
for bf in BUDGET_FRACS:
    budget = int(bf * len(sim)); row = {}
    for pol in ["uniform", "static", "ts_flat", "ts_static", "neighbor", "oracle"]:
        r = [run(pol, budget, s) for s in range(SEEDS)]; row[pol] = {"mean": float(np.mean(r)), "sd": float(np.std(r))}
    out[f"budget_{bf}"] = row; print(bf, json.dumps(row), flush=True)
json.dump(out, open(os.path.join(ROOT, "runs", "code", f"sim_{LABEL}.json"), "w"), indent=1)
print(f"\nlabel={LABEL} | budget | uniform | static triage | TS flat | TS + static prior | neighbor | oracle |\n|---|---|---|---|---|---|---|")
for k, row in out.items():
    u = row["uniform"]["mean"]; print(f"| {k} | {u:.0f} | " + " | ".join(f"{row[p]['mean']:.0f} ({row[p]['mean']/max(u,1):.2f}x)" for p in ["static", "ts_flat", "ts_static", "neighbor", "oracle"]) + " |")
