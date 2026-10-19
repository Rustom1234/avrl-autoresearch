"""L1a warm-start replay. Copy of sim_e21.py; the only change to the allocation logic is a `warm` flag in run_online.
Usage: sim_warm.py <feats.parquet> <label> <group_col> <cheap_col1,cheap_col2,...> [out.json]
Policies: uniform, agree_thr (label-free agreement threshold), greedy_warm_norefit (sanity: must equal agree_thr), greedy_warm, ts_warm
(start from the agreement ordering, then refit on audited labels), greedy_online, ts_online (cold, as E21), oracle.
Env BOOT_B=<n>: instead of the main table, run n pool bootstraps (resample the simulated pool with replacement, BOOT_SEEDS seeds each,
25% budget only; agree_thr, greedy_warm, ts_warm) and write out.json."""
import os, sys, json
import numpy as np, pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
MIN, K, SEEDS, BUDGET_FRACS = 3, 1, 8, (0.10, 0.25, 0.50)  # identical to sim_e21.py
path, LABEL, G, cheap = sys.argv[1], sys.argv[2], sys.argv[3], sys.argv[4].split(",")
out_path = sys.argv[5] if len(sys.argv) > 5 else None
df = pd.read_parquet(path)
cnt = df[G].value_counts(); df = df[df[G].isin(cnt[cnt >= MIN].index)].reset_index(drop=True)
rng0 = np.random.default_rng(0); groups = df[G].unique(); rng0.shuffle(groups)
fit_g = set(groups[: len(groups) // 4]); sim0 = df[~df[G].isin(fit_g)].reset_index(drop=True)  # same held-out-cells split as sim_e21.py

def set_pool(d):
    global sim, Xs, y, codes, n_b, members
    sim = d.reset_index(drop=True); Xs = sim[cheap].fillna(0).values; y = sim[LABEL].values.astype(int)
    codes, uniq = pd.factorize(sim[G].values); n_b = len(uniq); members = [np.where(codes == b)[0] for b in range(n_b)]
set_pool(sim0)
print(f"sim docs {len(sim)} buckets {n_b} base {y.mean():.4f} hits {int(y.sum())} median bucket {int(np.median([len(m) for m in members]))}", flush=True)

def run(policy, budget, seed):
    rng = np.random.default_rng(seed)
    if policy == "uniform": order = rng.permutation(len(y))
    elif policy == "agree_thr":  # label-free agreement threshold: agreement ascending, n_distinct descending, random tie-break
        order = np.lexsort((rng.random(len(y)), -sim.n_distinct.values, sim.agreement.values))
    elif policy == "oracle": order = np.argsort(-(y + 1e-6 * rng.random(len(y))))
    return int(y[order[:budget]].sum())  # K=1 audits in order, so the first `budget` entries are exactly what the sim_e21 loop would take

def bucket_prior(score):
    bp = pd.Series(score).groupby(codes).mean().values; return bp / bp.max()

def run_online(budget, seed, R=None, greedy=False, warm=False):
    """ts_online / greedy_online as in sim_e21.py. warm=True: score starts as the agree_thr ordering converted to a rank score in (0,1]
    (top-scored unseen doc = the one the threshold would audit next); the TS variant also starts bprior from the cell means of that score."""
    rng = np.random.default_rng(seed); R = R or max(budget // 8, 40)
    seen = np.zeros(len(y), bool)
    if warm:
        order = np.lexsort((rng.random(len(y)), -sim.n_distinct.values, sim.agreement.values))  # same first rng draw as agree_thr
        score = np.empty(len(y)); score[order] = (len(y) - np.arange(len(y))) / len(y); bprior = bucket_prior(score)
    else:
        score = rng.random(len(y)) * 1e-3; bprior = None
    alpha = np.ones(n_b); beta = np.ones(n_b); remaining = np.array([len(m) for m in members]); hits = spent = 0; next_fit = R
    while spent < budget:
        take = min(K, budget - spent); avail = remaining > 0
        if greedy:  # no layout: inspect the top-scored unseen docs anywhere
            mu = np.where(~seen)[0]; c = mu[np.argsort(-score[mu])][:take]; seen[c] = True; hits += int(y[c].sum()); spent += len(c)
        else:
            s = rng.beta(alpha + 10 * bprior, beta + 10 * (1 - bprior)) if bprior is not None else rng.beta(alpha, beta)
            s[~avail] = -1; b = int(np.argmax(s)); mu = members[b][~seen[members[b]]]; c = mu[np.argsort(-score[mu])][:take]
            seen[c] = True; remaining[b] -= len(c); h = int(y[c].sum()); alpha[b] += h; beta[b] += len(c) - h; hits += h; spent += len(c)
        if spent >= next_fit and 0 < y[seen].sum() < seen.sum():
            score = make_pipeline(StandardScaler(), LogisticRegression(max_iter=500, class_weight="balanced")).fit(Xs[seen], y[seen]).predict_proba(Xs)[:, 1]
            bprior = bucket_prior(score); next_fit += R
    return hits

POL = {"uniform": lambda b, s: run("uniform", b, s), "agree_thr": lambda b, s: run("agree_thr", b, s), "oracle": lambda b, s: run("oracle", b, s),
       "greedy_warm_norefit": lambda b, s: run_online(b, s, R=10**9, greedy=True, warm=True),  # sanity: refit never fires, must equal agree_thr per seed
       "greedy_warm": lambda b, s: run_online(b, s, greedy=True, warm=True), "ts_warm": lambda b, s: run_online(b, s, warm=True),
       "greedy_online": lambda b, s: run_online(b, s, greedy=True), "ts_online": lambda b, s: run_online(b, s)}

if os.environ.get("BOOT_B"):
    B, NS = int(os.environ["BOOT_B"]), int(os.environ.get("BOOT_SEEDS", 2)); brng = np.random.default_rng(12345); rows = []
    for i in range(B):
        set_pool(sim0.iloc[brng.integers(0, len(sim0), len(sim0))]); budget = int(0.25 * len(sim)); r = {"boot": i, "n_err": int(y.sum()), "budget": budget}
        for pol in ("agree_thr", "greedy_warm", "ts_warm"): r[pol] = [POL[pol](budget, s) for s in range(NS)]
        rows.append(r); print(i, r, flush=True)
    if out_path: json.dump({"n_sim": len(sim0), "budget_frac": 0.25, "seeds_per_resample": NS, "resamples": rows}, open(out_path, "w"), indent=1)
    sys.exit(0)

out = {"n_sim": int(len(sim)), "n_buckets": n_b, "hits": int(y.sum()), "paired": {}}
for bf in BUDGET_FRACS:
    budget = int(bf * len(sim)); row = {}
    for pol in POL:
        r = [POL[pol](budget, s) for s in range(SEEDS)]; row[pol] = {"mean": float(np.mean(r)), "sd": float(np.std(r)), "seeds": [int(x) for x in r]}
    out[f"budget_{bf}"] = row; out["budget_n"] = out.get("budget_n", {}); out["budget_n"][str(bf)] = budget
    thr = np.array(row["agree_thr"]["seeds"])
    out["paired"][f"budget_{bf}"] = {f"{p}_minus_agree_thr": {"seeds": [int(x) for x in np.array(row[p]["seeds"]) - thr], "mean": float((np.array(row[p]["seeds"]) - thr).mean()),
                                                           "sd": float((np.array(row[p]["seeds"]) - thr).std(ddof=1))} for p in ("greedy_warm", "ts_warm")}
if out_path: json.dump(out, open(out_path, "w"), indent=1)
print(f"\n{LABEL} by {G} | budget | uniform | agree_thr | greedy_warm | ts_warm | greedy_online (cold) | ts_online (cold) | oracle | greedy_warm-thr (paired SD) | ts_warm-thr (paired SD) |\n|---|---|---|---|---|---|---|---|---|---|")
for bf in BUDGET_FRACS:
    row = out[f"budget_{bf}"]; pr = out["paired"][f"budget_{bf}"]
    print(f"| {bf:.0%} | {row['uniform']['mean']:.0f} | " + " | ".join(f"{row[p]['mean']:.1f}±{row[p]['sd']:.1f}" for p in ["agree_thr", "greedy_warm", "ts_warm", "greedy_online", "ts_online", "oracle"])
          + " | " + " | ".join(f"{pr[k]['mean']:+.1f} ({pr[k]['sd']:.1f})" for k in ("greedy_warm_minus_agree_thr", "ts_warm_minus_agree_thr")) + " |")
