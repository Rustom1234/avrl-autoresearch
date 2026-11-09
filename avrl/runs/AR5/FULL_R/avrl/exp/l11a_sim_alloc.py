"""L11a: copy of exp/l1a_sim_alloc.py. Only differences: argv[7] = target n_sim rows (0 = keep the whole sim; else whole pids of the sim half are drawn at random, rng seeded from argv[8] and the split seed, until the row count is closest to the target, so the fit set, the static classifier and static_matched are unchanged), argv[8] = subsample seed, and NCALS = (25, 50, 100, 200, 400) in rollout mode too (still filtered by n <= budget // 2).
L1a: copy of exp/l1b_sim_alloc.py. Group mode (agreement column present) is unchanged apart from skipping an unused per-run member sort, so it reproduces runs/E04 exactly (regression gate).
ROLLOUT mode (judge_logit present, agreement absent: each row is one rollout; the table must contain no group aggregate) uses its own policy list: uniform, static (free labels, upper bound only), static_matched,
recal_charged_100/200/400, greedy_online (no layout), ts_online (bucket layout), oracle, and the label-free threshold family thr_judge_low (judge_p ascending), thr_boundary (boundary ascending),
thr_tokens_long (tokens descending), thr_trunc_first (truncated first, then tokens descending), all with random tie-breaks. Buckets are whole groups of the group column (cell100/cell30/pid), so the rollouts of a
prompt never straddle the fit/sim split when the group column is a cell or the pid.
Inherited from L1b: copy of runs/E04/sim_alloc.py with (a) a split-seed argument, (b) recal_charged_<n> policies (calibration labels charged to the budget), (c) per-seed hits for all policies.
Usage: l1b_sim_alloc.py <feats.parquet> <label> <group_col> <cheap_col1,cheap_col2,...> [out.json] [split_seed]
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
SPLIT_SEED = int(sys.argv[6]) if len(sys.argv) > 6 else 0
TARGET_N = int(sys.argv[7]) if len(sys.argv) > 7 else 0
SUB_SEED = int(sys.argv[8]) if len(sys.argv) > 8 else 0
FORBIDDEN = {"agreement", "n_distinct", "pass_rate", "majority", "majority_correct", "proxy", "proxy_error", "mean_tokens", "std_tokens", "any_truncated",
             "judge_top", "judge_margin", "judge_mean", "judge_std", "judge_sel", "judge_error"}
df = pd.read_parquet(path)
ROLL = "judge_logit" in df.columns and "agreement" not in df.columns
if ROLL: assert not (set(df.columns) & FORBIDDEN), set(df.columns) & FORBIDDEN; assert not (set(cheap) & FORBIDDEN)
NCALS = (25, 50, 100, 200, 400)
ORDER_POLS = {"uniform", "static", "static_matched", "oracle", "thr_agreement", "thr_margin", "thr_margin_split", "thr_agree_margin", "thr_judge_low", "thr_boundary", "thr_tokens_long", "thr_trunc_first"}
cnt = df[G].value_counts(); df = df[df[G].isin(cnt[cnt >= MIN].index)].reset_index(drop=True)
rng0 = np.random.default_rng(SPLIT_SEED); groups = df[G].unique(); rng0.shuffle(groups)
fit_g = set(groups[: len(groups) // 4]); fit = df[df[G].isin(fit_g)]; sim = df[~df[G].isin(fit_g)].reset_index(drop=True)
N_SIM_FULL = len(sim)
if TARGET_N > 0:  # L11a: keep whole pids (all rollouts of a prompt together), drawn at random until the row count is closest to the target
    assert ROLL and "pid" in sim.columns and sim.groupby("pid")[G].nunique().max() == 1
    pids = sim["pid"].unique(); np.random.default_rng([SUB_SEED, SPLIT_SEED, 11]).shuffle(pids)
    cum = np.cumsum(sim.groupby("pid").size().reindex(pids).values); keep = pids[: int(np.argmin(np.abs(cum - TARGET_N))) + 1]
    sim = sim[sim["pid"].isin(set(keep))].reset_index(drop=True)
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
    mem = None if policy in ORDER_POLS or policy.startswith("recal_charged_") else ([rng.permutation(m) for m in members] if policy == "ts_flat" else [m[np.argsort(-sim.static.values[m])] for m in members])
    ptr = np.zeros(n_b, int); remaining = np.array([len(m) for m in members])
    if policy == "uniform": order = rng.permutation(len(y))
    elif policy == "static_matched":  # fair static: spend half the budget on random labels, fit, rank the rest
        perm = rng.permutation(len(y)); tr = perm[: max(budget // 2, 1)]
        if y[tr].sum() == 0: order = perm
        else:
            sc = make_pipeline(StandardScaler(), LogisticRegression(max_iter=1000, class_weight="balanced")).fit(Xs[tr], y[tr]).predict_proba(Xs)[:, 1]
            sc[tr] = -1; order = np.concatenate([tr, np.argsort(-sc)])
    elif policy.startswith("recal_charged_"):  # fixed multi-feature rule: n_cal random verified groups (charged, hits counted) fit the logistic rule, rank the rest
        n_cal = int(policy.rsplit("_", 1)[1]); perm = rng.permutation(len(y)); cal = perm[:n_cal]
        if y[cal].sum() == 0 or y[cal].sum() == n_cal: order = perm
        else:
            sc = make_pipeline(StandardScaler(), LogisticRegression(max_iter=1000, class_weight="balanced")).fit(Xs[cal], y[cal]).predict_proba(Xs)[:, 1]
            sc[cal] = -1; order = np.concatenate([cal, np.argsort(-sc)[: len(y) - n_cal]])
    elif policy == "static": order = np.argsort(-sim.static.values)
    elif policy == "oracle": order = np.argsort(-(y + 1e-6 * rng.random(len(y))))
    elif policy == "thr_judge_low": order = np.lexsort((rng.random(len(y)), sim["judge_p"].values))  # lowest judge score first, random ties, no labels
    elif policy == "thr_boundary": order = np.lexsort((rng.random(len(y)), sim["boundary"].values))  # closest to the accept boundary first, random ties, no labels
    elif policy == "thr_tokens_long": order = np.lexsort((rng.random(len(y)), -sim["tokens"].values))  # longest completion first, random ties, no labels
    elif policy == "thr_trunc_first": order = np.lexsort((rng.random(len(y)), -sim["tokens"].values, -sim["truncated"].values))  # truncated first, then longest, random ties, no labels
    elif policy == "thr_agreement": order = np.lexsort((rng.random(len(y)), sim["agreement"].values))  # low agreement first, random ties, no labels
    elif policy == "thr_margin": order = np.lexsort((rng.random(len(y)), sim["judge_margin"].values))  # low judge margin first, random ties, no labels
    elif policy == "thr_margin_split": order = np.lexsort((rng.random(len(y)), np.where(sim["n_distinct"].values == 1, 1.0, sim["judge_margin"].values)))  # margin, unanimous groups last (robustness)
    elif policy == "thr_agree_margin": order = np.lexsort((rng.random(len(y)), np.where(sim["n_distinct"].values == 1, 1.0, sim["judge_margin"].values), sim["agreement"].values))  # agreement, ties by margin (robustness)
    while spent < budget:
        take = min(K, budget - spent)
        if policy in ORDER_POLS or policy.startswith("recal_charged_"):
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

if ROLL: POLS = ["uniform", "static", "static_matched", "greedy_online", "ts_online", "oracle", "thr_judge_low", "thr_boundary", "thr_tokens_long", "thr_trunc_first"]
else: POLS = ["uniform", "static", "static_matched", "ts_flat", "greedy_online", "ts_online", "ts_static", "neighbor", "oracle", "thr_agreement"] + (["thr_margin", "thr_margin_split", "thr_agree_margin"] if "judge_margin" in sim else [])
out = {"mode": "rollout" if ROLL else "group", "n_fit": int(len(fit)), "n_fit_pos": int(fit[LABEL].sum()), "n_sim": int(len(sim)), "n_buckets": n_b, "hits": int(y.sum()), "split_seed": SPLIT_SEED, "label": LABEL, "group_col": G, "features": cheap, "n_policy_seeds": SEEDS, "target_n_sim": TARGET_N, "subsample_seed": SUB_SEED, "n_sim_before_subsample": N_SIM_FULL, "n_sim_pids": int(sim["pid"].nunique()) if "pid" in sim.columns else None}
for bf in BUDGET_FRACS:
    budget = int(bf * len(sim)); row = {}
    for pol in POLS + [f"recal_charged_{n}" for n in NCALS if n <= budget // 2]:
        r = [run_online(budget, s, greedy=(pol == "greedy_online")) if pol in ("ts_online", "greedy_online") else run(pol, budget, s) for s in range(SEEDS)]; row[pol] = {"mean": float(np.mean(r)), "sd": float(np.std(r)), "per_seed": [int(v) for v in r]}
    out[f"budget_{bf}"] = row
if out_path: json.dump(out, open(out_path, "w"), indent=1)
print(f"\n{LABEL} by {G} | mode {out['mode']} | split {SPLIT_SEED} | mean errors caught (sd) per budget")
for bf in BUDGET_FRACS:
    row = out[f"budget_{bf}"]; u = row["uniform"]["mean"]
    print(f"| {bf:.0%} | " + " | ".join(f"{p} {row[p]['mean']:.1f}±{row[p]['sd']:.1f} ({row[p]['mean']/max(u,1):.2f}x)" for p in row), flush=True)
