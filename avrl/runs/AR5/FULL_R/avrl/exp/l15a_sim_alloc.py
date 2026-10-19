"""L15a: copy of exp/l13a_sim_alloc.py. Only differences: argv[10] STALE in {none, flip25, flip50, lowlen} corrupts the n = 200 free fit-half labels (and only those), identically for recal_free_200 and warm_online_200 (both read them through free_draw(), so the two stay paired per policy seed): none = L13a unchanged, flip25 / flip50 = each free label flipped independently with probability 0.25 / 0.5 using rng [policy seed, 15] on the labels (never the features), lowlen = the 200 free rows are drawn (same rng [policy seed, 12]) from the half of the fit rows with the lowest log_qlen (labels unchanged). New arm warm_online_200_w25 = warm_online_200 whose refits give the stale free labels sample_weight 0.25 (stream labels weight 1; the initial pre-refit fit has no stream labels and is left unweighted, so the arm equals warm_online_200 until its first refit); the weight is passed through the pipeline step name (logisticregression__sample_weight). Only the arms uniform, static, greedy_online, recal_free_200, warm_online_200, warm_online_200_w25 are run (rollout mode). argv[9] FLAG: "" / "refit" = refits on, "norefit" = identity-gate mode (no refits: warm_online_200 and its w25 twin must equal recal_free_200 per policy seed).
L13a: copy of exp/l12c_sim_alloc.py. Only differences: new policies warm_online_<n> for n in WARM_NS (100, 200, 400; skipped when argv[9] == "nowarm", the regression-gate mode that reproduces L12c exactly; argv[9] == "norefit" disables the refits, the identity-gate mode in which warm_online_n must equal recal_free_n per policy seed): the greedy online learner (no layout, K = 1) whose initial score is the recal_free_<n> logistic (same fit-half draw rng [seed, 12], same fit, so each warm arm is paired with recal_free_n per policy seed), refit every R = max(B // 8, 40) labels on the n free fit-half labels pooled with all seen sim labels (unweighted, same StandardScaler + balanced LogisticRegression, max_iter 500). Hits are counted on picks only. A one-class free draw falls back to the cold learner random score. Each arm builds its own rng from the policy seed.
L12c: copy of exp/l11a_sim_alloc.py. Only differences: new policies recal_free_<n> for n in NCALS (no budget filter; skipped when argv[9] == "nofree", which is the regression-gate mode that reproduces L11a exactly): draw n rows of the FIT half with a seeded rng (nested draws per policy seed), fit the same class-balanced logistic, rank the whole sim, spend the full budget B on the top-ranked picks (no labels charged, none counted as hits); a draw with 0 or n positives falls back to a random order, as recal_charged does.
L11a: copy of exp/l1a_sim_alloc.py. Only differences: argv[7] = target n_sim rows (0 = keep the whole sim; else whole pids of the sim half are drawn at random, rng seeded from argv[8] and the split seed, until the row count is closest to the target, so the fit set, the static classifier and static_matched are unchanged), argv[8] = subsample seed, and NCALS = (25, 50, 100, 200, 400) in rollout mode too (still filtered by n <= budget // 2).
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
FLAG = sys.argv[9] if len(sys.argv) > 9 else ""; assert FLAG in ("", "refit", "nowarm", "norefit"), FLAG
STALE = sys.argv[10] if len(sys.argv) > 10 else "none"; assert STALE in ("none", "flip25", "flip50", "lowlen"), STALE
STALE_P = {"none": 0.0, "flip25": 0.25, "flip50": 0.5, "lowlen": 0.0}; W_STALE = 0.25
NOWARM, NOREFIT = FLAG == "nowarm", FLAG == "norefit"
FORBIDDEN = {"agreement", "n_distinct", "pass_rate", "majority", "majority_correct", "proxy", "proxy_error", "mean_tokens", "std_tokens", "any_truncated",
             "judge_top", "judge_margin", "judge_mean", "judge_std", "judge_sel", "judge_error"}
df = pd.read_parquet(path)
ROLL = "judge_logit" in df.columns and "agreement" not in df.columns
if ROLL: assert not (set(df.columns) & FORBIDDEN), set(df.columns) & FORBIDDEN; assert not (set(cheap) & FORBIDDEN)
NCALS = (25, 50, 100, 200, 400); FREE_NS = (200,); WARM_NS = (200,)
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
LOWPOOL = np.argsort(fit["log_qlen"].values, kind="stable")[: len(fit) // 2] if STALE == "lowlen" else None

def free_draw(seed, n_cal):
    """L15a: positions (into fit) and labels of the n_cal free stale rows of a policy seed. Same draw rng [seed, 12] as L12c / L13a; lowlen draws from the lowest-log_qlen half of the fit rows; flipNN flips each label with probability STALE_P using rng [seed, 15]."""
    rf = np.random.default_rng([seed, 12])
    cal = rf.permutation(len(fit))[:n_cal] if LOWPOOL is None else LOWPOOL[rf.permutation(len(LOWPOOL))[:n_cal]]
    yc = fit[LABEL].values[cal].astype(int)
    if STALE_P[STALE] > 0: yc = np.where(np.random.default_rng([seed, 15]).random(n_cal) < STALE_P[STALE], 1 - yc, yc)
    return cal, yc
y = sim[LABEL].values.astype(int); codes, uniq = pd.factorize(sim[G].values); n_b = len(uniq)
members = [np.where(codes == b)[0] for b in range(n_b)]
prior = pd.Series(sim.static.values).groupby(codes).mean().values; prior = prior / prior.max()
print(f"sim docs {len(sim)} buckets {n_b} base {y.mean():.4f} hits {int(y.sum())} median bucket {int(np.median([len(m) for m in members]))}", flush=True)

def run(policy, budget, seed):
    rng = np.random.default_rng(seed); hits = spent = 0
    alpha = np.ones(n_b); beta = np.ones(n_b); last = None; last_hit = False
    mem = None if policy in ORDER_POLS or policy.startswith("recal_charged_") or policy.startswith("recal_free_") else ([rng.permutation(m) for m in members] if policy == "ts_flat" else [m[np.argsort(-sim.static.values[m])] for m in members])
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
    elif policy.startswith("recal_free_"):  # L12c: n_cal rows of the fit half (NOT charged, no hits counted) fit the same logistic rule, the whole budget goes to the top-ranked sim rows
        n_cal = int(policy.rsplit("_", 1)[1]); cal, yc = free_draw(seed, n_cal)
        if yc.sum() == 0 or yc.sum() == n_cal: order = rng.permutation(len(y))
        else: order = np.argsort(-make_pipeline(StandardScaler(), LogisticRegression(max_iter=1000, class_weight="balanced")).fit(X(fit)[cal], yc).predict_proba(Xs)[:, 1])
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
        if policy in ORDER_POLS or policy.startswith("recal_charged_") or policy.startswith("recal_free_"):
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

def run_warm(budget, seed, n_cal, refit=True, w_stale=1.0):
    """L13a warm_online_<n>: greedy online learner (no layout, K = 1) warm-started with the recal_free_<n> logistic. Between refits the picks are the unseen rows in descending score order (the same full-population argsort recal_free uses, so with refits disabled the picks equal recal_free_<n> even under exactly tied scores); refit every R labels on the n free fit-half labels pooled with the seen sim labels."""
    rng = np.random.default_rng(seed); R = max(budget // 8, 40) if refit else 10 ** 9
    cal, yc = free_draw(seed, n_cal); Xc = X(fit)[cal]
    score = rng.random(len(y)) * 1e-3  # cold-learner fallback when the free draw is one-class
    if 0 < yc.sum() < n_cal: score = make_pipeline(StandardScaler(), LogisticRegression(max_iter=1000, class_weight="balanced")).fit(Xc, yc).predict_proba(Xs)[:, 1]
    else: ONECLASS.setdefault(str(n_cal), []).append(seed)
    seen = np.zeros(len(y), bool); order = np.argsort(-score); ptr = 0; hits = spent = 0; next_fit = R
    while spent < budget:
        while seen[order[ptr]]: ptr += 1
        c = order[ptr]; seen[c] = True; ptr += 1; hits += int(y[c]); spent += 1
        if spent >= next_fit and spent < budget:
            yp = np.concatenate([yc, y[seen]])
            if 0 < yp.sum() < len(yp):
                fkw = {} if w_stale == 1.0 else {"logisticregression__sample_weight": np.concatenate([np.full(n_cal, w_stale), np.ones(int(seen.sum()))])}  # L15a w25: stale free labels down-weighted in every refit
                score = make_pipeline(StandardScaler(), LogisticRegression(max_iter=500, class_weight="balanced")).fit(np.vstack([Xc, Xs[seen]]), yp, **fkw).predict_proba(Xs)[:, 1]
                order = np.argsort(-score); ptr = 0; next_fit += R
    return hits

ONECLASS = {}
if ROLL: POLS = ["uniform", "static", "greedy_online"]
else: POLS = ["uniform", "static", "static_matched", "ts_flat", "greedy_online", "ts_online", "ts_static", "neighbor", "oracle", "thr_agreement"] + (["thr_margin", "thr_margin_split", "thr_agree_margin"] if "judge_margin" in sim else [])
out = {"mode": "rollout" if ROLL else "group", "n_fit": int(len(fit)), "n_fit_pos": int(fit[LABEL].sum()), "n_sim": int(len(sim)), "n_buckets": n_b, "hits": int(y.sum()), "split_seed": SPLIT_SEED, "label": LABEL, "group_col": G, "features": cheap, "n_policy_seeds": SEEDS, "target_n_sim": TARGET_N, "subsample_seed": SUB_SEED, "n_sim_before_subsample": N_SIM_FULL, "n_sim_pids": int(sim["pid"].nunique()) if "pid" in sim.columns else None, "free_ns": [n for n in FREE_NS if n <= len(fit)], "warm_ns": [] if NOWARM else [n for n in WARM_NS if n <= len(fit)], "warm_refit": not NOREFIT, "warm_oneclass_draws": {}, "stale": STALE, "stale_flip_p": STALE_P[STALE], "w_stale": W_STALE, "n_lowlen_pool": None if LOWPOOL is None else int(len(LOWPOOL))}
for bf in BUDGET_FRACS:
    budget = int(bf * len(sim)); row = {}
    for pol in POLS + [f"recal_free_{n}" for n in out["free_ns"]] + [f"warm_online_{n}" for n in out["warm_ns"]] + [f"warm_online_{n}_w25" for n in out["warm_ns"]]:
        if pol.startswith("warm_online_"): r = [run_warm(budget, s, int(pol.split("_")[2]), refit=not NOREFIT, w_stale=W_STALE if pol.endswith("_w25") else 1.0) for s in range(SEEDS)]
        else: r = [run_online(budget, s, greedy=(pol == "greedy_online")) if pol in ("ts_online", "greedy_online") else run(pol, budget, s) for s in range(SEEDS)]
        row[pol] = {"mean": float(np.mean(r)), "sd": float(np.std(r)), "per_seed": [int(v) for v in r]}
    out[f"budget_{bf}"] = row
out["warm_oneclass_draws"] = {k: sorted(set(v)) for k, v in ONECLASS.items()}
if out_path: json.dump(out, open(out_path, "w"), indent=1)
print(f"\n{LABEL} by {G} | mode {out['mode']} | split {SPLIT_SEED} | stale {STALE} | mean errors caught (sd) per budget")
for bf in BUDGET_FRACS:
    row = out[f"budget_{bf}"]; u = row["uniform"]["mean"]
    print(f"| {bf:.0%} | " + " | ".join(f"{p} {row[p]['mean']:.1f}±{row[p]['sd']:.1f} ({row[p]['mean']/max(u,1):.2f}x)" for p in row), flush=True)
