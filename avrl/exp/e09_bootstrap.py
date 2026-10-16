"""L9a: group-level bootstrap of the allocation replay for E09 (HelpSteer2, reward-model proxy vs
7B judge gold), the first point where the cheap signal is weak (cheap AUC 0.73, vs 0.83-0.91 on
E01/E08). Copied-and-modified from avrl/exp/e03b_bootstrap_matched.py per the project rule (copy and
modify, never edit prior experiment scripts in place); policy logic (static, static_matched,
greedy_online, ts_online) is sim_generic.py's run()/run_online() verbatim, K=1 (group-level budgets,
matches e03b's convention since feats.parquet here is one row per task-group, not per-doc).

No new rollouts, no GPU. Reuses avrl/runs/E09/feats.parquet (1,500 groups, label proxy_error).

Unlike e03b (one group_col, one budget, one feature set, static_matched vs ts_online only), this
script sweeps:
  - group_col in {cell100, cell30}
  - feature set in {cheap (9 cols), cheap+pc (9 cheap + pc0..pc15)}
  - budget_frac in {0.10, 0.25, 0.50}
  - comparisons: greedy_online vs static_matched, ts_online vs static_matched,
    greedy_online vs static (free labels)

For each (group_col, feature_set, budget) combo, the held-out-quarter split (fit_b/sim) is computed
once per (resample, group_col) and shared across feature sets and budgets; "static" (free-label
classifier, deterministic, fit on fit_b) is computed once per (resample, group_col, feature_set) and
shared across budgets (only the slice changes); static_matched spends half of THAT budget's own
labels on random draws before ranking the rest, verbatim from sim_generic.py's static_matched branch,
no labels beyond the audit budget. All four policies (static, static_matched, greedy_online,
ts_online) are evaluated against the same `sim` population and the same `budget_n = int(budget_frac *
len(sim))` within a resample/group_col/budget, so budget accounting is identical across policies.
static_matched/greedy_online/ts_online are averaged over 8 policy seeds (SEEDS, matches
p0/E03/e03b's convention); static has no seed (sd=0 by construction).

Bootstrap: resample the 1,500 groups with replacement (group-level, matches e03b), recompute the
full pipeline per resample. For each comparison, per resample: diff = hits_a - hits_b, rel = diff /
hits_b (skipped for that resample if hits_b == 0). 95% CI = 2.5/97.5 percentiles of the resample
distributions of diff and of rel.

Usage:
  e09_bootstrap.py [--feats F] [--n_boot N] [--seed S] [--out OUT]
"""
import argparse, json, os, time
import numpy as np, pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline

MIN, K, SEEDS = 3, 1, 8
LABEL = "proxy_error"
CHEAP = ["rm_max", "rm_mean", "rm_std", "rm_gap12", "mean_tokens", "std_tokens", "trunc_frac", "consistency", "log_qlen"]
PCS = [f"pc{i}" for i in range(16)]
FEATURE_SETS = {"cheap": CHEAP, "cheap_pc": CHEAP + PCS}
GROUP_COLS = ["cell100", "cell30"]
BUDGET_FRACS = (0.10, 0.25, 0.50)
COMPARISONS = [("greedy_online", "static_matched"), ("ts_online", "static_matched"), ("greedy_online", "static")]


def clf():
    return make_pipeline(StandardScaler(), LogisticRegression(max_iter=1000, class_weight="balanced"))


def run_online(Xs, y, codes, n_b, members, budget, seed, greedy):
    """sim_generic.py's run_online() verbatim (K=1): Thompson-over-buckets (or greedy, no layout),
    self-trained ranking + bucket prior refit every R verified labels."""
    rng = np.random.default_rng(seed)
    R = max(budget // 8, 40)
    seen = np.zeros(len(y), bool); score = rng.random(len(y)) * 1e-3; bprior = None
    alpha = np.ones(n_b); beta = np.ones(n_b); remaining = np.array([len(m) for m in members])
    hits = spent = 0; next_fit = R
    while spent < budget:
        take = min(K, budget - spent); avail = remaining > 0
        if greedy:
            mu = np.where(~seen)[0]
            if len(mu) == 0:
                break
            c = mu[np.argsort(-score[mu])][:take]
            seen[c] = True; hits += int(y[c].sum()); spent += len(c)
        else:
            if not avail.any():
                break
            s = rng.beta(alpha + 10 * bprior, beta + 10 * (1 - bprior)) if bprior is not None else rng.beta(alpha, beta)
            s[~avail] = -1; b = int(np.argmax(s)); mu = members[b][~seen[members[b]]]
            c = mu[np.argsort(-score[mu])][:take]
            seen[c] = True; remaining[b] -= len(c); h = int(y[c].sum()); alpha[b] += h; beta[b] += len(c) - h
            hits += h; spent += len(c)
        if spent >= next_fit and 0 < y[seen].sum() < seen.sum():
            score = clf().fit(Xs[seen], y[seen]).predict_proba(Xs)[:, 1]
            bp = pd.Series(score).groupby(codes).mean().values; bprior = bp / bp.max(); next_fit += R
    return hits


def split_sim(df, group_col):
    cnt = df[group_col].value_counts()
    df = df[df[group_col].isin(cnt[cnt >= MIN].index)].reset_index(drop=True)
    if len(df) == 0:
        return None
    rng0 = np.random.default_rng(0)
    buckets = df[group_col].unique(); rng0.shuffle(buckets)
    fit_b = set(buckets[: len(buckets) // 4])
    fit = df[df[group_col].isin(fit_b)]
    sim = df[~df[group_col].isin(fit_b)].reset_index(drop=True)
    if len(sim) == 0 or len(fit) == 0:
        return None
    return fit, sim


def compute_group_col(df, group_col):
    """Returns {feature_set: {budget_frac: {policy: hits, 'total_label1_in_sim':.., 'n_sim':..}}} or None."""
    split = split_sim(df, group_col)
    if split is None:
        return None
    fit, sim = split
    out = {}
    for fs_name, feats in FEATURE_SETS.items():
        Xs = sim[feats].fillna(0).values
        y = sim[LABEL].values.astype(int)
        codes, uniq = pd.factorize(sim[group_col].values)
        n_b = len(uniq)
        if n_b == 0 or y.sum() == 0:
            continue
        members = [np.where(codes == b)[0] for b in range(n_b)]

        Xf = fit[feats].fillna(0).values
        yf = fit[LABEL].values.astype(int)
        if 0 < yf.sum() < len(yf):
            static_prob = clf().fit(Xf, yf).predict_proba(Xs)[:, 1]
            static_order = np.argsort(-static_prob)
        else:
            static_order = np.arange(len(sim))  # degenerate free-label fit set: no signal, natural order

        budgets_out = {}
        for bf in BUDGET_FRACS:
            budget_n = int(bf * len(sim))
            if budget_n <= 0:
                continue
            hits_static = int(y[static_order[:budget_n]].sum())

            hits_sm = []
            for s in range(SEEDS):
                rng = np.random.default_rng(s)
                perm = rng.permutation(len(y)); tr = perm[: max(budget_n // 2, 1)]
                if y[tr].sum() == 0:
                    order = perm
                else:
                    sc = clf().fit(Xs[tr], y[tr]).predict_proba(Xs)[:, 1]
                    sc[tr] = -1
                    order = np.concatenate([tr, np.argsort(-sc)])
                hits_sm.append(int(y[order[:budget_n]].sum()))
            hits_static_matched = float(np.mean(hits_sm))

            hits_greedy = float(np.mean([run_online(Xs, y, codes, n_b, members, budget_n, s, True) for s in range(SEEDS)]))
            hits_ts = float(np.mean([run_online(Xs, y, codes, n_b, members, budget_n, s, False) for s in range(SEEDS)]))

            budgets_out[bf] = {
                "static": float(hits_static),
                "static_matched": hits_static_matched,
                "greedy_online": hits_greedy,
                "ts_online": hits_ts,
                "total_label1_in_sim": int(y.sum()),
                "n_sim": len(sim),
            }
        if budgets_out:
            out[fs_name] = budgets_out
    return out if out else None


def compute(df):
    """Full sweep over group_col for one dataframe (original or bootstrap-resampled)."""
    return {gc: compute_group_col(df, gc) for gc in GROUP_COLS}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--feats", default="/homes/rdubash/active-audit/avrl/runs/E09/feats.parquet")
    ap.add_argument("--n_boot", type=int, default=300)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default="/homes/rdubash/active-audit/avrl/runs/E09/bootstrap.json")
    args = ap.parse_args()

    t0 = time.time()
    G = pd.read_parquet(args.feats)
    n_groups = len(G)

    point = compute(G)

    rng = np.random.default_rng(args.seed)
    # collectors[(group_col, fs_name, budget_frac, a, b)] = {"diffs": [...], "rels": [...]}
    collectors = {}
    for gc in GROUP_COLS:
        pgc = point.get(gc)
        if pgc is None:
            continue
        for fs_name in pgc:
            for bf in pgc[fs_name]:
                for a, b in COMPARISONS:
                    collectors[(gc, fs_name, bf, a, b)] = {"diffs": [], "rels": []}

    n_boot_used = 0
    for i in range(args.n_boot):
        idx = rng.integers(0, n_groups, n_groups)
        dfb = G.iloc[idx].reset_index(drop=True)
        res = compute(dfb)
        n_boot_used += 1
        for gc in GROUP_COLS:
            rgc = res.get(gc)
            if rgc is None:
                continue
            for fs_name, budgets in rgc.items():
                for bf, row in budgets.items():
                    for a, b in COMPARISONS:
                        key = (gc, fs_name, bf, a, b)
                        if key not in collectors:
                            continue
                        ha, hb = row[a], row[b]
                        diff = ha - hb
                        collectors[key]["diffs"].append(diff)
                        if hb > 0:
                            collectors[key]["rels"].append(diff / hb)

    def pct_ci(vals):
        if len(vals) < 2:
            return None, None
        return float(np.percentile(vals, 2.5)), float(np.percentile(vals, 97.5))

    comparisons_out = []
    for gc in GROUP_COLS:
        pgc = point.get(gc)
        if pgc is None:
            continue
        for fs_name, budgets in pgc.items():
            for bf, row in budgets.items():
                for a, b in COMPARISONS:
                    key = (gc, fs_name, bf, a, b)
                    diffs = collectors[key]["diffs"]
                    rels = collectors[key]["rels"]
                    ha0, hb0 = row[a], row[b]
                    diff0 = ha0 - hb0
                    rel0 = diff0 / hb0 if hb0 > 0 else None
                    diff_lo, diff_hi = pct_ci(diffs)
                    rel_lo, rel_hi = pct_ci(rels)
                    comparisons_out.append({
                        "group_col": gc,
                        "feature_set": fs_name,
                        "budget_frac": bf,
                        "policy_a": a,
                        "policy_b": b,
                        "hits_a": ha0,
                        "hits_b": hb0,
                        "diff_point": diff0,
                        "diff_ci95": [diff_lo, diff_hi],
                        "rel_improve_point": rel0,
                        "rel_improve_ci95": [rel_lo, rel_hi],
                        "n_boot_used": len(diffs),
                        "total_label1_in_sim": row["total_label1_in_sim"],
                        "n_sim": row["n_sim"],
                    })

    out = {
        "id": "L9a",
        "feats": args.feats,
        "n_groups": n_groups,
        "label": LABEL,
        "group_cols": GROUP_COLS,
        "feature_sets": {k: v for k, v in FEATURE_SETS.items()},
        "budget_fracs": list(BUDGET_FRACS),
        "comparisons_spec": COMPARISONS,
        "n_boot_requested": args.n_boot,
        "n_boot_used": n_boot_used,
        "seeds_per_resample": SEEDS,
        "comparisons": comparisons_out,
        "runtime_sec": time.time() - t0,
        "note": ("static = free-label classifier fit once on a held-out quarter of group_col buckets, "
                 "deterministic ranking (sd=0, no policy-seed loop). static_matched = budget-matched: "
                 "spends half of THAT budget's labels on random draws, fits, ranks the rest, no labels "
                 "beyond the audit budget (sim_generic.py verbatim). greedy_online / ts_online = "
                 "self-trained, online-refit every R verified labels, greedy has no bucket layout, "
                 "ts_online Thompson-samples buckets. All four policies share the same fit_b/sim split "
                 "and the same budget_n = int(budget_frac * len(sim)) within a resample/group_col/budget, "
                 "so budget accounting is identical across policies. static_matched/greedy_online/"
                 "ts_online averaged over 8 policy seeds per resample (SEEDS)."),
    }
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    json.dump(out, open(args.out, "w"), indent=1)
    print(json.dumps({"n_boot_used": n_boot_used, "runtime_sec": out["runtime_sec"], "n_comparisons": len(comparisons_out)}, indent=1))


if __name__ == "__main__":
    main()
