"""L3a: same bootstrap significance test as E03, with "static" swapped for sim_generic.py's existing
"static_matched" policy as the threshold analog (E03 follow-up 1).

No new rollouts, no GPU. Reuses avrl/runs/p0/feats.parquet (already on disk). Copied from
avrl/exp/e03_bootstrap_sig.py per the project rule: copy and modify, never edit p0/E03 scripts in place.

Only change vs e03_bootstrap_sig.py: the "static" branch of fit_eval (free-label classifier fit once on a
held-out quarter of clusters, ranked deterministically) is replaced with sim_generic.py's static_matched
logic verbatim (runs/sim_generic.py lines 30-35): within the budget, spend half on random labels, fit
LogisticRegression on those, rank the remaining half of the budget by predicted probability -- no labels
spent beyond the audit budget. static_matched's own rng is seeded per-seed inside sim_generic.py's run()
(unlike free-label "static", which is deterministic, sd=0 across seeds), so it is averaged over the same
`seeds` as ts_online below.

The held-out-quarter split (fit_b), the "sim" population it produces, the ts_online logic,
corrupted_admitted_rate definition, and the 500-resample bootstrap procedure are byte-identical to E03 so
the two results are directly comparable. The held-out quarter is still excluded from "sim" even though
static_matched no longer uses it for fitting, so that ts_online's numbers (which only ever see "sim") stay
identical to E03's.

corrupted_admitted_rate = (# proxy_error=1 groups NOT selected for verification) / (# proxy_error=1 groups),
computed over the "sim" population (the 3/4 of clusters not used as the free-label fit pool in E03 -- same
population sim_generic.py itself reports hits/budget over).

Usage:
  e03b_bootstrap_matched.py --feats F --label L --group G --budget_frac B --n_boot N --seed S --out OUT
"""
import argparse, json, os, time
import numpy as np, pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline

MIN, K, SEEDS = 3, 1, 8  # matches avrl/runs/p0/sim_alloc.py (K=1 group per visit, group-level budgets)
CHEAP = ["agreement", "n_distinct", "mean_tokens", "std_tokens", "any_truncated", "log_qlen"]


def fit_eval(df, label, group_col, budget_frac, seeds=SEEDS):
    """One pass of sim_generic.py's setup + 'static_matched' and 'ts_online' policies.
    Returns (hits_static_matched_mean_over_seeds, hits_ts_online_mean_over_seeds, total_label1_in_sim)
    or None if degenerate."""
    cnt = df[group_col].value_counts()
    df = df[df[group_col].isin(cnt[cnt >= MIN].index)].reset_index(drop=True)
    if len(df) == 0:
        return None
    rng0 = np.random.default_rng(0)
    buckets = df[group_col].unique(); rng0.shuffle(buckets)
    fit_b = set(buckets[: len(buckets) // 4])
    sim = df[~df[group_col].isin(fit_b)].reset_index(drop=True)
    if len(sim) == 0:
        return None

    X = lambda d: d[CHEAP].fillna(0).values
    Xs = X(sim)
    y = sim[label].values.astype(int); codes, uniq = pd.factorize(sim[group_col].values); n_b = len(uniq)
    if n_b == 0:
        return None
    members = [np.where(codes == b)[0] for b in range(n_b)]
    total = int(y.sum())
    if total == 0:
        return None
    budget = int(budget_frac * len(sim))
    if budget <= 0:
        return None

    # static_matched: budget-matched threshold analog (sim_generic.py run(), policy=="static_matched", verbatim).
    # Spend half the budget on random labels, fit a classifier on those, rank the remaining half of the budget
    # by predicted probability. No labels spent beyond the audit budget. Averaged over `seeds` (own rng per seed).
    hits_sm_list = []
    for s in range(seeds):
        rng = np.random.default_rng(s)
        perm = rng.permutation(len(y)); tr = perm[: max(budget // 2, 1)]
        if y[tr].sum() == 0:
            order = perm
        else:
            sc = make_pipeline(StandardScaler(), LogisticRegression(max_iter=1000, class_weight="balanced")).fit(Xs[tr], y[tr]).predict_proba(Xs)[:, 1]
            sc[tr] = -1; order = np.concatenate([tr, np.argsort(-sc)])
        hits_sm_list.append(int(y[order[:budget]].sum()))
    hits_static_matched = float(np.mean(hits_sm_list))

    # ts_online: Thompson sampling over buckets, self-trained ranking refit every R verified labels (unchanged from E03)
    hits_list = []
    for s in range(seeds):
        rng = np.random.default_rng(s)
        R = max(budget // 8, 40)
        seen = np.zeros(len(y), bool); score = rng.random(len(y)) * 1e-3; bprior = None
        alpha = np.ones(n_b); beta = np.ones(n_b); remaining = np.array([len(m) for m in members])
        hits = spent = 0; next_fit = R
        while spent < budget:
            take = min(K, budget - spent); avail = remaining > 0
            if not avail.any():
                break
            sv = rng.beta(alpha + 10 * bprior, beta + 10 * (1 - bprior)) if bprior is not None else rng.beta(alpha, beta)
            sv[~avail] = -1; b = int(np.argmax(sv)); mu = members[b][~seen[members[b]]]
            c = mu[np.argsort(-score[mu])][:take]
            seen[c] = True; remaining[b] -= len(c); h = int(y[c].sum()); alpha[b] += h; beta[b] += len(c) - h
            hits += h; spent += len(c)
            if spent >= next_fit and 0 < y[seen].sum() < seen.sum():
                score = make_pipeline(StandardScaler(), LogisticRegression(max_iter=500, class_weight="balanced")).fit(Xs[seen], y[seen]).predict_proba(Xs)[:, 1]
                bp = pd.Series(score).groupby(codes).mean().values; bprior = bp / bp.max(); next_fit += R
        hits_list.append(hits)
    hits_ts_online = float(np.mean(hits_list))
    return hits_static_matched, hits_ts_online, total


def rates(hsm, ht, total):
    return (total - hsm) / total, (total - ht) / total  # static_matched_rate, learned_rate


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--feats", required=True)
    ap.add_argument("--label", required=True)
    ap.add_argument("--group", required=True)
    ap.add_argument("--budget_frac", type=float, required=True)
    ap.add_argument("--n_boot", type=int, default=500)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    t0 = time.time()
    G = pd.read_parquet(args.feats)
    n_groups = len(G)

    point = fit_eval(G, args.label, args.group, args.budget_frac)
    if point is None:
        raise RuntimeError("point estimate degenerate on the unresampled data, cannot proceed")
    hsm0, ht0, tot0 = point
    sm0, lrn0 = rates(hsm0, ht0, tot0)

    rng = np.random.default_rng(args.seed)
    diffs, skipped = [], 0
    for i in range(args.n_boot):
        idx = rng.integers(0, n_groups, n_groups)
        dfb = G.iloc[idx].reset_index(drop=True)
        r = fit_eval(dfb, args.label, args.group, args.budget_frac)
        if r is None:
            skipped += 1
            continue
        hsm, ht, tot = r
        sm, lrn = rates(hsm, ht, tot)
        diffs.append(sm - lrn)

    diffs = np.array(diffs)
    ci_lo, ci_hi = (float(np.percentile(diffs, 2.5)), float(np.percentile(diffs, 97.5))) if len(diffs) else (float("nan"), float("nan"))
    point_diff = sm0 - lrn0
    if len(diffs) == 0:
        verdict = "DEGENERATE"
    elif ci_lo > 0:
        verdict = "PASS"  # CI excludes zero, positive: learned beats static_matched, significant (confound explains E03)
    elif ci_hi < 0:
        verdict = "KILL_REVERSED"  # CI excludes zero, negative: static_matched still beats learned, significant (reversal is real)
    else:
        verdict = "KILL"  # CI includes zero: not distinguishable from resampling noise

    out = {
        "id": "L3a",
        "n_groups": n_groups,
        "budget_frac": args.budget_frac,
        "group_col": args.group,
        "label": args.label,
        "n_boot_requested": args.n_boot,
        "n_boot_used": int(len(diffs)),
        "n_boot_skipped_degenerate": skipped,
        "point_estimate": {
            "hits_static_matched_mean8seeds": hsm0,
            "hits_ts_online_mean8seeds": ht0,
            "total_label1_in_sim": tot0,
            "static_matched_corrupted_admitted_rate": sm0,
            "learned_corrupted_admitted_rate": lrn0,
            "diff_static_matched_minus_learned": point_diff,
        },
        "bootstrap": {
            "mean_diff": float(diffs.mean()) if len(diffs) else None,
            "sd_diff": float(diffs.std()) if len(diffs) else None,
            "ci95_lo": ci_lo,
            "ci95_hi": ci_hi,
        },
        "verdict": verdict,
        "pass": bool(verdict == "PASS"),
        "runtime_sec": time.time() - t0,
        "note": ("static_matched = budget-matched threshold analog (sim_generic.py's static_matched policy: "
                 "spends half the audit budget on random labels, fits a classifier on those, ranks the "
                 "remaining half of the budget by predicted probability, no labels beyond the audit budget, "
                 "averaged over 8 seeds). ts_online = learned analog (self-trained, online-refit every R "
                 "verified labels, Thompson sampling over cells), unchanged from E03. This removes E03's "
                 "flagged confound: static no longer gets a free-labeled quarter of clusters beyond the audit "
                 "budget that ts_online never gets."),
    }
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    json.dump(out, open(args.out, "w"), indent=1)
    print(json.dumps(out, indent=1))


if __name__ == "__main__":
    main()
