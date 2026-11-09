"""E03: bootstrap significance test, threshold vs learned allocator, at the budget phase-1 plans to spend.

No new rollouts, no GPU. Reuses avrl/runs/p0/feats.parquet (already on disk). Copies (does not import or edit)
the exact "static" and "ts_online" policy logic from runs/sim_generic.py (same as avrl/runs/p0/sim_alloc.py:
MIN=3, K=1 group per visit) and reruns it on bootstrap resamples of the groups.

Assumption (stated explicitly, per E03 plan): sim_generic.py's own policy names (static/static_matched/ts_flat/
ts_online/ts_static/neighbor/oracle) do not literally match allocators.py's phase-1 arm names (full/random/
threshold/learned). "static" (free-label-fit classifier, ranked once, not updated online) is used here as the
closest structural analog to the fixed "threshold" arm. "ts_online" (self-trained, refit every R verified
labels, Thompson sampling over cells) is used as the closest analog to "learned". Note: "static" gets a free
labeled quarter of clusters beyond the audit budget that "ts_online" does not get -- this asymmetry is inherent
to the existing harness, not something E03 introduces.

corrupted_admitted_rate = (# proxy_error=1 groups NOT selected for verification) / (# proxy_error=1 groups),
computed over the "sim" population (the 3/4 of clusters not used as the free-label fit pool -- same population
sim_generic.py itself reports hits/budget over).

Usage:
  e03_bootstrap_sig.py --feats F --label L --group G --budget_frac B --n_boot N --seed S --out OUT
"""
import argparse, json, os, time
import numpy as np, pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline

MIN, K, SEEDS = 3, 1, 8  # matches avrl/runs/p0/sim_alloc.py (K=1 group per visit, group-level budgets)
CHEAP = ["agreement", "n_distinct", "mean_tokens", "std_tokens", "any_truncated", "log_qlen"]


def fit_eval(df, label, group_col, budget_frac, seeds=SEEDS):
    """One pass of sim_generic.py's setup + 'static' and 'ts_online' policies.
    Returns (hits_static, hits_ts_online_mean_over_seeds, total_label1_in_sim) or None if degenerate."""
    cnt = df[group_col].value_counts()
    df = df[df[group_col].isin(cnt[cnt >= MIN].index)].reset_index(drop=True)
    if len(df) == 0:
        return None
    rng0 = np.random.default_rng(0)
    buckets = df[group_col].unique(); rng0.shuffle(buckets)
    fit_b = set(buckets[: len(buckets) // 4])
    fit = df[df[group_col].isin(fit_b)]; sim = df[~df[group_col].isin(fit_b)].reset_index(drop=True)
    if len(sim) == 0 or len(fit) == 0 or fit[label].nunique() < 2:
        return None

    X = lambda d: d[CHEAP].fillna(0).values
    clf = make_pipeline(StandardScaler(), LogisticRegression(max_iter=1000, class_weight="balanced")).fit(X(fit), fit[label])
    Xs = X(sim); sim = sim.copy(); sim["static"] = clf.predict_proba(Xs)[:, 1]
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

    # static: deterministic ranking by the free-label classifier, take the top `budget` (sd=0 across seeds, matches sim_generic.py)
    order = np.argsort(-sim["static"].values)
    hits_static = int(y[order[:budget]].sum())

    # ts_online: Thompson sampling over buckets, self-trained ranking refit every R verified labels
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
    return hits_static, hits_ts_online, total


def rates(hs, ht, total):
    return (total - hs) / total, (total - ht) / total  # threshold_rate, learned_rate


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
    hs0, ht0, tot0 = point
    thr0, lrn0 = rates(hs0, ht0, tot0)

    rng = np.random.default_rng(args.seed)
    diffs, skipped = [], 0
    for i in range(args.n_boot):
        idx = rng.integers(0, n_groups, n_groups)
        dfb = G.iloc[idx].reset_index(drop=True)
        r = fit_eval(dfb, args.label, args.group, args.budget_frac)
        if r is None:
            skipped += 1
            continue
        hs, ht, tot = r
        thr, lrn = rates(hs, ht, tot)
        diffs.append(thr - lrn)

    diffs = np.array(diffs)
    ci_lo, ci_hi = (float(np.percentile(diffs, 2.5)), float(np.percentile(diffs, 97.5))) if len(diffs) else (float("nan"), float("nan"))
    point_diff = thr0 - lrn0
    if len(diffs) == 0:
        verdict = "DEGENERATE"
    elif ci_lo > 0:
        verdict = "PASS"  # CI excludes zero, positive: learned beats threshold, significant
    elif ci_hi < 0:
        verdict = "KILL_REVERSED"  # CI excludes zero, negative: threshold beats learned, significant
    else:
        verdict = "KILL"  # CI includes zero: not distinguishable from resampling noise

    out = {
        "id": "E03",
        "n_groups": n_groups,
        "budget_frac": args.budget_frac,
        "group_col": args.group,
        "label": args.label,
        "n_boot_requested": args.n_boot,
        "n_boot_used": int(len(diffs)),
        "n_boot_skipped_degenerate": skipped,
        "point_estimate": {
            "hits_static": hs0,
            "hits_ts_online_mean8seeds": ht0,
            "total_label1_in_sim": tot0,
            "threshold_corrupted_admitted_rate": thr0,
            "learned_corrupted_admitted_rate": lrn0,
            "diff_threshold_minus_learned": point_diff,
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
        "note": ("static = threshold analog (free-label-fit classifier, ranked once, fit on a held-out quarter "
                 "of cell100 clusters, sd=0 across seeds). ts_online = learned analog (self-trained, online-refit "
                 "every R verified labels, Thompson sampling over cells). static has a labeling advantage (free "
                 "labels from a held-out cluster quarter, beyond the audit budget) that ts_online does not get; "
                 "this asymmetry is inherent to sim_generic.py's existing policy definitions, not an E03 change."),
    }
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    json.dump(out, open(args.out, "w"), indent=1)
    print(json.dumps(out, indent=1))


if __name__ == "__main__":
    main()
