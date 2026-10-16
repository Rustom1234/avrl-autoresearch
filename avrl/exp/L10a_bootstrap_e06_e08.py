"""L10a: bootstrap CI on greedy_online vs static_matched (plus two more comparisons) for E06 and E08,
the two already-computed sim_cell100_cheap.json points whose point estimates favor the allocator
(greedy_online beats static_matched at 25%/50% budget on both) but have never had a bootstrap CI run
on that specific comparison. See RESEARCH_LOG.md M01 critic note, 2026-10-01.

Copied from avrl/exp/e03b_bootstrap_matched.py per the project rule (copy and modify, never edit prior
experiment scripts in place). e03b only computes static_matched vs ts_online, for one hardcoded
agreement-based CHEAP list, one group_col, one budget_frac passed on the command line. This version
differs:
  - --cheap (comma list) replaces e03b's hardcoded CHEAP, so the same script runs E06 (6 cols:
    agreement, n_distinct, mean_tokens, std_tokens, any_truncated, log_qlen) and E08 (those 6 plus
    frac_syntax_error, frac_exception) without editing code. Confirmed against each point's own
    cheap-feature list in avrl/exp/e06_analyze_rundir.py and avrl/exp/E08_analyze.py.
  - adds greedy_online (self-trained ranking, no bucket layout) and static (free-label classifier fit
    once on a held-out quarter of group_col buckets), both lifted verbatim from avrl/exp/e09_bootstrap.py
    (which is itself copied-and-modified from this same e03b base), so the comparison set matches L9d's:
    greedy_online vs static_matched, ts_online vs static_matched, greedy_online vs static.
  - sweeps budget_frac over (0.10, 0.25, 0.50) instead of a single CLI value, computed once per
    resample/budget and sharing the fit_b/sim split and the free-label "static" fit across budgets
    (matches e09_bootstrap.py's convention, cheaper than re-splitting per budget).

label is hardcoded "proxy_error" (E06 and E08's own label column, confirmed against
avrl/exp/e06_analyze_rundir.py and avrl/exp/E08_analyze.py: both read G.proxy_error). group_col
defaults to cell100, L10a's and M01's primary grouping.

No new rollouts, no GPU. Reuses avrl/runs/E06/feats.parquet (3,000 groups) and
avrl/runs/E08/feats.parquet (600 groups), already on disk.

300 group-level bootstrap resamples x 8 policy seeds per resample, same config L9a-d used.

Usage:
  L10a_bootstrap_e06_e08.py --feats F --cheap c1,c2,... [--group_col cell100] [--n_boot 300] [--seed 0] --out OUT
"""
import argparse, json, os, time
import numpy as np, pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline

MIN, K, SEEDS = 3, 1, 8  # matches e03b_bootstrap_matched.py / e09_bootstrap.py (K=1 group per visit)
LABEL = "proxy_error"
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
        take = min(K, budget - spent)
        if greedy:
            mu = np.where(~seen)[0]
            if len(mu) == 0:
                break
            c = mu[np.argsort(-score[mu])][:take]
            seen[c] = True; hits += int(y[c].sum()); spent += len(c)
        else:
            avail = remaining > 0
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


def compute(df, group_col, cheap):
    """One pass: fit_b/sim split (sim_generic.py's split, verbatim), then static / static_matched /
    greedy_online / ts_online hits at every budget_frac. Returns {budget_frac: {...}} or None."""
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

    Xs = sim[cheap].fillna(0).values
    y = sim[LABEL].values.astype(int)
    codes, uniq = pd.factorize(sim[group_col].values); n_b = len(uniq)
    if n_b == 0 or y.sum() == 0:
        return None
    members = [np.where(codes == b)[0] for b in range(n_b)]

    Xf = fit[cheap].fillna(0).values
    yf = fit[LABEL].values.astype(int)
    if 0 < yf.sum() < len(yf):
        static_order = np.argsort(-clf().fit(Xf, yf).predict_proba(Xs)[:, 1])
    else:
        static_order = np.arange(len(sim))  # degenerate free-label fit set: no signal, natural order

    out = {}
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
                sc[tr] = -1; order = np.concatenate([tr, np.argsort(-sc)])
            hits_sm.append(int(y[order[:budget_n]].sum()))
        hits_static_matched = float(np.mean(hits_sm))

        hits_greedy = float(np.mean([run_online(Xs, y, codes, n_b, members, budget_n, s, True) for s in range(SEEDS)]))
        hits_ts = float(np.mean([run_online(Xs, y, codes, n_b, members, budget_n, s, False) for s in range(SEEDS)]))

        out[bf] = {
            "static": float(hits_static), "static_matched": hits_static_matched,
            "greedy_online": hits_greedy, "ts_online": hits_ts,
            "total_label1_in_sim": int(y.sum()), "n_sim": len(sim),
        }
    return out if out else None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--feats", required=True)
    ap.add_argument("--cheap", required=True, help="comma-separated cheap feature column names")
    ap.add_argument("--group_col", default="cell100")
    ap.add_argument("--n_boot", type=int, default=300)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    cheap = args.cheap.split(",")

    t0 = time.time()
    G = pd.read_parquet(args.feats)
    n_groups = len(G)

    point = compute(G, args.group_col, cheap)
    if point is None:
        raise RuntimeError("point estimate degenerate on the unresampled data, cannot proceed")

    rng = np.random.default_rng(args.seed)
    collectors = {(bf, a, b): {"diffs": [], "rels": []} for bf in point for a, b in COMPARISONS}

    n_boot_used = 0
    for _ in range(args.n_boot):
        idx = rng.integers(0, n_groups, n_groups)
        dfb = G.iloc[idx].reset_index(drop=True)
        res = compute(dfb, args.group_col, cheap)
        n_boot_used += 1
        if res is None:
            continue
        for bf, row in res.items():
            for a, b in COMPARISONS:
                key = (bf, a, b)
                if key not in collectors:
                    continue
                diff = row[a] - row[b]
                collectors[key]["diffs"].append(diff)
                if row[b] > 0:
                    collectors[key]["rels"].append(diff / row[b])

    def pct_ci(vals):
        if len(vals) < 2:
            return None, None
        return float(np.percentile(vals, 2.5)), float(np.percentile(vals, 97.5))

    comparisons_out = []
    for bf, row in point.items():
        for a, b in COMPARISONS:
            diffs = collectors[(bf, a, b)]["diffs"]; rels = collectors[(bf, a, b)]["rels"]
            diff0 = row[a] - row[b]
            rel0 = diff0 / row[b] if row[b] > 0 else None
            diff_lo, diff_hi = pct_ci(diffs); rel_lo, rel_hi = pct_ci(rels)
            comparisons_out.append({
                "budget_frac": bf, "policy_a": a, "policy_b": b,
                "hits_a": row[a], "hits_b": row[b], "diff_point": diff0,
                "diff_ci95": [diff_lo, diff_hi],
                "rel_improve_point": rel0, "rel_improve_ci95": [rel_lo, rel_hi],
                "n_boot_used": len(diffs),
                "total_label1_in_sim": row["total_label1_in_sim"], "n_sim": row["n_sim"],
            })

    out = {
        "id": "L10a",
        "feats": args.feats,
        "n_groups": n_groups,
        "label": LABEL,
        "group_col": args.group_col,
        "cheap_features": cheap,
        "budget_fracs": list(BUDGET_FRACS),
        "comparisons_spec": COMPARISONS,
        "n_boot_requested": args.n_boot,
        "n_boot_used": n_boot_used,
        "seeds_per_resample": SEEDS,
        "comparisons": comparisons_out,
        "runtime_sec": time.time() - t0,
        "note": ("static = free-label classifier fit once on a held-out quarter of group_col buckets, "
                 "deterministic ranking (sd=0). static_matched = budget-matched: spends half of the "
                 "budget's labels on random draws, fits, ranks the rest, no labels beyond the audit "
                 "budget (sim_generic.py verbatim). greedy_online / ts_online = self-trained, "
                 "online-refit every R verified labels; greedy has no bucket layout, ts_online "
                 "Thompson-samples buckets. All four policies share the same fit_b/sim split and the "
                 "same budget_n=int(budget_frac*len(sim)) within a resample/budget. "
                 "static_matched/greedy_online/ts_online averaged over 8 policy seeds per resample."),
    }
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    json.dump(out, open(args.out, "w"), indent=1)
    print(json.dumps({"n_boot_used": n_boot_used, "runtime_sec": out["runtime_sec"],
                       "n_comparisons": len(comparisons_out)}, indent=1))


if __name__ == "__main__":
    main()
