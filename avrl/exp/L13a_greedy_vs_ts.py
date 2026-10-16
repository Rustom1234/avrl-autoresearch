"""L13a: head-to-head greedy_online vs ts_online bootstrap, copied-and-modified from
avrl/exp/bootstrap_replay.py per the project rule (copy and modify, never edit prior experiment
scripts in place). Only two module constants differ from bootstrap_replay.py: BUDGET_FRACS is
restricted to (0.25,) (the primary budget used in every other ledger table, e.g. M01b/L12b) and
COMPARISONS is restricted to [("greedy_online", "ts_online")] (direct head-to-head, not each vs
static_matched). Everything else -- clf, run_online, split_sim, the percentile-CI bootstrap -- is
unchanged, so results are directly comparable to L12b's ts_online-vs-static_matched numbers.

Motivation (L12b follow-up 1 in RESEARCH_LOG.md): L12b found ts_online beats static_matched at the
25% budget on exactly 2 of 7 points (E09, E09d), the only two with a significant positive CI. This
script asks the sharper question L12b could not answer from stored aggregates alone: on those same
two points, does ts_online add anything over greedy_online (the simpler classifier-refit allocator
that M01b found winning everywhere else), or does greedy_online dominate there too. Restricted to
E09 and E09d only (the two points with an actual win to explain) to fit the 45-minute compute cap;
E09b and E09c tied against static_matched in L12b so are lower priority and not run here.

No new rollouts, no GPU, CPU-only.

Usage:
  bootstrap_replay.py --feats F.parquet [--label proxy_error] [--group cell100]
                       [--features a,b,c | --features auto] [--use_pcs]
                       [--n_boot 300] [--seeds 8] --out OUT.json

--features auto: every numeric column except the label, id-like columns (pid, id, *_id), and any
column named cell* or pc* (pc* is added separately by --use_pcs).
"""
import argparse, json, os, time
import numpy as np, pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline

MIN, K = 3, 1
BUDGET_FRACS = (0.25,)  # L13a: only the primary budget used across the ledger's headline tables
COMPARISONS = [("greedy_online", "ts_online")]  # L13a: direct head-to-head, not vs static_matched


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


def resolve_features(df, label, features_arg, use_pcs):
    if features_arg == "auto":
        numeric = df.select_dtypes(include=[np.number]).columns
        feats = [c for c in numeric if c != label and c not in ("pid", "id") and not c.endswith("_id")
                  and not c.startswith("cell") and not c.startswith("pc")]
    else:
        feats = features_arg.split(",")
    if use_pcs:
        feats = feats + [f"pc{i}" for i in range(16)]
    return feats


def compute_point(df, group_col, feats, label, seeds):
    """Returns {budget_frac: {policy: hits, 'total_label1_in_sim':.., 'n_sim':..}} or None."""
    split = split_sim(df, group_col)
    if split is None:
        return None
    fit, sim = split
    Xs = sim[feats].fillna(0).values
    y = sim[label].values.astype(int)
    codes, uniq = pd.factorize(sim[group_col].values)
    n_b = len(uniq)
    if n_b == 0 or y.sum() == 0:
        return None
    members = [np.where(codes == b)[0] for b in range(n_b)]

    Xf = fit[feats].fillna(0).values
    yf = fit[label].values.astype(int)
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
        for s in range(seeds):
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

        hits_greedy = float(np.mean([run_online(Xs, y, codes, n_b, members, budget_n, s, True) for s in range(seeds)]))
        hits_ts = float(np.mean([run_online(Xs, y, codes, n_b, members, budget_n, s, False) for s in range(seeds)]))

        budgets_out[bf] = {
            "static": float(hits_static),
            "static_matched": hits_static_matched,
            "greedy_online": hits_greedy,
            "ts_online": hits_ts,
            "total_label1_in_sim": int(y.sum()),
            "n_sim": len(sim),
        }
    return budgets_out if budgets_out else None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--feats", required=True)
    ap.add_argument("--label", default="proxy_error")
    ap.add_argument("--group", default="cell100")
    ap.add_argument("--features", required=True, help="comma list, or 'auto'")
    ap.add_argument("--use_pcs", action="store_true", help="append pc0..pc15 to the feature list")
    ap.add_argument("--n_boot", type=int, default=300)
    ap.add_argument("--seeds", type=int, default=8)
    ap.add_argument("--seed", type=int, default=0, help="RNG seed for the resample draws")
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    t0 = time.time()
    G = pd.read_parquet(args.feats)
    n_groups = len(G)
    feats = resolve_features(G, args.label, args.features, args.use_pcs)

    point = compute_point(G, args.group, feats, args.label, args.seeds)
    if point is None:
        raise SystemExit(f"no usable point for group={args.group} label={args.label} feats={feats}")

    rng = np.random.default_rng(args.seed)
    collectors = {(bf, a, b): {"diffs": [], "rels": []} for bf in point for a, b in COMPARISONS}

    n_boot_used = 0
    for i in range(args.n_boot):
        idx = rng.integers(0, n_groups, n_groups)
        dfb = G.iloc[idx].reset_index(drop=True)
        res = compute_point(dfb, args.group, feats, args.label, args.seeds)
        n_boot_used += 1
        if res is None:
            continue
        for bf, row in res.items():
            for a, b in COMPARISONS:
                key = (bf, a, b)
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
    for bf, row in point.items():
        for a, b in COMPARISONS:
            key = (bf, a, b)
            diffs = collectors[key]["diffs"]
            rels = collectors[key]["rels"]
            ha0, hb0 = row[a], row[b]
            diff0 = ha0 - hb0
            rel0 = diff0 / hb0 if hb0 > 0 else None
            diff_lo, diff_hi = pct_ci(diffs)
            rel_lo, rel_hi = pct_ci(rels)
            comparisons_out.append({
                "group_col": args.group,
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
        "id": "bootstrap_replay",
        "feats": args.feats,
        "n_groups": n_groups,
        "label": args.label,
        "group_col": args.group,
        "features": feats,
        "use_pcs": args.use_pcs,
        "budget_fracs": list(BUDGET_FRACS),
        "comparisons_spec": COMPARISONS,
        "n_boot_requested": args.n_boot,
        "n_boot_used": n_boot_used,
        "seeds_per_resample": args.seeds,
        "comparisons": comparisons_out,
        "runtime_sec": time.time() - t0,
        "note": ("static = free-label classifier fit once on a held-out quarter of group buckets, "
                 "deterministic ranking (sd=0, no policy-seed loop). static_matched = budget-matched: "
                 "spends half of THAT budget's labels on random draws, fits, ranks the rest, no labels "
                 "beyond the audit budget (sim_generic.py verbatim). greedy_online / ts_online = "
                 "self-trained, online-refit every R verified labels, greedy has no bucket layout, "
                 "ts_online Thompson-samples buckets. All four policies share the same fit_b/sim split "
                 "and the same budget_n = int(budget_frac * len(sim)) within a resample/budget, so "
                 "budget accounting is identical across policies. static_matched/greedy_online/"
                 "ts_online averaged over --seeds policy seeds per resample."),
    }
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    json.dump(out, open(args.out, "w"), indent=1)
    print(json.dumps({"n_boot_used": n_boot_used, "runtime_sec": out["runtime_sec"], "n_comparisons": len(comparisons_out)}, indent=1))


if __name__ == "__main__":
    main()
