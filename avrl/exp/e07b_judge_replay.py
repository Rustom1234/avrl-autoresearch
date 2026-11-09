"""L7b: allocation replay using judge-based ranking as the confidence signal.

Operational diagnostic (not a claim-level pass/kill): does judge_pick_share (E04/L5a's
gold-free judge signal) or a combined flag (low majority-vote agreement OR judge_error)
catch meaningfully more proxy_error groups per verification budget than majority-vote
agreement alone (E01's column)?

No new rollouts, no new judge calls. Reuses avrl/runs/p0/feats.parquet (agreement,
proxy_error) and avrl/runs/L5a/judge_scores.parquet (judge_pick_share, judge_error),
merged on pid. Each candidate signal ranks all 3,000 groups (verify most-suspicious
first); hits = proxy_error=1 groups caught within the top budget_frac of that ranking,
averaged over `seeds` random tie-breaks. Copied from e06a_selfverify_replay.py (L6a)
verbatim except for the self-verification columns swapped for the judge columns
(sv_agreement -> judge_pick_share, self_verify_error -> judge_error); does not edit
any p0/E01/E04/L5a/L6a script.

Usage: e07b_judge_replay.py --feats <feats.parquet> --judgescores <judge_scores.parquet> --out <result.json>
"""
import argparse, json, time
import numpy as np, pandas as pd


def hits_at_budget(order, y, budget):
    return int(y[order[:budget]].sum())


def ranked_hits(score_desc, y, budget, n, seeds):
    vals = []
    for s in range(seeds):
        rng = np.random.default_rng(s)
        jitter = rng.random(n) * 1e-9
        order = np.argsort(-(score_desc + jitter))
        vals.append(hits_at_budget(order, y, budget))
    return float(np.mean(vals)), float(np.std(vals))


def uniform_hits(y, budget, n, seeds):
    vals = []
    for s in range(seeds):
        rng = np.random.default_rng(1000 + s)
        order = rng.permutation(n)
        vals.append(hits_at_budget(order, y, budget))
    return float(np.mean(vals)), float(np.std(vals))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--feats", required=True)
    ap.add_argument("--judgescores", required=True)
    ap.add_argument("--label", default="proxy_error")
    ap.add_argument("--agree_threshold", type=float, default=0.75)  # E01's confident-subset cutoff
    ap.add_argument("--budget_fracs", default="0.10,0.25,0.50")
    ap.add_argument("--seeds", type=int, default=8)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    t0 = time.time()

    feats = pd.read_parquet(args.feats)
    jg = pd.read_parquet(args.judgescores)
    df = feats.merge(jg[["pid", "judge_pick_share", "judge_error"]], on="pid", how="inner")
    assert len(df) == len(feats), f"merge dropped rows: {len(feats)} -> {len(df)}"
    n = len(df)
    y = df[args.label].values.astype(int)
    agreement = df["agreement"].values
    judge_pick_share = df["judge_pick_share"].values
    combined_flag = ((agreement < args.agree_threshold) | (df["judge_error"].values == 1)).astype(float)

    budget_fracs = [float(x) for x in args.budget_fracs.split(",")]
    seeds = args.seeds
    n_error = int(y.sum())

    out = {"n": n, "n_error": n_error, "label": args.label, "agree_threshold": args.agree_threshold,
           "combined_flag_rate": float(combined_flag.mean()),
           "combined_flag_precision": float(y[combined_flag == 1].mean()) if combined_flag.sum() > 0 else None}
    rows = {}
    for bf in budget_fracs:
        budget = int(bf * n)
        agreement_only = ranked_hits(-agreement, y, budget, n, seeds)
        judge_only = ranked_hits(-judge_pick_share, y, budget, n, seeds)
        combined = ranked_hits(combined_flag, y, budget, n, seeds)
        uni = uniform_hits(y, budget, n, seeds)
        oracle = (float(min(budget, n_error)), 0.0)
        rows[f"budget_{bf}"] = {
            "budget_n": budget,
            "uniform": {"mean": uni[0], "sd": uni[1]},
            "agreement_only": {"mean": agreement_only[0], "sd": agreement_only[1]},
            "judge_only": {"mean": judge_only[0], "sd": judge_only[1]},
            "combined_flag": {"mean": combined[0], "sd": combined[1]},
            "oracle": {"mean": oracle[0], "sd": oracle[1]},
        }
    out["budgets"] = rows

    b25 = rows.get("budget_0.25")
    if b25:
        judge_rel_improve = (b25["judge_only"]["mean"] - b25["agreement_only"]["mean"]) / max(b25["agreement_only"]["mean"], 1e-9)
        combined_rel_improve = (b25["combined_flag"]["mean"] - b25["agreement_only"]["mean"]) / max(b25["agreement_only"]["mean"], 1e-9)
        out["judge_vs_agreement_rel_improve_25pct"] = float(judge_rel_improve)
        out["combined_vs_agreement_rel_improve_25pct"] = float(combined_rel_improve)
        best_rel_improve = max(judge_rel_improve, combined_rel_improve)
        out["verdict"] = "SUPPORT_SEPARATE_ARM" if best_rel_improve >= 0.10 else "NO_SUPPORT"

    out["runtime_sec"] = time.time() - t0
    json.dump(out, open(args.out, "w"), indent=1)
    print(json.dumps(out, indent=1))


if __name__ == "__main__":
    main()
