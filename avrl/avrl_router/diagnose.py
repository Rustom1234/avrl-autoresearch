"""Standalone regime diagnostic for the AVRL router.

Given a sample of rollout groups with a cheap grader's confidence and a flag
for whether the cheap grader disagreed with the expensive grader, report:
how informative the confidence signal is (AUC), how much verification budget
it takes to keep admitted corruption under a target, and a plain-language
regime label.

No training, no model. Pure numpy/pandas/sklearn on a saved sample.
"""

import argparse
import json
import os

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score


def _bootstrap_ci(stat_fn, n, n_boot, seed):
    """95% percentile CI over n_boot bootstrap resamples of size n.

    stat_fn(idx) returns a scalar for resample indices idx, or nan/None if
    the resample is degenerate (e.g. only one class present for an AUC).
    """
    rng = np.random.default_rng(seed)
    vals = []
    for _ in range(n_boot):
        idx = rng.integers(0, n, n)
        v = stat_fn(idx)
        if v is not None and np.isfinite(v):
            vals.append(v)
    if len(vals) < 2:
        return float("nan"), float("nan")
    return float(np.percentile(vals, 2.5)), float(np.percentile(vals, 97.5))


def _auc(conf, err):
    """AUC of (-confidence) predicting error. Higher = lower confidence predicts error."""
    if len(np.unique(err)) < 2:
        return float("nan")
    return float(roc_auc_score(err, -conf))


def _sort_by_confidence(conf, err):
    """Return err reordered so index 0 is the lowest-confidence group. Stable sort for ties."""
    order = np.argsort(conf, kind="stable")
    return err[order]


def _corrupted_at_k(err_sorted, k):
    """Fraction of all groups with a corrupted reward if the first k (lowest-confidence) are verified."""
    n = len(err_sorted)
    k = min(max(int(round(k)), 0), n)
    return float(err_sorted[k:].sum()) / n if n else float("nan")


def _threshold_corrupted(conf, err, budget):
    err_sorted = _sort_by_confidence(conf, err)
    return _corrupted_at_k(err_sorted, budget * len(err_sorted))


def _recommend_budget(conf, err, target):
    """Smallest budget on a 0.05 grid (0.05..1.0) with threshold-admitted corruption <= target."""
    err_sorted = _sort_by_confidence(conf, err)
    n = len(err_sorted)
    if n == 0:
        return float("nan")
    for i in range(1, 21):
        b = i / 20.0
        if _corrupted_at_k(err_sorted, b * n) <= target:
            return b
    return 1.0


def _recommend_realizable_budget(conf, err, err_point_sub, target, n_groups_per_step, n_boot, seed):
    """Candidate budgets realizable by a harness that verifies k of n_groups_per_step groups per
    step (k = 1..n_groups_per_step), i.e. the only budgets for which n_verify = round(budget *
    n_groups_per_step) actually equals k. For each candidate, compute threshold corruption (point
    estimate and bootstrap upper 95% bound) and random corruption (point estimate), same formulas
    as the 0.05 grid / `budgets` table. Returns (recommended_point, recommended_upper, candidates):
    the smallest candidate meeting `target` under each rule (point estimate vs. upper bound), and
    the full candidate list so both rules can be reported side by side. Falls back to 1.0 (full
    verification, which has zero corruption by construction) if no smaller candidate qualifies,
    same convention as _recommend_budget."""
    candidates = []
    rec_point, rec_upper = None, None
    for k in range(1, n_groups_per_step + 1):
        b = k / n_groups_per_step
        t_point = _threshold_corrupted(conf, err, b)
        _, t_hi = _bootstrap_ci(lambda idx: _threshold_corrupted(conf[idx], err[idx], b), len(err), n_boot, seed)
        r_point = (1 - b) * err_point_sub
        candidates.append({
            "budget": b, "k": k,
            "threshold_point": t_point, "threshold_upper": t_hi, "random_point": r_point,
        })
        if rec_point is None and t_point <= target:
            rec_point = b
        if rec_upper is None and np.isfinite(t_hi) and t_hi <= target:
            rec_upper = b
    if rec_point is None:
        rec_point = 1.0
    if rec_upper is None:
        rec_upper = 1.0
    return rec_point, rec_upper, candidates


def _resolve_target(target_corruption, base_error):
    """Resolve target_corruption into (target_mode, effective_target).

    target_corruption is either a float (absolute target, used as-is) or a
    string "rel:F" meaning F times base_error (the measured proxy error rate
    point estimate for this confidence column). A plain numeric string (e.g.
    "0.02", as now arrives from the CLI) is parsed as absolute, same result
    as passing the float 0.02.
    """
    if isinstance(target_corruption, str) and target_corruption.startswith("rel:"):
        factor = float(target_corruption[len("rel:"):])
        return "relative", factor * base_error
    return "absolute", float(target_corruption)


def _regime(auc_point, recommended_budget):
    # Bands calibrated on the project's in-training runs: vote agreement (AUC 0.87 to 0.91)
    # and response consistency (AUC 0.67 to 0.69) both beat random allocation in training,
    # reward-model margin (AUC 0.51 to 0.57) did not.
    if np.isnan(auc_point):
        return "undefined", "proxy error has no variance in this sample; AUC is undefined."
    if auc_point >= 0.85:
        return "strong", f"use the confidence-ranked threshold at the recommended budget ({recommended_budget:.2f})."
    if auc_point < 0.65:
        return "uninformative", (
            "this confidence signal is close to random, verification budget buys little per "
            "call, use a different confidence signal such as response consistency or raise the budget."
        )
    return "usable", (
        f"use the confidence-ranked threshold at the recommended budget ({recommended_budget:.2f}); "
        "the signal is moderate, so expect the relative reduction shown, not the strong-regime gain."
    )


def diagnose(df, confidence_cols, error_col, budgets, target_corruption, n_boot=1000, seed=0,
             n_groups_per_step=None, pick="point"):
    """Diagnose how useful one or more cheap-grader confidence columns are for budgeting verification.

    df: one row per rollout group.
    confidence_cols: column name or list of column names, higher = cheap grader more confident.
    error_col: 1 if the cheap grader disagreed with the expensive grader (would be corrupted if unverified).
    budgets: list of verification budget fractions in (0, 1], e.g. [0.1, 0.25, 0.5].
    target_corruption: target admitted-corruption rate used to pick a recommended budget.
    n_groups_per_step: if given, also compute a realizable-budget recommendation restricted to
        candidates k / n_groups_per_step (k = 1..n_groups_per_step) -- the only budgets a harness
        that verifies k of n_groups_per_step groups per step can actually realize. Stored per
        confidence column under "realizable". Default None leaves existing output unchanged.
    pick: "point" (default) or "upper" -- which bound of the realizable candidates' threshold
        corruption to require meeting `target_corruption` under. Only used when
        n_groups_per_step is given.
    Returns a plain dict, JSON-serializable.
    """
    if isinstance(confidence_cols, str):
        confidence_cols = [confidence_cols]

    err_all = df[error_col].to_numpy().astype(float)
    n = len(df)
    err_point = float(err_all.mean()) if n else float("nan")
    err_lo, err_hi = _bootstrap_ci(lambda idx: err_all[idx].mean(), n, n_boot, seed)

    result = {
        "n_groups": int(n),
        "error_col": error_col,
        "budgets": list(budgets),
        "target_corruption": target_corruption,
        "n_boot": n_boot,
        "seed": seed,
        "n_groups_per_step": n_groups_per_step,
        "pick": pick,
        "proxy_error_rate": {"point": err_point, "ci95_lo": err_lo, "ci95_hi": err_hi},
        "confidence_columns": {},
    }

    for i, col in enumerate(confidence_cols):
        sub = df[[col, error_col]].dropna()
        conf = sub[col].to_numpy().astype(float)
        err = sub[error_col].to_numpy().astype(float)
        m = len(sub)
        col_seed = seed + 1000 * (i + 1)

        auc_point = _auc(conf, err)
        auc_lo, auc_hi = _bootstrap_ci(lambda idx: _auc(conf[idx], err[idx]), m, n_boot, col_seed)
        err_point_sub = float(err.mean()) if m else float("nan")

        budget_out = {}
        for b in budgets:
            t_point = _threshold_corrupted(conf, err, b)
            t_lo, t_hi = _bootstrap_ci(lambda idx: _threshold_corrupted(conf[idx], err[idx], b), m, n_boot, col_seed)
            r_point = (1 - b) * err_point_sub
            r_lo, r_hi = _bootstrap_ci(lambda idx: (1 - b) * err[idx].mean(), m, n_boot, col_seed)
            rel_reduction = (r_point - t_point) / r_point if r_point > 0 else float("nan")
            budget_out[str(b)] = {
                "threshold": {"point": t_point, "ci95_lo": t_lo, "ci95_hi": t_hi},
                "random": {"point": r_point, "ci95_lo": r_lo, "ci95_hi": r_hi},
                "full_verification_floor": 0.0,
                "relative_reduction_vs_random": rel_reduction,
            }

        # Compact 0.05-grid of threshold vs. random corruption (point estimates only, no
        # bootstrap -- this is for eyeballing an override, the CI'd `budgets` table above is
        # unchanged), same grid _recommend_budget scans.
        grid_out = {}
        for i_grid in range(1, 21):
            b = i_grid / 20.0
            grid_out[f"{b:.2f}"] = {
                "threshold": _threshold_corrupted(conf, err, b),
                "random": (1 - b) * err_point_sub,
            }

        target_mode, effective_target = _resolve_target(target_corruption, err_point_sub)
        recommended_budget = _recommend_budget(conf, err, effective_target)
        regime, recommendation = _regime(auc_point, recommended_budget)

        realizable_out = None
        if n_groups_per_step is not None:
            rec_point, rec_upper, candidates = _recommend_realizable_budget(
                conf, err, err_point_sub, effective_target, n_groups_per_step, n_boot, col_seed)
            realizable_out = {
                "n_groups_per_step": n_groups_per_step,
                "pick": pick,
                "candidates": candidates,
                "recommended_point": rec_point,
                "recommended_upper": rec_upper,
                "recommended": rec_point if pick == "point" else rec_upper,
            }

        result["confidence_columns"][col] = {
            "n_used": int(m),
            "auc": {"point": auc_point, "ci95_lo": auc_lo, "ci95_hi": auc_hi},
            "budgets": budget_out,
            "grid": grid_out,
            "target_mode": target_mode,
            "effective_target": effective_target,
            "base_error": err_point_sub,
            "recommended_budget": recommended_budget,
            "regime": regime,
            "recommendation": recommendation,
            "realizable": realizable_out,
        }

    return result


def _grid_table_lines(c, indent=""):
    """Compact one-line-per-grid-budget table (threshold vs random corruption) from a
    confidence_columns[col] dict's "grid" field, for eyeballing an override. 20 grid points
    (0.05 to 1.0), so always <= 20 lines. Shared by diagnose.py's CLI table and
    p1_code_grpo.py's burn-in report, which both format the same diagnose() result."""
    lines = [f"{indent}grid (0.05 steps), threshold vs random corruption:"]
    for b, g in c["grid"].items():
        lines.append(f"{indent}  budget {b}: threshold {g['threshold']:.4f} vs random {g['random']:.4f}")
    return lines


def _realizable_table_lines(c, indent=""):
    """Candidate table for the realizable-budget rule (n_groups_per_step given): one line per
    k = 1..n_groups_per_step, plus both rules' recommendations. Shared by diagnose.py's CLI table
    and p1_code_grpo.py's burn-in report, same pattern as _grid_table_lines."""
    r = c["realizable"]
    lines = [f"{indent}realizable candidates (n_groups_per_step={r['n_groups_per_step']}, pick={r['pick']}):"]
    for cand in r["candidates"]:
        t_hi = cand["threshold_upper"]
        t_hi_s = f"{t_hi:.4f}" if np.isfinite(t_hi) else "nan"
        lines.append(
            f"{indent}  budget {cand['budget']:.4f} (k={cand['k']}): threshold point {cand['threshold_point']:.4f}, "
            f"threshold upper {t_hi_s}, random point {cand['random_point']:.4f}"
        )
    lines.append(f"{indent}recommended under point rule: {r['recommended_point']:.4f}")
    lines.append(f"{indent}recommended under upper rule: {r['recommended_upper']:.4f}")
    return lines


def _print_table(result):
    per = result["proxy_error_rate"]
    print(f"n groups: {result['n_groups']}")
    print(f"proxy error rate: {per['point']:.4f} (95% CI {per['ci95_lo']:.4f} to {per['ci95_hi']:.4f})")
    for col, c in result["confidence_columns"].items():
        print()
        print(f"confidence column: {col} (n_used={c['n_used']})")
        auc = c["auc"]
        if np.isnan(auc["point"]):
            print("  AUC: undefined (no error variance)")
        else:
            print(f"  AUC (low confidence predicts error): {auc['point']:.4f} (95% CI {auc['ci95_lo']:.4f} to {auc['ci95_hi']:.4f})")
        print(f"  regime: {c['regime']} - {c['recommendation']}")
        print(f"  target: {c['target_mode']} {c['effective_target']:.4f} (base error {c['base_error']:.4f})")
        print(f"  recommended budget for target corruption {result['target_corruption']}: {c['recommended_budget']:.2f}")
        for b, br in c["budgets"].items():
            t, r = br["threshold"], br["random"]
            print(
                f"  budget {b}: threshold corrupted {t['point']:.4f} (CI {t['ci95_lo']:.4f}-{t['ci95_hi']:.4f})"
                f" vs random {r['point']:.4f} (CI {r['ci95_lo']:.4f}-{r['ci95_hi']:.4f})"
                f", relative reduction {br['relative_reduction_vs_random']:.1%}"
                f", full-verification floor {br['full_verification_floor']:.2f}"
            )
        for line in _grid_table_lines(c, indent="  "):
            print(line)
        if c.get("realizable"):
            for line in _realizable_table_lines(c, indent="  "):
                print(line)


def main():
    ap = argparse.ArgumentParser(description="AVRL router regime diagnostic")
    ap.add_argument("--groups", required=True, help="path to a parquet file, one row per rollout group")
    ap.add_argument("--confidence", required=True, help="comma-separated confidence column name(s)")
    ap.add_argument("--error", default="proxy_error", help="column: 1 if cheap grader disagreed with expensive grader")
    ap.add_argument("--budgets", default="0.1,0.25,0.5", help="comma-separated budget fractions")
    ap.add_argument("--target_corruption", type=str, default=0.02,
                     help="target admitted-corruption rate: a float (absolute, default 0.02) "
                          "or 'rel:F' (F times the measured proxy error rate), e.g. rel:0.25")
    ap.add_argument("--n_boot", type=int, default=1000)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--n_groups_per_step", type=int, default=None,
                     help="if given, also recommend a budget realizable by a harness that "
                          "verifies k of this many groups per step (k = 1..n_groups_per_step)")
    ap.add_argument("--pick", choices=["point", "upper"], default="point",
                     help="with --n_groups_per_step: require the target under the point estimate "
                          "(default) or the bootstrap upper 95%% bound")
    ap.add_argument("--out", required=True, help="output directory for diagnosis.json")
    args = ap.parse_args()

    df = pd.read_parquet(args.groups)
    confidence_cols = args.confidence.split(",")
    budgets = [float(x) for x in args.budgets.split(",")]

    result = diagnose(df, confidence_cols, args.error, budgets, args.target_corruption, n_boot=args.n_boot, seed=args.seed,
                       n_groups_per_step=args.n_groups_per_step, pick=args.pick)
    _print_table(result)

    os.makedirs(args.out, exist_ok=True)
    out_path = os.path.join(args.out, "diagnosis.json")
    with open(out_path, "w") as f:
        json.dump(result, f, indent=2)
    print()
    print(f"wrote {out_path}")


if __name__ == "__main__":
    main()
