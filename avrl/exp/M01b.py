"""M01b: rebuild the regime-diagnostic table from M01.py using ONLY bootstrap-backed ground truth.
Copied-and-modified from M01.py per the project rule (copy and modify, never edit prior experiment
scripts in place). M01 used a mix of bootstrap CIs and qualitative citations for "actual winner";
M01b restricts to the points that have their own avrl/runs/<point>/bootstrap.json (greedy_online vs
static_matched at cell100/cheap features, same methodology across all of them since E01/E06/E08/E09
were produced by avrl/exp/bootstrap_replay.py and E09b/E09c/E09d by avrl/exp/e09_bootstrap.py, both
using sim_generic.py's policy code verbatim).

actual winner (per point, at the 25% budget, cell100/cheap):
  allocator  if the 95% CI on the greedy_online-vs-static_matched relative gain excludes zero on the
             positive side (CI lower bound > 0)
  threshold  if it excludes zero on the negative side (CI upper bound < 0)
  tie        otherwise (CI includes zero)

predicted winner: same 0.85 cheap-AUC cutoff as M01 (auc_cheap >= 0.85 -> threshold, else allocator).

No new rollouts, no GPU, CPU-only, reads existing result.json/bootstrap.json only.

Usage:
  M01b.py --runs_root avrl/runs --log avrl/RESEARCH_LOG.md
"""
import argparse
import datetime
import json
import os

AUC_CUTOFF = 0.85

# (point_id, run_dir name under runs_root)
POINTS = [
    ("E01/p0", "p0"),
    ("E06", "E06"),
    ("E08", "E08"),
    ("E09", "E09"),
    ("E09b", "E09b"),
    ("E09c", "E09c"),
    ("E09d", "E09d"),
]


def load_point(run_dir):
    with open(os.path.join(run_dir, "result.json")) as f:
        r = json.load(f)
    with open(os.path.join(run_dir, "bootstrap.json")) as f:
        b = json.load(f)
    comp = None
    for c in b["comparisons"]:
        if (c.get("group_col", b.get("group_col")) == "cell100" and abs(c.get("budget_frac", -1) - 0.25) < 1e-9
                and c["policy_a"] == "greedy_online" and c["policy_b"] == "static_matched"
                and c.get("feature_set", "cheap") == "cheap"):
            comp = c
            break
    if comp is None:
        raise SystemExit(f"no cell100/cheap/25%/greedy_online-vs-static_matched comparison in {run_dir}/bootstrap.json")
    return {
        "n": b["n_groups"],
        "auc_cheap": r["auc_cheap"],
        "neighbor_ratio_cell100": r["neighbor_ratio"]["cell100"],
        "rel_gain": comp["rel_improve_point"],
        "rel_ci": comp["rel_improve_ci95"],
    }


def actual_winner(rel_ci):
    lo, hi = rel_ci
    if lo is not None and lo > 0:
        return "allocator"
    if hi is not None and hi < 0:
        return "threshold"
    return "tie"


def predicted_winner(auc_cheap):
    return "threshold" if auc_cheap >= AUC_CUTOFF else "allocator"


def best_auc_cutoff(rows):
    """Brute-force the AUC cutoff (rule: predict allocator-win iff auc_cheap < cutoff) that best
    separates actual allocator wins from non-wins (threshold or tie), by accuracy. Candidates are
    midpoints between sorted unique AUC values, plus the extremes."""
    aucs = sorted(set(r["auc_cheap"] for r in rows))
    candidates = [aucs[0] - 0.01] + [(a + b) / 2 for a, b in zip(aucs, aucs[1:])] + [aucs[-1] + 0.01]
    is_allocator_win = [r["actual"] == "allocator" for r in rows]
    best_cut, best_acc = None, -1.0
    results = []
    for cut in candidates:
        pred_win = [r["auc_cheap"] < cut for r in rows]
        acc = sum(p == a for p, a in zip(pred_win, is_allocator_win)) / len(rows)
        results.append((cut, acc))
        if acc > best_acc:
            best_cut, best_acc = cut, acc
    return best_cut, best_acc, results


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs_root", default="avrl/runs")
    ap.add_argument("--log", default="avrl/RESEARCH_LOG.md")
    args = ap.parse_args()

    rows = []
    for point_id, run_dir_name in POINTS:
        run_dir = os.path.join(args.runs_root, run_dir_name)
        p = load_point(run_dir)
        actual = actual_winner(p["rel_ci"])
        predicted = predicted_winner(p["auc_cheap"])
        match = predicted == actual if actual != "tie" else None
        rows.append({
            "point_id": point_id,
            "n": p["n"],
            "auc_cheap": p["auc_cheap"],
            "neighbor_ratio_cell100": p["neighbor_ratio_cell100"],
            "rel_gain": p["rel_gain"],
            "rel_ci": p["rel_ci"],
            "predicted": predicted,
            "actual": actual,
            "match": match,
        })

    n_total = len(rows)
    n_match = sum(1 for r in rows if r["match"] is True)
    n_mismatch = sum(1 for r in rows if r["match"] is False)
    n_tie = sum(1 for r in rows if r["actual"] == "tie")

    match_frac = n_match / n_total
    mismatch_frac = n_mismatch / n_total
    pass_frac, kill_frac = 8 / 9, 3 / 9
    if match_frac >= pass_frac:
        verdict = "pass"
    elif mismatch_frac >= kill_frac:
        verdict = "kill"
    else:
        verdict = "inconclusive"

    best_cut, best_acc, cutoff_results = best_auc_cutoff(rows)

    result = {
        "id": "M01b",
        "hypothesis": "regime diagnostic (auc_cheap >= 0.85 cutoff), re-tested against bootstrap-backed "
                      "ground truth only (greedy_online vs static_matched, cell100/cheap, 25% budget CI)",
        "auc_cutoff": AUC_CUTOFF,
        "n_total": n_total,
        "n_match": n_match,
        "n_mismatch": n_mismatch,
        "n_tie": n_tie,
        "match_frac": match_frac,
        "pass_frac_required": pass_frac,
        "kill_frac_required": kill_frac,
        "verdict": verdict,
        "best_auc_cutoff": best_cut,
        "best_auc_cutoff_accuracy": best_acc,
        "rows": rows,
    }
    print(json.dumps(result, indent=1))

    # ledger entry
    today = datetime.date.today().isoformat()
    lines = []
    lines.append(f"## M01b {today} {verdict}")
    lines.append("")
    lines.append("Hypothesis: same regime diagnostic as M01 (cheap-feature AUC >= 0.85 -> predict threshold "
                 "wins, else predict allocator wins), re-tested using ONLY bootstrap-backed ground truth: "
                 "for every point with its own avrl/runs/<point>/bootstrap.json, actual winner is read off "
                 "the 95% CI on the greedy_online-vs-static_matched relative gain at the 25% budget "
                 "(cell100, cheap features) -- allocator if the CI excludes zero on the positive side, "
                 "threshold if it excludes zero on the negative side, tie otherwise. This removes the "
                 "qualitative/inferred ground truth that M01's critic note flagged as unreliable for E06 "
                 "and E08 (RESEARCH_LOG.md ## M01), by running avrl/exp/bootstrap_replay.py (E01/p0, E06, "
                 "E08, E09) and reusing avrl/exp/e09_bootstrap.py's existing output (E09b, E09c, E09d); "
                 "E04/L5a and E05 are dropped (no feats.parquet, so no bootstrap is possible).")
    lines.append("")
    lines.append(f"Run: no new rollouts, no GPU, CPU-only. Script avrl/exp/M01b.py (new file, does not edit "
                 f"M01.py or either bootstrap script) reads each point's result.json and bootstrap.json "
                 f"under avrl/runs/<point>/, pulls the cell100/cheap/25%-budget greedy_online-vs-"
                 f"static_matched comparison, and applies the rules above. {n_total} points qualify "
                 f"(E01/p0, E06, E08, E09, E09b, E09c, E09d).")
    lines.append("")
    lines.append("| point | n | cheap AUC | neighbor ratio (cell100) | 25% bootstrap rel. gain [95% CI] | predicted | actual | match |")
    lines.append("|---|---|---|---|---|---|---|---|")
    for r in rows:
        lo, hi = r["rel_ci"]
        ci_str = f"[{lo:+.1%}, {hi:+.1%}]" if lo is not None and hi is not None else "[n/a]"
        match_str = "MATCH" if r["match"] is True else "MISMATCH" if r["match"] is False else "TIE (no call)"
        lines.append(f"| {r['point_id']} | {r['n']} | {r['auc_cheap']:.3f} | {r['neighbor_ratio_cell100']:.3f} | "
                      f"{r['rel_gain']:+.1%} {ci_str} | {r['predicted']} | {r['actual']} | {match_str} |")
    lines.append("")
    lines.append(f"Verdict against M01's pre-registered rule (pass if >= 8/9 match; kill if >= 3/9 mismatch; "
                 f"with {n_total} points instead of 9, apply the same proportion: pass requires "
                 f"match_frac >= {pass_frac:.3f}, kill requires mismatch_frac >= {kill_frac:.3f}): "
                 f"{n_match}/{n_total} match ({match_frac:.1%}), {n_mismatch}/{n_total} mismatch "
                 f"({mismatch_frac:.1%}), {n_tie}/{n_total} tie. Result: {verdict.upper()}.")
    lines.append("")
    lines.append(f"AUC-cutoff note: brute-forcing the cutoff on these {n_total} points (rule: predict "
                 f"allocator-win iff auc_cheap < cutoff, scored against actual=='allocator' as the positive "
                 f"class) finds best accuracy {best_acc:.1%} at cutoff {best_cut:.3f} "
                 f"(vs the pre-registered 0.85). " +
                 ("The data-driven cutoff agrees with 0.85 to within rounding." if abs(best_cut - AUC_CUTOFF) < 0.02
                  else f"The data-driven cutoff ({best_cut:.3f}) differs from the pre-registered 0.85, "
                       f"which is weak evidence either way given only {n_total} points and ties that no "
                       f"single cutoff can resolve (a cutoff only distinguishes allocator-win from "
                       f"not-allocator-win; it cannot separate threshold-win from tie)."))
    entry = "\n".join(lines) + "\n"

    with open(args.log, "a") as f:
        f.write("\n" + entry)
    print("\n--- appended to", args.log, "---\n")
    print(entry)


if __name__ == "__main__":
    main()
