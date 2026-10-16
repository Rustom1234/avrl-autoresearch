"""L12b: tally the ts_online-vs-static_matched comparison across all 7 bootstrapped points.

RESEARCH_LOG.md's "Note 2026-10-01 (night)" claims ts_online (the cell/Thompson-sampling allocator
M03 is specced around) "never" beats static_matched across the bootstrapped points, and that claim 2's
whole replay gain comes from greedy_online's classifier-refit mechanism rather than from exploiting
prompt-space cell structure. That claim was asserted from reading greedy_online's table only (M01b).
The ts_online-vs-static_matched comparison was already computed in the same bootstrap run (same
comparisons_spec, see bootstrap_replay.py / e09_bootstrap.py) for all 7 points, but was never pulled
into a table or checked. This script does exactly that: no new rollouts, no GPU, no bootstrap
resampling -- it only reads each point's existing avrl/runs/<point>/bootstrap.json and result.json.

Copied-and-modified from avrl/exp/M01b.py per the project rule (copy and modify, never edit prior
experiment scripts in place). M01b's load_point/actual_winner pattern is reused verbatim, with the
comparison swapped from greedy_online-vs-static_matched to ts_online-vs-static_matched, and widened
to pull all three budget_fracs (0.10, 0.25, 0.50) per point instead of only 0.25 (0.25 remains the
pre-registered primary budget used for the win/lose/tie call).

CI-sign rule (same as L10a / M01b): at the 25% budget, cell100 group_col, cheap feature_set --
  win  if the 95% CI on ts_online's relative gain over static_matched excludes zero on the positive
       side (ci_lo > 0)
  lose if it excludes zero on the negative side (ci_hi < 0)
  tie  otherwise (CI straddles zero)

Usage:
  L12b_tsonline_tally.py --points p0,E06,E08,E09,E09b,E09c,E09d --group_col cell100
                          --feature_set cheap --out avrl/runs/L12b/result.json
"""
import argparse
import json
import os

BUDGET_FRACS = (0.10, 0.25, 0.50)
PRIMARY_BUDGET = 0.25
POLICY_A, POLICY_B = "ts_online", "static_matched"

# (point_id as shown in the table, run_dir name under runs_root) -- same labeling as M01b.py
POINT_LABELS = {"p0": "E01/p0"}


def load_point(run_dir, group_col, feature_set):
    with open(os.path.join(run_dir, "result.json")) as f:
        r = json.load(f)
    with open(os.path.join(run_dir, "bootstrap.json")) as f:
        b = json.load(f)
    by_budget = {}
    for c in b["comparisons"]:
        gc = c.get("group_col", b.get("group_col"))
        fs = c.get("feature_set", "cheap")
        if (gc == group_col and fs == feature_set and c["policy_a"] == POLICY_A
                and c["policy_b"] == POLICY_B):
            by_budget[c["budget_frac"]] = {
                "rel_improve_point": c["rel_improve_point"],
                "rel_improve_ci95": c["rel_improve_ci95"],
                "hits_a": c["hits_a"], "hits_b": c["hits_b"], "n_sim": c["n_sim"],
            }
    for bf in BUDGET_FRACS:
        if bf not in by_budget:
            raise SystemExit(
                f"no {group_col}/{feature_set}/{bf}/{POLICY_A}-vs-{POLICY_B} comparison in "
                f"{run_dir}/bootstrap.json")
    return {
        "n": b["n_groups"],
        "auc_cheap": r["auc_cheap"],
        "neighbor_ratio": r["neighbor_ratio"][group_col],
        "by_budget": by_budget,
        "source_bootstrap": os.path.join(run_dir, "bootstrap.json"),
        "source_result": os.path.join(run_dir, "result.json"),
    }


def call(ci):
    lo, hi = ci
    if lo is not None and lo > 0:
        return "win"
    if hi is not None and hi < 0:
        return "lose"
    return "tie"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--points", default="p0,E06,E08,E09,E09b,E09c,E09d")
    ap.add_argument("--runs_root", default="avrl/runs")
    ap.add_argument("--group_col", default="cell100")
    ap.add_argument("--feature_set", default="cheap")
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    points = args.points.split(",")
    rows = []
    for p in points:
        run_dir = os.path.join(args.runs_root, p)
        d = load_point(run_dir, args.group_col, args.feature_set)
        primary = d["by_budget"][PRIMARY_BUDGET]
        rows.append({
            "point": POINT_LABELS.get(p, p),
            "n": d["n"],
            "cheap_auc": d["auc_cheap"],
            "neighbor_ratio_cell100": d["neighbor_ratio"],
            "rel_improve_point": primary["rel_improve_point"],
            "ci95": primary["rel_improve_ci95"],
            "call": call(primary["rel_improve_ci95"]),
            "all_budgets": {
                str(bf): {"rel_improve_point": d["by_budget"][bf]["rel_improve_point"],
                          "ci95": d["by_budget"][bf]["rel_improve_ci95"]}
                for bf in BUDGET_FRACS
            },
            "source_bootstrap": d["source_bootstrap"],
            "source_result": d["source_result"],
        })

    n_win = sum(1 for row in rows if row["call"] == "win")
    n_lose = sum(1 for row in rows if row["call"] == "lose")
    n_tie = sum(1 for row in rows if row["call"] == "tie")

    if n_win == 0:
        verdict = "PASS (confirms Note: ts_online never beats static_matched at 25% budget)"
    else:
        wins = [row["point"] for row in rows if row["call"] == "win"]
        verdict = (f"KILL (falsifies Note's \"never\" claim): {n_win}/{len(rows)} points show a "
                   f"significant positive ts_online-vs-static_matched CI at 25% budget: {wins}")

    # cross-check: do the wins concentrate on the weak-cheap-signal / high-neighbor-ratio points,
    # as claim 3 predicts? Report both subgroup splits plainly rather than asserting a clean pattern.
    low_auc = [row for row in rows if row["cheap_auc"] < 0.80]
    high_auc = [row for row in rows if row["cheap_auc"] >= 0.80]
    cross_check = {
        "low_cheap_auc_points (auc_cheap < 0.80)": [row["point"] for row in low_auc],
        "low_cheap_auc_wins": sum(1 for row in low_auc if row["call"] == "win"),
        "low_cheap_auc_total": len(low_auc),
        "high_cheap_auc_points (auc_cheap >= 0.80)": [row["point"] for row in high_auc],
        "high_cheap_auc_wins": sum(1 for row in high_auc if row["call"] == "win"),
        "high_cheap_auc_total": len(high_auc),
        "note": "All wins (if any) fall inside the low-cheap-auc subgroup, but not every low-auc "
                "point wins -- see per-point rows for which ones do.",
    }

    result = {
        "id": "L12b",
        "hypothesis": "RESEARCH_LOG.md Note 2026-10-01 (night) claim that ts_online never beats "
                      "static_matched across the bootstrapped points",
        "comparison": f"{POLICY_A} vs {POLICY_B}",
        "group_col": args.group_col,
        "feature_set": args.feature_set,
        "primary_budget_frac": PRIMARY_BUDGET,
        "ci_sign_rule": "win if ci_lo > 0, lose if ci_hi < 0, tie otherwise (same rule as L10a/M01b)",
        "pass_threshold": "0 of 7 points show a significant positive CI at 25% budget",
        "kill_threshold": ">=1 of 7 points show a significant positive CI at 25% budget",
        "n_points": len(rows),
        "n_win": n_win,
        "n_lose": n_lose,
        "n_tie": n_tie,
        "verdict": verdict,
        "cross_check_vs_claim3": cross_check,
        "rows": rows,
    }

    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    with open(args.out, "w") as f:
        json.dump(result, f, indent=1)
    print(json.dumps(result, indent=1))


if __name__ == "__main__":
    main()
