"""M01: regime diagnostic, standalone validation across the nine existing ledger points.

Rule under test (lit/avrl/EXPERIMENT_BACKLOG.md, "M01" section): a point's own
already-measured cheap-feature AUC for proxy error decides a regime label.
  auc_cheap >= 0.85  -> STRONG regime -> predict the (budget-matched) threshold
                         allocator wins the allocation-replay comparison.
  auc_cheap <  0.85  -> WEAK regime   -> predict a learned/structure-aware
                         allocator wins the allocation-replay comparison.
This script reads each point's result.json (no new rollouts, no recomputation),
applies the cutoff, and checks the predicted winner against the winner already
recorded in the ledger (avrl/RESEARCH_LOG.md, avrl/PROJECT.md) for that point.

Nine points and where their numbers and their recorded winner come from:
  p0   (E01)        avrl/runs/p0/result.json           auc_cheap top-level field
  L5a  (E04/L5a)     avrl/runs/L5a/result.json          auc_cheap top-level field (no feats.parquet for this point; L5a's own result.json already carries it)
  E05                avrl/runs/E05/result.json          auc_cheap top-level field (no feats.parquet; selfverify-based result.json)
  E06                avrl/runs/E06/result.json          auc_cheap top-level field
  E08                avrl/runs/E08/result.json          auc_cheap top-level field
  E09                avrl/runs/E09/result.json          auc_cheap top-level field
  E09b               avrl/runs/E09b/result.json         auc_cheap top-level field
  E09c               avrl/runs/E09c/result.json         auc_cheap top-level field
  E09d               avrl/runs/E09d/result.json         auc_cheap top-level field

Recorded ("actual") winner per point, per the rule stated in the task spec
(kill/tie in the replay bootstrap = threshold wins; bootstrap CI excludes zero
on the positive side = allocator wins), read directly from RESEARCH_LOG.md:
  p0/E01 : threshold  - RESEARCH_LOG.md "## L4a" (budget-matched static vs
           ts_online, 10/25/50% budgets, bootstrap CI includes zero at all
           three -> KILL/tie each time); "## L3a" is the 25%-budget case of
           the same test.
  L5a    : threshold  - RESEARCH_LOG.md "## L7b" (judge_only ranking vs
           agreement_only, NO_SUPPORT, agreement/threshold strictly ahead at
           every budget); EXPERIMENT_BACKLOG.md priority note: "GSM8K is
           settled across four proxies ... the agreement threshold dominates
           every alternative at every budget."
  E05    : threshold  - RESEARCH_LOG.md "## L6a" (sv_agreement/combined_flag
           ranking vs agreement_only, NO_SUPPORT, agreement/threshold ahead
           at every budget).
  E06    : threshold  - no separate replay bootstrap was run on E06's own
           label; grouped with E01/L5a/E05 under the same qualitative
           verdict in EXPERIMENT_BACKLOG.md's priority note (same quote as
           L5a above). Flagged as the weakest-evidenced ground-truth cell.
  E08    : threshold  - RESEARCH_LOG.md "## E08" allocation-replay paragraph:
           the learned online allocators (greedy_online, ts_online) "sit in
           between and never clear the free-label static rule"; no separate
           bootstrap significance test exists for this point either, same
           caveat as E06.
  E09    : allocator  - RESEARCH_LOG.md "## L9a", PASS: greedy_online vs
           static_matched at 25% budget, cell100/cheap, 95% CI on relative
           gain [+3.9%, +25.5%], excludes zero positive.
  E09b   : allocator  - RESEARCH_LOG.md "## L9b", PASS: 25% budget CI
           [+5.3%, +26.2%].
  E09c   : threshold  - RESEARCH_LOG.md "## L9c", INCONCLUSIVE: 25% budget CI
           [-5.7%, +33.7%], includes zero (tie, per the task's own tie rule).
  E09d   : allocator  - RESEARCH_LOG.md "## L9d", PASS: 25% budget CI
           [+7.8%, +26.1%].

Usage:
  .venv/bin/python avrl/exp/M01.py --runs_root avrl/runs --out avrl/runs/M01/result.json
"""
import argparse
import json
import os

AUC_CUTOFF = 0.85

# (point_id, run_dir name under runs_root, actual winner, citation, confidence note)
POINTS = [
    ("E01/p0", "p0", "threshold",
     "RESEARCH_LOG.md ## L3a, ## L4a (static_matched vs ts_online, CI includes zero at 10/25/50%)",
     "bootstrap-backed"),
    ("E04/L5a", "L5a", "threshold",
     "RESEARCH_LOG.md ## L7b (judge_only/combined vs agreement_only, NO_SUPPORT at every budget); "
     "EXPERIMENT_BACKLOG.md priority note (\"agreement threshold dominates every alternative\")",
     "qualitative (no budget-matched bootstrap on this point's own label)"),
    ("E05", "E05", "threshold",
     "RESEARCH_LOG.md ## L6a (sv_agreement/combined_flag vs agreement_only, NO_SUPPORT at every budget)",
     "qualitative (no budget-matched bootstrap on this point's own label)"),
    ("E06", "E06", "threshold",
     "EXPERIMENT_BACKLOG.md priority note (\"agreement threshold dominates every alternative\"); "
     "no separate replay bootstrap exists for E06's own label",
     "qualitative, weakest-evidenced (grouped by inference, not its own replay test)"),
    ("E08", "E08", "threshold",
     "RESEARCH_LOG.md ## E08 (learned online allocators \"never clear the free-label static rule\")",
     "qualitative (no budget-matched bootstrap on this point's own label)"),
    ("E09", "E09", "allocator",
     "RESEARCH_LOG.md ## L9a, PASS, 25% budget CI [+3.9%, +25.5%] (greedy_online vs static_matched, cell100/cheap)",
     "bootstrap-backed"),
    ("E09b", "E09b", "allocator",
     "RESEARCH_LOG.md ## L9b, PASS, 25% budget CI [+5.3%, +26.2%]",
     "bootstrap-backed"),
    ("E09c", "E09c", "threshold",
     "RESEARCH_LOG.md ## L9c, INCONCLUSIVE, 25% budget CI [-5.7%, +33.7%] (includes zero -> tie)",
     "bootstrap-backed"),
    ("E09d", "E09d", "allocator",
     "RESEARCH_LOG.md ## L9d, PASS, 25% budget CI [+7.8%, +26.1%]",
     "bootstrap-backed"),
]


def load_point_stats(run_dir):
    with open(os.path.join(run_dir, "result.json")) as f:
        r = json.load(f)
    auc = r["auc_cheap"]
    nr100 = r["neighbor_ratio"]["cell100"]
    return auc, nr100


def diagnose(auc_cheap):
    regime = "STRONG" if auc_cheap >= AUC_CUTOFF else "WEAK"
    predicted = "threshold" if regime == "STRONG" else "allocator"
    return regime, predicted


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs_root", default="avrl/runs")
    ap.add_argument("--out", default="avrl/runs/M01/result.json")
    args = ap.parse_args()

    rows = []
    n_match = 0
    for point_id, run_dir_name, actual_winner, citation, confidence in POINTS:
        run_dir = os.path.join(args.runs_root, run_dir_name)
        auc_cheap, neighbor_ratio_cell100 = load_point_stats(run_dir)
        regime, predicted_winner = diagnose(auc_cheap)
        match = predicted_winner == actual_winner
        n_match += int(match)
        rows.append({
            "point_id": point_id,
            "run_dir": run_dir_name,
            "auc_cheap": auc_cheap,
            "neighbor_ratio_cell100": neighbor_ratio_cell100,
            "regime_label": regime,
            "predicted_winner": predicted_winner,
            "actual_winner": actual_winner,
            "match": match,
            "ground_truth_confidence": confidence,
            "citation": citation,
        })

    n_total = len(rows)
    n_mismatch = n_total - n_match
    result = {
        "id": "M01",
        "hypothesis": "two-statistic regime diagnostic (auc_cheap >= 0.85 cutoff) "
                      "correctly labels which allocation policy won, across the nine "
                      "ledger points",
        "auc_cutoff": AUC_CUTOFF,
        "n_total": n_total,
        "n_match": n_match,
        "n_mismatch": n_mismatch,
        "pass_threshold": "n_match >= 8 of 9",
        "kill_threshold": "n_mismatch >= 3 of 9",
        "verdict": (
            "pass" if n_match >= 8 else
            "kill" if n_mismatch >= 3 else
            "inconclusive"
        ),
        "rows": rows,
    }

    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    with open(args.out, "w") as f:
        json.dump(result, f, indent=1)

    print(json.dumps(result, indent=1))


if __name__ == "__main__":
    main()
