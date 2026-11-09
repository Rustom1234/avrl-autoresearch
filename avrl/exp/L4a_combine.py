"""L4a: stack L3a's 25%-budget result alongside the two new budget_frac
points (10%, 50%) run with the same byte-identical e03b_bootstrap_matched.py
script, into one result.json for the RESEARCH_LOG entry.

No new analysis: pulls the three already-computed result.json files and
reports each budget's point diff, bootstrap CI, and verdict side by side.
"""
import argparse, json


def load(path):
    d = json.load(open(path))
    return {
        "budget_frac": d["budget_frac"],
        "hits_static_matched": d["point_estimate"]["hits_static_matched_mean8seeds"],
        "hits_ts_online": d["point_estimate"]["hits_ts_online_mean8seeds"],
        "total_label1_in_sim": d["point_estimate"]["total_label1_in_sim"],
        "static_matched_rate": d["point_estimate"]["static_matched_corrupted_admitted_rate"],
        "ts_online_rate": d["point_estimate"]["learned_corrupted_admitted_rate"],
        "point_diff": d["point_estimate"]["diff_static_matched_minus_learned"],
        "ci95_lo": d["bootstrap"]["ci95_lo"],
        "ci95_hi": d["bootstrap"]["ci95_hi"],
        "verdict": d["verdict"],
        "ci_excludes_zero": d["bootstrap"]["ci95_lo"] > 0 or d["bootstrap"]["ci95_hi"] < 0,
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--b10", required=True)
    ap.add_argument("--b25", required=True)
    ap.add_argument("--b50", required=True)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    rows = {"0.10": load(args.b10), "0.25": load(args.b25), "0.50": load(args.b50)}
    any_significant = any(r["ci_excludes_zero"] for r in rows.values())
    out = {
        "id": "L4a",
        "hypothesis": "claim 2: does ts_online beat static_matched at the budget tails (10%, 50%) that bracket L3a's 25% null",
        "budgets": rows,
        "any_budget_ci_excludes_zero": any_significant,
        "verdict": "NO_SUPPORT_ANY_BUDGET" if not any_significant else "MIXED_SEE_BUDGETS",
    }
    json.dump(out, open(args.out, "w"), indent=1)
    print(json.dumps(out, indent=1))


if __name__ == "__main__":
    main()
