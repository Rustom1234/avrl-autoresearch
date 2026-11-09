"""AR1 analysis: experiments-to-verdict and cost per planner-context arm.

Reads avrl/runs/AR1/<arm>_s<seed>/result.json and trace.jsonl, prints a table
per arm (mean and per-seed values), the pre-registered reads, and a seed-paired
sign comparison of each arm against arm A. Writes avrl/runs/AR1/result.json.
"""
import json
import statistics as st
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
D = ROOT / "runs" / "AR1"


def load():
    runs = {}
    for p in sorted(D.glob("*_s*/result.json")):
        r = json.loads(p.read_text())
        trace = [json.loads(l) for l in (p.parent / "trace.jsonl").read_text().splitlines()]
        r["analyst_agreement"] = sum(1 for t in trace if t["verdict"] == t["header_verdict"]) / max(1, len(trace))
        r["final_correct"] = r["final_claims"].get("c1") == "pass" and r["final_claims"].get("c2") in ("kill", "tie") and r["final_claims"].get("c3") == "kill"
        runs.setdefault(r["arm"], []).append(r)
    return runs


def etv(r, max_iters=20):
    return r["experiments_to_verdict"] if r["experiments_to_verdict"] is not None else max_iters + 1


def main():
    runs = load()
    summary = {}
    print("| arm | seeds | experiments to verdict (censored at 21) | censored | flips | final correct | cost USD | planner input tokens | analyst agreement with real verdicts |")
    print("|---|---|---|---|---|---|---|---|---|")
    for arm in sorted(runs):
        rs = sorted(runs[arm], key=lambda r: r["seed"])
        e = [etv(r) for r in rs]
        row = {
            "n": len(rs), "etv_mean": st.mean(e), "etv_per_seed": e, "censored": sum(r["censored"] for r in rs),
            "flips": sum(r["flips"] for r in rs), "final_correct": sum(r["final_correct"] for r in rs),
            "cost_mean": st.mean(r["cost_usd"] for r in rs), "planner_tokens_mean": st.mean(r["planner_input_tokens"] for r in rs),
            "agreement_mean": st.mean(r["analyst_agreement"] for r in rs), "orders": {r["seed"]: r["order"] for r in rs},
        }
        summary[arm] = row
        print(f"| {arm} | {row['n']} | {row['etv_mean']:.1f} {row['etv_per_seed']} | {row['censored']} | {row['flips']} | {row['final_correct']}/{row['n']} | {row['cost_mean']:.2f} | {row['planner_tokens_mean']:.0f} | {row['agreement_mean']:.2f} |")

    if "A" in runs:
        a = {r["seed"]: r for r in runs["A"]}
        for arm in ("B", "C"):
            if arm not in runs:
                continue
            pairs = [(etv(a[r["seed"]]), etv(r)) for r in runs[arm] if r["seed"] in a]
            wins = sum(1 for x, y in pairs if y < x)
            ties = sum(1 for x, y in pairs if y == x)
            summary[arm]["vs_A"] = {"fewer_experiments": wins, "ties": ties, "n": len(pairs)}
            print(f"{arm} vs A, seed-paired: fewer experiments in {wins} of {len(pairs)} seeds ({ties} ties)")
    if "B" in runs and "C" in runs:
        b = {r["seed"]: r for r in runs["B"]}
        pairs = [(etv(b[r["seed"]]), etv(r)) for r in runs["C"] if r["seed"] in b]
        wins = sum(1 for x, y in pairs if y < x)
        print(f"C vs B, seed-paired: fewer experiments in {wins} of {len(pairs)} seeds ({sum(1 for x, y in pairs if x == y)} ties)")
        summary["C"]["vs_B"] = {"fewer_experiments": wins, "n": len(pairs)}

    # Pre-registered reads
    if {"A", "B", "C"} <= set(summary):
        A, B, C = summary["A"], summary["B"], summary["C"]
        read1 = (B["etv_mean"] < A["etv_mean"] and B["cost_mean"] < A["cost_mean"]) or (C["etv_mean"] < A["etv_mean"] and C["cost_mean"] < A["cost_mean"])
        read2 = C["etv_mean"] < B["etv_mean"]
        print(f"Read 1 (B or C beats A on experiments-to-verdict at lower cost): {'PASS' if read1 else 'NOT MET'}")
        print(f"Read 2 (C beats B on experiments-to-verdict): {'PASS' if read2 else 'NOT MET'}")
        summary["reads"] = {"read1_pass": read1, "read2_pass": read2}
    (D / "result.json").write_text(json.dumps(summary, indent=1))


if __name__ == "__main__":
    main()
