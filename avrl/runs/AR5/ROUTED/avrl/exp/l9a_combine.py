"""L9a combine (CPU only, no new computation): merges the per-pool L9a results into <out>/result.json and <out>/table.md.

Copied from avrl/exp/l7a_combine.py. Changes: imports the L9a module, adds the zero-refit gate, and renames the L9a arm.

Usage: python -I l9a_combine.py --pool_dirs OUT_L2b OUT_L4b --out OUTDIR
Overall verdict: PASS only if both pools PASS (mean over subset means above +1.0 pp). A pool with a failed gate makes the
overall verdict GATE_FAILED, and no table is written.
"""
import os, sys, json, argparse
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE)
import l9a_table_init_learned as L9

GATE_DICTS = ("fixed_reproduction", "frozen_3000_reproduction_L5a", "frozen_subset_reproduction_L6b", "zero_refit_reproduction")
COUNT_GATES = ("spend_gate", "table_spend_gate")
FLAG_GATES = ("fixed_identical_across_tables", "subset_distinct_prompts", "eval_question_overlap_zero")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pool_dirs", nargs="+", required=True)
    ap.add_argument("--out", required=True)
    a = ap.parse_args(); os.makedirs(a.out, exist_ok=True)
    parts = [json.load(open(os.path.join(d, "result.json"))) for d in a.pool_dirs]
    gates = {k: {} for k in GATE_DICTS}
    for p in parts:
        for k in GATE_DICTS: gates[k].update(p["gates"][k])
    for k in COUNT_GATES:
        gates[k] = {"checks": sum(p["gates"][k]["checks"] for p in parts), "failures": sum(p["gates"][k]["failures"] for p in parts),
                    "examples": [e for p in parts for e in p["gates"][k]["examples"]][:5],
                    "pass": all(bool(p["gates"][k]["pass"]) for p in parts)}
    for k in FLAG_GATES: gates[k] = all(bool(p["gates"][k]) for p in parts)
    gates["pass"] = all(bool(p["gates"]["pass"]) for p in parts)
    overlap = {k: v for p in parts for k, v in p["eval_question_overlap_with_p0"].items()}
    if any(p["verdict"] == "GATE_FAILED" for p in parts) or not gates["pass"]:
        out = {"experiment": "L9a", "verdict": "GATE_FAILED", "gates": gates, "eval_question_overlap_with_p0": overlap,
               "pool_result_files": [os.path.join(d, "result.json") for d in a.pool_dirs]}
        L9.write_json(a.out, out)
        print("GATE FAILED: no combined L9a verdict written", flush=True); sys.exit(1)

    out = dict(parts[0])  # settings are identical across pools
    pools = {k: v for p in parts for k, v in p["pools"].items()}
    detail = {}
    for p in parts:
        for s in p["subset_detail"]: detail.setdefault(s["subset"], {}).update(s)
    out.update({"gates": gates, "eval_question_overlap_with_p0": overlap, "pools": pools,
                "verdicts": {n: pools[n]["verdict"] for n in pools},
                "subset_detail": [detail[i] for i in sorted(detail)],
                "pool_result_files": [os.path.join(d, "result.json") for d in a.pool_dirs]})
    out["verdict"] = "PASS" if all(v == "PASS" for v in out["verdicts"].values()) else "KILL"
    L9.write_json(a.out, out); L9.write_table(a.out, out)
    for n in pools:
        p = pools[n]
        print(f"{n}: table_refit minus frozen 200-label {p['mean_pp']:+.2f} pp (sd over subset means {p['sd_pp']:.2f}, "
              f"{p['n_subsets_above_pass']} of {p['n_subsets']} subsets above +1.0), verdict {p['verdict']}", flush=True)
    print(f"L9a combined verdict: {out['verdict']} (per pool {out['verdicts']})", flush=True)
    print(f"wrote {os.path.join(a.out, 'result.json')} and table.md", flush=True)


if __name__ == "__main__":
    main()
