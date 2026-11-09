"""L10b combine (CPU only, no new computation): merges the per-pool L10b results into <out>/result.json and <out>/table.md.

Copied from avrl/exp/l9a_combine.py. Changes: imports the L10b module, the matched zero-refit gate, and a paired comparison
with L9a on the same 20 subsets (the subset draws and the frozen arm must match L9a before the comparison is reported).

Usage: python -I l10b_combine.py --pool_dirs OUT_L2b OUT_L4b --out OUTDIR
Overall verdict: PASS only if both pools PASS (mean over subset means above +1.0 pp). A pool with a failed gate makes the
overall verdict GATE_FAILED, and no table is written.
"""
import os, sys, json, argparse
import numpy as np
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE)
import l10b_matched_precision as L10

L9A_RESULT = "/homes/rdubash/avrl_live/ROUTED/avrl/runs/L9a/result.json"
GATE_DICTS = ("fixed_reproduction", "frozen_3000_reproduction_L5a", "frozen_subset_reproduction_L6b", "matched_zero_gate")
COUNT_GATES = ("spend_gate", "table_spend_gate")
FLAG_GATES = ("fixed_identical_across_tables", "subset_distinct_prompts", "eval_question_overlap_zero")


def l9a_compare(name, detail, l9):
    """Paired comparison with L9a on the same subsets. The subset draws and the frozen arm must match L9a, else this is an error."""
    d9 = {x["subset"]: x for x in l9["subset_detail"]}
    draws_match = all(d9[x["subset"]]["row_positions"] == x["row_positions"] for x in detail)
    frozen_err = max(abs(d9[x["subset"]][f"{name}_frozen_minus_fixed_pp"] - x[f"{name}_frozen_minus_fixed_pp"]) for x in detail)
    paired = np.asarray([x[f"{name}_table_refit_minus_frozen_pp"] - d9[x["subset"]][f"{name}_table_refit_minus_frozen_pp"]
                         for x in detail], float)
    return {"same_subset_draws": bool(draws_match), "frozen_arm_max_abs_diff_vs_L9a_pp": float(frozen_err),
            "paired_change_mean_pp": float(paired.mean()), "paired_change_sd_pp": float(paired.std(ddof=1)),
            "paired_change_per_subset_pp": [float(v) for v in paired]}


def write_table(out_dir, out):
    P = out["pools"]; names = list(P); G = out["gates"]; R9 = out["l9a_reference"]
    L = ["# L10b: matched-precision control for L9a, equal labels, 10% budget", "",
         "Rule: per pool, table_refit minus the frozen 200-label subset table, mean over 20 subsets (each subset mean over split seeds "
         "0 to 7, each split over 8 tie seeds). PASS only if the mean is above +1.0 pp on that pool; overall PASS needs both pools.", "",
         "| pool | L10b table_refit minus frozen (pp) | L10b sd over subset means | L10b subsets above +1.0 | L10b min to max subset mean (pp) | L10b verdict |",
         "|---|---|---|---|---|---|"]
    for n in names:
        p = P[n]
        L.append(f"| {n} | {p['mean_pp']:+.2f} | {p['sd_pp']:.2f} | {p['n_subsets_above_pass']} of {p['n_subsets']} | "
                 f"{p['min_subset_mean_pp']:+.2f} to {p['max_subset_mean_pp']:+.2f} | {p['verdict']} |")
    L += ["", "Beside the L9a numbers (same subsets, same frozen arm, same tie seeds):", "",
          "| pool | L9a table_refit minus frozen (pp) | L9a sd | L9a subsets above +1.0 | L10b minus L9a, paired per subset (pp) | sd of paired change | L9a verdict |",
          "|---|---|---|---|---|---|---|"]
    for n in names:
        r9 = R9[n]; c = out["l9a_compare"][n]
        L.append(f"| {n} | {r9['mean_pp']:+.2f} | {r9['sd_pp']:.2f} | {r9['n_subsets_above_pass']} of {r9['n_subsets']} | "
                 f"{c['paired_change_mean_pp']:+.2f} | {c['paired_change_sd_pp']:.2f} | {r9['verdict']} |")
    L += ["", "Context, same subsets and tie seeds:", "",
          "| pool | table_zero minus frozen (matched fold posterior, refits off) (pp) | frozen 200-label minus plain fixed (pp) | "
          "table_refit minus plain fixed (pp) | L9a table_refit minus plain fixed (pp) | plain fixed hit rate (%) | table_refit hit rate (%) |",
          "|---|---|---|---|---|---|---|"]
    for n in names:
        p = P[n]
        L.append(f"| {n} | {p['zero_minus_frozen_mean_pp']:+.2f} | {p['frozen_vs_fixed_mean_pp']:+.2f} | {p['learned_vs_fixed_mean_pp']:+.2f} | "
                 f"{R9[n]['learned_vs_fixed_mean_pp']:+.2f} | {p['fixed_pct_mean']:.2f} | {p['learned_pct_mean']:.2f} |")
    L += ["", "Gates (all must pass before a verdict):", "", "| gate | result |", "|---|---|"]
    for k in GATE_DICTS:
        L.append(f"| {k} | {'PASS' if all(v['pass'] for v in G[k].values()) else 'FAIL'} |")
    for k in COUNT_GATES:
        L.append(f"| {k} | {'PASS' if G[k]['pass'] else 'FAIL'} ({G[k]['checks']} checks, {G[k]['failures']} failures) |")
    for k in FLAG_GATES:
        L.append(f"| {k} | {'PASS' if G[k] else 'FAIL'} |")
    L += ["", "Assumptions (not in the spec): eval fold draw RandomState(400 + i).randint(0, 5) per eval row, one draw per pool; "
          "the zero-refit gate is replaced by the matched gate (table_zero equals frozen_pick on the matched score), because the "
          "L9a zero-refit gate cannot hold once the init is the fold table. The check step recomputes the means from result.json.", ""]
    with open(os.path.join(out_dir, "table.md"), "w") as fh: fh.write("\n".join(L))


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
        out = {"experiment": "L10b", "verdict": "GATE_FAILED", "gates": gates, "eval_question_overlap_with_p0": overlap,
               "pool_result_files": [os.path.join(d, "result.json") for d in a.pool_dirs]}
        L10.write_json(a.out, out)
        print("GATE FAILED: no combined L10b verdict written", flush=True); sys.exit(1)

    out = dict(parts[0])  # settings are identical across pools
    pools = {k: v for p in parts for k, v in p["pools"].items()}
    detail = {}
    for p in parts:
        for s in p["subset_detail"]: detail.setdefault(s["subset"], {}).update(s)
    subset_detail = [detail[i] for i in sorted(detail)]
    l9 = json.load(open(L9A_RESULT))
    l9_ref = {n: {k: l9["pools"][n][k] for k in ("mean_pp", "sd_pp", "n_subsets", "n_subsets_above_pass", "learned_vs_fixed_mean_pp",
                                                 "frozen_vs_fixed_mean_pp", "verdict")} for n in pools}
    compare = {n: l9a_compare(n, subset_detail, l9) for n in pools}
    for n in pools:
        assert compare[n]["same_subset_draws"], f"{n}: subset draws differ from L9a"
        assert compare[n]["frozen_arm_max_abs_diff_vs_L9a_pp"] < 1e-9, f"{n}: frozen arm differs from L9a"
    out.update({"gates": gates, "eval_question_overlap_with_p0": overlap, "pools": pools,
                "verdicts": {n: pools[n]["verdict"] for n in pools}, "subset_detail": subset_detail,
                "l9a_reference": l9_ref, "l9a_compare": compare, "l9a_result_file": L9A_RESULT,
                "pool_result_files": [os.path.join(d, "result.json") for d in a.pool_dirs]})
    out["verdict"] = "PASS" if all(v == "PASS" for v in out["verdicts"].values()) else "KILL"
    L10.write_json(a.out, out); write_table(a.out, out)
    for n in pools:
        p = pools[n]
        print(f"{n}: table_refit minus frozen 200-label {p['mean_pp']:+.2f} pp (sd over subset means {p['sd_pp']:.2f}, "
              f"{p['n_subsets_above_pass']} of {p['n_subsets']} subsets above +1.0), verdict {p['verdict']}; L9a {l9_ref[n]['mean_pp']:+.2f}; "
              f"paired change vs L9a {compare[n]['paired_change_mean_pp']:+.2f}", flush=True)
    print(f"L10b combined verdict: {out['verdict']} (per pool {out['verdicts']})", flush=True)
    print(f"wrote {os.path.join(a.out, 'result.json')} and table.md", flush=True)


if __name__ == "__main__":
    main()
