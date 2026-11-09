"""L13a combine (CPU only, no new computation): merges the per-pool L13a results into <out>/result.json and <out>/table.md.

Copied in spirit from avrl/exp/l10b_combine.py. Per full pool: table_shrunk minus the 200-label frozen table (the verdict), table_shrunk
minus table_zero (does shrinkage remove the refit cost), and table_shrunk minus the L10b table_refit (paired per subset; the subset
draws and the frozen arm must match L10b, else this is an error). The registered L4b pid cuts (pid <= 1318 test rows, pid > 1318
remaining train rows) are reported beside the full pools and are not claim verdicts. Overall PASS only if both full pools are above
+1.0 pp. A result with a failed gate stops the combine.

Usage: python -I l13a_combine.py --pool_dirs OUT_L2b OUT_L4b --cut_dirs OUT_CUT_LE1318 OUT_CUT_GT1318 --out OUTDIR
"""
import os, sys, json, argparse
import numpy as np

L10B = "/homes/rdubash/avrl_live/ROUTED/avrl/runs/L10b/result.json"
PASS_PP = 1.0


def load(d):
    r = json.load(open(os.path.join(d, "result.json")))
    if r.get("verdict") == "GATE_FAILED" or "pools" not in r or not r["gates"]["pass"]:
        sys.exit(f"{d}: verdict {r.get('verdict')}, gates pass {r.get('gates', {}).get('pass')}; no combine")
    return r


def paired_vs_l10b(name, det, l10):
    """table_shrunk minus L10b table_refit, per subset, after checking that the subset draws and frozen arm match L10b."""
    d10 = {x["subset"]: x for x in l10["subset_detail"]}
    draws = all(d10[x["subset"]]["row_positions"] == x["row_positions"] for x in det)
    frozen_err = max(abs(d10[x["subset"]][f"{name}_frozen_minus_fixed_pp"] - x[f"{name}_frozen_minus_fixed_pp"]) for x in det)
    if not draws or frozen_err > 1e-9:
        sys.exit(f"{name}: subset draws or frozen arm differ from L10b; the paired comparison is not valid")
    paired = np.asarray([x[f"{name}_table_shrunk_minus_frozen_pp"] - d10[x["subset"]][f"{name}_table_refit_minus_frozen_pp"]
                         for x in det], float)
    return {"same_subset_draws": bool(draws), "frozen_arm_max_abs_diff_vs_L10b_pp": float(frozen_err),
            "mean_pp": float(paired.mean()), "sd_pp": float(paired.std(ddof=1)), "per_subset_pp": [float(v) for v in paired]}


def summarise(d, r, name, l10):
    p = r["pools"][name]
    row = {"dir": d, "scope": r["scope"], "pid_cut": r["pid_cut"], "verdict": p["verdict"], "mean_pp": p["mean_pp"],
           "sd_pp": p["sd_pp"], "n_subsets": p["n_subsets"], "n_subsets_above_pass": p["n_subsets_above_pass"],
           "min_subset_mean_pp": p["min_subset_mean_pp"], "max_subset_mean_pp": p["max_subset_mean_pp"],
           "shrunk_minus_zero_pp": p["shrunk_minus_zero_mean_pp"], "zero_minus_frozen_pp": p["zero_minus_frozen_mean_pp"],
           "frozen_minus_fixed_pp": p["frozen_vs_fixed_mean_pp"], "shrunk_minus_fixed_pp": p["shrunk_vs_fixed_mean_pp"],
           "fixed_pct_mean": p["fixed_pct_mean"], "frozen_pct_mean": p["frozen_pct_mean"], "shrunk_pct_mean": p["shrunk_pct_mean"],
           "n_split_gaps_above_pass": p["n_split_gaps_above_pass"], "full_table_3000_mean_pp": p["full_table_3000_mean_pp"]}
    if l10 is not None:
        row["vs_L10b_table_refit"] = paired_vs_l10b(name, r["subset_detail"], l10)
        q = l10["pools"][name]
        row["L10b"] = {"table_refit_minus_frozen_pp": q["mean_pp"], "table_refit_sd_pp": q["sd_pp"],
                       "table_refit_subsets_above_pass": q["n_subsets_above_pass"], "table_zero_minus_frozen_pp": q["zero_minus_frozen_mean_pp"],
                       "table_refit_minus_table_zero_pp": q["mean_pp"] - q["zero_minus_frozen_mean_pp"], "verdict": q["verdict"]}
    return row


def write_table(out_dir, out):
    L = ["# L13a: offset-ridge refit anchored at the matched 160-label table, equal labels, 10% budget", "",
         f"Overall verdict: **{out['verdict']}** (PASS needs both full pools above +1.0 pp). Gates pass: {out['gates_pass']}.", "",
         "## Full pools (the verdict)", "",
         "| pool | table_shrunk minus frozen 200-label (pp) | sd over subset means | subsets above +1.0 | min to max subset mean (pp) | table_shrunk minus table_zero (pp) | table_zero minus frozen (pp) | verdict |",
         "|---|---|---|---|---|---|---|---|"]
    for n in ("L2b", "L4b"):
        s = out["full_pools"][n]
        L.append(f"| {n} | {s['mean_pp']:+.2f} | {s['sd_pp']:.2f} | {s['n_subsets_above_pass']} of {s['n_subsets']} | "
                 f"{s['min_subset_mean_pp']:+.2f} to {s['max_subset_mean_pp']:+.2f} | {s['shrunk_minus_zero_pp']:+.2f} | "
                 f"{s['zero_minus_frozen_pp']:+.2f} | {s['verdict']} |")
    L += ["", "## Against L10b (same subsets, frozen arm and tie seeds)", "",
          "| pool | L13a table_shrunk minus frozen (pp) | L10b table_refit minus frozen (pp) | table_shrunk minus L10b table_refit, paired (pp) | sd of paired change | L10b table_refit minus table_zero (pp) | L13a table_shrunk minus table_zero (pp) |",
          "|---|---|---|---|---|---|---|"]
    for n in ("L2b", "L4b"):
        s = out["full_pools"][n]; v = s["vs_L10b_table_refit"]; b = s["L10b"]
        L.append(f"| {n} | {s['mean_pp']:+.2f} | {b['table_refit_minus_frozen_pp']:+.2f} | {v['mean_pp']:+.2f} | {v['sd_pp']:.2f} | "
                 f"{b['table_refit_minus_table_zero_pp']:+.2f} | {s['shrunk_minus_zero_pp']:+.2f} |")
    L += ["", "## Registered L4b pid cuts (reported beside the full pools, not claim verdicts)", "",
          "| cut | table_shrunk minus frozen (pp) | sd | subsets above +1.0 | min to max | table_shrunk minus table_zero (pp) | table_zero minus frozen (pp) | frozen minus fixed (pp) |",
          "|---|---|---|---|---|---|---|---|"]
    for k, s in out["cuts"].items():
        L.append(f"| {k} | {s['mean_pp']:+.2f} | {s['sd_pp']:.2f} | {s['n_subsets_above_pass']} of {s['n_subsets']} | "
                 f"{s['min_subset_mean_pp']:+.2f} to {s['max_subset_mean_pp']:+.2f} | {s['shrunk_minus_zero_pp']:+.2f} | "
                 f"{s['zero_minus_frozen_pp']:+.2f} | {s['frozen_minus_fixed_pp']:+.2f} |")
    L += ["", "Gate detail, per run: see result.json (gates) and each run.log.", ""]
    open(os.path.join(out_dir, "table.md"), "w").write("\n".join(L))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pool_dirs", nargs=2, required=True)
    ap.add_argument("--cut_dirs", nargs=2, required=True)
    ap.add_argument("--out", required=True)
    a = ap.parse_args(); os.makedirs(a.out, exist_ok=True)
    l10 = json.load(open(L10B))
    full, cuts = {}, {}
    for d in a.pool_dirs:
        r = load(d)
        for name in r["pools"]:
            full[name] = summarise(d, r, name, l10)
    for d in a.cut_dirs:
        r = load(d)
        c = r["pid_cut"]; name = next(iter(r["pools"]))
        if c["min"] is None:
            key = f"{name} pid <= {c['max']} (registered test rows)"
        else:
            key = f"{name} pid >= {c['min']} (remaining train rows)"
        cuts[key] = summarise(d, r, name, None)
    gates_pass = len(full) == 2
    verdict = "PASS" if all(full[n]["verdict"] == "PASS" for n in full) else "KILL"
    out = {"experiment": "L13a", "verdict": verdict, "verdicts": {n: full[n]["verdict"] for n in full},
           "gates_pass": bool(gates_pass), "full_pools": full, "cuts": cuts,
           "rule": "per full pool, table_shrunk minus the 200-label frozen table, mean over 20 subsets; PASS if above +1.0 pp on both pools",
           "L10b_reference": L10B}
    with open(os.path.join(a.out, "result.json"), "w") as fh: json.dump(out, fh, indent=1)
    write_table(a.out, out)
    for n in full:
        s = full[n]; v = s["vs_L10b_table_refit"]
        print(f"{n}: table_shrunk minus frozen {s['mean_pp']:+.2f} (sd {s['sd_pp']:.2f}, {s['n_subsets_above_pass']} of 20 above +1.0), "
              f"table_shrunk minus table_zero {s['shrunk_minus_zero_pp']:+.2f}, table_shrunk minus L10b table_refit {v['mean_pp']:+.2f} "
              f"(L10b table_refit minus frozen {s['L10b']['table_refit_minus_frozen_pp']:+.2f}), verdict {s['verdict']}")
    for k, s in cuts.items():
        print(f"cut {k}: table_shrunk minus frozen {s['mean_pp']:+.2f} (sd {s['sd_pp']:.2f}, {s['n_subsets_above_pass']} of 20 above +1.0), "
              f"table_shrunk minus table_zero {s['shrunk_minus_zero_pp']:+.2f}, frozen minus fixed {s['frozen_minus_fixed_pp']:+.2f}")
    print(f"L13a combined verdict: {verdict} (per pool {out['verdicts']})")


if __name__ == "__main__":
    main()
