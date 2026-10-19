"""Independent check of runs/L18a/result.json. Recomputes every per-split gap from the saved hit counts only (raw_hits).

Imports nothing from the experiment modules (numpy and json only). It checks that:
  - each split holds one hit count per tie seed, within 0 to budget, for the fixed, table and greedy arms;
  - greedy_strata minus the 3,000-label table, table minus plain fixed, and greedy minus plain fixed, per split, match the
    values in result.json to 1e-9 pp;
  - the pool mean, sd (ddof 1), count above +1.0 and verdict match result.json;
  - table minus plain fixed per split matches runs/L5a/result.json to 1e-6 pp on both pools (the comparator check).
Writes <out>/indep_check.json and prints a log. Exit 0 if every check passes, else 1.

Usage: python -I l18a_check.py --result runs/L18a/result.json --out runs/L18a
"""
import os, sys, json, argparse
import numpy as np

PASS_PP = 1.0
TOL_PP = 1e-9
L5A_TOL_PP = 1e-6
L5A_REF = "/homes/rdubash/avrl_live/ROUTED/avrl/runs/L5a/result.json"


def pct(hits, budget):
    return 100.0 * np.asarray(hits, float) / budget


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--result", required=True)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    r = json.load(open(a.result))
    splits = [str(s) for s in r["splits"]]; n_seeds = int(r["tie_seeds"])
    l5a = json.load(open(L5A_REF))["pools"]

    log, ok = [], True
    shape_ok = True
    for pool, raw in r["raw_hits"].items():
        for s in splits:
            x = raw[s]; b = x["budget"]
            for arm in ("fixed_hits", "frozen_hits", "greedy_hits"):
                h = x[arm]
                if len(h) != n_seeds or min(h) < 0 or max(h) > b:
                    shape_ok = False
    ok = ok and shape_ok
    log.append(f"hit-count shape: {n_seeds} tie seeds per split and arm, 0 <= hits <= budget: {'PASS' if shape_ok else 'FAIL'}")

    per = {}
    for pool, raw in r["raw_hits"].items():
        g_t, t_f, g_f = [], [], []
        for s in splits:
            x = raw[s]; b = x["budget"]
            g = pct(x["greedy_hits"], b); z = pct(x["frozen_hits"], b); f = pct(x["fixed_hits"], b)
            g_t.append(float((g - z).mean())); t_f.append(float((z - f).mean())); g_f.append(float((g - f).mean()))
        m = np.asarray(g_t, float)
        per[pool] = {"greedy_minus_table": g_t, "table_minus_fixed": t_f, "greedy_minus_fixed": g_f,
                     "mean_pp": float(m.mean()), "sd_pp": float(m.std(ddof=1)), "n_above": int((m > PASS_PP).sum()),
                     "verdict": "PASS" if m.mean() > PASS_PP else "KILL"}

    checks = {}
    for pool, c in per.items():
        R = r["pools"][pool]
        d_gt = max(abs(p - q) for p, q in zip(c["greedy_minus_table"], R["greedy_minus_table_pp_per_split"]))
        d_zf = max(abs(p - q) for p, q in zip(c["table_minus_fixed"], R["table_minus_fixed_pp_per_split"]))
        d_gx = max(abs(p - q) for p, q in zip(c["greedy_minus_fixed"], R["greedy_minus_fixed_pp_per_split"]))
        d_mean = abs(c["mean_pp"] - R["mean_pp"]); d_sd = abs(c["sd_pp"] - R["sd_pp"])
        verdict_ok = c["verdict"] == R["verdict"] and c["n_above"] == R["n_splits_above_pass"]
        l5 = l5a[pool]["gap_pp_per_split"]
        d_l5a = max(abs(p - float(q)) for p, q in zip(c["table_minus_fixed"], l5))
        checks[pool] = {"max_abs_diff_greedy_minus_table_pp": d_gt, "max_abs_diff_table_minus_fixed_pp": d_zf,
                        "max_abs_diff_greedy_minus_fixed_pp": d_gx, "abs_diff_mean_pp": d_mean, "abs_diff_sd_pp": d_sd,
                        "verdict_and_count_match": verdict_ok, "table_minus_fixed_vs_L5a_max_abs_diff_pp": d_l5a,
                        "table_minus_fixed_mean_recomputed_pp": float(np.mean(c["table_minus_fixed"])),
                        "L5a_mean_pp": float(l5a[pool]["mean_pp"]),
                        "pass": bool(max(d_gt, d_zf, d_gx) <= TOL_PP and d_mean <= TOL_PP and d_sd <= TOL_PP and verdict_ok
                                     and d_l5a <= L5A_TOL_PP)}
        ok = ok and checks[pool]["pass"]
        log.append(f"{pool}: recomputed greedy_strata minus 3,000-label table mean {c['mean_pp']:+.4f} pp, sd {c['sd_pp']:.4f}, "
                   f"splits above +1.0 {c['n_above']} of {len(splits)}, verdict {c['verdict']}")
        log.append(f"{pool}: max abs diff vs result.json greedy minus table {d_gt:.2e} pp, table minus fixed {d_zf:.2e} pp, "
                   f"greedy minus fixed {d_gx:.2e} pp, mean {d_mean:.2e}, sd {d_sd:.2e}, verdict and count match {verdict_ok}")
        log.append(f"{pool}: table minus plain fixed, recomputed from hits, mean {checks[pool]['table_minus_fixed_mean_recomputed_pp']:+.4f} "
                   f"pp; runs/L5a/result.json mean {checks[pool]['L5a_mean_pp']:+.4f} pp; max abs diff per split vs L5a "
                   f"{d_l5a:.2e} pp (tol {L5A_TOL_PP:g})")
        log.append(f"{pool}: independent check {'PASS' if checks[pool]['pass'] else 'FAIL'}")

    verdict = "PASS" if all(v == "PASS" for v in (c["verdict"] for c in per.values())) else "KILL"
    log.append(f"overall verdict recomputed from hits: {verdict}; result.json verdict: {r['verdict']}; "
               f"match {verdict == r['verdict']}")
    ok = ok and verdict == r["verdict"]
    log.append(f"independent check overall: {'PASS' if ok else 'FAIL'}")

    out = {"check": "L18a independent recomputation from raw_hits", "source": a.result, "per_pool": per, "checks": checks,
           "overall_verdict_recomputed": verdict, "overall_check_pass": bool(ok)}
    with open(os.path.join(a.out, "indep_check.json"), "w") as fh: json.dump(out, fh, indent=1)
    print("\n".join(log), flush=True)
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
