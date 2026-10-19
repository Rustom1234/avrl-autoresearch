"""L12b: does the frozen 200-label table's stratum ordering transfer across the two row sets of the L4b pool?

Assumptions, stated up front:
  - stratum key = agreement x n_distinct x any_truncated, as in l5a_frozen_table.py (STRATUM) and l6a_test_rows.py.
  - error label = proxy_error (majority vote wrong against gold), as in L5a and L6a.
  - row sets are runs/L4b/feats.parquet split at pid <= --test_pid_max (registered test rows) and
    pid > --test_pid_max (remaining-train rows).
  - the strata are the p0 strata (runs/p0/feats.parquet). A stratum with zero docs on one side is left out of
    that comparison and listed.
  - per-stratum error rate = raw mean of proxy_error on that row set (not the frozen posterior).
  - reference values are read from the saved L13a per-cut results, not recomputed. No refit, no new arm,
    no rollouts. CPU only, no GPU, no ssh, no pip.

Usage: python -I l12b_stratum_transfer.py --p0 P0 --l4b L4B --test_pid_max 1318 --boot 1000 --out OUT_DIR
"""
import argparse
import json
import os
import warnings

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

STRATUM = ["agreement", "n_distinct", "any_truncated"]
LABEL = "proxy_error"
EXPECTED_TEST, EXPECTED_REMAINING = 1319, 1473
EXPECTED_P0_STRATA, EXPECTED_L4B_STRATA = 36, 35
PASS_ABOVE, KILL_BELOW = 0.8, 0.5
SENS_MIN_DOCS = 10
MIN_STRATA_FOR_RHO = 3
SEED = 12
L13A_DIR = "/homes/rdubash/avrl_live/ROUTED/avrl/runs/L13a"
REF_DIRS = {"test": "L4b_pid_le1318", "remaining": "L4b_pid_gt1318"}
LEDGER_REF_PP = {"test": 1.22, "remaining": 8.58}
REF_TOL = 0.005
VERDICT_TEXT = {
    "PASS": "PASS: stratum ordering transfers (rho above 0.8)",
    "KILL": "KILL: stratum ordering does not transfer (rho below 0.5)",
    "INCONCLUSIVE": "INCONCLUSIVE: rho between 0.5 and 0.8, recorded as neither pass nor kill",
    "UNDEFINED": "UNDEFINED: fewer than 3 shared strata",
}


def stratum_label(t):
    return f"agreement={t[0]:.4f} n_distinct={int(t[1])} any_truncated={int(t[2])}"


def codes_for(df, universe):
    return universe.get_indexer(pd.MultiIndex.from_frame(df[STRATUM]))


def per_stratum(codes, err, K):
    n = np.bincount(codes, minlength=K).astype(float)
    e = np.bincount(codes, weights=err, minlength=K)
    rate = np.full(K, np.nan)
    ok = n > 0
    rate[ok] = e[ok] / n[ok]
    return n, rate


def rho(x, y):
    if len(x) < MIN_STRATA_FOR_RHO:
        return float("nan")
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return float(spearmanr(x, y).correlation)


def band(r):
    if np.isnan(r):
        return "UNDEFINED"
    if r > PASS_ABOVE:
        return "PASS"
    if r < KILL_BELOW:
        return "KILL"
    return "INCONCLUSIVE"


def load_ref(l13a_dir, side):
    path = os.path.join(l13a_dir, REF_DIRS[side], "result.json")
    with open(path) as f:
        pool = json.load(f)["pools"]["L4b"]
    val = float(pool["frozen_vs_fixed_mean_pp"])
    return {"file": path, "frozen_minus_fixed_pp": val, "eval_rows": pool["eval_rows"],
            "ledger_pp": LEDGER_REF_PP[side], "matches_ledger": bool(abs(val - LEDGER_REF_PP[side]) <= REF_TOL)}


def clean(o):
    if isinstance(o, dict):
        return {k: clean(v) for k, v in o.items()}
    if isinstance(o, (list, tuple)):
        return [clean(v) for v in o]
    if isinstance(o, np.ndarray):
        return clean(o.tolist())
    if isinstance(o, np.generic):
        o = o.item()
    if isinstance(o, float) and not np.isfinite(o):
        return None
    return o


def fmt(x):
    if x is None or (isinstance(x, float) and np.isnan(x)):
        return "n/a"
    return f"{x:.3f}"


def write_table(path, out):
    p, b, g = out["primary"], out["primary"]["bootstrap"], out["gates"]
    s, ref = out["sensitivity"], out["reference_l13a"]
    L = [f"# L12b: stratum-ordering transfer across the two L4b row sets", "",
         f"**{out['verdict']}**", "",
         "Primary metric: Spearman rho between per-stratum proxy-error rates on the registered test rows "
         "(pid <= 1318) and the remaining-train rows (pid > 1318), over the strata present on both sides. "
         "Pass above 0.8, kill below 0.5, inconclusive in between.", "",
         "| metric | value |", "|---|---|",
         f"| primary rho | {fmt(p['rho'])} |",
         f"| 95% bootstrap CI ({b['draws']} prompt resamples per row set) | [{fmt(p['ci95'][0])}, {fmt(p['ci95'][1])}] |",
         f"| strata in primary | {out['strata']['both_sides']} |",
         f"| sensitivity rho (strata with >= {SENS_MIN_DOCS} docs on each side) | {fmt(s['rho'])} (n = {s['n_strata']}) |",
         f"| sensitivity band same as primary | {'yes' if s['same_band_as_primary'] else 'no'} ({s['band']}) |",
         f"| mean strata used per bootstrap draw | {b['mean_strata_per_draw']:.1f} |",
         f"| bootstrap draws with undefined rho | {b['undefined_draws']} |", "",
         "## Gates", "", "| gate | pass |", "|---|---|"]
    L += [f"| {k} | {'yes' if v else 'NO'} |" for k, v in g.items()]
    L += ["", "## Reference: saved L13a per-cut results (frozen minus fixed, pp)", "",
          "| row set | L13a file value | ledger | eval rows | matches ledger (tol 0.005) |", "|---|---|---|---|---|"]
    for side in ("test", "remaining"):
        r = ref[side]
        L.append(f"| {side} | {r['frozen_minus_fixed_pp']:+.2f} | {r['ledger_pp']:+.2f} | {r['eval_rows']} | "
                 f"{'yes' if r['matches_ledger'] else 'NO'} |")
    L += ["", "## Strata", "",
          f"- p0 strata: {out['strata']['p0_strata']}; present in L4b: {out['strata']['l4b_present']}",
          f"- absent from L4b: {', '.join(out['strata']['absent_from_l4b']) or 'none'}",
          f"- one side only (excluded from primary): {', '.join(out['strata']['one_side_only']) or 'none'}", "",
          "| agreement | n_distinct | any_truncated | p0 docs | test docs | test occupancy | test error rate | "
          "remaining docs | remaining occupancy | remaining error rate | in primary | in sensitivity |",
          "|---|---|---|---|---|---|---|---|---|---|---|---|"]
    for r in out["per_stratum"]:
        L.append(f"| {r['agreement']:.4f} | {r['n_distinct']} | {r['any_truncated']} | {r['p0_docs']} | "
                 f"{r['test_docs']} | {r['test_occupancy']:.3f} | {fmt(r['test_error_rate'])} | "
                 f"{r['remaining_docs']} | {r['remaining_occupancy']:.3f} | {fmt(r['remaining_error_rate'])} | "
                 f"{'yes' if r['in_primary'] else 'no'} | {'yes' if r['in_sensitivity'] else 'no'} |")
    with open(path, "w") as f:
        f.write("\n".join(L) + "\n")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--p0", required=True)
    ap.add_argument("--l4b", required=True)
    ap.add_argument("--test_pid_max", type=int, default=1318)
    ap.add_argument("--boot", type=int, default=1000)
    ap.add_argument("--out", required=True)
    ap.add_argument("--l13a_dir", default=L13A_DIR)
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    print(f"L12b: p0={a.p0} l4b={a.l4b} test_pid_max={a.test_pid_max} boot={a.boot} seed={SEED}", flush=True)

    p0 = pd.read_parquet(a.p0)
    l4 = pd.read_parquet(a.l4b)
    need = STRATUM + [LABEL, "pid"]
    gates = {
        "no_null_fields": bool(p0[need].isna().sum().sum() == 0 and l4[need].isna().sum().sum() == 0),
        "l4b_pid_unique": bool(l4["pid"].is_unique),
    }
    universe = pd.MultiIndex.from_frame(p0[STRATUM].drop_duplicates().sort_values(STRATUM).reset_index(drop=True))
    K = len(universe)
    test = l4[l4["pid"] <= a.test_pid_max].reset_index(drop=True)
    rem = l4[l4["pid"] > a.test_pid_max].reset_index(drop=True)
    gates["row_sets_cover_l4b"] = bool(len(test) + len(rem) == len(l4))
    gates["test_rows_1319"] = bool(len(test) == EXPECTED_TEST)
    gates["remaining_rows_1473"] = bool(len(rem) == EXPECTED_REMAINING)
    cp0, ct, cr = codes_for(p0, universe), codes_for(test, universe), codes_for(rem, universe)
    gates["every_row_in_one_p0_stratum"] = bool((cp0 >= 0).all() and (ct >= 0).all() and (cr >= 0).all())
    gates["p0_strata_36"] = bool(K == EXPECTED_P0_STRATA)
    # Rows matching no p0 stratum are dropped here; the gate above already records the failure.
    tm, rm = ct >= 0, cr >= 0
    test, ct = test[tm].reset_index(drop=True), ct[tm]
    rem, cr = rem[rm].reset_index(drop=True), cr[rm]
    et = test[LABEL].to_numpy(float)
    er = rem[LABEL].to_numpy(float)

    n_p0 = np.bincount(cp0[cp0 >= 0], minlength=K)
    n_t, r_t = per_stratum(ct, et, K)
    n_r, r_r = per_stratum(cr, er, K)
    present = (n_t + n_r) > 0
    gates["l4b_present_strata_35"] = bool(present.sum() == EXPECTED_L4B_STRATA)
    absent = [stratum_label(universe[i]) for i in np.flatnonzero(~present)]
    one_side = [stratum_label(universe[i]) for i in np.flatnonzero((n_t > 0) ^ (n_r > 0))]
    both = (n_t > 0) & (n_r > 0)
    sens = (n_t >= SENS_MIN_DOCS) & (n_r >= SENS_MIN_DOCS)
    print(f"gates: {gates}", flush=True)
    print(f"test rows {len(test)}, remaining rows {len(rem)}, p0 strata {K}, L4b strata present {int(present.sum())}, "
          f"absent {absent}, one side only {one_side}", flush=True)

    rho_primary = rho(r_t[both], r_r[both])
    rho_sens = rho(r_t[sens], r_r[sens])
    band_p, band_s = band(rho_primary), band(rho_sens)
    print(f"primary rho {rho_primary:.4f} over {int(both.sum())} strata ({band_p}); "
          f"sensitivity rho {rho_sens:.4f} over {int(sens.sum())} strata ({band_s})", flush=True)

    rng = np.random.default_rng(SEED)
    boot_rho = np.empty(a.boot)
    boot_k = np.empty(a.boot, dtype=int)
    for b in range(a.boot):
        it = rng.integers(0, len(ct), len(ct))
        ir = rng.integers(0, len(cr), len(cr))
        nt, rt = per_stratum(ct[it], et[it], K)
        nr, rr = per_stratum(cr[ir], er[ir], K)
        m = (nt > 0) & (nr > 0)
        boot_k[b] = int(m.sum())
        boot_rho[b] = rho(rt[m], rr[m])
    ok = ~np.isnan(boot_rho)
    ci = [float(np.percentile(boot_rho[ok], 2.5)), float(np.percentile(boot_rho[ok], 97.5))]
    print(f"bootstrap 95% CI [{ci[0]:.4f}, {ci[1]:.4f}] over {int(ok.sum())} defined draws", flush=True)

    refs = {side: load_ref(a.l13a_dir, side) for side in REF_DIRS}
    for side in refs:
        print(f"reference {side}: frozen minus fixed {refs[side]['frozen_minus_fixed_pp']:+.4f} pp "
              f"(ledger {LEDGER_REF_PP[side]:+.2f}, matches {refs[side]['matches_ledger']})", flush=True)

    rows = []
    for i in range(K):
        rows.append({
            "stratum": stratum_label(universe[i]), "agreement": float(universe[i][0]),
            "n_distinct": int(universe[i][1]), "any_truncated": int(universe[i][2]),
            "p0_docs": int(n_p0[i]),
            "test_docs": int(n_t[i]), "test_occupancy": n_t[i] / len(test), "test_error_rate": r_t[i],
            "remaining_docs": int(n_r[i]), "remaining_occupancy": n_r[i] / len(rem), "remaining_error_rate": r_r[i],
            "in_primary": bool(both[i]), "in_sensitivity": bool(sens[i]),
        })

    gates_pass = all(gates.values())
    out = {
        "experiment": "L12b",
        "hypothesis": ("The frozen 200-label table's stratum ordering (36 strata of agreement x n_distinct x "
                       "any_truncated, fitted on p0) transfers across the two row sets of the L4b pool."),
        "label": LABEL, "stratum": STRATUM, "scope": "rollout-side strata; does not test prompt-space structure",
        "inputs": {"p0": a.p0, "l4b": a.l4b, "test_pid_max": a.test_pid_max, "boot": a.boot, "seed": SEED,
                   "l13a_dir": a.l13a_dir},
        "row_sets": {"test": {"rows": len(test), "rule": f"pid <= {a.test_pid_max}"},
                     "remaining": {"rows": len(rem), "rule": f"pid > {a.test_pid_max}"}},
        "strata": {"p0_strata": K, "l4b_present": int(present.sum()), "absent_from_l4b": absent,
                   "both_sides": int(both.sum()), "one_side_only": one_side,
                   "sensitivity_min_docs_per_side": SENS_MIN_DOCS},
        "gates": gates, "gates_pass": bool(gates_pass),
        "primary": {"metric": "spearman_rho_per_stratum_error_rate", "rho": rho_primary, "ci95": ci,
                    "n_strata": int(both.sum()), "band": band_p, "pass_above": PASS_ABOVE, "kill_below": KILL_BELOW,
                    "bootstrap": {"draws": a.boot, "seed": SEED, "unit": "prompts, resampled within each row set",
                                  "undefined_draws": int((~ok).sum()), "mean_strata_per_draw": float(boot_k.mean())}},
        "sensitivity": {"rho": rho_sens, "n_strata": int(sens.sum()), "band": band_s,
                        "same_band_as_primary": bool(band_s == band_p)},
        "reference_l13a": refs,
        "verdict_band": band_p,
        "verdict": VERDICT_TEXT[band_p] if gates_pass else "GATE FAIL: no verdict recorded",
        "per_stratum": rows,
    }
    with open(os.path.join(a.out, "result.json"), "w") as f:
        json.dump(clean(out), f, indent=2)
    write_table(os.path.join(a.out, "table.md"), clean(out))
    print(f"verdict: {out['verdict']}", flush=True)
    return 0 if gates_pass else 2


if __name__ == "__main__":
    raise SystemExit(main())
