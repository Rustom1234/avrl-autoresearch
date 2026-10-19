"""L9b: descriptive gate audit on the existing GSM8K majority-vote groups (CPU only, no generation).

For each policy family (E01, E10) and each agreement cutoff, report the confident-subset group count n,
proxy-error count k, point rate k/n, a Wilson 95% interval, and two flags: rate >= 0.03 (the gate, applied
as E10 applied it) and k >= 100 (the error floor). Reads only the two groups.parquet files; writes result.json.
Pre-registered in avrl/RESEARCH_LOG.md, block "L9b pre-registration 2026-10-18".
"""
import argparse, json, math, os
import numpy as np
import pandas as pd

Z = 1.959963984540054
GATE_RATE = 0.03
ERROR_FLOOR = 100
G = 8
# Logged confident-subset values (agreement >= 0.75) that the 0.75 row must reproduce.
LOGGED_075 = {"E01": {"n": 1551, "rate": 0.01096067053513862}, "E10": {"n": 2158, "k": 35, "rate": 0.016218721037998145}}


def wilson(k, n):
    p = k / n
    denom = 1 + Z * Z / n
    centre = (p + Z * Z / (2 * n)) / denom
    half = Z * math.sqrt(p * (1 - p) / n + Z * Z / (4 * n * n)) / denom
    return [centre - half, centre + half]


def load_checked(path):
    df = pd.read_parquet(path, columns=["pid", "gold", "majority", "agreement", "proxy_error", "majority_correct"])
    assert len(df) == 3000 and df.pid.is_unique, f"{path}: expected 3000 unique prompts, got {len(df)}"
    eighths = df.agreement * G
    assert (eighths == np.round(eighths)).all(), f"{path}: agreement is not a multiple of 1/{G}"
    assert set(df.proxy_error.unique()) <= {0, 1}, f"{path}: proxy_error not 0/1"
    recomputed = (df.majority != df.gold).astype(int)
    checks = {
        "n_groups": int(len(df)),
        "mismatch_proxy_error_vs_raw_majority_gold": int((recomputed != df.proxy_error).sum()),
        "mismatch_proxy_error_vs_1_minus_majority_correct": int((df.proxy_error != 1 - df.majority_correct).sum()),
        "proxy_error_total": int(df.proxy_error.sum()),
    }
    return df, checks


def audit_family(df, cutoffs):
    rows = []
    for c in cutoffs:
        sub = df[df.agreement >= c]
        n, k = len(sub), int(sub.proxy_error.sum())
        rate = k / n if n else float("nan")
        rows.append({
            "cutoff": c, "n": n, "k": k, "rate": rate,
            "wilson95": wilson(k, n) if n else [float("nan"), float("nan")],
            "flag_rate_ge_3pct": bool(n and rate >= GATE_RATE),
            "flag_k_ge_100": bool(k >= ERROR_FLOOR),
            "meets_both": bool(n and rate >= GATE_RATE and k >= ERROR_FLOOR),
        })
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--p0", required=True)
    ap.add_argument("--e10", required=True)
    ap.add_argument("--cutoffs", nargs="+", type=float, required=True)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)

    inputs = {"E01": a.p0, "E10": a.e10}
    families, cells, ok = {}, [], True
    for fam, path in inputs.items():
        df, checks = load_checked(path)
        rows = audit_family(df, a.cutoffs)
        families[fam] = {"source": path, "checks": checks, "rows": rows}
        ok &= checks["mismatch_proxy_error_vs_raw_majority_gold"] == 0 and checks["mismatch_proxy_error_vs_1_minus_majority_correct"] == 0
        cells += [(fam, r["cutoff"]) for r in rows if r["meets_both"]]
        if fam in LOGGED_075:
            r75 = next((r for r in rows if r["cutoff"] == 0.75), None)
            if r75 is not None:
                logged = LOGGED_075[fam]
                match = r75["n"] == logged["n"] and abs(r75["rate"] - logged["rate"]) < 1e-9
                if "k" in logged:
                    match = match and r75["k"] == logged["k"]
                families[fam]["logged_075_check"] = {"logged": logged, "computed_n": r75["n"], "computed_k": r75["k"],
                                                    "computed_rate": r75["rate"], "match": bool(match)}
                ok &= bool(match)

    gate_unreachable = len(cells) == 0
    result = {
        "id": "L9b", "metric": "per family and cutoff: n, k, point rate, Wilson 95%, flags (rate >= 0.03, k >= 100)",
        "gate_rate": GATE_RATE, "error_floor": ERROR_FLOOR, "cutoffs": a.cutoffs, "G": G,
        "families": families,
        "cells_meeting_both_flags": [{"family": f, "cutoff": c} for f, c in cells],
        "gate_unreachable_on_both_families": bool(gate_unreachable),
        "checks_passed": bool(ok),
        "verdict": ("descriptive: hypothesis holds, no cutoff meets both flags on either family; "
                    "claim 2 has no pre-registered verdict on GSM8K majority vote at G 8"
                    if gate_unreachable else
                    "descriptive: hypothesis does not hold; cells meeting both flags are a finding for a new "
                    "pre-registration (needs Rustom's decision on a gate); no verdict changes"),
    }
    with open(os.path.join(a.out, "result.json"), "w") as fh:
        json.dump(result, fh, indent=2)

    print(f"checks_passed={ok} gate_unreachable={gate_unreachable}")
    for fam in inputs:
        print(f"\n{fam}  {families[fam]['checks']}")
        print(f"{'cutoff':>7} {'n':>6} {'k':>5} {'rate':>8} {'wilson95':>22} {'rate>=3%':>9} {'k>=100':>7} {'both':>5}")
        for r in families[fam]["rows"]:
            lo, hi = r["wilson95"]
            print(f"{r['cutoff']:>7.3f} {r['n']:>6} {r['k']:>5} {r['rate']:>8.4f} {f'[{lo:.4f}, {hi:.4f}]':>22} "
                  f"{str(r['flag_rate_ge_3pct']):>9} {str(r['flag_k_ge_100']):>7} {str(r['meets_both']):>5}")
        if "logged_075_check" in families[fam]:
            print(f"logged 0.75 check: {families[fam]['logged_075_check']}")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
