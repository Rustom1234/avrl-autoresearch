"""AR6 analysis: planner routing policies, experiments-to-verdict per policy, seed-paired comparisons, pre-registered reads.

Reads runs/AR6/<tag_>POLICY_s<seed>/result.json. Block 1 has no tag (Haiku cheap tier); block 2 tags are the open
model names (open7b, open14b). Writes runs/AR6/result_<block>.json and prints the tables.
"""
import json
import re
import statistics as st
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
D = ROOT / "runs" / "AR6"
POLICIES = ["HAIKU", "SONNET", "CONSIST3", "VERBAL", "STALL", "STRONG_EARLY", "RANDOM25"]
CAP = 21


def load(block):
    runs = {}
    pat = re.compile(r"^(?:(?P<tag>[a-z0-9]+)_)?(?P<pol>[A-Z_0-9]+)_s(?P<seed>\d+)$")
    for p in D.glob("*_s*/result.json"):
        m = pat.match(p.parent.name)
        if not m:
            continue
        tag = m.group("tag") or "haiku"
        if tag != block:
            continue
        r = json.loads(p.read_text())
        runs.setdefault(r["policy"], {})[r["seed"]] = r
    return runs


def etv(r):
    return r["experiments_to_verdict"] if r["experiments_to_verdict"] is not None else CAP


def boot_mean_ci(x, B=10000, seed=0):
    rng = np.random.default_rng(seed)
    x = np.asarray(x, float)
    if len(x) < 2:
        return (float(x.mean()), float(x.mean()))
    bs = rng.choice(x, (B, len(x)), replace=True).mean(axis=1)
    return (float(np.percentile(bs, 5)), float(np.percentile(bs, 95)))


def paired(a, b):
    """seeds in both; returns wins of b over a (fewer experiments), ties, n, mean diff b-a with 90 percent bootstrap CI."""
    seeds = sorted(set(a) & set(b))
    d = [etv(b[s]) - etv(a[s]) for s in seeds]
    if not d:
        return {}
    return {"n": len(seeds), "b_fewer": sum(1 for x in d if x < 0), "ties": sum(1 for x in d if x == 0), "mean_diff": st.mean(d), "ci90": boot_mean_ci(d)}


def main():
    for block in ("haiku", "open7b", "open14b"):
        runs = load(block)
        if not runs:
            continue
        print(f"\n===== block {block} (cheap tier = {block}) =====")
        print("| policy | seeds | experiments to verdict mean [90% CI] | per seed | censored | strong-pick fraction | cost per run USD | flips |")
        print("|---|---|---|---|---|---|---|---|")
        summ = {}
        for pol in POLICIES:
            rs = runs.get(pol, {})
            if not rs:
                continue
            e = [etv(rs[s]) for s in sorted(rs)]
            lo, hi = boot_mean_ci(e)
            row = {"n": len(e), "etv_mean": st.mean(e), "etv_ci90": [lo, hi], "etv_per_seed": e, "censored": sum(rs[s]["censored"] for s in rs),
                   "strong_frac": st.mean(rs[s]["strong_pick_fraction"] for s in rs), "cost": st.mean(rs[s]["cost_usd"] for s in rs), "flips": sum(rs[s]["flips"] for s in rs)}
            summ[pol] = row
            print(f"| {pol} | {row['n']} | {row['etv_mean']:.1f} [{lo:.1f}, {hi:.1f}] | {e} | {row['censored']} | {row['strong_frac']:.2f} | {row['cost']:.2f} | {row['flips']} |")
        comps = {}
        if "SONNET" in runs:
            for pol in POLICIES:
                if pol in runs and pol != "SONNET":
                    comps[f"{pol}_vs_SONNET"] = paired(runs["SONNET"], runs[pol])
        for a, b in (("CONSIST3", "STALL"), ("VERBAL", "STALL"), ("RANDOM25", "STALL"), ("RANDOM25", "CONSIST3"), ("RANDOM25", "VERBAL"), ("HAIKU", "STALL")):
            if a in runs and b in runs:
                comps[f"{b}_vs_{a}"] = paired(runs[a], runs[b])
        print("\nseed-paired (negative mean diff = second policy needs fewer experiments):")
        for k, v in comps.items():
            if v:
                print(f"  {k:24s} n={v['n']} fewer={v['b_fewer']} ties={v['ties']} mean_diff={v['mean_diff']:+.2f} ci90=[{v['ci90'][0]:+.2f}, {v['ci90'][1]:+.2f}]")
        reads = {}
        if {"HAIKU", "SONNET"} <= set(summ):
            reads["R1_cheap_slower_by_1"] = summ["HAIKU"]["etv_mean"] - summ["SONNET"]["etv_mean"] >= 1.0
            cands = [p for p in ("CONSIST3", "VERBAL", "STALL") if p in summ]
            reads["R2_cascade_within_1_of_sonnet_at_le_40pct_strong"] = any(summ[p]["etv_mean"] <= summ["SONNET"]["etv_mean"] + 1.0 and summ[p]["strong_frac"] <= 0.40 for p in cands)
            reads["R2_which"] = [p for p in cands if summ[p]["etv_mean"] <= summ["SONNET"]["etv_mean"] + 1.0 and summ[p]["strong_frac"] <= 0.40]
            if {"STALL", "CONSIST3", "VERBAL"} <= set(summ):
                reads["R3_stall_at_least_as_fast_at_le_strong"] = all(summ["STALL"]["etv_mean"] <= summ[p]["etv_mean"] and summ["STALL"]["strong_frac"] <= summ[p]["strong_frac"] + 1e-9 for p in ("CONSIST3", "VERBAL"))
            if "RANDOM25" in summ and cands:
                best = min(cands, key=lambda p: summ[p]["etv_mean"])
                reads["R4_random_slower_than_best_cascade"] = {"best_cascade": best, "random_mean": summ["RANDOM25"]["etv_mean"], "best_mean": summ[best]["etv_mean"], "random_strong_frac": summ["RANDOM25"]["strong_frac"], "best_strong_frac": summ[best]["strong_frac"], "pass": summ["RANDOM25"]["etv_mean"] > summ[best]["etv_mean"]}
        print("\nreads:", json.dumps(reads, indent=1))
        (D / f"result_{block}.json").write_text(json.dumps({"summary": summ, "paired": comps, "reads": reads}, indent=1, default=float))


if __name__ == "__main__":
    main()
