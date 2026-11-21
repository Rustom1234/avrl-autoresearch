#!/usr/bin/env python3
"""Analysis of the memory study S6, as written down in curation/design/S6_PREREG.md.

    python curation/loop/analysis_s6.py --root curation/runs/S6 --anchors curation/runs/R1 [--out curation/runs/S6/analysis.json]

Cells: P0 (plain cheap planner), PM (memory in the prompt), PW (memory in the weights), K (an old final file, no model).
Anchors SS, CC and SC are the R1 loops (same pool, engine and seeds; run earlier, not concurrent).
Rules: T1 (PW gains more than P0, exact one-sided permutation test) and T2 (at least 3 of 4 PW loops at the strong
level, which is 85 percent of the SS mean gain). Everything else is descriptive. Gap recovered, extra cost, advantage
and their intervals come from the frozen C1 analysis (analysis_c1.py), unchanged.
"""
import argparse, json, random, re, statistics as st, sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import analysis_c1 as A
from table import raw_runs, perm_p, ROOT

CELLS, ANCHORS, REACH = ("P0", "PM", "PW"), ("SS", "CC", "SC"), 0.85
PAIRS = (("PW", "P0"), ("PW", "PM"), ("PM", "P0"))


def gain_diff(gains, x, y, n_boot=A.N_BOOT, seed=A.BOOT_SEED):
    """Mean gain of cell x minus cell y, with a bootstrap interval (loops resampled within cells)."""
    rng = random.Random(seed)
    d = sorted(st.mean(rng.choice(gains[x]) for _ in gains[x]) - st.mean(rng.choice(gains[y]) for _ in gains[y])
               for _ in range(n_boot))
    return {"diff": st.mean(gains[x]) - st.mean(gains[y]), "ci": [d[int(0.025 * n_boot)], d[int(0.975 * n_boot) - 1]]}


def cached(pool, raw_sealed, prefix="S6K_"):
    """Old final files run unchanged on this pool: {name: {"gain": mean over seeds, "seeds": n}}."""
    runs = {}
    for f in sorted((ROOT / "curation/results" / pool).glob(prefix + "*_s*.json")):
        name = re.match(r"%s(.+)_s\d+$" % prefix, f.stem).group(1)
        runs.setdefault(name, []).append(json.loads(f.read_text())["sealed_bpb"])
    return {k: {"gain": raw_sealed - st.mean(v), "seeds": len(v)} for k, v in runs.items()}


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--root", required=True)
    ap.add_argument("--anchors", required=True)
    ap.add_argument("--pool", default="v2b")
    ap.add_argument("--out", default=None)
    a = ap.parse_args(argv)
    raw_sealed = st.mean(r["sealed_bpb"] for r in raw_runs(a.pool))
    loops = [l for l in A.load(a.anchors, raw_sealed) if l["cell"] in ANCHORS] + \
            [l for l in A.load(a.root, raw_sealed) if l["cell"] in CELLS]
    lost = [l["id"] for l in loops if l["lost_slot"] or l["gain"] is None]
    loops = [l for l in loops if l["id"] not in lost]
    cells = {c: [(l["gain"], l["usd"]) for l in loops if l["cell"] == c] for c in ANCHORS + CELLS}
    cells = {c: v for c, v in cells.items() if v}
    gains = {c: [x[0] for x in v] for c, v in cells.items()}
    g, usd = {c: st.mean(v) for c, v in gains.items()}, {c: st.mean(x[1] for x in v) for c, v in cells.items()}
    level = REACH * g["SS"]
    boot = A.bootstrap(cells)
    res = {"raw_sealed": raw_sealed, "strong_level": level, "left_out": lost, "cells": {}, "pairs": {}}
    print("raw sealed bpb %.4f; strong level: gain of at least %.4f (%.0f percent of the SS mean)" % (raw_sealed, level, 100 * REACH))
    if lost:
        print("left out (lost a slot or has no sealed score): %s" % " ".join(lost))
    print("\n%-4s %2s %8s %8s  %-8s %6s %6s %6s  %s" % ("cell", "n", "gain", "usd", "at level", "gap", "cost", "adv", "loop gains"))
    for c in ANCHORS + CELLS:
        if c not in cells:
            continue
        gr, xc, adv = A.advantage(g, usd, c)
        row = {"n": len(gains[c]), "gain": g[c], "usd": usd[c], "at_level": sum(x >= level for x in gains[c]),
               "gap": gr, "extra_cost": xc, "advantage": adv, "gains": gains[c], "boot": boot.get(c)}
        res["cells"][c] = row
        ci = "" if c not in boot else "  adv 95%% [%.2f, %.2f]" % tuple(boot[c]["adv"])
        print("%-4s %2d %8.4f %8.2f  %d of %d   %6.2f %6.2f %6.2f  %s%s" % (
            c, row["n"], g[c], usd[c], row["at_level"], row["n"], gr, xc, adv, " ".join("%.4f" % x for x in gains[c]), ci))
    print()
    for x, y in PAIRS:
        if x in gains and y in gains:
            d = gain_diff(gains, x, y)
            d["perm_p"] = perm_p(gains[x], gains[y])
            res["pairs"]["%s-%s" % (x, y)] = d
            print("%s minus %s: gain difference %+.4f, 95%% [%+.4f, %+.4f], one-sided permutation p %.3f" % (
                x, y, d["diff"], d["ci"][0], d["ci"][1], d["perm_p"]))
    if "PW" in gains and "P0" in gains:
        t1 = res["pairs"]["PW-P0"]
        res["T1"] = t1["diff"] > 0 and t1["perm_p"] <= 0.05
        res["T2"] = res["cells"]["PW"]["at_level"] >= 3
        print("\nT1 (training helps: PW above P0, p at most 0.05): %s" % ("HOLDS" if res["T1"] else "does not hold"))
        print("T2 (at least 3 of 4 PW loops at the strong level): %s" % ("HOLDS" if res["T2"] else "does not hold"))
    k = cached(a.pool, raw_sealed)
    if k:
        res["K"] = {"files": k, "at_level": sum(v["gain"] >= level for v in k.values()), "n": len(k),
                    "gain": st.mean(v["gain"] for v in k.values())}
        print("\nK (old final file, no model): mean gain %.4f, %d of %d at the strong level" % (
            res["K"]["gain"], res["K"]["at_level"], len(k)))
        for name, v in sorted(k.items()):
            print("  %-6s gain %.4f (%d seeds)%s" % (name, v["gain"], v["seeds"], "" if v["gain"] >= level else "  below the level"))
    if a.out:
        Path(a.out).write_text(json.dumps({"result": res, "loops": loops}, indent=1))
    return 0


if __name__ == "__main__":
    sys.exit(main())
