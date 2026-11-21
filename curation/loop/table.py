#!/usr/bin/env python3
"""Stage B table: per loop and per cell sealed gain and model dollars, the strong-minus-cheap gap and its test.

    python curation/loop/table.py --root curation/runs/B1 --pool v2 [--strong SS --cheap QT]

Sealed gain of a loop = mean sealed bpb of the raw pool (the 8 pilot runs) minus the loop's confirmation mean.
The test is an exact permutation test on the loop gains (all ways to split the loops into the two cells), one-sided.
"""
import argparse, itertools, json, statistics as st
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]


def raw_runs(pool):
    return [json.loads(p.read_text()) for p in sorted((ROOT / "curation/results" / pool).glob("raw_s*.json"))]


def perm_p(a, b):
    """One-sided p for mean(a) > mean(b): share of splits with a difference at least as large as observed."""
    obs, allv, n = st.mean(a) - st.mean(b), a + b, len(a)
    hits = tot = 0
    for idx in itertools.combinations(range(len(allv)), n):
        x = [allv[i] for i in idx]
        y = [allv[i] for i in range(len(allv)) if i not in idx]
        tot += 1
        hits += (st.mean(x) - st.mean(y)) >= obs - 1e-12
    return hits / tot


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--root", required=True)
    ap.add_argument("--pool", default="v2")
    ap.add_argument("--strong", default="SS")
    ap.add_argument("--cheap", default="QT")
    a = ap.parse_args(argv)
    raw = raw_runs(a.pool)
    anchor, sd = st.mean(r["sealed_bpb"] for r in raw), st.stdev(r["sealed_bpb"] for r in raw)
    print("raw pool %s: sealed %.4f (single-run sd %.4f, %d runs)" % (a.pool, anchor, sd, len(raw)))
    cells = {}
    print("%-8s %8s %8s %7s %7s %6s" % ("loop", "sealed", "gain", "accept", "usd", "infra"))
    for f in sorted(Path(a.root).glob("*/final.json")):
        d = json.loads(f.read_text())
        if d["sealed_bpb"] is None:
            print("%-8s no confirmation result" % d["loop_id"])
            continue
        g = anchor - d["sealed_bpb"]
        cells.setdefault(d["cell"], []).append((g, d["model_usd"]))
        print("%-8s %8.4f %+8.4f %7d %7.2f %6d" % (d["loop_id"], d["sealed_bpb"], g, d["n_accept"], d["model_usd"],
                                                 d.get("infra_failures", 0)))
    for c, v in sorted(cells.items()):
        g = [x[0] for x in v]
        print("cell %-4s n %d  gain %.4f (sd %.4f)  dollars per loop %.2f" % (
            c, len(v), st.mean(g), st.stdev(g) if len(g) > 1 else 0.0, st.mean(x[1] for x in v)))
    if a.strong in cells and a.cheap in cells:
        s, q = [x[0] for x in cells[a.strong]], [x[0] for x in cells[a.cheap]]
        gap = st.mean(s) - st.mean(q)
        print("gap %s minus %s = %.4f bpb = %.1f single-run noise sds; one-sided permutation p = %.3f" % (
            a.strong, a.cheap, gap, gap / sd, perm_p(s, q)))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
