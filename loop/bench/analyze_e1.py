#!/usr/bin/env python3
"""E1 analysis (checks of loop/design/E1_PREREG.md): extra held-out progress from slots 41 to 80.

    python loop/bench/analyze_e1.py
"""
import json, math, random
from collections import defaultdict
from pathlib import Path
from statistics import fmean, stdev

R = Path(__file__).resolve().parents[1] / "runs"
a = {r["loop"]: r for r in json.load(open(R / "BG1_R/confirmation/summary.json"))}
b = {r["loop"]: r for r in json.load(open(R / "E1/confirmation/summary.json"))}
span = {}                                   # held-out start minus reference, per task
for r in a.values():
    if abs(r["progress"]) > 1e-9:
        span[r["task"]] = (r["base"] - r["final"]) / r["progress"]
rows = []
for k, r in sorted(b.items()):
    same = (R / "E1" / k / "incumbent.py").read_text() == (R / "BG1_R" / k / "incumbent.py").read_text()
    extra = 0.0 if same else r["progress"] - a[k]["progress"]
    rep = (a[k]["inloop"] - r["inloop"]) / span[r["task"]]
    rows.append(dict(loop=k, task=r["task"], cell=r["cell"], p40=a[k]["progress"], p80=r["progress"], same=same, extra=extra, rep=rep))
    print("%-12s at 40 %.3f  at 80 %.3f  extra %+.3f  in-loop reported %+.3f%s" % (k, a[k]["progress"], r["progress"], extra, rep, "  (no accept after 40)" if same else ""))
ex = [r["extra"] for r in rows]
m, s = fmean(ex), stdev(ex)
t = m / (s / math.sqrt(len(ex))) if s else 0.0
# one-sided p from the t distribution by simulation-free normal approximation is too loose at n=28; use a sign-flip test
rng, hit = random.Random(0), 0
for _ in range(20000):
    hit += fmean(x * rng.choice((-1, 1)) for x in ex) >= m - 1e-12
px = (hit + 1) / 20001
print("\nloops %d, changed after slot 40: %d" % (len(rows), sum(not r["same"] for r in rows)))
print("X more: mean extra %.3f (needs > 0.05), t = %.2f, sign-flip p = %.4f -> %s" % (m, t, px, "HOLDS" if m > 0.05 and px < 0.05 else "FAILS"))
g = lambda c, rs=rows: fmean(r["extra"] for r in rs if r["cell"] == c)
obs, hit = g("SS") - g("HH"), 0
by = defaultdict(list)
for r in rows:
    by[r["task"]].append(r)
for _ in range(20000):
    sh = []
    for grp in by.values():
        lab = [r["cell"] for r in grp]
        rng.shuffle(lab)
        sh += [dict(r, cell=l) for r, l in zip(grp, lab)]
    hit += g("SS", sh) - g("HH", sh) >= obs - 1e-12
ps = (hit + 1) / 20001
print("S strong: extra SS %.3f, HH %.3f, p = %.4f -> %s" % (g("SS"), g("HH"), ps, "HOLDS" if obs > 0 and ps < 0.05 else "FAILS"))
rep = fmean(r["rep"] for r in rows)
print("F phantom: in-loop reported %.3f, held-out extra %.3f (needs reported >= 2 x extra) -> %s" % (rep, m, "HOLDS" if rep >= 2 * m else "FAILS"))
for c in ("SS", "HH"):
    v = [r for r in rows if r["cell"] == c]
    print("%s: at 40 %.3f, at 80 %.3f, reported %.3f" % (c, fmean(r["p40"] for r in v), fmean(r["p40"] + r["extra"] for r in v), fmean(r["rep"] for r in v)))
