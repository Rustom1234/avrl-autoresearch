#!/usr/bin/env python3
"""When does screening ideas before the test pay, and with which model?  Arithmetic on measured numbers.

    python3 loop/ideas/price_rule.py --root loop/runs/B1

Policy (m, f): scorer m rates every proposed idea, only the top fraction f (inside the loop) is tested.
  cost per promising idea = (ask + score_m + f * test) / (p * r_m(f))
  without screening       = (ask + test) / p
p = share of ideas that are promising, r_m(f) = share of the promising ideas that the top f holds (measured on the
retro set: ideas the E1 continuation tested, scored blind). The ratio of the two does not depend on p.
Prices: ask and score are the measured nominal dollars per idea; the price of one test is the free variable.
Intervals: bootstrap over loops, 1,000 draws.
"""
import argparse, glob, json, random
from collections import defaultdict
from pathlib import Path

ap = argparse.ArgumentParser()
ap.add_argument("--root", required=True)
ap.add_argument("--ask", type=float, default=0.0005, help="USD per proposed idea (Haiku planner, measured)")
a = ap.parse_args()
root = Path(a.root).resolve()
truth = {}
for f in sorted(glob.glob(str(root / "retro" / "*.json"))):
    for i in json.loads(Path(f).read_text()):
        truth[(Path(f).stem, i["id"])] = i["z1"] is not None and i["z1"] >= 0.75
sc, cost, n_sc = defaultdict(dict), defaultdict(float), defaultdict(int)
for f in sorted(glob.glob(str(root / "scores" / "retro_*.jsonl"))):
    for r in map(json.loads, open(f)):
        cost[r["model"]] += r["cost_usd_nominal"]
        n_sc[r["model"]] += len(r["scores"])
        for iid, p in r["scores"].items():
            sc[r["model"]][(r["loop"], iid)] = p
price = {m: cost[m] / max(1, n_sc[m]) for m in sc}
common = set.intersection(*[set(v) for v in sc.values()]) & set(truth)
loops = sorted({k[0] for k in common})
print("ideas scored by all scorers: %d in %d loops, promising: %d (p = %.3f)" % (
    len(common), len(loops), sum(truth[k] for k in common), sum(truth[k] for k in common) / len(common)))
print("score price per idea (USD nominal):", {m: round(v, 4) for m, v in price.items()})
FR = (0.1, 0.2, 0.3, 0.5)


def lift(val, keys):
    by = defaultdict(list)
    for k in keys:
        by[k[0]].append(k)
    got, tot, tested = {f: 0 for f in FR}, 0, {f: 0 for f in FR}
    for lp, kk in by.items():
        srt = sorted(kk, key=lambda k: -val[k[0].split("#")[0], k[1]] if isinstance(k[0], str) and "#" in k[0] else -val[k])
        tot += sum(T[k] for k in kk)
        for f in FR:
            top = srt[:max(1, round(f * len(kk)))]
            got[f] += sum(T[k] for k in top)
            tested[f] += len(top)
    return {f: (got[f] / tot if tot else float("nan"), tested[f] / len(keys)) for f in FR}


T = dict(truth)
base = {m: lift(sc[m], list(common)) for m in sc}
rng = random.Random(5)
boot = {m: {f: [] for f in FR} for m in sc}
by_loop = defaultdict(list)
for k in common:
    by_loop[k[0]].append(k)
for _ in range(1000):
    keys = []
    for n, lp in enumerate(rng.choice(loops) for _ in loops):
        for k in by_loop[lp]:
            k2 = ("%s#%d" % (lp, n), k[1])
            T[k2] = truth[k]
            keys.append(k2)
    for m in sc:
        r = lift(sc[m], keys)
        for f in FR:
            if r[f][0] == r[f][0]:
                boot[m][f].append(r[f][0])
print("\nShare of the promising ideas inside the top fraction f of scores (r), with 95% intervals")
print("| scorer | f = 0.1 | 0.2 | 0.3 | 0.5 |\n|---|---|---|---|---|")
for m in sc:
    cells = []
    for f in FR:
        b = sorted(boot[m][f])
        cells.append("%.2f (%.2f to %.2f)" % (base[m][f][0], b[int(0.025 * len(b))], b[int(0.975 * len(b))]))
    print("| %s | %s |" % (m, " | ".join(cells)))
print("\nCost per promising idea, relative to testing everything (below 1 = screening is cheaper), by price of one test")
grid = (0.002, 0.01, 0.05, 0.25, 1.0, 5.0)
print("| policy | " + " | ".join("%g USD" % e for e in grid) + " |\n|---|" + "---|" * len(grid))
best = {e: (1.0, "test everything") for e in grid}
for m in sc:
    for f in FR:
        r, ftrue = base[m][f]
        if not r:
            continue
        row = []
        for e in grid:
            ratio = (a.ask + price[m] + ftrue * e) / (r * (a.ask + e))
            row.append("%.2f" % ratio)
            if ratio < best[e][0]:
                best[e] = (ratio, "%s, top %d%%" % (m, f * 100))
        print("| %s, top %d%% | %s |" % (m, f * 100, " | ".join(row)))
print("| cheapest | " + " | ".join("%s (%.2f)" % (best[e][1], best[e][0]) for e in grid) + " |")
print("\nBreak-even price of one test (screening pays above it):")
for m in sc:
    for f in (0.2, 0.3):
        r, ftrue = base[m][f]
        if r and r > ftrue:
            e0 = ((a.ask + price[m]) / r - a.ask) / (1 - ftrue / r)
            print("  %s, top %d%%: %.4f USD; with very expensive tests the cost per promising idea falls to %.2f of testing everything" % (
                m, f * 100, max(0.0, e0), ftrue / r))
