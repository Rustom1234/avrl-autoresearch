#!/usr/bin/env python3
"""Phantom gains from real noise: replay accept rules on a pool of scores of one unchanged script.

    python loop/null/replay.py --pool FILE --field val [--sigma S] [--loops 20000]

FILE is a jsonl of jobs of the unchanged starting script (lower score is better). Every "candidate" in the replay
is another draw from that pool, so the true gain of anything a rule accepts is exactly zero. Reported per rule: the
gain the loop would report (starting bar minus the final bar), in score units and in noise sds; how many edits it
keeps; the share of loops that keep at least one. Next to each number, the same replay with Gaussian noise of the
same sd, to see whether the real noise has heavier tails or host effects than the model used so far.

Rules:
  ours      the engine of this thesis: z = (bar - score) / sigma; reject below 0.75; accept at 2.0 or more; in
            between run it again and accept when the mean z of the two runs is at least 1.5. The bar starts as the
            mean of 10 baseline runs and becomes the mean of the accepted candidate's own runs. 20 and 40 slots.
  strict    keep whenever one run beats the best score so far (the rule of the original autoresearch loop), 100
            experiments.
  two-seed  keep when one run beats the bar by 2 sigma or more and a second run also beats the bar; the new bar
            is the mean of the two (a 2-sigma rule with a confirmation seed), 93 experiments.
sigma is 1.15 times the sd of the pool unless given (the rule used to set sigma in P2B and CG1).
"""
import argparse, json, random
from statistics import fmean, stdev

ap = argparse.ArgumentParser()
ap.add_argument("--pool", required=True)
ap.add_argument("--field", default="val")
ap.add_argument("--sigma", type=float, default=None)
ap.add_argument("--loops", type=int, default=20000)
a = ap.parse_args()
rows = [r for r in map(json.loads, open(a.pool)) if r.get(a.field) is not None]
pool = [r[a.field] for r in rows]
sd, mu = stdev(pool), fmean(pool)
sigma = a.sigma or 1.15 * sd
print("pool: %d jobs, mean %.5f, sd %.5f, sigma used %.5f" % (len(pool), mu, sd, sigma))
hosts = sorted({r.get("host", "") for r in rows})
if len(hosts) > 1:
    print("per host: " + "; ".join("%s mean %.5f sd %.5f n %d" % (h, fmean(v), stdev(v), len(v)) for h in hosts
                                   for v in [[r[a.field] for r in rows if r.get("host") == h]]))
srt = sorted((x - mu) / sd for x in pool)
print("tails: min %.2f sd, max %.2f sd; share beyond 2 sd %.3f (Gaussian 0.046)" % (srt[0], srt[-1], sum(abs(x) > 2 for x in srt) / len(srt)))


def ours(draw, slots):
    bar, keeps = fmean(draw() for _ in range(10)), 0
    start, used = bar, 0
    while used < slots:
        x = draw(); used += 1
        z = (bar - x) / sigma
        if z < 0.75:
            continue
        if z >= 2.0:
            bar, keeps = x, keeps + 1
            continue
        if used >= slots:
            break
        y = draw(); used += 1
        if ((bar - x) / sigma + (bar - y) / sigma) / 2 >= 1.5:
            bar, keeps = (x + y) / 2, keeps + 1
    return start - bar, keeps


def strict(draw, n):
    best = draw(); start, keeps = best, 0
    for _ in range(n):
        x = draw()
        if x < best:
            best, keeps = x, keeps + 1
    return start - best, keeps


def two_seed(draw, n):
    bar = draw(); start, keeps, used = bar, 0, 0
    while used < n:
        x = draw(); used += 1
        if bar - x >= 2 * sigma and used < n:
            y = draw(); used += 1
            if y < bar:
                bar, keeps = (x + y) / 2, keeps + 1
    return start - bar, keeps


def run(rule, n, draw, rng):
    g, k = zip(*(rule(draw, n) for _ in range(a.loops)))
    return fmean(g), fmean(k), sum(x > 0 for x in k) / len(k)


print("\n| Rule | Experiments | Reported gain, real noise | in sds | Keeps | Loops with a keep | Same with Gaussian noise (gain in sds, keeps) |\n|---|---|---|---|---|---|---|")
for name, rule, n in (("ours", ours, 20), ("ours", ours, 40), ("strict", strict, 100), ("two-seed", two_seed, 93)):
    rng = random.Random(1)
    g, k, p = run(rule, n, lambda: rng.choice(pool), rng)
    rg = random.Random(2)
    gg, gk, gp = run(rule, n, lambda: rg.gauss(mu, sd), rg)
    print("| %s | %d | %.5f | %.2f | %.2f | %.0f%% | %.2f, %.2f |" % (name, n, g, g / sd, k, 100 * p, gg / sd, gk))
