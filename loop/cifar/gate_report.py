#!/usr/bin/env python3
"""CG1 report: the four checks of loop/design/CG1_PREREG.md from confirmation/summary.json and the loop ledgers.

    python loop/cifar/gate_report.py --root loop/runs/CG1 [--labels loop/runs/CG1/labels.json]

labels.json maps "<loop_id>:<slot>" of an accepted SS edit to a short idea label (written by hand from the diffs);
without it check C lists the accepted edits and is not evaluated.
"""
import argparse, itertools, json, math, random
from pathlib import Path
from statistics import fmean, stdev

ap = argparse.ArgumentParser()
ap.add_argument("--root", required=True)
ap.add_argument("--labels", default=None)
a = ap.parse_args()
root = Path(a.root)
S = json.loads((root / "confirmation/summary.json").read_text())
cells = {}
for lid, v in sorted(S["loops"].items()):
    if v["gain"] is not None:
        cells.setdefault(v["cell"], []).append(v["gain"])


def boot(v, n=10000, seed=0):
    r = random.Random(seed)
    m = sorted(fmean(r.choice(v) for _ in v) for _ in range(n))
    return m[int(0.025 * n)], m[int(0.975 * n)]


def t_p_one_sided(v):
    """One-sample t test of mean > 0. p from the t distribution by numerical integration (no scipy needed)."""
    n, m, s = len(v), fmean(v), stdev(v)
    if s == 0:
        return float("inf"), 0.0 if m > 0 else 1.0
    t, df = m / (s / math.sqrt(n)), n - 1
    pdf = lambda x: math.gamma((df + 1) / 2) / (math.sqrt(df * math.pi) * math.gamma(df / 2)) * (1 + x * x / df) ** (-(df + 1) / 2)
    steps, hi = 20000, 60.0
    h = (hi - t) / steps
    return t, max(0.0, sum(pdf(t + (i + 0.5) * h) for i in range(steps)) * h) if t < hi else 0.0


def perm_p(x, y):
    """Exact one-sided permutation p for mean(x) > mean(y)."""
    pool, obs, n, hit, tot = x + y, fmean(x) - fmean(y), len(x), 0, 0
    tsum = sum(pool)
    for idx in itertools.combinations(range(len(pool)), n):
        sx = sum(pool[i] for i in idx)
        d = sx / n - (tsum - sx) / (len(pool) - n)
        hit += d >= obs - 1e-12
        tot += 1
    return hit / tot


print("| Cell | Loops | Mean sealed gain | 95% interval | Loops that changed the script | Gains |")
print("|---|---|---|---|---|---|")
for c, g in cells.items():
    lo, hi = boot(g)
    ch = sum(1 for v in S["loops"].values() if v["cell"] == c and v["changed"])
    print("| %s | %d | %.4f | %.4f to %.4f | %d | %s |" % (c, len(g), fmean(g), lo, hi, ch, " ".join("%.3f" % x for x in g)))
ss, n, hh = cells.get("SS", []), cells.get("N", []), cells.get("HH", [])
t, p = t_p_one_sided(ss)
print("\nA headroom: mean gain(SS) = %.4f (needs >= 0.05); t = %.2f, one-sided p = %.4f -> %s" % (
    fmean(ss), t, p, "PASS" if fmean(ss) >= 0.05 and p < 0.05 else "FAIL"))
pb = perm_p(ss, n)
print("B menu: gain(N) / gain(SS) = %.2f (needs <= 0.33); permutation p = %.4f -> %s" % (
    fmean(n) / fmean(ss) if fmean(ss) else float("nan"), pb, "PASS" if fmean(n) <= fmean(ss) / 3 and pb < 0.05 else "FAIL"))
print("D gap: gain(HH) / gain(SS) = %.2f (needs <= 0.67) -> %s" % (
    fmean(hh) / fmean(ss) if fmean(ss) else float("nan"), "PASS" if fmean(hh) <= 2 * fmean(ss) / 3 else "FAIL"))

labels = json.loads(Path(a.labels).read_text()) if a.labels else {}
acc = []
for lid, v in sorted(S["loops"].items()):
    if v["cell"] not in ("SS", "HH"):
        continue
    inc = 0.0
    for line in open(root / lid / "ledger.jsonl"):
        r = json.loads(line)
        if r.get("decision") == "accept":
            new = r["incumbent_after"]["mean"] if r.get("incumbent_after") else None
            before = r["incumbent_before"]["mean"]
            acc.append((v["cell"], lid, r["slot"], before - new if new is not None else None, r.get("description", "")))
print("\nAccepted edits (in-loop gain = drop of the incumbent's mean cost):")
for cell, lid, slot, g, d in acc:
    print("  %s %s slot %d gain %s [%s] %s" % (cell, lid, slot, "%.4f" % g if g is not None else "?",
                                              labels.get("%s:%d" % (lid, slot), "unlabelled"), d[:160]))
if labels:
    tot, by = sum(g for c, _, _, g, _ in acc if c == "SS"), {}
    for c, lid, slot, g, _ in acc:
        if c == "SS":
            k = labels.get("%s:%d" % (lid, slot), "unlabelled")
            by[k] = by.get(k, 0.0) + g
    top = max(by.items(), key=lambda kv: kv[1])
    print("\nC one trick: shares of the summed in-loop SS gain:", ", ".join("%s %.0f%%" % (k, 100 * g / tot) for k, g in sorted(by.items(), key=lambda kv: -kv[1])))
    print("C: largest share %.0f%% (%s), needs <= 60%% -> %s" % (100 * top[1] / tot, top[0], "PASS" if top[1] / tot <= 0.6 else "FAIL"))
