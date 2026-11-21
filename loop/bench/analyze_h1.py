#!/usr/bin/env python3
"""H1 analysis (checks and predictions of loop/design/H1_PREREG.md).

    python loop/bench/analyze_h1.py

Reads the held-out summary of H1 (cells GG, HG, GH) and of BG1_R (anchors HH and N), model dollars from each loop's
calls.jsonl, slot outcomes from each ledger.jsonl, and the idea-gap scores of loop/k1/h1_gap.json.
"""
import glob, json, random
from collections import Counter, defaultdict
from pathlib import Path
from statistics import fmean

R = Path(__file__).resolve().parents[1] / "runs"
rows = [r for r in json.load(open(R / "H1/confirmation/summary.json")) if r["cell"] in ("GG", "HG", "GH")]
for r in rows:
    r["root"] = "H1"
anch = [r for r in json.load(open(R / "BG1_R/confirmation/summary.json")) if r["cell"] in ("HH", "N", "SS")]
for r in anch:
    r["root"] = "BG1_R"
rows += anch
for r in rows:
    d = R / r["root"] / r["loop"]
    r["cost"] = sum(json.loads(l).get("cost_usd_nominal") or 0 for l in open(d / "calls.jsonl")) if (d / "calls.jsonl").exists() else 0.0
    r["dec"] = Counter(json.loads(l).get("decision") for l in open(d / "ledger.jsonl") if json.loads(l).get("kind") == "candidate")
CELLS = ("N", "GG", "HG", "GH", "HH", "SS")
tasks = sorted({r["task"] for r in rows})
fam = {r["task"]: r["family"] for r in rows}


def mean(rs, cell, key="progress", task=None):
    v = [r[key] for r in rs if r["cell"] == cell and (task is None or r["task"] == task)]
    return fmean(v) if v else float("nan")


def recovered(rs, x, key="progress"):
    g, h = mean(rs, "GG", key), mean(rs, "HH", key)
    return (mean(rs, x, key) - g) / (h - g) if h != g else float("nan")


def perm_p(rs, x, y, n=20000):
    """one-sided: mean(x) - mean(y) under shuffling of cell labels inside tasks"""
    rs = [r for r in rs if r["cell"] in (x, y)]
    obs = mean(rs, x) - mean(rs, y)
    rng, hit = random.Random(0), 0
    by = defaultdict(list)
    for r in rs:
        by[r["task"]].append(r)
    for _ in range(n):
        sh = []
        for t, g in by.items():
            lab = [r["cell"] for r in g]
            rng.shuffle(lab)
            sh += [dict(r, cell=l) for r, l in zip(g, lab)]
        hit += mean(sh, x) - mean(sh, y) >= obs - 1e-12
    return (hit + 1) / (n + 1)


def boot(rs, fn, n=4000):
    rng, vals = random.Random(1), []
    by = defaultdict(list)
    for r in rs:
        by[(r["task"], r["cell"])].append(r)
    for _ in range(n):
        v = fn([rng.choice(g) for g in by.values() for _ in g])
        if v == v and abs(v) < 1e6:
            vals.append(v)
    vals.sort()
    return vals[int(0.025 * len(vals))], vals[int(0.975 * len(vals))]


print("== mean held-out progress per task (loops) ==\n| task | " + " | ".join(CELLS) + " |\n|---|" + "---|" * len(CELLS))
for t in tasks:
    print("| %s | " % t + " | ".join("%.3f (%d)" % (mean(rows, c, task=t), sum(r["cell"] == c and r["task"] == t for r in rows)) for c in CELLS) + " |")
print("\n== model dollars per loop, and coder slot outcomes ==")
for name in ("admet", "packing"):
    fr = [r for r in rows if r["family"] == name]
    print("\n### %s" % name)
    print("progress: " + ", ".join("%s %.3f" % (c, mean(fr, c)) for c in CELLS))
    print("cost USD: " + ", ".join("%s %.2f" % (c, mean(fr, c, "cost")) for c in CELLS))
    for c in ("GG", "HG", "GH", "HH"):
        d = Counter()
        for r in fr:
            if r["cell"] == c:
                d.update(r["dec"])
        tot = sum(d.values())
        print("  %s slots %d: " % (c, tot) + ", ".join("%s %.0f%%" % (k, 100 * v / tot) for k, v in d.most_common()))
    gg, hh = mean(fr, "GG"), mean(fr, "HH")
    p = perm_p(fr, "HH", "GG")
    G = gg <= 2 * hh / 3 and p < 0.05
    print("G gap: GG / HH = %.2f (needs <= 0.67), HH above GG p = %.4f -> %s" % (gg / hh, p, "HOLDS" if G else "FAILS"))
    for x, lab in (("HG", "P planner"), ("GH", "C coder")):
        rec, ec = recovered(fr, x), recovered(fr, x, "cost")
        lo, hi = boot(fr, lambda s: recovered(s, x))
        mlo, mhi = boot(fr, lambda s: recovered(s, x) - recovered(s, x, "cost"))
        print("%s: gap recovered by %s = %.2f (95%% %.2f to %.2f; needs >= 0.5) -> %s; extra cost share %.2f; thesis metric %.2f (95%% %.2f to %.2f)" % (
            lab, x, rec, lo, hi, ("HOLDS" if rec >= 0.5 else "FAILS") if G else "descriptive", ec, rec - ec, mlo, mhi))
    print("HG above GG p = %.4f; GH above GG p = %.4f; GH above HG p = %.4f; HH above GH p = %.4f" % (
        perm_p(fr, "HG", "GG"), perm_p(fr, "GH", "GG"), perm_p(fr, "GH", "HG"), perm_p(fr, "HH", "GH")))

print("\n== idea-gap predictions ==")
gap = json.load(open(R.parent / "k1/h1_gap.json"))
key = {t: t for t in tasks}
ok = [t for t in tasks if mean(rows, "GH", task=t) >= 2 * mean(rows, "HH", task=t) / 3]
print("1. GH >= 2/3 HH in %d of %d tasks (needs 5): %s -> %s" % (len(ok), len(tasks), ", ".join(ok), "HOLDS" if len(ok) >= 5 else "FAILS"))
# pooled over tasks: mean of task means, so that each task counts once
tm = lambda c: fmean(mean(rows, c, task=t) for t in tasks)
rg, rh = (tm("GH") - tm("GG")) / (tm("HH") - tm("GG")), (tm("HG") - tm("GG")) / (tm("HH") - tm("GG"))
print("2. pooled gap recovered: GH %.2f, HG %.2f -> %s" % (rg, rh, "HOLDS" if rg > rh else "FAILS"))
sc = [gap[key[t]]["gap_top5"] for t in tasks]
df = [mean(rows, "HH", task=t) - mean(rows, "GH", task=t) for t in tasks]
rank = lambda v: [sorted(v).index(x) for x in v]
a, b = rank(sc), rank(df)
n = len(a)
rho = 1 - 6 * sum((x - y) ** 2 for x, y in zip(a, b)) / (n * (n * n - 1))
print("3. Spearman(score, HH - GH) = %.2f -> %s" % (rho, "HOLDS" if rho > 0 else "FAILS"))
for t, s, d in zip(tasks, sc, df):
    print("   %s score %.2f, HH - GH %.3f" % (t, s, d))
