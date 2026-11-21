#!/usr/bin/env python3
"""A first simulator of the research loop, fitted on the ledgers of BG1_R and H1 and tested on E1.

    python loop/sim/fit.py

Unit: one experiment = one charged ledger row (a candidate, a replicate or a crash fix). For every experiment the
table has the task, the planner and coder model, its index in the loop, the number of edits accepted so far, the
experiments since the last accept, whether it was accepted, and the in-loop gain of an accept on the progress scale
((bar before - bar after) / (held-out start - reference)).

Model (discrete-time hazard): P(accept at this experiment) = logistic(task family, coder, planner, log index,
accepts so far, log experiments since last accept). Gain of an accept: resampled from the observed gains of the
same family with the same number of earlier accepts (0, 1, 2+).

Test: the model is fitted on experiments 1 to 40 only (BG1_R, H1) and asked for the number of accepts in
experiments 41 to 80 of the 28 loops that E1 continued.

Use: the stopping rule of the Pandora's box index in this model is "stop when hazard x expected gain x value of
progress < cost of one experiment". Printed: the expected progress kept and experiments saved by that rule for a
range of values, and by plain patience rules, replayed on the real ledgers.
"""
import glob, json, math, random
from collections import defaultdict
from pathlib import Path
from statistics import fmean

import numpy as np
from sklearn.linear_model import LogisticRegression

R = Path(__file__).resolve().parents[1] / "runs"
span, fam, held = {}, {}, {}
for root in ("BG1_R", "H1", "E1"):
    for r in json.load(open(R / root / "confirmation/summary.json")):
        fam[r["task"]] = r["family"]
        held[(root, r["loop"])] = r["progress"]
        if abs(r["progress"]) > 1e-9:
            span[r["task"]] = (r["base"] - r["final"]) / r["progress"]
MODEL = {"S": "sonnet", "H": "haiku", "G": "gemma"}


def loops(root):
    for f in sorted(glob.glob(str(R / root / "*/ledger.jsonl"))):
        d = Path(f).parent
        cfg = json.loads((d / "config.json").read_text())
        if cfg.get("cell") in (None, "N"):
            continue
        rows, n_acc, since, e = [], 0, 0, 0
        for l in open(f):
            x = json.loads(l)
            if not x.get("charged"):
                continue
            e += 1
            acc = x.get("decision") == "accept"
            gain = (x["incumbent_before"]["mean"] - x["incumbent_after"]["mean"]) / span[cfg["task"]] if acc else 0.0
            cost = sum(c.get("cost_usd_nominal") or 0 for c in x.get("model_calls") or [])
            rows.append(dict(root=root, loop=d.name, task=cfg["task"], fam=fam[cfg["task"]], planner=cfg["cell"][0],
                             coder=cfg["cell"][1], e=e, n_acc=n_acc, since=since, acc=int(acc), gain=gain, cost=cost,
                             bad=int(x.get("decision") in ("crash", "invalid", "abort"))))
            n_acc, since = (n_acc + 1, 0) if acc else (n_acc, since + 1)
        yield d.name, rows


data = {root: dict(loops(root)) for root in ("BG1_R", "H1", "E1")}
train = [r for root in ("BG1_R", "H1") for rows in data[root].values() for r in rows]
with open(Path(__file__).parent / "slots.jsonl", "w") as f:
    for root in data:
        for rows in data[root].values():
            for r in rows:
                f.write(json.dumps(r) + "\n")


def feats(r):
    return [r["fam"] == "packing", r["coder"] == "G", r["coder"] == "S", r["planner"] == "G", r["planner"] == "S",
            math.log(r["e"]), min(r["n_acc"], 3), math.log1p(r["since"])]


clf = LogisticRegression(C=1.0, max_iter=2000).fit(np.array([feats(r) for r in train], float), [r["acc"] for r in train])
haz = lambda r: float(clf.predict_proba(np.array([feats(r)], float))[0, 1])
names = ["packing", "coder Gemma", "coder Sonnet", "planner Gemma", "planner Sonnet", "log index", "accepts so far", "log since last accept"]
print("fitted on %d experiments of %d loops, %d accepts" % (len(train), sum(len(data[k]) for k in ("BG1_R", "H1")), sum(r["acc"] for r in train)))
print("hazard coefficients: " + ", ".join("%s %+.2f" % (n, c) for n, c in zip(names, clf.coef_[0])) + ", intercept %+.2f" % clf.intercept_[0])
gains = defaultdict(list)
for r in train:
    if r["acc"]:
        gains[(r["fam"], min(r["n_acc"], 2))].append(r["gain"])
print("mean in-loop gain of an accept (progress units): " + "; ".join("%s after %d accepts %.3f (n=%d)" % (k[0], k[1], fmean(v), len(v)) for k, v in sorted(gains.items())))
print("mean model cost per experiment (USD): " + ", ".join("%s%s %.4f" % (p, c, fmean(r["cost"] for r in train if r["planner"] == p and r["coder"] == c))
                                                         for p, c in sorted({(r["planner"], r["coder"]) for r in train})))

# ---- test on E1: simulate experiments 41..80 from each loop's state at 40
rng, HC = random.Random(0), {}
obs = sum(r["acc"] for rows in data["E1"].values() for r in rows if r["e"] > 40)
n_late = sum(1 for rows in data["E1"].values() for r in rows if r["e"] > 40)
sims = []
for _ in range(2000):
    tot = 0
    for name, rows in data["E1"].items():
        first = [r for r in rows if r["e"] <= 40]
        if not first:
            continue
        s = dict(first[-1])
        n_acc, since = s["n_acc"] + s["acc"], 0 if s["acc"] else s["since"] + 1
        for e in range(41, 41 + sum(1 for r in rows if r["e"] > 40)):
            s.update(e=e, n_acc=n_acc, since=since)
            key = (s["fam"], s["planner"], s["coder"], e, min(n_acc, 3), since)
            if key not in HC:
                HC[key] = haz(s)
            if rng.random() < HC[key]:
                tot, n_acc, since = tot + 1, n_acc + 1, 0
            else:
                since += 1
    sims.append(tot)
sims.sort()
print("\ntest on E1 (%d late experiments): accepts observed %d, predicted mean %.1f (95%% of simulations %d to %d)" % (
    n_late, obs, fmean(sims), sims[int(0.025 * len(sims))], sims[int(0.975 * len(sims))]))

# ---- stopping rules replayed on the real ledgers of BG1_R and H1 (40 experiments each)
allrows = [rows for root in ("BG1_R", "H1") for rows in data[root].values()]
from sklearn.model_selection import GroupKFold
X, y, g = np.array([feats(r) for r in train], float), np.array([r["acc"] for r in train]), np.array([r["root"] + r["loop"] for r in train])
for tr, te in GroupKFold(5).split(X, y, g):
    m = LogisticRegression(C=1.0, max_iter=2000).fit(X[tr], y[tr])
    gk = defaultdict(list)
    for i in tr:
        if train[i]["acc"]:
            gk[(train[i]["fam"], min(train[i]["n_acc"], 2))].append(train[i]["gain"])
    for i, p in zip(te, m.predict_proba(X[te])[:, 1]):
        train[i]["h_cv"] = float(p)
        train[i]["eg_cv"] = fmean(gk.get((train[i]["fam"], min(train[i]["n_acc"], 2))) or [0.02])
full = fmean(sum(r["gain"] for r in rows) for rows in allrows)
print("\nreplay on %d real loops; full loop: in-loop progress %.3f, %.1f experiments" % (len(allrows), full, fmean(len(rows) for rows in allrows)))
print("| rule | experiments used | share of experiments saved | share of in-loop progress kept |\n|---|---|---|---|")


def replay(stop):
    used, kept = [], []
    for rows in allrows:
        g = 0.0
        for i, r in enumerate(rows):
            if stop(r):
                used.append(i)
                break
            g += r["gain"]
        else:
            used.append(len(rows))
        kept.append(g)
    return fmean(used), fmean(kept)


for k in (5, 8, 12, 16, 20):
    u, g = replay(lambda r: r["since"] >= k)
    print("| patience %d (stop after %d experiments without an accept) | %.1f | %.0f%% | %.0f%% |" % (k, k, u, 100 * (1 - u / fmean(len(r) for r in allrows)), 100 * g / full))
eg = {k: fmean(v) for k, v in gains.items()}
for price in (0.02, 0.01, 0.005, 0.0025):      # cost of one experiment, in progress units (cost / value of progress)
    u, g = replay(lambda r: r["h_cv"] * r["eg_cv"] < price)
    print("| index rule (fitted on other loops), one experiment priced at %.4f progress | %.1f | %.0f%% | %.0f%% |" % (price, u, 100 * (1 - u / fmean(len(r) for r in allrows)), 100 * g / full))
