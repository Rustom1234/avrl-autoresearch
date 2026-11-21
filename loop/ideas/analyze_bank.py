#!/usr/bin/env python3
"""Read the stall bank: which source's ideas pass, and who can tell beforehand.

    python3 loop/ideas/analyze_bank.py --root loop/runs/B1 [--e1 loop/runs/E1]

Works on a run in progress (ideas not yet tested are left out). An idea's outcome:
  ran       a job ran for it (not aborted by the coder, not stopped by the static check)
  z1        z of its first run against the frozen incumbent (positive = better); the engine rejects below 0.75,
            accepts at 2.0 or more, and reruns in between (accept if z over both runs is at least 1.5)
  promising z1 at least 0.75 (it earned a second run or a direct accept)
  accepted  the engine's final decision
An idea proposed by several sources counts for each of them.
"""
import argparse, glob, json, math, random
from collections import defaultdict
from pathlib import Path

ap = argparse.ArgumentParser()
ap.add_argument("--root", required=True)
ap.add_argument("--e1", default=None)
ap.add_argument("--stage", default="s", help="shard folder prefix: s = first stage (the loop's own coder), cs = Sonnet coder")
a = ap.parse_args()
root = Path(a.root).resolve()
TAU = 0.824
NAME = {"haiku": "Haiku", "sonnet": "Sonnet", "opus": "Opus"}


def outcomes(d):
    out = {}
    for f in sorted(glob.glob(str(d / (a.stage + "[0-9]*") / "ledger.jsonl"))):
        for l in open(f):
            r = json.loads(l)
            iid = r.get("idea_id")
            if not iid or not iid.startswith("b"):
                continue
            o = out.setdefault(iid, {"ran": False, "z1": None, "accepted": False, "crash": False, "cost": 0.0,
                                     "first": None, "runs": 0, "z2": None})
            o["cost"] += sum(c.get("cost_usd_nominal") or 0 for c in r.get("model_calls") or [])
            if o["first"] is None:
                o["first"] = r["decision"]
            if r.get("job_id"):
                o["ran"] = True
                o["runs"] += 1
            if r.get("z1") is not None and o["z1"] is None:
                o["z1"] = r["z1"]
            if r.get("z2") is not None:
                o["z2"] = r["z2"]
            if r["decision"] == "crash":
                o["crash"] = True
            if r["decision"] == "accept":
                o["accepted"] = True
    return out


rows = []      # one per (proposal, source)
ideas_all = {}
for d in sorted(p.parent for p in root.glob("*/ideas.json")):
    out = outcomes(d)
    ask = {(r["model"], r["ctx"]): r for r in map(json.loads, open(d / "proposals.jsonl"))}
    for p in json.loads((d / "props.json").read_text()):
        o = out.get(p["idea"])
        if o is None:
            continue
        done = o["first"] is not None and not (o["first"] == "replicate" and o["runs"] < 2)
        if not done:
            continue
        rows.append(dict(p, loop=d.name, **o, ask_cost=ask[(p["model"], p["ctx"])]["cost_usd_nominal"] / max(1, len(ask[(p["model"], p["ctx"])]["items"]))))
    for i in json.loads((d / "ideas.json").read_text()):
        if i["id"] in out:
            ideas_all[(d.name, i["id"])] = dict(i, **out[i["id"]])
n_ideas = sum(len(json.loads((d / "ideas.json").read_text())) for d in (p.parent for p in root.glob("*/ideas.json")))
print("ideas decided so far: %d of %d\n" % (len({(r["loop"], r["idea"]) for r in rows}), n_ideas))


def zc(z):
    return max(-5.0, min(5.0, z))


def line(label, x):
    n = len(x)
    if not n:
        return "| %s | 0 | | | | | | | | |" % label
    ran = [r for r in x if r["ran"] and r["z1"] is not None]
    prom = sum(r["z1"] >= 0.75 for r in ran)
    return "| %s | %d | %.2f | %.2f | %s | %d (%.3f) | %d (%.3f) | %.3f | %.3f | %.2f |" % (
        label, n, 1 - sum(r["ran"] for r in x) / n, sum(r["crash"] for r in x) / n,
        "%+.2f" % (sum(zc(r["z1"]) for r in ran) / len(ran)) if ran else "-",
        prom, prom / n, sum(r["accepted"] for r in x), sum(r["accepted"] for r in x) / n,
        sum(r["ask_cost"] for r in x) / n, sum(r["cost"] for r in x) / n,
        sum(r["sim_hist_bge"] < TAU for r in x) / n)


HEAD = ("| %s | ideas | no run | crashed | mean z1 | promising (rate) | accepted (rate) | ask USD per idea | test USD per idea "
        "| share new to the loop |\n|---|---|---|---|---|---|---|---|---|---|")
print("By source (planner model, context), all loops pooled\n" + HEAD % "source")
for m in ("haiku", "sonnet", "opus"):
    for c in ("c", "f"):
        print(line("%s, %s" % (NAME[m], "continued" if c == "c" else "fresh"), [r for r in rows if r["model"] == m and r["ctx"] == c]))
print("\nBy planner model\n" + HEAD % "model")
for m in ("haiku", "sonnet", "opus"):
    print(line(NAME[m], [r for r in rows if r["model"] == m]))
print("\nBy context\n" + HEAD % "context")
for c in ("c", "f"):
    print(line("continued" if c == "c" else "fresh", [r for r in rows if r["ctx"] == c]))
print("\nBy loop\n" + HEAD % "loop")
for lp in sorted({r["loop"] for r in rows}):
    print(line(lp, [r for r in rows if r["loop"] == lp]))
print("\nBy closeness to the ideas the loop had already proposed (bge cosine)\n" + HEAD % "closeness")
for lo, hi, lab in ((0, 0.74, "below 0.74"), (0.74, TAU, "0.74 to 0.824"), (TAU, 0.88, "0.824 to 0.88"), (0.88, 2, "0.88 and above")):
    print(line(lab, [r for r in rows if lo <= r["sim_hist_bge"] < hi]))


# ---------- permutation tests inside loops (labels shuffled among the ideas of the same loop)
def perm(key, a_val, b_val, stat, n=5000):
    rng = random.Random(1)
    byloop = defaultdict(list)
    for r in rows:
        if r[key] in (a_val, b_val):
            byloop[r["loop"]].append(r)

    def diff(lab):
        xa = [stat(r) for lp in byloop for r, g in zip(byloop[lp], lab[lp]) if g == a_val]
        xb = [stat(r) for lp in byloop for r, g in zip(byloop[lp], lab[lp]) if g == b_val]
        return (sum(xa) / len(xa) - sum(xb) / len(xb)) if xa and xb else 0.0

    lab = {lp: [r[key] for r in v] for lp, v in byloop.items()}
    obs = diff(lab)
    hit = 0
    for _ in range(n):
        sh = {lp: rng.sample(v, len(v)) for lp, v in lab.items()}
        hit += abs(diff(sh)) >= abs(obs) - 1e-12
    return obs, hit / n


prom = lambda r: float(r["ran"] and r["z1"] is not None and r["z1"] >= 0.75)
zval = lambda r: zc(r["z1"]) if (r["ran"] and r["z1"] is not None) else -5.0
print("\nDifferences (first minus second), permutation p inside loops")
print("| comparison | promising rate | p | mean z1, a failed run counted as -5 | p |\n|---|---|---|---|---|")
for key, x, y in (("ctx", "f", "c"), ("model", "opus", "haiku"), ("model", "sonnet", "haiku"), ("model", "opus", "sonnet")):
    if any(r[key] == x for r in rows) and any(r[key] == y for r in rows):
        d1, p1 = perm(key, x, y, prom)
        d2, p2 = perm(key, x, y, zval)
        print("| %s minus %s | %+.3f | %.3f | %+.2f | %.3f |" % (x, y, d1, p1, d2, p2))


# ---------- who can tell beforehand
def auc(pos, neg):
    if not pos or not neg:
        return float("nan")
    return sum((p > q) + 0.5 * (p == q) for p in pos for q in neg) / (len(pos) * len(neg))


def spearman(x, y):
    def rk(v):
        o = sorted(range(len(v)), key=lambda i: v[i])
        r = [0.0] * len(v)
        i = 0
        while i < len(o):
            j = i
            while j + 1 < len(o) and v[o[j + 1]] == v[o[i]]:
                j += 1
            for k in range(i, j + 1):
                r[o[k]] = (i + j) / 2
            i = j + 1
        return r
    if len(x) < 3:
        return float("nan")
    a_, b_ = rk(x), rk(y)
    ma, mb = sum(a_) / len(a_), sum(b_) / len(b_)
    den = math.sqrt(sum((u - ma) ** 2 for u in a_) * sum((v - mb) ** 2 for v in b_))
    return sum((u - ma) * (v - mb) for u, v in zip(a_, b_)) / den if den else float("nan")


def scorer_report(name, truth):
    """truth: {(loop, id): dict(ran, z1, accepted)}"""
    files = sorted(glob.glob(str(root / "scores" / ("%s_*.jsonl" % name))))
    if not files:
        return
    sc = defaultdict(lambda: defaultdict(dict))     # model -> (loop, id) -> rep -> p
    cost = defaultdict(float)
    for f in files:
        for r in map(json.loads, open(f)):
            cost[r["model"]] += r["cost_usd_nominal"]
            for iid, p in r["scores"].items():
                sc[r["model"]][(r["loop"], iid)][r["rep"]] = p
    print("\nScores given before the test (%s set): do they rank the ideas that turn out promising above the rest?" % name)
    print("| scorer | ideas scored and decided | promising | AUC promising, inside loops | AUC accepted | rank correlation with z1 "
          "| promising found in the top fifth | agreement of the two passes | USD per idea |\n|---|---|---|---|---|---|---|---|---|")
    for m in ("haiku", "sonnet", "opus", "all three"):
        if m == "all three":
            keys = set.intersection(*[set(sc[x]) for x in ("haiku", "sonnet", "opus")]) if all(sc[x] for x in ("haiku", "sonnet", "opus")) else set()
            val = {k: sum(sum(sc[x][k].values()) / len(sc[x][k]) for x in ("haiku", "sonnet", "opus")) / 3 for k in keys}
        else:
            val = {k: sum(v.values()) / len(v) for k, v in sc[m].items()}
        ks = [k for k in val if k in truth]
        if not ks:
            continue
        byloop = defaultdict(list)
        for k in ks:
            byloop[k[0]].append(k)
        aucs, w, rhos, top, npos = [], [], [], 0, 0
        for lp, kk in byloop.items():
            pos = [val[k] for k in kk if prom(truth[k])]
            neg = [val[k] for k in kk if not prom(truth[k])]
            if pos and neg:
                aucs.append(auc(pos, neg))
                w.append(len(pos))
            rhos.append(spearman([val[k] for k in kk], [zval(truth[k]) for k in kk]))
            cut = sorted((val[k] for k in kk), reverse=True)[max(0, math.ceil(len(kk) / 5) - 1)]
            top += sum(1 for k in kk if prom(truth[k]) and val[k] >= cut)
            npos += len(pos)
        # pooled AUC on ranks inside loops
        pr = {}
        for lp, kk in byloop.items():
            srt = sorted(kk, key=lambda k: val[k])
            for i, k in enumerate(srt):
                pr[k] = (i + 0.5) / len(kk)
        a_acc = auc([pr[k] for k in ks if truth[k]["accepted"]], [pr[k] for k in ks if not truth[k]["accepted"]])
        rel = "-"
        if m != "all three":
            two = [(v[0], v[1]) for k, v in sc[m].items() if 0 in v and 1 in v]
            rel = "%.2f" % spearman([t[0] for t in two], [t[1] for t in two]) if two else "-"
        rr = [r for r in rhos if r == r]
        print("| %s | %d | %d | %s | %s | %s | %s | %s | %s |" % (
            m, len(ks), npos, "%.2f" % (sum(x * y for x, y in zip(aucs, w)) / sum(w)) if w else "-",
            "%.2f" % a_acc if a_acc == a_acc else "-", "%+.2f" % (sum(rr) / len(rr)) if rr else "-",
            "%d of %d" % (top, npos) if npos else "-", rel,
            "%.4f" % (cost[m] / max(1, len(sc[m])) if m != "all three" else sum(cost.values()) / max(1, len(val)))))


scorer_report("bank", {k: v for k, v in ideas_all.items()
                       if v["first"] is not None and not (v["first"] == "replicate" and v["runs"] < 2)})
retro = {}
for f in sorted(glob.glob(str(root / "retro" / "*.json"))):
    for i in json.loads(Path(f).read_text()):
        retro[(Path(f).stem, i["id"])] = {"ran": i["ran"], "z1": i["z1"], "accepted": i["accepted"]}
if retro:
    scorer_report("retro", retro)
    x = list(retro.values())
    print("\nretro set: %d ideas, %d ran, %d promising, %d accepted" % (
        len(x), sum(v["ran"] for v in x), sum(prom(v) for v in x), sum(v["accepted"] for v in x)))

# ---------- the E1 continuation of the same loops (same model, same context, slots 40 to 79)
if a.e1:
    print("\nE1 continuation of the same loops (planner Haiku, continued, the loop's own backlog and refills)")
    print("| loop | ideas tested | promising (rate) | accepted |\n|---|---|---|---|")
    T = [0, 0, 0]
    for lp in sorted({r["loop"] for r in rows}):
        seen, n, pr, acc = set(), 0, 0, set()
        for l in open(Path(a.e1) / lp / "ledger.jsonl"):
            r = json.loads(l)
            if r["slot"] < 40:
                continue
            if r["decision"] == "accept":
                acc.add(r["idea_id"])
            if r["kind"] == "candidate" and r["idea_id"] not in seen:
                seen.add(r["idea_id"])
                n += 1
                pr += int(r.get("z1") is not None and r["z1"] >= 0.75)
        print("| %s | %d | %d (%.3f) | %d |" % (lp, n, pr, pr / max(1, n), len(acc)))
        T = [T[0] + n, T[1] + pr, T[2] + len(acc)]
    print("| all | %d | %d (%.3f) | %d |" % (T[0], T[1], T[1] / max(1, T[0]), T[2]))
