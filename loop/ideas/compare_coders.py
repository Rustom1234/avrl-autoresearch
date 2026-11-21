#!/usr/bin/env python3
"""The same ideas against the same frozen incumbent, implemented by two coders.

    python3 loop/ideas/compare_coders.py --root loop/runs/B1 [--a s --b cs]

Folders s0, s1: the loop's own coder (Haiku). Folders cs0, cs1: Sonnet as coder and repairer. Only ideas decided in
both stages are compared (paired).
"""
import argparse, glob, json, random
from collections import defaultdict
from pathlib import Path

ap = argparse.ArgumentParser()
ap.add_argument("--root", required=True)
ap.add_argument("--a", default="s")
ap.add_argument("--b", default="cs")
a = ap.parse_args()
root = Path(a.root).resolve()


def outcomes(d, stage):
    o = {}
    for f in sorted(glob.glob(str(d / (stage + "[0-9]*") / "ledger.jsonl"))):
        for l in open(f):
            r = json.loads(l)
            iid = r.get("idea_id")
            if not iid or not iid.startswith("b"):
                continue
            x = o.setdefault(iid, {"z1": None, "accepted": False, "first": None, "runs": 0, "crash": False, "cost": 0.0})
            x["cost"] += sum(c.get("cost_usd_nominal") or 0 for c in r.get("model_calls") or [])
            if x["first"] is None:
                x["first"] = r["decision"]
            x["runs"] += int(bool(r.get("job_id")))
            if r.get("z1") is not None and x["z1"] is None:
                x["z1"] = r["z1"]
            x["crash"] |= r["decision"] == "crash"
            x["accepted"] |= r["decision"] == "accept"
    return {k: v for k, v in o.items() if v["first"] is not None and not (v["first"] == "replicate" and v["runs"] < 2)}


pairs = []
for d in sorted(p.parent for p in root.glob("*/ideas.json")):
    A, B = outcomes(d, a.a), outcomes(d, a.b)
    src = {i["id"]: i["sources"] for i in json.loads((d / "ideas.json").read_text())}
    for k in sorted(set(A) & set(B)):
        pairs.append((d.name, k, A[k], B[k], src[k]))
print("ideas decided in both stages: %d" % len(pairs))
zc = lambda x: max(-5.0, min(5.0, x["z1"])) if x["z1"] is not None else -5.0
prom = lambda x: x["z1"] is not None and x["z1"] >= 0.75


def row(label, sel):
    n = len(sel)
    if not n:
        return
    f = lambda i, fn: sum(fn(p[i]) for p in sel) / n
    print("| %s | %d | %.2f / %.2f | %+.2f / %+.2f | %d / %d | %d / %d | %.4f / %.4f |" % (
        label, n, f(2, lambda x: x["z1"] is not None), f(3, lambda x: x["z1"] is not None), f(2, zc), f(3, zc),
        sum(prom(p[2]) for p in sel), sum(prom(p[3]) for p in sel), sum(p[2]["accepted"] for p in sel),
        sum(p[3]["accepted"] for p in sel), f(2, lambda x: x["cost"]), f(3, lambda x: x["cost"])))


print("\nEach cell: Haiku coder / Sonnet coder")
print("| ideas from | n | valid run | mean z1 (failed run = -5) | promising | accepted | test USD per idea |\n|---|---|---|---|---|---|---|")
row("all", pairs)
for m in ("haiku", "sonnet", "opus"):
    row("%s planner" % m, [p for p in pairs if any(s[0] == m for s in p[4])])
for c, lab in (("c", "continued context"), ("f", "fresh context")):
    row(lab, [p for p in pairs if any(s[1] == c for s in p[4])])
for lp in sorted({p[0] for p in pairs}):
    row(lp, [p for p in pairs if p[0] == lp])
both = [p for p in pairs if p[2]["z1"] is not None and p[3]["z1"] is not None]
if both:
    d = [min(5, max(-5, p[3]["z1"])) - min(5, max(-5, p[2]["z1"])) for p in both]
    rng, m = random.Random(2), sum(d) / len(d)
    hit = sum(abs(sum(x * rng.choice((-1, 1)) for x in d) / len(d)) >= abs(m) for _ in range(5000)) / 5000
    print("\nideas with a valid run under both coders: %d; mean z1 Sonnet coder minus Haiku coder: %+.2f (sign-flip p = %.3f)" % (len(both), m, hit))
    print("promising under: Haiku only %d, Sonnet only %d, both %d" % (
        sum(prom(p[2]) and not prom(p[3]) for p in pairs), sum(prom(p[3]) and not prom(p[2]) for p in pairs),
        sum(prom(p[2]) and prom(p[3]) for p in pairs)))
