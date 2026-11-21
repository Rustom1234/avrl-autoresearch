#!/usr/bin/env python3
"""Analysis of the main curation study C1, as written down in curation/design/C1_PREREG.md.

    python curation/loop/analysis_c1.py --root curation/runs/C1 [--pool v2] [--out curation/runs/C1/analysis.json]

Per loop: sealed gain = mean sealed bpb of the raw pool minus the loop's confirmation mean; dollars = model dollars.
Per cell P, with anchors CC (all cheap) and SS (all strong):
  gap recovered = (gain P - gain CC) / (gain SS - gain CC)
  extra cost    = (dollars P - dollars CC) / (dollars SS - dollars CC)
  advantage     = gap recovered - extra cost
Intervals: bootstrap over loops within cells (the anchors are resampled too), 10,000 resamples, fixed seed.
Gates: G2 (the anchors differ) and V (the damaged loop DH gains less than SS), exact one-sided permutation tests.
A loop that lost a slot to a provider or server failure is listed and left out; the rule is to run it again.
"""
import argparse, json, random, statistics as st, sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from table import raw_runs, perm_p

STRONG, CHEAP, DAMAGED = "SS", "CC", "DH"
CURVE_AT = (1, 2, 4, 8, 12)
N_BOOT, BOOT_SEED = 10000, 20261110


def lost_slot(ledger):
    return any(c.get("error_class") == "infra" and not c["ok"] for r in ledger for c in r["calls"])


def curve(ledger):
    """Dev gain of the current best over the starting file after each slot in CURVE_AT (same seed for both)."""
    start = next((r["inc"] for r in ledger if r.get("inc") is not None), None)   # slots that fail to run store none
    best, out = start, {}
    for r in ledger:
        if r["decision"] == "accept":
            best = r["dev"]
        if r["slot"] in CURVE_AT:
            out[r["slot"]] = 0.0 if start is None else start - best
    return out


def load(root, raw_sealed):
    loops = []
    for f in sorted(Path(root).glob("*/final.json")):
        d = json.loads(f.read_text())
        led = [json.loads(l) for l in open(f.parent / "ledger.jsonl")]
        state = json.loads((f.parent / "state.json").read_text())
        loops.append({"id": d["loop_id"], "cell": d["cell"], "sealed": d["sealed_bpb"], "usd": d["model_usd"],
                      "gain": None if d["sealed_bpb"] is None else raw_sealed - d["sealed_bpb"],
                      "n_accept": d["n_accept"], "infra_waits": d.get("infra_failures", 0),
                      "resumes": state.get("resumes", 0), "lost_slot": lost_slot(led), "curve": curve(led)})
    return loops


def advantage(g, c, p):
    """(gap recovered, extra cost, advantage) of cell p from dicts of cell mean gain and cell mean dollars."""
    gr = (g[p] - g[CHEAP]) / (g[STRONG] - g[CHEAP])
    xc = (c[p] - c[CHEAP]) / (c[STRONG] - c[CHEAP])
    return gr, xc, gr - xc


def bootstrap(cells, n_boot=N_BOOT, seed=BOOT_SEED):
    """cells: {cell: [(gain, usd), ...]}. Returns {cell: {"adv": [lo, hi], "gap": [lo, hi], "p_pos": share above 0}}."""
    rng, names = random.Random(seed), sorted(cells)
    draws = {p: [] for p in names if p not in (STRONG, CHEAP)}
    for _ in range(n_boot):
        g, c = {}, {}
        for p in names:
            s = [rng.choice(cells[p]) for _ in cells[p]]
            g[p], c[p] = st.mean(x[0] for x in s), st.mean(x[1] for x in s)
        if g[STRONG] - g[CHEAP] <= 0 or c[STRONG] - c[CHEAP] <= 0:
            continue                                  # the ratio has no meaning in such a resample; counted below
        for p in draws:
            draws[p].append(advantage(g, c, p))
    out = {}
    for p, d in draws.items():
        q = lambda v: [sorted(v)[int(0.025 * len(v))], sorted(v)[int(0.975 * len(v)) - 1]]
        out[p] = {"gap": q([x[0] for x in d]), "adv": q([x[2] for x in d]),
                  "p_pos": sum(x[2] > 0 for x in d) / len(d), "n_valid": len(d)}
    return out


def analyse(loops, raw_sd):
    used = [l for l in loops if l["gain"] is not None and not l["lost_slot"]]
    cells = {}
    for l in used:
        cells.setdefault(l["cell"], []).append((l["gain"], l["usd"]))
    res = {"n_loops": len(loops), "left_out": [l["id"] for l in loops if l not in used],
           "resumed": {l["id"]: l["resumes"] for l in loops if l["resumes"]},
           "infra_waits": {l["id"]: l["infra_waits"] for l in loops if l["infra_waits"]}, "cells": {}}
    g = {p: st.mean(x[0] for x in v) for p, v in cells.items()}
    c = {p: st.mean(x[1] for x in v) for p, v in cells.items()}
    for p, v in sorted(cells.items()):
        res["cells"][p] = {"n": len(v), "gain": g[p], "gain_sd": st.stdev(x[0] for x in v) if len(v) > 1 else None,
                           "usd": c[p], "accepts": st.mean(l["n_accept"] for l in used if l["cell"] == p),
                           "curve": {k: st.mean(l["curve"][k] for l in used if l["cell"] == p and k in l["curve"])
                                     for k in CURVE_AT if any(k in l["curve"] for l in used if l["cell"] == p)}}
    if STRONG not in cells or CHEAP not in cells:
        return res
    ss, cc = [x[0] for x in cells[STRONG]], [x[0] for x in cells[CHEAP]]
    gap = g[STRONG] - g[CHEAP]
    p2 = perm_p(ss, cc)
    res["G2"] = {"gap": gap, "gap_in_sds": gap / raw_sd, "p": p2, "pass": gap >= 2 * raw_sd and p2 <= 0.05}
    if DAMAGED in cells:
        pv = perm_p(ss, [x[0] for x in cells[DAMAGED]])
        res["V"] = {"ss_minus_dh": g[STRONG] - g[DAMAGED], "p": pv, "pass": pv <= 0.05}
    if not res["G2"]["pass"]:
        res["verdict"] = "NO GAP"
        return res
    boot = bootstrap(cells)
    for p in boot:
        gr, xc, adv = advantage(g, c, p)
        res["cells"][p].update(gap_recovered=gr, extra_cost=xc, advantage=adv, adv_ci=boot[p]["adv"],
                               gap_ci=boot[p]["gap"], p_adv_pos=boot[p]["p_pos"])
    res["verdict"] = "SCORED" if res.get("V", {}).get("pass") else "SHALLOW"
    for name, cell in (("H1", "SC"), ("H2", "F1")):
        if cell in boot:
            res[name] = {"cell": cell, "advantage": res["cells"][cell]["advantage"], "ci": boot[cell]["adv"],
                         "ci_above_zero": boot[cell]["adv"][0] > 0,
                         "confirmed": boot[cell]["adv"][0] > 0 and res["verdict"] == "SCORED"}
    return res


def show(res, raw_sealed, raw_sd):
    print("raw pool sealed %.4f (single-run sd %.4f); %d loops read" % (raw_sealed, raw_sd, res["n_loops"]))
    for k in ("left_out", "resumed", "infra_waits"):
        if res[k]:
            print("%s: %s" % (k.replace("_", " "), res[k]))
    print("%-4s %2s %8s %7s %7s %7s %7s %7s  %-16s %6s" % ("cell", "n", "gain", "sd", "usd", "accept", "gap", "cost",
                                                          "advantage (95%)", "P>0"))
    for p, v in res["cells"].items():
        line = "%-4s %2d %8.4f %7s %7.2f %7.1f" % (p, v["n"], v["gain"], "%.4f" % v["gain_sd"] if v["gain_sd"] is not None
                                                 else "", v["usd"], v["accepts"])
        if "advantage" in v:
            line += " %7.2f %7.2f  %5.2f (%5.2f, %5.2f) %5.2f" % (v["gap_recovered"], v["extra_cost"], v["advantage"],
                                                                  v["adv_ci"][0], v["adv_ci"][1], v["p_adv_pos"])
        print(line)
    print("dev gain of the current best after slot k:")
    for p, v in res["cells"].items():
        print("  %-4s %s" % (p, "  ".join("%d: %.4f" % (k, x) for k, x in v["curve"].items())))
    if "G2" in res:
        print("G2 anchors differ: gap %.4f = %.1f sds, p = %.3f: %s" % (
            res["G2"]["gap"], res["G2"]["gap_in_sds"], res["G2"]["p"], "PASS" if res["G2"]["pass"] else "FAIL"))
    if "V" in res:
        print("V damaged loop below SS: difference %.4f, p = %.3f: %s" % (
            res["V"]["ss_minus_dh"], res["V"]["p"], "PASS" if res["V"]["pass"] else "FAIL"))
    for h in ("H1", "H2"):
        if h in res:
            print("%s (%s advantage above 0): %.2f (%.2f, %.2f), interval above zero: %s, confirmed: %s" % (
                h, res[h]["cell"], res[h]["advantage"], res[h]["ci"][0], res[h]["ci"][1], res[h]["ci_above_zero"],
                res[h]["confirmed"]))
    print("verdict: %s" % res.get("verdict", "anchors missing"))


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--root", required=True)
    ap.add_argument("--pool", default="v2")
    ap.add_argument("--out", default=None)
    a = ap.parse_args(argv)
    raw = [r["sealed_bpb"] for r in raw_runs(a.pool)]
    raw_sealed, raw_sd = st.mean(raw), st.stdev(raw)
    loops = load(a.root, raw_sealed)
    res = analyse(loops, raw_sd)
    show(res, raw_sealed, raw_sd)
    if a.out:
        Path(a.out).write_text(json.dumps({"result": res, "loops": loops}, indent=1))
    return 0


if __name__ == "__main__":
    sys.exit(main())
