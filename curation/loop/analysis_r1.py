#!/usr/bin/env python3
"""Analysis of the router test R1, as written down in curation/design/R1_PREREG.md.

    python curation/loop/analysis_r1.py --root curation/runs/R1 [--pool v2b] [--out curation/runs/R1/analysis.json]

The cell table, gate G2 and the advantage intervals come from the frozen C1 analysis (analysis_c1.py), unchanged.
Added here:
- differences in advantage between two cells (learned table minus SC, and between the two tables), with the same
  bootstrap (loops resampled within cells, anchors too, 10,000 resamples, fixed seed);
- how many loops of a cell reach the strong level (sealed gain at least 85 percent of the SS cell mean);
- what the learned tables did: slots with a strong planner, slots with a strong coder, and the first slot that
  started at the top gain level;
- a sensitivity check (--rescored): the same numbers when each loop's final file is scored on four seeds instead
  of two (the two of the loop plus the extra runs written by rescore_finals.py). The main result stays the
  two-seed one.
"""
import argparse, json, random, statistics as st, sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import analysis_c1 as A
from table import raw_runs

PAIRS = (("LA", "SC"), ("LB", "SC"), ("LA", "LB"))
TABLES = ("LA", "LB")
REACH = 0.85
TOP = 14                # noise sds at which the R1 tables stop paying (their top gain level)


def differences(cells, pairs=PAIRS, n_boot=A.N_BOOT, seed=A.BOOT_SEED):
    """cells: {cell: [(gain, usd), ...]}. {"X-Y": {"diff": advantage X minus advantage Y, "ci": [lo, hi]}}."""
    pairs = [p for p in pairs if p[0] in cells and p[1] in cells]
    mean = lambda rows, i: st.mean(x[i] for x in rows)
    g, c = {p: mean(v, 0) for p, v in cells.items()}, {p: mean(v, 1) for p, v in cells.items()}
    point = {(x, y): A.advantage(g, c, x)[2] - A.advantage(g, c, y)[2] for x, y in pairs}
    rng, draws = random.Random(seed), {p: [] for p in pairs}
    for _ in range(n_boot):
        s = {p: [rng.choice(v) for _ in v] for p, v in sorted(cells.items())}
        bg, bc = {p: mean(v, 0) for p, v in s.items()}, {p: mean(v, 1) for p, v in s.items()}
        if bg[A.STRONG] - bg[A.CHEAP] <= 0 or bc[A.STRONG] - bc[A.CHEAP] <= 0:
            continue
        for x, y in pairs:
            draws[x, y].append(A.advantage(bg, bc, x)[2] - A.advantage(bg, bc, y)[2])
    q = lambda v: [sorted(v)[int(0.025 * len(v))], sorted(v)[int(0.975 * len(v)) - 1]]
    return {"%s-%s" % p: {"diff": point[p], "ci": q(draws[p])} for p in pairs}


def route(run_dir):
    """What one routed loop did, from its slot records."""
    cfg = json.loads((run_dir / "config.json").read_text())
    rows = [json.loads(l) for l in open(run_dir / "routes.jsonl")]
    top = next((r["slot"] for r in rows if (r.get("gain_sd") or 0) >= TOP), None)
    return {"strong_planner": sum(r["planner"] == cfg["strong"] for r in rows), "strong_coder": sum(r["coder"] == cfg["strong"] for r in rows),
            "first_slot_at_top": top, "slots": len(rows)}


def rescored(loops, results_dir, raw_sealed, prefix="R1final_"):
    """Copies of the loops with sealed bpb and gain taken over the loop's two confirmation runs plus every extra run
    <prefix><loop id>_s<seed>.json in results_dir. Returns (loops, number of loops that had extra runs)."""
    out, n = [], 0
    for l in loops:
        xs = [json.loads(f.read_text())["sealed_bpb"] for f in sorted(Path(results_dir).glob("%s%s_s*.json" % (prefix, l["id"])))]
        if xs and l["sealed"] is not None:
            sealed = (2 * l["sealed"] + sum(xs)) / (2 + len(xs))
            l, n = dict(l, sealed=sealed, gain=raw_sealed - sealed), n + 1
        out.append(l)
    return out, n


def by_cell(loops):
    cells = {}
    for l in loops:
        if l["gain"] is not None and not l["lost_slot"]:
            cells.setdefault(l["cell"], []).append((l["gain"], l["usd"]))
    return cells


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--root", required=True)
    ap.add_argument("--pool", default="v2b")
    ap.add_argument("--out", default=None)
    ap.add_argument("--rescored", default=None, help="folder with the extra runs of the final files (sensitivity check)")
    a = ap.parse_args(argv)
    raw = [r["sealed_bpb"] for r in raw_runs(a.pool)]
    raw_sealed, raw_sd = st.mean(raw), st.stdev(raw)
    loops = A.load(a.root, raw_sealed)
    res = A.analyse(loops, raw_sd)
    A.show(res, raw_sealed, raw_sd)
    print("(the H1 line and the verdict line above belong to C1's rules; R1 has no damaged cell and its rules are below)")
    cells = by_cell(loops)
    extra = {}
    if res.get("G2", {}).get("pass"):
        bar = REACH * res["cells"][A.STRONG]["gain"]
        extra["reach"] = {p: [sum(x[0] >= bar for x in v), len(v)] for p, v in sorted(cells.items())}
        print("loops at the strong level (sealed gain at least %.4f): %s" % (bar, "  ".join(
            "%s %d/%d" % (p, k, n) for p, (k, n) in extra["reach"].items())))
        extra["differences"] = differences(cells)
        for name, d in extra["differences"].items():
            print("advantage %s: %+.2f (%+.2f, %+.2f)" % (name, d["diff"], d["ci"][0], d["ci"][1]))
        for h, cell in (("R1", "LA"), ("R2", "LB")):
            if cell in res["cells"]:
                lo = res["cells"][cell]["adv_ci"][0]
                print("%s (%s advantage above 0): %.2f, lower end %.2f: %s" % (h, cell, res["cells"][cell]["advantage"], lo,
                                                                               "holds" if lo > 0 else "does not hold"))
        if "LA-SC" in extra["differences"]:
            d = extra["differences"]["LA-SC"]
            print("R3 (LA advantage above SC): %+.2f, lower end %+.2f: %s" % (d["diff"], d["ci"][0],
                                                                              "holds" if d["ci"][0] > 0 else "does not hold"))
    extra["routes"] = {l["id"]: route(Path(a.root) / l["id"]) for l in loops if l["cell"] in TABLES}
    for i, r in extra["routes"].items():
        print("%s: strong planner in %d slots, strong coder in %d, first slot that started at the top level: %s" % (
            i, r["strong_planner"], r["strong_coder"], r["first_slot_at_top"]))
    if a.rescored:
        loops4, n = rescored(loops, a.rescored, raw_sealed)
        res4 = A.analyse(loops4, raw_sd)
        print("\nsensitivity, final files scored on four seeds (%d of %d loops have extra runs):" % (n, len(loops)))
        extra["four_seeds"] = {"result": res4, "n_rescored": n}
        if res4.get("G2", {}).get("pass"):
            for p, c in sorted(res4["cells"].items()):
                print("  %-3s gain %.4f" % (p, c["gain"]) + ("  advantage %.2f (%.2f, %.2f)" % (c["advantage"], c["adv_ci"][0], c["adv_ci"][1])
                                                              if c.get("adv_ci") else ""))
            extra["four_seeds"]["differences"] = differences(by_cell(loops4))
            for name, d in extra["four_seeds"]["differences"].items():
                print("  advantage %s: %+.2f (%+.2f, %+.2f)" % (name, d["diff"], d["ci"][0], d["ci"][1]))
        else:
            print("  G2 does not pass on four seeds")
    if a.out:
        Path(a.out).write_text(json.dumps({"result": res, "r1": extra, "loops": loops}, indent=1))
    return 0


if __name__ == "__main__":
    sys.exit(main())
