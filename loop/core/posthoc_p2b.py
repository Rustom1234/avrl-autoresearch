#!/usr/bin/env python3
"""P2B, post hoc and descriptive (not in the preregistration): where does a loop's gain come from?

    .venv/bin/python -m loop.core.posthoc_p2b --run loop/runs/P2B

Reads analysis/loops.csv, every loop's final idea list (state.json) and ledger. Prints four tables:
1. final settings of the kept script by cell (total batch, depth below 8);
2. sealed gain by final settings, and the share of the variance of gain they explain;
3. which idea lists held the two batch-size cuts and a shallower model, at which position, and what happened.
4. the same match on the first lists of the offline ladder (loop/runs/LADDER1), which were written before the run.
Ideas are matched by keyword, so the counts in tables 3 and 4 are approximate.
"""
import argparse, collections, csv, json, re, statistics as st
from pathlib import Path

CELLS = ("SS", "ST", "F", "R", "H", "M", "QT")
SKIP = re.compile(r"ramp|curriculum|schedule|larger|stagger|dynamic|adaptive|sequence|warmup", re.I)
QUARTER = re.compile(r"2\s*(\*\*|\^)\s*17|131[,.]?072|131\s*k|quarter", re.I)
HALF = re.compile(r"2\s*(\*\*|\^)\s*18|262[,.]?144|262\s*k|halv|half|smaller total batch|reduce total_batch|lower total batch", re.I)
SHALLOW = re.compile(r"shallow|depth\s*(=|to|from 8 to|of)?\s*6\b|depth 8 to 6|fewer layers", re.I)


def idea_kind(b):
    text = b["title"] + " " + b["spec"][:200]
    if SHALLOW.search(text):
        return "shallower"
    if "batch" not in (b["title"] + b["spec"]).lower() or SKIP.search(b["title"]):
        return None
    if QUARTER.search(text):
        return "batch 2^17"
    return "batch 2^18" if HALF.search(text) else None


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--run", default="loop/runs/P2B")
    run = Path(ap.parse_args().run)
    rows = [r for r in csv.DictReader(open(run / "analysis/loops.csv"))]
    skipped = [r["loop_id"] for r in rows if not r["final_tbs"]]
    rows = [r for r in rows if r["final_tbs"]]
    for r in rows:
        r["tbs"], r["shallow"], r["gain"] = int(float(r["final_tbs"])), int(r["final_depth"]) < 8, float(r["sealed_gain_global"])
    print("loops without a readable final batch size (left out of tables 1 and 2): %s\n" % skipped)

    print("1. Final settings by cell (loops)\n| Cell | batch 2^17 | 2^18 | 2^19 | depth below 8 | no accepted change |\n|---|---|---|---|---|---|")
    for c in CELLS:
        r = [x for x in rows if x["cell"] == c]
        b = collections.Counter(x["tbs"] for x in r)
        print("| %s | %d | %d | %d | %d | %d |" % (c, b[2**17], b[2**18], b[2**19], sum(x["shallow"] for x in r),
                                                 sum(int(x["accepts"]) == 0 for x in r)))

    groups = collections.defaultdict(list)
    for x in rows:
        groups[(x["tbs"], x["shallow"])].append(x["gain"])
    print("\n2. Sealed gain by final settings (all cells)\n| Total batch | Depth below 8 | Loops | Mean gain | Min | Max |\n|---|---|---|---|---|---|")
    for (tbs, sh), v in sorted(groups.items()):
        name = "2^%d" % (tbs.bit_length() - 1) if tbs & (tbs - 1) == 0 else "%d" % tbs
        print("| %s | %s | %d | %.4f | %.4f | %.4f |" % (name, "yes" if sh else "no", len(v), st.mean(v), min(v), max(v)))
    y = [x["gain"] for x in rows]
    sst = sum((v - st.mean(y)) ** 2 for v in y)
    by_cell = collections.defaultdict(list)
    for x in rows:
        by_cell[x["cell"]].append(x["gain"])
    for name, g in (("the two final settings", groups), ("the cell", by_cell)):
        sse = sum((v - st.mean(vs)) ** 2 for vs in g.values() for v in vs)
        print("share of the variance of gain explained by %s: %.2f" % (name, 1 - sse / sst))

    print("\n3. Ideas on the list (loops with the idea; median position on the list; loops that tried it; loops that kept it)")
    print("| Cell | batch 2^18 | batch 2^17 | shallower model |\n|---|---|---|---|")
    for c in CELLS:
        out = []
        for kind in ("batch 2^18", "batch 2^17", "shallower"):
            has, pos, tried, kept = 0, [], 0, 0
            for d in sorted(run.glob(c + "_??")):
                backlog = json.load(open(d / "state.json"))["backlog"]
                ids = [(i, b["id"]) for i, b in enumerate(backlog) if idea_kind(b) == kind]
                if not ids:
                    continue
                mine = [r for r in map(json.loads, open(d / "ledger.jsonl")) if r["idea_id"] in {x[1] for x in ids}]
                has, tried, kept = has + 1, tried + bool(mine), kept + any(r["decision"] == "accept" for r in mine)
                pos.append(ids[0][0] + 1)
            out.append("%d; %s; %d; %d" % (has, ("%g" % st.median(pos)) if pos else "-", tried, kept))
        print("| %s | %s |" % (c, " | ".join(out)))

    ladder = run.parent / "LADDER1/lists.jsonl"          # first lists written before the run, no GPU
    if ladder.exists():
        print("\n4. The offline ladder lists, scored again (lists of 12 with: a 2^18 cut; a 2^17 cut; both among the first 5 ideas)")
        print("| Model | 2^18 cut | 2^17 cut | both in the first 5 | median position of the 2^18 cut |\n|---|---|---|---|---|")
        by = collections.defaultdict(list)
        for r in map(json.loads, open(ladder)):
            kinds = [idea_kind(b) for b in r["items"] or []]
            by[r["model"]].append([kinds.index(k) + 1 if k in kinds else None for k in ("batch 2^18", "batch 2^17")])
        for m, v in by.items():
            half = [h for h, _ in v if h]
            print("| %s | %d | %d | %d | %s |" % (m, len(half), sum(q is not None for _, q in v),
                                                 sum(bool(h and q and max(h, q) <= 5) for h, q in v), ("%g" % st.median(half)) if half else "-"))


if __name__ == "__main__":
    main()
