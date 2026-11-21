"""SR6 table: level 3 chains with the automatic term check against the stored chains without it.

    iso_table.py --tool ../runs/SR6 --base ../runs/SR4 ../runs/SR5

Rules and claims: loop/design/SR6_PREREG.md.
"""
import argparse, glob, json
from pathlib import Path

import numpy as np

import advice2 as A2

PROBS, SRC, CELLS = ["problems_advice", "problems_advice2"], "problems_main", ("000", "100")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tool", required=True)
    ap.add_argument("--base", nargs="+", required=True)
    a = ap.parse_args()
    fails = glob.glob(a.tool + "/trie/*/r*/**/failure.json", recursive=True)
    recs = [json.load(open(f)) for f in glob.glob(a.tool + "/trie/*/r*/**/record.json", recursive=True)]
    print("tool arm: %d sessions, %d failed sessions, %.2f USD nominal, models %s" % (
        len(recs), len(fails), sum(r["cost_usd"] for r in recs), sorted({m for r in recs for m in r["models_used"]})))
    tool, _, _ = A2.load([a.tool], PROBS, SRC)
    base, _, _ = A2.load(a.base, PROBS, SRC)
    tool, base = tool[3], base[3]
    probs = sorted(set.intersection(*[set(d[c]) for d in (tool, base) for c in CELLS]))
    f = lambda d, c, ps: float(np.mean([np.mean(d[c][p]) for p in ps]))
    print("\ncell  told     without check      with check        difference (95%%, resampling %d problems)" % len(probs))
    for c in CELLS:
        nb, nt = [x for p in probs for x in base[c][p]], [x for p in probs for x in tool[c][p]]
        d = A2.boot(lambda ps: f(tool, c, ps) - f(base, c, ps), probs)
        print("%-5s %-8s %2d of %2d (%.2f)    %2d of %2d (%.2f)    %+.3f (%+.3f to %+.3f)" % (
            c, {"000": "nothing", "100": "S"}[c], sum(nb), len(nb), f(base, c, probs), sum(nt), len(nt), f(tool, c, probs), *d))
    t1 = A2.boot(lambda ps: float(np.mean([f(tool, c, ps) - f(base, c, ps) for c in CELLS])), probs)
    print("T1 mean difference over both cells: %+.3f (95%% %+.3f to %+.3f) -> %s" % (*t1, "PASSED" if t1[1] > 0 else "FAILED"))
    same = {"bio_L3_s1", "chem_L3_s2", "chem_L3_s3", "matsci_L3_s1"}       # S and U on the same input
    for name, ps in (("S and U on different inputs", [p for p in probs if p not in same]), ("S and U on the same input", [p for p in probs if p in same])):
        print("  %-28s (%d problems): " % (name, len(ps)) + ", ".join(
            "cell %s %d of %d against %d of %d" % (c, sum(x for p in ps for x in tool[c][p]), sum(len(tool[c][p]) for p in ps),
                                                  sum(x for p in ps for x in base[c][p]), sum(len(base[c][p]) for p in ps)) for c in CELLS))

    print("\nuse of the term check, per chain")
    for c, suffix in (("000", "__G"), ("100", "__R100")):
        rows = []
        for d in sorted(Path(a.tool).glob("trie/*%s/r*" % suffix)):
            parts = [json.loads(l) for f in d.rglob("parts.jsonl") for l in open(f)]
            chain = sorted((json.loads(f.read_text()) for f in d.rglob("record.json")), key=lambda r: len(r["path"]))
            solved = any(r["val"] is not None and r["val"] <= 1e-10 for r in chain)
            mix = any(p.get("mixing") for p in parts)
            allr = any(p.get("mixing") and p.get("single") and all(p["single"].values()) for p in parts)
            rows.append((len(parts), sum(1 for p in parts if p.get("text")), mix, allr, solved, sum(r["n_evals"] for r in chain)))
        n = len(rows)
        print("cell %s: %d chains, complete-law checks per chain %.1f, with a term check %.1f" % (
            c, n, np.mean([r[5] for r in rows]), np.mean([r[1] for r in rows])))
        print("  chains that got 'mixing part RIGHT' at least once: %d of %d; of these solved: %d" % (
            sum(r[2] for r in rows), n, sum(r[2] and r[4] for r in rows)))
        print("  chains that got every group RIGHT at least once (without being exact): %d; of these solved: %d" % (
            sum(r[3] for r in rows), sum(r[3] and r[4] for r in rows)))
        print("  chains that never got the mixing part right: %d; of these solved: %d" % (
            sum(not r[2] for r in rows), sum((not r[2]) and r[4] for r in rows)))


if __name__ == "__main__":
    main()
