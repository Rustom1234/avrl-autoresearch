#!/usr/bin/env python3
"""Progress and outcome table of a run of the ShinkaEvolve packing task.

    python loop/shinka/progress.py --root loop/runs/S1

Per loop: evaluations done, best valid sum of radii over all jobs, and the evaluation index and model dollars at
which it first reached 2.6359 and 2.635977 (model dollars = list-price cost of all calls made up to that job).
"""
import argparse, glob, json
from pathlib import Path

ap = argparse.ArgumentParser()
ap.add_argument("--root", required=True)
a = ap.parse_args()
LEVELS = (2.6359, 2.635977)
print("| loop | evaluations | best sum | exact-valid | " + " | ".join("evals, USD to %.6f" % l for l in LEVELS) + " | last improvement at | total USD |\n|---|---|---|---|---|---|---|---|")
for d in sorted(Path(a.root).glob("s26_*")):
    if not (d / "ledger.jsonl").exists():
        continue
    calls = [json.loads(l) for l in open(d / "calls.jsonl")]
    rows = [json.loads(l) for l in open(d / "ledger.jsonl")]
    rows = [r for r in rows if r.get("charged")]
    best, hit, last, ev = 0.0, {}, 0, True
    for i, r in enumerate(rows, 1):
        v = -(r.get("official_val_bpb") or 0.0)
        if r.get("status") == "ok" and v > best + 1e-9:
            best, last = v, i
            f = d / "jobs" / (r.get("job_id") or "") / "result.json"
            ev = json.load(open(f)).get("exact_valid") if f.exists() else None
        for l in LEVELS:
            if l not in hit and best >= l:
                hit[l] = (i, sum(c.get("cost_usd_nominal") or 0 for c in calls if c["ts"] <= r["ts"]))
    print("| %s | %d | %.7f | %s | %s | %d | %.2f |" % (d.name, len(rows), best, ev, " | ".join("%d, %.2f" % hit[l] if l in hit else "not reached" for l in LEVELS),
                                                 last, sum(c.get("cost_usd_nominal") or 0 for c in calls)))
