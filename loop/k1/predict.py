#!/usr/bin/env python3
"""K1: gap score of each BG1 task from the offline idea lists (loop/k1/bg1_lists.jsonl). No outcome is read."""
import json, sys
from pathlib import Path
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from gap import gap
L = {}
for r in map(json.loads, open(HERE / "bg1_lists.jsonl")):
    if r["items"]:
        L.setdefault((r["task"], r["model"]), []).append(r["items"])
out = {}
for t in sorted({k[0] for k in L}):
    g5 = gap(L[(t, "sonnet")], L[(t, "haiku")], 5)
    out[t] = {"gap_top5": g5[0], "d_cheap": g5[1], "d_strong": g5[2], "gap_top10": gap(L[(t, "sonnet")], L[(t, "haiku")], 10)[0],
              "gap_all": gap(L[(t, "sonnet")], L[(t, "haiku")], 30)[0],
              "reverse_top5": gap(L[(t, "haiku")], L[(t, "sonnet")], 5)[0]}
    print("%-28s gap top5 %.2f (to haiku %.3f, to own %.3f)  top10 %.2f  all %.2f  reverse (haiku ideas missing from sonnet) %.2f" % (
        t, g5[0], g5[1], g5[2], out[t]["gap_top10"], out[t]["gap_all"], out[t]["reverse_top5"]))
(HERE / "bg1_gap.json").write_text(json.dumps(out, indent=1))
