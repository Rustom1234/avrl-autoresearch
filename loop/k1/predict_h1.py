#!/usr/bin/env python3
"""H1: idea-gap score of each BG1 task between Haiku (strong side) and Gemma 12B (cheap side), from offline first
idea lists (haiku lists: bg1_lists.jsonl; gemma lists: h1_lists.jsonl). No outcome of any Gemma loop is read."""
import json, sys
from pathlib import Path
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from gap import gap
L = {}
for f in ("bg1_lists.jsonl", "h1_lists.jsonl"):
    for r in map(json.loads, open(HERE / f)):
        if r["items"]:
            L.setdefault((r["task"], r["model"]), []).append(r["items"])
out = {}
for t in sorted({k[0] for k in L}):
    g = gap(L[(t, "haiku")], L[(t, "gemma4-12b")], 5)
    s = gap(L[(t, "sonnet")], L[(t, "gemma4-12b")], 5)
    out[t] = {"gap_top5": g[0], "d_cheap": g[1], "d_strong": g[2], "sonnet_vs_gemma_top5": s[0]}
    print("%-28s haiku vs gemma: gap top5 %.2f (to gemma %.3f, to own %.3f)   sonnet vs gemma %.2f" % (t, g[0], g[1], g[2], s[0]))
(HERE / "h1_gap.json").write_text(json.dumps(out, indent=1))
