#!/usr/bin/env python3
"""K1: the knowledge gap between a strong and a cheap planner, measured from their idea lists alone (no GPU run).

    python loop/k1/gap.py

For one task, take several first idea lists from the strong planner and several from a cheap planner. Embed every
idea (title and spec) with a small sentence-embedding model. For each strong idea among the first TOP of its list:
  d_cheap  = distance to the nearest idea in the cheap planner's lists (1 - cosine similarity)
  d_strong = distance to the nearest idea in the OTHER lists of the strong planner
gap = mean d_cheap / mean d_strong. About 1 means the cheap planner covers the strong planner's leading ideas as
well as the strong planner covers itself from one sample to the next, so there is nothing only the strong model
knows. Well above 1 means the strong planner leads with ideas the cheap planner does not produce.
To keep the comparison fair, the cheap pool is cut to the same number of lists as the "other strong lists" pool.
"""
import json, os, random, sys
from pathlib import Path
from statistics import fmean

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")
import numpy as np
from sentence_transformers import SentenceTransformer

REPO = Path(__file__).resolve().parents[2]
_model = None


def embed(items):
    global _model
    if _model is None:
        _model = SentenceTransformer("all-MiniLM-L6-v2", local_files_only=True)
    return _model.encode(["%s. %s" % (i["title"], i["spec"]) for i in items], normalize_embeddings=True)


def gap(strong_lists, cheap_lists, top=5, seed=0):
    """strong_lists, cheap_lists: lists of idea lists. Returns (gap, mean d_cheap, mean d_strong)."""
    rng = random.Random(seed)
    S = [embed(l) for l in strong_lists if l]
    C = [embed(l) for l in cheap_lists if l]
    dc, ds = [], []
    for i, s in enumerate(S):
        others = [x for j, x in enumerate(S) if j != i]
        k = min(len(others), len(C))
        if k == 0:
            continue
        po = np.vstack(rng.sample(others, k))
        pc = np.vstack(rng.sample(C, k))
        for v in s[:top]:
            ds.append(1 - float((po @ v).max()))
            dc.append(1 - float((pc @ v).max()))
    return fmean(dc) / fmean(ds), fmean(dc), fmean(ds)


if __name__ == "__main__":
    # Retrospective check on the two tasks whose outcome is known.
    L = {}
    for line in open(REPO / "loop/runs/LADDER1/lists.jsonl"):
        r = json.loads(line)
        if r.get("ok") and r.get("items"):
            L.setdefault(r["model"], []).append(r["items"])
    print("GPT task (known: only the strong planner led with the winning move; weak planners got 0.2 to 0.4 of its gain)")
    for m in ("haiku", "gemma4-12b", "qwen35-9b", "qwen35-4b", "sonnet"):
        if m == "sonnet":
            a, b = L["sonnet"][:6], L["sonnet"][6:]
            g = gap(a, b)
        else:
            g = gap(L["sonnet"], L[m])
        print("  sonnet vs %-11s gap %.2f  (to cheap %.3f, to own other lists %.3f)" % ((m,) + g))
    cg = {"SS": [], "HH": []}
    for cell in cg:
        for i in range(1, 7):
            s = json.loads((REPO / "loop/runs/CG1" / ("%s_%02d" % (cell, i)) / "state.json").read_text())
            cg[cell].append(sorted([b for b in s["backlog"] if int(b["id"][1:]) <= 30], key=lambda b: b["id"]))
    print("CIFAR task (known: Haiku did at least as well as Sonnet)")
    print("  sonnet vs haiku       gap %.2f  (to cheap %.3f, to own other lists %.3f)" % gap(cg["SS"], cg["HH"]))
    for top in (3, 10, 30):
        print("  top %d: GPT sonnet vs haiku %.2f, vs qwen35-4b %.2f; CIFAR sonnet vs haiku %.2f" % (
            top, gap(L["sonnet"], L["haiku"], top)[0], gap(L["sonnet"], L["qwen35-4b"], top)[0], gap(cg["SS"], cg["HH"], top)[0]))
