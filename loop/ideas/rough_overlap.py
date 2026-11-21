#!/usr/bin/env python3
"""Rough first look: do different models propose the same ideas?  No model calls.

    .venv/bin/python loop/ideas/rough_overlap.py [--frac 0.1]

Two proposals count as "the same idea" when the cosine of their sentence embeddings (bge-small-en-v1.5, local
cache, CPU) is at least tau.  tau is set from pairs that are different by construction: two items of one idea list
(the model was asked for distinct ideas).  tau = the 99th percentile of those pairs, so about 1 percent of pairs
known to be different would be called the same.  This is a surface measure and only a first look; the proper
version groups ideas with a calibrated judge.

Part A, clean test: loop/k1/*_lists.jsonl.  Same prompt, no loop state: 6 tasks x 3 models x 3 lists of about 30.
Part B, in the loops: candidate descriptions in the ledgers of BG1_R, H1, E1.
"""
import argparse, glob, json, os, random
from collections import defaultdict
from pathlib import Path
from statistics import fmean

os.environ["HF_HUB_OFFLINE"] = "1"
os.environ.setdefault("OMP_NUM_THREADS", "8")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
import numpy as np

L = Path(__file__).resolve().parents[1]
ap = argparse.ArgumentParser()
ap.add_argument("--frac", type=float, default=1.0, help="fraction of the data to use (memory check)")
a = ap.parse_args()
rng = random.Random(0)

# ---------- load
lists = []  # (task, model, rep, [text])
for f in ("k1/bg1_lists.jsonl", "k1/h1_lists.jsonl"):
    for l in open(L / f):
        r = json.loads(l)
        if r.get("ok"):
            m = {"gemma4-12b": "gemma"}.get(r["model"], r["model"])
            lists.append((r["task"], m, r["rep"], ["%s. %s" % (i.get("title", ""), i.get("spec", "")) for i in r["items"]]))
TASK = {"bbb": "bbb_martins", "caco": "caco2_wang", "clear": "clearance_hepatocyte_az", "lipo": "lipophilicity_astrazeneca", "p26": "pack26", "p32": "pack32"}
loops = {}  # (task, planner, loop) -> [(order, text, accepted)]; an E1 ledger continues and replaces the BG1_R one
for root in ("BG1_R", "H1", "E1"):
    for f in sorted(glob.glob(str(L / "runs" / root / "*" / "ledger.jsonl"))):
        name = Path(f).parent.name
        parts = name.split("_")
        cell = parts[-2]
        if cell == "N":
            continue
        task = next((v for p, v in TASK.items() if name.startswith(p)), parts[0])
        rows = []
        for i, l in enumerate(open(f)):
            d = json.loads(l)
            if d.get("kind") == "candidate" and d.get("description") and not d.get("attempt"):  # first try of an idea only
                rows.append((i, d["description"], d.get("decision") == "accept"))
        loops[(task, cell[0], name)] = rows
if a.frac < 1:
    lists = [x for x in lists if x[0] in ("pack26",)]
    loops = {k: v for k, v in loops.items() if k[0] == "pack26"}
for k in loops:
    loops[k].sort(key=lambda t: t[0])

# ---------- embed
import torch
from sentence_transformers import SentenceTransformer

torch.set_num_threads(8)
model = SentenceTransformer("BAAI/bge-small-en-v1.5", device="cpu")
texts = [t for x in lists for t in x[3]] + [t for v in loops.values() for _, t, _ in v]
E = model.encode(texts, batch_size=64, normalize_embeddings=True, show_progress_bar=False).astype(np.float32)
pos = 0
LE = []
for x in lists:
    LE.append(E[pos:pos + len(x[3])])
    pos += len(x[3])
PE = {}
for k, v in loops.items():
    PE[k] = E[pos:pos + len(v)]
    pos += len(v)
print("ideas in lists: %d, proposals in loops: %d" % (sum(len(x[3]) for x in lists), sum(len(v) for v in loops.values())))

# ---------- threshold from pairs that are different by construction
within = np.concatenate([(e @ e.T)[np.triu_indices(len(e), 1)] for e in LE])
tau = float(np.quantile(within, 0.99))
print("tau = %.3f (99th percentile of %d within-list pairs; 95th %.3f, median %.3f)\n" % (tau, len(within), np.quantile(within, 0.95), np.median(within)))


def cover(A, B, t):
    """share of rows of A that have a match in B"""
    return float(((A @ B.T).max(axis=1) >= t).mean()) if len(A) and len(B) else float("nan")


def part_a(t):
    MODELS = ("gemma", "haiku", "sonnet")
    by = defaultdict(dict)
    for (task, m, rep, _), e in zip(lists, LE):
        by[task][(m, rep)] = e
    tot = defaultdict(list)
    newsame, newother = defaultdict(list), defaultdict(list)
    print("A1. share of one list's ideas that also appear in ANOTHER list (rows: whose ideas; columns: found in a list by)")
    print("| task | from | in gemma | in haiku | in sonnet |\n|---|---|---|---|---|")
    for task in sorted(by):
        for m in MODELS:
            row = []
            for m2 in MODELS:
                v = [cover(e, e2, t) for (a1, r1), e in by[task].items() if a1 == m for (a2, r2), e2 in by[task].items() if a2 == m2 and (a1, r1) != (a2, r2)]
                row.append(fmean(v) if v else float("nan"))
                tot[(m, m2)] += v
            print("| %s | %s | %s |" % (task, m, " | ".join("%.2f" % x for x in row)))
        # new ideas added by one more list, after two lists of model m are known
        for m in MODELS:
            reps = [r for (a1, r) in by[task] if a1 == m]
            for hold in reps:
                base = np.concatenate([by[task][(m, r)] for r in reps if r != hold])
                newsame[m].append(1 - cover(by[task][(m, hold)], base, t))
                for m2 in MODELS:
                    if m2 != m:
                        for (a2, r2), e2 in by[task].items():
                            if a2 == m2:
                                newother[(m, m2)].append(1 - cover(e2, base, t))
    print("\nA2. all tasks pooled, same table")
    print("| from | in gemma | in haiku | in sonnet |\n|---|---|---|---|")
    for m in MODELS:
        print("| %s | %s |" % (m, " | ".join("%.2f" % fmean(tot[(m, m2)]) for m2 in MODELS)))
    print("\nA3. after two lists (about 60 ideas) from one model: share of the NEXT list's ideas that are new")
    print("| known model | next list from the same model | from gemma | from haiku | from sonnet |\n|---|---|---|---|---|")
    for m in MODELS:
        print("| %s | %.2f | %s |" % (m, fmean(newsame[m]), " | ".join("-" if m2 == m else "%.2f" % fmean(newother[(m, m2)]) for m2 in MODELS)))
    print("\nA4. distinct ideas among the 3 lists of one model (about 90 proposals), and the Good-Turing chance that the next one is new")
    print("| task | model | proposals | distinct | seen once | chance next is new |\n|---|---|---|---|---|---|")
    for task in sorted(by):
        for m in MODELS:
            e = np.concatenate([v for (a1, r), v in sorted(by[task].items()) if a1 == m])
            n = len(e)
            S = e @ e.T >= t
            lab = list(range(n))

            def find(i):
                while lab[i] != i:
                    lab[i] = lab[lab[i]]
                    i = lab[i]
                return i

            for i, j in zip(*np.nonzero(np.triu(S, 1))):
                lab[find(int(i))] = find(int(j))
            sizes = defaultdict(int)
            for i in range(n):
                sizes[find(i)] += 1
            f1 = sum(v == 1 for v in sizes.values())
            print("| %s | %s | %d | %d | %d | %.2f |" % (task, m, n, len(sizes), f1, f1 / n))


def part_b(t):
    print("\nB1. inside the loops: share of proposals that are new to their own loop, by position in the loop (first tries of an idea only)")
    seq = defaultdict(list)  # (task, planner, loop) joined over roots
    for k in loops:
        seq[k] = (PE[k], [x[2] for x in loops[k]])
    bins = ((0, 10), (10, 20), (20, 40), (40, 80), (80, 200))
    agg = defaultdict(list)
    for (task, pl, name), (e, acc) in seq.items():
        S = e @ e.T
        new = [True] + [bool(S[i, :i].max() < t) for i in range(1, len(e))]
        for lo, hi in bins:
            if len(new) > lo:
                agg[(pl, lo)].append(fmean(new[lo:hi]))
    print("| planner | proposals 1-10 | 11-20 | 21-40 | 41-80 | 81+ |\n|---|---|---|---|---|---|")
    for pl in "GHS":
        print("| %s | %s |" % (pl, " | ".join("%.2f (%d)" % (fmean(agg[(pl, lo)]), len(agg[(pl, lo)])) if agg[(pl, lo)] else "-" for lo, _ in bins)))
    print("\nB2. share of a loop's proposals that also appear in ANOTHER loop on the same task (rows: whose; columns: other loop's planner)")
    tot = defaultdict(list)
    for (task, pl, name), (e, _) in seq.items():
        for (task2, pl2, name2), (e2, _) in seq.items():
            if task2 == task and name2 != name:
                tot[(pl, pl2)].append(cover(e, e2, t))
    print("| from planner | in a G loop | in an H loop | in an S loop |\n|---|---|---|---|")
    for pl in "GHS":
        print("| %s | %s |" % (pl, " | ".join("%.2f (%d)" % (fmean(tot[(pl, p2)]), len(tot[(pl, p2)])) if tot[(pl, p2)] else "-" for p2 in "GHS")))
    print("\nB3. accepted proposals: share that some loop with the OTHER planner on the same task also proposed")
    print("| accepted in loops of | n accepted | also proposed by G loops | by H loops | by S loops |\n|---|---|---|---|---|")
    for pl in "GHS":
        A = defaultdict(list)
        for (task, p1, name), (e, acc) in seq.items():
            if p1 == pl and any(acc):
                A[task].append(e[np.array(acc)])
        row, n = [], 0
        for p2 in "GHS":
            hit = tot_n = 0
            for task, es in A.items():
                other = [e for (t2, q, nm), (e, _) in seq.items() if t2 == task and q == p2]
                if not other or p2 == pl:
                    continue
                ea, eo = np.concatenate(es), np.concatenate(other)
                hit += int(((ea @ eo.T).max(axis=1) >= t).sum())
                tot_n += len(ea)
            row.append("%.2f (%d)" % (hit / tot_n, tot_n) if tot_n else "-")
        n = sum(len(e) for es in A.values() for e in es)
        print("| %s | %d | %s |" % (pl, n, " | ".join(row)))


for label, t in (("main", tau), ("stricter", min(0.99, tau + 0.03)), ("looser", tau - 0.03)):
    print("\n==================== threshold %s: tau = %.3f ====================" % (label, t))
    part_a(t)
    if label == "main":
        part_b(t)
