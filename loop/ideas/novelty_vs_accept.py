#!/usr/bin/env python3
"""Do repeated ideas pass less often than new ones?  Existing ledgers only, no model calls.

    .venv/bin/python loop/ideas/novelty_vs_accept.py

For every tested idea of the loops in BG1_R, H1 and E1 (an E1 ledger continues and replaces the BG1_R one): its
closeness to the ideas tested earlier in the same loop (largest cosine of bge-small embeddings of the edit
descriptions), and whether it was accepted (any record of that idea with decision accept, so accepts that came on
the replicate run are counted). Reported: accept rate by closeness, split by position in the loop.
"""
import glob, json, os
from collections import defaultdict
from pathlib import Path

os.environ["HF_HUB_OFFLINE"] = "1"
os.environ.setdefault("OMP_NUM_THREADS", "8")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
import numpy as np

L = Path(__file__).resolve().parents[1]
TAU = 0.824
loops = {}
for root in ("BG1_R", "H1", "E1"):
    for f in sorted(glob.glob(str(L / "runs" / root / "*" / "ledger.jsonl"))):
        name = Path(f).parent.name
        cell = name.split("_")[-2]
        if cell == "N":
            continue
        rows, acc, ran = [], defaultdict(bool), defaultdict(bool)
        for l in open(f):
            d = json.loads(l)
            key = (d.get("idea_id"), )
            if d.get("decision") == "accept":
                acc[d.get("idea_id")] = True
            if d.get("job_id"):
                ran[d.get("idea_id")] = True
            if d.get("kind") == "candidate" and d.get("description") and not d.get("attempt"):
                rows.append((d.get("idea_id"), d["description"], d.get("slot", 0)))
        seen, seq = set(), []
        for iid, text, slot in rows:
            if iid in seen:
                continue
            seen.add(iid)
            seq.append((text, acc[iid], ran[iid], slot))
        loops[name] = (cell, seq)          # later roots replace earlier ones for the same loop name

import torch
from sentence_transformers import SentenceTransformer

torch.set_num_threads(8)
model = SentenceTransformer("BAAI/bge-small-en-v1.5", device="cpu")
texts = [t for _, seq in loops.values() for t, _, _, _ in seq]
E = model.encode(texts, batch_size=64, normalize_embeddings=True, show_progress_bar=False).astype(np.float32)
pos = 0
rows = []   # (planner, position, max sim to earlier, accepted, ran, slot)
for name, (cell, seq) in loops.items():
    e = E[pos:pos + len(seq)]
    pos += len(seq)
    S = e @ e.T
    for i, (_, a, r, slot) in enumerate(seq):
        rows.append((cell[0], i, float(S[i, :i].max()) if i else 0.0, a, r, slot))
print("loops: %d, tested ideas: %d, accepted: %d" % (len(loops), len(rows), sum(r[3] for r in rows)))
BINS = ((0, 0.70, "below 0.70 (clearly new)"), (0.70, 0.78, "0.70 to 0.78"), (0.78, TAU, "0.78 to 0.824"),
        (TAU, 2, "0.824 and above (repeat)"))


def table(sel, title):
    print("\n%s\n| closeness to earlier ideas of the loop | ideas | accepted | rate |\n|---|---|---|---|" % title)
    for lo, hi, lab in BINS:
        x = [r for r in sel if lo <= r[2] < hi]
        print("| %s | %d | %d | %s |" % (lab, len(x), sum(r[3] for r in x), "%.3f" % (sum(r[3] for r in x) / len(x)) if x else "-"))


table(rows, "All tested ideas")
table([r for r in rows if r[1] >= 10], "Ideas after the first ten of their loop")
table([r for r in rows if r[5] >= 20], "Ideas tested in slot 20 or later")
table([r for r in rows if r[5] >= 40], "Ideas tested in slot 40 or later (the E1 continuation)")
for pl in "GHS":
    table([r for r in rows if r[0] == pl and r[1] >= 10], "Planner %s, after the first ten" % pl)

# ---------- the same split with a continuous outcome: z of the first run (positive = better than the incumbent)
zrows, pos = [], 0
for name, (cell, seq) in loops.items():
    pos += len(seq)
zmap = {}
for root in ("BG1_R", "H1", "E1"):
    for f in sorted(glob.glob(str(L / "runs" / root / "*" / "ledger.jsonl"))):
        name = Path(f).parent.name
        if name.split("_")[-2] == "N":
            continue
        zs, votes = {}, {}
        for l in open(f):
            d = json.loads(l)
            if d.get("kind") == "candidate" and d.get("idea_id") not in zs and (d.get("job_id") or d.get("status")):
                zs[d["idea_id"]] = (d.get("z1"), d.get("decision"))
        st = json.loads((Path(f).parent / "state.json").read_text())
        for i in st["backlog"]:
            votes[i["id"]] = i
        zmap[name] = (zs, votes)
pos, out = 0, []
for name, (cell, seq) in loops.items():
    e = E[pos:pos + len(seq)]
    pos += len(seq)
    S = e @ e.T
    ids, seen = [], set()
    for l in open(next(p for p in [L / "runs" / r / name / "ledger.jsonl" for r in ("E1", "H1", "BG1_R")] if p.exists())):
        d = json.loads(l)
        if d.get("kind") == "candidate" and d.get("description") and not d.get("attempt") and d.get("idea_id") not in seen:
            seen.add(d.get("idea_id"))
            ids.append(d.get("idea_id"))
    zs, votes = zmap[name]
    for i, iid in enumerate(ids[:len(seq)]):
        z1, dec = zs.get(iid, (None, None))
        it = votes.get(iid)
        yes = None
        if it is not None and it["status"] == "done":
            yes = it["alpha"] - 1 - int(seq[i][1])     # the three voter samples: prior alpha = 1 + yes votes
        out.append((cell[0], i, float(S[i, :i].max()) if i else 0.0, z1, dec, seq[i][1], yes, seq[i][3]))


def ztable(sel, title):
    print("\n%s\n| closeness | ideas | crashed or invalid | mean z | z above 1 | z above 2 | z below -2 |\n|---|---|---|---|---|---|---|" % title)
    for lo, hi, lab in BINS:
        x = [r for r in sel if lo <= r[2] < hi]
        z = [r[3] for r in x if r[3] is not None]
        if not x or not z:
            continue
        print("| %s | %d | %.2f | %+.2f | %.3f | %.3f | %.2f |" % (lab, len(x), 1 - len(z) / len(x), float(np.mean(np.clip(z, -10, 10))),
                                                             np.mean([v > 1 for v in z]), np.mean([v > 2 for v in z]), np.mean([v < -2 for v in z])))


ztable([r for r in out if r[1] >= 10], "First-run z by closeness, ideas after the first ten of their loop (z clipped to plus or minus 10 for the mean)")
ztable([r for r in out if r[7] >= 40], "Same, slot 40 or later")
print("\nVoter (three Haiku yes/no samples before testing) against the outcome, ideas after the first ten")
print("| yes votes | ideas | accepted | rate | z above 1 | z above 2 |\n|---|---|---|---|---|---|")
for y in range(4):
    x = [r for r in out if r[1] >= 10 and r[6] == y]
    z = [r[3] for r in x if r[3] is not None]
    if x:
        print("| %d | %d | %d | %.3f | %s | %s |" % (y, len(x), sum(r[5] for r in x), sum(r[5] for r in x) / len(x),
                                                  "%.3f" % np.mean([v > 1 for v in z]) if z else "-", "%.3f" % np.mean([v > 2 for v in z]) if z else "-"))
