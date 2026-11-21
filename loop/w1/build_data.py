#!/usr/bin/env python3
"""W1 data: one row per candidate edit that reached a GPU in the logged loops, with its outcome.

    python loop/w1/build_data.py

For every loop of the GPT-task runs (P2, P2F, P2B) and the CIFAR gate (CG1), the ledger is replayed in order to
recover the incumbent file each candidate was written against (the incumbent changes only at an accept, and the
accepted candidate's file is stored in its slot directory). Row fields: run, task, loop, cell, slot, coder model,
idea title and spec, edit class, sha of the incumbent and of the candidate (texts in files/<sha>.py), status of the
first run, z1 (z score of the first run against the incumbent), crash, promising (z1 >= 0.75), kept (the experiment
ended in accept). Loops are read one at a time; nothing large is held in memory.
"""
import glob, hashlib, json, os
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
OUT = Path(__file__).resolve().parent
BASES = {"gpt": REPO / "loop/substrate/base/train_ampere.py", "cifar": REPO / "loop/cifar/base/train.py"}
RUNS = [("P2", "gpt"), ("P2F", "gpt"), ("P2B", "gpt"), ("CG1", "cifar")]


def put(text):
    sha = hashlib.sha256(text.encode()).hexdigest()[:16]
    f = OUT / "files" / (sha + ".py")
    if not f.exists():
        f.write_text(text)
    return sha


n = 0
with open(OUT / "data.jsonl", "w") as out:
    for run, task in RUNS:
        for d in sorted(glob.glob(str(REPO / "loop/runs" / run / "*" / "ledger.jsonl"))):
            d = Path(d).parent
            try:
                cfg = json.loads((d / "config.json").read_text())
                state = json.loads((d / "state.json").read_text())
            except FileNotFoundError:
                continue
            if cfg.get("cell") == "N" or not (d / "calls.jsonl").exists():
                continue
            base = Path(cfg.get("base_train_py") or BASES[task])
            if not base.exists():
                base = BASES[task]
            coder = {}
            for line in open(d / "calls.jsonl"):
                c = json.loads(line)
                t = c.get("tag", "").split(":")
                if len(t) == 3 and t[1] in ("coder", "repairer") and c.get("ok"):
                    coder.setdefault(int(t[2]), {})[t[1]] = c.get("served_model") or c.get("model_name")
            items = {i["id"]: i for i in state.get("backlog", [])}
            inc, groups = base.read_text(), {}
            recs = [json.loads(x) for x in open(d / "ledger.jsonl")]
            for r in recs:
                if r.get("job_id") is None or r.get("decision") == "infra" or r.get("idea_id") is None:
                    continue
                groups.setdefault(r["idea_id"], []).append(r)
            done = set()
            for r in recs:
                iid = r.get("idea_id")
                if iid not in groups or iid in done:
                    continue
                done.add(iid)
                g = groups[iid]
                first = g[0]
                f = d / "slots" / str(first["slot"]) / "train.py"
                # a crash-fix or replicate slot holds the same candidate under a later slot number
                cand = f.read_text() if f.exists() else None
                last_f = d / "slots" / str(g[-1]["slot"]) / "train.py"
                kept = g[-1]["decision"] == "accept"
                if cand is not None and cand != inc and first.get("kind") == "candidate":
                    it = items.get(first.get("idea_id"), {})
                    out.write(json.dumps(dict(
                        run=run, task=task, loop=d.name, cell=cfg.get("cell"), slot=first["slot"],
                        coder=coder.get(first["slot"], {}).get("coder"), title=it.get("title") or first.get("description", "")[:80],
                        spec=it.get("spec") or first.get("description", ""), family=it.get("family"),
                        edit_class=first.get("edit_class"), inc=put(inc), cand=put(cand), status=first.get("status"),
                        z1=first.get("z1"), crash=first.get("status") in ("crash", "timeout"),
                        promising=(first.get("z1") or -9) >= 0.75, kept=kept, harm=cfg.get("harm_rate", 0) > 0)) + "\n")
                    n += 1
                if kept and last_f.exists():
                    inc = last_f.read_text()
print(n, "rows,", len(os.listdir(OUT / "files")), "files")
