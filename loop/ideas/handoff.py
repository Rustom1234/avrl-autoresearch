#!/usr/bin/env python3
"""Hand the tail of a slow shard to helper shards on idle machines.

    handoff.py split --root loop/runs/B1 --shards <config.json> ... --helpers 2 2 3 4 [--keep 5] [--out shards_x.json]
    handoff.py watch --root loop/runs/B1

split  each named shard keeps its next --keep untested ideas (the first of them is the one in progress); the rest go
       round robin to new shard folders (next free number, same stage prefix, same config otherwise). Writes
       <root>/<out> (helper configs, for launch_bank.py) and <root>/handoff.json (per original: config, ideas it keeps).
watch  ends an original shard (SIGTERM to its own process group: quiet, in user space, no restart) once every idea it
       keeps is decided and no follow-up run is pending, so that it does not walk into ideas the helpers now own.
       The condition must hold on two polls in a row (2 s apart) before the signal is sent.
"""
import argparse, json, os, re, signal, time
from pathlib import Path

import psutil


def log(root, msg):
    with open(root / "launch.log", "a") as f:
        f.write("%s %s\n" % (time.strftime("%m-%d %H:%M:%S"), msg))


def cmd_split(a):
    root = Path(a.root).resolve()
    assert len(a.helpers) == len(a.shards)
    nxt, out, hand = {}, [], []
    for p, h in zip(a.shards, a.helpers):
        p = Path(p).resolve()
        cfg = json.loads(p.read_text())
        st = json.loads((p.parent / "state.json").read_text())
        untried = [i["id"] for i in st["backlog"] if i["times_tried"] == 0 and i["status"] == "open"]
        own, rest = untried[:a.keep], untried[a.keep:]
        hand.append({"config": str(p), "loop_id": cfg["loop_id"], "own": own})
        folder, loop_dir = p.parent.name, p.parent.parent
        prefix = re.match(r"([a-z]+)\d+$", folder).group(1)
        key = (str(loop_dir), prefix)
        if key not in nxt:
            nxt[key] = 1 + max(int(re.match(r"[a-z]+(\d+)$", d.name).group(1)) for d in loop_dir.iterdir()
                               if re.match(r"%s\d+$" % prefix, d.name))
        for j in range(h):
            items = rest[j::h]
            if not items:
                continue
            name = "%s%d" % (prefix, nxt[key])
            nxt[key] += 1
            c = dict(cfg, items=items, loop_id=cfg["loop_id"][:-len(folder)] + name, run_dir=str(loop_dir / name),
                     n_slots=3 * len(items) + 3, rng_seed=cfg["rng_seed"] + 100 * nxt[key])
            c.pop("host", None)
            (loop_dir / name).mkdir()
            (loop_dir / name / "config.json").write_text(json.dumps(c, indent=1))
            out.append(str(loop_dir / name / "config.json"))
            print("%s: %d ideas from %s" % (c["loop_id"], len(items), cfg["loop_id"]))
        print("%s keeps %d: %s" % (cfg["loop_id"], len(own), " ".join(own)))
    (root / a.out).write_text(json.dumps(out, indent=1))
    (root / "handoff.json").write_text(json.dumps(hand, indent=1))


def find_proc(config):
    for q in psutil.process_iter(["cmdline"]):
        cl = q.info["cmdline"] or []
        if config in cl and any(x.endswith("bank.py") for x in cl) and "python" in os.path.basename(cl[0]):
            return q
    return None


def cmd_watch(a):
    root = Path(a.root).resolve()
    todo = {h["loop_id"]: h for h in json.loads((root / "handoff.json").read_text())}
    hits = dict.fromkeys(todo, 0)
    while todo:
        time.sleep(2)
        for lid, h in list(todo.items()):
            q = find_proc(h["config"])
            if q is None:
                log(root, "%s handoff: process already gone" % lid)
                del todo[lid]
                continue
            try:
                s = json.loads((Path(h["config"]).parent / "state.json").read_text())
            except ValueError:
                continue
            tried = {i["id"]: i["times_tried"] for i in s["backlog"]}
            ok = s["cand"] is None and s["in_flight_job"] is None and all(tried[k] > 0 for k in h["own"])
            hits[lid] = hits[lid] + 1 if ok else 0
            if hits[lid] >= 2:
                if os.getpgid(q.pid) == q.pid:
                    os.killpg(q.pid, signal.SIGTERM)
                else:
                    q.terminate()
                log(root, "%s handoff: its %d ideas are decided, process %d ended (helpers own the rest)" % (
                    lid, len(h["own"]), q.pid))
                del todo[lid]
    log(root, "handoff watcher finished")


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    sub = ap.add_subparsers(dest="cmd", required=True)
    p = sub.add_parser("split")
    p.add_argument("--root", required=True)
    p.add_argument("--shards", nargs="+", required=True)
    p.add_argument("--helpers", nargs="+", type=int, required=True)
    p.add_argument("--keep", type=int, default=5)
    p.add_argument("--out", default="shards_handoff.json")
    p = sub.add_parser("watch")
    p.add_argument("--root", required=True)
    a = ap.parse_args()
    {"split": cmd_split, "watch": cmd_watch}[a.cmd](a)
