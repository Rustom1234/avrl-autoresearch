#!/usr/bin/env python3
"""Start every shard of a bank (or router) run at once and watch it. No retries of any kind.

    python loop/ideas/launch_bank.py --root loop/runs/B1 [--per-host 3] [--entry loop/ideas/bank.py]

Reads <root>/shards.json (config paths). Each shard is one engine process on this machine; its jobs go to a lab
machine by ssh (the harness limits each job to 8 threads and stops it quietly above its own memory limit).
Watches, once a minute:
  login     a model call that failed for a login or credential reason: writes <root>/STOP, every engine stops.
  memory    resident memory of a shard's process tree above --rss-gb: that shard is ended with SIGTERM (quiet, in
            user space) and is not started again.
Each process also runs under prlimit --as, a quiet limit on address space.
"""
import argparse, json, os, re, signal, subprocess, sys, time
from pathlib import Path

import psutil

ap = argparse.ArgumentParser()
ap.add_argument("--root", required=True)
ap.add_argument("--per-host", type=int, default=3)
ap.add_argument("--hosts", nargs="+", default=["willow", "walnut", "pine", "cherry"])
ap.add_argument("--entry", default="loop/ideas/bank.py")
ap.add_argument("--rss-gb", type=float, default=3.0)
ap.add_argument("--shards-file", default="shards.json")
a = ap.parse_args()
root = Path(a.root).resolve()
REPO = Path(__file__).resolve().parents[2]
AUTH = re.compile(r"login|authenticat|credential|unauthorized|\b401\b|invalid api key|oauth", re.I)


def log(msg):
    with open(root / "launch.log", "a") as f:
        f.write("%s %s\n" % (time.strftime("%m-%d %H:%M:%S"), msg))


cfgs = [Path(p) for p in json.loads((root / a.shards_file).read_text())]
todo = [c for c in cfgs if not (c.parent / "done").exists()]
assert len(todo) <= a.per_host * len(a.hosts), "more shards than places: %d" % len(todo)
procs = {}
for n, c in enumerate(todo):
    cfg = json.loads(c.read_text())
    cfg.setdefault("host", a.hosts[n % len(a.hosts)])     # a shard that already ran keeps its machine
    c.write_text(json.dumps(cfg, indent=1))
    resume = (c.parent / "state.json").exists()
    cmd = ["prlimit", "--as=%d" % (32 << 30), str(REPO / ".venv/bin/python"), str(REPO / a.entry), "run", "--config",
           str(c)]
    f = open(c.parent / "loop.log", "a")
    p = subprocess.Popen(cmd, stdout=f, stderr=subprocess.STDOUT, cwd=str(REPO), start_new_session=True,
                         env=dict(os.environ, CUDA_VISIBLE_DEVICES="", AR_CLAUDE_MAX_CONCURRENT="8"))
    procs[cfg["loop_id"]] = (p, c.parent, 0)
    log("%s start on %s pid %d%s" % (cfg["loop_id"], cfg["host"], p.pid, " (resumed)" if resume else ""))

seen = {}
while procs:
    time.sleep(60)
    for lid, (p, d, _) in list(procs.items()):
        rc = p.poll()
        if rc is not None:
            log("%s rc=%d" % (lid, rc))
            del procs[lid]
            continue
        try:
            tree = [psutil.Process(p.pid)] + psutil.Process(p.pid).children(recursive=True)
            rss = sum(q.memory_info().rss for q in tree if q.is_running()) / 2 ** 30
        except psutil.Error:
            rss = 0.0
        if rss > a.rss_gb:
            log("%s MEMORY: process tree at %.1f GB, ending this shard (no restart)" % (lid, rss))
            os.killpg(p.pid, signal.SIGTERM)
        calls = d / "calls.jsonl"
        if calls.exists():
            size = calls.stat().st_size
            with open(calls) as f:
                f.seek(seen.get(lid, 0))
                for line in f:
                    try:
                        r = json.loads(line)
                    except ValueError:
                        continue
                    if not r.get("ok") and AUTH.search(str(r.get("error") or "")):
                        log("%s LOGIN error, stopping the run: %s" % (lid, str(r.get("error"))[:200]))
                        (root / "STOP").write_text("login\n")
            seen[lid] = size
log("launcher finished")
