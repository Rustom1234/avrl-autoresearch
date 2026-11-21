#!/usr/bin/env python3
"""Run the loops of a plan, one loop at a time per lab machine.

    python loop/cifar/launch.py --plan loop/runs/<RUN>/plan.json

plan.json: {"hosts": [...], "loops": [{"loop_id", "cmd": "loop" or "nloop", "config": {...engine config without host}}]}
in the order they should start. Each host has a worker thread that takes the next loop that has not started, writes
its config (adding the host) and runs task.py for it; a loop that was interrupted (state.json exists, no "done" file)
is resumed on the host it started on. A loop whose process exits with an error is tried again after a minute, at
most 5 times, never after a memory stop (those end as a normal crash result inside the loop). The loop processes
run on this machine; only the GPU jobs run on the hosts. Create <run>/STOP to stop after the loops now running.
"""
import argparse, json, os, subprocess, sys, threading, time
from pathlib import Path

ap = argparse.ArgumentParser()
ap.add_argument("--plan", required=True)
a = ap.parse_args()
plan_path = Path(a.plan).resolve()
plan, root = json.loads(plan_path.read_text()), plan_path.parent
REPO = Path(__file__).resolve().parents[2]
lock = threading.Lock()


def log(msg):
    with lock, open(root / "launch.log", "a") as f:
        f.write("%s %s\n" % (time.strftime("%m-%d %H:%M:%S"), msg))


def take(host):
    """Next loop for this host: one it started and did not finish, else the first that nobody has started."""
    with lock:
        fresh = None
        for lp in plan["loops"]:
            d = root / lp["loop_id"]
            if (d / "done").exists() or lp.get("taken"):
                continue
            cfg_file = d / "config.json"
            if cfg_file.exists():
                if json.loads(cfg_file.read_text())["host"] == host:
                    lp["taken"] = True
                    return lp
            elif fresh is None:
                fresh = lp
        if fresh is not None:
            fresh["taken"] = True
            d = root / fresh["loop_id"]
            d.mkdir(parents=True, exist_ok=True)
            (d / "config.json").write_text(json.dumps(dict(fresh["config"], loop_id=fresh["loop_id"], host=host,
                                                           run_dir=str(d)), indent=1))
        return fresh


def worker(host):
    while not (root / "STOP").exists():
        lp = take(host)
        if lp is None:
            return
        d = root / lp["loop_id"]
        for attempt in range(5):
            cmd = [str(REPO / ".venv/bin/python"), str(REPO / "loop/cifar/task.py"), lp["cmd"], "--config",
                   str(d / "config.json")] + (["--resume"] if (d / "state.json").exists() else [])
            log("%s start on %s (attempt %d)" % (lp["loop_id"], host, attempt + 1))
            with open(d / "loop.log", "a") as f:
                # no GPU for the loop process itself (the engine's idea-dedup embedder would otherwise sit on the
                # stopwatch GPU of this machine)
                rc = subprocess.call(cmd, stdout=f, stderr=subprocess.STDOUT, cwd=str(REPO),
                                     env=dict(os.environ, CUDA_VISIBLE_DEVICES=""))
            log("%s rc=%d" % (lp["loop_id"], rc))
            if rc == 0:
                (d / "done").write_text(time.strftime("%Y-%m-%d %H:%M:%S\n"))
                break
            time.sleep(60)


threads = [threading.Thread(target=worker, args=(h,)) for h in plan["hosts"]]
for t in threads:
    t.start()
for t in threads:
    t.join()
log("launcher finished")
