#!/usr/bin/env python3
"""Stop a bank run cleanly when a lab machine gets busy.

    load_guard.py --root loop/runs/B2 --hosts willow walnut pine cherry spruce --shards a/config.json ... [--max-load 24]

Once a minute reads the 1 minute load of every host (ssh; /proc/loadavg on this machine). If a host is above
--max-load on two checks in a row, writes <root>/STOP: every engine stops between slots, running jobs finish, nothing
is killed and nothing is started again. Ends when every named shard has its done file, or after it stopped the run.
One line per check goes to <root>/load.log.
"""
import argparse, socket, subprocess, time
from pathlib import Path

ap = argparse.ArgumentParser()
ap.add_argument("--root", required=True)
ap.add_argument("--hosts", nargs="+", required=True)
ap.add_argument("--shards", nargs="+", required=True)
ap.add_argument("--max-load", type=float, default=24.0)
a = ap.parse_args()
root = Path(a.root).resolve()
me = socket.gethostname().split(".")[0]


def load(h):
    try:
        if h == me:
            return float(Path("/proc/loadavg").read_text().split()[0])
        r = subprocess.run(["ssh", "-o", "ConnectTimeout=6", "-o", "BatchMode=yes", h, "cat /proc/loadavg"],
                           capture_output=True, text=True, timeout=20)
        return float(r.stdout.split()[0])
    except Exception:
        return None                     # unreachable: unknown, not a reason to stop


high = dict.fromkeys(a.hosts, 0)
while True:
    if all((Path(c).parent / "done").exists() for c in a.shards):
        print("load guard: every shard is done")
        break
    if (root / "STOP").exists():
        print("load guard: STOP file present, leaving")
        break
    L = {h: load(h) for h in a.hosts}
    with open(root / "load.log", "a") as f:
        f.write("%s %s\n" % (time.strftime("%m-%d %H:%M:%S"),
                             " ".join("%s=%s" % (h, "?" if v is None else "%.1f" % v) for h, v in L.items())))
    for h, v in L.items():
        high[h] = high[h] + 1 if v is not None and v > a.max_load else 0
    busy = [h for h in a.hosts if high[h] >= 2]
    if busy:
        (root / "STOP").write_text("load\n")
        print("load guard: %s above %.0f twice in a row (%s), wrote STOP" % (" ".join(busy), a.max_load, L))
        break
    time.sleep(60)
