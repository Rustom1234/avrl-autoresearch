#!/usr/bin/env python3
"""Held-out check of bank ideas: one sealed job per script, the same call loop/bench/finish.py makes.

    confirm.py --out loop/runs/B1/confirmation --task clearance_hepatocyte_az --hosts willow walnut \
               base=loop/runs/B1/clear_HH_1/incumbent0.py b009=loop/runs/B1/clear_HH_1/s0/slots/4/train.py ...

A sealed job trains on the whole training file with 5 seeds and scores on the benchmark's held-out molecules (packing:
5 fresh seeds). One try per script; a result file that exists is kept. Prints the sealed loss of every script and its
difference from the first one given (the base).
"""
import argparse, importlib.util, json, sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

LOOP = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(LOOP / "core"))
_spec = importlib.util.spec_from_file_location("bench_task", LOOP / "bench" / "task.py")
B = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(B)

ap = argparse.ArgumentParser()
ap.add_argument("--out", required=True)
ap.add_argument("--task", required=True)
ap.add_argument("--hosts", nargs="+", required=True)
ap.add_argument("scripts", nargs="+", help="key=path")
a = ap.parse_args()
out = Path(a.out).resolve()
out.mkdir(parents=True, exist_ok=True)
want = [(s.split("=", 1)[0], str(Path(s.split("=", 1)[1]).resolve())) for s in a.scripts]


def run(i):
    key, f = want[i]
    rf = out / (key + ".json")
    if rf.exists():
        return
    svc = B.CpuJobs(a.hosts[i % len(a.hosts)], out / "jobs", a.task, job_timeout_s=900)
    jid, r = svc.submit(f, seed=7, sealed=True, tag="confirm-" + key, kind="confirm"), None
    while r is None:
        r = svc.wait(jid, 1800)
    rf.write_text(json.dumps(r))


with ThreadPoolExecutor(len(a.hosts)) as ex:
    list(ex.map(run, range(len(want))))
base = None
for key, f in want:
    r = json.loads((out / (key + ".json")).read_text())
    v = r.get("sealed_val_bpb")
    base = v if base is None else base
    print("%-12s status=%s sealed=%s diff_from_base=%s wall=%.0fs peak=%.2fGB" % (
        key, r["status"], "%.4f" % v if v is not None else None,
        "%+.4f" % (v - base) if v is not None and base is not None else None, r.get("wall_s") or 0, r.get("peak_rss_gb") or 0))
