#!/usr/bin/env python3
"""Baseline calibration of the bench tasks: 10 jobs of the starting script per task (seeds 1 to 10), two at a time
per lab machine, then consts.json: per task the anchor (mean baseline loss), sd, sigma_used = 1.15 sd (the P2B rule),
the job time and the size of the training file.

    python loop/bench/calibrate.py            run missing jobs and write consts.json
"""
import json, sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from statistics import fmean, stdev

HERE = Path(__file__).resolve().parent
sys.argv = sys.argv[:1]
import importlib.util
spec = importlib.util.spec_from_file_location("btask", HERE / "task.py")
B = importlib.util.module_from_spec(spec)
spec.loader.exec_module(B)
HOSTS = ["willow", "walnut", "pine", "cherry", "spruce"]
out = HERE / "calib"
todo = [(t, s) for s in range(1, 11) for t in B.TASKS]


def run(i):
    t, s = todo[i]
    f = out / ("%s_%d.json" % (t, s))
    if f.exists():
        return
    svc = B.CpuJobs(HOSTS[i % len(HOSTS)], out / "jobs", t)
    r = None
    jid = svc.submit(str(B.base_path(t)), seed=s, sealed=False, tag="calib-%s-%d" % (t, s), kind="calib")
    while r is None:
        r = svc.wait(jid, 900)
    f.write_text(json.dumps(r))


out.mkdir(exist_ok=True)
with ThreadPoolExecutor(10) as ex:
    list(ex.map(run, range(len(todo))))
consts = {}
for t in B.TASKS:
    rs = [json.loads((out / ("%s_%d.json" % (t, s))).read_text()) for s in range(1, 11)]
    bad = [r["status"] for r in rs if r["status"] != "ok"]
    assert not bad, (t, bad)
    v = [r["official_val_bpb"] for r in rs]
    n_train = sum(1 for _ in open(HERE / "data/admet_group" / t / "train_val.csv")) - 1 if B.TASKS[t]["family"] == "admet" else 0
    consts[t] = dict(anchor=fmean(v), sd=stdev(v), sigma_used=1.15 * stdev(v), n_anchor=len(v),
                     job_s=fmean(r["wall_s"] for r in rs), n_train=n_train, values=v,
                     peak_gb=max(r.get("peak_rss_gb", 0) for r in rs))
    print("%-28s anchor %.5f sd %.5f job %.0f s peak %.2f GB" % (t, consts[t]["anchor"], consts[t]["sd"], consts[t]["job_s"], consts[t]["peak_gb"]))
(HERE / "consts.json").write_text(json.dumps(consts, indent=1))
