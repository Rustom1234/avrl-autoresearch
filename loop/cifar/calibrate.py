#!/usr/bin/env python3
"""Turn the calibration jobs (calib/jobs/<host>_<name>_<i>/result.json) into consts.json, the frozen numbers of the task.

    python loop/cifar/calibrate.py [--write]

t_host        mean over a host's baseline jobs of the median run time; t_ref is their geometric mean over hosts
gpu_offsets   ln(t_host / t_ref): what the engine subtracts from a cost measured on that host
a_ref_val, a_ref_sealed   mean baseline accuracy (percent) on the validation and the sealed half of the test set
k             slope of accuracy (percent, all 10,000 test images) against ln(time) over the jobs at 0.9, 1.0 and
              1.1 times the baseline epochs, time taken relative to the host's baseline
sigma_used    1.15 times the sd of the baseline jobs' own costs (the P2B rule for the noise of one job)
bench_host    median time of the fixed benchmark workload per host (the stopwatch check of harness.py)
"""
import glob, json, math, os, sys
from statistics import fmean, median, stdev

HERE = os.path.dirname(os.path.abspath(__file__))
jobs = {}
for f in sorted(glob.glob(os.path.join(HERE, "calib/jobs/*/result.json"))):
    r = json.load(open(f))
    host, name = os.path.basename(os.path.dirname(f)).split("_")[:2]
    if r["status"] != "ok":
        print("not ok:", f, r["status"], r.get("error") or r.get("reason"))
        continue
    jobs.setdefault(name, []).append(dict(r, host=host))
base = jobs["base"]
hosts = sorted({j["host"] for j in base})
t_host = {h: fmean(j["time_median"] for j in base if j["host"] == h) for h in hosts}
t_ref = math.exp(fmean(math.log(t) for t in t_host.values()))
off = {h: math.log(t_host[h] / t_ref) for h in hosts}
a_val, a_sealed = fmean(j["acc_val"] for j in base), fmean(j["acc_sealed"] for j in base)
pts = [(math.log(j["time_median"] / t_host[j["host"]]), (j["acc_val"] + j["acc_sealed"]) / 2)
       for name in ("e090", "base", "e110") for j in jobs.get(name, [])]
mx, my = fmean(p[0] for p in pts), fmean(p[1] for p in pts)
k = sum((x - mx) * (y - my) for x, y in pts) / sum((x - mx) ** 2 for x, y in pts)
cost = lambda j: math.log(j["time_median"] / t_ref) - off[j["host"]] + (a_val - j["acc_val"]) / k
costs = [cost(j) for j in base]
sd = stdev(costs)
bench = {h: median(b for name in jobs for j in jobs[name] if j["host"] == h for b in j["bench"]) for h in hosts}
spread = max(abs(b / bench[j["host"]] - 1) for name in jobs for j in jobs[name] for b in j["bench"])
consts = dict(t_ref=t_ref, t_host=t_host, gpu_offsets=off, a_ref_val=a_val, a_ref_sealed=a_sealed, k=k,
              sigma_used=1.15 * sd, sd_cost=sd, n_anchor=len(base), bench_host=bench, bench_max_dev=spread,
              anchor_costs=costs)
for name in ("e090", "base", "e110"):
    js = jobs.get(name, [])
    print("%s: %d jobs, acc10k %.3f, rel time %.4f, run acc sd %.3f, run time sd %.3f s" % (
        name, len(js), fmean((j["acc_val"] + j["acc_sealed"]) / 2 for j in js),
        fmean(j["time_median"] / t_host[j["host"]] for j in js), fmean(j["acc_val_sd"] for j in js),
        fmean(j["time_sd"] for j in js)))
print(json.dumps({a: (round(b, 5) if isinstance(b, float) else b) for a, b in consts.items() if a != "anchor_costs"}, indent=1))
print("peak memory GB:", max(j.get("peak_rss_gb", 0) for name in jobs for j in jobs[name]),
      " warmup s:", round(fmean(j["warmup_s"] for name in jobs for j in jobs[name])),
      " job wall s:", round(fmean(j["wall_s"] for name in jobs for j in jobs[name])))
if "--write" in sys.argv:
    json.dump(consts, open(os.path.join(HERE, "consts.json"), "w"), indent=1)
    print("wrote consts.json")
