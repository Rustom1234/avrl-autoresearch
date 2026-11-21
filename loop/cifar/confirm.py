#!/usr/bin/env python3
"""Confirmation block of a run: score every loop's final train.py and the baseline on the sealed half of the test set.

    python loop/cifar/confirm.py --root loop/runs/CG1 [--jobs-per-loop 2] [--base-per-host 2]

Each loop whose final incumbent differs from the baseline gets --jobs-per-loop fresh jobs on different lab machines;
the baseline gets --base-per-host jobs on every machine, in the same session. Per job the sealed cost has the host's
frozen time offset subtracted. gain of a loop = mean baseline sealed cost minus mean sealed cost of its final script
(positive is better); a loop that accepted nothing has gain 0 and runs no job. Job ids are kept in
confirmation/state.json, so the command can be run again after an interruption and only waits for or resubmits
what is missing. Writes confirmation/summary.json.
"""
import argparse, hashlib, json, sys, time
from pathlib import Path
from statistics import fmean

HERE = Path(__file__).resolve().parent
sys.argv, argv = sys.argv[:1], sys.argv[1:]
import task  # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("--root", required=True)
ap.add_argument("--jobs-per-loop", type=int, default=2)
ap.add_argument("--base-per-host", type=int, default=2)
a = ap.parse_args(argv)
root = Path(a.root).resolve()
plan = json.loads((root / "plan.json").read_text())
consts_path = HERE / "consts.json"
consts = json.loads(consts_path.read_text())
out = root / "confirmation"
out.mkdir(exist_ok=True)
state_file = out / "state.json"
state = json.loads(state_file.read_text()) if state_file.exists() else {}
hosts = plan["hosts"]
gpus = {h: task.LabGPU(h, out / "jobs", consts_path) for h in hosts}
sha = lambda p: hashlib.sha256(Path(p).read_bytes()).hexdigest()
base_sha = sha(task.BASE)

want, i = [], 0    # (key, file, host, seed)
for h in hosts:
    for j in range(a.base_per_host):
        want.append(("BASE:%s:%d" % (h, j), str(task.BASE), h, 9000 + j))
for lp in plan["loops"]:
    inc = root / lp["loop_id"] / "incumbent.py"
    if not (root / lp["loop_id"] / "done").exists():
        print("not finished, skipped:", lp["loop_id"])
        continue
    if sha(inc) == base_sha:
        continue
    for j in range(a.jobs_per_loop):
        want.append(("%s:%d" % (lp["loop_id"], j), str(inc), hosts[i % len(hosts)], 9100 + j))
        i += 1

results = {}
for rnd in range(4):   # a job lost to a machine going down is submitted again, at most 3 times
    for key, f, h, seed in want:
        if key not in results and key not in state:
            state[key] = {"job": gpus[h].submit(f, seed=seed, sealed=True, tag="confirm-" + key, kind="confirm"), "host": h}
            state_file.write_text(json.dumps(state, indent=1))
    for key, f, h, seed in want:
        if key in results:
            continue
        r = None
        while r is None:
            r = gpus[h].wait(state[key]["job"], 3600)
        if r["status"] == "infra_fail":
            print("infra_fail, will resubmit:", key, r.get("error"))
            del state[key]
            time.sleep(30)
        else:
            results[key] = r
    state_file.write_text(json.dumps(state, indent=1))
    if len(results) == len(want):
        break

adj = lambda r: None if r.get("sealed_val_bpb") is None else r["sealed_val_bpb"] - consts["gpu_offsets"][r["gpu"]]
base = [adj(r) for k, r in results.items() if k.startswith("BASE:") and adj(r) is not None]
b = fmean(base)
loops = {}
for lp in plan["loops"]:
    lid = lp["loop_id"]
    if not (root / lid / "done").exists():
        continue
    rs = [r for k, r in results.items() if k.rsplit(":", 1)[0] == lid]
    vals = [adj(r) for r in rs if adj(r) is not None]
    changed = sha(root / lid / "incumbent.py") != base_sha
    loops[lid] = {"cell": lp["config"]["cell"], "block": lp["config"]["block"], "changed": changed,
                  "gain": (b - fmean(vals)) if vals else (0.0 if not changed else None), "sealed_costs": vals,
                  "statuses": [r["status"] for r in rs],
                  "time": [r.get("time_median") for r in rs], "acc_sealed": [r.get("acc_sealed") for r in rs]}
(out / "summary.json").write_text(json.dumps({"baseline_sealed_mean": b, "baseline_sealed": base, "loops": loops}, indent=1))
for cell in sorted({v["cell"] for v in loops.values()}):
    g = [v["gain"] for v in loops.values() if v["cell"] == cell and v["gain"] is not None]
    print("%s: n=%d mean sealed gain %.4f  %s" % (cell, len(g), fmean(g), " ".join("%.3f" % x for x in g)))
