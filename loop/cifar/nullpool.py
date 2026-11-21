#!/usr/bin/env python3
"""Null pool for the CIFAR task: the unchanged starting script, scored again and again.

    python loop/cifar/nullpool.py --root loop/runs/CG1_NULL --hosts willow walnut cherry spruce --per-host 40

Each job is an ordinary loop job (24 timed runs) with its own seed, plus the sealed score. No edit is ever applied,
so whatever gain an accept rule finds in these jobs is noise. One job at a time per machine; a job that does not
end with status ok is recorded and not submitted again. Create <root>/STOP to stop after the jobs now running.
"""
import argparse, importlib.util, json, sys, threading
from pathlib import Path

HERE = Path(__file__).resolve().parent
argv, sys.argv = sys.argv[1:], sys.argv[:1]
spec = importlib.util.spec_from_file_location("ctask", HERE / "task.py")
task = importlib.util.module_from_spec(spec)
spec.loader.exec_module(task)
ap = argparse.ArgumentParser()
ap.add_argument("--root", required=True)
ap.add_argument("--hosts", nargs="+", required=True)
ap.add_argument("--per-host", type=int, default=40)
a = ap.parse_args(argv)
root = Path(a.root).resolve()
root.mkdir(parents=True, exist_ok=True)
lock = threading.Lock()


def chain(hi, host):
    gpu = task.LabGPU(host, root / "jobs", HERE / "consts.json")
    for i in range(a.per_host):
        if (root / "STOP").exists():
            return
        seed = 5000 + 100 * hi + i
        jid = gpu.submit(task.BASE, seed=seed, sealed=True, tag="null-%s-%d" % (host, i), kind="null")
        r = gpu.wait(jid, 1200) or {"status": "no result in 1200 s"}
        with lock, open(root / "pool.jsonl", "a") as f:
            f.write(json.dumps({"host": host, "i": i, "seed": seed, "job": jid, "status": r.get("status"),
                                "val": r.get("official_val_bpb"), "sealed": r.get("sealed_val_bpb")}) + "\n")
        if r.get("status") != "ok":
            print(host, i, "status", r.get("status"), "- chain stopped", flush=True)
            return


ts = [threading.Thread(target=chain, args=(hi, h)) for hi, h in enumerate(a.hosts)]
[t.start() for t in ts]
[t.join() for t in ts]
print("null pool finished", flush=True)
