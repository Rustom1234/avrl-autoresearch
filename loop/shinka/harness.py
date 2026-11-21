#!/usr/bin/env python3
"""Harness of the ShinkaEvolve circle-packing task (26 circles), for the unchanged loop engine.

    python harness.py CAND --task pack26s --out DIR [--timeout 300]

One evaluation, as in ShinkaEvolve's examples/circle_packing/evaluate.py: the candidate's run_packing() is called
once and must return (centers, radii, reported_sum). Validity is their check with zero tolerance (function
adapted_validate_packing below is their logic, atol = 0). loss = minus the reported sum; an invalid packing is a
crash. The packing is saved in DIR/packing.npz, and a second check in exact rational arithmetic is recorded
(field exact_valid) but does not change the score.

Shared-machine safety as in loop/bench/harness.py: 16 GB address-space limit (a runaway allocation fails inside
Python), a watchdog that ends the job quietly at --mem-gb (default 6) or after --timeout seconds, one thread.
"""
import argparse, importlib.util, json, os, resource, socket, sys, threading, time, traceback

ap = argparse.ArgumentParser()
ap.add_argument("cand")
ap.add_argument("--task", default="pack26s")
ap.add_argument("--out", required=True)
ap.add_argument("--seed", type=int, default=42)
ap.add_argument("--sealed", action="store_true")
ap.add_argument("--timeout", type=float, default=300)
ap.add_argument("--mem-gb", type=float, default=6.0)
ap.add_argument("--job-id", default=None)
args = ap.parse_args()
N = 26
for v in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ[v] = "1"
os.environ["CUDA_VISIBLE_DEVICES"] = ""
resource.setrlimit(resource.RLIMIT_AS, (16 << 30, 16 << 30))
os.makedirs(args.out, exist_ok=True)
RESULT = os.path.join(args.out, "result.json")
T0 = time.time()
res = {"job_id": args.job_id, "gpu": socket.gethostname().split(".")[0], "task": args.task, "seed": args.seed,
       "sealed": args.sealed, "status": None, "official_val_bpb": None, "sealed_val_bpb": None,
       "integrity_failed": [], "log_tail": []}
_done = threading.Event()


def finish(status, **kw):
    if _done.is_set():
        return
    _done.set()
    res.update(status=status, wall_s=round(time.time() - T0, 1), **kw)
    with open(RESULT + ".tmp", "w") as f:
        json.dump(res, f, indent=1)
    os.replace(RESULT + ".tmp", RESULT)
    sys.stdout.flush()
    os._exit(0)


def self_gb():
    for l in open("/proc/self/smaps_rollup"):
        if l.startswith("Pss:"):
            return int(l.split()[1]) / 1048576.0
    return 0.0


def watchdog():
    peak = 0.0
    while not _done.is_set():
        try:
            g = self_gb()
            peak = max(peak, g)
            res["peak_rss_gb"] = round(peak, 2)
            with open(os.path.join(args.out, "heartbeat"), "w") as f:
                f.write("%d %.2f\n" % (time.time(), g))
            if g > args.mem_gb:
                finish("crash", reason="memory_limit", log_tail=["the job used %.1f GB of memory; the limit is %.1f GB" % (g, args.mem_gb)])
            if time.time() - T0 > args.timeout:
                finish("timeout", log_tail=["the job did not finish within %d s" % args.timeout])
        except Exception:
            pass
        time.sleep(1.0)


threading.Thread(target=watchdog, daemon=True).start()
import numpy as np

logf = open(os.path.join(args.out, "cand.log"), "w")
real_out = os.dup(1)


def tail(n=40):
    logf.flush()
    return open(os.path.join(args.out, "cand.log")).read().splitlines()[-n:]


def validate(centers, radii, reported_sum, atol=0.0):
    """ShinkaEvolve's adapted_validate_packing, same order of checks; returns an error message or None."""
    if centers.shape != (N, 2):
        return "Centers shape incorrect. Expected (%d, 2), got %s" % (N, centers.shape)
    if radii.shape != (N,):
        return "Radii shape incorrect. Expected (%d,), got %s" % (N, radii.shape)
    if not np.all(np.isfinite(centers)) or not np.all(np.isfinite(radii)) or not np.isfinite(reported_sum):
        return "Non-finite values found in centers, radii, or reported_sum."
    if np.any(radii < 0):
        return "Negative radii found"
    if not np.isclose(np.sum(radii), reported_sum, atol=atol):
        return "Sum of radii (%.6f) does not match reported (%.6f)" % (np.sum(radii), reported_sum)
    for i in range(N):
        x, y = centers[i]
        r = radii[i]
        if x - r < -atol or x + r > 1 + atol or y - r < -atol or y + r > 1 + atol:
            return "Circle %d (x=%.4f, y=%.4f, r=%.4f) is outside unit square." % (i, x, y, r)
    for i in range(N):
        for j in range(i + 1, N):
            dist = np.sqrt(np.sum((centers[i] - centers[j]) ** 2))
            if dist < radii[i] + radii[j] - atol:
                return "Circles %d & %d overlap. Dist: %.6g, Sum Radii: %.6g" % (i, j, dist, radii[i] + radii[j])
    return None


def exact_valid(c, r):
    from fractions import Fraction as F
    C = [(F(float(x)), F(float(y))) for x, y in c]
    R = [F(float(x)) for x in r]
    if any(min(x - q, y - q, 1 - x - q, 1 - y - q) < 0 for (x, y), q in zip(C, R)):
        return False
    return all((C[i][0] - C[j][0]) ** 2 + (C[i][1] - C[j][1]) ** 2 >= (R[i] + R[j]) ** 2 for i in range(N) for j in range(i))


try:
    os.dup2(logf.fileno(), 1)
    os.dup2(logf.fileno(), 2)
    spec = importlib.util.spec_from_file_location("train", args.cand)
    cand = importlib.util.module_from_spec(spec)
    sys.modules["train"] = cand
    spec.loader.exec_module(cand)
    out = cand.run_packing()
    centers, radii, reported = np.asarray(out[0], dtype=np.float64), np.asarray(out[1], dtype=np.float64), float(out[2])
    err = validate(centers, radii, reported)
    if err:
        raise RuntimeError("invalid packing: " + err)
except BaseException:
    traceback.print_exc()
    os.dup2(real_out, 1)
    finish("crash", log_tail=tail())
os.dup2(real_out, 1)
np.savez(os.path.join(args.out, "packing.npz"), centers=centers, radii=radii)
loss = -float(np.sum(radii))
res.update(sum=float(np.sum(radii)), exact_valid=bool(exact_valid(centers, radii)), run_s=round(time.time() - T0, 1))
res["official_val_bpb"] = loss
res["sealed_val_bpb"] = loss if args.sealed else None
finish("ok", log_tail=tail(10) + ["loss: %.9f" % loss])
