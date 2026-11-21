#!/usr/bin/env python3
"""Locked harness of the CIFAR speedrun task: runs one candidate train.py on this machine's GPU and scores it.

    python harness.py CAND --out DIR [--seed 42] [--runs 24] [--sealed] [--consts consts.json] [--timeout 900]

One job = one process = one "slot" of a loop. The candidate defines make_model(), train(model, images, labels) and
predict(model, images) (see base/train.py). The harness
  1. waits for this host's GPU lock, so two of my jobs never share a GPU (the GPU is the stopwatch),
  2. calls train and predict once on random data (compilation is not timed),
  3. for each of --runs runs: seeds torch, times train + predict on the real data with a synchronized wall clock,
     and computes accuracy itself from the returned logits. The candidate never sees test images during training
     and never sees test labels at all.
  4. writes DIR/result.json.

Score (lower is better; the engine stores it in its field official_val_bpb):
    cost = ln(median run time / T_REF) + (A_REF - mean accuracy in percent) / K
T_REF and A_REF are the baseline's time and accuracy, K is the accuracy (percent) the baseline gains per unit of
ln(time) when its epochs are scaled. So the baseline costs 0, a change that is 5 percent faster at equal accuracy
costs about -0.05, and trading epochs for accuracy along the baseline's own curve changes nothing. Without --consts
the cost is not computed (calibration mode).

The 10,000 test images are split once into a validation half (official score, used inside loops) and a sealed half
(reported only with --sealed; used for audits and for the final confirmation).

Shared-machine safety: a watchdog thread ends the job quietly (status crash, reason memory_limit) when the resident
memory of this process tree passes --mem-gb, long before the kernel would act, and ends it with status timeout after
--timeout seconds. A job is reported as infra_fail (retried, not charged) when another process uses the GPU or the
fixed benchmark workload runs more than 10 percent off this host's calibrated speed.
"""
import argparse, fcntl, importlib.util, json, math, os, socket, subprocess, sys, threading, time, traceback

ap = argparse.ArgumentParser()
ap.add_argument("cand")
ap.add_argument("--out", required=True)
ap.add_argument("--seed", type=int, default=42)
ap.add_argument("--runs", type=int, default=24)
ap.add_argument("--sealed", action="store_true")
ap.add_argument("--consts", default=None)
ap.add_argument("--timeout", type=float, default=900)
ap.add_argument("--mem-gb", type=float, default=9.0)
ap.add_argument("--job-id", default=None)
ap.add_argument("--data", default="/homes/rdubash/speedrun/cifar10")
args = ap.parse_args()

HOST = socket.gethostname().split(".")[0]
os.environ.setdefault("TORCHINDUCTOR_COMPILE_THREADS", "1")   # no pool of compile workers on a shared machine
os.makedirs(args.out, exist_ok=True)
RESULT = os.path.join(args.out, "result.json")
T_START = time.time()
BENCH_TOL = 0.10
SPLIT_SEED = 20261113
res = {"job_id": args.job_id, "gpu": HOST, "seed": args.seed, "sealed": args.sealed, "status": None,
       "official_val_bpb": None, "sealed_val_bpb": None, "integrity_failed": [], "log_tail": []}
_done = threading.Event()


def finish(status, **kw):
    """Write result.json once (atomically) and leave."""
    if _done.is_set():
        return
    _done.set()
    res.update(status=status, wall_s=round(time.time() - T_START, 1), **kw)
    tmp = RESULT + ".tmp"
    with open(tmp, "w") as f:
        json.dump(res, f, indent=1)
    os.replace(tmp, RESULT)
    sys.stdout.flush()
    os._exit(0)


def tree_rss_gb():
    """Memory of this process and its descendants, in GB. Uses Pss (shared pages counted once across the tree),
    because forked compile workers share most of their pages with the parent."""
    out = subprocess.run(["ps", "-u", str(os.getuid()), "-o", "pid=,ppid="], capture_output=True, text=True).stdout
    kids = {}
    for l in out.splitlines():
        pid, ppid = (int(x) for x in l.split())
        kids.setdefault(ppid, []).append(pid)
    todo, total = [os.getpid()], 0
    while todo:
        p = todo.pop()
        todo += kids.get(p, [])
        try:
            for l in open("/proc/%d/smaps_rollup" % p):
                if l.startswith("Pss:"):
                    total += int(l.split()[1])
        except OSError:
            pass
    return total / 1048576.0


def watchdog():
    peak = 0.0
    while not _done.is_set():
        try:
            g = tree_rss_gb()
            peak = max(peak, g)
            res["peak_rss_gb"] = round(peak, 2)
            with open(os.path.join(args.out, "heartbeat"), "w") as f:
                f.write("%d %.2f\n" % (time.time(), g))
            if g > args.mem_gb:
                finish("crash", reason="memory_limit", log_tail=["the job used %.1f GB of host memory; the limit is "
                                                                 "%.1f GB" % (g, args.mem_gb)])
            if res.get("t_lock") and time.time() - res["t_lock"] > args.timeout:
                finish("timeout", log_tail=["the job did not finish within %d s" % args.timeout])
        except Exception:
            pass
        time.sleep(1.0)


threading.Thread(target=watchdog, daemon=True).start()

# One job of mine per GPU at a time: the lock lives on the host's local disk, so it is per host.
lock = open("/tmp/cifar_gpu_%d.lock" % os.getuid(), "w")
fcntl.flock(lock, fcntl.LOCK_EX)
res["t_lock"] = time.time()


def gpu_others():
    """Compute processes on the GPU that are not part of this job."""
    out = subprocess.run(["nvidia-smi", "--query-compute-apps=pid,used_memory", "--format=csv,noheader,nounits"],
                         capture_output=True, text=True).stdout
    return [l.strip() for l in out.splitlines() if l.strip() and int(l.split(",")[0]) != os.getpid()]


others = gpu_others()
for _ in range(30):   # a job of mine that has just released the lock may still be leaving the GPU: give it 30 s
    if not others:
        break
    time.sleep(1)
    others = gpu_others()
if others:
    finish("infra_fail", error="GPU busy before start: %s" % others)

import torch
import torch.nn.functional as F

consts = json.load(open(args.consts)) if args.consts else None


def bench():
    """Seconds for a fixed GPU workload (120 forward and backward passes of a small conv net), used as a stopwatch check."""
    g = torch.Generator(device="cuda").manual_seed(0)
    x = torch.randn(1024, 3, 32, 32, device="cuda", generator=g).half().contiguous(memory_format=torch.channels_last)
    net = torch.nn.Sequential(torch.nn.Conv2d(3, 64, 3, padding=1), torch.nn.SiLU(), torch.nn.MaxPool2d(2),
                              torch.nn.Conv2d(64, 256, 3, padding=1), torch.nn.SiLU(), torch.nn.MaxPool2d(2),
                              torch.nn.Conv2d(256, 256, 3, padding=1)).cuda().half().to(memory_format=torch.channels_last)
    def step():
        net.zero_grad(set_to_none=True)
        net(x).float().square().mean().backward()
    for _ in range(10):
        step()
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    for _ in range(120):
        step()
    torch.cuda.synchronize()
    return time.perf_counter() - t0


try:
    tr = torch.load(os.path.join(args.data, "train.pt"), map_location="cuda", weights_only=True)
    te = torch.load(os.path.join(args.data, "test.pt"), map_location="cuda", weights_only=True)
    train_images, train_labels = tr["images"], tr["labels"]
    test_images, test_labels = te["images"], te["labels"]
    perm = torch.randperm(len(test_labels), generator=torch.Generator().manual_seed(SPLIT_SEED)).cuda()
    val_idx, sealed_idx = perm[:5000], perm[5000:]
    b0 = bench()
except Exception:
    finish("infra_fail", error=traceback.format_exc()[-1500:])

logf = open(os.path.join(args.out, "cand.log"), "w")


def tail(n=40):
    logf.flush()
    return open(os.path.join(args.out, "cand.log")).read().splitlines()[-n:]


accs_v, accs_s, times = [], [], []
real_out = os.dup(1)
try:
    os.dup2(logf.fileno(), 1)
    os.dup2(logf.fileno(), 2)
    spec = importlib.util.spec_from_file_location("train", args.cand)
    cand = importlib.util.module_from_spec(spec)
    sys.modules["train"] = cand
    spec.loader.exec_module(cand)
    model = cand.make_model()
    torch.manual_seed(args.seed)
    g = torch.Generator(device="cuda").manual_seed(1)
    cand.train(model, torch.randint(0, 256, train_images.shape, device="cuda", dtype=torch.uint8, generator=g),
               torch.randint(0, 10, train_labels.shape, device="cuda", generator=g))
    cand.predict(model, torch.randint(0, 256, test_images.shape, device="cuda", dtype=torch.uint8, generator=g))
    torch.cuda.synchronize()
    res["warmup_s"] = round(time.time() - res["t_lock"], 1)
    budget = 2.5 * consts["t_host"].get(HOST, consts["t_ref"]) if consts else None
    for i in range(args.runs):
        torch.manual_seed(args.seed * 1000 + i)
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        cand.train(model, train_images, train_labels)
        logits = cand.predict(model, test_images)
        torch.cuda.synchronize()
        times.append(time.perf_counter() - t0)
        if tuple(logits.shape) != (len(test_labels), 10) or not bool(torch.isfinite(logits.float()).all()):
            raise RuntimeError("predict must return finite logits of shape (%d, 10), got %s"
                               % (len(test_labels), tuple(logits.shape)))
        ok = (logits.argmax(1) == test_labels)
        accs_v.append(100.0 * ok[val_idx].float().mean().item())
        accs_s.append(100.0 * ok[sealed_idx].float().mean().item())
        if budget and i == 2 and sorted(times)[1] > budget:
            os.dup2(real_out, 1)
            finish("overbudget", overbudget=True, times=times,
                   log_tail=["a run took %.1f s; the limit is %.1f s (2.5 times the baseline)" % (sorted(times)[1], budget)])
except Exception:
    traceback.print_exc()
    os.dup2(real_out, 1)
    finish("crash", log_tail=tail(), n_done=len(times))
os.dup2(real_out, 1)

b1 = bench()
others = gpu_others()
n = len(times)
t_med = sorted(times)[n // 2]
mean = lambda v: sum(v) / len(v)
sd = lambda v: (sum((x - mean(v)) ** 2 for x in v) / max(1, len(v) - 1)) ** 0.5
res.update(n=n, time_median=t_med, time_mean=mean(times), time_sd=sd(times), acc_val=mean(accs_v), acc_val_sd=sd(accs_v),
           acc_sealed=mean(accs_s) if args.sealed else None, times=times, accs_val=accs_v,
           accs_sealed=accs_s if args.sealed else None, bench=[b0, b1])
# Work carried from one run to the next would show as a first run that is slower or less accurate than the rest.
if n >= 6:
    rest_t, rest_a = sorted(times[1:])[(n - 1) // 2], mean(accs_v[1:])
    if times[0] > 1.3 * rest_t or times[0] < 0.7 * rest_t or abs(accs_v[0] - rest_a) > 1.5:
        res["integrity_failed"].append("first_run_differs")
if others:
    finish("infra_fail", error="GPU shared during the job: %s" % others)
if consts:
    ref_b = consts.get("bench_host", {}).get(HOST)
    if ref_b and max(abs(b0 / ref_b - 1), abs(b1 / ref_b - 1)) > BENCH_TOL:
        finish("infra_fail", error="stopwatch check failed: bench %.3f / %.3f s against %.3f s" % (b0, b1, ref_b))
    cost = lambda which, acc: math.log(t_med / consts["t_ref"]) + (consts["a_ref_" + which] - acc) / consts["k"]
    res["official_val_bpb"] = cost("val", res["acc_val"])
    if args.sealed:
        res["sealed_val_bpb"] = cost("sealed", mean(accs_s))
    res["log_tail"] = ["runs: %d" % n, "median run time: %.3f s (baseline on this machine: %.3f s)"
                       % (t_med, consts["t_host"].get(HOST, consts["t_ref"])),
                       "mean validation accuracy: %.3f percent (baseline: %.3f)" % (res["acc_val"], consts["a_ref_val"]),
                       "cost: %.4f" % res["official_val_bpb"]]
finish("ok")
