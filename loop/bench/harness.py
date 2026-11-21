#!/usr/bin/env python3
"""Locked CPU harness of the bench tasks: runs one candidate train.py on one task and scores it.

    python harness.py CAND --task NAME --out DIR [--seed 42] [--sealed] [--threads 8] [--timeout 600]

Result: DIR/result.json with a loss, lower is better (the engine stores it in its field official_val_bpb).

Family "admet" (tasks.json): the candidate defines fit_predict(train_smiles, train_y, test_smiles, seed).
  In a loop: 5-fold cross-validation inside the public training file, repeated with 3 fold assignments (15 fits),
  folds made of whole Murcko scaffolds (molecules that share a ring skeleton stay together, as in the benchmark's
  own split). loss = mean over the 15 folds of MAE, or 1 - AUROC, or 1 - Spearman. The assignments depend on --seed.
  With --sealed (confirmation only, never inside a loop): train on the whole training file with 5 seeds, predict the
  benchmark's held-out test file, loss per seed; sealed loss = their mean. The candidate never receives test labels.
Family "packing": the candidate defines solve(n, seed, time_limit) -> (centers, radii). Three seeds; each call has
  TIME_LIMIT seconds of CPU time and is void above 1.5 times that. The packing is checked exactly here.
  loss = minus the mean sum of radii. --sealed uses five fresh seeds instead.

Shared-machine safety: an address-space limit of 16 GB makes a runaway allocation fail inside Python as a
MemoryError (an ordinary crash result); a watchdog thread ends the job quietly at --mem-gb (default 6) of resident
memory or after --timeout seconds. No GPU is visible to the job.
"""
import argparse, importlib.util, json, math, os, resource, socket, subprocess, sys, threading, time, traceback

ap = argparse.ArgumentParser()
ap.add_argument("cand")
ap.add_argument("--task", required=True)
ap.add_argument("--out", required=True)
ap.add_argument("--seed", type=int, default=42)
ap.add_argument("--sealed", action="store_true")
ap.add_argument("--threads", type=int, default=8)
ap.add_argument("--timeout", type=float, default=600)
ap.add_argument("--mem-gb", type=float, default=6.0)
ap.add_argument("--job-id", default=None)
args = ap.parse_args()

HERE = os.path.dirname(os.path.abspath(__file__))
TASK = json.load(open(os.path.join(HERE, "tasks.json")))[args.task]
TIME_LIMIT = 30.0          # CPU seconds per solve() call (packing)
TOL = 1e-10                # allowed overlap or overhang of a packing
threads = 1 if TASK["family"] == "packing" else args.threads
for v in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "NUMEXPR_NUM_THREADS", "LOKY_MAX_CPU_COUNT"):
    os.environ[v] = str(threads)
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


def tree_gb():
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
            g = tree_gb()
            peak = max(peak, g)
            res["peak_rss_gb"] = round(peak, 2)
            with open(os.path.join(args.out, "heartbeat"), "w") as f:
                f.write("%d %.2f\n" % (time.time(), g))
            if g > args.mem_gb:
                finish("crash", reason="memory_limit",
                       log_tail=["the job used %.1f GB of memory; the limit is %.1f GB" % (g, args.mem_gb)])
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


def load_candidate():
    spec = importlib.util.spec_from_file_location("train", args.cand)
    cand = importlib.util.module_from_spec(spec)
    sys.modules["train"] = cand
    spec.loader.exec_module(cand)
    return cand


# ------------------------------------------------------------------ admet

def metric_loss(y, p):
    from scipy.stats import spearmanr
    from sklearn.metrics import mean_absolute_error, roc_auc_score
    y, p = np.asarray(y, dtype=float), np.asarray(p, dtype=float)
    if p.shape != y.shape or not np.isfinite(p).all():
        raise RuntimeError("fit_predict must return one finite number per test molecule, got shape %s" % (p.shape,))
    m = TASK["metric"]
    if m == "mae":
        return float(mean_absolute_error(y, p))
    if m == "auroc":
        return 1.0 - float(roc_auc_score(y, p))
    rho = spearmanr(y, p).correlation
    return 1.0 - (0.0 if rho != rho else float(rho))


def scaffold_folds(smiles, seed, k=5):
    """Fold index per molecule; molecules with the same Murcko scaffold share a fold; folds of about equal size."""
    from rdkit import Chem
    from rdkit.Chem.Scaffolds import MurckoScaffold
    groups = {}
    for i, s in enumerate(smiles):
        try:
            sc = MurckoScaffold.MurckoScaffoldSmiles(mol=Chem.MolFromSmiles(s))
        except Exception:
            sc = s
        groups.setdefault(sc, []).append(i)
    order = list(groups.values())
    np.random.default_rng(seed).shuffle(order)
    order.sort(key=len, reverse=True)
    fold, sizes = np.zeros(len(smiles), dtype=int), [0] * k
    for g in order:
        j = int(np.argmin(sizes))
        fold[g] = j
        sizes[j] += len(g)
    return fold


def run_admet(cand):
    import pandas as pd
    d = os.path.join(HERE, "data", "admet_group", args.task)
    tv = pd.read_csv(os.path.join(d, "train_val.csv"))
    smiles, y = tv["Drug"].tolist(), tv["Y"].to_numpy()
    if args.sealed:
        te = pd.read_csv(os.path.join(d, "test.csv"))
        losses = [metric_loss(te["Y"].to_numpy(), cand.fit_predict(list(smiles), y.copy(), te["Drug"].tolist(), 100 + s))
                  for s in range(5)]
        return None, float(np.mean(losses)), {"sealed_losses": losses}
    losses = []
    for rep in range(3):   # three different fold assignments, so that one lucky split decides less
        fold = scaffold_folds(smiles, args.seed * 10 + rep)
        for j in range(5):
            tr, va = np.where(fold != j)[0], np.where(fold == j)[0]
            p = cand.fit_predict([smiles[i] for i in tr], y[tr].copy(), [smiles[i] for i in va], args.seed)
            losses.append(metric_loss(y[va], p))
    return float(np.mean(losses)), None, {"fold_losses": losses}


# ------------------------------------------------------------------ packing

def check_packing(n, out):
    c, r = np.asarray(out[0], dtype=np.float64), np.asarray(out[1], dtype=np.float64)
    if c.shape != (n, 2) or r.shape != (n,) or not (np.isfinite(c).all() and np.isfinite(r).all()):
        raise RuntimeError("solve must return centers of shape (%d, 2) and radii of shape (%d,)" % (n, n))
    if (r < 0).any():
        raise RuntimeError("a radius is negative")
    edge = np.max(np.concatenate([r - c[:, 0], r - c[:, 1], c[:, 0] + r - 1, c[:, 1] + r - 1]))
    if edge > TOL:
        raise RuntimeError("a circle leaves the unit square by %.3g" % edge)
    d = np.sqrt(((c[:, None] - c[None]) ** 2).sum(-1))
    over = r[:, None] + r[None] - d
    np.fill_diagonal(over, -1.0)
    if over.max() > TOL:
        i, j = np.unravel_index(np.argmax(over), over.shape)
        raise RuntimeError("circles %d and %d overlap by %.3g" % (i, j, over.max()))
    return float(r.sum())


def run_packing(cand):
    n = TASK["n"]
    seeds = [900000 + args.seed * 10 + i for i in range(5)] if args.sealed else [args.seed * 10 + i for i in range(3)]
    sums, cpu = [], []
    for s in seeds:
        t = time.process_time()
        out = cand.solve(n, s, TIME_LIMIT)
        used = time.process_time() - t
        if used > 1.5 * TIME_LIMIT:
            raise RuntimeError("solve used %.0f s of CPU time; the limit is %.0f s" % (used, TIME_LIMIT))
        sums.append(check_packing(n, out))
        cpu.append(used)
    loss = -float(np.mean(sums))
    extra = {"sums": sums, "cpu_s": cpu, "best": max(sums)}
    return (None, loss, extra) if args.sealed else (loss, None, extra)


try:
    os.dup2(logf.fileno(), 1)
    os.dup2(logf.fileno(), 2)
    cand = load_candidate()
    official, sealed, extra = (run_admet if TASK["family"] == "admet" else run_packing)(cand)
except BaseException:
    traceback.print_exc()
    os.dup2(real_out, 1)
    finish("crash", log_tail=tail())
os.dup2(real_out, 1)
res.update(extra)
# The engine needs an official value on every run; in a sealed (confirmation) job it is the sealed value.
res["official_val_bpb"] = official if official is not None else sealed
res["sealed_val_bpb"] = sealed
res["log_tail"] = ["loss: %.5f" % res["official_val_bpb"]]
finish("ok")
