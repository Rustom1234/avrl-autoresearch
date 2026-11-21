#!/usr/bin/env python3
"""Confirmation and report of a bench run.

    python loop/bench/finish.py --root loop/runs/BG1

Confirmation: one sealed job for the starting script of every task and one for every loop's final train.py that
differs from it (harness.py --sealed: drug-property tasks are trained on the whole training file with 5 seeds and
scored on the benchmark's held-out test molecules; packing runs 5 fresh seeds, the same 5 for every script).
progress of a loop = (sealed loss of the start - sealed loss of its final script) / (sealed loss of the start -
reference loss), where the reference is the leaderboard's first place or the best published packing (tasks.json).
0 = no better than the start, 1 = level with the reference. A loop that accepted nothing has progress 0.
Results are kept in confirmation/*.json, so the command can be run again. Then the checks of BG1_PREREG.md.
"""
import argparse, hashlib, importlib.util, json, math, random, sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from statistics import fmean, stdev

HERE = Path(__file__).resolve().parent
argv, sys.argv = sys.argv[1:], sys.argv[:1]
spec = importlib.util.spec_from_file_location("btask", HERE / "task.py")
B = importlib.util.module_from_spec(spec)
spec.loader.exec_module(B)
ap = argparse.ArgumentParser()
ap.add_argument("--root", required=True)
a = ap.parse_args(argv)
root = Path(a.root).resolve()
plan = json.loads((root / "plan.json").read_text())
out = root / "confirmation"
out.mkdir(exist_ok=True)
hosts = sorted(set(plan["hosts"]))
sha = lambda p: hashlib.sha256(Path(p).read_bytes()).hexdigest()
tasks = sorted({lp["config"]["task"] for lp in plan["loops"]})
want = [("BASE_" + t, t, str(B.base_path(t))) for t in tasks]
for lp in plan["loops"]:
    d, t = root / lp["loop_id"], lp["config"]["task"]
    if (d / "done").exists() and sha(d / "incumbent.py") != sha(B.base_path(t)):
        want.append((lp["loop_id"], t, str(d / "incumbent.py")))


def run(i):
    key, t, f = want[i]
    rf = out / (key + ".json")
    if rf.exists() and json.loads(rf.read_text())["status"] != "infra_fail":
        return
    svc = B.CpuJobs(hosts[i % len(hosts)], out / "jobs", t, job_timeout_s=900)
    jid, r = svc.submit(f, seed=7, sealed=True, tag="confirm-" + key, kind="confirm"), None
    while r is None:
        r = svc.wait(jid, 1800)
    rf.write_text(json.dumps(r))


with ThreadPoolExecutor(10) as ex:
    list(ex.map(run, range(len(want))))
R = lambda key: json.loads((out / (key + ".json")).read_text())


def ref_loss(t):
    T = B.TASKS[t]
    return -T["ref"] if T["family"] == "packing" else (T["ref"] if T["metric"] == "mae" else 1 - T["ref"])


def show(t, loss):
    """The task's own metric from a loss, and its rank on the leaderboard if there is one."""
    T = B.TASKS[t]
    if T["family"] == "packing":
        return "%.5f" % -loss
    v = loss if T["metric"] == "mae" else 1 - loss
    better = sum(1 for b in T["board"] if (b < v if T["metric"] == "mae" else b > v))
    return "%.3f (rank %d of %d)" % (v, better + 1, len(T["board"]) + 1)


rows = []
for lp in plan["loops"]:
    lid, t, cell = lp["loop_id"], lp["config"]["task"], lp["config"]["cell"]
    if not (root / lid / "done").exists():
        continue
    base = R("BASE_" + t)["sealed_val_bpb"]
    f = out / (lid + ".json")
    if f.exists():
        r = R(lid)
        final = r["sealed_val_bpb"] if r["status"] == "ok" else None
    else:
        final = base
    prog = None if final is None else (base - final) / (base - ref_loss(t))
    st = json.loads((root / lid / "state.json").read_text())
    rows.append(dict(loop=lid, task=t, family=B.TASKS[t]["family"], cell=cell, base=base, final=final, progress=prog,
                     inloop=st["incumbent"]["mean_adjusted"]))
(out / "summary.json").write_text(json.dumps(rows, indent=1))


def t_p(v):
    n, m, s = len(v), fmean(v), stdev(v)
    if s == 0:
        return float("inf"), 0.0 if m > 0 else 1.0
    t, df = m / (s / math.sqrt(n)), n - 1
    pdf = lambda x: math.gamma((df + 1) / 2) / (math.sqrt(df * math.pi) * math.gamma(df / 2)) * (1 + x * x / df) ** (-(df + 1) / 2)
    h = (80.0 - t) / 40000
    return t, (max(0.0, sum(pdf(t + (i + 0.5) * h) for i in range(40000)) * h) if t < 80 else 0.0)


def perm_p(rows, x, y, n=20000):
    """One-sided p for mean(progress of cell x) > mean(cell y), cell labels shuffled inside each task."""
    rs = [r for r in rows if r["cell"] in (x, y) and r["progress"] is not None]
    d = lambda lab: fmean(r["progress"] for r, l in zip(rs, lab) if l == x) - fmean(r["progress"] for r, l in zip(rs, lab) if l == y)
    obs, rng, hit = d([r["cell"] for r in rs]), random.Random(0), 0
    by = {}
    for i, r in enumerate(rs):
        by.setdefault(r["task"], []).append(i)
    for _ in range(n):
        lab = [None] * len(rs)
        for idx in by.values():
            ls = [rs[i]["cell"] for i in idx]
            rng.shuffle(ls)
            for i, l in zip(idx, ls):
                lab[i] = l
        hit += d(lab) >= obs - 1e-12
    return hit / n


print("| Task | Start | Reference | Cell | Final scores of its loops | Mean progress |")
print("|---|---|---|---|---|---|")
for t in tasks:
    for cell in ("N", "SS", "HH"):
        rs = [r for r in rows if r["task"] == t and r["cell"] == cell and r["progress"] is not None]
        if rs:
            print("| %s | %s | %s | %s | %s | %.2f |" % (t, show(t, rs[0]["base"]), show(t, ref_loss(t)), cell,
                                                        "; ".join(show(t, r["final"]) for r in rs), fmean(r["progress"] for r in rs)))
for fam in ("admet", "packing"):
    fr = [r for r in rows if r["family"] == fam and r["progress"] is not None]
    if not fr:
        continue
    g = {c: [r["progress"] for r in fr if r["cell"] == c] for c in ("N", "SS", "HH")}
    print("\n%s: mean progress N %.3f (n=%d), SS %.3f (n=%d), HH %.3f (n=%d)" % (
        fam, fmean(g["N"]), len(g["N"]), fmean(g["SS"]), len(g["SS"]), fmean(g["HH"]), len(g["HH"])))
    t, p = t_p(g["SS"])
    print("  A headroom: SS %.3f (needs >= 0.25), t = %.2f, p = %.4f -> %s" % (fmean(g["SS"]), t, p, "PASS" if fmean(g["SS"]) >= 0.25 and p < 0.05 else "FAIL"))
    pb = perm_p(fr, "SS", "N")
    print("  B menu: N / SS = %.2f (needs <= 0.33), p = %.4f -> %s" % (fmean(g["N"]) / fmean(g["SS"]), pb, "PASS" if fmean(g["N"]) <= fmean(g["SS"]) / 3 and pb < 0.05 else "FAIL"))
    print("  D gap: HH / SS = %.2f (needs <= 0.67), SS above HH p = %.4f -> %s" % (fmean(g["HH"]) / fmean(g["SS"]), perm_p(fr, "SS", "HH"), "PASS" if fmean(g["HH"]) <= 2 * fmean(g["SS"]) / 3 else "FAIL"))
