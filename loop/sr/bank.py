#!/usr/bin/env python3
"""Collect attempts: every model tries every problem k times, each attempt stored once.

    bank.py build --problems loop/sr/problems --root loop/runs/SR0 --models haiku sonnet fable --k 3 [--workers 4]
    bank.py table --root loop/runs/SR0 [--tau 1e-10]

build   runs the missing attempts (problem, model, k) with session.run_attempt and stores each under
        <root>/attempts/<problem>/<model>/<k>/record.json. Safe to stop and start again: stored attempts are kept.
        Stops by itself on: a <root>/STOP file, a login failure (writes STOP), the dollar cap --max-usd, or a rate
        limit that does not clear (the loop's breaker waits; a limited attempt is put back at most twice).
        A session that breaks for any other reason is logged in failures.jsonl and is NOT tried again.
        Waits while the 1 minute load of this machine is above --max-load (never adds to a busy machine).
table   successes per problem and model at tolerance tau on the validation split, with held-out confirmation.

An attempt never sees the validation or test files; they are read only by our scorer after the session ends.
"""
import argparse, json, shutil, sys, threading, time
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))


def rec_path(root, p, m, k):
    return root / "attempts" / p / m / str(k) / "record.json"


def load(root):
    out = []
    for f in sorted(Path(root).glob("attempts/*/*/*/record.json")):
        r = json.loads(f.read_text())
        if r.get("ok"):
            r["k"] = int(f.parent.name)
            out.append(r)
    return out


def cmd_build(a):
    import session as S
    root, pr = Path(a.root).resolve(), Path(a.problems).resolve()
    root.mkdir(parents=True, exist_ok=True)
    probs = sorted(d.name for d in pr.iterdir() if (d / "card.json").exists())
    if a.only:
        probs = [p for p in probs if any(s in p for s in a.only)]
    jobs = [(p, m, k) for k in range(a.k) for p in probs for m in a.models if not rec_path(root, p, m, k).exists()]
    lock, state = threading.Lock(), {"usd": 0.0, "done": 0, "fail": 0, "stop": None}

    def log(msg):
        with lock, open(root / "bank.log", "a") as f:
            f.write("%s %s\n" % (time.strftime("%m-%d %H:%M:%S"), msg))

    log("build: %d attempts to run (%d problems, models %s, k %d, workers %d)" % (
        len(jobs), len(probs), " ".join(a.models), a.k, a.workers))

    def one(job):
        p, m, k = job
        for again in range(3):
            if state["stop"] or (root / "STOP").exists():
                return
            while float(Path("/proc/loadavg").read_text().split()[0]) > a.max_load:
                time.sleep(30)
            d = rec_path(root, p, m, k).parent
            if d.exists():
                shutil.rmtree(d)
            r = S.run_attempt(pr / p, m, d, budget=a.budget)
            with lock:
                state["usd"] += r.get("cost_usd") or 0.0
                over = state["usd"] > a.max_usd
            if r["ok"]:
                r["k"] = k
                (d / "record.json").write_text(json.dumps(r, indent=1))
                with lock:
                    state["done"] += 1
                log("%s %s %d: evals %s val %s ood %s usd %.4f wall %.0fs" % (
                    p, m, k, r.get("n_evals"), "%.1e" % r["val"] if r.get("val") is not None else None,
                    "%.1e" % r["test_ood"] if r.get("test_ood") is not None else None, r["cost_usd"], r["wall_s"]))
            else:
                with lock, open(root / "failures.jsonl", "a") as f:
                    f.write(json.dumps(dict(r, k=k)) + "\n")
                shutil.move(str(d), str(d.parent / ("%d_failed_%d" % (k, int(time.time())))))
                log("%s %s %d: FAILED (%s) %s" % (p, m, k, r["error_class"], (r["error"] or "")[:160]))
                if r["error_class"] == "auth":
                    (root / "STOP").write_text("login\n")
                    state["stop"] = "login failure"
                    return
                if r["error_class"] == "rate" and again < 2:
                    S.C._trip_breaker(S.C._backoff(again), "sr bank rate limit")
                    continue                     # the breaker makes every caller wait; then this attempt once more
                with lock:
                    state["fail"] += 1
            if over:
                state["stop"] = "dollar cap %.2f reached" % a.max_usd
            return

    with ThreadPoolExecutor(a.workers) as ex:
        list(ex.map(one, jobs))
    log("build finished: %d stored, %d failed, %.2f USD nominal%s" % (
        state["done"], state["fail"], state["usd"], ", stopped: " + state["stop"] if state["stop"] else ""))
    print(open(root / "bank.log").read().splitlines()[-1])


def cmd_table(a):
    R = load(a.root)
    by = defaultdict(list)
    for r in R:
        by[(r["problem"], r["model"])].append(r)
    models = [m for m in ("haiku", "sonnet", "opus", "fable") if any(k[1] == m for k in by)]
    hit = lambda r, s: r.get(s) is not None and r[s] <= a.tau
    print("success = validation NMSE at most %g; in brackets: also at most %g outside the training range" % (a.tau, a.tau))
    print("| problem | " + " | ".join(models) + " |\n|---|" + "---|" * len(models))
    tot = {m: [0, 0, 0, 0.0, 0.0] for m in models}
    for p in sorted({k[0] for k in by}):
        cells = []
        for m in models:
            rs = by.get((p, m), [])
            s, c = sum(hit(r, "val") for r in rs), sum(hit(r, "val") and hit(r, "test_ood") for r in rs)
            cells.append("%d (%d) of %d" % (s, c, len(rs)) if rs else "-")
            t = tot[m]
            t[0] += s; t[1] += c; t[2] += len(rs); t[3] += sum(r["cost_usd"] for r in rs); t[4] += sum(r["wall_s"] for r in rs)
        print("| %s | %s |" % (p, " | ".join(cells)))
    print("| all | " + " | ".join("%d (%d) of %d" % tuple(tot[m][:3]) for m in models) + " |")
    print("\n| model | attempts | success rate | USD per attempt | USD per success | minutes per attempt |\n|---|---|---|---|---|---|")
    for m in models:
        s, c, n, usd, wall = tot[m]
        if n:
            print("| %s | %d | %.2f | %.4f | %s | %.1f |" % (m, n, s / n, usd / n, "%.3f" % (usd / s) if s else "none", wall / n / 60))
    lv = defaultdict(lambda: defaultdict(lambda: [0, 0]))
    for (p, m), rs in by.items():
        L = p.split("_")[1]
        lv[L][m][0] += sum(hit(r, "val") for r in rs); lv[L][m][1] += len(rs)
    print("\n| level | " + " | ".join(models) + " |\n|---|" + "---|" * len(models))
    for L in sorted(lv):
        print("| %s | %s |" % (L, " | ".join("%d of %d" % tuple(lv[L][m]) for m in models)))


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    p = sub.add_parser("build")
    p.add_argument("--problems", required=True)
    p.add_argument("--root", required=True)
    p.add_argument("--models", nargs="+", required=True)
    p.add_argument("--k", type=int, required=True)
    p.add_argument("--workers", type=int, default=4)
    p.add_argument("--budget", type=int, default=14)
    p.add_argument("--max-usd", type=float, default=40.0)
    p.add_argument("--max-load", type=float, default=16.0)
    p.add_argument("--only", nargs="*", help="keep problems whose id contains one of these strings")
    p = sub.add_parser("table")
    p.add_argument("--root", required=True)
    p.add_argument("--tau", type=float, default=1e-10)
    a = ap.parse_args()
    {"build": cmd_build, "table": cmd_table}[a.cmd](a)
