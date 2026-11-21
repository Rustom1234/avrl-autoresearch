#!/usr/bin/env python3
"""Attempts that build on each other, stored once and shared by every policy that takes the same path.

    trie.py run --problems loop/sr/problems_w0 --root loop/runs/SR0C --plans "haiku*6" "haiku*3,fable" \
                [--reps 1] [--only L2_ L6_] [--workers 4]
    trie.py table --root loop/runs/SR0C

A path is a list of models, for example haiku,haiku,fable. Step i is one session by that model which is shown the
three best candidates found by the earlier steps of the same path (none at step 0). The record of a step is stored
at <root>/trie/<problem>/r<rep>/<m0>/<m1>/.../record.json, so two plans with the same beginning share those steps
(this is the bank of the Nov 18 plan: an attempt is made once and returned to every policy that asks for it).
A path ends at the first step that reaches the target. No step is run twice; a broken session ends its path.
"""
import argparse, json, sys, threading, time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
TAU = 1e-10


def parse(plan):
    out = []
    for part in plan.split(","):
        m, _, n = part.partition("*")
        out += [m.strip()] * int(n or 1)
    return out


def node(root, problem, rep, arms):
    return Path(root) / "trie" / problem / ("r%d" % rep) / Path(*arms)


def solved(r):
    return r.get("val") is not None and r["val"] <= TAU


def step(S, root, pdir, rep, arms, budget, log):
    """Make sure the last step of `arms` exists (its ancestors must exist). Returns its record or None."""
    d = node(root, pdir.name, rep, arms)
    if (d / "record.json").exists():
        return json.loads((d / "record.json").read_text())
    if (d / "failure.json").exists():
        return None
    anc = [json.loads((node(root, pdir.name, rep, arms[:i]) / "record.json").read_text()) for i in range(1, len(arms))]
    cands = {}
    for a in anc:
        if a.get("best"):
            cands.setdefault(a["best"]["code"], a["best"]["train_nmse"])
    parents = [{"code": c, "train_nmse": v} for c, v in sorted(cands.items(), key=lambda kv: kv[1])[:3]]
    r = S.run_attempt(pdir, arms[-1], d / "work", budget=budget, parents=parents)
    r["path"], r["rep"] = arms, rep
    r["parent_best"] = parents[0]["train_nmse"] if parents else None
    if not r["ok"]:
        (d / "failure.json").write_text(json.dumps(r))
        log("%s r%d %s: FAILED (%s) %s" % (pdir.name, rep, ",".join(arms), r["error_class"], (r["error"] or "")[:120]))
        return None
    (d / "record.json").write_text(json.dumps(r, indent=1))
    log("%s r%d %s: train %s val %s usd %.4f" % (
        pdir.name, rep, ",".join(a[0] for a in arms), "%.1e" % r["best"]["train_nmse"] if r.get("best") else None,
        "%.1e" % r["val"] if r.get("val") is not None else None, r["cost_usd"]))
    return r


def cmd_run(a):
    import session as S
    root, pr = Path(a.root).resolve(), Path(a.problems).resolve()
    root.mkdir(parents=True, exist_ok=True)
    probs = sorted(d for d in pr.iterdir() if (d / "card.json").exists() and (not a.only or any(s in d.name for s in a.only)))
    plans = [parse(p) for p in a.plans]
    lock, state = threading.Lock(), {"usd": 0.0, "stop": None}

    def log(msg):
        with lock, open(root / "trie.log", "a") as f:
            f.write("%s %s\n" % (time.strftime("%m-%d %H:%M:%S"), msg))

    def one(job):
        pdir, rep = job
        for plan in plans:                       # plans of one (problem, rep) run in turn: they share steps
            for i in range(1, len(plan) + 1):
                if state["stop"] or (root / "STOP").exists():
                    return
                while float(Path("/proc/loadavg").read_text().split()[0]) > a.max_load:
                    time.sleep(30)
                new = not (node(root, pdir.name, rep, plan[:i]) / "record.json").exists()
                r = step(S, root, pdir, rep, plan[:i], a.budget, log)
                if r is None:
                    f = node(root, pdir.name, rep, plan[:i]) / "failure.json"
                    if f.exists() and json.loads(f.read_text()).get("error_class") == "auth":
                        (root / "STOP").write_text("login\n")
                        state["stop"] = "login failure"
                    break
                if new:
                    with lock:
                        state["usd"] += r["cost_usd"]
                        if state["usd"] > a.max_usd:
                            state["stop"] = "dollar cap %.2f reached" % a.max_usd
                if solved(r):
                    break

    log("run: %d problems x %d reps, plans %s" % (len(probs), a.reps, a.plans))
    with ThreadPoolExecutor(a.workers) as ex:
        list(ex.map(one, [(p, r) for r in range(a.reps) for p in probs]))
    log("run finished: %.2f USD nominal for new steps%s" % (state["usd"], ", stopped: " + state["stop"] if state["stop"] else ""))
    print(open(root / "trie.log").read().splitlines()[-1])


def cmd_table(a):
    recs = [json.loads(f.read_text()) for f in sorted(Path(a.root).glob("trie/**/record.json"))]
    print("| problem | rep | path | parent best | this step best | solved | USD |\n|---|---|---|---|---|---|---|")
    for r in sorted(recs, key=lambda r: (r["problem"], r["rep"], len(r["path"]), r["path"])):
        f = lambda v: "%.0e" % v if v is not None else "-"
        print("| %s | %d | %s | %s | %s | %s | %.3f |" % (
            r["problem"], r["rep"], "".join(m[0] for m in r["path"]), f(r["parent_best"]),
            f((r.get("best") or {}).get("train_nmse")), "yes" if solved(r) else "", r["cost_usd"]))


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    p = sub.add_parser("run")
    p.add_argument("--problems", required=True)
    p.add_argument("--root", required=True)
    p.add_argument("--plans", nargs="+", required=True)
    p.add_argument("--reps", type=int, default=1)
    p.add_argument("--only", nargs="*")
    p.add_argument("--workers", type=int, default=4)
    p.add_argument("--budget", type=int, default=14)
    p.add_argument("--max-usd", type=float, default=10.0)
    p.add_argument("--max-load", type=float, default=16.0)
    p = sub.add_parser("table")
    p.add_argument("--root", required=True)
    a = ap.parse_args()
    {"run": cmd_run, "table": cmd_table}[a.cmd](a)
