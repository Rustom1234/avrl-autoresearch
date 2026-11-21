#!/usr/bin/env python3
"""Can a model tell, before the test, which ideas will pass?  One score per idea, blind to the outcome.

    score.py bank  --root loop/runs/B1                  score the ideas of the stall bank (before or while they run)
    score.py retro --root loop/runs/B1 --e1 loop/runs/E1 --src loop/runs/BG1_R
                                                        score ideas the E1 continuation tested against the slot-40
                                                        incumbent (outcomes exist, the scorer does not see them)

The scorer sees what the loop knew at slot 40 (program card, incumbent file, results table, notes) and a shuffled
list of ideas, and gives each a probability from 0 to 100 that the idea, implemented as one edit, would be accepted.
Scorers: haiku, sonnet, opus. Two passes per scorer with different orders (the second measures how stable a score
is). One try per call: a failed call stays recorded and is not repeated.
Output: <root>/scores/<set>_<loop>.jsonl, one line per call.
"""
import argparse, importlib.util, json, random, sys, threading
from pathlib import Path

HERE = Path(__file__).resolve().parent
LOOP = HERE.parent
sys.path.insert(0, str(LOOP / "core"))
_spec = importlib.util.spec_from_file_location("bench_task", LOOP / "bench" / "task.py")
bench = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(bench)
import ledger                              # noqa: E402
import prompts as P0                       # noqa: E402
from services import ModelService          # noqa: E402

MODELS = ("haiku", "sonnet", "opus")
SCHEMA = {"type": "object", "required": ["scores"], "properties": {"scores": {"type": "array", "items": {
    "type": "object", "required": ["id", "p"], "properties": {"id": {"type": "string"}, "p": {"type": "number"}}}}}}
BATCH = 25


def prompt(inc_src, table, memo, family_stats, ideas):
    lst = "\n".join("- [%s] %s: %s" % (i["id"], i["title"], i["spec"]) for i in ideas)
    return "\n\n".join([
        P0._file(inc_src), "## Results so far\n%s" % table,
        "## Notes from earlier experiments\n%s\nFamily outcomes: %s" % (memo or "none yet", P0._fam(family_stats)),
        "## Candidate ideas\n%s" % lst,
        "## Task\nEach candidate idea would be implemented by a coder as one edit to train.py and run once. It is "
        "accepted only if its loss is lower than the incumbent's by clearly more than the noise. For every idea give "
        "p, your probability from 0 to 100 that it would be accepted. Most ideas at this stage fail, so use the full "
        'range and keep high numbers for the few you would bet on. Reply as JSON: {"scores": [{"id", "p"}]} with one '
        "entry per idea."])


USE_MODELS, USE_REPS = list(MODELS), (0, 1)


def score_set(root, name, lid, task, inc_src, table, memo, fam, ideas, seed):
    """All scorer calls for one loop's idea list."""
    consts = json.loads((LOOP / "bench" / "consts.json").read_text())
    P = bench.make_prompts(task, consts[task])
    d = root / "scores"
    d.mkdir(exist_ok=True)
    out = d / ("%s_%s.jsonl" % (name, lid))
    have = {(r["model"], r["rep"], r["batch"]) for r in map(json.loads, open(out))} if out.exists() else set()
    models = ModelService(str(d), "SC_%s_%s" % (name, lid), max_tokens=8192)
    for m in USE_MODELS:
        for rep in USE_REPS:
            order = list(ideas)
            random.Random(seed * 100 + rep * 10 + MODELS.index(m)).shuffle(order)
            for b in range(0, len(order), BATCH):
                if (m, rep, b) in have:
                    continue
                chunk = order[b:b + BATCH]
                r = models.call("scorer", m, prompt(inc_src, table, memo, fam, chunk), system=P.program_card(80),
                                schema=SCHEMA)
                sc = {str(x["id"]).strip("[] "): float(x["p"]) for x in (r.obj or {}).get("scores", [])} if r.ok else {}
                with open(out, "a") as f:
                    f.write(json.dumps({"loop": lid, "model": m, "rep": rep, "batch": b, "ok": r.ok, "error": r.error,
                                        "ids": [i["id"] for i in chunk], "scores": sc,
                                        "cost_usd_nominal": r.cost_usd_nominal, "latency_s": r.latency_s}) + "\n")
                print("%s %s %s rep%d batch%d ok=%s n=%d cost=%.3f" % (name, lid, m, rep, b, r.ok, len(sc),
                                                                    r.cost_usd_nominal), flush=True)


def cmd_bank(a):
    root = Path(a.root).resolve()
    jobs = []
    for d in sorted(p.parent for p in root.glob("*/ideas.json")):
        fz = json.loads((d / "frozen.json").read_text())
        ideas = json.loads((d / "ideas.json").read_text())
        table = ledger.results_table(fz["records"], fz["incumbent"])
        jobs.append((root, "bank", d.name, fz["task"], (d / "incumbent0.py").read_text(), table, fz["memo"],
                     fz["family_stats"], ideas, fz["config"]["rng_seed"]))
    run(jobs)


def cmd_retro(a):
    root, e1, src = Path(a.root).resolve(), Path(a.e1).resolve(), Path(a.src).resolve()
    (root / "retro").mkdir(exist_ok=True)
    jobs = []
    for d in sorted(p.parent for p in e1.glob("*/ledger.jsonl")):
        lid = d.name
        st0 = json.loads((src / lid / "state.json").read_text())      # the loop at slot 40
        cfg0 = json.loads((src / lid / "config.json").read_text())
        items = {i["id"]: i for i in json.loads((d / "state.json").read_text())["backlog"]}
        inc0 = st0["incumbent"]
        ideas, seen = [], set()
        for r in ledger.Ledger(d / "ledger.jsonl").read():
            if r["slot"] < 40 or r["kind"] != "candidate" or r["idea_id"] in seen or r["idea_id"] not in items:
                continue
            if abs(r["incumbent_before"]["mean"] - inc0["mean_adjusted"]) > 1e-12:
                continue                                              # tested after an accept: another incumbent
            seen.add(r["idea_id"])
            it = items[r["idea_id"]]
            ideas.append({"id": it["id"], "title": it["title"], "spec": it["spec"], "family": it["family"],
                          "z1": r.get("z1"), "status": r.get("status"), "decision": r.get("decision"),
                          "ran": r.get("job_id") is not None, "slot": r["slot"],
                          "accepted": False, "alpha": it["alpha"], "beta": it["beta"]})
        acc = {r["idea_id"] for r in ledger.Ledger(d / "ledger.jsonl").read() if r["decision"] == "accept" and r["slot"] >= 40}
        for i in ideas:
            i["accepted"] = i["id"] in acc
        (root / "retro" / (lid + ".json")).write_text(json.dumps(ideas, indent=1))
        records0 = ledger.Ledger(src / lid / "ledger.jsonl").read()
        table = ledger.results_table(records0, inc0)
        blind = [{k: i[k] for k in ("id", "title", "spec")} for i in ideas]
        jobs.append((root, "retro", lid, cfg0["task"], Path(inc0["path"]).read_text(), table, st0["memo"],
                     st0["family_stats"], blind, cfg0["rng_seed"]))
    if a.loops:
        jobs = [j for j in jobs if j[2] in a.loops]
    print("retro: %d loops, %d ideas" % (len(jobs), sum(len(j[8]) for j in jobs)), flush=True)
    run(jobs)


def run(jobs, workers=6):
    it = iter(jobs)
    lock = threading.Lock()

    def work():
        while True:
            with lock:
                j = next(it, None)
            if j is None:
                return
            score_set(*j)

    ts = [threading.Thread(target=work) for _ in range(workers)]
    for t in ts:
        t.start()
    for t in ts:
        t.join()


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    sub = ap.add_subparsers(dest="cmd", required=True)
    p = sub.add_parser("bank")
    p.add_argument("--root", required=True)
    p = sub.add_parser("retro")
    p.add_argument("--root", required=True)
    p.add_argument("--e1", required=True)
    p.add_argument("--src", required=True)
    p.add_argument("--loops", nargs="*")
    p.add_argument("--models", nargs="+", default=list(MODELS))
    p.add_argument("--reps", type=int, default=2)
    a = ap.parse_args()
    if a.cmd == "retro":
        USE_MODELS[:] = a.models
        USE_REPS = tuple(range(a.reps))
    {"bank": cmd_bank, "retro": cmd_retro}[a.cmd](a)
