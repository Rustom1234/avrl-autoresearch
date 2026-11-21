#!/usr/bin/env python3
"""K1: first idea lists for the six BG1 tasks, written outside any loop (no GPU run, nothing scored).

    python loop/k1/lists.py

For each task and each of sonnet and haiku, REPS calls with exactly the prompt a loop's first planner call gets
(the task's program card as system prompt; the starting script, an empty results table and no notes as the
message). Output: loop/k1/bg1_lists.jsonl, one line per list. Existing lines are kept, so it can be rerun.
"""
import importlib.util, json, sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]
sys.argv = sys.argv[:1]
spec = importlib.util.spec_from_file_location("btask", REPO / "loop/bench/task.py")
B = importlib.util.module_from_spec(spec)
spec.loader.exec_module(B)
sys.path.insert(0, str(REPO / "loop/models"))
import call as C

REPS, OUT = 3, HERE / "bg1_lists.jsonl"
consts = json.loads((REPO / "loop/bench/consts.json").read_text())
have = set()
if OUT.exists():
    have = {(r["task"], r["model"], r["rep"]) for r in map(json.loads, open(OUT)) if r["items"]}
todo = [(t, m, r) for t in B.TASKS for m in ("sonnet", "haiku") for r in range(REPS) if (t, m, r) not in have]


def one(job):
    t, m, r = job
    P = B.make_prompts(t, consts[t])
    user = P.planner(B.base_path(t).read_text(), 30, "no experiments yet", "", {}, [])
    res = C.call(m, user, system=P.program_card(40), schema=P.SCHEMAS["planner_backlog"], max_tokens=8192,
                 tag="k1:%s:%s:%d" % (t, m, r), log_path=str(HERE / "calls.jsonl"), timeout_s=900)
    return dict(task=t, model=m, rep=r, ok=res.ok, items=(res.obj or {}).get("items", []) if res.ok else [])


with ThreadPoolExecutor(6) as ex, open(OUT, "a") as f:
    for row in ex.map(one, todo):
        f.write(json.dumps(row) + "\n")
        f.flush()
        print(row["task"], row["model"], row["rep"], row["ok"], len(row["items"]), flush=True)
