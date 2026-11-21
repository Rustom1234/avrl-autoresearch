#!/usr/bin/env python3
"""Write plan.json for a run of the bench tasks.

    python loop/bench/make_plan.py --root loop/runs/BG1 --seed 21 --slots 40 --per-host 2 \\
        --spec caco2_wang:2,lipophilicity_astrazeneca:2,bbb_martins:2,clearance_hepatocyte_az:2,pack26:3,pack32:3 --cells N,SS,HH

For every task and every repetition, one loop per cell; the whole list is shuffled by the seed. Cells as in
loop/cifar/make_plan.py: N random search over the Settings block (no model), SS Sonnet as planner, coder and
repairer, HH Haiku in those roles; analyst Sonnet, critic Opus, voter Haiku in both. Each lab machine runs
--per-host loops at a time (jobs are CPU jobs of at most 8 threads). Refuses to overwrite an existing plan.
"""
import argparse, json, random
from pathlib import Path

HERE = Path(__file__).resolve().parent
HOSTS = ["willow", "walnut", "pine", "cherry", "spruce"]
SHORT = {"caco2_wang": "caco", "lipophilicity_astrazeneca": "lipo", "bbb_martins": "bbb",
         "clearance_hepatocyte_az": "clear", "pack26": "p26", "pack32": "p32"}


def table(cell):
    m = {"SS": "sonnet", "HH": "haiku"}[cell]
    t = {"analyst": "sonnet", "critic": "opus", "voter": "haiku", "coder": m, "repairer": m}
    t.update({k: m for k in ("planner", "planner_backlog", "planner_cheap", "planner_refresh", "planner_widen")})
    return t


ap = argparse.ArgumentParser()
ap.add_argument("--root", required=True)
ap.add_argument("--seed", type=int, required=True)
ap.add_argument("--slots", type=int, default=40)
ap.add_argument("--per-host", type=int, default=2)
ap.add_argument("--spec", required=True)
ap.add_argument("--cells", default="N,SS,HH")
a = ap.parse_args()
root = Path(a.root).resolve()
assert not (root / "plan.json").exists(), "plan.json exists"
root.mkdir(parents=True, exist_ok=True)
loops = []
for item in a.spec.split(","):
    task, reps = item.split(":")
    for r in range(1, int(reps) + 1):
        for cell in a.cells.split(","):
            cfg = {"arm": "D", "cell": cell, "task": task, "rep": r, "n_slots": a.slots, "consts": str(HERE / "consts.json"),
                   "voter_model": "haiku", "critic_model": "opus", "vote_k": 3, "job_timeout_s": 600, "timeout_s": 3600,
                   "routing": None if cell == "N" else table(cell)}
            loops.append({"loop_id": "%s_%s_%d" % (SHORT[task], cell, r), "cmd": "nloop" if cell == "N" else "loop", "config": cfg})
random.Random(a.seed).shuffle(loops)
for i, lp in enumerate(loops):
    lp["config"]["rng_seed"] = a.seed * 1000 + i
(root / "plan.json").write_text(json.dumps({"hosts": [h for h in HOSTS for _ in range(a.per_host)], "seed": a.seed, "loops": loops}, indent=1))
print("wrote", root / "plan.json", "with", len(loops), "loops:", " ".join(l["loop_id"] for l in loops))
