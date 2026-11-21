#!/usr/bin/env python3
"""Write plan.json for a run of the ShinkaEvolve circle-packing task.

    python loop/shinka/make_plan.py --root loop/runs/S1 --seed 41 --slots 150 --cells SS:3,HH:3,SH:3 --per-host 2

Cells are two letters, planner then coder (the repairer goes with the coder): S = Sonnet, H = Haiku. Analyst
Sonnet, critic Opus, voter Haiku in every cell, as in BG1 and H1. Refuses to overwrite an existing plan.
"""
import argparse, json, random
from pathlib import Path

HERE = Path(__file__).resolve().parent
M = {"H": "haiku", "S": "sonnet"}
ap = argparse.ArgumentParser()
ap.add_argument("--root", required=True)
ap.add_argument("--seed", type=int, required=True)
ap.add_argument("--slots", type=int, default=150)
ap.add_argument("--cells", default="SS:3,HH:3,SH:3")
ap.add_argument("--per-host", type=int, default=2)
ap.add_argument("--hosts", default="willow,walnut,pine,cherry,spruce")
a = ap.parse_args()
root = Path(a.root).resolve()
assert not (root / "plan.json").exists(), "plan.json exists"
root.mkdir(parents=True, exist_ok=True)
loops = []
for item in a.cells.split(","):
    cell, reps = item.split(":")
    p, c = M[cell[0]], M[cell[1]]
    t = {"analyst": "sonnet", "critic": "opus", "voter": "haiku", "coder": c, "repairer": c}
    t.update({k: p for k in ("planner", "planner_backlog", "planner_cheap", "planner_refresh", "planner_widen")})
    for r in range(1, int(reps) + 1):
        cfg = {"arm": "D", "cell": cell, "task": "pack26s", "rep": r, "n_slots": a.slots, "consts": str(HERE / "consts.json"),
               "base_train_py": str(HERE / "initial_shinka.py"), "voter_model": "haiku", "critic_model": "opus", "vote_k": 3,
               "job_timeout_s": 300, "timeout_s": 3600, "routing": t}
        loops.append({"loop_id": "s26_%s_%d" % (cell, r), "cmd": "loop", "config": cfg})
random.Random(a.seed).shuffle(loops)
for i, lp in enumerate(loops):
    lp["config"]["rng_seed"] = a.seed * 1000 + i
(root / "plan.json").write_text(json.dumps({"hosts": [h for h in a.hosts.split(",") for _ in range(a.per_host)], "seed": a.seed, "loops": loops}, indent=1))
print("wrote", root / "plan.json", "with", len(loops), "loops:", " ".join(l["loop_id"] for l in loops))
