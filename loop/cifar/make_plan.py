#!/usr/bin/env python3
"""Write plan.json for a run of the CIFAR task.

    python loop/cifar/make_plan.py --root loop/runs/CG1 --seed 11 --blocks 6 --cells N,SS,N,HH [--slots 20]

One block holds one loop per listed cell (a cell may be listed twice), in an order shuffled by the seed. Cells:
  N    random search over the settings block, no model
  SS   Sonnet in every role
  HH   Haiku as planner, coder and repairer
  ST   Sonnet plans, Haiku codes and repairs
  TS   Haiku plans, Sonnet codes and repairs
  F    Sonnet writes the first idea list only; everything else as HH
In every model cell the analyst is Sonnet, the critic Opus and the voter Haiku, as P2B held them fixed across cells
(P2B's voter was an open-weight model on a lab GPU; here the lab GPUs are the stopwatches, so the voter is Haiku).
Refuses to overwrite an existing plan.
"""
import argparse, json, random
from pathlib import Path

HERE = Path(__file__).resolve().parent
HOSTS = ["willow", "walnut", "pine", "cherry", "spruce"]


def table(cell):
    p, c = {"SS": ("sonnet", "sonnet"), "HH": ("haiku", "haiku"), "ST": ("sonnet", "haiku"),
            "TS": ("haiku", "sonnet"), "F": ("haiku", "haiku")}[cell]
    t = {"analyst": "sonnet", "critic": "opus", "voter": "haiku", "coder": c, "repairer": c}
    t.update({k: p for k in ("planner", "planner_backlog", "planner_cheap", "planner_refresh", "planner_widen")})
    if cell == "F":
        t["planner_backlog"] = "sonnet"
    return t


ap = argparse.ArgumentParser()
ap.add_argument("--root", required=True)
ap.add_argument("--seed", type=int, required=True)
ap.add_argument("--blocks", type=int, required=True)
ap.add_argument("--cells", required=True)
ap.add_argument("--slots", type=int, default=20)
a = ap.parse_args()
root = Path(a.root).resolve()
assert not (root / "plan.json").exists(), "plan.json exists"
root.mkdir(parents=True, exist_ok=True)
rng, loops, count = random.Random(a.seed), [], {}
for b in range(1, a.blocks + 1):
    cells = a.cells.split(",")
    rng.shuffle(cells)
    for cell in cells:
        count[cell] = count.get(cell, 0) + 1
        cfg = {"arm": "D", "cell": cell, "block": b, "n_slots": a.slots, "rng_seed": a.seed * 1000 + len(loops),
               "consts": str(HERE / "consts.json"), "voter_model": "haiku", "critic_model": "opus", "vote_k": 3,
               "job_timeout_s": 900, "timeout_s": 3600, "runs": 24,
               "routing": None if cell == "N" else table(cell)}
        loops.append({"loop_id": "%s_%02d" % (cell, count[cell]), "cmd": "nloop" if cell == "N" else "loop",
                      "config": cfg})
(root / "plan.json").write_text(json.dumps({"hosts": HOSTS, "seed": a.seed, "loops": loops}, indent=1))
print("wrote", root / "plan.json", "with", len(loops), "loops:", " ".join(l["loop_id"] for l in loops))
