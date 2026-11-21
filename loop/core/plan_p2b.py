#!/usr/bin/env python3
"""P2B plan builder: the P2 replication with first-list (F), random-table (R) and optional mid-planner (M) cells.

    python -m loop.core.plan_p2b --root loop/runs/P2B --seed N --n-blocks 12 [--mid-planner MODEL|none]

A thin layer over plan_p2: it swaps plan_p2's cell list and routing_table for the duration of one make_plan call, so
slots, spend ceiling, sentinels, confirm seeds, loop defaults, block shuffling and config writing are P2's own.
Cells, one loop per cell per block (random order inside the block, as in P2):
  SS, ST, QT  the P2 routing tables, unchanged.
  H           the SS table with harm_rate 0.5, in every block.
  F           the QT table, but the planner call that writes the initial backlog (planner_backlog) is sonnet.
  R           a random table: for each routable key (the 5 planner roles, the 3 coder edit classes, the repairer) a
              fair coin picks the SS-cell or the QT-cell model. Coins come from random.Random seeded by (seed, block),
              are stored in the loop config ("r_draws") and in plan.json, and the loop's own routing file
              routing/<loop_id>.json, e.g. R_01 is fixed for the whole loop.
  M           only with --mid-planner MODEL: the QT table with every planner role set to MODEL.
Refuses to run when ROOT/plan.json exists or when the seed equals P2's seed (2).
"""
import argparse
import json
import random
import sys
from contextlib import contextmanager
from pathlib import Path

HERE = Path(__file__).resolve().parent
for _p in (str(HERE.parents[1]), str(HERE)):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import plan_p2 as P  # noqa: E402
from confirmation import atomic_write  # noqa: E402

P2_SEED = 2
CLASSES = ("constant", "schedule", "structural")
R_KEYS = P.PLANNER_ROLES + tuple("coder_by_class." + c for c in CLASSES) + ("repairer",)


def _get(table, key):
    """The model a P2 table gives a routable key (SS has no coder_by_class: every class is its coder)."""
    if key.startswith("coder_by_class."):
        return table.get("coder_by_class", {}).get(key.split(".")[1], table["coder"] if "coder" in table else "sonnet")
    return table[key]


def r_draws(seed, block):
    """{key: "SS" or "QT"}: one fair coin per routable key, from random.Random seeded by (seed, block)."""
    rng = random.Random("p2b:%d:%d" % (seed, block))
    return {k: rng.choice(("SS", "QT")) for k in R_KEYS}


def r_table(draws, structural_model="haiku", repairer_model="haiku"):
    """The R-cell routing table for one set of draws: QT's table with the SS model wherever a coin says SS."""
    src = {"SS": P.routing_table("SS", structural_model, repairer_model),
           "QT": P.routing_table("QT", structural_model, repairer_model)}
    table = json.loads(json.dumps(src["QT"]))
    for key, side in draws.items():
        model = _get(src[side], key)
        if key.startswith("coder_by_class."):
            table["coder_by_class"][key.split(".")[1]] = model
        else:
            table[key] = model
    return table


def cell_table(cell, mid_planner, structural_model="haiku", repairer_model="haiku"):
    """Routing table of a P2B cell (R gets a placeholder here: its real tables are per loop)."""
    if cell in ("SS", "ST", "QT"):
        return P.routing_table(cell, structural_model, repairer_model)
    table = P.routing_table("QT", structural_model, repairer_model)
    if cell == "F":
        table["planner_backlog"] = "sonnet"
    elif cell == "M":
        table.update({r: mid_planner for r in P.PLANNER_ROLES})
    return table


@contextmanager
def p2b_cells(mid_planner):
    """plan_p2 with the P2B cell list and routing tables; restored on exit."""
    old = P.FACTORIAL, P.routing_table
    cells = ("SS", "ST", "QT", "F", "R") + (("M",) if mid_planner else ())
    P.FACTORIAL = cells
    P.routing_table = lambda cell, s="haiku", r="haiku": (
        old[1](cell, s, r) if cell in ("SS", "ST", "QT") else cell_table(cell, mid_planner, s, r))
    try:
        yield
    finally:
        P.FACTORIAL, P.routing_table = old


def make_plan(root, seed, n_blocks=12, mid_planner=None):
    """P2's make_plan with P2B cells, then the per-loop R tables. Returns the plan dict (main writes plan.json)."""
    if seed == P2_SEED:
        P.die("seed %d is P2's seed; P2B needs a different one" % seed)
    root = Path(root).resolve()
    with p2b_cells(mid_planner):
        plan = P.make_plan(root, seed, n_blocks=n_blocks, harm_blocks=n_blocks)
        (root / "routing" / "R.json").unlink()   # the placeholder; every R loop has its own table
        for loop in plan["loops"]:
            if loop["cell"] != "R":
                continue
            draws = r_draws(seed, loop["block"])
            table = r_table(draws)
            P.check_models(table, P.usable_models())
            path = root / "routing" / ("%s.json" % loop["loop_id"])
            atomic_write(path, json.dumps(table, indent=1, sort_keys=True))
            cfg_path = Path(loop["config"])
            cfg = json.loads(cfg_path.read_text())
            cfg.update(routing=str(path), r_draws=draws)
            atomic_write(cfg_path, json.dumps(cfg, indent=1, sort_keys=True))
            loop["r_draws"] = draws
    plan.update(study="P2B", mid_planner=mid_planner)
    return plan


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--root", required=True)
    ap.add_argument("--seed", type=int, required=True)
    ap.add_argument("--n-blocks", type=int, default=12)
    ap.add_argument("--mid-planner", default="none", help="model for the M cell planner roles, or none")
    a = ap.parse_args(argv)
    root = Path(a.root).resolve()
    if (root / "plan.json").exists():
        P.die("%s exists; the plan is frozen, delete it by hand to redo it" % (root / "plan.json"))
    root.mkdir(parents=True, exist_ok=True)
    mid = None if a.mid_planner.lower() == "none" else a.mid_planner
    plan = make_plan(root, a.seed, a.n_blocks, mid)
    atomic_write(root / "plan.json", json.dumps(plan, indent=1))
    for b, ids in enumerate(plan["blocks"], start=1):
        print("block %2d: %s" % (b, " ".join(ids)))
    print("%d loops in %d blocks; wrote %s" % (len(plan["loops"]), len(plan["blocks"]), root / "plan.json"))
    return 0


if __name__ == "__main__":
    sys.exit(main())
