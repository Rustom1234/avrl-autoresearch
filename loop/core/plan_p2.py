#!/usr/bin/env python3
"""P2 plan builder (loop/design/P2_PREREG.md): the planner x coder factorial plus the harm control.

    python -m loop.core.plan_p2 --root DIR --seed 2 [--structural-model haiku] [--repairer-model haiku]
                                [--n-blocks 12] [--harm-blocks 6]

Writes DIR/routing/{SS,ST,QS,QT}.json (frozen routing tables), DIR/<loop_id>/config.json for every loop and, last,
DIR/plan.json. Refuses to run when DIR/plan.json exists. Every loop runs the arm D engine; only its routing table
(and, for H, harm_rate) differs. Cell letters: first = planner (S Sonnet, Q qwen35-4b), second = coder and repairer
(S Sonnet, T the per-class table with a Haiku-or-other repairer). H is the harm control: the SS table, harm_rate 0.5.
The controller and confirmation read plan.json unchanged (plus its sentinel_every and confirm_seeds keys).

Where it sits: the first step of P2, run once before launch. Next come the launch sentinels and anchor_p2.py (which
fills in sigma_used), then the controller, confirmation.py and analysis_p2.py.

Terms: a loop is one automated research loop of N_SLOTS slots (a slot is one 300 second GPU training run, scored by
val_bpb, validation bits per byte, lower is better). A routing table maps each LLM role of the loop (planner, coder,
repairer, voter, analyst, critic) to the model that fills it. A cell is one routing configuration. A block is one
replicate of the design: every factorial cell once, in random order. Loops launch in block order (P2_PREREG), so
drift over the run (GPU load, time of day) is spread evenly over the cells instead of landing on one of them.
"""
import argparse
import json
import random
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]
for _p in (str(REPO), str(HERE)):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from confirmation import atomic_write  # noqa: E402

MODELS_JSON = REPO / "loop" / "models" / "models.json"   # the model registry: every name a table may use
FACTORIAL = ("SS", "ST", "QS", "QT")   # the 2 x 2 cells; H is added separately in block_layout
# the planner keys of a routing table; all are set explicitly, see routing_table
PLANNER_ROLES = ("planner_backlog", "planner_widen", "planner_refresh", "planner", "planner_cheap")
PLANNER_MODEL = {"S": "sonnet", "Q": "qwen35-4b"}
N_SLOTS = 20   # GPU training runs per loop
SPEND_CEILING_TOKENS = 60_000_000   # per-loop token ceiling, copied into every loop config
HARM_RATE = 0.5   # in H loops, the probability that a coder edit is replaced by a no-op (arms.py, _harm)
SENTINEL_EVERY = 45   # the controller submits one baseline (sentinel) run per this many loop slots
CONFIRM_SEEDS = [1001, 1002]   # confirmation reruns every loop's final train.py once per seed (confirmation.py)
# Shared by every loop config: all loops run the arm D engine, so cells differ only in routing table (and harm_rate).
LOOP_DEFAULTS = {"arm": "D", "n_slots": N_SLOTS, "job_timeout_s": 900, "timeout_s": 7200,
                 "spend_ceiling_tokens": SPEND_CEILING_TOKENS, "critic_model": "opus", "voter_model": "gemma4-12b",
                 "vote_k": 3, "free_action_caps": {"slot": 2, "run": 40}}


def die(msg):
    """Print msg to stderr and exit with status 2 (a usage or precondition error)."""
    print(msg, file=sys.stderr)
    raise SystemExit(2)


def routing_table(cell, structural_model="haiku", repairer_model="haiku"):
    """The frozen routing table of one factorial cell. Only model names as string values: the engine's per-slot health
    check looks up every string value in the model registry. Planner roles are listed one by one because the engine
    sends planner_backlog and planner_widen to Sonnet unless the table names them."""
    planner, coder = cell
    table = {r: PLANNER_MODEL[planner] for r in PLANNER_ROLES}
    if coder == "S":
        table.update(coder="sonnet", repairer="sonnet")
    else:
        table.update(coder_by_class={"constant": "qwen35-4b", "schedule": "gemma4-12b", "structural": structural_model},
                     repairer=repairer_model)
    table.update(voter="gemma4-12b", analyst="sonnet", critic="opus")
    return table


def usable_models():
    """Names of the registry models that are not disabled."""
    return {m["name"] for m in json.loads(MODELS_JSON.read_text()) if not m.get("disabled")}


def check_models(table, known):
    """Exit unless every model named in a routing table (top level and coder_by_class) is a usable registry model.
    Checked here so that a typo fails before launch, not at a loop's first health check."""
    names = {v for v in list(table.values()) + list(table.get("coder_by_class", {}).values()) if isinstance(v, str)}
    bad = sorted(names - known)
    if bad:
        die("model(s) %s not in %s (or disabled); known: %s" % (", ".join(bad), MODELS_JSON, ", ".join(sorted(known))))


def block_layout(seed, n_blocks=12, harm_blocks=6):
    """[[cell, ...] per block]: SS, ST, QS and QT once per block in random order, plus H at a random position in
    harm_blocks randomly chosen blocks. Deterministic in seed."""
    rng = random.Random(seed)   # one generator, so the whole layout follows from the plan seed
    with_h = set(rng.sample(range(n_blocks), harm_blocks))
    blocks = []
    for b in range(n_blocks):
        cells = list(FACTORIAL)
        rng.shuffle(cells)
        if b in with_h:
            cells.insert(rng.randrange(len(cells) + 1), "H")
        blocks.append(cells)
    return blocks


def make_plan(root, seed, structural_model="haiku", repairer_model="haiku", n_blocks=12, harm_blocks=6):
    """Writes the routing tables and every loop's config.json, returns the plan dict (plan.json is written by main)."""
    if not 0 <= harm_blocks <= n_blocks:
        die("--harm-blocks must be between 0 and --n-blocks")
    root = Path(root).resolve()
    known = usable_models()
    tables, paths = {}, {}
    for cell in FACTORIAL:
        tables[cell] = routing_table(cell, structural_model, repairer_model)
        check_models(tables[cell], known)
        paths[cell] = root / "routing" / ("%s.json" % cell)
    paths["H"] = paths["SS"]   # the harm control runs the SS table
    (root / "routing").mkdir(parents=True, exist_ok=True)
    for cell in FACTORIAL:
        atomic_write(paths[cell], json.dumps(tables[cell], indent=1, sort_keys=True))
    # loop ids count each cell in launch order (SS_01, SS_02, ...); rng_seed is distinct per loop and fixed by the seed
    loops, blocks, count = [], [], {}
    for b, cells in enumerate(block_layout(seed, n_blocks, harm_blocks), start=1):
        ids = []
        for cell in cells:
            count[cell] = count.get(cell, 0) + 1
            lid = "%s_%02d" % (cell, count[cell])
            run_dir = root / lid
            cfg = dict(LOOP_DEFAULTS, loop_id=lid, cell=cell, block=b, rng_seed=seed * 1000 + len(loops),
                       harm_rate=HARM_RATE if cell == "H" else 0.0, routing=str(paths[cell]), run_dir=str(run_dir),
                       stop_file=str(run_dir / "STOP_AT"), anchor_file=str(root / "anchor.json"))
            run_dir.mkdir(parents=True, exist_ok=True)
            atomic_write(run_dir / "config.json", json.dumps(cfg, indent=1, sort_keys=True))
            loops.append({"loop_id": lid, "arm": "D", "cell": cell, "block": b, "index": len(loops),
                          "run_dir": str(run_dir), "config": str(run_dir / "config.json"),
                          "stop_file": cfg["stop_file"]})
            ids.append(lid)
        blocks.append(ids)
    # sigma_used stays None here: anchor_p2.py fills it in once the launch sentinels have been run
    return {"seed": seed, "root": str(root), "n_slots": N_SLOTS, "spend_ceiling_tokens": SPEND_CEILING_TOKENS,
            "blocks": blocks, "loops": loops, "anchor_file": str(root / "anchor.json"), "sigma_used": None,
            "sentinel_every": SENTINEL_EVERY, "confirm_seeds": CONFIRM_SEEDS}


def main(argv=None):
    """Command line entry point. plan.json is written last and never overwritten: once it exists the layout,
    seeds and routing tables of the run are frozen."""
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--root", required=True)
    ap.add_argument("--seed", type=int, required=True)
    ap.add_argument("--structural-model", default="haiku", help="coder for structural edits in the T cells")
    ap.add_argument("--repairer-model", default="haiku", help="repairer in the T cells")
    ap.add_argument("--n-blocks", type=int, default=12)
    ap.add_argument("--harm-blocks", type=int, default=6)
    a = ap.parse_args(argv)
    root = Path(a.root).resolve()
    if (root / "plan.json").exists():
        die("%s exists; the plan is frozen, delete it by hand to redo it" % (root / "plan.json"))
    root.mkdir(parents=True, exist_ok=True)
    plan = make_plan(root, a.seed, a.structural_model, a.repairer_model, a.n_blocks, a.harm_blocks)
    atomic_write(root / "plan.json", json.dumps(plan, indent=1))
    for b, ids in enumerate(plan["blocks"], start=1):
        print("block %2d: %s" % (b, " ".join(ids)))
    print("%d loops in %d blocks; wrote %s" % (len(plan["loops"]), len(plan["blocks"]), root / "plan.json"))
    return 0


if __name__ == "__main__":
    sys.exit(main())
