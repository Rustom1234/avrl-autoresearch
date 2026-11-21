#!/usr/bin/env python3
"""Cell N of P2B: a loop with no language model (random search over eight settings of train.py).

P2B_PREREG.md, "No-LLM baseline": each slot draws uniformly one of eight settings and a direction, applied to the
current incumbent: times 0.5 or times 2 for TOTAL_BATCH_SIZE, EMBEDDING_LR, UNEMBEDDING_LR, MATRIX_LR, SCALAR_LR and
WEIGHT_DECAY, minus 2 or plus 2 for DEPTH, minus 0.2 or plus 0.2 for WARMDOWN_RATIO (clipped to 0.1 and 0.9). A move
already rejected on the same incumbent is redrawn. Everything after the proposal is the engine's own code (arms.py,
arm D): the GPU queue, the z rule with replication, the tripwire audit, integrity flags, the ledger. No model is
called, so model cost is zero: there is no planner, coder, repairer, critic or analyst.

Choices the preregistration leaves open (also listed in loop/design/P2B_CELL_N_NOTES.md):
- A move that cannot change the script (WARMDOWN_RATIO already at its clip, DEPTH below 2, a batch smaller than one
  sequence) is redrawn and never reaches a GPU.
- When TOTAL_BATCH_SIZE falls below DEVICE_BATCH_SIZE * MAX_SEQ_LEN, DEVICE_BATCH_SIZE is lowered with it, because the
  script asserts that the total is a multiple of the per-device batch. It is never raised above the base value (64).
- When every possible move has been rejected on the current incumbent, the rejected list of that incumbent is cleared.
- A candidate that crashes is rejected and costs its slot; there is no crash-fix slot, since that needs a model.
- No critic review (borderline accepts, audits, periodic reviews): those are model calls.

Commands (run from the repository root):
  python -m loop.core.random_search plan --root loop/runs/P2B_N --anchor loop/runs/P2B/anchor.json --seed 3
  python -m loop.core.random_search controller run --plan loop/runs/P2B_N/plan.json --max-inflight 8 --window-hours 20
  python -m loop.core.random_search loop --config loop/runs/P2B_N/N_01/config.json [--resume]
"""
import argparse
import json
import re
import shutil
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))   # sibling modules import by bare name, as in run_loop.py

import run_loop
from arms import LoopEngine
from linter import lint

SCALED = ("TOTAL_BATCH_SIZE", "EMBEDDING_LR", "UNEMBEDDING_LR", "MATRIX_LR", "SCALAR_LR", "WEIGHT_DECAY")
SETTINGS = SCALED + ("DEPTH", "WARMDOWN_RATIO")
MOVES = tuple((name, d) for name in SETTINGS for d in ("down", "up"))
MAX_SEQ_LEN = 2048          # prepare_pinned.py; train.py asserts TOTAL_BATCH_SIZE % (DEVICE_BATCH_SIZE * MAX_SEQ_LEN) == 0
BASE_DEVICE_BATCH = 64      # the base value; 128 does not fit an A100 40GB
N_LOOPS, N_SLOTS = 12, 20


def _line(name):
    """Regex for the assignment line of one module-level setting: (prefix)(value)(trailing spaces and comment)."""
    return re.compile(r"^(%s\s*=\s*)([^#\n]+?)([ \t]*(?:#.*)?)$" % name, re.M)


def read_value(src, name):
    """Current value of a setting. The right-hand side is a number or simple arithmetic such as 2**19."""
    m = _line(name).search(src)
    if not m:
        raise ValueError("setting %s not found in train.py" % name)
    return eval(m.group(2), {"__builtins__": {}})   # our own incumbent file, numbers only


def _set(src, name, value):
    """Rewrite the value of a setting, keeping the rest of its line."""
    return _line(name).sub(lambda m: m.group(1) + repr(value) + m.group(3), src, count=1)


def apply_move(src, name, direction):
    """Apply one move to a train.py source. Returns (new_source, old_value, new_value), or None when the move cannot
    change the script (clipped, or out of range)."""
    old = read_value(src, name)
    up = direction == "up"
    if name == "TOTAL_BATCH_SIZE":
        new = int(old * 2) if up else int(old // 2)
        if new < MAX_SEQ_LEN:
            return None
    elif name in SCALED:
        new = float("%.6g" % (old * (2.0 if up else 0.5)))
    elif name == "DEPTH":
        new = int(old) + (2 if up else -2)
        if new < 2:
            return None
    else:   # WARMDOWN_RATIO
        new = round(min(0.9, max(0.1, old + (0.2 if up else -0.2))), 3)
    if new == old:
        return None
    out = _set(src, name, new)
    if name == "TOTAL_BATCH_SIZE":
        dev = min(BASE_DEVICE_BATCH, new // MAX_SEQ_LEN)
        if dev != read_value(out, "DEVICE_BATCH_SIZE"):
            out = _set(out, "DEVICE_BATCH_SIZE", dev)
    return out, old, new


class RandomSearchEngine(LoopEngine):
    """LoopEngine whose candidates come from random moves and not from a model. Only the proposal step and the model
    hooks are replaced; the slot loop, the GPU runs and the decision are inherited unchanged."""

    def _call(self, role, user, **kw):
        raise RuntimeError("cell N makes no model calls (asked for role %s)" % role)

    def _health_check(self):
        self.override = {}

    def _critic(self, new_src, numbers, reason):
        return None

    def _memo(self):
        return None

    def _finish(self, cand, accepted):
        """Remember a rejected move against the incumbent it was tried on, then the engine's own bookkeeping."""
        if not accepted and cand.get("move"):
            tried = self.state.setdefault("n_rejected", {}).setdefault(self.state["incumbent"]["sha256"], [])
            if cand["move"] not in tried:
                tried.append(cand["move"])
        super()._finish(cand, accepted)

    def _attempt(self):
        """One random move on the incumbent, run on a GPU. Returns "ran", or "skipped" / "charged" if the linter
        refuses the move (not expected: the moves only change constants)."""
        s = self.state
        inc_src = self._inc_src()
        tried = s.setdefault("n_rejected", {}).setdefault(s["incumbent"]["sha256"], [])
        options = [(m, r) for m, r in ((m, apply_move(inc_src, *m)) for m in MOVES) if r is not None]
        fresh = [(m, r) for m, r in options if "%s:%s" % m not in tried]
        if not fresh:   # every move was rejected on this incumbent: start over
            del tried[:]
            fresh = options
        (name, direction), (new, old, val) = self.rng.choice(fresh)
        move = "%s:%s" % (name, direction)
        ctx = {"idea_id": "n%d.%d" % (s["slot_index"], s["attempt"]), "item_id": None, "edit_class": None,
               "description": "random search: %s %s, %r -> %r" % (name, direction, old, val)}
        lr = lint(new, inc_src)
        lint_rec = {"ok": lr.ok, "hard": lr.hard, "warnings": lr.warnings}
        ctx["edit_class"] = lr.edit_class
        if not lr.ok:
            tried.append(move)
            return self._abandon("lint_fail", "lint", ctx, free=True, lint=lint_rec)
        path = self._stage(new)
        (self._slot_dir() / "lint.json").write_text(json.dumps(lint_rec))
        # is_fix=True: a crash ends the experiment (the engine grants a crash-fix slot only to a first candidate, and
        # that slot is a repairer call).
        s["cand"] = {"idea_id": ctx["idea_id"], "description": ctx["description"], "edit_class": ctx["edit_class"],
                     "item_id": None, "train_py": path, "lint": lint_rec, "critic": None, "runs": [], "next": None,
                     "is_fix": True, "z1": None, "move": move}
        res = self._run_gpu("candidate", path, self.cfg["candidate_seed"], False)
        self._on_result("candidate", res)
        return "ran"


def build_services(cfg):
    """The real GPU queue and clock, and no model service at all."""
    from services import Clock, GPUService, Services
    return Services(None, GPUService(job_timeout_s=cfg.get("job_timeout_s", 600), queue_kwargs=cfg.get("queue")),
                    Clock())


def cmd_loop(a, services=None):
    """Run one cell N loop; same contract and exit codes as run_loop.py."""
    cfg = run_loop.load_config(a.config)
    state_file = Path(cfg["run_dir"]) / "state.json"
    if state_file.exists() and not a.resume:
        print("%s exists; pass --resume to continue that loop" % state_file, file=sys.stderr)
        return 2
    engine = RandomSearchEngine(cfg, services or build_services(cfg))
    status = engine.run()
    s = engine.state
    print("%s: %s after %d/%d slots, claimed best official %s" % (
        cfg["loop_id"], status, s["slot_index"], s["n_slots"], s["claimed_best_official"]))
    return 0 if status == "done" or engine.stop_reason == "stop_file" else 3


def cmd_plan(a):
    """Write <root>/plan.json, one config per loop and a copy of the study's anchor.json. Refuses to overwrite."""
    root = Path(a.root).resolve()
    if (root / "plan.json").exists():
        raise SystemExit("%s exists; the plan is frozen" % (root / "plan.json"))
    anchor = Path(a.anchor).resolve()
    json.loads(anchor.read_text())["anchor_mean"]   # must be a frozen anchor file
    root.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(anchor, root / "anchor.json")
    loops = []
    for b in range(1, N_LOOPS + 1):
        lid = "N_%02d" % b
        run_dir = root / lid
        run_dir.mkdir(exist_ok=True)
        cfg = {"loop_id": lid, "arm": "D", "cell": "N", "block": b, "rng_seed": a.seed * 1000 + 100 + b,
               "run_dir": str(run_dir), "stop_file": str(run_dir / "STOP_AT"), "anchor_file": str(root / "anchor.json"),
               "routing": None, "n_slots": N_SLOTS, "job_timeout_s": 900, "timeout_s": 7200}
        (run_dir / "config.json").write_text(json.dumps(cfg, indent=1, sort_keys=True))
        loops.append({"loop_id": lid, "arm": "D", "cell": "N", "block": b, "index": b - 1, "run_dir": str(run_dir),
                      "config": str(run_dir / "config.json"), "stop_file": cfg["stop_file"]})
    plan = {"seed": a.seed, "root": str(root), "n_slots": N_SLOTS, "spend_ceiling_tokens": 0,
            "blocks": [[l["loop_id"]] for l in loops], "loops": loops, "anchor_file": str(root / "anchor.json"),
            "sentinel_every": 45, "confirm_seeds": [1001, 1002], "study": "P2B", "cell": "N",
            "anchor_source": str(anchor)}
    (root / "plan.json").write_text(json.dumps(plan, indent=1))
    print("wrote %s (%d loops)" % (root / "plan.json", len(loops)))
    return 0


def cmd_controller(rest):
    """The study's controller (run, confirm, status), with loops started through this file instead of run_loop.py."""
    import controller
    controller.LOOP_CMD = [sys.executable, "-m", "loop.core.random_search", "loop"]
    return controller.main(rest)


def main(argv=None):
    argv = list(sys.argv[1:] if argv is None else argv)
    if argv[:1] == ["controller"]:
        return cmd_controller(argv[1:])
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    sub = ap.add_subparsers(dest="cmd", required=True)
    p = sub.add_parser("loop")
    p.add_argument("--config", required=True)
    p.add_argument("--resume", action="store_true")
    p = sub.add_parser("plan")
    p.add_argument("--root", required=True)
    p.add_argument("--anchor", required=True)
    p.add_argument("--seed", type=int, default=3)
    a = ap.parse_args(argv)
    return cmd_loop(a) if a.cmd == "loop" else cmd_plan(a)


if __name__ == "__main__":
    sys.exit(main())
