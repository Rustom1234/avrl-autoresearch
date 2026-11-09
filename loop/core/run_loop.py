#!/usr/bin/env python3
"""Run one loop: python run_loop.py --config config.json [--resume]

A loop is one automated research run of one arm (routing and verification configuration A to D, see arms.py): it
spends n_slots GPU training runs (slots) trying to lower val_bpb (validation bits per byte) of a small GPT. This file
is only the command line around the engine. The controller (controller.py) starts one such process per loop.

Builds the real services (loop/models call layer, QueueClient to hannah, the GPU cluster) and runs a LoopEngine until
n_slots are used, the stop file says stop, or an infra failure outlasts its retries. Exit code 0 when the loop is done
or was stopped by its stop file (the controller's symmetric stop), 3 when infra outlasted the retries (the controller
resumes it), 2 for a usage or config error. Config fields are listed in ENGINE.md.
"""
import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))   # sibling modules (arms, services) import by bare name

# Config keys the engine cannot run without. The anchor is the baseline's mean val_bpb and run count, gpu_offsets the
# frozen per-GPU corrections, sigma_used the run-to-run noise sd used by the acceptance rules.
REQUIRED = ("loop_id", "arm", "run_dir", "n_slots", "rng_seed", "sigma_used", "gpu_offsets", "anchor")
DEFAULT_BASE = Path(__file__).resolve().parents[1] / "substrate" / "base" / "train_ampere.py"


def load_config(path):
    """Read the config and fill in what the controller writes in another shape: `anchor_file` (the controller's
    anchor.json: anchor_mean, n, gpu_offsets, sigma_used) stands in for anchor, gpu_offsets and sigma_used, and
    `routing` may be the path of routing_table.json (null for arms A to C)."""
    cfg = json.loads(Path(path).read_text())
    if cfg.get("anchor_file") and "anchor" not in cfg:
        a = json.loads(Path(cfg["anchor_file"]).read_text())
        cfg["anchor"] = {"mean": a["anchor_mean"], "n": a["n"]}
        cfg.setdefault("gpu_offsets", a["gpu_offsets"])
        cfg.setdefault("sigma_used", a["sigma_used"])
    if isinstance(cfg.get("routing"), str):
        cfg["routing"] = json.loads(Path(cfg["routing"]).read_text())
    cfg["routing"] = cfg.get("routing") or {}
    cfg.setdefault("base_train_py", str(DEFAULT_BASE))
    missing = [k for k in REQUIRED if k not in cfg]
    if missing:
        raise SystemExit("config %s lacks: %s" % (path, ", ".join(missing)))
    return cfg


def build_services(cfg):
    """The real services: model calls logged under run_dir, the GPU queue client and the wall clock."""
    from services import Clock, GPUService, ModelService, Services
    return Services(ModelService(cfg["run_dir"], cfg["loop_id"]),
                    GPUService(job_timeout_s=cfg.get("job_timeout_s", 600), queue_kwargs=cfg.get("queue")), Clock())


def main(argv=None, services=None):
    """Parse arguments, build the engine and run it. services: optional replacement for the real services (tests
    pass fakes). Returns the exit code described in the module docstring."""
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--config", required=True)
    ap.add_argument("--resume", action="store_true", help="continue from <run_dir>/state.json")
    a = ap.parse_args(argv)
    cfg = load_config(a.config)
    state_file = Path(cfg["run_dir"]) / "state.json"
    # An existing state.json means this loop already started. Requiring --resume keeps a rerun from being mistaken
    # for a fresh loop; the engine itself always continues from state.json when it is there.
    if state_file.exists() and not a.resume:
        print("%s exists; pass --resume to continue that loop" % state_file, file=sys.stderr)
        return 2
    if a.resume and not state_file.exists():
        print("no state.json under %s, starting fresh" % cfg["run_dir"])
    from arms import LoopEngine
    engine = LoopEngine(cfg, services or build_services(cfg))
    status = engine.run()
    s = engine.state
    print("%s: %s after %d/%d slots, claimed best official %s" % (
        cfg["loop_id"], status, s["slot_index"], s["n_slots"], s["claimed_best_official"]))
    # A stop requested through the stop file is a normal ending; any other stop is infra trouble the controller retries.
    return 0 if status == "done" or engine.stop_reason == "stop_file" else 3


if __name__ == "__main__":
    sys.exit(main())
