#!/usr/bin/env python3
"""E1: continue every model loop of a finished bench run for more slots.

    python loop/bench/make_ext.py --src loop/runs/BG1_R --root loop/runs/E1 --slots 80

Each SS and HH loop directory of the source run is copied (ledger, backlog, memo, incumbent: the whole state), its
slot budget is raised and its state set back to running, so the frozen launcher resumes it as if the loop had been
planned with that many slots from the start. Paths inside config and state are pointed at the copy. Loops are
assigned to the lab machines in turn (one resumed loop at a time per machine).
"""
import argparse, json, shutil
from pathlib import Path

HOSTS = ["willow", "walnut", "pine", "cherry", "spruce"]
ap = argparse.ArgumentParser()
ap.add_argument("--src", required=True)
ap.add_argument("--root", required=True)
ap.add_argument("--slots", type=int, default=80)
a = ap.parse_args()
src, root = Path(a.src).resolve(), Path(a.root).resolve()
assert not (root / "plan.json").exists(), "plan.json exists"
root.mkdir(parents=True, exist_ok=True)
plan = json.loads((src / "plan.json").read_text())
loops = [lp for lp in plan["loops"] if lp["cmd"] == "loop"]
for i, lp in enumerate(loops):
    lid = lp["loop_id"]
    s, d = (src / lid).resolve(), root / lid
    shutil.copytree(s, d, ignore=shutil.ignore_patterns("done", "__pycache__", "jobs"))
    (d / "jobs").mkdir()
    lp["config"]["n_slots"] = a.slots
    cfg = json.loads((d / "config.json").read_text())
    cfg.update(n_slots=a.slots, host=HOSTS[i % len(HOSTS)], run_dir=str(d))
    (d / "config.json").write_text(json.dumps(cfg, indent=1))
    st = json.loads((d / "state.json").read_text())
    assert st["status"] == "done" and st["slot_index"] == st["n_slots"] and st["in_flight_job"] is None, lid
    st.update(n_slots=a.slots, status="running")
    st["incumbent"]["path"] = st["incumbent"]["path"].replace(str(s), str(d))
    assert Path(st["incumbent"]["path"]).exists(), st["incumbent"]["path"]
    (d / "state.json").write_text(json.dumps(st, indent=1))
(root / "plan.json").write_text(json.dumps({"hosts": HOSTS, "seed": plan["seed"], "loops": loops}, indent=1))
print("wrote", root, "with", len(loops), "loops to continue to", a.slots, "slots")
