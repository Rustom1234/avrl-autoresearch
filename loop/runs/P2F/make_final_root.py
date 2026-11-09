"""Build the final P2 root for confirmation and analysis (P2 deviation 1).

loop/runs/P2F holds one symlink per loop: the 32 loops of loop/runs/P2 with no refused Claude call, and the 28 reruns
in loop/runs/P2R. plan.json lists all 60 in the original block order with run_dir set to the real loop directory.
anchor.json is the frozen P2 anchor; launch_sentinels.jsonl is P2's; sentinels.jsonl concatenates the in-run sentinels
of P2 and P2R (field source added). Refuses to overwrite.
"""
import json
import shutil
from pathlib import Path

REPO = Path(__file__).resolve().parents[3]
P2, P2R, P2F = REPO / "loop/runs/P2", REPO / "loop/runs/P2R", REPO / "loop/runs/P2F"


def jl(path):
    return [json.loads(l) for l in open(path)] if path.exists() else []


def main():
    if (P2F / "plan.json").exists():
        raise SystemExit("refusing to overwrite loop/runs/P2F/plan.json")
    plan, rerun = json.loads((P2 / "plan.json").read_text()), json.loads((P2R / "plan.json").read_text())
    rerun_ids = {l["loop_id"] for l in rerun["loops"]}
    states = {}
    loops = []
    for l in plan["loops"]:
        src = (P2R if l["loop_id"] in rerun_ids else P2) / l["loop_id"]
        st = json.loads((src / "state.json").read_text())
        if st.get("status") != "done":
            raise SystemExit("%s is not done (%s)" % (src, st.get("status")))
        states[l["loop_id"]] = str(src)
        (P2F / l["loop_id"]).symlink_to(src)
        loops.append(dict(l, run_dir=str(src), config=str(src / "config.json"), stop_file=str(src / "STOP_AT"),
                          source="P2R" if l["loop_id"] in rerun_ids else "P2"))
    shutil.copy(P2 / "anchor.json", P2F / "anchor.json")
    shutil.copy(P2 / "launch_sentinels.jsonl", P2F / "launch_sentinels.jsonl")
    with open(P2F / "sentinels.jsonl", "w") as f:
        for name, root in (("P2", P2), ("P2R", P2R)):
            for r in jl(root / "sentinels.jsonl"):
                f.write(json.dumps(dict(r, source=name)) + "\n")
    new = dict(plan, root=str(P2F), anchor_file=str(P2F / "anchor.json"), loops=loops,
               combined_from={"P2": str(P2), "P2R": str(P2R)}, rerun_ids=sorted(rerun_ids))
    (P2F / "plan.json").write_text(json.dumps(new, indent=1))
    print("%d loops (%d reruns) linked into %s" % (len(loops), len(rerun_ids), P2F))


if __name__ == "__main__":
    main()
