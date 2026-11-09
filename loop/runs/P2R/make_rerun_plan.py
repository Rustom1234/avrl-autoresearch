"""Build the P2 rerun root (P2 deviation 1, 2026-11-07).

From 14:19 to about 16:10 UTC on 2026-11-07 every Claude call failed with HTTP 403 (an account-level refusal by the
model provider). The engine counted these as model failures, so 28 loops burned slots on failed calls. This script
copies those 28 loops' configs unchanged (same rng_seed, routing table, harm_rate, anchor values) into loop/runs/P2R so
they can be rerun from scratch. The 32 loops with no refused call stay in loop/runs/P2 untouched. A loop is affected
if any Claude call in its calls.jsonl returned 403.
"""
import json
import shutil
from pathlib import Path

REPO = Path("/homes/rdubash/active-audit")
P2, P2R = REPO / "loop/runs/P2", REPO / "loop/runs/P2R"


def affected():
    out = []
    for d in sorted(p for p in P2.iterdir() if (p / "config.json").exists()):
        calls = [json.loads(l) for l in open(d / "calls.jsonl")] if (d / "calls.jsonl").exists() else []
        if any(c.get("provider") == "claude" and not c.get("ok") and "403" in (c.get("error") or "") for c in calls):
            out.append(d.name)
    return out


def main():
    if (P2R / "plan.json").exists():
        raise SystemExit("refusing to overwrite loop/runs/P2R/plan.json")
    plan = json.loads((P2 / "plan.json").read_text())
    ids = set(affected())
    shutil.copy(P2 / "anchor.json", P2R / "anchor.json")
    loops = []
    for l in plan["loops"]:
        if l["loop_id"] not in ids:
            continue
        cfg = json.loads(Path(l["config"]).read_text())
        run_dir = P2R / l["loop_id"]
        run_dir.mkdir()
        cfg.update(run_dir=str(run_dir), stop_file=str(run_dir / "STOP_AT"), anchor_file=str(P2R / "anchor.json"))
        (run_dir / "config.json").write_text(json.dumps(cfg, indent=1, sort_keys=True))
        loops.append(dict(l, run_dir=str(run_dir), config=str(run_dir / "config.json"), stop_file=str(run_dir / "STOP_AT")))
    new = {k: v for k, v in plan.items() if k not in ("loops", "blocks", "root", "anchor_file")}
    new.update(root=str(P2R), anchor_file=str(P2R / "anchor.json"), loops=loops,
               blocks=[[i for i in b if i in ids] for b in plan["blocks"]], rerun_of=str(P2),
               rerun_reason="P2 deviation 1: model provider refused every call (HTTP 403) 2026-11-07 14:19 to about 16:10 UTC")
    (P2R / "plan.json").write_text(json.dumps(new, indent=1))
    print("%d loops: %s" % (len(loops), " ".join(l["loop_id"] for l in loops)))


if __name__ == "__main__":
    main()
