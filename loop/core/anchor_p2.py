#!/usr/bin/env python3
"""Freeze the P2 anchor from the live launch sentinels (loop/design/P2_PREREG.md, Schedule and blocking).

    python -m loop.core.anchor_p2 --root DIR [--min-n 14]

Reads DIR/launch_sentinels.jsonl (written by `controller.py launch-sentinels`), needs at least --min-n runs with status
ok, and writes DIR/anchor.json: anchor_mean = mean of the runs, sd = their sample sd, sigma_used = 1.15 x sd (the R1
rule), gpu_offsets = 0 for every GPU seen (R1's offsets did not hold, so GPU is a blocking factor in the analysis
instead). Then writes sigma_used into DIR/plan.json. Refuses to overwrite an existing anchor.json.

Terms: val_bpb is validation bits per byte after the 300 second training run, lower is better. A launch sentinel is
a run of the unmodified baseline train.py on the live GPU workers just before the loops start. The anchor is the
mean of those runs: the baseline value every loop starts from. sigma_used is the run-to-run noise scale that the
loops' acceptance rule works with. A GPU offset is a per-GPU correction to a run's val_bpb (all zero in P2).

Where it sits: after plan_p2.py and `controller.py launch-sentinels`, before the loops start. Freezing the anchor
and sigma before any loop runs keeps the acceptance rule independent of the results it will judge.
"""
import argparse
import json
import statistics
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]
for _p in (str(REPO), str(HERE)):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from confirmation import atomic_write, read_json, read_jsonl  # noqa: E402

SIGMA_INFLATION = 1.15   # R1: sigma_used = 1.15 x the sd of the live launch sentinels
SIGMA_RULE = "1.15 x sd of the live launch sentinels (R1 rule); the sbatch noise job is not comparable"
OFFSETS_RULE = "zero: R1 offsets did not hold; GPU is a blocking factor in analysis"


def die(msg):
    """Print msg to stderr and exit with status 2 (a usage or precondition error)."""
    print(msg, file=sys.stderr)
    raise SystemExit(2)


def build_anchor(rows, min_n=14):
    """The anchor.json dict from the launch sentinel rows. Only rows with status ok and a val_bpb count; fewer than
    min_n of them is an error, since their sd is frozen as the noise scale of the whole run. The sample sd (n - 1)
    is used, and 'values' keeps every run so the numbers can be recomputed from the file."""
    ok = [r for r in rows if r.get("status") == "ok" and r.get("official_val_bpb") is not None]
    if len(ok) < min_n:
        die("only %d launch sentinels with status ok, need at least %d; rerun launch-sentinels to top up" % (
            len(ok), min_n))
    vals = [float(r["official_val_bpb"]) for r in ok]
    sd = statistics.stdev(vals)
    gpus = sorted({int(r["gpu"]) for r in ok if r.get("gpu") is not None})
    return {"anchor_mean": round(statistics.fmean(vals), 6), "n": len(vals), "sd": round(sd, 6),
            "gpu_offsets": {str(g): 0.0 for g in gpus}, "sigma_used": round(SIGMA_INFLATION * sd, 6),
            "sigma_rule": SIGMA_RULE, "offsets_rule": OFFSETS_RULE,
            "values": [{"gpu": r.get("gpu"), "val": v, "job_id": r.get("job_id"), "seed": r.get("seed")}
                       for r, v in zip(ok, vals)],
            "gpus": gpus, "frozen_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())}


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--root", required=True)
    ap.add_argument("--min-n", type=int, default=14)
    a = ap.parse_args(argv)
    root = Path(a.root).resolve()
    path, plan_path = root / "anchor.json", root / "plan.json"
    if path.exists():
        die("%s exists; the anchor is frozen at launch, delete it by hand to recompute" % path)
    plan = read_json(plan_path)
    if not isinstance(plan, dict):
        die("%s is missing or unreadable: run plan_p2 first" % plan_path)
    anchor = build_anchor(read_jsonl(root / "launch_sentinels.jsonl"), a.min_n)
    atomic_write(path, json.dumps(anchor, indent=1))
    plan["sigma_used"] = anchor["sigma_used"]
    atomic_write(plan_path, json.dumps(plan, indent=1))
    print("anchor mean %.6f  n %d  sd %.6f  sigma_used %.6f  GPUs seen %s (offsets all zero)" % (
        anchor["anchor_mean"], anchor["n"], anchor["sd"], anchor["sigma_used"], anchor["gpus"]))
    print("wrote %s and sigma_used into %s" % (path, plan_path))
    return 0


if __name__ == "__main__":
    sys.exit(main())
