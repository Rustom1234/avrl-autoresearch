#!/usr/bin/env python3
"""Run the Step 6 loops: the launcher of C1 and R1 (launch_c1.py, unchanged) starting routed_s6.py.

    python curation/loop/launch_s6.py --root curation/runs/S6 --hosts local willow --per-host 3

The plan (curation/runs/S6/plan.json) is written by hand and frozen in S6_PREREG.md. A loop of the plan may name a
"memory" file, which is passed on to routed_s6.py.
"""
import argparse, sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import launch_c1 as L   # noqa: E402

_command = L.command


def command(plan, lp, root, host):
    cmd = _command(plan, lp, root, host)
    cmd[1] = str(HERE / "routed_s6.py")
    if lp.get("memory"):
        cmd[2:2] = ["--memory", lp["memory"]]
    return cmd


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--root", required=True)
    ap.add_argument("--hosts", nargs="+", default=["local"], help='"local" is this machine')
    ap.add_argument("--per-host", type=int, default=2)
    ap.add_argument("--blocks", type=int, nargs="+", default=None)
    L.command = command
    return L.run(ap.parse_args(argv))


if __name__ == "__main__":
    sys.exit(main())
