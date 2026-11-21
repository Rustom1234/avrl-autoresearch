#!/usr/bin/env python3
"""Score the final file of every finished loop of a study on extra seeds (sensitivity check of R1_PREREG.md).

    python curation/loop/rescore_finals.py --root curation/runs/R1 --pool v2b --seeds 1003 1004

Runs curation/task/run_recipes.py (same GPU locks as the loops). Results: curation/results/<pool>/<prefix><loop>_s<seed>.json.
"""
import argparse, json, subprocess, sys
from pathlib import Path

HERE = Path(__file__).resolve().parent


def jobs(root, prefix):
    return ["%s%s=%s" % (prefix, f.parent.name, json.loads(f.read_text())["incumbent"]) for f in sorted(Path(root).glob("*/final.json"))]


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--root", required=True)
    ap.add_argument("--pool", required=True)
    ap.add_argument("--seeds", nargs="+", type=int, default=[1003, 1004])
    ap.add_argument("--hosts", nargs="+", default=["local", "willow", "cherry"])
    ap.add_argument("--prefix", default="R1final_")
    a = ap.parse_args(argv)
    return subprocess.call([sys.executable, str(HERE.parent / "task/run_recipes.py"), "--pool", a.pool, "--hosts", *a.hosts,
                            "--seeds", *map(str, a.seeds), "--jobs", *jobs(a.root, a.prefix)])


if __name__ == "__main__":
    sys.exit(main())
