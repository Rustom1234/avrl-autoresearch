"""A second instance of the v2 recipe: same token shares, another seed, written to data/curation/pool_v2b.

    python build_pool_v2b.py [--seed 1]

A new seed gives a new wiki split (target sample, dev and sealed sets) and new pool documents. It is the held-out
pool of the router test R1. build_pool.py names its output after the preset, so the v2 shares are registered under
the name v2b here and pool_v2 is never written.
"""
import argparse, sys
import build_pool

ap = argparse.ArgumentParser()
ap.add_argument("--seed", type=int, default=1)
a = ap.parse_args()
assert a.seed != 0, "seed 0 is pool v2 itself"
build_pool.PRESETS["v2b"] = build_pool.PRESETS["v2"]
sys.argv = [sys.argv[0], "--preset", "v2b", "--seed", str(a.seed)]
build_pool.main()
