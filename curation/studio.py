#!/usr/bin/env python3
"""Curation studio (demo): for a new pool, check the memory of past strong loops before paying for any model.

    curation/.venv/bin/python curation/studio.py --pool v3 --hosts local willow

1. Library: the final curate.py of every finished loop in the given cells (default: the strong-planner cells of C1).
2. Screen: run every library file on the pool with one seed and rank by dev bpb (the sealed score is not used here).
3. Confirm: run the best file on a second seed and compare its sealed bpb with the raw pool on the same seeds.
4. Route: if the best file beats the raw pool by more than --min-gain on both seeds, reuse it (0 USD of model calls);
   otherwise the pool needs a planning loop, and S6/R1 say which: strong planner with the cheap coder.
Training runs go through run_recipes.py (one run per GPU, memory-capped runner). Start mem_guard.py on each host first.
"""
import argparse, json, subprocess, sys, time
from pathlib import Path

HERE = Path(__file__).resolve().parent
PY = str(HERE / ".venv/bin/python")
SC_USD = 2.69            # R1: model dollars of one loop with the strong planner and the cheap coder


def library(roots, cells):
    """{name: final file} for every finished loop of the given cells."""
    lib = {}
    for root in roots:
        for f in sorted(Path(root).glob("*/final.json")):
            d = json.loads(f.read_text())
            if d["cell"] in cells and d.get("incumbent"):
                lib["LIB_%s_%s" % (Path(root).name, d["loop_id"])] = d["incumbent"]
    return lib


def run(pool, hosts, seed, jobs):
    """Run jobs ({name: file}, or "raw") on one seed and return {name: result}. Finished results are reused."""
    res_dir = HERE / "results" / pool
    args = ["raw" if f == "raw" else "%s=%s" % (n, f) for n, f in jobs.items()]
    todo = [a for n, a in zip(jobs, args) if not (res_dir / ("%s_s%d.json" % (n, seed))).exists()]
    if todo:
        subprocess.run([PY, str(HERE / "task/run_recipes.py"), "--pool", pool, "--hosts", *hosts, "--seeds", str(seed),
                        "--jobs", *todo], stdout=subprocess.DEVNULL, check=False)
    out = {}
    for n in jobs:
        f = res_dir / ("%s_s%d.json" % (n, seed))
        for _ in range(12):                      # a file written on another host can take a while to show up on NFS
            if f.exists():
                break
            time.sleep(5)
        out[n] = json.loads(f.read_text()) if f.exists() else None
    return out


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--pool", required=True)
    ap.add_argument("--hosts", nargs="+", default=["local"])
    ap.add_argument("--library", nargs="+", default=[str(HERE / "runs/C1")])
    ap.add_argument("--cells", nargs="+", default=["SS", "SC"])
    ap.add_argument("--seeds", nargs=2, type=int, default=[0, 1], help="screening seed and confirmation seed")
    ap.add_argument("--min-gain", type=float, default=0.022, help="bpb below raw needed on both seeds (3 noise sds)")
    a = ap.parse_args(argv)
    lib = library(a.library, a.cells)
    s0, s1 = a.seeds
    print("pool %s: %d files in the library; screening on seed %d" % (a.pool, len(lib), s0), flush=True)
    raw0 = run(a.pool, a.hosts, s0, {"raw": "raw"})["raw"]
    screen = {n: r for n, r in run(a.pool, a.hosts, s0, lib).items() if r}
    if not raw0 or not screen:
        print("no usable runs; see curation/results/%s" % a.pool)
        return 1
    ranked = sorted(screen, key=lambda n: screen[n]["dev_bpb"])
    print("\n%-16s %9s %9s" % ("file", "dev bpb", "below raw"))
    print("%-16s %9.4f" % ("raw pool", raw0["dev_bpb"]))
    for n in ranked:
        print("%-16s %9.4f %9.4f" % (n, screen[n]["dev_bpb"], raw0["dev_bpb"] - screen[n]["dev_bpb"]))
    best = ranked[0]
    second = run(a.pool, a.hosts, s1, {"raw": "raw", best: lib[best]})
    if not second["raw"] or not second[best]:
        print("the confirmation run failed; see curation/results/%s" % a.pool)
        return 1
    gains = [raw0["sealed_bpb"] - screen[best]["sealed_bpb"], second["raw"]["sealed_bpb"] - second[best]["sealed_bpb"]]
    print("\nbest file: %s (%s)" % (best, lib[best]))
    print("sealed bpb below raw: %.4f on seed %d, %.4f on seed %d (mean %.4f)" % (gains[0], s0, gains[1], s1, sum(gains) / 2))
    if min(gains) > a.min_gain:
        print("ROUTE: reuse the stored file. Model calls: 0 USD. Training runs spent on the check: %d." % (len(lib) + 1))
    else:
        print("ROUTE: no stored file fits this pool. Run a loop with the strong planner and the cheap coder "
              "(about %.2f USD of model calls in R1)." % SC_USD)
    return 0


if __name__ == "__main__":
    sys.exit(main())
