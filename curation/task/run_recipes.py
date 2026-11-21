"""Run a list of curators on the lab GPUs, one run at a time per GPU (same lock files as the loop engine, so it can
share the machines with running loops). Run on spruce.

    python curation/task/run_recipes.py --pool v3 --hosts local willow cherry --seeds 0 1 --jobs raw mix:clean_wiki \
        tfidf0.8 name=/path/to/curate.py
A job is "raw", a mode of baselines/curators.py (dedup, heuristic, tfidf0.5 ...), "mix:..." (baselines/oracle_mix.py)
or NAME=FILE. Results: curation/results/<pool>/<name>_s<seed>.json; finished results are skipped.
"""
import argparse, fcntl, os, queue, shlex, subprocess, sys, threading
from pathlib import Path

HERE = Path(__file__).resolve().parent
PY = str(HERE.parent / ".venv/bin/python")
SCRATCH = "/tmp/arloop_curation_%s" % os.environ.get("USER", "u")


def spec(job, pool):
    """(result name, curator file, extra environment)."""
    labels = str(HERE.parents[1] / "data/curation" / ("pool_%s" % pool) / "labels_HIDDEN.json")
    if "=" in job and not job.startswith("mix:"):
        name, path = job.split("=", 1)
        return name, path, {}
    if job == "raw":
        return "raw", str(HERE / "curate.py"), {}
    if job.startswith("mix:"):
        name = "orc_" + job[4:].replace("clean_wiki", "cw").replace("wiki_", "").replace("dmg_", "").replace(":tail=", "_tail_")
        return name, str(HERE / "baselines/oracle_mix.py"), {"CURATE_MODE": job, "CURATION_ORACLE_LABELS": labels}
    env = {"CURATE_MODE": job}
    if job.startswith("oracle"):
        env["CURATION_ORACLE_LABELS"] = labels
    return job, str(HERE / "baselines/curators.py"), env


def worker(host, q, pool, out_dir):
    while True:
        try:
            job, seed = q.get_nowait()
        except queue.Empty:
            return
        name, path, env = spec(job, pool)
        out = out_dir / ("%s_s%d.json" % (name, seed))
        if out.exists():
            continue
        cmd = [PY, str(HERE / "run.py"), "--curate", path, "--seed", str(seed), "--out", str(out), "--pool", pool]
        line = " ".join(["%s=%s" % (k, shlex.quote(v)) for k, v in env.items()] + [shlex.quote(c) for c in cmd])
        full = ["bash", "-c", line] if host == "local" else ["ssh", "-o", "BatchMode=yes", host, line]
        os.makedirs(SCRATCH, exist_ok=True)
        with open("%s/gpu_%s.lock" % (SCRATCH, host), "w") as lock:
            fcntl.flock(lock, fcntl.LOCK_EX)
            p = subprocess.run(full, capture_output=True, text=True)
        print("%s seed %d on %s: %s" % (name, seed, host, "ok" if out.exists() else "FAILED " + p.stderr.strip()[-400:]),
              flush=True)


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--pool", required=True)
    ap.add_argument("--hosts", nargs="+", default=["local", "willow", "cherry"])
    ap.add_argument("--seeds", nargs="+", type=int, default=[0, 1])
    ap.add_argument("--jobs", nargs="+", required=True)
    a = ap.parse_args()
    out_dir = HERE.parent / "results" / a.pool
    out_dir.mkdir(parents=True, exist_ok=True)
    q = queue.Queue()
    for seed in a.seeds:
        for job in a.jobs:
            q.put((job, seed))
    threads = [threading.Thread(target=worker, args=(h, q, a.pool, out_dir)) for h in a.hosts]
    [t.start() for t in threads]
    [t.join() for t in threads]
    print("all recipes finished", flush=True)


if __name__ == "__main__":
    sys.exit(main())
