#!/usr/bin/env python3
"""Run a set of curation loops, a fixed number at a time per GPU host. Loops that already finished are skipped and
unfinished ones resume, so the same command can be run again after a stop.

    python curation/loop/launch.py --root curation/runs/B1 --pool v2 --sigma 0.0169 --loops 4 \
        --cells SS=fable,fable QT=qwen35-4b,qwen35-4b --hosts local willow cherry --per-host 2

Loop ids are <cell>_<nn>; loop nn of every cell uses seed nn (the same random documents in the planner prompt).
Model calls are made from this machine; only the trainer runs on the other hosts.
"""
import argparse, queue, subprocess, sys, threading
from pathlib import Path

HERE = Path(__file__).resolve().parent


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--root", required=True)
    ap.add_argument("--pool", default="v2")
    ap.add_argument("--sigma", type=float, required=True)
    ap.add_argument("--slots", type=int, default=12)
    ap.add_argument("--loops", type=int, default=4, help="loops per cell")
    ap.add_argument("--cells", nargs="+", required=True, help="NAME=planner,coder")
    ap.add_argument("--hosts", nargs="+", default=["local"], help='"local" is this machine')
    ap.add_argument("--per-host", type=int, default=2)
    a = ap.parse_args(argv)
    root = Path(a.root).resolve()
    (root / "logs").mkdir(parents=True, exist_ok=True)
    jobs = queue.Queue()
    for n in range(1, a.loops + 1):               # block by block, so cells stay balanced over hosts and time
        for cell in a.cells:
            name, models = cell.split("=")
            jobs.put((name, n, models.split(",")))

    def worker(host):
        while True:
            try:
                name, n, (planner, coder) = jobs.get_nowait()
            except queue.Empty:
                return
            lid = "%s_%02d" % (name, n)
            run_dir = root / lid
            if (run_dir / "final.json").exists():
                continue
            cmd = [sys.executable, str(HERE / "engine.py"), "--run-dir", str(run_dir), "--planner", planner, "--coder", coder,
                   "--pool", a.pool, "--slots", str(a.slots), "--seed", str(n), "--sigma", str(a.sigma), "--cell", name]
            if (run_dir / "config.json").exists():
                cmd.append("--resume")             # the stored config (and its GPU host) is used
            elif host != "local":
                cmd += ["--gpu-host", host]
            with open(root / "logs" / (lid + ".log"), "a") as log:
                rc = subprocess.run(cmd, stdout=log, stderr=subprocess.STDOUT).returncode
            print("%s on %s: exit %d" % (lid, host, rc), flush=True)

    threads = [threading.Thread(target=worker, args=(h,)) for h in a.hosts for _ in range(a.per_host)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    print("all loops finished", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
