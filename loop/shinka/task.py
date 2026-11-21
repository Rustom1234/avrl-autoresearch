#!/usr/bin/env python3
"""ShinkaEvolve's circle-packing task (26 circles) for the unchanged loop engine: their starting program, their
validity check, one evaluation per slot, five minutes per evaluation.

    python loop/shinka/task.py loop --config CFG [--resume]

Reuses loop/bench/task.py (prompt builder, static check, entry point) and replaces, in this process only: the task
table entry, the program card, the required function, and the harness that scores a job.
"""
import importlib.util, json, re, shlex, socket, subprocess, sys, time
from pathlib import Path

HERE = Path(__file__).resolve().parent
spec = importlib.util.spec_from_file_location("bench_task", HERE.parent / "bench" / "task.py")
bt = importlib.util.module_from_spec(spec)
sys.modules["bench_task"] = bt
spec.loader.exec_module(bt)

bt.TASKS["pack26s"] = {"family": "packing", "n": 26, "ref": 2.635983}
bt.SIGS["packing"] = ("run_packing", 0)
bt.EXAMPLES["packing"] = ("    centers[0] = [0.5, 0.5]", "    centers[0] = [0.5, 0.52]")
# The numbered directions are the task message of ShinkaEvolve's run_evo.py, word for word, so that both systems
# get the same domain hints.
bt.PACK = """# autoresearch program card

Goal: lower the loss of train.py, a script that packs %(n)d circles into the unit square so that the sum of their
radii is as large as possible. The circles may have different radii.

loss = minus the sum of radii of the packing the script returns, lower is better. In results tables the loss is the
"official" column. One job is one evaluation: run_packing() is called once.

The starting script places the circles on two rings by hand, with no optimisation; its sum of radii is about
%(score).4f. The best known result for the sum of radii when packing 26 circles in a unit square is 2.635.

Key directions to explore:
1. The optimal arrangement likely involves variable-sized circles
2. A pure hexagonal arrangement may not be optimal due to edge effects
3. The densest known circle packings often use a hybrid approach
4. The optimization routine is critically important - simple physics-based models with carefully tuned parameters
5. Consider strategic placement of circles at square corners and edges
6. Adjusting the pattern to place larger circles at the center and smaller at the edges
7. The math literature suggests special arrangements for specific values of n
8. You can use the scipy optimize package (e.g. LP or SLSQP) to optimize the radii given center locations and constraints

Be creative and try to find a new solution better than the best known result.

What you may change: train.py only, and anything in it.

Harness facts (enforced, stated so that no slot is wasted):
- train.py must keep defining run_packing() with no arguments, returning (centers, radii, sum_radii): arrays of
  shape (26, 2) and (26,) and the sum of the radii.
- A job is stopped after 5 minutes of wall time on one CPU thread; leave a margin (finish within about 4 minutes).
- The packing is checked in double precision with zero tolerance: every circle inside [0, 1] x [0, 1], no two
  circles overlapping, and the reported sum equal to the sum of the radii. An invalid packing is a crash, so
  shrink the radii by a tiny margin before returning.
- Available packages: numpy and scipy. Nothing else. No file or network access. Do not paste coordinates of a known
  packing: the script must compute its answer.
- Results differ by about %(sigma).0e between identical runs only if the script uses unseeded randomness or the
  clock; a fixed seed makes a result repeatable.
"""


class ShinkaJobs(bt.CpuJobs):
    def submit(self, train_py_path, *, seed, sealed, tag, kind, gpu_affinity=None):
        jid = "%s_%d" % (re.sub(r"[^A-Za-z0-9_.-]", "-", tag), time.time_ns() // 1000000)
        out = self.jobs / jid
        out.mkdir()
        cmd = [bt.PY, str(HERE / "harness.py"), str(train_py_path), "--task", self.task, "--out", str(out), "--seed",
               str(seed), "--timeout", str(self.job_timeout_s), "--job-id", jid] + (["--sealed"] if sealed else [])
        (out / "cmd.json").write_text(json.dumps({"cmd": cmd, "host": self.host, "kind": kind, "t": time.time()}))
        log = str(out / "harness.log")
        if self.host == socket.gethostname().split(".")[0]:
            subprocess.Popen(cmd, stdout=open(log, "w"), stderr=subprocess.STDOUT, stdin=subprocess.DEVNULL,
                             start_new_session=True, cwd=str(HERE))
        else:
            remote = "cd %s && setsid nohup %s > %s 2>&1 < /dev/null" % (HERE, shlex.join(cmd), shlex.quote(log))
            subprocess.run(["ssh", "-f", "-o", "BatchMode=yes", "-o", "ConnectTimeout=10", self.host, remote],
                           check=True, timeout=40)
        return jid


bt.CpuJobs = ShinkaJobs
if __name__ == "__main__":
    sys.exit(bt.main())
