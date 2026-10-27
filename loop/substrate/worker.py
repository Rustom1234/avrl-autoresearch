#!/usr/bin/env python3
"""GPU worker for the autoresearch job queue. Pure stdlib, Python 3.10+.

    python worker.py --gpu N --root ROOT --base BASE --python PY [--poll 5] [--once]

Claims jobs from ROOT/queue/pending by atomic rename, runs the candidate train.py
(plus the harness tail) in results/<job_id>/run, writes results/<job_id>/result.json,
and moves the job file to queue/done. See README.md for the protocol.

Maintenance modes (used by start_workers.sbatch and QueueClient, no GPU needed):
    --requeue-stale             move dead or overdue queue/running jobs back to pending
    --requeue-all-running       same, for every running job (nothing can be alive at sbatch start)
    --supervise --gpus 0,1,2,3  kill workers with a stale heartbeat, print their gpu ids, requeue
"""
import argparse
import fcntl
import hashlib
import json
import math
import os
import re
import secrets
import shutil
import signal
import socket
import subprocess
import sys
import time
from pathlib import Path

from harness_tail import PINNED_EVAL_CODE_SHA256, PINNED_PIPELINE_CODE_SHA256, build_tail

# sha256 of prepare.py at karpathy/autoresearch@228791f; BASE/pinned_hashes.json overrides.
PINNED_PREPARE_SHA256 = "4f2ba9cbb8ba8c4a3d35be405a913e2f3be3af9aea103ed52ef7b2a662058150"
DEFAULT_CACHE_SRC = "/home/rustom/.cache/autoresearch"
DEFAULT_SEALED_DIR = DEFAULT_CACHE_SRC + "/sealed"
FIXED_ENV = {
    "HF_HUB_OFFLINE": "1",
    "TRANSFORMERS_OFFLINE": "1",
    "HF_HOME": "/home/rustom/.cache/huggingface",
    "PYTHONUNBUFFERED": "1",
}
SUMMARY_INT = ("num_steps", "depth")
SUMMARY_KEYS = ("val_bpb", "training_seconds", "total_seconds", "peak_vram_mb", "mfu_percent",
                "total_tokens_M", "num_steps", "num_params_M", "depth")
SUMMARY_RE = re.compile(r"^(%s):\s+(\S+)\s*$" % "|".join(SUMMARY_KEYS))
SEED_RES = (re.compile(r"\btorch\.manual_seed\(\s*42\s*\)"),
            re.compile(r"\btorch\.cuda\.manual_seed\(\s*42\s*\)"))
MISMATCH_TOL = 1e-4
LOG_READ_BYTES = 1 << 20
OVERBUDGET_S = 450          # tail start minus launch: interpreter start + compile + 300 s training + train.py's eval
CLOCK_SAMPLE_AFTER_S = 60   # sample the GPU clock once, this far into the run (training is busy by then)
HB_STALE_S = 180            # heartbeat older than this: the worker is dead or hung
OVERRUN_GRACE_S = 120       # a running job older than timeout_s + this is stuck
YOUNG_S = 60                # a just-claimed job is not "orphaned" before this age
MAX_RETRIES = 2


class JobError(Exception):
    """The job could not be run (bad job file, missing train.py, prepare.py mismatch)."""


def atomic_write_json(path, obj):
    path = Path(path)
    tmp = path.with_name(path.name + ".tmp%d" % os.getpid())
    tmp.write_text(json.dumps(obj))
    os.replace(tmp, path)


def read_json(path):
    try:
        return json.loads(Path(path).read_text())
    except (OSError, ValueError):
        return None


def load_pins(base):
    """Expected hashes: BASE/pinned_hashes.json (per key), else module defaults. `data_files`
    (file name -> sha256 of token_bytes.pt, tokenizer.pkl, the val shard) has no default: unpinned."""
    pins = {"prepare_py_sha256": PINNED_PREPARE_SHA256,
            "evaluate_bpb_code_sha256": PINNED_EVAL_CODE_SHA256,
            "pipeline_code_sha256": PINNED_PIPELINE_CODE_SHA256,
            "data_files": {}}
    f = Path(base) / "pinned_hashes.json"
    if f.exists():
        pins.update(json.loads(f.read_text()))
    return pins


def check_prepare(base, expected):
    """Return prepare.py bytes if its sha256 equals expected, else raise JobError."""
    data = (Path(base) / "prepare.py").read_bytes()
    got = hashlib.sha256(data).hexdigest()
    if got != expected:
        raise JobError("prepare.py sha256 mismatch: got %s, pinned %s" % (got, expected))
    return data


def patch_seed(src, seed):
    """Replace torch.manual_seed(42) / torch.cuda.manual_seed(42). Returns (src, n_lines, patched)."""
    total, hit_all = 0, True
    for rx, call in zip(SEED_RES, ("torch.manual_seed(%d)", "torch.cuda.manual_seed(%d)")):
        src, n = rx.subn(call % seed, src)
        total += n
        hit_all = hit_all and n > 0
    return src, total, hit_all


def finite(x):
    return float(x) if isinstance(x, (int, float)) and not isinstance(x, bool) and math.isfinite(x) else None


def parse_printed(text):
    """Summary lines exactly as train.py prints them; the last occurrence of each key wins."""
    printed = {}
    for line in text.splitlines():
        m = SUMMARY_RE.match(line)
        if m:
            try:
                v = float(m.group(2))
            except ValueError:
                continue
            printed[m.group(1)] = int(v) if m.group(1) in SUMMARY_INT else v
    return printed


def mismatch(printed_bpb, official_bpb):
    """True if the two scores disagree by more than MISMATCH_TOL, or only one of them exists."""
    a, b = finite(printed_bpb), finite(official_bpb)
    if a is None and b is None:
        return False
    if a is None or b is None:
        return True
    return abs(a - b) > MISMATCH_TOL


def read_tail(path):
    with open(path, "rb") as fh:
        fh.seek(0, os.SEEK_END)
        fh.seek(max(0, fh.tell() - LOG_READ_BYTES))
        return fh.read().decode("utf-8", errors="replace")


def load_receipt(path, nonce, exit_code):
    """The harness receipt, only if the file exists, carries this job's nonce and the child exited 0."""
    d = read_json(path)
    if exit_code != 0 or not isinstance(d, dict) or not nonce or d.get("nonce") != nonce:
        return None
    return d


def integrity_failures(h):
    """Names of the harness integrity checks that failed: any of them nulls the top-level scores."""
    bad = [k for k in ("constants_ok", "prepare_code_hash_ok", "pipeline_code_hash_ok", "canary_ok",
                       "data_files_ok") if h.get(k) is False]
    bad += [k for k in ("target_leak_flag", "score_shape_bad") if h.get(k) is True]
    return bad


def sync_tree(src, dst):
    """Mirror the files of directory src into dst: copy missing or differently sized ones, drop extras."""
    src, dst = Path(src), Path(dst)
    dst.mkdir(parents=True, exist_ok=True)
    files = [p for p in sorted(src.iterdir()) if p.is_file()]
    for p in files:
        d = dst / p.name
        if d.exists() and d.stat().st_size == p.stat().st_size:
            continue
        tmp = dst / (".%s.tmp%d" % (p.name, os.getpid()))
        shutil.copyfile(p, tmp)
        os.replace(tmp, d)
    keep = {p.name for p in files}
    for d in list(dst.iterdir()):
        if d.is_file() and not d.name.startswith(".") and d.name not in keep:
            d.unlink()


# ---- stale job handling (shared by the worker CLI, the sbatch supervisor and QueueClient) ----
def requeue_stale(root, now=None, all_running=False, hb_stale_s=HB_STALE_S, grace_s=OVERRUN_GRACE_S,
                  young_s=YOUNG_S, max_retries=MAX_RETRIES):
    """Move stuck jobs from queue/running back to pending with retries+1; after max_retries retries finish
    them as status infra_fail. A running job is stuck if no fresh worker heartbeat names it (and it is older
    than young_s), or it is older than timeout_s + grace_s. The claim time is the file's mtime."""
    root, now = Path(root), time.time() if now is None else now
    q = root / "queue"
    live = set()
    for f in (root / "workers").glob("gpu*.json"):
        hb = read_json(f)
        if isinstance(hb, dict) and hb.get("current_job") and now - float(hb.get("ts", 0)) <= hb_stale_s:
            live.add(hb["current_job"])
    done = {"requeued": [], "infra_fail": []}
    for p in sorted((q / "running").glob("*.json")):
        try:
            age = now - p.stat().st_mtime
        except OSError:
            continue
        job = read_json(p)
        job = job if isinstance(job, dict) else {}
        try:
            timeout = int(job.get("timeout_s") or 600)
        except (TypeError, ValueError):
            timeout = 600
        job_id = p.stem
        if not (all_running or age > timeout + grace_s or (job_id not in live and age > young_s)):
            continue
        taken = p.with_name("." + p.name + ".requeue")
        try:
            os.rename(p, taken)  # atomic: if the worker just finished the job this fails and we leave it
        except FileNotFoundError:
            continue
        retries = int(job.get("retries", 0) or 0)
        if retries >= max_retries:
            rdir = root / "results" / job_id
            rdir.mkdir(parents=True, exist_ok=True)
            atomic_write_json(rdir / "result.json", {
                "job_id": job_id, "status": "infra_fail", "retries": retries, "finished": now,
                "official_val_bpb": None, "sealed_val_bpb": None, "tag": job.get("tag", ""),
                "meta": job.get("meta", {}), "error": "worker died or job overran %d times" % (retries + 1)})
            os.replace(taken, q / "done" / p.name)
            done["infra_fail"].append(job_id)
        else:
            job["retries"] = retries + 1
            tmp = q / "pending" / ("." + p.name + ".tmp")
            tmp.write_text(json.dumps(job))
            os.replace(tmp, q / "pending" / p.name)
            taken.unlink()
            done["requeued"].append(job_id)
    return done


def _cmdline(pid):
    try:
        return Path("/proc/%d/cmdline" % int(pid)).read_bytes().replace(b"\0", b" ").decode(errors="replace")
    except (OSError, ValueError, TypeError):
        return ""


def kill_heartbeat_processes(hb):
    """SIGKILL the trainer's process group and the worker named in a heartbeat, after checking that the
    pids still look like ours (pid reuse). Returns the pids signalled."""
    killed = []
    cpg, pid = hb.get("child_pgid"), hb.get("pid")
    if cpg and "train.py" in _cmdline(cpg):
        try:
            os.killpg(int(cpg), signal.SIGKILL)
            killed.append(int(cpg))
        except OSError:
            pass
    if pid and "worker.py" in _cmdline(pid):
        try:
            if os.getpgid(int(pid)) == int(pid):  # leader (setsid in the sbatch): take its whole group
                os.killpg(int(pid), signal.SIGKILL)
            else:
                os.kill(int(pid), signal.SIGKILL)
            killed.append(int(pid))
        except OSError:
            pass
    return killed


def supervise_once(root, gpus, now=None, started_at=None, hb_stale_s=HB_STALE_S, grace_s=OVERRUN_GRACE_S):
    """Kill the workers (and their trainers) whose heartbeat is stale, or whose job overran timeout_s +
    grace_s; requeue stuck jobs. Returns (gpus to restart, requeue summary)."""
    root, now = Path(root), time.time() if now is None else now
    restart = []
    for g in gpus:
        hb = read_json(root / "workers" / ("gpu%d.json" % g))
        if not isinstance(hb, dict):
            # never wrote a heartbeat: dead on arrival, but give a fresh start time to come up
            if started_at is not None and now - started_at > hb_stale_s:
                restart.append(g)
            continue
        stale = now - float(hb.get("ts", 0)) > hb_stale_s
        overdue = False
        job_id = hb.get("current_job")
        if job_id and not stale:
            running = root / "queue" / "running" / (job_id + ".json")
            job = read_json(running)
            try:
                timeout = int((job or {}).get("timeout_s") or 600)
                overdue = now - running.stat().st_mtime > timeout + grace_s
            except (OSError, TypeError, ValueError):
                pass
        if stale or overdue:
            kill_heartbeat_processes(hb)
            restart.append(g)
    return restart, requeue_stale(root, now, hb_stale_s=hb_stale_s, grace_s=grace_s)


class Worker:
    kill_grace = 10.0   # seconds between SIGTERM and SIGKILL on timeout
    kill_wait = 30.0    # seconds to wait for the leader after SIGKILL before giving up on it

    def __init__(self, gpu, root, base, python, poll=5.0, sealed_dir=DEFAULT_SEALED_DIR,
                 cache_src=DEFAULT_CACHE_SRC, stage_root=None):
        self.gpu, self.root, self.base = int(gpu), Path(root), Path(base)
        self.python, self.poll = str(python), float(poll)
        self.cache_src, self.sealed_src = Path(cache_src), Path(sealed_dir)
        self.stage_root = Path(stage_root or "/tmp/arcache_%d" % os.getuid())
        self.local_sealed = str(self.stage_root / "sealed")
        self.staging_seconds = None
        self.child_pgid = None
        self.host = socket.gethostname()
        self.q = self.root / "queue"
        for d in (self.q / "pending", self.q / "running", self.q / "done", self.root / "inbox",
                  self.root / "results", self.root / "workers"):
            d.mkdir(parents=True, exist_ok=True)

    # ---- node-local data staging ----
    def stage(self):
        """Copy data, tokenizer and the sealed shard to node-local disk once (under a lock, so the four
        workers do not race); the child runs with HOME=stage_root so prepare.py finds the local copy."""
        t = time.time()
        self.stage_root.mkdir(parents=True, exist_ok=True)
        with open(self.stage_root / ".lock", "w") as lock:
            fcntl.flock(lock, fcntl.LOCK_EX)
            cache = self.stage_root / ".cache" / "autoresearch"
            for sub in ("data", "tokenizer"):
                sync_tree(self.cache_src / sub, cache / sub)
            if self.sealed_src.is_dir():
                sync_tree(self.sealed_src, self.local_sealed)
        self.staging_seconds = time.time() - t

    def ensure_staged(self):
        if self.staging_seconds is None:
            self.stage()

    # ---- queue ----
    def heartbeat(self, current_job=None):
        atomic_write_json(self.root / "workers" / ("gpu%d.json" % self.gpu),
                          {"ts": time.time(), "gpu": self.gpu, "host": self.host,
                           "current_job": current_job, "pid": os.getpid(), "child_pgid": self.child_pgid})

    def stopped(self):
        return (self.root / "STOP").exists()

    def claim(self):
        """Atomically move the best pending job to running. Returns its running path or None.
        Order: gpu_affinity == my gpu, then no affinity, then other gpus' jobs (only if nothing else is
        pending); within a class priority desc, then mtime asc."""
        cands = []
        for p in (self.q / "pending").glob("*.json"):
            try:
                mtime = p.stat().st_mtime
            except OSError:
                continue
            prio, aff = 0, None  # unreadable job: claim it anyway so it gets an error result
            try:
                job = json.loads(p.read_text())
                prio = int(job.get("priority", 0))
                aff = job.get("gpu_affinity")
            except (OSError, ValueError, TypeError, AttributeError):
                pass
            if isinstance(aff, bool) or not isinstance(aff, int):
                aff = None
            cands.append((0 if aff == self.gpu else 1 if aff is None else 2, -prio, mtime, p.name, p))
        for *_, name, p in sorted(cands, key=lambda t: t[:4]):
            dst = self.q / "running" / name
            try:
                os.utime(p)  # the mtime of a running job is its claim time (see requeue_stale)
                os.rename(p, dst)
            except FileNotFoundError:
                continue  # another worker won the race
            return dst
        return None

    def run(self, once=False):
        """Main loop. Returns a process exit code."""
        try:
            check_prepare(self.base, load_pins(self.base)["prepare_py_sha256"])
        except (JobError, OSError, ValueError) as e:
            print("worker gpu%d: refusing to start: %s" % (self.gpu, e), file=sys.stderr, flush=True)
            return 3
        try:
            self.stage()
        except OSError as e:
            print("worker gpu%d: data staging failed: %s" % (self.gpu, e), file=sys.stderr, flush=True)
            return 4
        while True:
            self.heartbeat(None)
            if self.stopped():
                print("worker gpu%d: STOP file found, exiting" % self.gpu, flush=True)
                return 0
            running = self.claim()
            if running is not None:
                self.process(running)
            if once:
                return 0
            if running is None:
                time.sleep(self.poll)

    # ---- GPU / load observation ----
    def nvidia_smi(self, *args):
        """stdout of nvidia-smi, or None if it is missing or fails (tests replace this method)."""
        try:
            r = subprocess.run(["nvidia-smi", *args], capture_output=True, text=True, timeout=10)
        except (OSError, subprocess.TimeoutExpired):
            return None
        return r.stdout if r.returncode == 0 else None

    def gpu_procs(self):
        out = self.nvidia_smi("--query-compute-apps=pid,used_memory", "--format=csv,noheader", "-i", str(self.gpu))
        if out is None:
            return None
        procs = []
        for line in out.splitlines():
            parts = [x.strip() for x in line.split(",")]
            if parts and parts[0].isdigit():
                procs.append({"pid": int(parts[0]), "used_memory": parts[1] if len(parts) > 1 else None})
        return procs

    def gpu_clock_mhz(self):
        out = self.nvidia_smi("--query-gpu=clocks.sm", "--format=csv,noheader,nounits", "-i", str(self.gpu))
        try:
            return int(float(out.strip().splitlines()[0]))
        except (AttributeError, IndexError, ValueError):
            return None

    def snapshot(self, pgid):
        """Load average and the compute apps on my GPU; foreign = not in the trainer's process group."""
        procs = self.gpu_procs()
        foreign = []
        for p in procs or []:
            try:
                mine = pgid is not None and os.getpgid(p["pid"]) == pgid
            except OSError:
                mine = pgid is not None and p["pid"] == pgid  # already exited, still listed
            if not mine:
                foreign.append(p["pid"])
        try:
            load = list(os.getloadavg())
        except OSError:
            load = None
        return {"loadavg": load, "gpu_procs": procs, "foreign_pids": foreign}

    # ---- one job ----
    def process(self, running_path):
        job_id = running_path.stem
        self.heartbeat(job_id)
        t0 = time.time()
        job = {}
        try:
            job = json.loads(running_path.read_text())
            if not isinstance(job, dict):
                raise JobError("job file is not a JSON object")
            res = self.execute(job_id, job, t0)
        except Exception as e:  # a bad job must never kill the worker
            res = self.base_result(job_id, job if isinstance(job, dict) else {}, t0)
            res.update(status="error", error="%s: %s" % (type(e).__name__, e))
        res["finished"] = time.time()
        res["wall_seconds"] = res["finished"] - res["started"]
        rdir = self.root / "results" / job_id
        rdir.mkdir(parents=True, exist_ok=True)
        atomic_write_json(rdir / "result.json", res)
        try:
            os.replace(running_path, self.q / "done" / running_path.name)
        except FileNotFoundError:
            pass  # the supervisor requeued it from under us; the result above stays for forensics
        self.child_pgid = None
        self.heartbeat(None)
        return res

    def base_result(self, job_id, job, t0):
        return {"job_id": job_id, "status": None, "gpu": self.gpu, "host": self.host,
                "started": t0, "finished": None, "wall_seconds": None, "exit_code": None,
                "printed": {}, "harness": {}, "harness_missing": None, "official_val_bpb": None,
                "sealed_val_bpb": None, "integrity_failed": [], "mismatch_flag": False,
                "target_leak_flag": None, "eval_batch_size_used": None, "seed": job.get("seed"),
                "seed_patched": False, "seed_lines_matched": 0, "seed_ok": None,
                "sealed": bool(job.get("sealed", False)), "retries": job.get("retries", 0),
                "startup_plus_training_s": None, "overbudget": None, "data_source": None,
                "staging_seconds": None, "load": {}, "load_flagged": None, "gpu_clock_mhz": None,
                "tag": job.get("tag", ""), "meta": job.get("meta", {}), "error": None, "log_tail": []}

    def prepare_run(self, job_id, job, res):
        if job.get("kind", "train") != "train":
            raise JobError("unsupported kind %r" % job.get("kind"))
        train_py = Path(str(job.get("train_py", "")))
        if not train_py.is_absolute() or not train_py.is_file():
            raise JobError("train_py is not an existing absolute path: %r" % job.get("train_py"))
        seed = job.get("seed")
        if seed is not None and (isinstance(seed, bool) or not isinstance(seed, int)):
            raise JobError("seed must be an int or null, got %r" % (seed,))
        rdir = self.root / "results" / job_id
        if rdir.exists():
            shutil.rmtree(rdir)
        run = rdir / "run"
        run.mkdir(parents=True)
        pins = load_pins(self.base)
        (run / "prepare.py").write_bytes(check_prepare(self.base, pins["prepare_py_sha256"]))
        try:
            src = train_py.read_text(encoding="utf-8")
        except UnicodeDecodeError as e:
            raise JobError("train.py is not utf-8: %s" % e)
        if seed is not None:
            src, res["seed_lines_matched"], res["seed_patched"] = patch_seed(src, seed)
        tail = build_tail(pins["evaluate_bpb_code_sha256"], pins["pipeline_code_sha256"], pins["data_files"])
        (run / "train.py").write_text(src.rstrip("\n") + "\n" + tail, encoding="utf-8")
        return rdir, run

    def execute(self, job_id, job, t0):
        res = self.base_result(job_id, job, t0)
        self.ensure_staged()
        rdir, run = self.prepare_run(job_id, job, res)
        timeout = int(job.get("timeout_s") or 600)
        seed = job.get("seed")
        nonce = secrets.token_hex(8)
        rpath = run / ("hr_%s.json" % secrets.token_hex(8))
        rpath.unlink(missing_ok=True)
        env = os.environ.copy()
        env.update(FIXED_ENV)
        env.update(HOME=str(self.stage_root), CUDA_VISIBLE_DEVICES=str(self.gpu),
                   TORCHINDUCTOR_CACHE_DIR="/tmp/inductor_gpu%d" % self.gpu,
                   AR_NONCE=nonce, AR_RESULT_PATH=str(rpath))
        env.pop("AR_SEALED_DIR", None)
        env.pop("AR_SEED", None)
        if job.get("sealed"):
            if not Path(self.local_sealed).is_dir():
                raise JobError("sealed job but no sealed dir at %s (source %s)" % (self.local_sealed, self.sealed_src))
            env["AR_SEALED_DIR"] = self.local_sealed
        if seed is not None:
            env["AR_SEED"] = str(seed)
        res["data_source"], res["staging_seconds"] = "local", self.staging_seconds
        res["load"]["before"] = self.snapshot(None)

        timed_out = False
        clock = None
        res["started"] = launch = time.time()
        env["AR_T0"] = repr(launch)
        with open(rdir / "run.log", "wb") as log:
            proc = subprocess.Popen([self.python, "train.py"], cwd=run, env=env, stdin=subprocess.DEVNULL,
                                    stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
            self.child_pgid = proc.pid
            self.heartbeat(job_id)
            try:
                while True:
                    try:
                        proc.wait(timeout=max(0.05, min(self.poll, launch + timeout - time.time())))
                        break
                    except subprocess.TimeoutExpired:
                        self.heartbeat(job_id)
                        if clock is None and time.time() - launch >= CLOCK_SAMPLE_AFTER_S:
                            clock = self.gpu_clock_mhz()
                        if time.time() >= launch + timeout:
                            timed_out = True
                            self.kill_group(proc)
                            break
                res["load"]["after"] = self.snapshot(proc.pid)
                if clock is None:
                    clock = self.gpu_clock_mhz()
            finally:
                self.kill_group(proc, term=False)  # reap stragglers so the GPU is free for the next job
        res["gpu_clock_mhz"] = clock
        res["load_flagged"] = any(res["load"][k]["foreign_pids"] for k in res["load"])

        text = read_tail(rdir / "run.log")
        res["exit_code"] = proc.returncode
        res["printed"] = printed = parse_printed(text)
        res["log_tail"] = text.splitlines()[-40:]
        shutil.copy2(run / "train.py", rdir / "train.py")
        if rpath.exists():
            shutil.copy2(rpath, rdir / "harness_result.json")
        receipt = load_receipt(rpath, nonce, proc.returncode)
        res["harness_missing"] = receipt is None
        res["harness"] = harness = receipt or {
            "error": "harness receipt missing, without this job's nonce, or the trainer exited nonzero"}

        res["status"] = ("timeout" if timed_out else
                         "crash" if proc.returncode != 0 or "val_bpb" not in printed else "ok")
        official = finite(harness.get("official_val_bpb"))
        res["integrity_failed"] = failed = integrity_failures(harness)
        suppress = receipt is None or bool(failed)
        res["official_val_bpb"] = None if suppress else official
        res["sealed_val_bpb"] = None if suppress else finite(harness.get("sealed_val_bpb"))
        res["target_leak_flag"] = harness.get("target_leak_flag")
        res["eval_batch_size_used"] = harness.get("eval_batch_size_used")
        res["seed_ok"] = harness.get("seed_ok")
        res["mismatch_flag"] = mismatch(printed.get("val_bpb"), official)
        tail_start = finite(harness.get("tail_start_epoch"))
        if tail_start is not None:
            res["startup_plus_training_s"] = tail_start - launch
            res["overbudget"] = res["startup_plus_training_s"] > OVERBUDGET_S
        if res["status"] == "ok":
            if seed is not None and (res["seed_lines_matched"] != 2 or res["seed_ok"] is False):
                res["status"] = "invalid_seed"
            elif res["overbudget"]:
                res["status"] = "overbudget"
        return res

    def kill_group(self, proc, term=True):
        """SIGTERM the process group, wait kill_grace for the leader, then SIGKILL the whole group and
        wait at most kill_wait for the leader (a process stuck in D state must not hang the worker)."""
        pgid = proc.pid  # start_new_session=True makes the child its own group leader
        try:
            if term:
                os.killpg(pgid, signal.SIGTERM)
                try:
                    proc.wait(timeout=self.kill_grace)
                except subprocess.TimeoutExpired:
                    pass
            os.killpg(pgid, signal.SIGKILL)
        except ProcessLookupError:
            pass
        try:
            proc.wait(timeout=self.kill_wait)
        except subprocess.TimeoutExpired:
            print("worker gpu%d: pid %d did not exit after SIGKILL, continuing" % (self.gpu, pgid),
                  file=sys.stderr, flush=True)


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--root", required=True)
    ap.add_argument("--gpu", type=int)
    ap.add_argument("--base")
    ap.add_argument("--python")
    ap.add_argument("--poll", type=float, default=5.0)
    ap.add_argument("--once", action="store_true", help="process at most one job, then exit")
    ap.add_argument("--sealed-dir", default=DEFAULT_SEALED_DIR, help="source of the sealed shard")
    ap.add_argument("--cache-src", default=DEFAULT_CACHE_SRC, help="source of data/ and tokenizer/")
    ap.add_argument("--stage-root", default=None, help="node-local copy, default /tmp/arcache_<uid>")
    ap.add_argument("--requeue-stale", action="store_true")
    ap.add_argument("--requeue-all-running", action="store_true")
    ap.add_argument("--supervise", action="store_true")
    ap.add_argument("--gpus", default="0,1,2,3")
    ap.add_argument("--stale-s", type=float, default=HB_STALE_S)
    ap.add_argument("--started-at", type=float, default=None)
    a = ap.parse_args(argv)
    if a.requeue_stale or a.requeue_all_running:
        print(json.dumps(requeue_stale(a.root, all_running=a.requeue_all_running, hb_stale_s=a.stale_s)))
        return 0
    if a.supervise:
        restart, moved = supervise_once(a.root, [int(g) for g in a.gpus.split(",")],
                                        started_at=a.started_at, hb_stale_s=a.stale_s)
        print(json.dumps(moved), file=sys.stderr, flush=True)
        for g in restart:
            print(g)
        return 0
    if a.gpu is None or not a.base or not a.python:
        ap.error("--gpu, --base and --python are required to run a worker")
    signal.signal(signal.SIGTERM, lambda *_: sys.exit(143))  # unwinds into execute()'s finally: no orphan trainer
    return Worker(a.gpu, a.root, a.base, a.python, a.poll, a.sealed_dir, a.cache_src, a.stage_root).run(once=a.once)


if __name__ == "__main__":
    sys.exit(main())
