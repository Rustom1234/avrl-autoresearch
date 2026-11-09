#!/usr/bin/env python3
"""Run controller for the routed autoresearch experiment (DESIGN.md sections 2, 11, 12, 14; PREREG.md).

    controller.py plan             --root DIR [--seed N] [--base-config cfg.json]
    controller.py launch-sentinels --root DIR [--n 8]
    controller.py anchor           --root DIR [--sigma-table FILE] [--sentinels FILE]
    controller.py run              --plan DIR/plan.json [--max-inflight 6] [--window-hours 24] [--resume]
    controller.py confirm          --plan DIR/plan.json
    controller.py status           --plan DIR/plan.json

The loops themselves are subprocesses (`python -m loop.core.run_loop --config CFG [--resume]`, exit 0 when the run is
complete). This file never implements loop logic: it launches them, watches their state.json, restarts crashes, feeds
sentinel baselines through the queue, applies the symmetric stop, and runs the confirmation block.
"""
# Reader's guide (comments, because the docstring above is also the --help text).
#
# One experiment is many loops sharing one GPU queue. A loop is one LoopEngine run of one arm (arms.py): it edits a
# 300 second GPT training script and is scored by val_bpb (validation bits per byte, lower is better).
#   arm             loop configuration: A base, B adds verification, C adds planning, D adds model routing
#   slot            one GPU training run; every loop gets the same number of slots
#   block           a group of consecutive loops in the launch order, one A and one D each (some blocks add a B or a
#                   C), shuffled inside the block, so every arm is spread over the whole run window
#   anchor          the unmodified baseline: its mean val_bpb, the per-GPU offsets and sigma_used (the noise sd used by
#                   the acceptance rules), all frozen in anchor.json before any loop starts
#   sentinel        a baseline run fed through the queue during the experiment, to see whether the noise level drifts
#   symmetric stop  when the time window or the token ceiling is reached, every running loop is told to stop at the
#                   slot index it has reached and unstarted loops are skipped, so the stop treats all arms alike
#   confirmation    after the loops end, every final incumbent (best accepted script) is rerun (confirmation.py)
# Order of commands for a run: plan, launch-sentinels, anchor, run (restart_wrapper.sh keeps it alive), confirm.
# Files under --root: plan.json, anchor.json, launch_sentinels.jsonl, controller_state.json, controller.log,
# spend.jsonl, sentinels.jsonl, confirmation/, and one directory per loop (config.json, state.json, ledger.jsonl,
# calls.jsonl, run_loop.log). The controller keeps its own state in controller_state.json, so it can be killed and
# resumed at any time; loops keep running without it and are adopted again on resume.
import argparse
import fcntl
import json
import os
import random
import signal
import statistics
import subprocess
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]  # directory that holds the `loop` package; loops are launched from here
for _p in (str(REPO), str(HERE)):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import confirmation  # noqa: E402
from confirmation import BASE_TRAIN, append_jsonl, atomic_write, read_json, read_jsonl  # noqa: E402

try:
    from loop.models.budget import tokens_used
except Exception:  # no budget module: same shape ({provider: {token type: count}}), summed from calls.jsonl here
    def tokens_used(log_path):
        out = {}
        for r in read_jsonl(log_path):
            u = out.setdefault(r.get("provider") or "?", {})
            for k, v in (r.get("usage") or {}).items():
                u[k] = u.get(k, 0) + int(v or 0)
        return out


def claude_tokens(log_path):
    """Claude tokens of all types (input, output, cache read and write) in one calls.jsonl; open models are free."""
    return sum(tokens_used(log_path).get("claude", {}).values())

LOOP_CMD = [sys.executable, "-m", "loop.core.run_loop"]
NOISE_TABLE = REPO / "loop" / "runs" / "noise" / "sigma_table.txt"
ROUTING_TABLE = REPO / "loop" / "runs" / "calibration" / "routing_table.json"

N_SLOTS = 20                  # slots per loop
SPEND_CEILING_TOKENS = 60_000_000   # Claude tokens of all types, summed over all loops; above it: symmetric stop
SIGMA_INFLATION = 1.15        # PREREG: sigma_used = 1.15 x residual sd of the noise job
NOISE_RESIDUAL_SD = 0.005857  # PREREG value, used by `status` as the live sigma reference
SENTINEL_EVERY = 12           # one sentinel baseline job per this many completed loop slots, all loops together
MAX_RESTARTS = 3              # crash restarts per loop before it is marked failed
STOP_RESERVE_H = 4.5          # hours kept after the loop stop: 4 for the confirmation block, 0.5 spare (hour 19.5 of 24)
SPEND_EVERY_S = 1800          # seconds between two spend checks
STOP_GRACE_S = 3600           # after the symmetric stop, loops that have not exited by then are terminated
DRAIN_S = 1800                # after the last loop ends, how long to wait for outstanding sentinel jobs
TERMINAL = ("done", "stopped", "failed", "skipped")   # loop statuses that need no further attention


def die(msg):
    """Usage or setup error: exit status 2, which restart_wrapper.sh does not retry."""
    print(msg, file=sys.stderr)
    raise SystemExit(2)


def _now():
    """Wall clock in epoch seconds. A module-level function so tests can replace it."""
    return time.time()


def _sleep(s):
    """Sleep s seconds. A module-level function so tests can replace it."""
    time.sleep(s)


def _iso(t):
    """Epoch seconds to an ISO 8601 UTC string for controller.log."""
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime(t))


# ---- plan ----
def make_plan(root, seed=0, base=None):
    """8 blocks, each one A and one D; blocks 1, 3, 5 add a B, blocks 2, 4, 6 add a C; order inside a block from
    random.Random(seed). Writes <root>/<loop_id>/config.json for every loop and returns the plan dict."""
    root = Path(root).resolve()
    base = dict(base or {})
    rng = random.Random(seed)
    loops, blocks, count = [], [], {}
    for b in range(1, 9):
        arms = ["A", "D"] + (["B"] if b in (1, 3, 5) else []) + (["C"] if b in (2, 4, 6) else [])
        rng.shuffle(arms)
        ids = []
        for arm in arms:
            count[arm] = count.get(arm, 0) + 1
            lid = "%s_%02d" % (arm, count[arm])
            run_dir = root / lid
            cfg = dict(base, loop_id=lid, arm=arm, block=b, rng_seed=seed * 1000 + len(loops),
                       run_dir=str(run_dir), stop_file=str(run_dir / "STOP_AT"), anchor_file=str(root / "anchor.json"),
                       routing=str(ROUTING_TABLE) if arm == "D" else None,
                       n_slots=base.get("n_slots", N_SLOTS),
                       spend_ceiling_tokens=base.get("spend_ceiling_tokens", SPEND_CEILING_TOKENS))
            run_dir.mkdir(parents=True, exist_ok=True)
            atomic_write(run_dir / "config.json", json.dumps(cfg, indent=1, sort_keys=True))
            loops.append({"loop_id": lid, "arm": arm, "block": b, "index": len(loops), "run_dir": str(run_dir),
                          "config": str(run_dir / "config.json"), "stop_file": cfg["stop_file"]})
            ids.append(lid)
        blocks.append(ids)
    return {"seed": seed, "root": str(root), "n_slots": base.get("n_slots", N_SLOTS),
            "spend_ceiling_tokens": base.get("spend_ceiling_tokens", SPEND_CEILING_TOKENS),
            "blocks": blocks, "loops": loops}


# ---- anchor ----
def read_noise_table(path):
    """sigma_table.txt rows (idx seed gpu val_bpb steps tokens_M total_s peak_mb exit) -> [(gpu, val_bpb)]."""
    runs = []
    for line in Path(path).read_text().splitlines()[1:]:
        p = line.split()
        if len(p) >= 9 and p[8] == "0":
            runs.append((int(p[2]), float(p[3])))
    return runs


def residual_sd(runs):
    """sd of the runs after subtracting each GPU's own mean (dof = n minus number of GPUs)."""
    groups = {}
    for g, v in runs:
        groups.setdefault(g, []).append(v)
    # Pooled within-GPU sd: differences between GPUs are removed by the offsets, so they must not inflate sigma.
    dof = len(runs) - len(groups)
    ss = sum((v - sum(vs) / len(vs)) ** 2 for vs in groups.values() for v in vs)
    return (ss / dof) ** 0.5 if dof > 0 else None


def compute_anchor(noise, live):
    """noise, live: [(gpu, val_bpb)]. Anchor = mean of all runs; GPU offset = per-GPU mean minus grand mean;
    sigma_used = 1.15 x residual sd of the noise runs only (PREREG: fixed from the noise job, not re-estimated)."""
    runs = list(noise) + list(live)
    mean = sum(v for _, v in runs) / len(runs)
    groups = {}
    for g, v in runs:
        groups.setdefault(g, []).append(v)
    sd_noise = residual_sd(noise)
    return {"anchor_mean": mean, "n": len(runs), "n_noise": len(noise), "n_live": len(live),
            "gpu_offsets": {str(g): sum(vs) / len(vs) - mean for g, vs in sorted(groups.items())},
            "gpu_counts": {str(g): len(vs) for g, vs in sorted(groups.items())},
            "sigma_residual_noise": sd_noise, "sigma_residual_all": residual_sd(runs),
            "sigma_used": round(SIGMA_INFLATION * sd_noise, 6)}


def sentinel_runs(rows):
    """Sentinel result rows -> [(gpu, val_bpb)], keeping only runs that finished ok with a value and a GPU id."""
    return [(int(r["gpu"]), float(r["official_val_bpb"])) for r in rows
            if r.get("status") == "ok" and r.get("official_val_bpb") is not None and r.get("gpu") is not None]


def cmd_anchor(args):
    """Compute the anchor from the noise table plus the launch sentinels and write anchor.json. It refuses to
    overwrite: the anchor is fixed before any loop runs, so no later result can move the bar."""
    path = Path(args.root) / "anchor.json"
    if path.exists():
        die("%s exists; the anchor is frozen at launch, delete it by hand to recompute" % path)
    sentinels = args.sentinels or str(Path(args.root) / "launch_sentinels.jsonl")
    anchor = compute_anchor(read_noise_table(args.sigma_table), sentinel_runs(read_jsonl(sentinels)))
    anchor["sigma_table"], anchor["sentinels"] = str(args.sigma_table), sentinels
    atomic_write(path, json.dumps(anchor, indent=1, sort_keys=True))
    print("anchor mean %.6f  n %d (%d noise + %d launch sentinels)" % (
        anchor["anchor_mean"], anchor["n"], anchor["n_noise"], anchor["n_live"]))
    for g, off in anchor["gpu_offsets"].items():
        print("  gpu %s offset %+.5f (n=%d)" % (g, off, anchor["gpu_counts"][g]))
    print("residual sd: noise runs %.6f (frozen basis), all runs %.6f" % (
        anchor["sigma_residual_noise"], anchor["sigma_residual_all"] or float("nan")))
    print("sigma_used %.6f (1.15 x noise residual sd)" % anchor["sigma_used"])
    print("wrote", path)


# ---- sentinels ----
def sentinel_seed(tag, k):
    """Deterministic fresh seed, well away from the noise job (1 to 8, 42) and confirmation (1001 to 1004) seeds."""
    return random.Random("%s-%d" % (tag, k)).randint(100000, 999999)


def cmd_launch_sentinels(args):
    """The launch baseline runs for the anchor: n runs of the unmodified baseline, four at a time."""
    path = Path(args.root) / "launch_sentinels.jsonl"
    have = len(sentinel_runs(read_jsonl(path)))
    client = confirmation.make_client()
    k = len(read_jsonl(path))
    while have < args.n and k < 2 * args.n:  # bounded: a dead queue must not loop forever
        batch = range(k + 1, k + 1 + min(4, args.n - have))
        jobs = [{"train_py": str(BASE_TRAIN), "seed": sentinel_seed("launch", i), "sealed": False,
                 "tag": "sentinel-launch", "meta": {"launch_sentinel": i}} for i in batch]
        for i, job, (jid, res, attempt) in zip(batch, jobs, confirmation.run_jobs(client, jobs)):
            res = res or {}
            append_jsonl(path, {"k": i, "job_id": jid, "seed": job["seed"], "status": res.get("status") or "no_result",
                                "official_val_bpb": res.get("official_val_bpb"), "gpu": res.get("gpu"),
                                "attempt": attempt})
        k += len(jobs)
        have = len(sentinel_runs(read_jsonl(path)))
        print("launch sentinels ok: %d / %d" % (have, args.n))
    if have < args.n:
        die("only %d of %d launch sentinels succeeded; rerun to try again" % (have, args.n))


# ---- processes ----
def find_pid(cfg):
    """Pid of a live process whose argv has `--config <cfg>` (a loop from before a controller restart), else None."""
    for d in os.listdir("/proc"):
        if not d.isdigit() or int(d) == os.getpid():
            continue
        try:
            argv = Path("/proc/%s/cmdline" % d).read_bytes().decode(errors="replace").split("\0")
        except OSError:
            continue
        if "--config" in argv and argv[argv.index("--config") + 1:argv.index("--config") + 2] == [cfg]:
            return int(d)
    return None


class Controller:
    """Supervises all loops of one plan: starts up to max_inflight at a time, restarts crashes, submits sentinels,
    watches the token spend and the time window, and applies the symmetric stop. State lives in
    controller_state.json (self.st), saved every poll cycle.
    
    window_hours is the whole wall-clock allowance; loops are stopped STOP_RESERVE_H before its end to leave room
    for the confirmation block. client: a queue client for sentinels (built lazily; tests pass a fake)."""
    def __init__(self, plan_path, max_inflight=6, window_hours=24.0, poll_s=30.0, client=None):
        self.root = Path(plan_path).resolve().parent
        self.plan = read_json(plan_path)
        self.loops = {l["loop_id"]: l for l in self.plan["loops"]}
        self.order = list(self.loops)
        self.max_inflight, self.poll_s, self.client = max_inflight, poll_s, client
        self.stop_after_s = (window_hours - STOP_RESERVE_H) * 3600
        self.ceiling = self.plan.get("spend_ceiling_tokens", SPEND_CEILING_TOKENS)
        self.state_path = self.root / "controller_state.json"
        # procs: loops this process started (Popen objects). adopted: loop_id -> pid of loops found alive from before a
        # controller restart; they are not our children, so they are watched through /proc instead.
        self.procs, self.adopted = {}, {}
        self.st = None

    # -- bookkeeping --
    def log(self, msg):
        """Append a timestamped line to controller.log and echo it to stdout."""
        line = "%s %s" % (_iso(_now()), msg)
        with open(self.root / "controller.log", "a") as f:
            f.write(line + "\n")
        print(line, flush=True)

    def save(self):
        """Write controller_state.json atomically."""
        atomic_write(self.state_path, json.dumps(self.st, indent=1, sort_keys=True))

    def loop_state(self, lid):
        """The loop's own state.json as a dict ({} if it does not exist yet). The controller only ever reads it."""
        return read_json(Path(self.loops[lid]["run_dir"]) / "state.json") or {}

    def get_client(self):
        """The queue client, built on first use so that commands and tests that never submit do not need the cluster."""
        if self.client is None:
            self.client = confirmation.make_client()
        return self.client

    def total_slots(self):
        """Slots completed so far, summed over all loops."""
        return sum(ls["slot_index"] for ls in self.st["loops"].values())

    # -- start and resume --
    def init_state(self, resume):
        """Start a fresh controller state, or with resume reload the saved one and adopt loops that are still alive.
        Refuses to start over an existing state without --resume, so a run is never restarted by accident."""
        old = read_json(self.state_path)
        if old is None and self.state_path.exists():
            die("%s exists but is unreadable; fix it by hand" % self.state_path)
        if old and not resume:
            die("%s exists; pass --resume to continue that run" % self.state_path)
        if old:
            self.st = old
            for lid in self.order:
                ls = self.st["loops"][lid]
                if ls["status"] in ("pending", "running"):
                    pid = find_pid(self.loops[lid]["config"])
                    if pid:  # still alive from before the restart: watch it, do not launch a second one
                        ls["status"] = "running"
                        self.adopted[lid] = ls["pid"] = pid
                        self.log("adopted %s pid %d" % (lid, pid))
                ls["slot_index"] = max(ls["slot_index"], int(self.loop_state(lid).get("slot_index") or 0))
            self.log("resume: %s" % ", ".join("%s=%s" % (k, v) for k, v in sorted(self.counts().items())))
        else:
            self.st = {"start": _now(), "stopping": None, "stop_t": None, "stop_slots": {},
                       "sentinels_submitted": 0, "sentinels_out": {}, "last_spend_check": None, "tokens": 0,
                       "loops": {lid: {"status": "pending", "restarts": 0, "pid": None, "slot_index": 0}
                                 for lid in self.order}}
            self.log("start: %d loops, max in flight %d, loop stop after %.2f h" % (
                len(self.order), self.max_inflight, self.stop_after_s / 3600))

    def counts(self):
        """Number of loops per status, for example {'running': 6, 'pending': 10}."""
        c = {}
        for ls in self.st["loops"].values():
            c[ls["status"]] = c.get(ls["status"], 0) + 1
        return c

    # -- loop subprocesses --
    def launch(self, lid):
        """Start the loop as a detached subprocess (own session, output to run_loop.log). It gets --resume when the
        loop already has a state.json, so a restart continues the loop instead of starting it again."""
        run_dir = Path(self.loops[lid]["run_dir"])
        resume = (run_dir / "state.json").exists()
        cmd = LOOP_CMD + ["--config", self.loops[lid]["config"]] + (["--resume"] if resume else [])
        with open(run_dir / "run_loop.log", "ab") as out:
            p = subprocess.Popen(cmd, cwd=REPO, stdin=subprocess.DEVNULL, stdout=out, stderr=subprocess.STDOUT,
                                 start_new_session=True)
        self.procs[lid] = p
        self.adopted.pop(lid, None)
        ls = self.st["loops"][lid]
        ls["status"], ls["pid"] = "running", p.pid
        self.log("launched %s pid %d%s" % (lid, p.pid, " (--resume)" if resume else ""))

    def alive(self, lid):
        """True while the loop's process exists: our own child, or an adopted pid still running with the same config."""
        if lid in self.procs:
            return self.procs[lid].poll() is None
        pid = self.adopted.get(lid)
        return bool(pid) and find_pid(self.loops[lid]["config"]) == pid

    def poll_loops(self):
        """One pass over the running loops: refresh slot counts from their state.json, mark loops that exited cleanly
        as done or stopped, and restart the ones that died (up to MAX_RESTARTS, then failed)."""
        for lid in self.order:
            ls = self.st["loops"][lid]
            if ls["status"] != "running":
                continue
            running = self.alive(lid)
            ss = self.loop_state(lid)
            ls["slot_index"] = max(ls["slot_index"], int(ss.get("slot_index") or 0))
            if running:
                continue
            rc = self.procs[lid].returncode if lid in self.procs else None   # None: adopted, exit code unknown
            if rc in (None, 0) and ss.get("status") in ("done", "stopped"):
                ls["status"] = ss["status"]
                self.log("%s %s at slot %d" % (lid, ss["status"], ls["slot_index"]))
                continue
            ls["restarts"] += 1
            if ls["restarts"] > MAX_RESTARTS:
                ls["status"] = "failed"
                self.log("%s FAILED: exit %s, state %r, restart limit %d reached" % (
                    lid, rc, ss.get("status"), MAX_RESTARTS))
            else:
                self.log("%s crashed (exit %s, state %r); restart %d of %d" % (
                    lid, rc, ss.get("status"), ls["restarts"], MAX_RESTARTS))
                self.launch(lid)

    def launch_pending(self):
        """Start pending loops in plan order until max_inflight are running. Nothing new starts once a stop has
        begun."""
        if self.st["stopping"]:
            return
        running = self.counts().get("running", 0)
        for lid in self.order:
            if running >= self.max_inflight:
                break
            if self.st["loops"][lid]["status"] == "pending":
                self.launch(lid)
                running += 1

    # -- symmetric stop and spend --
    def begin_stop(self, reason):
        """Every running loop gets a stop file holding its own current slot index; loops not yet started never start."""
        st = self.st
        st["stopping"], st["stop_t"] = reason, _now()
        for lid in self.order:
            ls = st["loops"][lid]
            if ls["status"] == "pending":
                ls["status"] = "skipped"
            elif ls["status"] == "running":
                ls["slot_index"] = max(ls["slot_index"], int(self.loop_state(lid).get("slot_index") or 0))
                atomic_write(self.loops[lid]["stop_file"], str(ls["slot_index"]))
                st["stop_slots"][lid] = ls["slot_index"]
        self.log("SYMMETRIC STOP (%s): stop slots %s, skipped %d unstarted loops" % (
            reason, st["stop_slots"], self.counts().get("skipped", 0)))

    def check_spend(self, now):
        """Every SPEND_EVERY_S seconds: sum the Claude tokens over all loops' calls.jsonl, log the total to spend.jsonl,
        and begin the symmetric stop if it is over the ceiling."""
        last = self.st["last_spend_check"]
        if last is not None and now - last < SPEND_EVERY_S:
            return
        total = 0
        for lid in self.order:
            try:
                total += claude_tokens(str(Path(self.loops[lid]["run_dir"]) / "calls.jsonl"))
            except FileNotFoundError:
                pass
        over = total > self.ceiling
        self.st["last_spend_check"], self.st["tokens"] = now, total
        append_jsonl(self.root / "spend.jsonl", {"t": now, "tokens": total, "ceiling": self.ceiling, "over": over})
        if over:
            self.log("spend %d tokens is over the ceiling %d" % (total, self.ceiling))
            self.begin_stop("spend")

    def kill_stragglers(self):
        """Terminate loops that are still running STOP_GRACE_S after the symmetric stop, and mark them failed."""
        for lid in self.order:
            ls = self.st["loops"][lid]
            if ls["status"] == "running":
                self.log("%s did not exit %d s after the stop; terminating pid %s" % (lid, STOP_GRACE_S, ls["pid"]))
                try:
                    os.killpg(ls["pid"], signal.SIGTERM)  # loops are session leaders (start_new_session)
                except (ProcessLookupError, PermissionError):
                    pass
                ls["status"] = "failed"

    # -- sentinel baselines --
    def submit_sentinels(self):
        """Submit one baseline run per SENTINEL_EVERY completed loop slots (all loops together). Seeds are
        deterministic in (plan seed, k). The count is saved right after each submit, so a resume never submits a
        sentinel twice."""
        st = self.st
        if st["stopping"]:
            return
        due = self.total_slots() // self.plan.get("sentinel_every", SENTINEL_EVERY)
        while st["sentinels_submitted"] < due:
            k = st["sentinels_submitted"] + 1
            seed = sentinel_seed("plan%s" % self.plan["seed"], k)
            try:
                jid = self.get_client().submit(str(BASE_TRAIN), seed=seed, sealed=False,
                                               timeout_s=confirmation.JOB_TIMEOUT_S, tag="sentinel",
                                               meta={"sentinel": k})
            except Exception as e:  # queue unreachable: try again next cycle, never stop the loops for it
                self.log("sentinel %d submit failed: %r" % (k, e))
                return
            st["sentinels_submitted"] = k
            st["sentinels_out"][jid] = {"k": k, "seed": seed, "slots_at_submit": self.total_slots(), "t": _now()}
            self.save()
            self.log("sentinel %d submitted (%s) at %d completed slots" % (k, jid, self.total_slots()))

    def collect_sentinels(self):
        """Poll outstanding sentinel jobs without blocking and append each finished one to sentinels.jsonl."""
        out = self.st["sentinels_out"]
        if not out:
            return
        try:
            res = self.get_client().wait(list(out), timeout_s=0, poll_interval=0)
        except Exception as e:
            self.log("sentinel poll failed: %r" % (e,))
            return
        for jid, r in res.items():
            if r is None:
                continue
            m = out.pop(jid)
            append_jsonl(self.root / "sentinels.jsonl", {
                "k": m["k"], "job_id": jid, "seed": m["seed"], "status": r.get("status"),
                "official_val_bpb": r.get("official_val_bpb"), "sealed_val_bpb": r.get("sealed_val_bpb"),
                "gpu": r.get("gpu"), "slots_at_submit": m["slots_at_submit"], "submitted": m["t"],
                "recorded": _now()})

    # -- main loop --
    def run(self, resume=False):
        """The supervision loop, one cycle every poll_s seconds: poll loops, check the window and the spend (or, after
        a stop, the grace period), launch pending loops, handle sentinels, save. Ends when every loop is terminal and
        the sentinel jobs are collected or DRAIN_S has passed. Returns 0."""
        self.init_state(resume)
        t_done = None
        try:
            while True:
                now = _now()
                self.poll_loops()
                if not self.st["stopping"]:
                    if now - self.st["start"] >= self.stop_after_s:
                        self.begin_stop("window")
                    else:
                        self.check_spend(now)
                elif now - self.st["stop_t"] > STOP_GRACE_S:
                    self.kill_stragglers()
                self.launch_pending()
                self.submit_sentinels()
                self.collect_sentinels()
                self.save()
                if all(ls["status"] in TERMINAL for ls in self.st["loops"].values()):
                    t_done = t_done or now
                    if not self.st["sentinels_out"] or now - t_done > DRAIN_S:
                        break
                _sleep(self.poll_s)
        finally:
            self.save()
        if self.st["sentinels_out"]:
            self.log("gave up on %d outstanding sentinel jobs" % len(self.st["sentinels_out"]))
        self.log("finished: %s, %d slots, %d sentinels" % (
            ", ".join("%s=%s" % kv for kv in sorted(self.counts().items())), self.total_slots(),
            self.st["sentinels_submitted"]))
        return 0


# ---- commands ----
def cmd_plan(args):
    """Write plan.json and every loop's config.json. It refuses to overwrite an existing plan."""
    root = Path(args.root).resolve()
    if (root / "plan.json").exists():
        die("%s exists; the plan is frozen, delete it by hand to redo it" % (root / "plan.json"))
    root.mkdir(parents=True, exist_ok=True)
    plan = make_plan(root, args.seed, read_json(args.base_config, {}) if args.base_config else {})
    atomic_write(root / "plan.json", json.dumps(plan, indent=1))
    for b, ids in enumerate(plan["blocks"], start=1):
        print("block %d: %s" % (b, " ".join(ids)))
    print("wrote", root / "plan.json")


def cmd_run(args):
    """Run the Controller under an exclusive file lock, so two controllers can never supervise the same plan.
    Returns 3 when the lock is held, a code restart_wrapper.sh treats as a crash and retries."""
    plan = Path(args.plan).resolve()
    root = plan.parent
    if not read_json(plan, {}).get("loops"):
        die("%s is missing or has no loops: run `plan` first" % plan)
    if not (root / "anchor.json").exists():
        die("%s missing: run `anchor` before `run`" % (root / "anchor.json"))
    lock = open(root / "controller.lock", "w")
    try:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    except BlockingIOError:
        print("another controller holds %s" % lock.name, file=sys.stderr)
        return 3
    return Controller(plan, args.max_inflight, args.window_hours, args.poll_s).run(resume=args.resume)


def cmd_confirm(args):
    """Run the confirmation block (confirmation.py) and print its summary table."""
    plan_path = Path(args.plan).resolve()
    summary = confirmation.run_confirmation(plan_path.parent, read_json(plan_path))
    print_confirmation(summary)


def print_confirmation(summary):
    """Print one row per loop (ok runs, mean official and sealed val_bpb, improvement) and the anchor line."""
    print("%-6s %-3s %5s %9s %9s %12s" % ("loop", "arm", "ok/4", "official", "sealed", "improvement"))
    f = lambda x, p=5: "-" if x is None else "%.*f" % (p, x)
    for lid, d in sorted(summary["loops"].items()):
        print("%-6s %-3s %5s %9s %9s %12s" % (lid, d["arm"], "%d/%d" % (d["n_ok"], d["n_runs"]),
                                              f(d["mean_official"], 4), f(d["mean_sealed"], 4), f(d["improvement"])))
    a = summary["anchor"]
    print("anchor: %d ok, %d failed, official minus sealed offset %s" % (
        a["n_ok"], a["n_failed"], f(a["offset_official_minus_sealed"])))


def cmd_status(args):
    """Read-only progress report from the files on disk: loop counts, slots, spend, the live sigma from sentinels
    against the noise-job sigma, a row per loop, and confirmation progress. Safe to run while `run` is active."""
    plan_path = Path(args.plan).resolve()
    root, plan = plan_path.parent, read_json(plan_path)
    st = read_json(root / "controller_state.json")
    if not st:
        print("no controller_state.json yet: `run` has not started")
        return
    now = time.time()
    c = {}
    for ls in st["loops"].values():
        c[ls["status"]] = c.get(ls["status"], 0) + 1
    print("%s  elapsed %.2f h  stopping: %s" % (root, (now - st["start"]) / 3600, st["stopping"] or "no"))
    print("loops: " + "  ".join("%s %d" % kv for kv in sorted(c.items())))
    rows = read_jsonl(root / "sentinels.jsonl")
    line = "slots %d  sentinels %d submitted, %d recorded  spend %s / %s tokens" % (
        sum(ls["slot_index"] for ls in st["loops"].values()), st["sentinels_submitted"], len(rows),
        st["tokens"], plan.get("spend_ceiling_tokens"))
    print(line)
    anchor = read_json(root / "anchor.json") or {}
    obs = sentinel_runs(rows)
    if len(obs) >= 4:
        off = anchor.get("gpu_offsets", {})
        sd = statistics.stdev([v - off.get(str(g), 0.0) for g, v in obs])  # GPU-offset adjusted, as in decisions
        ref = anchor.get("sigma_residual_noise") or NOISE_RESIDUAL_SD
        print("live sigma %.5f from %d sentinels vs noise-job %.5f (ratio %.2f%s)" % (
            sd, len(obs), ref, sd / ref, ", OUTSIDE 25 percent: replay B, C, D offline" if abs(sd / ref - 1) > 0.25
            else ""))
    print("%-5s %-3s %-8s %7s %9s %3s %4s %7s" % ("loop", "arm", "status", "slot", "inc_mean", "n", "rst", "age_s"))
    for l in plan["loops"]:
        ls, ss = st["loops"][l["loop_id"]], read_json(Path(l["run_dir"]) / "state.json") or {}
        inc = ss.get("incumbent") or {}
        mean = inc.get("mean_adjusted")
        upd = ss.get("updated")
        print("%-5s %-3s %-8s %3d/%-3s %9s %3s %4d %7s" % (
            l["loop_id"], l["arm"], ls["status"], ls["slot_index"], ss.get("n_slots", plan["n_slots"]),
            "-" if mean is None else "%.5f" % mean, inc.get("n", "-"), ls["restarts"],
            "-" if not isinstance(upd, (int, float)) else int(now - upd)))
    res = read_jsonl(root / "confirmation" / "results.jsonl")
    waves = read_json(root / "confirmation" / "waves.json")
    if waves:
        print("confirmation: %d / %d jobs recorded, %d failed" % (
            len(res), sum(len(w["jobs"]) for w in waves), sum(1 for r in res if r["status"] != "ok")))


def main(argv=None):
    """Parse the subcommand and dispatch. Returns the process exit code."""
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    p = sub.add_parser("plan")
    p.add_argument("--root", required=True)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--base-config")
    p = sub.add_parser("launch-sentinels")
    p.add_argument("--root", required=True)
    p.add_argument("--n", type=int, default=8)
    p = sub.add_parser("anchor")
    p.add_argument("--root", required=True)
    p.add_argument("--sigma-table", default=str(NOISE_TABLE))
    p.add_argument("--sentinels")
    p = sub.add_parser("run")
    p.add_argument("--plan", required=True)
    p.add_argument("--max-inflight", type=int, default=6)
    p.add_argument("--window-hours", type=float, default=24.0)
    p.add_argument("--poll-s", type=float, default=30.0)
    p.add_argument("--resume", action="store_true")
    for name in ("confirm", "status"):
        sub.add_parser(name).add_argument("--plan", required=True)
    args = ap.parse_args(argv)
    fn = {"plan": cmd_plan, "launch-sentinels": cmd_launch_sentinels, "anchor": cmd_anchor, "run": cmd_run,
          "confirm": cmd_confirm, "status": cmd_status}[args.cmd]
    return fn(args) or 0


if __name__ == "__main__":
    sys.exit(main())
