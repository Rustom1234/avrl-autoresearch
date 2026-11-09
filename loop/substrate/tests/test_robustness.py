"""Staging, load logging, affinity, stuck children, stale-job requeue and the sbatch supervisor."""
import json
import os
import shutil
import signal
import subprocess
import sys
import threading
import time
from pathlib import Path

import pytest

import worker
from conftest import approx, expected_bpb, sha

SUB = Path(worker.__file__).resolve().parent
E1, E2 = expected_bpb(1.25), expected_bpb(2.5)


# ---- G: node-local staging and the HOME override ----
def test_staging_copies_to_the_stage_root_and_the_child_uses_that_copy(env):
    r = env.run_job("g1", sealed=True)
    assert r["data_source"] == "local" and r["staging_seconds"] >= 0
    assert (env.stage_data / "shard_00000.parquet").read_text() == "1.5"
    assert sha(env.stage_tok / "token_bytes.pt") == sha(env.src / "tokenizer" / "token_bytes.pt")
    assert (env.stage_sealed / "shard_06541.parquet").read_text() == "2.5"  # sealed lives outside .cache/autoresearch
    assert "HOME %s" % env.stage in (env.root / "results" / "g1" / "run.log").read_text()
    h = r["harness"]
    assert h["val_shard_path"] == str(env.stage_data / "shard_06542.parquet")
    assert h["constants_ok"] is True  # DATA_DIR still ends .cache/autoresearch/data under the override
    assert (r["official_val_bpb"], r["sealed_val_bpb"]) == (approx(E1), approx(E2))


def test_stage_is_idempotent_resyncs_changed_sizes_and_prunes(env):
    env.make_worker().stage()
    f = env.stage_data / "shard_00000.parquet"
    m = f.stat().st_mtime_ns
    time.sleep(0.02)
    env.make_worker().stage()
    assert f.stat().st_mtime_ns == m  # same size: not copied again
    (env.src / "data" / "shard_00000.parquet").write_text("a much longer replacement")
    (env.stage_data / "stale.parquet").write_text("x")  # not in the source any more
    env.make_worker().stage()
    assert f.read_text() == "a much longer replacement" and not (env.stage_data / "stale.parquet").exists()


def test_concurrent_staging_is_safe(env):
    ws = [env.make_worker(gpu=g) for g in range(4)]
    ths = [threading.Thread(target=w.stage) for w in ws]
    for t in ths:
        t.start()
    for t in ths:
        t.join(timeout=60)
    assert all(w.staging_seconds is not None for w in ws)
    assert sorted(p.name for p in env.stage_data.iterdir()) == ["shard_00000.parquet", "shard_06542.parquet"]
    assert (env.stage_data / "shard_06542.parquet").read_text() == "1.25"


def test_staging_failure_stops_the_worker_before_claiming(env):
    env.put_job("sf1")
    shutil.rmtree(env.src / "data")
    assert env.make_worker().run() == 4
    assert (env.root / "queue" / "pending" / "sf1.json").exists()


def test_sealed_job_without_a_sealed_dir_is_an_error_result(env):
    shutil.rmtree(env.src / "sealed")
    r = env.run_job("sf2", sealed=True)
    assert r["status"] == "error" and "sealed" in r["error"]


# ---- H: load logging and GPU observation ----
def _smi(apps="", clock="1410\n", calls=None):
    def smi(self, *args):
        if calls is not None:
            calls.append(args)
        return apps if args[0].startswith("--query-compute-apps") else clock
    return smi


def test_foreign_gpu_process_sets_load_flagged_and_clock_is_sampled_once(env, monkeypatch):
    calls = []
    monkeypatch.setattr(worker.Worker, "nvidia_smi", _smi("424242, 1024 MiB\n", calls=calls))
    r = env.run_job("h1", worker_obj=env.make_worker(gpu=2))
    assert r["load_flagged"] is True and r["gpu_clock_mhz"] == 1410
    assert r["load"]["before"]["foreign_pids"] == r["load"]["after"]["foreign_pids"] == [424242]
    assert r["load"]["before"]["gpu_procs"] == [{"pid": 424242, "used_memory": "1024 MiB"}]
    assert len(r["load"]["before"]["loadavg"]) == len(r["load"]["after"]["loadavg"]) == 3
    assert all(a[-2:] == ("-i", "2") for a in calls)  # only ever asks about my GPU
    assert sum(a[0].startswith("--query-gpu") for a in calls) == 1


def test_quiet_gpu_and_missing_nvidia_smi_do_not_flag(env, monkeypatch):
    monkeypatch.setattr(worker.Worker, "nvidia_smi", _smi(""))
    assert env.run_job("h2")["load_flagged"] is False
    monkeypatch.setattr(worker.Worker, "nvidia_smi", lambda self, *a: None)  # no nvidia-smi at all
    r = env.run_job("h3")
    assert r["load_flagged"] is False and r["gpu_clock_mhz"] is None and r["load"]["before"]["gpu_procs"] is None


def test_clock_is_sampled_mid_run_when_the_job_is_long_enough(env, monkeypatch):
    calls = []
    monkeypatch.setattr(worker.Worker, "nvidia_smi", _smi("", clock="1395\n", calls=calls))
    monkeypatch.setattr(worker, "CLOCK_SAMPLE_AFTER_S", 1.0)
    monkeypatch.setenv("FAKE_MODE", "hang")
    w = env.make_worker(poll=0.5)
    w.kill_grace = 1.0
    r = env.run_job("h4", worker_obj=w, timeout_s=3)
    assert r["status"] == "timeout" and r["gpu_clock_mhz"] == 1395
    assert sum(a[0].startswith("--query-gpu") for a in calls) == 1


def test_snapshot_treats_the_trainers_process_group_as_mine(env, monkeypatch):
    child = subprocess.Popen(["sleep", "30"], start_new_session=True)
    gone = subprocess.Popen(["true"])
    gone.wait()
    try:
        listing = "%d, 100 MiB\n%d, 200 MiB\n%d, 5 MiB\n" % (child.pid, os.getpid(), gone.pid)
        monkeypatch.setattr(worker.Worker, "nvidia_smi", lambda self, *a: listing)
        w = env.make_worker()
        assert w.snapshot(child.pid)["foreign_pids"] == [os.getpid(), gone.pid]
        assert w.snapshot(gone.pid)["foreign_pids"] == [child.pid, os.getpid()]  # exited leader still listed: mine
        assert w.snapshot(None)["foreign_pids"] == [child.pid, os.getpid(), gone.pid]  # before launch: all foreign
    finally:
        child.kill()
        child.wait()


# ---- I: a child that will not die must not hang the worker ----
def test_kill_group_gives_up_on_an_unkillable_child(env, capsys):
    class Stuck:
        pid = 2 ** 22 + 12345  # no such process: killpg raises ProcessLookupError

        def wait(self, timeout=None):
            raise subprocess.TimeoutExpired("x", timeout)

    w = env.make_worker()
    w.kill_grace, w.kill_wait = 0.01, 0.05
    t0 = time.time()
    w.kill_group(Stuck())
    w.kill_group(Stuck(), term=False)
    assert time.time() - t0 < 5
    assert "did not exit after SIGKILL" in capsys.readouterr().err


def test_retries_field_is_echoed(env):
    assert env.run_job("rt1", retries=1)["retries"] == 1


# ---- I: stale running jobs ----
def test_requeue_stale_rules(env):
    env.put_running("orphan", age=100)  # nobody's heartbeat names it, older than 60 s
    env.put_running("alive", age=150)
    env.write_heartbeat(0, "alive", age=5)  # a fresh heartbeat names it
    env.put_running("deadhb", age=100)
    env.write_heartbeat(1, "deadhb", age=400)  # only a stale heartbeat names it
    env.put_running("overrun", age=100 + 120 + 30, timeout_s=100)
    env.write_heartbeat(2, "overrun", age=1)  # worker alive, but the job is past timeout_s + 120
    env.put_running("young", age=10)  # just claimed: its worker may not have written the heartbeat yet
    out = worker.requeue_stale(env.root)
    assert sorted(out["requeued"]) == ["deadhb", "orphan", "overrun"] and out["infra_fail"] == []
    pend = env.root / "queue" / "pending"
    for j in out["requeued"]:
        assert json.loads((pend / (j + ".json")).read_text())["retries"] == 1
    assert sorted(p.stem for p in (env.root / "queue" / "running").glob("*.json")) == ["alive", "young"]
    assert not [p for p in (env.root / "queue" / "running").iterdir() if p.name.startswith(".")]  # no stray temp files


def test_retries_count_up_to_two_then_infra_fail(env):
    env.put_running("r", age=500)
    q = env.root / "queue"
    for expect in (1, 2):
        assert worker.requeue_stale(env.root)["requeued"] == ["r"]
        assert json.loads((q / "pending" / "r.json").read_text())["retries"] == expect
        os.rename(q / "pending" / "r.json", q / "running" / "r.json")  # a worker claims it again ...
        os.utime(q / "running" / "r.json", (time.time() - 500, time.time() - 500))  # ... and dies again
    assert worker.requeue_stale(env.root) == {"requeued": [], "infra_fail": ["r"]}
    res = json.loads((env.root / "results" / "r" / "result.json").read_text())
    assert res["status"] == "infra_fail" and res["retries"] == 2 and res["official_val_bpb"] is None
    assert (q / "done" / "r.json").exists()
    assert not list((q / "running").glob("*.json")) and not list((q / "pending").glob("*.json"))


def test_requeue_all_running_ignores_age_and_heartbeats(env):
    env.put_running("young", age=1)
    env.put_running("alive", age=1)
    env.write_heartbeat(0, "alive", age=1)
    out = worker.requeue_stale(env.root, all_running=True)
    assert sorted(out["requeued"]) == ["alive", "young"]


def test_requeued_job_runs_again_from_a_clean_results_dir(env):
    env.put_running("again", age=100, train_py=str(Path(__file__).parent / "fake_train.py"))
    stale = env.root / "results" / "again" / "run"
    stale.mkdir(parents=True)
    (stale / "junk").write_text("from the dead attempt")
    worker.requeue_stale(env.root)
    env.make_worker().run(once=True)
    r = json.loads((env.root / "results" / "again" / "result.json").read_text())
    assert r["status"] == "ok" and r["retries"] == 1 and not (stale / "junk").exists()


def _spawn(tmp, name, gpu=None, new_session=True):
    d = tmp / ("proc_" + name)
    d.mkdir(exist_ok=True)
    script = d / name
    script.write_text("import time\ntime.sleep(600)\n")
    args = [sys.executable, str(script)] + (["--gpu", str(gpu)] if gpu is not None else [])
    return subprocess.Popen(args, start_new_session=new_session)


def _dead(p, timeout=5):
    try:
        p.wait(timeout=timeout)
        return True
    except subprocess.TimeoutExpired:
        return False


def test_supervise_kills_a_stale_worker_and_its_trainer_and_requeues(env):
    w, t = _spawn(env.tmp, "worker.py", gpu=0), _spawn(env.tmp, "train.py")
    try:
        env.put_running("j", age=500)
        env.write_heartbeat(0, "j", age=400, pid=w.pid, child_pgid=t.pid)
        restart, moved = worker.supervise_once(env.root, [0, 1])
        assert restart == [0] and moved["requeued"] == ["j"]  # gpu 1 never wrote a heartbeat: left alone here
        assert _dead(w) and _dead(t)
    finally:
        for p in (w, t):
            p.kill()


def test_supervise_leaves_a_fresh_worker_alone(env):
    w = _spawn(env.tmp, "worker.py", gpu=0)
    try:
        env.write_heartbeat(0, None, age=2, pid=w.pid)
        assert worker.supervise_once(env.root, [0])[0] == []
        time.sleep(0.2)
        assert w.poll() is None
    finally:
        w.kill()


def test_supervise_does_not_kill_a_reused_pid_or_the_callers_process_group(env):
    other = _spawn(env.tmp, "unrelated.py")  # a pid that is not a worker.py
    w = _spawn(env.tmp, "worker.py", gpu=0, new_session=False)  # not a group leader: only that pid may be killed
    try:
        env.write_heartbeat(0, None, age=400, pid=other.pid)
        assert worker.supervise_once(env.root, [0])[0] == [0]
        time.sleep(0.2)
        assert other.poll() is None
        env.write_heartbeat(0, None, age=400, pid=w.pid)
        worker.supervise_once(env.root, [0])
        assert _dead(w)  # and this test process is still here
    finally:
        other.kill()
        w.kill()


def test_supervise_restarts_a_worker_whose_job_overran(env):
    w = _spawn(env.tmp, "worker.py", gpu=2)
    try:
        env.put_running("slow", age=100 + 120 + 30, timeout_s=100)
        env.write_heartbeat(2, "slow", age=1, pid=w.pid)  # heartbeating, but stuck on a job past its limit
        restart, moved = worker.supervise_once(env.root, [2])
        assert restart == [2] and moved["requeued"] == ["slow"] and _dead(w)
    finally:
        w.kill()


def test_worker_that_never_wrote_a_heartbeat_is_restarted_after_the_grace_period(env):
    assert worker.supervise_once(env.root, [3], started_at=time.time() - 10)[0] == []
    assert worker.supervise_once(env.root, [3], started_at=time.time() - 1000)[0] == [3]


def test_maintenance_cli_modes(env):
    env.put_running("orphan", age=500)
    run = lambda *a: subprocess.run([sys.executable, str(SUB / "worker.py"), "--root", str(env.root), *a],
                                    capture_output=True, text=True, timeout=60)
    p = run("--supervise", "--gpus", "0,1", "--started-at", str(time.time() - 1000))
    assert p.returncode == 0 and p.stdout.split() == ["0", "1"]  # restart list on stdout, summary on stderr
    assert json.loads(p.stderr.strip().splitlines()[-1])["requeued"] == ["orphan"]
    env.put_running("again", age=500)
    p = run("--requeue-stale")
    assert json.loads(p.stdout)["requeued"] == ["again"]
    env.put_running("fresh", age=1)
    assert json.loads(run("--requeue-all-running").stdout)["requeued"] == ["fresh"]


# ---- I: the sbatch supervisor loop, run for real against a stalled worker ----
@pytest.mark.skipif(shutil.which("taskset") is None or 0 not in os.sched_getaffinity(0),
                    reason="needs taskset and CPU 0")
def test_sbatch_restarts_a_stalled_worker_and_requeues_running_jobs_at_start(env):
    code = env.root / "code"
    code.mkdir(parents=True)
    for f in ("worker.py", "harness_tail.py"):
        shutil.copy(SUB / f, code / f)
    env.put_running("left_by_dead_job", age=1)  # a previous Slurm job died holding it
    run_env = dict(os.environ, ROOT=str(env.root), BASE=str(env.base), PY=sys.executable, NGPU="1", CORES_PER="1",
                   INTERVAL="1", STALE_S="3", POLL="0.2",
                   WORKER_EXTRA="--cache-src %s --stage-root %s --sealed-dir %s" % (env.src, env.stage, env.src / "sealed"))
    p = subprocess.Popen(["bash", str(SUB / "start_workers.sbatch")], env=run_env, stdout=subprocess.PIPE,
                         stderr=subprocess.STDOUT, start_new_session=True, text=True)
    hb = env.root / "workers" / "gpu0.json"
    try:
        def pid_now():
            try:
                return json.loads(hb.read_text())["pid"]
            except (OSError, ValueError, KeyError):
                return None

        def wait_for(cond, timeout=40):
            end = time.time() + timeout
            while time.time() < end:
                v = cond()
                if v:
                    return v
                time.sleep(0.1)
            return None

        pid1 = wait_for(pid_now)
        assert pid1, "worker never wrote a heartbeat"
        pend = env.root / "queue" / "pending" / "left_by_dead_job.json"
        done = env.root / "queue" / "done" / "left_by_dead_job.json"
        assert pend.exists() or done.exists()  # moved out of running at start (the worker may already have run it)
        assert not (env.root / "queue" / "running" / "left_by_dead_job.json").exists() or done.exists()
        os.kill(pid1, signal.SIGSTOP)  # the worker hangs: no more heartbeats
        pid2 = wait_for(lambda: (pid_now() not in (None, pid1)) and pid_now())
        assert pid2, "supervisor did not restart the stalled worker"
        assert _pid_gone(pid1)
        assert "restarting worker gpu0" in (env.root / "logs" / "supervisor.log").read_text()
        (env.root / "STOP").write_text("")
        assert p.wait(timeout=30) == 0
    finally:
        if p.poll() is None:
            os.killpg(p.pid, signal.SIGKILL)
        try:
            os.kill(pid_now() or 0, signal.SIGKILL)
        except (OSError, TypeError):
            pass


def _pid_gone(pid, timeout=10):
    end = time.time() + timeout
    while time.time() < end:
        try:
            stat = Path("/proc/%d/stat" % pid).read_text()
        except OSError:
            return True
        if stat.rsplit(")", 1)[1].split()[0] == "Z":
            return True
        time.sleep(0.1)
    return False
