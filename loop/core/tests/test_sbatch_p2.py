"""start_workers.sbatch: GPU_LIST, FIRST_JOB and the busy-GPU preflight, run for real against fake nvidia-smi and a fake
worker interpreter (no GPU, no Slurm). The fake interpreter logs its arguments and touches STOP when asked to supervise,
so the script's loop ends after one round."""
import os
import shutil
import signal
import stat
import subprocess
import time
from pathlib import Path

import pytest

SBATCH = Path(__file__).resolve().parents[2] / "substrate" / "start_workers.sbatch"

FAKE_PY = '''#!/bin/bash
echo "$*" >> "$FAKE_LOG"
case "$*" in *--supervise*) touch "$ROOT/STOP";; esac
'''


def fake_smi(rows):
    return "#!/bin/bash\n" + "".join("echo '%s'\n" % r for r in rows)


def run_script(tmp_path, smi=None, **env):
    """Returns (exit code, argument lines the fake interpreter saw, root, preflight log text)."""
    root, bindir = tmp_path / "root", tmp_path / "bin"
    bindir.mkdir()
    py = bindir / "fakepy"
    py.write_text(FAKE_PY)
    py.chmod(py.stat().st_mode | stat.S_IEXEC)
    if smi is not None:
        exe = bindir / "nvidia-smi"
        exe.write_text(smi)
        exe.chmod(exe.stat().st_mode | stat.S_IEXEC)
    log = tmp_path / "fake.log"
    run_env = dict(os.environ, ROOT=str(root), BASE=str(tmp_path), PY=str(py), CORES_PER="1", INTERVAL="1",
                   FAKE_LOG=str(log), PATH="%s:%s" % (bindir, os.environ["PATH"]), **env)
    (root / "code").mkdir(parents=True)
    pre = env.get("PRE_STOP")
    if pre:
        (root / "STOP").write_text("")
    p = subprocess.run(["bash", str(SBATCH)], env=run_env, capture_output=True, text=True, timeout=60)
    lines = log.read_text().splitlines() if log.exists() else []
    plog = root / "logs" / "preflight.log"
    return p.returncode, lines, root, plog.read_text() if plog.exists() else ""


def workers(lines):
    return sorted(int(l.split("--gpu ")[1].split()[0]) for l in lines if "--gpu " in l)


def supervised(lines):
    (line,) = [l for l in lines if "--supervise" in l]
    return line.split("--gpus ")[1].split()[0]


pytestmark = pytest.mark.skipif(shutil.which("taskset") is None or not {0, 1, 2, 3} <= os.sched_getaffinity(0),
                                reason="needs taskset and CPUs 0 to 3")


def test_syntax():
    assert subprocess.run(["bash", "-n", str(SBATCH)]).returncode == 0


def test_defaults_start_gpus_0_to_3_and_requeue_at_start(tmp_path):
    smi = fake_smi(["0, 15 MiB", "1, 15 MiB", "2, 15 MiB", "3, 15 MiB"])
    rc, lines, root, plog = run_script(tmp_path, smi)
    assert rc == 0 and workers(lines) == [0, 1, 2, 3] and supervised(lines) == "0,1,2,3"
    assert any("--requeue-all-running" in l for l in lines) and plog == ""


def test_busy_gpus_are_skipped_logged_and_not_supervised(tmp_path):
    smi = fake_smi(["0, 15 MiB", "1, 2001 MiB", "2, 2000 MiB", "3, 40000 MiB"])
    rc, lines, root, plog = run_script(tmp_path, smi)
    assert rc == 0 and workers(lines) == [0, 2]            # exactly 2000 MiB is allowed, more is not
    assert supervised(lines) == "0,2"
    assert "gpu1 has 2001 MiB in use" in plog and "gpu3 has 40000 MiB in use" in plog and "gpu0" not in plog


def test_gpu_list_with_first_job_zero_leaves_stop_and_running_jobs_alone(tmp_path):
    smi = fake_smi(["0, 15 MiB", "1, 15 MiB", "2, 15 MiB", "3, 15 MiB"])
    rc, lines, root, _ = run_script(tmp_path, smi, GPU_LIST="2 3", FIRST_JOB="0", PRE_STOP="1")
    assert rc == 0 and (root / "STOP").exists()             # STOP of the first job survives, so no supervisor round
    assert workers(lines) == [2, 3] and not any("--requeue-all-running" in l for l in lines)


def test_gpu_list_with_commas_and_first_job_one_clears_stop(tmp_path):
    smi = fake_smi(["0, 15 MiB", "1, 15 MiB", "2, 15 MiB", "3, 15 MiB"])
    rc, lines, root, _ = run_script(tmp_path, smi, GPU_LIST="1,3", PRE_STOP="1")
    assert workers(lines) == [1, 3] and supervised(lines) == "1,3"
    assert any("--requeue-all-running" in l for l in lines)   # the supervise round then touched STOP again


def test_ngpu_still_works_without_gpu_list(tmp_path):
    smi = fake_smi(["0, 15 MiB", "1, 15 MiB", "2, 15 MiB", "3, 15 MiB"])
    rc, lines, _, _ = run_script(tmp_path, smi, NGPU="2")
    assert workers(lines) == [0, 1] and supervised(lines) == "0,1"


def test_no_usable_gpu_means_no_worker_and_a_failed_job(tmp_path):
    smi = fake_smi(["0, 5000 MiB", "1, 5000 MiB"])
    rc, lines, root, plog = run_script(tmp_path, smi, NGPU="2")
    assert rc == 1 and workers(lines) == [] and not any("--supervise" in l for l in lines)
    assert "no usable GPU left" in plog


def test_a_failing_nvidia_smi_does_not_block_the_workers(tmp_path):
    rc, lines, root, plog = run_script(tmp_path, "#!/bin/bash\nexit 9\n", NGPU="2")
    assert rc == 0 and workers(lines) == [0, 1] and "nvidia-smi unavailable" in plog


def test_a_gpu_missing_from_nvidia_smi_is_skipped(tmp_path):
    rc, lines, root, plog = run_script(tmp_path, fake_smi(["0, 15 MiB"]), NGPU="2")
    assert workers(lines) == [0] and "gpu1 not reported" in plog


def test_preflight_can_be_switched_off(tmp_path):
    rc, lines, _, plog = run_script(tmp_path, fake_smi(["0, 30000 MiB"]), NGPU="1", PREFLIGHT="0")
    assert workers(lines) == [0] and plog == ""


# ---- SIGTERM from Slurm (time limit, scancel): the setsid workers must not outlive the job
SLEEPY_PY = '''#!/bin/bash
case "$*" in
  *--requeue-all-running*) exit 0;;
  *--supervise*) if [ -n "$FAKE_RESTART" ] && [ ! -e "$FAKE_PIDS.once" ]; then touch "$FAKE_PIDS.once"; echo 0; fi; exit 0;;
esac
echo "$$ $*" >> "$FAKE_PIDS"
while :; do sleep 1; done
'''


def pid_alive(pid):
    """Alive and not a zombie (a killed child of a script that has exited may stay a zombie for a moment)."""
    try:
        return not (Path("/proc/%d/status" % pid).read_text().split("State:")[1].split()[0].startswith("Z"))
    except (OSError, IndexError):
        return False


def worker_lines(pids_file):
    """[(pid, gpu)] in start order, from the lines the fake worker interpreter appended."""
    if not pids_file.exists():
        return []
    return [(int(l.split()[0]), int(l.split("--gpu ")[1].split()[0])) for l in pids_file.read_text().splitlines()]


def wait_for(cond, seconds):
    end = time.time() + seconds
    while time.time() < end:
        if cond():
            return True
        time.sleep(0.1)
    return cond()


def term_run(tmp_path, n_lines, **env):
    """Start the script with long-lived fake workers, wait for n_lines worker starts, send SIGTERM. Returns the exit code,
    the seconds the script needed to exit, the (pid, gpu) list and the root."""
    root, bindir = tmp_path / "root", tmp_path / "bin"
    bindir.mkdir()
    py = bindir / "fakepy"
    py.write_text(SLEEPY_PY)
    py.chmod(py.stat().st_mode | stat.S_IEXEC)
    pids = tmp_path / "pids"
    (root / "code").mkdir(parents=True)
    run_env = dict(os.environ, ROOT=str(root), BASE=str(tmp_path), PY=str(py), CORES_PER="1", NGPU="2", PREFLIGHT="0",
                   FAKE_PIDS=str(pids), **env)
    proc = subprocess.Popen(["bash", str(SBATCH)], env=run_env)
    started = []
    try:
        assert wait_for(lambda: len(worker_lines(pids)) >= n_lines, 15)
        started = worker_lines(pids)
        assert all(pid_alive(pid) for pid, _ in started[:2])
        t0 = time.time()
        proc.send_signal(signal.SIGTERM)
        rc = proc.wait(timeout=30)
        return rc, time.time() - t0, started, root
    finally:
        if proc.poll() is None:
            proc.kill()
        for pid, _ in started:
            try:
                os.killpg(pid, signal.SIGKILL)
            except OSError:
                pass


def test_sigterm_ends_the_workers_the_job_started_and_does_not_wait_for_the_supervisor_sleep(tmp_path):
    rc, seconds, started, root = term_run(tmp_path, 2, INTERVAL="60")
    assert rc == 143 and seconds < 10                       # the 60 s supervisor sleep does not delay the trap
    assert sorted(g for _, g in started) == [0, 1]
    assert wait_for(lambda: not any(pid_alive(pid) for pid, _ in started), 5)
    log = (root / "logs" / "supervisor.log").read_text()
    assert "stop_workers: signalled workers" in log and all(str(pid) in log for pid, _ in started)


def test_sigterm_signals_the_restarted_worker_not_the_dead_one(tmp_path):
    rc, seconds, started, root = term_run(tmp_path, 3, INTERVAL="1", FAKE_RESTART="1")
    gs = [g for _, g in started]
    assert rc == 143 and sorted(gs[:2]) == [0, 1] and gs[2:] == [0]   # both started (either order), then gpu0 restarted
    latest = {g: pid for pid, g in started}                          # the last start per GPU
    assert wait_for(lambda: not pid_alive(latest[0]) and not pid_alive(latest[1]), 5)
    assert str(latest[0]) in (root / "logs" / "supervisor.log").read_text()
