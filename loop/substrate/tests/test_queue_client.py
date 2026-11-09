"""QueueClient tests. `_run` is monkeypatched: either a recorder, or a loopback that executes the
remote shell scripts locally (bash -c) against a temp ROOT, so the real scripts are exercised."""
import json
import re
import shutil
import subprocess
import sys
import threading
import time
from pathlib import Path
from types import SimpleNamespace

import pytest

import queue_client
from queue_client import QueueClient, QueueError
from conftest import approx, expected_bpb


def cp(rc=0, out="", err=""):
    return subprocess.CompletedProcess([], rc, out, err)


@pytest.fixture
def loop(env, monkeypatch):
    """QueueClient whose ssh/scp run locally against env.root; returns (client, calls)."""
    root = env.root
    env.make_worker()  # creates the ROOT directory tree
    calls = []

    def fake_run(self, cmd):
        calls.append(cmd)
        if cmd[0] == "ssh":
            return subprocess.run(["bash", "-c", cmd[-1]], capture_output=True, text=True, timeout=60)
        if cmd[0] == "scp":
            dest = cmd[-1].split(":", 1)[1]
            if not Path(dest).parent.is_dir():
                return cp(1, "", "No such file or directory")
            shutil.copy(cmd[-2], dest)
            return cp()
        raise AssertionError(cmd)

    monkeypatch.setattr(QueueClient, "_run", fake_run)
    return QueueClient(root=str(root), backoff_s=0), calls


@pytest.fixture
def local_train(tmp_path):
    p = tmp_path / "cand.py"
    p.write_text("print('hi')\n")
    return p


# ---- recorder-level tests ----
def test_submit_commands_and_job_json(monkeypatch, local_train):
    calls = []
    monkeypatch.setattr(QueueClient, "_run", lambda self, cmd: (calls.append(cmd), cp())[1])
    c = QueueClient()
    jid = c.submit(local_train, seed=3, sealed=True, timeout_s=900, priority=2, tag="my tag/x", meta={"a": 1})
    assert re.fullmatch(r"\d+_my_tag_x_[0-9a-f]{8}", jid)
    scp, ssh = calls
    assert scp[:5] == ["scp", "-o", "BatchMode=yes", "-o", "ConnectTimeout=15"]
    assert scp[-2:] == [str(local_train), "rustom@hannah:/home/rustom/arloop/inbox/%s.train.py" % jid]
    assert ssh[:6] == ["ssh", "-o", "BatchMode=yes", "-o", "ConnectTimeout=15", "rustom@hannah"]
    script = ssh[-1]
    body = re.search(r"<<'(AR_EOF_\w+)'\n(.*)\n\1\n", script).group(2)
    assert json.loads(body) == {
        "job_id": jid, "kind": "train", "train_py": "/home/rustom/arloop/inbox/%s.train.py" % jid,
        "seed": 3, "sealed": True, "timeout_s": 900, "priority": 2, "tag": "my tag/x", "meta": {"a": 1}}
    assert len(calls) == 2  # scp + ONE ssh


def test_submit_missing_file_and_bad_meta_fail_before_any_transfer(monkeypatch, local_train):
    calls = []
    monkeypatch.setattr(QueueClient, "_run", lambda self, cmd: (calls.append(cmd), cp())[1])
    c = QueueClient()
    with pytest.raises(FileNotFoundError):
        c.submit("/no/such/file.py")
    with pytest.raises(TypeError):
        c.submit(local_train, meta={"x": object()})
    assert calls == []


def test_poll_is_one_ssh_call(monkeypatch):
    calls = []
    out = "@@AR_BEGIN a\n{\"status\": \"ok\"}\n@@AR_END\n"
    monkeypatch.setattr(QueueClient, "_run", lambda self, cmd: (calls.append(cmd), cp(0, out))[1])
    assert QueueClient().poll(["a", "b", "c"]) == {"a": {"status": "ok"}, "b": None, "c": None}
    assert len(calls) == 1 and calls[0][0] == "ssh"
    assert QueueClient().poll([]) == {}
    with pytest.raises(ValueError):
        QueueClient().poll(["a;rm -rf /"])


def test_retries_with_exponential_backoff(monkeypatch):
    sleeps, rcs = [], iter([255, 255, 0])
    monkeypatch.setattr(queue_client.time, "sleep", sleeps.append)
    monkeypatch.setattr(QueueClient, "_run", lambda self, cmd: cp(next(rcs), "ok", "connection refused"))
    assert QueueClient(backoff_s=2.0)._ssh("true") == "ok"
    assert sleeps == [2.0, 4.0]


def test_gives_up_after_three_attempts(monkeypatch):
    n = []
    monkeypatch.setattr(queue_client.time, "sleep", lambda s: None)
    monkeypatch.setattr(QueueClient, "_run", lambda self, cmd: (n.append(1), cp(255, "", "no route"))[1])
    with pytest.raises(QueueError, match="3 attempts.*no route"):
        QueueClient()._ssh("true")
    assert len(n) == 3


def test_timeout_is_retried(monkeypatch):
    seq = iter([subprocess.TimeoutExpired("ssh", 1), cp(0, "fine")])
    def run(self, cmd):
        r = next(seq)
        if isinstance(r, Exception):
            raise r
        return r
    monkeypatch.setattr(queue_client.time, "sleep", lambda s: None)
    monkeypatch.setattr(QueueClient, "_run", run)
    assert QueueClient()._ssh("true") == "fine"


# ---- loopback tests (real shell scripts, local filesystem) ----
def test_submit_writes_pending_job_atomically(loop, env, local_train):
    c, _ = loop
    jid = c.submit(local_train, tag="t", meta={"note": "it's a $(echo pwn) `x` \"q\"\nline2"})
    pend = env.root / "queue" / "pending"
    assert [p.name for p in pend.iterdir()] == [jid + ".json"]  # no leftover tmp file
    job = json.loads((pend / (jid + ".json")).read_text())
    assert job["meta"]["note"] == "it's a $(echo pwn) `x` \"q\"\nline2"
    assert (env.root / "inbox" / (jid + ".train.py")).read_text() == "print('hi')\n"


def test_status(loop, env):
    c, _ = loop
    env.make_worker(gpu=0).heartbeat("j9")
    env.make_worker(gpu=1).heartbeat(None)
    env.put_job("p1")
    env.put_job("p2")
    (env.root / "queue" / "running" / "r1.json").write_text("{}")
    (env.root / "queue" / "done" / "d1.json").write_text("{}")
    st = c.status()
    assert (st["pending"], st["running"], st["done"]) == (2, 1, 1)
    assert st["running_jobs"] == ["r1"] and st["stop"] is False
    assert set(st["workers"]) == {0, 1} and st["workers"][0]["current_job"] == "j9"
    assert 0 <= st["workers"][0]["age_s"] < 30
    c.stop_workers()
    assert (env.root / "STOP").exists() and c.status()["stop"] is True


def test_cancel(loop, env):
    c, _ = loop
    env.put_job("c1")
    (env.root / "queue" / "running" / "c2.json").write_text("{}")
    assert c.cancel("c1") is True
    assert (env.root / "queue" / "done" / "c1.json").exists()
    assert c.poll(["c1"])["c1"]["status"] == "cancelled"
    assert c.cancel("c1") is False  # already gone from pending
    assert c.cancel("c2") is False  # already claimed
    assert not (env.root / "results" / "c2").exists()


def test_wait_returns_when_results_appear(loop, env):
    c, _ = loop

    def later():
        time.sleep(0.4)
        d = env.root / "results" / "w1"
        d.mkdir(parents=True)
        (d / "result.json").write_text(json.dumps({"job_id": "w1", "status": "ok"}))

    threading.Thread(target=later).start()
    out = c.wait(["w1"], timeout_s=20, poll_interval=0.1)
    assert out["w1"]["status"] == "ok"


def test_wait_times_out_with_none(loop):
    c, _ = loop
    t0 = time.time()
    assert c.wait(["nope"], timeout_s=0.5, poll_interval=0.1) == {"nope": None}
    assert time.time() - t0 < 5


def test_wait_raises_after_repeated_poll_failures(monkeypatch):
    monkeypatch.setattr(queue_client.time, "sleep", lambda s: None)
    monkeypatch.setattr(QueueClient, "_run", lambda self, cmd: cp(255, "", "down"))
    with pytest.raises(QueueError):
        QueueClient().wait(["x"], timeout_s=100, poll_interval=0)


def test_full_chain_submit_worker_poll(loop, env, local_train):
    """submit -> worker (real fake_train run) -> poll, no hannah involved."""
    c, _ = loop
    jid = c.submit(Path(__file__).parent / "fake_train.py", seed=5, tag="chain")
    env.make_worker().run(once=True)
    res = c.wait([jid], timeout_s=20, poll_interval=0.1)[jid]
    assert res["status"] == "ok" and res["official_val_bpb"] == approx(expected_bpb(1.25))
    assert res["seed_patched"] is True and res["seed_ok"] is True
    assert res["tag"] == "chain"


def test_requeue_stale_runs_worker_maintenance_mode_on_the_login_side(loop, env):
    c, calls = loop
    code = env.root / "code"
    code.mkdir(parents=True, exist_ok=True)
    for f in ("worker.py", "harness_tail.py"):
        shutil.copy(Path(queue_client.__file__).parent / f, code / f)
    c.python = sys.executable
    env.put_running("orphan", age=500)
    env.put_running("exhausted", age=500, retries=2)
    env.put_running("alive", age=100)
    env.write_heartbeat(0, "alive", age=2)
    n = len(calls)
    assert c.requeue_stale() == {"requeued": ["orphan"], "infra_fail": ["exhausted"]}
    assert len(calls) == n + 1 and calls[n][0] == "ssh" and "--requeue-stale" in calls[n][-1]  # one ssh call
    assert json.loads((env.root / "queue" / "pending" / "orphan.json").read_text())["retries"] == 1
    assert json.loads((env.root / "results" / "exhausted" / "result.json").read_text())["status"] == "infra_fail"
    assert c.poll(["exhausted"])["exhausted"]["status"] == "infra_fail"  # and the loop controller can see it
    assert (env.root / "queue" / "running" / "alive.json").exists()
