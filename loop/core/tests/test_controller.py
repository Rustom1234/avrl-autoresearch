import fcntl
import json
import os
import subprocess
import sys
import threading
import time
from pathlib import Path

import pytest

import confirmation
import controller as K

WRAPPER = Path(K.HERE) / "restart_wrapper.sh"

# A stand-in for `python -m loop.core.run_loop`: same CLI and state.json contract, no model, no GPU.
FAKE_RUN_LOOP = r'''
import argparse, json, os, sys, time
from pathlib import Path
ap = argparse.ArgumentParser()
ap.add_argument("--config", required=True)
ap.add_argument("--resume", action="store_true")
a = ap.parse_args()
cfg = json.load(open(a.config))
rd, lid = Path(cfg["run_dir"]), cfg["loop_id"]

def event(kind):
    with open(rd.parent / "events.log", "a") as f:
        f.write("%s %s %.4f\n" % (kind, lid, time.time()))

def write(slot, status):
    tmp = rd / ".state.tmp"
    tmp.write_text(json.dumps({"loop_id": lid, "arm": cfg["arm"], "slot_index": slot, "n_slots": cfg["fake_slots"],
                               "incumbent": {"path": str(rd / "incumbent.py"), "sha256": "0",
                                             "mean_adjusted": 1.14, "n": 1},
                               "status": status, "in_flight_job": None, "updated": time.time()}))
    os.replace(tmp, rd / "state.json")

with open(rd / "launches.jsonl", "a") as f:
    f.write(json.dumps({"resume": a.resume, "argv": sys.argv[1:]}) + "\n")
slot, ran = 0, 0
if a.resume and (rd / "state.json").exists():
    slot = json.load(open(rd / "state.json"))["slot_index"]
(rd / "incumbent.py").write_text("# incumbent\n")
event("start")
write(slot, "running")
marker = rd / "crashed"
while slot < cfg["fake_slots"]:
    stop = Path(cfg["stop_file"])
    if stop.exists() and slot >= int(stop.read_text()):
        write(slot, "stopped")
        event("end")
        sys.exit(0)
    time.sleep(cfg["fake_sleep"])
    slot, ran = slot + 1, ran + 1
    with open(rd / "calls.jsonl", "a") as f:
        f.write(json.dumps({"provider": "claude", "usage": {"input": cfg.get("fake_tokens", 0), "output": 0,
                                                            "cache_read": 0, "cache_write": 0}}) + "\n")
    write(slot, "running")
    if (cfg.get("fake_crash_always") and ran == 1) or (cfg.get("fake_crash_at") == slot and not marker.exists()):
        marker.write_text("x")
        event("end")
        sys.exit(3)
write(slot, "done")
event("end")
'''


class FakeClient:
    """Stands in for QueueClient: records submissions, answers every wait at once with a synthetic result."""

    def __init__(self):
        self.submitted = []

    def submit(self, train_py, **kw):
        jid = "job%03d" % len(self.submitted)
        self.submitted.append(dict(kw, job_id=jid, train_py=str(train_py)))
        return jid

    def wait(self, job_ids, timeout_s, poll_interval=20):
        return {jid: {"job_id": jid, "status": "ok", "official_val_bpb": 1.14, "sealed_val_bpb": None,
                      "gpu": int(jid[3:]) % 4} for jid in job_ids}


@pytest.fixture(autouse=True)
def isolate(monkeypatch, tmp_path):
    def boom():
        raise AssertionError("a test tried to build a real QueueClient")
    monkeypatch.setattr(confirmation, "make_client", boom)
    yield
    subprocess.run(["pkill", "-f", str(tmp_path)], check=False)  # stray fake loops


@pytest.fixture
def world(tmp_path, monkeypatch):
    """A plan of the first n loops of a real make_plan, with fake loops and fast polling."""
    script = tmp_path / "fake_run_loop.py"
    script.write_text(FAKE_RUN_LOOP)
    monkeypatch.setattr(K, "LOOP_CMD", [sys.executable, str(script)])

    def build(n=6, **base):
        root = tmp_path / "root"
        base = dict({"fake_slots": 4, "fake_sleep": 0.15}, **base)
        plan = K.make_plan(root, seed=5, base=base)
        plan["loops"] = plan["loops"][:n]
        (root / "plan.json").write_text(json.dumps(plan))
        (root / "anchor.json").write_text("{}")
        return root, plan

    return build


def ctl(root, client=None, **kw):
    return K.Controller(root / "plan.json", poll_s=0.1, client=client, **kw)


def launches(root, lid):
    return [json.loads(l) for l in (root / lid / "launches.jsonl").read_text().splitlines()]


def max_overlap(root):
    ev = sorted((float(t), 1 if k == "start" else -1) for k, _, t in
                (l.split() for l in (root / "events.log").read_text().splitlines()))
    cur = peak = 0
    for _, d in ev:
        cur += d
        peak = max(peak, cur)
    return peak


# ---------------- plan ----------------
def test_plan_structure(tmp_path):
    plan = K.make_plan(tmp_path, seed=1, base={"n_slots": 20, "custom": "kept"})
    assert len(plan["blocks"]) == 8 and len(plan["loops"]) == 22
    arms = [l["arm"] for l in plan["loops"]]
    assert {a: arms.count(a) for a in "ABCD"} == {"A": 8, "D": 8, "B": 3, "C": 3}
    by_id = {l["loop_id"]: l for l in plan["loops"]}
    for b, ids in enumerate(plan["blocks"], start=1):
        want = ["A", "D"] + (["B"] if b in (1, 3, 5) else []) + (["C"] if b in (2, 4, 6) else [])
        assert sorted(by_id[i]["arm"] for i in ids) == sorted(want)
    assert [l["loop_id"] for l in plan["loops"]] == [i for ids in plan["blocks"] for i in ids]  # launch order
    assert set(by_id) >= {"A_01", "A_08", "D_01", "D_08", "B_01", "B_03", "C_01", "C_03"}
    assert {l["loop_id"] for l in plan["loops"] if l["block"] in (1, 3, 5) and l["arm"] == "B"} == {"B_01", "B_02", "B_03"}
    cfgs = [json.loads(Path(l["config"]).read_text()) for l in plan["loops"]]
    assert len({c["rng_seed"] for c in cfgs}) == 22 and len({c["run_dir"] for c in cfgs}) == 22
    for c, l in zip(cfgs, plan["loops"]):
        assert (c["loop_id"], c["arm"], c["custom"], c["n_slots"]) == (l["loop_id"], l["arm"], "kept", 20)
        assert c["stop_file"] == l["stop_file"] and c["anchor_file"] == str(tmp_path / "anchor.json")
        assert c["spend_ceiling_tokens"] == 60_000_000
        assert (c["routing"] is not None) == (l["arm"] == "D")


def test_plan_deterministic_for_a_seed(tmp_path):
    a = K.make_plan(tmp_path / "a", seed=3)
    b = K.make_plan(tmp_path / "b", seed=3)
    strip = lambda p: [(l["loop_id"], l["block"], l["index"]) for l in p["loops"]]
    assert strip(a) == strip(b) and a["blocks"] == b["blocks"]
    assert all(json.loads(Path(l["config"]).read_text())["rng_seed"] == 3000 + i for i, l in enumerate(a["loops"]))
    orders = {tuple(map(tuple, K.make_plan(tmp_path / ("s%d" % s), seed=s)["blocks"])) for s in range(5)}
    assert len(orders) > 1  # the seed changes the within-block order


def test_cmd_plan_refuses_to_overwrite(tmp_path):
    assert K.main(["plan", "--root", str(tmp_path / "r"), "--seed", "2"]) == 0
    with pytest.raises(SystemExit):
        K.main(["plan", "--root", str(tmp_path / "r")])


# ---------------- anchor ----------------
def test_anchor_arithmetic_on_a_small_table(tmp_path, capsys):
    (tmp_path / "t.txt").write_text(
        "idx seed gpu val_bpb steps tokens_M total_s peak_mb exit\n"
        "r01 42 0 1.10 1 1 1 1 0\nr02 42 0 1.12 1 1 1 1 0\nr03 42 1 1.20 1 1 1 1 0\nr04 42 1 1.22 1 1 1 1 0\n"
        "r05 42 1 9.99 1 1 1 1 1\n")  # nonzero exit: ignored
    with open(tmp_path / "launch_sentinels.jsonl", "w") as f:
        for r in ({"gpu": 0, "status": "ok", "official_val_bpb": 1.14}, {"gpu": 1, "status": "ok", "official_val_bpb": 1.18},
                  {"gpu": 1, "status": "crash", "official_val_bpb": None}):
            f.write(json.dumps(r) + "\n")
    assert K.main(["anchor", "--root", str(tmp_path), "--sigma-table", str(tmp_path / "t.txt")]) == 0
    a = json.loads((tmp_path / "anchor.json").read_text())
    assert a["n"] == 6 and (a["n_noise"], a["n_live"]) == (4, 2)
    assert a["anchor_mean"] == pytest.approx((1.10 + 1.12 + 1.20 + 1.22 + 1.14 + 1.18) / 6)  # 1.16
    assert a["gpu_offsets"]["0"] == pytest.approx(1.12 - 1.16) and a["gpu_offsets"]["1"] == pytest.approx(1.20 - 1.16)
    assert a["gpu_counts"] == {"0": 3, "1": 3}
    sd = (0.0004 / 2) ** 0.5  # noise runs only: SS within GPU = 2 * 2 * 0.01^2, dof = 4 - 2
    assert a["sigma_residual_noise"] == pytest.approx(sd)
    assert a["sigma_used"] == pytest.approx(round(1.15 * sd, 6))
    out = capsys.readouterr().out
    assert "anchor mean 1.160000" in out and "sigma_used" in out and "offset" in out
    with pytest.raises(SystemExit):  # frozen: a second call must not silently recompute
        K.main(["anchor", "--root", str(tmp_path), "--sigma-table", str(tmp_path / "t.txt")])


def test_anchor_reproduces_the_prereg_numbers_from_the_real_noise_table():
    a = K.compute_anchor(K.read_noise_table(K.NOISE_TABLE), [])
    assert a["n"] == 16 and a["anchor_mean"] == pytest.approx(1.143597625)
    assert a["sigma_residual_noise"] == pytest.approx(0.005857, abs=5e-7)
    assert a["sigma_used"] == pytest.approx(0.006736, abs=1e-9)
    prereg = {"0": -0.00306, "1": 0.00474, "2": 0.00166, "3": -0.00334}
    assert {g: round(v, 5) for g, v in a["gpu_offsets"].items()} == pytest.approx(prereg, abs=6e-6)


def test_launch_sentinels_feed_the_anchor(tmp_path, monkeypatch):
    client = FakeClient()
    monkeypatch.setattr(confirmation, "make_client", lambda: client)
    K.cmd_launch_sentinels(type("A", (), {"root": str(tmp_path), "n": 8}))
    rows = confirmation.read_jsonl(tmp_path / "launch_sentinels.jsonl")
    assert len(rows) == 8 and len(client.submitted) == 8
    assert len({r["seed"] for r in rows}) == 8 and all(r["seed"] >= 100000 for r in rows)
    assert all(j["sealed"] is False and j["train_py"] == str(confirmation.BASE_TRAIN) for j in client.submitted)
    assert len(K.sentinel_runs(rows)) == 8
    K.cmd_launch_sentinels(type("A", (), {"root": str(tmp_path), "n": 8}))  # already done: submits nothing more
    assert len(client.submitted) == 8


# ---------------- run ----------------
def test_run_respects_max_inflight_and_submits_a_sentinel_per_12_slots(world, capsys):
    root, plan = world(6)
    client = FakeClient()
    assert ctl(root, client, max_inflight=3).run() == 0
    assert max_overlap(root) == 3  # never above 3, and the cap is actually used
    st = json.loads((root / "controller_state.json").read_text())
    assert {ls["status"] for ls in st["loops"].values()} == {"done"} and st["stopping"] is None
    assert [l["resume"] for lid in st["loops"] for l in launches(root, lid)] == [False] * 6
    assert sum(ls["slot_index"] for ls in st["loops"].values()) == 24
    assert len(client.submitted) == 2 == st["sentinels_submitted"]  # 24 slots // 12
    for j in client.submitted:
        assert (j["train_py"], j["tag"], j["sealed"]) == (str(confirmation.BASE_TRAIN), "sentinel", False)
        assert "gpu_affinity" not in j
    assert len({j["seed"] for j in client.submitted}) == 2
    rows = confirmation.read_jsonl(root / "sentinels.jsonl")
    assert [r["k"] for r in rows] == [1, 2] and all(r["status"] == "ok" for r in rows)
    for r in rows:  # submitted when the 12k-th slot completed, give or take one poll interval of 3 loops
        assert 12 * r["k"] <= r["slots_at_submit"] < 12 * r["k"] + 6
    assert "finished" in (root / "controller.log").read_text()
    K.main(["status", "--plan", str(root / "plan.json")])
    out = capsys.readouterr().out
    assert "slots 24" in out and all(l["loop_id"] in out for l in plan["loops"])


def test_cli_run_with_the_fake_queue(world, monkeypatch):
    root, plan = world(3)
    client = FakeClient()
    monkeypatch.setattr(confirmation, "make_client", lambda: client)
    assert K.main(["run", "--plan", str(root / "plan.json"), "--poll-s", "0.1", "--max-inflight", "2"]) == 0
    assert len(client.submitted) == 1  # 12 slots
    with pytest.raises(SystemExit):  # state exists and --resume was not given
        K.main(["run", "--plan", str(root / "plan.json"), "--poll-s", "0.1"])
    assert K.main(["run", "--plan", str(root / "plan.json"), "--poll-s", "0.1", "--resume"]) == 0  # all finished: no-op
    assert len(client.submitted) == 1 and len(launches(root, plan["loops"][0]["loop_id"])) == 1


def test_run_needs_the_anchor_and_a_free_lock(world):
    root, _ = world(1)
    (root / "anchor.json").unlink()
    with pytest.raises(SystemExit) as e:
        K.main(["run", "--plan", str(root / "plan.json")])
    assert e.value.code == 2  # the wrapper treats 2 as "do not retry"
    with pytest.raises(SystemExit) as e:
        K.main(["run", "--plan", str(root / "nope.json")])
    assert e.value.code == 2
    (root / "anchor.json").write_text("{}")
    with open(root / "controller.lock", "w") as held:
        fcntl.flock(held, fcntl.LOCK_EX)
        assert K.main(["run", "--plan", str(root / "plan.json")]) == 3


def test_crashed_loop_is_restarted_with_resume(world):
    root, plan = world(3)
    cfg_path = Path(plan["loops"][0]["config"])
    cfg_path.write_text(json.dumps(dict(json.loads(cfg_path.read_text()), fake_crash_at=2)))
    c = ctl(root, FakeClient(), max_inflight=3)
    c.run()
    lid = plan["loops"][0]["loop_id"]
    assert [l["resume"] for l in launches(root, lid)] == [False, True]
    assert "--resume" in launches(root, lid)[1]["argv"]
    ls = c.st["loops"][lid]
    assert (ls["status"], ls["restarts"], ls["slot_index"]) == ("done", 1, 4)
    assert all(c.st["loops"][l["loop_id"]]["restarts"] == 0 for l in plan["loops"][1:])
    assert "crashed" in (root / "controller.log").read_text()


def test_loop_that_keeps_crashing_fails_after_three_restarts(world):
    root, plan = world(2)
    cfg_path = Path(plan["loops"][0]["config"])
    cfg_path.write_text(json.dumps(dict(json.loads(cfg_path.read_text()), fake_crash_always=True)))
    c = ctl(root, FakeClient(), max_inflight=2)
    assert c.run() == 0  # the run still completes
    bad, good = (l["loop_id"] for l in plan["loops"])
    assert [l["resume"] for l in launches(root, bad)] == [False, True, True, True]  # 1 launch + 3 restarts
    assert (c.st["loops"][bad]["status"], c.st["loops"][good]["status"]) == ("failed", "done")


def test_symmetric_stop_at_the_configured_hour(world, monkeypatch):
    root, plan = world(5, fake_slots=40)
    offset = [0.0]
    monkeypatch.setattr(K, "_now", lambda: time.time() + offset[0])
    threading.Timer(1.0, lambda: offset.__setitem__(0, 19.5 * 3600)).start()
    c = ctl(root, FakeClient(), max_inflight=2, window_hours=24)
    start = time.time()
    c.run()
    assert time.time() - start < 15  # stopped by the stop file, not by running 40 slots
    st = c.st
    assert st["stopping"] == "window" and st["stop_t"] - st["start"] >= 19.5 * 3600
    first, second, *rest = plan["loops"]
    assert set(st["stop_slots"]) == {first["loop_id"], second["loop_id"]}  # the two in flight
    for l in (first, second):
        n = st["stop_slots"][l["loop_id"]]
        assert Path(l["stop_file"]).read_text() == str(n) and n >= 1
        ss = json.loads((root / l["loop_id"] / "state.json").read_text())
        assert ss["status"] == "stopped" and n <= ss["slot_index"] < 40
        assert st["loops"][l["loop_id"]]["status"] == "stopped"
    for l in rest:  # never started
        assert st["loops"][l["loop_id"]]["status"] == "skipped" and not (root / l["loop_id"] / "launches.jsonl").exists()


def test_spend_ceiling_applies_the_same_stop(world, monkeypatch):
    root, plan = world(3, fake_slots=40, fake_tokens=100, spend_ceiling_tokens=1500)
    monkeypatch.setattr(K, "SPEND_EVERY_S", 0.2)
    c = ctl(root, FakeClient(), max_inflight=3)
    c.run()
    st = c.st
    assert st["stopping"] == "spend" and st["tokens"] > 1500
    spend = confirmation.read_jsonl(root / "spend.jsonl")
    assert len(spend) >= 2 and spend[-1]["over"] is True and not any(r["over"] for r in spend[:-1])
    assert spend[-1]["ceiling"] == 1500
    assert set(st["stop_slots"]) == {l["loop_id"] for l in plan["loops"]}
    for l in plan["loops"]:
        assert Path(l["stop_file"]).read_text() == str(st["stop_slots"][l["loop_id"]])
        assert st["loops"][l["loop_id"]]["status"] == "stopped"
        assert json.loads((root / l["loop_id"] / "state.json").read_text())["slot_index"] < 40


def test_spend_counts_only_claude_tokens_across_loops(world):
    root, plan = world(2, spend_ceiling_tokens=1000)
    for l, n in zip(plan["loops"], (600, 600)):
        with open(Path(l["run_dir"]) / "calls.jsonl", "w") as f:
            f.write(json.dumps({"provider": "claude", "usage": {"input": n - 6, "output": 5, "cache_read": 0, "cache_write": 1}}) + "\n")
            f.write(json.dumps({"provider": "vllm", "usage": {"input": 10 ** 9, "output": 0}}) + "\n")
            f.write("not json\n")
    assert K.claude_tokens(str(Path(plan["loops"][0]["run_dir"]) / "calls.jsonl")) == 600
    c = ctl(root, FakeClient())
    c.init_state(False)
    c.check_spend(1000.0)  # 600 + 600 over the 1000 ceiling although each loop alone is under it
    assert c.st["stopping"] == "spend" and c.st["tokens"] == 1200
    assert all(ls["status"] == "skipped" for ls in c.st["loops"].values())  # nothing had started


def test_resume_adopts_running_loops_and_skips_finished_ones(world, monkeypatch):
    root, plan = world(6, fake_slots=5)
    first = ctl(root, FakeClient(), max_inflight=2)

    class Died(Exception):
        pass

    def die_once_two_are_done(_):
        st = json.loads((root / "controller_state.json").read_text())
        if sum(ls["status"] == "done" for ls in st["loops"].values()) >= 2:
            raise Died

    monkeypatch.setattr(K, "_sleep", die_once_two_are_done)
    with pytest.raises(Died):  # the controller "dies"; its loops, each a session leader, keep running
        first.run()
    before = json.loads((root / "controller_state.json").read_text())
    done_before = [lid for lid, ls in before["loops"].items() if ls["status"] == "done"]
    running_before = [lid for lid, ls in before["loops"].items() if ls["status"] == "running"]
    assert len(done_before) == 2 and len(running_before) == 2
    monkeypatch.setattr(K, "_sleep", time.sleep)
    second = ctl(root, FakeClient(), max_inflight=2)
    with pytest.raises(SystemExit):  # a fresh start must not overwrite a run in progress
        ctl(root, FakeClient()).run(resume=False)
    second.run(resume=True)
    assert set(second.adopted) == set(running_before)  # found by their --config argv, not relaunched
    assert {ls["status"] for ls in second.st["loops"].values()} == {"done"}
    for lid in before["loops"]:
        assert len(launches(root, lid)) == 1  # finished and adopted loops were never launched a second time
    assert second.st["start"] == before["start"]
    assert second.st["sentinels_submitted"] == 30 // 12


DRIVER = """
import sys
sys.path[:0] = [{core!r}, {tests!r}]
import confirmation, controller as K, test_controller as T
K.LOOP_CMD = [sys.executable, {script!r}]
confirmation.make_client = lambda: T.FakeClient()
sys.exit(K.main(["run", "--plan", {plan!r}, "--poll-s", "0.1", "--max-inflight", "2", "--resume"]))
"""


def test_kill_minus_9_and_resume(world, tmp_path):
    root, plan = world(6, fake_slots=6, fake_sleep=0.3)
    driver = tmp_path / "driver.py"
    driver.write_text(DRIVER.format(core=str(K.HERE), tests=str(Path(__file__).parent),
                                    script=K.LOOP_CMD[1], plan=str(root / "plan.json")))
    p = subprocess.Popen([sys.executable, str(driver)], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    state = root / "controller_state.json"
    deadline = time.time() + 30
    while time.time() < deadline:
        st = confirmation.read_json(state)
        if st and sum(ls["status"] == "done" for ls in st["loops"].values()) >= 2:
            break
        time.sleep(0.03)
    p.kill()  # SIGKILL: no cleanup of any kind
    p.wait()
    before = confirmation.read_json(state)
    assert sum(ls["status"] == "done" for ls in before["loops"].values()) >= 2
    r = subprocess.run([sys.executable, str(driver)], capture_output=True, text=True, timeout=60)  # lock was released
    assert r.returncode == 0, r.stderr
    st = json.loads(state.read_text())
    assert {ls["status"] for ls in st["loops"].values()} == {"done"} and st["start"] == before["start"]
    assert all(len(launches(root, l["loop_id"])) == 1 for l in plan["loops"])  # none relaunched or restarted
    assert "adopted" in (root / "controller.log").read_text()
    assert st["sentinels_submitted"] == 36 // 12


def test_resume_relaunches_a_loop_that_died_with_the_controller(world):
    root, plan = world(2)
    c = ctl(root, FakeClient(), max_inflight=2)
    c.init_state(False)
    lid = plan["loops"][0]["loop_id"]
    c.st["loops"][lid].update(status="running", pid=2 ** 22 + 1)  # recorded as running, process gone
    run_dir = root / lid
    (run_dir / "state.json").write_text(json.dumps({"slot_index": 1, "status": "running"}))
    c.save()
    again = ctl(root, FakeClient(), max_inflight=2)
    again.run(resume=True)
    assert [l["resume"] for l in launches(root, lid)] == [True]  # restarted, with --resume since state.json exists
    assert again.st["loops"][lid]["status"] == "done" and again.st["loops"][lid]["restarts"] == 1


# ---------------- restart wrapper ----------------
def wrapper(tmp_path, codes, extra_env=None):
    root = tmp_path / "w"
    root.mkdir(exist_ok=True)
    (root / "plan.json").write_text("{}")
    stub = tmp_path / "stub.sh"
    stub.write_text("#!/usr/bin/env bash\necho \"$@\" >> %s/calls\nn=$(wc -l < %s/calls)\ncodes=(%s)\nexit ${codes[$((n-1))]}\n"
                    % (root, root, " ".join(map(str, codes))))
    stub.chmod(0o755)
    env = dict(os.environ, PY=str(stub), SLEEP_S="0.05", **(extra_env or {}))
    return root, subprocess.run(["bash", str(WRAPPER), str(root / "plan.json"), "--max-inflight", "3"],
                                env=env, capture_output=True, text=True, timeout=30)


def test_wrapper_resumes_until_the_controller_exits_zero(tmp_path):
    root, r = wrapper(tmp_path, [1, 137, 0])
    assert r.returncode == 0
    calls = (root / "calls").read_text().splitlines()
    assert len(calls) == 3
    assert all("run --plan %s --resume --max-inflight 3" % (root / "plan.json") in c for c in calls)
    assert (root / "controller.log").read_text().count("resuming") == 2


def test_wrapper_gives_up_on_a_usage_error(tmp_path):
    root, r = wrapper(tmp_path, [2, 0])
    assert r.returncode == 2 and len((root / "calls").read_text().splitlines()) == 1


def test_wrapper_refuses_a_second_instance(tmp_path):
    root = tmp_path / "w"
    root.mkdir()
    with open(root / "wrapper.lock", "w") as held:
        fcntl.flock(held, fcntl.LOCK_EX)
        r = subprocess.run(["bash", str(WRAPPER), str(root / "plan.json")], capture_output=True, text=True, timeout=30)
    assert r.returncode == 1 and "already running" in r.stderr
