import json
import os
import re
import subprocess
import sys
import threading
import time

import pytest

import worker
from conftest import HERE, approx, expected_bpb, is_dead

E_OFFICIAL, E_SEALED = expected_bpb(1.25), expected_bpb(2.5)


# ---- claim ----
def test_claim_order_priority_then_age(env):
    w = env.make_worker()
    now = time.time()
    env.put_job("old_p0", mtime=now - 100, priority=0)
    env.put_job("new_p5", mtime=now - 10, priority=5)
    env.put_job("old_p5", mtime=now - 50, priority=5)
    env.put_job("newest_p0", mtime=now - 1, priority=0)
    order = []
    while (p := w.claim()) is not None:
        order.append(p.stem)
        assert p.parent.name == "running"
    assert order == ["old_p5", "new_p5", "old_p0", "newest_p0"]
    assert not list((env.root / "queue" / "pending").glob("*.json"))


def test_claim_prefers_own_affinity_then_none_then_foreign(env):
    w = env.make_worker(gpu=1)
    now = time.time()
    env.put_job("foreign_old", mtime=now - 100, gpu_affinity=0, priority=9)
    env.put_job("none_old", mtime=now - 90)
    env.put_job("mine_new", mtime=now - 10, gpu_affinity=1)
    env.put_job("none_new", mtime=now - 5)
    env.put_job("mine_old", mtime=now - 50, gpu_affinity=1)
    env.put_job("bad_aff", mtime=now - 1, gpu_affinity="x")  # not an int: counts as no affinity
    order = []
    while (p := w.claim()) is not None:
        order.append(p.stem)
    # a mismatched-affinity job is taken only when nothing else is pending, whatever its priority
    assert order == ["mine_old", "mine_new", "none_old", "none_new", "bad_aff", "foreign_old"]


def test_claim_stamps_claim_time_on_the_running_file(env):
    w = env.make_worker()
    env.put_job("a", mtime=time.time() - 3600)
    p = w.claim()
    assert abs(time.time() - p.stat().st_mtime) < 10


def test_claim_skips_job_stolen_by_another_worker(env, monkeypatch):
    w = env.make_worker()
    env.put_job("a", mtime=time.time() - 10)
    env.put_job("b", mtime=time.time() - 5)
    real_rename, calls = os.rename, []

    def flaky(src, dst):
        calls.append(src)
        if len(calls) == 1:
            raise FileNotFoundError(src)  # lost the race for "a"
        return real_rename(src, dst)

    monkeypatch.setattr(worker.os, "rename", flaky)
    assert w.claim().stem == "b"


def test_claim_ignores_tmp_files(env):
    w = env.make_worker()
    env.put_job("a")
    (env.root / "queue" / "pending" / ".b.json.tmp").write_text("{")
    assert w.claim().stem == "a"
    assert w.claim() is None


# ---- seed patching and seed binding ----
def test_patch_seed_counts():
    src = "torch.manual_seed(42)\n  torch.cuda.manual_seed( 42 )\nx = torch.manual_seed(1337)\n"
    out, n, ok = worker.patch_seed(src, 7)
    assert (n, ok) == (2, True)
    assert out == "torch.manual_seed(7)\n  torch.cuda.manual_seed(7)\nx = torch.manual_seed(1337)\n"
    assert worker.patch_seed("torch.manual_seed(42)\n", 7)[1:] == (1, False)
    assert worker.patch_seed("print(1)\n", 7)[1:] == (0, False)


def test_seed_end_to_end(env):
    r = env.run_job("s7", seed=7)
    assert (r["seed"], r["seed_patched"], r["seed_lines_matched"], r["seed_ok"]) == (7, True, 2, True)
    assert r["status"] == "ok" and r["harness"]["initial_seed"] == 7
    assert "SEEDS [7, 7]" in (env.root / "results" / "s7" / "run.log").read_text()
    r = env.run_job("snone", seed=None)
    assert (r["seed_patched"], r["seed_lines_matched"], r["seed_ok"]) == (False, 0, None)
    assert r["status"] == "ok"
    assert "SEEDS [42, 42]" in (env.root / "results" / "snone" / "run.log").read_text()


def _variant(env, name, old, new):
    p = env.tmp / name
    text = (HERE / "fake_train.py").read_text()
    assert old in text
    p.write_text(text.replace(old, new))
    return p


@pytest.mark.parametrize("old,new,matched", [
    ("torch.manual_seed(42)\ntorch.cuda.manual_seed(42)\n", "pass\npass\n", 0),
    ("torch.manual_seed(42)\ntorch.cuda.manual_seed(42)\n", "torch.manual_seed(42)\npass\n", 1),
])
def test_seed_lines_not_matching_twice_is_invalid_seed(env, old, new, matched):
    train = _variant(env, "noseed.py", old, new)
    r = env.run_job("is1", seed=7, train_py=train)
    assert r["seed_lines_matched"] == matched and r["seed_patched"] is False
    assert r["status"] == "invalid_seed"
    assert r["official_val_bpb"] == approx(E_OFFICIAL)  # the measurement itself is kept


def test_candidate_reseeding_after_the_patched_lines_is_invalid_seed(env, monkeypatch):
    monkeypatch.setenv("FAKE_MODE", "reseed")  # calls torch.manual_seed(999) later in the script
    r = env.run_job("rs1", seed=7)
    assert r["seed_lines_matched"] == 2 and r["seed_ok"] is False and r["harness"]["initial_seed"] == 999
    assert r["status"] == "invalid_seed"
    r = env.run_job("rs2", seed=None)  # no seed requested: nothing to bind
    assert r["status"] == "ok" and r["seed_ok"] is None


# ---- run outcomes ----
def test_status_ok_with_parsed_fields(env):
    w = env.make_worker(gpu=2)
    beats = []
    real = w.heartbeat
    w.heartbeat = lambda current_job=None: (beats.append(current_job), real(current_job))
    r = env.run_job("ok1", worker_obj=w, tag="mytag", meta={"k": 1})
    assert r["status"] == "ok" and r["exit_code"] == 0 and r["gpu"] == 2
    p = r["printed"]
    assert p.pop("val_bpb") == approx(E_OFFICIAL)
    assert p == {"training_seconds": 1.5, "total_seconds": 2.5, "peak_vram_mb": 1234.5,
                 "mfu_percent": 12.34, "total_tokens_M": 10.5, "num_steps": 42, "num_params_M": 50.3, "depth": 8}
    assert isinstance(p["num_steps"], int)
    h = r["harness"]
    assert h["official_val_bpb"] == approx(E_OFFICIAL) and h["sealed_val_bpb"] is None and h["error"] is None
    assert h["constants_ok"] is True and h["prepare_code_hash_ok"] is True
    assert h["pipeline_code_hash_ok"] is True and h["constants_bad"] == [] and h["pipeline_bad"] == []
    assert h["wall_from_AR_T0"] > 0 and h["harness_eval_seconds"] >= 0
    assert r["official_val_bpb"] == approx(E_OFFICIAL) and r["sealed_val_bpb"] is None
    assert r["mismatch_flag"] is False and r["harness_missing"] is False and r["integrity_failed"] == []
    assert r["eval_batch_size_used"] == 64 and r["target_leak_flag"] is False
    assert (r["tag"], r["meta"]) == ("mytag", {"k": 1})
    assert r["finished"] >= r["started"] and r["wall_seconds"] == pytest.approx(r["finished"] - r["started"])
    assert any("harness_official_val_bpb:" in line for line in r["log_tail"])
    assert r["data_source"] == "local" and r["staging_seconds"] >= 0 and r["overbudget"] is False
    assert 0 < r["startup_plus_training_s"] < 120 and r["retries"] == 0
    rd = env.root / "results" / "ok1"
    for name in ("run.log", "train.py", "harness_result.json", "result.json", "run/prepare.py", "run/train.py"):
        assert (rd / name).exists(), name
    assert (env.root / "queue" / "done" / "ok1.json").exists()
    assert not (env.root / "queue" / "running" / "ok1.json").exists()
    assert "ok1" in beats and beats[-1] is None
    hb = json.loads((env.root / "workers" / "gpu2.json").read_text())
    assert set(hb) == {"ts", "gpu", "host", "current_job", "pid", "child_pgid"} and hb["current_job"] is None


def test_child_launch_contract(env, monkeypatch):
    seen = []
    real_popen = worker.subprocess.Popen

    def spy(*a, **kw):
        seen.append(dict(kw["env"], cwd=str(kw["cwd"]), sess=kw["start_new_session"]))
        return real_popen(*a, **kw)

    monkeypatch.setattr(worker.subprocess, "Popen", spy)
    env.run_job("envj", worker_obj=env.make_worker(gpu=3), seed=5)
    env.run_job("envk", worker_obj=env.make_worker(gpu=3))
    s, k = seen
    assert s["CUDA_VISIBLE_DEVICES"] == "3" and s["TORCHINDUCTOR_CACHE_DIR"] == "/tmp/inductor_gpu3"
    assert s["HF_HUB_OFFLINE"] == "1" and s["TRANSFORMERS_OFFLINE"] == "1"
    assert s["HF_HOME"] == "/home/rustom/.cache/huggingface"  # explicit, since HOME moves
    assert s["HOME"] == str(env.stage)
    assert float(s["AR_T0"]) > 0 and "AR_SEALED_DIR" not in s and s["sess"] is True
    assert s["cwd"].endswith("results/envj/run")
    assert re.fullmatch(r"[0-9a-f]{16}", s["AR_NONCE"]) and s["AR_NONCE"] != k["AR_NONCE"]
    assert re.fullmatch(r"hr_[0-9a-f]{16}\.json", os.path.basename(s["AR_RESULT_PATH"]))
    assert os.path.dirname(s["AR_RESULT_PATH"]) == s["cwd"] and s["AR_RESULT_PATH"] != k["AR_RESULT_PATH"]
    assert s["AR_SEED"] == "5" and "AR_SEED" not in k


def test_crash_has_log_tail(env, monkeypatch):
    monkeypatch.setenv("FAKE_MODE", "crash")
    r = env.run_job("c1")
    assert r["status"] == "crash" and r["exit_code"] == 1
    assert any("RuntimeError: boom" in line for line in r["log_tail"])
    assert r["printed"] == {} and r["official_val_bpb"] is None
    assert r["harness_missing"] is True and r["harness"]["error"]
    assert (env.root / "queue" / "done" / "c1.json").exists()


@pytest.mark.parametrize("ignore_term", [False, True])
def test_timeout_kills_process_group(env, monkeypatch, ignore_term):
    monkeypatch.setenv("FAKE_MODE", "hang")
    if ignore_term:
        monkeypatch.setenv("FAKE_IGNORE_TERM", "1")
    w = env.make_worker(poll=0.5)
    w.kill_grace = 1.0
    t0 = time.time()
    r = env.run_job("h1", worker_obj=w, timeout_s=3)
    assert r["status"] == "timeout"
    assert 3 <= time.time() - t0 < 14
    assert r["exit_code"] in (-15, -9)
    grandchild = int((env.tmp / "child.pid").read_text())
    for _ in range(40):
        if is_dead(grandchild):
            break
        time.sleep(0.1)
    assert is_dead(grandchild)


def test_mismatch_flag(env, monkeypatch):
    monkeypatch.setenv("FAKE_MODE", "mismatch")
    r = env.run_job("m1")
    assert r["status"] == "ok" and r["mismatch_flag"] is True
    assert r["printed"]["val_bpb"] == pytest.approx(E_OFFICIAL + 0.01, abs=1e-5)
    assert r["official_val_bpb"] == approx(E_OFFICIAL)


def test_mismatch_helper():
    assert worker.mismatch(1.0, 1.00005) is False
    assert worker.mismatch(1.0, 1.0002) is True
    assert worker.mismatch(None, None) is False
    assert worker.mismatch(1.0, None) is True and worker.mismatch(None, 1.0) is True
    assert worker.mismatch(float("nan"), 1.0) is True


def test_tampered_evaluate_bpb_cannot_move_the_score(env, monkeypatch):
    monkeypatch.setenv("FAKE_MODE", "tamper")  # prepare.evaluate_bpb = lambda *a, **k: 0.1
    r = env.run_job("t1")
    h = r["harness"]
    assert r["status"] == "ok" and h["prepare_code_hash_ok"] is False
    assert h["official_via_prepare"] == 0.1  # the tampered function's answer
    assert h["official_val_bpb"] == approx(E_OFFICIAL)  # the harness scored from raw logits itself
    assert r["official_val_bpb"] is None and r["sealed_val_bpb"] is None  # and the flag still nulls the scores
    assert r["integrity_failed"] == ["prepare_code_hash_ok"]


# ---- sealed ----
def test_sealed_evaluation(env):
    r = env.run_job("sl1", sealed=True)
    assert r["sealed"] is True
    assert r["harness"]["official_val_bpb"] == approx(E_OFFICIAL)
    assert r["harness"]["sealed_val_bpb"] == approx(E_SEALED) and E_SEALED != pytest.approx(E_OFFICIAL)
    assert r["official_val_bpb"] == approx(E_OFFICIAL) and r["sealed_val_bpb"] == approx(E_SEALED)
    assert r["mismatch_flag"] is False


def test_sealed_dir_not_leaked_to_unsealed_jobs(env, monkeypatch):
    monkeypatch.setenv("AR_SEALED_DIR", str(env.src / "sealed"))  # stray value in the worker's own env
    r = env.run_job("sl2", sealed=False)
    assert r["sealed_val_bpb"] is None and r["harness"]["sealed_val_bpb"] is None


def test_sealed_dir_with_two_parquets_reports_error(env):
    (env.src / "sealed" / "shard_00001.parquet").write_text("9")
    r = env.run_job("sl3", sealed=True)
    assert r["status"] == "ok" and r["official_val_bpb"] == approx(E_OFFICIAL) and r["sealed_val_bpb"] is None
    assert "exactly one .parquet" in r["harness"]["error"]


# ---- receipt authentication (nonce, random result path, exit code 0) ----
def test_load_receipt_accepts_only_nonce_path_and_clean_exit(tmp_path):
    p = tmp_path / "r.json"
    p.write_text(json.dumps({"nonce": "abc", "official_val_bpb": 1.0}))
    assert worker.load_receipt(p, "abc", 0)["official_val_bpb"] == 1.0
    assert worker.load_receipt(p, "abd", 0) is None  # wrong nonce
    assert worker.load_receipt(p, "abc", 1) is None  # trainer did not exit 0
    assert worker.load_receipt(p, "abc", None) is None  # trainer would not die
    assert worker.load_receipt(p, "", 0) is None
    assert worker.load_receipt(tmp_path / "missing.json", "abc", 0) is None
    p.write_text("[1]")
    assert worker.load_receipt(p, "abc", 0) is None
    p.write_text("{not json")
    assert worker.load_receipt(p, "abc", 0) is None


@pytest.mark.parametrize("mode", ["wrongnonce", "nopath", "norecept"])
def test_unauthenticated_receipt_means_harness_missing(env, monkeypatch, mode):
    monkeypatch.setenv("FAKE_MODE", mode)
    r = env.run_job("rc_" + mode)
    assert r["status"] == "ok" and r["exit_code"] == 0  # train.py itself ran fine
    assert r["harness_missing"] is True
    assert r["official_val_bpb"] is None and r["sealed_val_bpb"] is None
    assert r["mismatch_flag"] is True  # train.py printed a score nothing could confirm


def test_receipt_from_a_trainer_that_exits_nonzero_is_rejected(env, monkeypatch):
    monkeypatch.setenv("FAKE_MODE", "exit3")  # atexit -> os._exit(3), after the tail wrote a good receipt
    r = env.run_job("ex3")
    assert r["exit_code"] == 3 and r["status"] == "crash"
    assert r["harness_missing"] is True and r["official_val_bpb"] is None
    assert json.loads((env.root / "results" / "ex3" / "harness_result.json").read_text())["official_val_bpb"]


def test_preexisting_file_at_the_result_path_is_deleted_before_launch(env, monkeypatch):
    monkeypatch.setenv("FAKE_MODE", "norecept")  # the trainer writes no receipt of its own
    monkeypatch.setattr(worker.secrets, "token_hex", lambda n: "ab" * n)  # fixed nonce and file name
    real_prepare_run = worker.Worker.prepare_run

    def plant(self, job_id, job, res):
        rdir, run = real_prepare_run(self, job_id, job, res)
        (run / ("hr_" + "ab" * 8 + ".json")).write_text(json.dumps({"nonce": "ab" * 8, "official_val_bpb": 0.0}))
        return rdir, run

    monkeypatch.setattr(worker.Worker, "prepare_run", plant)
    r = env.run_job("pre1")
    assert r["harness_missing"] is True and r["official_val_bpb"] is None  # the planted receipt did not survive


# ---- external time check ----
def test_overbudget_flags_status_and_keeps_scores(env, monkeypatch):
    r = env.run_job("ob0")
    assert r["overbudget"] is False and r["status"] == "ok" and 0 < r["startup_plus_training_s"] < worker.OVERBUDGET_S
    monkeypatch.setattr(worker, "OVERBUDGET_S", 0.0)
    r = env.run_job("ob1")
    assert r["overbudget"] is True and r["status"] == "overbudget"
    assert r["official_val_bpb"] == approx(E_OFFICIAL)
    assert r["startup_plus_training_s"] == pytest.approx(r["harness"]["tail_start_epoch"] - r["started"])


# ---- refusal and bad jobs ----
def test_prepare_hash_mismatch_refusal(env):
    with open(env.base / "prepare.py", "a") as fh:
        fh.write("\n# tampered\n")
    env.put_job("p1")
    w = env.make_worker()
    assert w.run() == 3  # refuses to start, queue untouched
    assert (env.root / "queue" / "pending" / "p1.json").exists()
    # and a job that reaches execute() with a bad base is refused too, without running anything
    res = w.process(w.claim())
    assert res["status"] == "error" and "sha256 mismatch" in res["error"]
    assert not (env.root / "results" / "p1" / "run.log").exists()
    assert json.loads((env.root / "results" / "p1" / "result.json").read_text())["status"] == "error"


def test_default_pin_used_when_no_pinned_hashes_file(env):
    (env.base / "pinned_hashes.json").unlink()  # fake prepare.py != real pinned sha256
    assert env.make_worker().run() == 3


@pytest.mark.parametrize("raw,needle", [("{not json", "JSONDecodeError"), ("[1, 2]", "not a JSON object")])
def test_invalid_job_file_gets_error_result(env, raw, needle):
    (env.root / "queue" / "pending").mkdir(parents=True)
    (env.root / "queue" / "pending" / "bad.json").write_text(raw)
    env.make_worker().run(once=True)
    r = json.loads((env.root / "results" / "bad" / "result.json").read_text())
    assert r["status"] == "error" and needle in r["error"]
    assert (env.root / "queue" / "done" / "bad.json").exists()


def test_missing_train_py_gets_error_result(env):
    r = env.run_job("nt", train_py="/nonexistent/train.py")
    assert r["status"] == "error" and "train_py" in r["error"]


# ---- STOP ----
def test_stop_file_exits_without_claiming(env):
    env.put_job("s1")
    w = env.make_worker()
    (env.root / "STOP").write_text("")
    assert w.run() == 0
    assert (env.root / "queue" / "pending" / "s1.json").exists()
    assert json.loads((env.root / "workers" / "gpu0.json").read_text())["current_job"] is None


def test_loop_processes_then_stops_on_stop_file(env):
    w = env.make_worker(poll=0.1)
    env.put_job("l1")
    codes = []
    th = threading.Thread(target=lambda: codes.append(w.run()))
    th.start()
    done = env.root / "queue" / "done" / "l1.json"
    for _ in range(100):
        if done.exists():
            break
        time.sleep(0.1)
    assert done.exists()
    (env.root / "STOP").write_text("")
    th.join(timeout=10)
    assert not th.is_alive() and codes == [0]


def test_once_processes_exactly_one_job(env):
    env.put_job("o1", mtime=time.time() - 10)
    env.put_job("o2")
    env.make_worker().run(once=True)
    assert (env.root / "queue" / "done" / "o1.json").exists()
    assert (env.root / "queue" / "pending" / "o2.json").exists()


def test_cli_entry_point(env):
    env.put_job("cli1")
    p = subprocess.run([sys.executable, str(worker.__file__), "--gpu", "1", "--root", str(env.root),
                        "--base", str(env.base), "--python", sys.executable, "--once",
                        "--sealed-dir", str(env.src / "sealed"), "--cache-src", str(env.src),
                        "--stage-root", str(env.stage)], capture_output=True, text=True, timeout=120)
    assert p.returncode == 0, p.stderr
    assert json.loads((env.root / "results" / "cli1" / "result.json").read_text())["gpu"] == 1
