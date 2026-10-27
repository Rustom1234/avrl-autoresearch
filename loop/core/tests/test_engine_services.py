"""services.py wrappers and run_loop.py, with the network, hannah and the claude CLI replaced by fakes."""
import json
from types import SimpleNamespace

import pytest

import run_loop
import services
from fakes import BASE, FakeClock, FakeGPUService, FakeModelService, FakeServices, Infra, reply


class QCPlain:
    """QueueClient as it is today: no kind or gpu_affinity parameters."""

    def __init__(self):
        self.calls = []

    def submit(self, path, *, seed=None, sealed=False, timeout_s=600, priority=0, tag="", meta=None):
        self.calls.append(dict(path=path, seed=seed, sealed=sealed, timeout_s=timeout_s, tag=tag, meta=meta))
        return "j1"

    def wait(self, ids, timeout_s, poll_interval=20):
        return {ids[0]: {"job_id": ids[0], "status": "ok"}}


class QCFull(QCPlain):
    def submit(self, path, *, seed=None, sealed=False, timeout_s=600, priority=0, tag="", meta=None, kind="train",
               gpu_affinity=None):
        self.calls.append(dict(kind=kind, gpu_affinity=gpu_affinity, seed=seed, meta=meta))
        return "j2"


def test_gpu_service_adapts_to_the_queue_client_signature():
    plain, full = QCPlain(), QCFull()
    assert services.GPUService(plain, job_timeout_s=500).submit("/x/train.py", seed=7, sealed=True, tag="L:audit:3",
                                                                kind="audit") == "j1"
    assert plain.calls[0] == dict(path="/x/train.py", seed=7, sealed=True, timeout_s=500, tag="L:audit:3",
                                  meta={"kind": "audit", "gpu_affinity": None})
    services.GPUService(full).submit("/x/train.py", seed=1, sealed=False, tag="t", kind="replicate", gpu_affinity=2)
    assert full.calls[0]["kind"] == "replicate" and full.calls[0]["gpu_affinity"] == 2


def test_gpu_service_wait_returns_result_none_on_timeout_and_infra_fail_on_transport_error():
    svc = services.GPUService(QCPlain())
    assert svc.wait("j1", 10) == {"job_id": "j1", "status": "ok"}

    class Timeout(QCPlain):
        def wait(self, ids, timeout_s, poll_interval=20):
            return {ids[0]: None}

    class Broken(QCPlain):
        def wait(self, ids, timeout_s, poll_interval=20):
            raise RuntimeError("ssh died")

    assert services.GPUService(Timeout()).wait("j1", 1) is None
    r = services.GPUService(Broken()).wait("j1", 1)
    assert r["status"] == "infra_fail" and "ssh died" in r["error"] and "job_id" not in r


def test_model_service_tags_logs_and_wraps_the_call_layer(tmp_path, monkeypatch):
    svc = services.ModelService(tmp_path, "loopX")
    seen = []

    def fake_call(model, user, **kw):
        seen.append((model, user, kw))
        return SimpleNamespace(ok=True, obj={"vote": "yes"}, model=model)

    monkeypatch.setattr(svc._m, "call", fake_call)
    monkeypatch.setattr(svc._c, "call", fake_call)
    r = svc.call("coder", "sonnet", "hi", system="sys", schema={"type": "object"}, slot=3, temperature=0.5,
                 extra={"a": 1})
    model, user, kw = seen[0]
    assert (model, user) == ("sonnet", "hi") and r.ok
    assert kw["tag"] == "loopX:coder:3" and kw["log_path"] == str(tmp_path / "calls.jsonl")
    assert kw["system"] == "sys" and kw["temperature"] == 0.5 and kw["extra"] == {"a": 1}
    out = svc.sample_k("voter", "haiku", "q", 3, "vote", schema={"type": "object"}, slot=4)
    assert out["values"] == ["yes"] * 3 and seen[-1][2]["tag"] == "loopX:voter:4"
    assert svc.is_open("qwen35-9b") and not svc.is_open("sonnet") and svc.fallback_of("qwen35-9b") == "haiku"


def test_model_service_turns_context_overflow_into_a_failed_result(tmp_path, monkeypatch):
    svc = services.ModelService(tmp_path, "loopX")

    def boom(*a, **k):
        raise svc._m.ContextTooLong("too long")

    monkeypatch.setattr(svc._m, "call", boom)
    r = svc.call("coder", "qwen35-9b", "x")
    assert not r.ok and r.error_class == "context" and "too long" in r.error


def test_clock_is_real_time():
    c = services.Clock()
    assert c.now() > 1.7e9


# ------------------------------------------------------------------ run_loop

def write_config(tmp_path, **over):
    cfg = {"loop_id": "cli", "arm": "A", "run_dir": str(tmp_path / "run"), "n_slots": 2, "rng_seed": 1,
           "routing": {}, "sigma_used": 0.0067, "gpu_offsets": {"0": 0.0, "1": 0.001},
           "anchor": {"mean": 1.1436, "n": 24}, "base_train_py": str(BASE), "stop_file": str(tmp_path / "STOP"),
           "free_action_caps": {"slot": 2, "run": 40}, "critic_model": "opus", "voter_model": "haiku", "vote_k": 3,
           "timeout_s": 60}
    cfg.update(over)
    path = tmp_path / "config.json"
    path.write_text(json.dumps(cfg))
    return str(path)


def fake_services():
    return FakeServices(FakeModelService({"base_step": [reply(-0.02), reply(0.01)]}), FakeGPUService(noise_sd=0),
                        FakeClock())


def test_cli_runs_a_loop_to_done_and_guards_the_state_file(tmp_path, capsys):
    cfg = write_config(tmp_path)
    assert run_loop.main(["--config", cfg], services=fake_services()) == 0
    state = json.loads((tmp_path / "run" / "state.json").read_text())
    assert state["status"] == "done" and state["slot_index"] == 2 and state["arm"] == "A"
    assert "done after 2/2 slots" in capsys.readouterr().out
    assert run_loop.main(["--config", cfg], services=fake_services()) == 2           # would clobber a loop
    assert run_loop.main(["--config", cfg, "--resume"], services=fake_services()) == 0   # nothing left to do


def test_cli_resume_without_a_state_file_starts_fresh(tmp_path):
    assert run_loop.main(["--config", write_config(tmp_path), "--resume"], services=fake_services()) == 0


def test_cli_exit_codes_for_stop_file_infra_and_bad_config(tmp_path):
    (tmp_path / "STOP").write_text("")
    cfg = write_config(tmp_path)
    assert run_loop.main(["--config", cfg], services=fake_services()) == 0          # the controller's stop is a normal exit
    assert json.loads((tmp_path / "run" / "state.json").read_text())["status"] == "stopped"
    (tmp_path / "STOP").unlink()
    broken = FakeServices(FakeModelService({"base_step": [Infra()] * 9}), FakeGPUService(), FakeClock())
    (tmp_path / "b").mkdir()
    cfg2 = write_config(tmp_path / "b", run_dir=str(tmp_path / "b" / "run"), max_infra_retries=2)
    assert run_loop.main(["--config", cfg2], services=broken) == 3                  # infra outlasted the retries
    bad = tmp_path / "bad.json"
    bad.write_text(json.dumps({"loop_id": "x"}))
    with pytest.raises(SystemExit) as e:
        run_loop.main(["--config", str(bad)], services=fake_services())
    assert "arm" in str(e.value)


def test_cli_reads_the_controllers_config_shape(tmp_path):
    """The controller writes anchor_file and a routing path instead of anchor, gpu_offsets, sigma_used and routing."""
    anchor = {"anchor_mean": 1.1436, "n": 24, "gpu_offsets": {"0": -0.003, "1": 0.005}, "sigma_used": 0.006736}
    (tmp_path / "anchor.json").write_text(json.dumps(anchor))
    (tmp_path / "routing_table.json").write_text(json.dumps({"voter": "gemma4-12b", "coder_by_class": {}}))
    cfg_path = write_config(tmp_path, anchor_file=str(tmp_path / "anchor.json"), routing=str(tmp_path / "routing_table.json"))
    cfg = json.loads(open(cfg_path).read())
    for k in ("anchor", "gpu_offsets", "sigma_used", "base_train_py"):
        cfg.pop(k)
    open(cfg_path, "w").write(json.dumps(cfg))
    loaded = run_loop.load_config(cfg_path)
    assert loaded["anchor"] == {"mean": 1.1436, "n": 24} and loaded["sigma_used"] == 0.006736
    assert loaded["gpu_offsets"] == {"0": -0.003, "1": 0.005} and loaded["routing"]["voter"] == "gemma4-12b"
    assert loaded["base_train_py"].endswith("substrate/base/train_ampere.py")
    cfg["routing"] = None
    open(cfg_path, "w").write(json.dumps(cfg))
    assert run_loop.load_config(cfg_path)["routing"] == {}
    assert run_loop.main(["--config", cfg_path], services=fake_services()) == 0


def test_build_services_wires_the_real_classes_without_touching_the_network(tmp_path, monkeypatch):
    cfg = json.loads(open(write_config(tmp_path)).read())
    made = {}

    class FakeQC:
        def __init__(self, **kw):
            made["queue"] = kw

        def submit(self, path, *, seed=None, sealed=False, timeout_s=600, priority=0, tag="", meta=None):
            raise AssertionError("no network in tests")

    services._path("substrate")
    import queue_client
    monkeypatch.setattr(queue_client, "QueueClient", FakeQC)
    svc = run_loop.build_services({**cfg, "queue": {"host": "u@h", "root": "/r"}, "job_timeout_s": 700})
    assert made["queue"] == {"host": "u@h", "root": "/r"} and svc.gpu.job_timeout_s == 700
    assert isinstance(svc.models, services.ModelService) and isinstance(svc.clock, services.Clock)
    assert svc.models.log_path == str(tmp_path / "run" / "calls.jsonl")
