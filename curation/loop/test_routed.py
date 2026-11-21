"""Routing policies of the curation loop, on the fake models and trainer of test_engine.py."""
import json, sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import routed as R
from test_engine import FakeModels, FakeRunner, GOOD


def loop(tmp_path, policy="fixed:cheap,cheap", slots=6, harm=0.0, models=None, runner=None):
    cfg = {"loop_id": "L", "run_dir": str(tmp_path), "cell": "X", "pool": "v2", "slots": slots, "seed": 1, "sigma": 0.02,
           "gpu_host": None, "policy": policy, "strong": "strong", "cheap": "cheap", "harm": harm, "routing": None}
    return R.RoutedLoop(cfg, models=models or FakeModels(), runner=runner or FakeRunner())


def routes(tmp_path):
    return [json.loads(l) for l in open(tmp_path / "routes.jsonl")]


def test_state():
    s = R.state_of(["accept", "reject", "fail", "no_idea"], 5, 12)
    assert s == {"slot": 5, "left": 8, "stall": 3, "fails": 2, "n_accept": 1}
    assert R.state_of([], 1, 12)["stall"] == 0


def test_fixed_and_first(tmp_path):
    lp = loop(tmp_path / "a", "fixed:strong,cheap", slots=2)
    lp.run()
    assert [(r["planner"], r["coder"]) for r in routes(tmp_path / "a")] == [("strong", "cheap")] * 2
    lp = loop(tmp_path / "b", "first", slots=3)
    lp.run()
    assert [r["planner"] for r in routes(tmp_path / "b")] == ["strong", "cheap", "cheap"]
    assert [m for role, m, _, _ in lp.models.calls if role == "coder"] == ["strong", "cheap", "cheap"]


def test_escalate_after_k_stalled_slots(tmp_path):
    # every slot is rejected: cheap, cheap, strong, cheap, cheap, strong
    loop(tmp_path, "escalate:2").run()
    assert [r["planner"] for r in routes(tmp_path)] == ["cheap", "cheap", "strong", "cheap", "cheap", "strong"]
    # an accept in slot 2 resets the count
    d = tmp_path / "b"
    loop(d, "escalate:2", runner=FakeRunner(dev={2: 1.90})).run()
    assert [r["planner"] for r in routes(d)] == ["cheap", "cheap", "cheap", "cheap", "strong", "cheap"]


def test_random_is_seeded_and_mixed(tmp_path):
    loop(tmp_path / "a", "random", slots=12).run()
    loop(tmp_path / "b", "random", slots=12).run()
    a = [(r["planner"], r["coder"]) for r in routes(tmp_path / "a")]
    assert a == [(r["planner"], r["coder"]) for r in routes(tmp_path / "b")]
    assert len(set(a)) >= 3


def test_harm_replaces_half_of_the_files_by_the_incumbent(tmp_path):
    run = FakeRunner(dev={k: 1.90 for k in range(1, 13)})        # every real file would be accepted
    lp = loop(tmp_path, "fixed:strong,strong", slots=12, harm=0.5, runner=run)
    lp.run()
    rs = routes(tmp_path)
    harmed = [r["harmed"] for r in rs]
    assert 3 <= sum(harmed) <= 9
    assert all((r["decision"] == "reject") == r["harmed"] or r["decision"] == "reject" for r in rs)
    assert all(r["usd"] > 0 for r in rs)                           # the harmed call is still paid for


def test_until_stops_strong_after_the_first_accept(tmp_path):
    loop(tmp_path, "until:1", slots=5, runner=FakeRunner(dev={3: 1.90})).run()
    assert [r["planner"] for r in routes(tmp_path)] == ["strong", "strong", "strong", "cheap", "cheap"]


def test_cut_off_slot_is_scored_again_after_resume(tmp_path):
    lp = loop(tmp_path, "fixed:strong,strong", slots=2)
    lp.run_slot()
    stale = tmp_path / "runs" / "s02_0_s11.json"                   # slot 2 was cut off after its run was stored
    stale.write_text(json.dumps({"dev_bpb": 1.0}))
    loop(tmp_path, "fixed:strong,strong", slots=2).run_slot()
    assert not stale.exists()


def test_a_result_the_gpu_host_wrote_but_this_host_cannot_see_yet_is_fetched(tmp_path, monkeypatch):
    seen, answer = [], {"rc": 0}

    def fake_run(cmd, **kw):
        seen.append(cmd)
        if "cat " in cmd[-1]:                      # the check on the GPU host
            return R.SimpleNamespace(returncode=answer["rc"], stdout='{"dev_bpb": 1.9}', stderr="")
        return R.SimpleNamespace(returncode=0, stdout="", stderr="")   # the run: exit 0, no file visible here

    monkeypatch.setattr(R.E.subprocess, "run", fake_run)
    monkeypatch.setattr(R.E, "SCRATCH", str(tmp_path / "scratch"))
    lp = loop(tmp_path / "a")
    lp.cfg["gpu_host"] = "farhost"
    assert lp.train(tmp_path / "x.py", 11, "s01_1") == ({"dev_bpb": 1.9}, None)
    assert [c[3] for c in seen] == ["farhost", "farhost"] and seen[1][-1].endswith("runs/s01_1_s11.json")
    answer["rc"] = 1                               # the host has no result either: a real failure
    assert lp.train(tmp_path / "x.py", 12, "s01_1") == (None, "run failed (exit 0)")
