"""Cell N (random search, no model): the moves, and the engine run on the fake GPU service."""
import json
import re

import pytest

import random_search as RS
from fakes import BASE, FakeClock, FakeGPUService, FakeServices, Killed, make_config
from linter import lint


class BatchGPU(FakeGPUService):
    """Fake worker whose true val_bpb depends on one setting: 0.02 better when TOTAL_BATCH_SIZE is 2**18."""
    def submit(self, train_py_path, **kw):
        job_id = super().submit(train_py_path, **kw)
        res = self.jobs[job_id]
        src = open(train_py_path).read()
        if res.get("official_val_bpb") is not None and RS.read_value(src, "TOTAL_BATCH_SIZE") == 2 ** 18:
            res["official_val_bpb"] -= 0.02
        return job_id


def engine(tmp_path, gpu=None, **over):
    cfg = make_config(tmp_path, "D", n_slots=20, **over)
    svc = FakeServices(None, gpu or FakeGPUService(noise_sd=0.0), FakeClock())
    return RS.RandomSearchEngine(cfg, svc), svc


def test_moves_on_base():
    src = BASE.read_text()
    assert len(RS.MOVES) == 16
    for name, direction in RS.MOVES:
        new, old, val = RS.apply_move(src, name, direction)
        assert RS.read_value(new, name) == val != old
        compile(new, "train.py", "exec")
        assert lint(new, src).ok, (name, direction, lint(new, src).hard)
        # nothing but the setting (and the per-device batch that follows the total batch) changes
        changed = [a for a, b in zip(src.splitlines(), new.splitlines()) if a != b]
        assert 1 <= len(changed) <= 2 and len(src.splitlines()) == len(new.splitlines())
    assert RS.apply_move(src, "EMBEDDING_LR", "down")[2] == 0.3
    assert RS.apply_move(src, "DEPTH", "up")[2] == 10
    assert RS.apply_move(src, "WARMDOWN_RATIO", "up")[2] == 0.7


def test_clips_and_batch_assert():
    src = BASE.read_text()
    for _ in range(2):
        src = RS.apply_move(src, "WARMDOWN_RATIO", "up")[0]
    assert RS.read_value(src, "WARMDOWN_RATIO") == 0.9
    assert RS.apply_move(src, "WARMDOWN_RATIO", "up") is None       # clipped: not a move
    src = BASE.read_text()
    for _ in range(3):
        src = RS.apply_move(src, "DEPTH", "down")[0]
    assert RS.read_value(src, "DEPTH") == 2 and RS.apply_move(src, "DEPTH", "down") is None
    src = BASE.read_text()
    while True:   # the script's own assert must hold at every batch size on the way down
        total, dev = RS.read_value(src, "TOTAL_BATCH_SIZE"), RS.read_value(src, "DEVICE_BATCH_SIZE")
        assert total % (dev * RS.MAX_SEQ_LEN) == 0 and 1 <= dev <= 64
        r = RS.apply_move(src, "TOTAL_BATCH_SIZE", "down")
        if r is None:
            break
        src = r[0]
    assert total == RS.MAX_SEQ_LEN
    up = RS.apply_move(src, "TOTAL_BATCH_SIZE", "up")[0]
    assert RS.read_value(up, "DEVICE_BATCH_SIZE") == 2


def test_run_has_no_model_calls_and_no_repeats(tmp_path):
    eng, svc = engine(tmp_path)
    assert eng.run() == "done" and eng.state["slot_index"] == 20
    recs = eng.records
    assert all(not r.get("model_calls") for r in recs)
    assert all(r["decision"] == "reject" for r in recs) and len(recs) == 20   # noise 0, no move helps
    # 16 distinct moves on the unchanged incumbent, then the list is cleared and four more are drawn
    moves = [re.search(r"random search: (\S+ \S+),", r["description"]).group(1) for r in recs]
    assert len(set(moves[:16])) == 16 and len(set(moves[16:])) == 4
    assert json.loads((eng.run_dir / "state.json").read_text())["status"] == "done"


def test_good_move_is_kept(tmp_path):
    eng, svc = engine(tmp_path, gpu=BatchGPU(noise_sd=0.0))
    assert eng.run() == "done"
    assert RS.read_value(eng._inc_src(), "TOTAL_BATCH_SIZE") == 2 ** 18
    assert eng.state["incumbent"]["mean_adjusted"] == pytest.approx(1.1436 - 0.02, abs=1e-6)
    assert sum(r["decision"] == "accept" for r in eng.records) == 1


def test_crash_costs_one_slot_and_no_fix(tmp_path):
    gpu = FakeGPUService(noise_sd=0.0, patch={1: {"status": "crash", "official_val_bpb": None}})
    eng, svc = engine(tmp_path, gpu=gpu)
    eng.run_one_slot()
    assert eng.state["slot_index"] == 1 and eng.state["cand"] is None
    assert [r["decision"] for r in eng.records] == ["crash"]
    eng.run_one_slot()
    assert eng.records[1]["kind"] == "candidate" and eng.records[1]["decision"] == "reject"
    assert eng.records[0]["description"] != eng.records[1]["description"]   # the crashed move is not drawn again


def test_resume_after_kill(tmp_path):
    ref, _ = engine(tmp_path / "ref")
    ref.run()
    gpu = FakeGPUService(noise_sd=0.0, kill_on_wait=5)
    eng, svc = engine(tmp_path / "killed", gpu=gpu)
    with pytest.raises(Killed):
        eng.run()
    eng2 = RS.RandomSearchEngine(eng.cfg, svc)
    assert eng2.run() == "done" and len(gpu.submits) == 20
    assert [r["description"] for r in eng2.records] == [r["description"] for r in ref.records]
