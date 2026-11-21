"""The added parts of the R1 analysis, on made-up cells."""
import json, sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import analysis_r1 as R1

CELLS = {"SS": [(0.10, 10.0)] * 4, "CC": [(0.00, 0.0)] * 4, "SC": [(0.10, 2.0)] * 4, "LA": [(0.10, 1.0)] * 4,
         "LB": [(0.05, 1.0), (0.15, 1.0), (0.10, 1.0), (0.10, 1.0)]}


def test_difference_is_the_gap_between_two_advantages():
    d = R1.differences(CELLS, n_boot=300)
    assert abs(d["LA-SC"]["diff"] - 0.1) < 1e-9          # same gain, one tenth of the anchor cost gap cheaper
    assert all(abs(x - 0.1) < 1e-9 for x in d["LA-SC"]["ci"])            # no spread in these cells, so none in the interval
    assert abs(d["LB-SC"]["diff"] - 0.1) < 1e-9 and d["LB-SC"]["ci"][0] < 0.1 < d["LB-SC"]["ci"][1]
    assert abs(d["LA-LB"]["diff"]) < 1e-9


def test_pairs_with_a_missing_cell_are_skipped():
    assert set(R1.differences({k: v for k, v in CELLS.items() if k != "LB"}, n_boot=50)) == {"LA-SC"}


def test_route_counts_strong_slots_and_finds_the_top(tmp_path):
    (tmp_path / "config.json").write_text(json.dumps({"strong": "fable", "cheap": "gemma"}))
    rows = [{"slot": 1, "planner": "fable", "coder": "fable", "gain_sd": 0.0}, {"slot": 2, "planner": "fable", "coder": "gemma", "gain_sd": 4.0},
            {"slot": 3, "planner": "gemma", "coder": "gemma", "gain_sd": 15.0}, {"slot": 4, "planner": "gemma", "coder": "gemma", "gain_sd": 15.0}]
    (tmp_path / "routes.jsonl").write_text("".join(json.dumps(r) + "\n" for r in rows))
    assert R1.route(tmp_path) == {"strong_planner": 2, "strong_coder": 1, "first_slot_at_top": 3, "slots": 4}


def test_rescored_uses_four_seeds(tmp_path):
    loops = [{"id": "LA_01", "cell": "LA", "sealed": 1.90, "gain": 0.10, "usd": 1.0, "lost_slot": False},
             {"id": "SC_01", "cell": "SC", "sealed": 1.92, "gain": 0.08, "usd": 2.0, "lost_slot": False}]
    for seed, v in ((1003, 1.94), (1004, 1.98)):
        (tmp_path / ("R1final_LA_01_s%d.json" % seed)).write_text(json.dumps({"sealed_bpb": v}))
    out, n = R1.rescored(loops, tmp_path, raw_sealed=2.0)
    assert n == 1 and abs(out[0]["sealed"] - 1.93) < 1e-9 and abs(out[0]["gain"] - 0.07) < 1e-9
    assert out[1] == loops[1] and loops[0]["sealed"] == 1.90           # no extra runs: unchanged; input not modified
    assert R1.by_cell(out) == {"LA": [(out[0]["gain"], 1.0)], "SC": [(0.08, 2.0)]}


def test_rescore_jobs(tmp_path):
    import rescore_finals
    (tmp_path / "LA_01").mkdir()
    (tmp_path / "LA_01" / "final.json").write_text(json.dumps({"incumbent": "/x/cand.py"}))
    (tmp_path / "LA_02").mkdir()                                       # unfinished loop: no job
    assert rescore_finals.jobs(tmp_path, "R1final_") == ["R1final_LA_01=/x/cand.py"]
