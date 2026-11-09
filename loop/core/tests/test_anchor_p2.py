"""P2 anchor: arithmetic, refusals, and the hand-off to plan.json and run_loop."""
import json
import statistics
from pathlib import Path

import pytest

import anchor_p2 as A
import run_loop

R1 = Path(__file__).resolve().parents[2] / "runs" / "R1"
VALS = [1.110, 1.112, 1.114, 1.116, 1.118, 1.111, 1.113, 1.115, 1.117, 1.119, 1.112, 1.114, 1.116, 1.113, 1.115, 1.117]


def write_root(root, vals=VALS, extra=(), plan=True):
    root.mkdir(parents=True, exist_ok=True)
    rows = [{"k": i + 1, "job_id": "j%d" % i, "seed": 1000 + i, "status": "ok", "official_val_bpb": v, "gpu": i % 2,
             "attempt": 1} for i, v in enumerate(vals)] + list(extra)
    (root / "launch_sentinels.jsonl").write_text("".join(json.dumps(r) + "\n" for r in rows))
    if plan:
        (root / "plan.json").write_text(json.dumps({"seed": 2, "n_slots": 20, "sigma_used": None, "loops": []}))
    return root


def test_arithmetic_and_plan_update(tmp_path):
    root = write_root(tmp_path / "r", extra=[{"status": "crash", "official_val_bpb": None, "gpu": 3},
                                            {"status": "timeout", "official_val_bpb": 9.9, "gpu": 2}])
    assert A.main(["--root", str(root)]) == 0
    a = json.loads((root / "anchor.json").read_text())
    sd = statistics.stdev(VALS)
    assert a["n"] == 16 and a["anchor_mean"] == pytest.approx(statistics.fmean(VALS), abs=5e-7)
    assert a["sd"] == pytest.approx(sd, abs=5e-7) and a["sigma_used"] == pytest.approx(1.15 * sd, abs=5e-7)
    assert a["gpus"] == [0, 1] and a["gpu_offsets"] == {"0": 0.0, "1": 0.0}   # failed runs and their GPUs are not used
    assert a["offsets_rule"].startswith("zero: R1 offsets did not hold") and "1.15" in a["sigma_rule"]
    assert len(a["values"]) == 16 and a["values"][0] == {"gpu": 0, "val": VALS[0], "job_id": "j0", "seed": 1000}
    assert a["frozen_at"].endswith("Z")
    plan = json.loads((root / "plan.json").read_text())
    assert plan["sigma_used"] == a["sigma_used"] and plan["seed"] == 2 and plan["n_slots"] == 20   # other keys kept
    assert not list(root.glob(".*.tmp")) and not list(root.glob("*.tmp"))


def test_gpu_offsets_are_zero_for_every_gpu_seen(tmp_path):
    vals = [1.10 + 0.01 * (i % 4) for i in range(16)]   # a strong GPU effect on purpose
    root = write_root(tmp_path / "r", vals)
    rows = [json.loads(l) for l in (root / "launch_sentinels.jsonl").read_text().splitlines()]
    for r in rows:
        r["gpu"] = int(round((r["official_val_bpb"] - 1.10) / 0.01))
    (root / "launch_sentinels.jsonl").write_text("".join(json.dumps(r) + "\n" for r in rows))
    A.main(["--root", str(root)])
    a = json.loads((root / "anchor.json").read_text())
    assert a["gpus"] == [0, 1, 2, 3] and set(a["gpu_offsets"].values()) == {0.0}


def test_needs_min_n_ok_runs(tmp_path):
    root = write_root(tmp_path / "r", VALS[:13])
    with pytest.raises(SystemExit) as e:
        A.main(["--root", str(root)])
    assert e.value.code == 2 and not (root / "anchor.json").exists()
    assert json.loads((root / "plan.json").read_text())["sigma_used"] is None
    assert A.main(["--root", str(root), "--min-n", "13"]) == 0


def test_refuses_to_overwrite_and_leaves_files_alone(tmp_path):
    root = write_root(tmp_path / "r")
    A.main(["--root", str(root)])
    before = ((root / "anchor.json").read_text(), (root / "plan.json").read_text())
    with pytest.raises(SystemExit):
        A.main(["--root", str(root)])
    assert ((root / "anchor.json").read_text(), (root / "plan.json").read_text()) == before


def test_needs_a_plan_and_writes_no_anchor_without_it(tmp_path):
    root = write_root(tmp_path / "r", plan=False)
    with pytest.raises(SystemExit):
        A.main(["--root", str(root)])
    assert not (root / "anchor.json").exists()


def test_run_loop_reads_the_anchor(tmp_path):
    root = write_root(tmp_path / "r")
    A.main(["--root", str(root)])
    cfg = tmp_path / "cfg.json"
    cfg.write_text(json.dumps({"loop_id": "SS_01", "arm": "D", "run_dir": str(tmp_path / "SS_01"), "n_slots": 20,
                               "rng_seed": 1, "anchor_file": str(root / "anchor.json"), "routing": None}))
    c = run_loop.load_config(cfg)
    a = json.loads((root / "anchor.json").read_text())
    assert c["anchor"] == {"mean": a["anchor_mean"], "n": 16} and c["sigma_used"] == a["sigma_used"]
    assert c["gpu_offsets"] == {"0": 0.0, "1": 0.0}


@pytest.mark.skipif(not (R1 / "launch_sentinels.jsonl").exists(), reason="R1 data not present")
def test_reproduces_the_r1_numbers_apart_from_the_offsets(tmp_path):
    root = tmp_path / "r"
    root.mkdir()
    (root / "launch_sentinels.jsonl").write_text((R1 / "launch_sentinels.jsonl").read_text())
    (root / "plan.json").write_text("{}")
    A.main(["--root", str(root), "--min-n", "16"])
    a, r1 = json.loads((root / "anchor.json").read_text()), json.loads((R1 / "anchor.json").read_text())
    assert (a["anchor_mean"], a["sd"], a["sigma_used"], a["n"]) == (r1["anchor_mean"], r1["sd"], r1["sigma_used"], 16)
    assert set(a["gpu_offsets"]) == set(r1["gpu_offsets"]) and set(a["gpu_offsets"].values()) == {0.0}
