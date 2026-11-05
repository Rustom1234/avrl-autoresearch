import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from cost_report import price_calls, report  # noqa: E402

PRICES = {
    "source": "test",
    "per_million_tokens": {"cheap": {"input": 1.0, "output": 2.0}},
    "gpu_rental_per_hour": {"low": 0.1, "high": 0.2},
    "n_open_model_servers": 2,
}


def _write(path, rows):
    path.write_text("".join(json.dumps(r) + "\n" for r in rows))


def test_price_calls_splits_claude_and_open():
    rows = [
        {"served_model": "sonnet", "usage": {"input": 100, "output": 50}, "cost_usd_plan": 0.5, "ts": 0},
        {"served_model": "cheap", "usage": {"input": 1_000_000, "output": 500_000}, "latency_s": 3600, "ts": 7200},
    ]
    t = price_calls(rows, PRICES)
    assert t["claude_usd"] == 0.5 and t["claude_tokens"] == 150
    assert abs(t["open_usd"] - 2.0) < 1e-9  # 1 M input at 1.0 + 0.5 M output at 2.0
    assert t["open_calls"] == 1 and t["open_busy_s"] == 3600
    assert t["t_min"] == 0 and t["t_max"] == 7200


def test_report_per_arm_totals_and_rental(tmp_path):
    run = tmp_path / "run"
    for lid, arm, rows, decisions in [
        ("A_01", "A", [{"served_model": "sonnet", "usage": {"input": 10, "output": 10}, "cost_usd_plan": 1.0, "ts": 0}], ["accept", "reject"]),
        ("D_01", "D", [{"served_model": "haiku", "usage": {"input": 10, "output": 10}, "cost_usd_plan": 0.2, "ts": 0},
                       {"served_model": "cheap", "usage": {"input": 1_000_000, "output": 0}, "latency_s": 1800, "ts": 36000}],
         ["accept", "accept"]),
    ]:
        d = run / lid
        d.mkdir(parents=True)
        (d / "state.json").write_text(json.dumps({"loop_id": lid, "arm": arm}))
        _write(d / "calls.jsonl", rows)
        _write(d / "ledger.jsonl", [{"decision": x} for x in decisions])
    r = report(run, PRICES)
    a, dd = r["per_arm"]["A"], r["per_arm"]["D"]
    assert a["total_usd_per_run"] == 1.0 and a["total_usd_per_accept"] == 1.0
    assert abs(dd["total_usd_per_run"] - 1.2) < 1e-9 and abs(dd["total_usd_per_accept"] - 0.6) < 1e-9
    w = r["whole_run"]
    assert abs(w["open_busy_gpu_hours"] - 0.5) < 1e-9
    assert abs(w["run_window_hours"] - 10.0) < 1e-9
    assert [round(x, 6) for x in w["dedicated_rental_usd"]] == [2.0, 4.0]  # 10 h x 2 servers x 0.1 to 0.2
