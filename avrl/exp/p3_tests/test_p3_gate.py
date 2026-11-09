"""Day 0 gate numbers, level loading and frame building of p3_effort_analyze, from rows written by the real runner code."""
import pytest

import p3_effort_analyze as A
import p3_effort_run as P
from p3_helpers import fake_result, good_output, make_items

TOKENS = {"default": 600, "low": 300, "medium": 600, "high": 900}


def write(run_dir, items, tokens=TOKENS, bad=()):
    def fake(model, user, **kw):
        lv = kw["effort"] or "default"
        if (user, lv) in bad:
            return fake_result(obj={"problems": []}, out_tokens=tokens[lv])
        return fake_result(obj=good_output(70), out_tokens=tokens[lv])

    P.run_sweep(items, P.LEVELS, run_dir, 10_000, workers=1, call_fn=fake)


def test_gate_passes_with_token_spread_and_valid_output(tmp_path):
    items = make_items(20)
    write(tmp_path, items)
    g = A.day0_gate(tmp_path, items)
    assert g["median_ratio_high_over_low"] == pytest.approx(3.0)
    assert g["validity_all"] == 1.0 and g["pass"] is True
    assert g["per_level"]["high"]["median_out_tokens"] == 900 and g["per_level"]["low"]["n_valid"] == 20
    assert g["total_cost_usd"] == pytest.approx(80 * 0.002)


def test_gate_fails_without_token_spread(tmp_path):
    items = make_items(20)
    write(tmp_path, items, tokens={"default": 300, "low": 300, "medium": 310, "high": 380})  # ratio 1.27
    g = A.day0_gate(tmp_path, items)
    assert g["median_ratio_high_over_low"] == pytest.approx(380 / 300) and g["pass"] is False


def test_gate_fails_below_95_percent_valid(tmp_path):
    items = make_items(20)
    bad = {(items[i]["text"], "high") for i in range(2)}  # 2 of 80 invalid: 97.5 percent, still a pass
    write(tmp_path, items, bad=bad)
    assert A.day0_gate(tmp_path, items)["pass"] is True
    bad = {(items[i]["text"], lv) for i in range(3) for lv in ("high", "low")}  # 6 of 80 invalid: 92.5 percent
    write(tmp_path / "b", items, bad=bad)
    g = A.day0_gate(tmp_path / "b", items)
    assert g["validity_all"] == pytest.approx(74 / 80) and g["pass"] is False
    assert g["per_level"]["high"]["n_valid"] == 17


def test_default_is_matched_to_the_closest_level(tmp_path):
    items = make_items(20)
    write(tmp_path, items, tokens={"default": 560, "low": 300, "medium": 600, "high": 900})
    g = A.day0_gate(tmp_path, items)
    assert g["default_closest_level"] == "medium" and g["default_within_25pct_of_closest"] is True
    assert g["default_vs_levels"]["medium"]["median_ratio_default_over_level"] == pytest.approx(560 / 600)
    write(tmp_path / "c", items, tokens={"default": 1500, "low": 300, "medium": 600, "high": 900})
    g = A.day0_gate(tmp_path / "c", items)
    assert g["default_closest_level"] == "high" and g["default_within_25pct_of_closest"] is False


def test_load_level_takes_the_first_valid_row_and_skips_invalid(tmp_path):
    items = make_items(1)
    P.run_sweep(items, ["low"], tmp_path, 5, workers=1, call_fn=lambda *a, **k: fake_result(obj={"problems": []}))
    P.run_sweep(items, ["low"], tmp_path, 5, workers=1, call_fn=lambda *a, **k: fake_result(obj=good_output(40), cost=0.5))
    P.run_sweep(items, ["low"], tmp_path, 5, workers=1, call_fn=lambda *a, **k: fake_result(obj=good_output(90)))  # already done: not called
    r = A.load_level(tmp_path, "low")["it00"]
    assert r["structured_output"]["judgment_confidence"] == 40
    st = A.call_stats(tmp_path, "low", {"it00"})
    assert st["rows"] == 2 and st["invalid_rows"] == 1 and st["items_with_valid_row"] == 1 and st["total_cost_usd"] == pytest.approx(0.502)


def test_build_frame_keeps_only_complete_cases_and_scores_with_k1(tmp_path):
    items = make_items(3)  # it00 control, it01 N-big, it02 N-mid
    outs = {"it00": good_output(80), "it01": good_output(60, lines=[5]), "it02": good_output(50, lines=[9])}
    P.run_sweep(items, ["low", "high"], tmp_path, 100, workers=1,
                call_fn=lambda m, u, **kw: fake_result(obj=outs[[it["item_id"] for it in items if it["text"] == u][0]]))
    sonnet = {iid: ({"lines": {5}}, 0.012) for iid in ("it00", "it01")}  # it02 has no Sonnet pass 1
    weights = {"it00": 0.02, "it01": 0.001, "it02": 0.001}
    fr = A.build_frame(items, {lv: A.load_level(tmp_path, lv) for lv in ("low", "high")}, sonnet, weights, ["low", "high"])
    assert list(fr.item_id) == ["it00", "it01"]
    assert list(fr.cor_low) == [1.0, 1.0]  # control: no lines; N-big flagged at 5
    assert list(fr.s_cor) == [0.0, 1.0]  # Sonnet flagged line 5 on the control (false alarm) and hit the plant
    assert list(fr.conf_high) == [80.0, 60.0] and list(fr.tok_low) == [100.0, 100.0] and list(fr.w) == [0.02, 0.001]


@pytest.mark.skipif(not (P.AR3_DIR / "items.parquet").exists(), reason="AR3 data not present")
def test_main_runs_end_to_end_on_the_real_day0_items_with_mocked_rows(tmp_path):
    items, _ = P.select_items()
    d0 = P.day0_items(items)
    P.run_sweep(d0, P.LEVELS, tmp_path, 1000, workers=1,
                call_fn=lambda m, u, **kw: fake_result(obj=good_output(65, lines=[3]), out_tokens=TOKENS[kw["effort"] or "default"]))
    out = tmp_path / "analysis.json"
    res = A.main(["--run-dir", str(tmp_path), "--items", "day0", "--splits", "burn,test", "--B", "20", "--out", str(out), "--day0-gate"])
    import json
    saved = json.loads(out.read_text())
    assert saved["splits"]["burn"]["n_items"] == 20 and saved["splits"]["test"]["n_items"] == 0
    assert saved["day0_gate"]["pass"] is True and res["meta"]["calls"]["high"]["items_with_valid_row"] == 20
    assert saved["splits"]["burn"]["levels"]["low"]["accuracy"] is not None
