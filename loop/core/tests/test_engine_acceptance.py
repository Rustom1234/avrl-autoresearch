"""acceptance.py: hand-computed numbers. sigma_used 0.0067, incumbent 1.1436 over 24 runs (the initial anchor).

Denominators: one candidate run  0.0067 * sqrt(1 + 1/24)    = 0.0067 * 1.020621 = 0.0068382
              two candidate runs 0.0067 * sqrt(1/2 + 1/24)  = 0.0067 * 0.735980 = 0.0049311
"""
import math

import pytest

import acceptance as acc

SIGMA = 0.0067
INC = {"mean_adjusted": 1.1436, "n": 24}
ONE = SIGMA * math.sqrt(1 + 1 / 24)
TWO = SIGMA * math.sqrt(0.5 + 1 / 24)


def test_denominators_match_hand_values():
    assert ONE == pytest.approx(0.0068382, abs=2e-7)
    assert TWO == pytest.approx(0.0049311, abs=2e-7)


def test_adjust_subtracts_gpu_offset_with_int_or_str_keys():
    off = {0: -0.00306, 1: 0.00474}
    assert acc.adjust(1.15, 1, off) == pytest.approx(1.14526)
    assert acc.adjust(1.15, 0, off) == pytest.approx(1.15306)
    assert acc.adjust(1.15, 1, {"1": 0.00474}) == pytest.approx(1.14526)
    assert acc.adjust(1.15, 3, off) == 1.15


def test_z_score_one_and_two_runs():
    assert acc.z_score(1.1436, 24, [1.1336], SIGMA) == pytest.approx(0.01 / ONE)   # 1.4624
    assert acc.z_score(1.1436, 24, [1.1336], SIGMA) == pytest.approx(1.4624, abs=2e-4)
    assert acc.z_score(1.1436, 24, [1.1336, 1.1336], SIGMA) == pytest.approx(2.0280, abs=2e-4)
    assert acc.z_score(1.1436, 24, [1.1386, 1.1286], SIGMA) == pytest.approx(2.0280, abs=2e-4)   # mean of the runs
    assert acc.z_score(1.1436, 24, [1.1536], SIGMA) < 0                                           # worse than incumbent
    assert acc.z_score(1.1436, 1, [1.1336], SIGMA) == pytest.approx(0.01 / (SIGMA * math.sqrt(2)))  # n_i = 1


@pytest.mark.parametrize("gain,expected", [
    (0.0050, "reject"),       # z1 0.7312
    (0.0052, "replicate"),    # z1 0.7605
    (0.0136, "replicate"),    # z1 1.9888
    (0.0137, "accept"),       # z1 2.0034
    (-0.0100, "reject"),
])
def test_decide_verified_first_run_bands(gain, expected):
    d, z = acc.decide_verified(INC, [1.1436 - gain], SIGMA)
    assert d == expected
    assert z == pytest.approx(gain / ONE)


@pytest.mark.parametrize("gain,expected", [(0.0073, "reject"), (0.0074, "accept")])   # z2 1.4804 and 1.5007
def test_decide_verified_after_replicate(gain, expected):
    d, z = acc.decide_verified(INC, [1.1436 - gain] * 2, SIGMA)
    assert d == expected
    assert z == pytest.approx(gain / TWO)


def test_decide_verified_uses_runs_behind_the_incumbent():
    inc = {"mean_adjusted": 1.1336, "n": 2}                      # an earlier accept with a replicate
    d, z = acc.decide_verified(inc, [1.1236], SIGMA)             # gain 0.01, denominator 0.0067 * sqrt(1.5) = 0.008206
    assert z == pytest.approx(0.01 / (SIGMA * math.sqrt(1.5)))
    assert z == pytest.approx(1.2186, abs=2e-4)
    assert d == "replicate"


def test_decide_base_is_strictly_lower():
    assert acc.decide_base(1.1436, 1.1436) == "reject"
    assert acc.decide_base(1.1436, 1.14359) == "accept"
    assert acc.decide_base(1.1436, 1.2) == "reject"


def test_tripwire_single_step_limit_is_5_sigma_sqrt2():
    limit = 5 * SIGMA * math.sqrt(2)
    assert limit == pytest.approx(0.04737, abs=1e-5)
    assert acc.tripwire(1.1436, 1.1436 - 0.0474, 1.0, SIGMA)
    assert not acc.tripwire(1.1436, 1.1436 - 0.0471, 1.0, SIGMA)


def test_tripwire_cumulative_gain_over_anchor():
    assert acc.tripwire(1.12, 1.1099, 1.1436, SIGMA)        # 0.0337 over the anchor, step only 0.0101
    assert not acc.tripwire(1.12, 1.1140, 1.1436, SIGMA)    # 0.0296 over the anchor
    assert not acc.tripwire(1.1436, 1.1436, 1.1436, SIGMA)


def clean():
    return {"status": "ok", "official_val_bpb": 1.14, "mismatch_flag": False, "target_leak_flag": False,
            "canary_ok": True, "harness_missing": False, "seed": 5, "seed_patched": True, "seed_lines_matched": 2,
            "overbudget": False, "harness": {"constants_ok": True, "prepare_code_hash_ok": True,
                                             "pipeline_code_hash_ok": True}}


def test_integrity_flags_clean_result_has_none():
    assert acc.integrity_flags(clean()) == []


@pytest.mark.parametrize("change,flag", [
    ({"mismatch_flag": True}, "mismatch"),
    ({"target_leak_flag": True}, "target_leak"),
    ({"canary_ok": False}, "canary_failed"),
    ({"harness_missing": True}, "harness_missing"),
    ({"harness": {"error": "harness_result.json missing or unreadable"}}, "harness_missing"),
    ({"harness": {"constants_ok": False}}, "constants"),
    ({"harness": {"prepare_code_hash_ok": False}}, "hash"),
    ({"harness": {"pipeline_code_hash_ok": False}}, "hash"),
    ({"status": "overbudget"}, "overbudget"),
    ({"overbudget": True}, "overbudget"),
    ({"status": "invalid_seed"}, "invalid_seed"),
    ({"seed_lines_matched": 1}, "invalid_seed"),
    ({"seed_patched": False}, "invalid_seed"),
    ({"official_val_bpb": None}, "no_official"),
])
def test_each_integrity_flag(change, flag):
    r = {**clean(), **change}
    assert flag in acc.integrity_flags(r)


def test_integrity_flags_top_level_beats_nested_and_multiple_flags_accumulate():
    r = {**clean(), "canary_ok": True, "harness": {"canary_ok": False}}
    assert acc.integrity_flags(r) == []                                    # top level is consulted first
    r = {**clean(), "target_leak_flag": True, "mismatch_flag": True, "official_val_bpb": None}
    assert acc.integrity_flags(r) == ["mismatch", "target_leak", "no_official"]
    assert acc.integrity_flags({**clean(), "seed": None, "seed_lines_matched": 0}) == []   # unseeded job


def test_integrity_flags_clean_worker_result_with_null_error_key():
    # Regression: the worker always writes harness["error"] = None on clean runs; that must not count as missing.
    from acceptance import integrity_flags
    result = {"status": "ok", "official_val_bpb": 1.1, "harness_missing": False, "mismatch_flag": False,
              "target_leak_flag": False, "integrity_failed": [], "seed": 42, "seed_patched": True,
              "seed_lines_matched": 2, "overbudget": False,
              "harness": {"error": None, "canary_ok": True, "constants_ok": True,
                          "prepare_code_hash_ok": True, "pipeline_code_hash_ok": True}}
    assert integrity_flags(result) == []
    result["harness"]["error"] = "tail crashed"
    assert "harness_missing" in integrity_flags(result)
    result["harness"]["error"] = None
    result["integrity_failed"] = ["canary_ok"]
    assert "worker:canary_ok" in integrity_flags(result)
