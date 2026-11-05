import pytest

import consistency
from consistency import sample_k
from conftest import cli_json, ok_cli

SCHEMA = {"type": "object", "properties": {"label": {}}, "required": ["label"]}


def run(claude, labels, **kw):
    claude([ok_cli(cli_json("", {"label": v})) for v in labels])
    return sample_k("haiku", "q", len(labels), key="label", schema=SCHEMA, log_path="l.jsonl", **kw)


def test_agreement_math(claude):
    r = run(claude, ["a", "a", "b", "a", "c"])
    assert r["values"] == ["a", "a", "b", "a", "c"] and r["modal_value"] == "a"
    assert r["agreement"] == pytest.approx(3 / 5) and r["n_ok"] == 5
    assert len(r["results"]) == 5 and all(x.ok for x in r["results"])


def test_unanimous_and_tie_goes_to_first_seen(claude):
    assert run(claude, [1, 1, 1])["agreement"] == 1.0
    r = run(claude, ["x", "y", "y", "x"])
    assert r["modal_value"] == "x" and r["agreement"] == 0.5


def test_failed_calls_excluded_from_denominator(claude):
    # two good calls, then one call that fails all 3 attempts (schema violation)
    claude([ok_cli(cli_json("", {"label": "a"})), ok_cli(cli_json("", {"label": "a"}))]
           + [ok_cli(cli_json("", {"nolabel": 1}))] * 3)
    r = sample_k("haiku", "q", 3, key="label", schema=SCHEMA, log_path="l.jsonl")
    assert r["n_ok"] == 2 and r["agreement"] == 1.0 and len(r["results"]) == 3 and not r["results"][2].ok


def test_all_failed(claude):
    claude([ok_cli(cli_json("", {"nolabel": 1}))] * 6)
    r = sample_k("haiku", "q", 2, key="label", schema=SCHEMA, attempts=3, log_path="l.jsonl")
    assert r["n_ok"] == 0 and r["agreement"] == 0.0 and r["modal_value"] is None and r["values"] == []


def test_unhashable_values(claude):
    r = run(claude, [[1, 2], [1, 2], {"k": 1}])
    assert r["modal_value"] == [1, 2] and r["agreement"] == pytest.approx(2 / 3)


def test_vllm_samples_are_independent_calls(http):
    from conftest import vllm_body
    s = http([(200, vllm_body('{"label": "a"}')), (200, vllm_body('{"label": "b"}')), (200, vllm_body('{"label": "a"}'))])
    r = sample_k("qwen7b-awq", "q", 3, key="label", schema=SCHEMA, temperature=0.8, log_path="l.jsonl")
    assert len(s.calls) == 3 and r["modal_value"] == "a" and r["agreement"] == pytest.approx(2 / 3)
    assert all(c[1]["temperature"] == 0.8 for c in s.calls)
