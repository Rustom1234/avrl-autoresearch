import hashlib
import json
import sys
from pathlib import Path

import pytest

import harness_tail
import worker
from conftest import EVAL_TOKENS, PYKEY, approx, expected_bpb, sha

SUB = Path(__file__).resolve().parent.parent
REF = SUB / "base" / "prepare_pinned.py"
E1, E2 = expected_bpb(1.25), expected_bpb(2.5)


# ---- the tail source and the pins ----
def test_tail_is_valid_python_without_placeholders_or_em_dash():
    compile(harness_tail.HARNESS_TAIL, "tail", "exec")
    assert chr(0x2014) not in harness_tail.HARNESS_TAIL
    for placeholder in ("__PINNED__", "__PIPELINE__", "__PIPELINE_NAMES__", "__DATA_FILES__"):
        assert placeholder not in harness_tail.HARNESS_TAIL


def test_tail_does_not_read_candidate_batch_size_or_trust_math_log():
    assert "DEVICE_BATCH_SIZE" not in harness_tail.HARNESS_TAIL
    assert "0.6931471805599453" in harness_tail.HARNESS_TAIL and "import math" not in harness_tail.HARNESS_TAIL


def test_observed_hashes_equal_harness_tail_helper(env):
    """The recipe inlined in the tail (incl. nested code objects) equals harness_tail's."""
    r = env.run_job("h1")
    hashes = harness_tail.prepare_code_hashes((env.base / "prepare.py").read_text())
    assert r["harness"]["eval_code_sha256"] == hashes.pop("evaluate_bpb")
    assert r["harness"]["pipeline_code_sha256"] == hashes
    assert set(hashes) == set(harness_tail.PIPELINE_FUNCS)


def test_unknown_python_version_gives_none_and_keeps_official(env):
    (env.base / "pinned_hashes.json").write_text(json.dumps({
        "prepare_py_sha256": hashlib.sha256((env.base / "prepare.py").read_bytes()).hexdigest(),
        "evaluate_bpb_code_sha256": {}, "pipeline_code_sha256": {}}))
    r = env.run_job("u1")
    assert r["harness"]["prepare_code_hash_ok"] is None and r["harness"]["pipeline_code_hash_ok"] is None
    assert r["official_val_bpb"] == approx(E1)
    assert r["harness"]["eval_code_sha256"]  # observed hash is reported so it can be pinned


def test_nested_code_hash_is_stable_and_sensitive():
    src = "def f():\n    def g():\n        return %d\n    return g\n"
    a1 = compile(src % 1, "a", "exec").co_consts[0]
    a2 = compile(src % 1, "other_path.py", "exec").co_consts[0]  # new code objects, new addresses and paths
    b = compile(src % 2, "a", "exec").co_consts[0]
    assert harness_tail.eval_code_hash(a1) == harness_tail.eval_code_hash(a2)
    assert harness_tail.eval_code_hash(a1) != harness_tail.eval_code_hash(b)


def test_changed_constant_is_detected():
    """Same opcodes, different constant: co_code alone would miss it, the consts hash does not."""
    a = compile("def f(x):\n    return x * 2\n", "a", "exec").co_consts[0]
    b = compile("def f(x):\n    return x * 3\n", "a", "exec").co_consts[0]
    assert a.co_code == b.co_code
    assert harness_tail.eval_code_hash(a) != harness_tail.eval_code_hash(b)


def test_pinned_hashes_json_matches_code_constants():
    pins = json.loads((SUB / "pinned_hashes.json").read_text())
    assert pins["prepare_py_sha256"] == worker.PINNED_PREPARE_SHA256
    assert pins["evaluate_bpb_code_sha256"] == harness_tail.PINNED_EVAL_CODE_SHA256
    assert pins["pipeline_code_sha256"] == harness_tail.PINNED_PIPELINE_CODE_SHA256
    versions = {"3.10", "3.12", "3.13", "3.14"}
    assert set(pins["evaluate_bpb_code_sha256"]) == set(pins["pipeline_code_sha256"]) == versions
    for table in pins["pipeline_code_sha256"].values():
        assert set(table) == set(harness_tail.PIPELINE_FUNCS)
    # empty until the first smoke job, then file name -> sha256 as measured there (never guessed)
    assert all(len(h) == 64 and set(h) <= set("0123456789abcdef") for h in pins.get("data_files", {}).values())


@pytest.mark.skipif(not REF.exists(), reason="reference clone not present")
def test_pins_match_reference_prepare():
    data = REF.read_bytes()
    assert hashlib.sha256(data).hexdigest() == worker.PINNED_PREPARE_SHA256
    if PYKEY not in harness_tail.PINNED_EVAL_CODE_SHA256:
        pytest.skip("no pin for Python " + PYKEY)
    hashes = harness_tail.prepare_code_hashes(data.decode())
    assert hashes.pop("evaluate_bpb") == harness_tail.PINNED_EVAL_CODE_SHA256[PYKEY]
    assert hashes == harness_tail.PINNED_PIPELINE_CODE_SHA256[PYKEY]


# ---- A: harness-owned scoring ----
def test_harness_owned_score_matches_closed_form_and_the_prepare_path(env):
    r = env.run_job("a1", sealed=True)
    h = r["harness"]
    assert h["official_tokens_scored"] == h["sealed_tokens_scored"] == EVAL_TOKENS == 20971520
    assert h["official_bytes_scored"] == EVAL_TOKENS // 8 * 28  # token k has k bytes, y cycles 0..7
    assert h["score_shape_bad"] is False
    assert h["official_val_bpb"] == approx(E1) and h["sealed_val_bpb"] == approx(E2)
    assert h["official_via_prepare"] == approx(E1) and abs(h["official_via_prepare_diff"]) < 1e-5
    assert r["official_val_bpb"] == approx(E1) and r["sealed_val_bpb"] == approx(E2)


def test_score_does_not_depend_on_the_candidates_loss_path(env, monkeypatch):
    monkeypatch.setenv("FAKE_MODE", "lying_loss")  # forward(x, y) returns half the true loss
    r = env.run_job("a2")
    h = r["harness"]
    assert h["official_val_bpb"] == approx(E1)  # from raw logits
    assert h["official_via_prepare"] == approx(E1 / 2)  # what the pinned function reports
    assert h["official_via_prepare_diff"] == approx(E1 / 2)  # recorded
    assert r["printed"]["val_bpb"] == approx(E1 / 2)  # and what train.py printed
    assert h["target_leak_flag"] is True  # the two paths disagree: the leak check sees it too
    assert r["official_val_bpb"] is None  # so the top-level score is withheld


def test_wrong_token_count_is_score_shape_bad(env, monkeypatch):
    monkeypatch.setenv("FAKE_MODE", "tamper_short")  # loader delivers 2047 columns instead of 2048
    r = env.run_job("a3", sealed=True)
    h = r["harness"]
    assert h["score_shape_bad"] is True
    assert h["official_tokens_scored"] == 160 * 64 * 2047 != EVAL_TOKENS
    assert h["official_val_bpb"] is None and h["sealed_val_bpb"] is None
    assert set(r["integrity_failed"]) == {"score_shape_bad", "pipeline_code_hash_ok"}
    assert r["official_val_bpb"] is None


# ---- B: known-answer canary ----
def test_canary_passes_on_an_honest_process(env):
    h = env.run_job("b1")["harness"]
    assert h["canary_ok"] is True
    assert h["canary_value"] == pytest.approx(3.25, abs=1e-6)  # 13 bits * 7/8 masked tokens / 3.5 bytes per token
    assert h["canary_expected"] == pytest.approx(3.25, abs=1e-12) and abs(h["canary_value"] - h["canary_expected"]) < 1e-6


@pytest.mark.parametrize("mode", ["tamper_item", "tamper_ce"])
def test_canary_catches_patched_tensor_item_or_cross_entropy(env, monkeypatch, mode):
    monkeypatch.setenv("FAKE_MODE", mode)  # item() / F.cross_entropy results scaled by 0.9
    r = env.run_job("b_" + mode)
    h = r["harness"]
    assert h["canary_ok"] is False and abs(h["canary_value"] - h["canary_expected"]) > 0.1
    assert r["integrity_failed"] == ["canary_ok"]
    assert r["official_val_bpb"] is None and h["official_val_bpb"] is not None  # raw kept for forensics


# ---- C: pinned data files ----
def test_data_files_are_hashed_and_unpinned_means_none(env):
    h = env.run_job("c1")["harness"]
    assert h["data_files_sha256"] == {
        "shard_06542.parquet": sha(env.stage_data / "shard_06542.parquet"),
        "token_bytes.pt": sha(env.stage_tok / "token_bytes.pt"),
        "tokenizer.pkl": sha(env.stage_tok / "tokenizer.pkl")}
    assert h["data_files_ok"] is None


def _pin_data_files(env, **override):
    files = {"shard_06542.parquet": sha(env.src / "data" / "shard_06542.parquet"),
             "token_bytes.pt": sha(env.src / "tokenizer" / "token_bytes.pt"),
             "tokenizer.pkl": sha(env.src / "tokenizer" / "tokenizer.pkl")}
    files.update(override)
    (env.base / "pinned_hashes.json").write_text(json.dumps(dict(env.pins, data_files=files)))


def test_matching_data_file_pins(env):
    _pin_data_files(env)
    r = env.run_job("c2")
    assert r["harness"]["data_files_ok"] is True and r["official_val_bpb"] == approx(E1)


@pytest.mark.parametrize("name", ["shard_06542.parquet", "token_bytes.pt", "tokenizer.pkl"])
def test_wrong_data_file_pin_nulls_the_scores(env, name):
    _pin_data_files(env, **{name: "0" * 64})
    r = env.run_job("c3_" + name.split(".")[0])
    assert r["harness"]["data_files_ok"] is False and r["integrity_failed"] == ["data_files_ok"]
    assert r["official_val_bpb"] is None and r["harness"]["official_val_bpb"] == approx(E1)


def test_a_doctored_token_bytes_file_is_caught_by_its_pin(env):
    _pin_data_files(env)  # pins the genuine file
    import torch
    torch.save(torch.tensor([0, 1, 2, 3, 4, 5, 6, 700], dtype=torch.int32), env.src / "tokenizer" / "token_bytes.pt")
    r = env.run_job("c4")  # staging notices the changed size? same dtype/length: content differs, size may too
    assert r["harness"]["data_files_ok"] is False
    assert r["harness"]["official_val_bpb"] != approx(E1)  # the doctored table changes the score ...
    assert r["official_val_bpb"] is None  # ... and the pin withholds it


# ---- tamper detection: (FAKE_MODE, flag that must flip, list holding the culprit, culprit) ----
TAMPERS = [
    ("tamper_loader", "pipeline_code_hash_ok", "pipeline_bad", "make_dataloader"),
    ("tamper_docs", "pipeline_code_hash_ok", "pipeline_bad", "_document_batches"),
    ("tamper_tokbytes", "pipeline_code_hash_ok", "pipeline_bad", "get_token_bytes"),
    ("tamper_listfiles", "pipeline_code_hash_ok", "pipeline_bad", "list_parquet_files"),
    ("tamper_encode", "pipeline_code_hash_ok", "pipeline_bad", "Tokenizer.encode"),
    ("tamper_tokenizer_cls", "pipeline_code_hash_ok", "pipeline_bad", "Tokenizer.type"),
    ("tamper_val_filename", "constants_ok", "constants_bad", "VAL_FILENAME"),
    ("tamper_data_dir", "constants_ok", "constants_bad", "DATA_DIR"),
    ("tamper_vocab", "constants_ok", "constants_bad", "VOCAB_SIZE"),
    ("tamper_tokdir", "constants_ok", "constants_bad", "TOKENIZER_DIR"),
]


@pytest.mark.parametrize("mode,flag,listing,culprit", TAMPERS)
def test_tampered_pipeline_flips_flag_and_nulls_official(env, monkeypatch, mode, flag, listing, culprit):
    monkeypatch.setenv("FAKE_MODE", mode)
    r = env.run_job("t_" + mode, sealed=True)
    h = r["harness"]
    assert r["status"] == "ok" and h["error"] is None
    assert h[flag] is False and h[listing] == [culprit]
    for other in {"constants_ok", "pipeline_code_hash_ok", "prepare_code_hash_ok", "canary_ok"} - {flag}:
        assert h[other] is True, other  # only the tampered piece is flagged
    assert h["official_val_bpb"] is not None  # raw value kept for forensics ...
    assert r["official_val_bpb"] is None and r["sealed_val_bpb"] is None  # ... but not exposed as official
    assert r["integrity_failed"] == [flag]


def test_tampered_make_dataloader_changes_the_score_but_is_caught(env, monkeypatch):
    monkeypatch.setenv("FAKE_MODE", "tamper_loader")  # feeds the shard value 0.1 instead of the val shard
    r = env.run_job("tl")
    assert r["harness"]["official_val_bpb"] == approx(expected_bpb(0.1)) != approx(E1)
    assert r["harness"]["pipeline_code_hash_ok"] is False and r["official_val_bpb"] is None


def test_val_shard_receipt_binds_the_data(env):
    expect = sha(env.src / "data" / "shard_06542.parquet")
    h = env.run_job("rc1", sealed=True)["harness"]
    assert h["val_shard_sha256"] == expect  # the official shard, not the sealed one
    assert h["val_shard_path"] == str(env.stage_data / "shard_06542.parquet")  # the staged, node-local copy
    assert h["sealed_val_bpb"] == approx(E2)


def test_val_shard_receipt_follows_the_shard_actually_used(env, monkeypatch):
    monkeypatch.setenv("FAKE_MODE", "tamper_data_dir")
    h = env.run_job("rc2")["harness"]
    assert h["val_shard_path"] == str(env.alt / "shard_06542.parquet")
    assert h["val_shard_sha256"] == sha(env.alt / "shard_06542.parquet")


# ---- fixed eval batch size and CUDA OOM fallback ----
def test_eval_batch_is_fixed_at_64_whatever_the_candidate_says(env):
    r = env.run_job("b64")  # fake_train sets DEVICE_BATCH_SIZE = 128
    assert r["eval_batch_size_used"] == r["harness"]["eval_batch_size_used"] == 64
    assert r["harness"]["sealed_eval_batch_size_used"] is None


@pytest.mark.parametrize("kind", ["runtime", "torch"])
@pytest.mark.parametrize("above", [32, 16])
def test_oom_at_64_retries_smaller(env, monkeypatch, above, kind):
    monkeypatch.setenv("FAKE_OOM_ABOVE", str(above))  # evaluation and the fake model's forward refuse bigger batches
    monkeypatch.setenv("FAKE_OOM_KIND", kind)  # RuntimeError("out of memory") or torch.OutOfMemoryError
    r = env.run_job("o%d%s" % (above, kind), sealed=True)
    h = r["harness"]
    assert h["error"] is None and r["status"] == "ok", r["log_tail"]
    assert r["eval_batch_size_used"] == h["eval_batch_size_used"] == above
    assert h["sealed_eval_batch_size_used"] == above  # sealed starts from the size that worked
    assert h["target_leak_flag"] is False  # the leak check retried too
    assert (r["official_val_bpb"], r["sealed_val_bpb"]) == (approx(E1), approx(E2))
    assert h["official_via_prepare"] == approx(E1)


def test_oom_at_every_size_is_reported(env, monkeypatch):
    monkeypatch.setenv("FAKE_OOM_ABOVE", "8")
    r = env.run_job("o8")
    assert "out of memory" in r["harness"]["error"] and "out of memory" in r["harness"]["target_leak_error"]
    assert r["official_val_bpb"] is None and r["eval_batch_size_used"] is None
    assert r["mismatch_flag"] is True  # train.py printed a score the harness could not reproduce


# ---- target leak check ----
def test_unmodified_forward_has_no_leak(env):
    r = env.run_job("lk0")
    h = r["harness"]
    assert h["target_leak_flag"] is False and h["target_leak_error"] is None
    assert abs(h["target_leak_loss_a"] - h["target_leak_loss_b"]) < 1e-5  # softcap is shared by both paths
    assert r["target_leak_flag"] is False


def test_forward_reading_targets_trips_the_flag_and_cannot_improve_the_score(env, monkeypatch):
    monkeypatch.setenv("FAKE_MODE", "leak")
    r = env.run_job("lk1", sealed=True)
    h = r["harness"]
    assert h["error"] is None
    assert h["target_leak_loss_b"] - h["target_leak_loss_a"] > 1.0
    assert h["target_leak_flag"] is True and r["target_leak_flag"] is True
    assert h["official_val_bpb"] == approx(E1)  # harness-owned scoring never passes targets: no benefit from the leak
    assert h["official_via_prepare"] < h["official_val_bpb"] - 0.1  # while the pinned path is flattered
    assert r["official_val_bpb"] is None and r["sealed_val_bpb"] is None
    assert r["integrity_failed"] == ["target_leak_flag"]


def test_changed_forward_signature_gives_null_flag_but_is_still_scored(env, monkeypatch):
    monkeypatch.setenv("FAKE_MODE", "badsig")  # forward(self, idx): no targets argument
    r = env.run_job("lk2")
    assert r["target_leak_flag"] is None and "TypeError" in r["harness"]["target_leak_error"]
    assert r["official_val_bpb"] == approx(E1)  # scoring needs only model(x): the leak check was not the last line
    assert "via prepare" in r["harness"]["error"]  # the candidate-owned loss path is what broke


# ---- no torch at all: the tail degrades, the receipt is still written and authenticated ----
def test_no_torch_degrades_gracefully(env, monkeypatch):
    monkeypatch.setenv("FAKE_NOTORCH", "1")
    r = env.run_job("nt1", sealed=True)
    h = r["harness"]
    assert r["status"] == "ok" and r["harness_missing"] is False
    assert "torch unavailable" in h["error"] and "torch unavailable" in h["target_leak_error"]
    assert h["official_val_bpb"] is None and h["canary_ok"] is None and r["official_val_bpb"] is None
    assert h["constants_ok"] is True and h["prepare_code_hash_ok"] is True and h["pipeline_code_hash_ok"] is True
