"""prompts.py: program card facts, which roles see the full train.py, schemas against the fakes, house rules."""
import pathlib

import patch
import prompts as P
import services
from fakes import BASE, FakeModelService

SRC = BASE.read_text()
ITEM = {"id": "i001", "title": "Raise MATRIX_LR", "spec": "Increase MATRIX_LR to 0.05.", "family": "optimizer",
        "status": "open"}
FAMILY = {"optimizer": {"alpha": 2, "beta": 1}}


def test_card_states_the_harness_facts_plainly():
    card = P.program_card(20)
    for fact in ("A100 40 GB", "DEVICE_BATCH_SIZE 64", "300 s", "val_bpb", "450 s", "owned by the harness",
                 "invalidates the run", "prepare.py is read only", "do not reference prepare.py internals",
                 "recomputed by the harness", "20 slots", "SEARCH/REPLACE", "at most 2 per slot"):
        assert fact in card, fact
    assert "abort" not in card.lower()                       # the card is the same for every arm
    assert P.program_card(20) == P.program_card(20) and "12 slots" in P.program_card(12)


def test_card_example_block_parses_with_the_real_patch_module():
    assert patch.parse_blocks(P.EXAMPLE) == [
        ("WEIGHT_DECAY = 0.2      # cautious weight decay for Muon\n",
         "WEIGHT_DECAY = 0.1      # cautious weight decay for Muon\n")]
    new, rep = patch.apply_text(SRC, P.EXAMPLE)
    assert rep["ok"] and "WEIGHT_DECAY = 0.1 " in new


def test_train_py_is_in_full_only_for_base_step_coder_and_repairer():
    marker = "class MuonAdamW"
    full = [P.base_step(SRC, "tbl", "tail", True), P.coder(SRC, ITEM, "tbl", "memo", FAMILY, True),
            P.repairer_lint(SRC, [{"rule": "forbidden-import", "line": 3, "message": "no"}]),
            P.repairer_crash(SRC, "Traceback ...")]
    assert all(marker in p for p in full)
    other = [P.planner(SRC, 30, "tbl", "memo", FAMILY, [ITEM]), P.voter(ITEM), P.analyst("tbl", "memo", FAMILY),
             P.critic("diff text", "numbers", "why")]
    assert not any(marker in p for p in other)


def test_abort_is_offered_only_when_allowed():
    assert "abort_reason" in P.base_step(SRC, "t", "x", True) and "abort_reason" not in P.base_step(SRC, "t", "x", False).replace(
        '"abort": false, "abort_reason": ""', "")
    assert "set \"abort\": true" in P.coder(SRC, ITEM, "t", "", {}, True)
    assert "set \"abort\": true" not in P.coder(SRC, ITEM, "t", "", {}, False)


def test_role_prompts_carry_their_inputs():
    c = P.coder(SRC, ITEM, "TABLE", "MEMO TEXT", FAMILY, False)
    assert all(x in c for x in ("TABLE", "MEMO TEXT", ITEM["spec"], "optimizer a=2 b=1", "Family: optimizer"))
    pl = P.planner(SRC, 30, "TABLE", "MEMO", FAMILY, [ITEM], widen=True)
    assert "30 ideas" in pl and "stalled" in pl and "[i001] Raise MATRIX_LR" in pl and "DEVICE_BATCH_SIZE = 64" in pl
    assert "stalled" not in P.planner(SRC, 30, "TABLE", "MEMO", FAMILY, [], widen=False)
    assert "forbidden-import (line 3)" in P.repairer_lint(SRC, [{"rule": "forbidden-import", "line": 3, "message": "x"}])
    assert "RuntimeError: boom" in P.repairer_crash(SRC, "RuntimeError: boom")
    assert "```diff\nDIFF\n```" in P.critic("DIFF", "z = 1.0", "borderline accept")
    assert P.retry_suffix("old", "") == "" and "old" in P.retry_suffix("old", "no match") and "no match" in P.retry_suffix("old", "no match")


def test_hyper_block_lists_the_hyperparameter_assignments():
    hb = P.hyper_block(SRC)
    assert "MATRIX_LR = 0.04" in hb and "DEVICE_BATCH_SIZE = 64" in hb and "class" not in hb


def test_schemas_accept_what_the_fakes_return():
    services._path("models")
    import call
    m = FakeModelService()
    for role, schema in P.SCHEMAS.items():
        obj = m.call(role, "sonnet", "x").obj
        assert call.validate(obj, schema) is None, role
    assert call.validate({"vote": "maybe"}, P.VOTE_SCHEMA) is not None
    assert call.validate({"items": [{"title": "t", "spec": "s", "family": "weird"}]}, P.BACKLOG_SCHEMA) is not None


def test_no_em_dash_in_any_file_this_build_wrote():
    dash = chr(0x2014)
    core = pathlib.Path(__file__).resolve().parents[1]
    files = [core / n for n in ("acceptance.py", "ledger.py", "scheduler.py", "prompts.py", "arms.py", "services.py",
                                "run_loop.py", "ENGINE.md")]
    files += sorted((core / "tests").glob("test_engine_*.py")) + [core / "tests" / "fakes.py"]
    assert all(f.exists() for f in files)
    bad = [f.name for f in files if dash in f.read_text()]
    assert not bad, bad
    # prompt templates (the baseline train.py itself has one em dash in a comment, which is not ours to edit)
    assert dash not in P.program_card(20) + P.base_step("", "t", "x", True) + P.planner("", 30, "t", "m", {}, [])
