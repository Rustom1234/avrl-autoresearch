"""End-to-end tests of LoopEngine on the fakes: acceptance flow, free actions, infra, crash, resume, arms C and D."""
import json
import math
import re

import pytest

import ledger as L
from fakes import (FakeClock, FakeGPUService, FakeModelService, Infra, Killed, ModelFail, abort_reply,
                   bad_search_reply, make_engine, remove_line_reply, reply, violating_reply)

SIGMA = 0.0067
ONE = SIGMA * math.sqrt(1 + 1 / 24)        # denominator, one candidate run against the 24-run anchor
TWO = SIGMA * math.sqrt(0.5 + 1 / 24)      # two candidate runs
STATE_KEYS = {"loop_id", "arm", "slot_index", "n_slots", "incumbent", "free_actions_run", "free_actions_slot",
              "in_flight_job", "backlog", "family_stats", "consecutive_rejections", "memo", "rng_state",
              "claimed_best_official", "status", "started", "updated", "served_counts", "designated_open_calls",
              "open_served_calls"}
MODEL_CALL_KEYS = {"role", "requested_model", "served_model", "cost_usd_plan", "cost_usd_nominal", "tokens_in",
                   "tokens_out", "latency_s"}


def run(tmp_path, arm, scripts=None, gpu=None, models=None, **cfg):
    models = models or FakeModelService(scripts)
    gpu = gpu or FakeGPUService(noise_sd=0)
    eng, svc = make_engine(tmp_path, arm, models, gpu, **cfg)
    status = eng.run()
    assert len(charged(eng)) == eng.state["slot_index"]       # the invariant that ties the ledger to the slot count
    return eng, svc, status

def charged(eng):
    return [r for r in eng.records if r["charged"]]


def read_state(eng):
    return json.loads((eng.run_dir / "state.json").read_text())


# ------------------------------------------------------------------ every arm, default fakes

@pytest.mark.parametrize("arm", ["A", "B", "C", "D"])
def test_each_arm_runs_six_slots_and_writes_state_and_ledger(tmp_path, arm):
    eng, svc, status = run(tmp_path, arm, gpu=FakeGPUService(), n_slots=6)
    assert status == "done"
    assert len(charged(eng)) == 6 and len(eng.records) == 6          # one record per slot, nothing else happened
    st = read_state(eng)
    assert STATE_KEYS <= set(st) and st["slot_index"] == 6 and st["status"] == "done"
    assert [r["slot"] for r in eng.records] == list(range(6))
    on_disk = [json.loads(x) for x in (eng.run_dir / "ledger.jsonl").read_text().splitlines()]
    assert on_disk == eng.records
    for r in on_disk:
        assert set(r) >= set(L.FIELDS) and r["decision"] in L.DECISIONS and r["kind"] in L.KINDS
        for c in r["model_calls"]:
            assert set(c) == MODEL_CALL_KEYS
    assert len(svc.gpu.submits) == 6
    assert not list(eng.run_dir.glob("*.tmp"))                         # atomic writes leave no temp file
    assert (eng.run_dir / "incumbent.py").exists() and (eng.run_dir / "slots" / "0" / "train.py").exists()
    if arm in ("A", "B"):
        assert {c[0] for c in svc.models.calls} == {"base_step"}      # one model call per slot, no other roles
    for r in on_disk:
        assert (r["lint"] is None) == (arm == "A")                    # arm A has no linter


def test_claimed_best_and_incumbent_follow_accepts(tmp_path):
    eng, svc, _ = run(tmp_path, "B", {"base_step": [reply(-0.02)]}, n_slots=1)
    (r,) = eng.records
    assert r["decision"] == "accept" and r["z1"] == pytest.approx(0.02 / ONE)
    inc = eng.state["incumbent"]
    assert inc["n"] == 1 and inc["mean_adjusted"] == pytest.approx(1.1236) and inc["raw_values"] == [pytest.approx(1.1236)]
    assert eng.state["claimed_best_official"] == pytest.approx(1.1236)
    assert "# fx:-0.02" in open(inc["path"]).read()
    assert r["incumbent_before"] == {"mean": 1.1436, "n": 24} and r["incumbent_after"] == {"mean": inc["mean_adjusted"], "n": 1}


# ------------------------------------------------------------------ acceptance rule

def test_b_replicates_when_z1_is_in_the_band_then_accepts_on_z2(tmp_path):
    gpu = FakeGPUService(noise_sd=0)
    eng, svc, _ = run(tmp_path, "B", {"base_step": [reply(-0.010)]}, gpu=gpu, n_slots=2)
    c, r = eng.records
    assert (c["kind"], c["decision"]) == ("candidate", "replicate")
    assert c["z1"] == pytest.approx(0.010 / ONE) and c["z2"] is None
    assert (r["kind"], r["decision"]) == ("replicate", "accept")
    assert r["z2"] == pytest.approx(0.010 / TWO) and r["z1"] == pytest.approx(c["z1"])
    assert gpu.submits[0]["seed"] == 42 and gpu.submits[1]["seed"] not in (None, 42)    # fresh seed from the loop RNG
    assert gpu.submits[1]["sealed"] is False and gpu.submits[1]["kind"] == "replicate"
    assert eng.state["incumbent"]["n"] == 2 and eng.state["incumbent"]["mean_adjusted"] == pytest.approx(1.1336)
    assert charged(eng) == eng.records and eng.state["slot_index"] == 2


def test_b_rejects_after_a_replicate_that_does_not_confirm(tmp_path):
    gpu = FakeGPUService(noise_sd=0, bias={2: 0.0095})                  # replicate gain 0.0005, mean gain 0.00525
    eng, svc, _ = run(tmp_path, "B", {"base_step": [reply(-0.010)]}, gpu=gpu, n_slots=2)
    assert [r["decision"] for r in eng.records] == ["replicate", "reject"]
    assert eng.records[1]["z2"] == pytest.approx(0.00525 / TWO) and eng.records[1]["z2"] < 1.5
    assert eng.state["incumbent"]["n"] == 24                            # unchanged


@pytest.mark.parametrize("fx,decision", [(-0.004, "reject"), (-0.0140, "accept")])
def test_b_single_run_decisions_outside_the_band(tmp_path, fx, decision):
    eng, svc, _ = run(tmp_path, "B", {"base_step": [reply(fx)]}, n_slots=1)
    assert [r["decision"] for r in eng.records] == [decision] and len(svc.gpu.submits) == 1


def test_tripwire_forces_a_sealed_audit_before_accepting(tmp_path):
    gpu = FakeGPUService(noise_sd=0)
    eng, svc, _ = run(tmp_path, "B", {"base_step": [reply(-0.06)]}, gpu=gpu, n_slots=2)
    c, a = eng.records
    assert (c["kind"], c["decision"]) == ("candidate", "audit")         # z1 8.8 would have been a plain accept
    assert c["flags"] == ["tripwire"] and a["flags"] == []              # the analysis counts "tripwire" separately
    assert (a["kind"], a["decision"]) == ("audit", "accept")
    assert gpu.submits[1]["sealed"] is True and gpu.submits[1]["seed"] not in (None, 42)
    assert a["sealed_val_bpb"] is not None and a["z2"] == pytest.approx(0.06 / TWO)
    assert eng.state["incumbent"]["n"] == 2


def test_audit_that_does_not_hold_up_rejects(tmp_path):
    gpu = FakeGPUService(noise_sd=0, bias={2: 0.11})                    # audit value +0.05, mean gain 0.005, z2 1.01
    eng, svc, _ = run(tmp_path, "B", {"base_step": [reply(-0.06)]}, gpu=gpu, n_slots=2)
    assert [r["decision"] for r in eng.records] == ["audit", "reject"]
    assert eng.state["incumbent"]["n"] == 24


def test_audit_without_a_sealed_value_is_invalid(tmp_path):
    gpu = FakeGPUService(noise_sd=0, patch={2: {"sealed_val_bpb": None}})
    eng, svc, _ = run(tmp_path, "B", {"base_step": [reply(-0.06)]}, gpu=gpu, n_slots=2)
    assert eng.records[1]["decision"] == "invalid" and "sealed_missing" in eng.records[1]["flags"]
    assert eng.state["incumbent"]["n"] == 24


def test_tripwire_applies_to_arm_a_as_well(tmp_path):
    eng, svc, _ = run(tmp_path, "A", {"base_step": [reply(-0.06)]}, n_slots=2)
    assert [(r["kind"], r["decision"]) for r in eng.records] == [("candidate", "audit"), ("audit", "accept")]


def test_cumulative_tripwire_triggers_over_several_accepts(tmp_path):
    # 1st: z1 2.34 accept. 2nd: gain 0.016 over a 1-run incumbent, z1 1.69, so replicate; z2 1.95 would accept, but the
    # cumulative gain over the anchor is 0.032 > 0.03, so an audit comes first.
    eng, svc, _ = run(tmp_path, "B", {"base_step": [reply(-0.016), reply(-0.016)]}, n_slots=4)
    assert [(r["kind"], r["decision"]) for r in eng.records] == [
        ("candidate", "accept"), ("candidate", "replicate"), ("replicate", "audit"), ("audit", "accept")]
    assert svc.gpu.submits[3]["sealed"] is True and eng.state["incumbent"]["n"] == 3
    assert eng.records[2]["flags"] == ["tripwire"]


# ------------------------------------------------------------------ arm A rule

def test_a_keeps_only_strictly_lower_raw_values(tmp_path):
    scripts = {"base_step": [reply(-0.001), reply(0.001), reply(0.0)]}
    eng, svc, _ = run(tmp_path, "A", scripts, n_slots=3)
    assert [r["decision"] for r in eng.records] == ["accept", "reject", "reject"]    # the third ties the bar exactly
    assert eng.state["incumbent"]["raw_values"] == [pytest.approx(1.1426)]
    assert all(r["z2"] is None for r in eng.records)


def test_a_decides_on_raw_values_and_b_on_gpu_adjusted_values(tmp_path):
    off = {0: 0.05}                                                      # GPU0 reads 0.05 high
    a, _, _ = run(tmp_path / "a", "A", {"base_step": [reply(-0.02)]}, gpu=FakeGPUService(noise_sd=0, offsets=off),
                  gpu_offsets=off, n_slots=1)
    b, _, _ = run(tmp_path / "b", "B", {"base_step": [reply(-0.02)]}, gpu=FakeGPUService(noise_sd=0, offsets=off),
                  gpu_offsets=off, n_slots=1)
    assert a.records[0]["official_val_bpb"] == pytest.approx(1.1736) and a.records[0]["decision"] == "reject"
    assert b.records[0]["adjusted_val_bpb"] == pytest.approx(1.1236) and b.records[0]["decision"] == "accept"


# ------------------------------------------------------------------ integrity

@pytest.mark.parametrize("arm", ["A", "B"])
def test_integrity_flag_rejects_and_charges_the_slot(tmp_path, arm):
    eng, svc, _ = run(tmp_path, arm, {"base_step": [reply(-0.05, "LEAK_ME")]}, n_slots=1)
    (r,) = eng.records
    assert r["decision"] == "invalid" and "target_leak" in r["flags"] and r["charged"]
    assert eng.state["slot_index"] == 1 and eng.state["incumbent"]["n"] == 24
    assert eng.state["claimed_best_official"] is None


def test_flagged_run_in_arm_c_is_reviewed_by_the_critic(tmp_path):
    eng, svc, _ = run(tmp_path, "C", {"coder": [reply(-0.05, "LEAK_ME")]}, n_slots=1)
    (r,) = eng.records
    assert r["decision"] == "invalid" and r["critic"] == {"ok": True, "issues": [], "veto": False}
    assert ("critic", "opus") in [(c[0], c[1]) for c in svc.models.calls]


# ------------------------------------------------------------------ infra failures

def test_gpu_infra_failure_does_not_charge_a_slot(tmp_path):
    clock = FakeClock()
    gpu = FakeGPUService(noise_sd=0, infra_on={1})
    eng, svc = make_engine(tmp_path, "B", FakeModelService({"base_step": [reply(-0.02)]}), gpu, clock=clock, n_slots=1)
    assert eng.run() == "done"
    infra, cand = eng.records
    assert infra["decision"] == "infra" and infra["charged"] is False and infra["status"] == "engine:infra:gpu:infra_fail"
    assert cand["decision"] == "accept" and eng.state["slot_index"] == 1 and len(gpu.submits) == 2
    assert clock.sleeps == [5] and eng.state["infra_events"] == 1


def test_model_infra_failure_waits_and_retries_the_same_step(tmp_path):
    clock = FakeClock()
    models = FakeModelService({"base_step": [Infra(), reply(-0.02)]})
    eng, svc = make_engine(tmp_path, "A", models, FakeGPUService(noise_sd=0), clock=clock, n_slots=1)
    assert eng.run() == "done"
    assert [r["decision"] for r in eng.records] == ["infra", "accept"] and eng.state["slot_index"] == 1
    assert eng.state["free_actions_run"] == 0 and clock.sleeps == [5]
    assert len(eng.records[0]["model_calls"]) == 1                      # the failed call's cost is still booked


def test_infra_does_not_update_posteriors_in_arm_c(tmp_path):
    gpu = FakeGPUService(noise_sd=0, infra_on={1})
    eng, svc, _ = run(tmp_path, "C", {"coder": [reply(0.02)]}, gpu=gpu, n_slots=1)
    assert eng.state["experiments"] == 1 and eng.state["consecutive_rejections"] == 1
    fam = eng.state["family_stats"]
    assert sum(v["alpha"] + v["beta"] - 2 for v in fam.values()) == 1    # exactly one outcome recorded, not two


def test_endless_infra_stops_the_loop_without_charging(tmp_path):
    models = FakeModelService({"base_step": [Infra()] * 10})
    eng, svc = make_engine(tmp_path, "A", models, FakeGPUService(), max_infra_retries=2, n_slots=3)
    assert eng.run() == "stopped" and read_state(eng)["status"] == "stopped"
    assert eng.state["slot_index"] == 0 and len(models.calls) == 3


def test_stop_file_stops_between_slots_and_a_resume_finishes(tmp_path):
    stop = tmp_path / "STOP"

    def touch(user):
        stop.write_text("")
        return reply(0.01)

    eng, svc = make_engine(tmp_path, "A", FakeModelService({"base_step": [reply(0.01), touch]}),
                           FakeGPUService(noise_sd=0), n_slots=4)
    assert eng.run() == "stopped" and eng.state["slot_index"] == 2
    stop.unlink()
    eng2, _ = make_engine(tmp_path, "A", FakeModelService(), FakeGPUService(noise_sd=0), n_slots=4)
    assert eng2.run() == "done" and eng2.state["slot_index"] == 4 and len(eng2.records) == 4


def test_stop_file_holding_a_slot_number_gives_the_symmetric_stop(tmp_path):
    (tmp_path / "STOP").write_text("2\n")
    eng, svc = make_engine(tmp_path, "A", FakeModelService(), FakeGPUService(noise_sd=0), n_slots=5)
    assert eng.run() == "stopped" and eng.state["slot_index"] == 2 and len(eng.records) == 2
    (tmp_path / "STOP").write_text("4")
    eng2, _ = make_engine(tmp_path, "A", FakeModelService(), FakeGPUService(noise_sd=0), n_slots=5)
    assert eng2.run() == "stopped" and eng2.state["slot_index"] == 4


# ------------------------------------------------------------------ free actions

def test_patch_failure_gets_two_free_retries_with_the_applier_hint(tmp_path):
    scripts = {"base_step": [bad_search_reply(), reply(-0.02)]}
    eng, svc, _ = run(tmp_path, "A", scripts, n_slots=1)
    (r,) = eng.records
    assert r["decision"] == "accept" and len(r["model_calls"]) == 2
    assert eng.state["free_actions_run"] == 1
    first, second = svc.models.calls[0][2], svc.models.calls[1][2]
    assert "could not be used" not in first and "could not be used" in second
    assert "THIS_LINE_DOES_NOT_EXIST" in second                         # the previous edits are shown back


def test_third_patch_failure_in_a_slot_charges_the_slot(tmp_path):
    scripts = {"base_step": [bad_search_reply()] * 3 + [reply(-0.02)]}
    eng, svc, _ = run(tmp_path, "B", scripts, n_slots=2)
    first, second = eng.records
    assert first["decision"] == "invalid" and first["charged"] and first["status"] == "engine:patch_fail"
    assert first["flags"] == []                                         # flags are integrity flags only
    assert first["job_id"] is None and len(first["model_calls"]) == 3
    assert second["decision"] == "accept" and second["slot"] == 1
    assert eng.state["free_actions_run"] == 2 and eng.state["free_actions_slot"] == 0   # per-slot count resets


def test_run_cap_on_free_actions(tmp_path):
    caps = {"slot": 2, "run": 1}
    scripts = {"base_step": [bad_search_reply()] * 4}
    eng, svc, _ = run(tmp_path, "A", scripts, n_slots=2, free_action_caps=caps)
    assert [r["decision"] for r in eng.records] == ["invalid", "invalid"]
    assert len(svc.models.calls) == 3 and eng.state["free_actions_run"] == 1    # 2 calls in slot 0, 1 call in slot 1


def test_syntax_error_is_retried_like_a_patch_failure(tmp_path):
    broken = {"description": "x", "abort": False, "abort_reason": "",
              "edits_text": "<<<<<<< SEARCH\nWARMUP_RATIO = 0.0      # fraction of time budget for LR warmup\n=======\n"
                            "WARMUP_RATIO = (\n>>>>>>> REPLACE\n"}
    eng, svc, _ = run(tmp_path, "A", {"base_step": [broken, reply(-0.02)]}, n_slots=1)
    assert [r["decision"] for r in eng.records] == ["accept"] and eng.state["free_actions_run"] == 1
    assert "does not" in svc.models.calls[1][2]


def test_model_that_never_matches_the_schema_is_a_charged_failure(tmp_path):
    eng, svc, _ = run(tmp_path, "A", {"base_step": [ModelFail()] * 3}, n_slots=1)
    assert eng.records[0]["decision"] == "invalid" and eng.records[0]["status"] == "engine:model_fail"


def test_abort_is_a_free_action_in_verified_arms_and_ignored_in_arm_a(tmp_path):
    b, svc, _ = run(tmp_path / "b", "B", {"base_step": [abort_reply(), reply(-0.02)]}, n_slots=1)
    ab, acc = b.records
    assert ab["decision"] == "abort" and ab["charged"] is False and ab["job_id"] is None
    assert acc["decision"] == "accept" and acc["attempt"] == 1 and b.state["free_actions_run"] == 1
    assert '"abort": true' in svc.models.calls[0][2]
    a, svc_a, _ = run(tmp_path / "a", "A", {"base_step": [abort_reply(), reply(-0.02)]}, n_slots=1)
    assert [r["decision"] for r in a.records] == ["accept"]             # arm A is not offered abort, so it is not honoured
    assert '"abort": true' not in svc_a.models.calls[0][2]


def test_abort_over_the_cap_charges_the_slot(tmp_path):
    caps = {"slot": 1, "run": 40}
    eng, svc, _ = run(tmp_path, "B", {"base_step": [abort_reply(), abort_reply(), reply(-0.02)]}, n_slots=2,
                      free_action_caps=caps)
    assert [(r["decision"], r["charged"]) for r in eng.records] == [("abort", False), ("abort", True), ("accept", True)]
    assert eng.state["slot_index"] == 2


# ------------------------------------------------------------------ linter (B, C, D)

def test_lint_failure_gets_one_free_repair(tmp_path):
    scripts = {"base_step": [violating_reply(-0.02)], "repairer": [remove_line_reply("import subprocess")]}
    eng, svc, _ = run(tmp_path, "B", scripts, n_slots=1)
    (r,) = eng.records
    assert r["decision"] == "accept" and r["lint"]["ok"] is True and r["flags"] == []
    assert [c["role"] for c in r["model_calls"]] == ["base_step", "repairer"]
    assert "import subprocess" not in open(eng.state["incumbent"]["path"]).read()
    assert eng.state["free_actions_run"] == 1


def test_second_lint_failure_skips_the_idea_for_free(tmp_path):
    scripts = {"base_step": [violating_reply(), reply(-0.02)], "repairer": [reply(0.0, tag="still-bad")]}
    eng, svc, _ = run(tmp_path, "B", scripts, n_slots=1)
    skip, ok = eng.records
    assert skip["decision"] == "lint_fail" and skip["charged"] is False and skip["lint"]["ok"] is False
    assert skip["lint"]["hard"][0]["rule"] == "forbidden-import" and skip["job_id"] is None
    assert ok["decision"] == "accept" and ok["attempt"] == 1 and eng.state["free_actions_run"] == 2


def test_lint_failure_with_no_free_action_left_is_charged(tmp_path):
    eng, svc, _ = run(tmp_path, "B", {"base_step": [violating_reply()]}, n_slots=1,
                      free_action_caps={"slot": 0, "run": 40})
    (r,) = eng.records
    assert r["decision"] == "lint_fail" and r["charged"] and len(svc.gpu.submits) == 0
    assert "repairer" not in svc.models.roles()


def test_arm_a_runs_without_the_linter(tmp_path):
    eng, svc, _ = run(tmp_path, "A", {"base_step": [violating_reply(-0.02)]}, n_slots=1)
    assert eng.records[0]["lint"] is None and len(svc.gpu.submits) == 1 and "repairer" not in svc.models.roles()


# ------------------------------------------------------------------ crashes

@pytest.mark.parametrize("arm", ["A", "B", "C"])
def test_crash_gets_one_fix_attempt_that_costs_a_slot(tmp_path, arm):
    role = "coder" if arm == "C" else "base_step"
    scripts = {role: [reply(-0.02, "CRASH_ME")], "repairer": [remove_line_reply("# CRASH_ME")]}
    eng, svc, _ = run(tmp_path, arm, scripts, n_slots=2)
    crash, fix = eng.records
    assert (crash["kind"], crash["decision"], crash["official_val_bpb"]) == ("candidate", "crash", None)
    assert (fix["kind"], fix["decision"]) == ("crash_fix", "accept") and fix["idea_id"] == crash["idea_id"]
    assert eng.state["slot_index"] == 2 and svc.gpu.submits[1]["kind"] == "crash_fix"
    fix_prompt = [c for c in svc.models.calls if c[0] == "repairer"][0][2]
    assert "Traceback (most recent call last)" in fix_prompt and "RuntimeError: boom" in fix_prompt
    assert "CRASH_ME" not in open(eng.state["incumbent"]["path"]).read()


def test_a_second_crash_is_final(tmp_path):
    scripts = {"base_step": [reply(0.0, "CRASH_ME"), reply(-0.02)], "repairer": [reply(0.0, tag="no fix")]}
    eng, svc, _ = run(tmp_path, "B", scripts, n_slots=3)
    assert [(r["kind"], r["decision"]) for r in eng.records] == [
        ("candidate", "crash"), ("crash_fix", "crash"), ("candidate", "accept")]


def test_crash_fix_that_cannot_be_applied_still_costs_the_slot(tmp_path):
    scripts = {"base_step": [reply(0.0, "CRASH_ME"), reply(-0.02)], "repairer": [bad_search_reply()] * 3}
    eng, svc, _ = run(tmp_path, "A", scripts, n_slots=3)
    assert [(r["kind"], r["decision"]) for r in eng.records] == [
        ("candidate", "crash"), ("crash_fix", "invalid"), ("candidate", "accept")]
    assert eng.state["slot_index"] == 3


# ------------------------------------------------------------------ resume

def test_resume_polls_the_in_flight_job_instead_of_resubmitting(tmp_path):
    gpu = FakeGPUService(noise_sd=0, kill_on_wait=1)
    eng, svc = make_engine(tmp_path, "B", FakeModelService({"base_step": [reply(-0.02)]}), gpu, n_slots=2)
    with pytest.raises(Killed):
        eng.run()
    st = read_state(eng)
    assert st["slot_index"] == 0 and st["in_flight_job"]["job_id"] == "job001" and st["cand"]["runs"] == []
    assert not (eng.run_dir / "ledger.jsonl").exists()
    eng2, svc2 = make_engine(tmp_path, "B", FakeModelService(), gpu, n_slots=2)   # a new process reads state.json
    assert eng2.run() == "done"
    assert [s["n"] for s in gpu.submits] == [1, 2]                      # job 1 was polled, never resubmitted
    assert gpu.waits.count("job001") == 2
    first = eng2.records[0]
    assert first["job_id"] == "job001" and first["decision"] == "accept" and first["slot"] == 0
    assert eng2.state["slot_index"] == 2 and read_state(eng2)["in_flight_job"] is None


def test_resume_continues_a_candidate_waiting_for_its_replicate(tmp_path):
    gpu = FakeGPUService(noise_sd=0)
    eng, svc = make_engine(tmp_path, "B", FakeModelService({"base_step": [reply(-0.010)]}), gpu, n_slots=2)
    eng.run_one_slot()
    assert read_state(eng)["cand"]["next"] == "replicate"
    eng2, _ = make_engine(tmp_path, "B", FakeModelService(), gpu, n_slots=2)
    assert eng2.run() == "done"
    assert [(r["kind"], r["decision"]) for r in eng2.records] == [("candidate", "replicate"), ("replicate", "accept")]
    assert gpu.submits[1]["kind"] == "replicate"


def test_resume_restores_the_rng_so_picks_do_not_change(tmp_path):
    full, _, _ = run(tmp_path / "full", "C", n_slots=4)
    part, svc = make_engine(tmp_path / "part", "C", FakeModelService(), FakeGPUService(noise_sd=0.006), n_slots=4)
    part.run_one_slot()
    part.run_one_slot()
    resumed, _ = make_engine(tmp_path / "part", "C", FakeModelService(), FakeGPUService(noise_sd=0.006), n_slots=4)
    resumed.run()
    assert [r["idea_id"] for r in resumed.records][:2] == [r["idea_id"] for r in full.records][:2]


# ------------------------------------------------------------------ arm C

def test_c_builds_a_backlog_seeds_posteriors_and_updates_them(tmp_path):
    eng, svc, _ = run(tmp_path, "C", gpu=FakeGPUService(), n_slots=6)
    s = eng.state
    roles = svc.models.roles()
    assert roles.count("planner_backlog") == 1 and roles.count("voter") == 90 and len(s["backlog"]) == 30
    assert s["dedup_mode"] == "ngram"
    done = [i for i in s["backlog"] if i["status"] == "done"]
    open_ = [i for i in s["backlog"] if i["status"] == "open"]
    assert len(done) == s["experiments"] and len(open_) == 30 - len(done)         # the backlog shrinks
    assert all(i["times_tried"] == 1 for i in done)
    # every open item still carries its prior: 1 + yes, 1 + no with 3 votes, so alpha + beta == 5
    assert all(i["alpha"] + i["beta"] == 5 for i in open_)
    # a done item's posterior moved by exactly one outcome
    assert all(i["alpha"] + i["beta"] == 6 for i in done)
    fam = s["family_stats"]
    assert sum(v["alpha"] + v["beta"] - 2 for v in fam.values()) == s["experiments"]
    assert {r["idea_id"] for r in eng.records} <= {i["id"] for i in s["backlog"]}


def test_c_is_deterministic_for_a_seed(tmp_path):
    a, _, _ = run(tmp_path / "x", "C", n_slots=5, gpu=FakeGPUService(seed=3))
    b, _, _ = run(tmp_path / "y", "C", n_slots=5, gpu=FakeGPUService(seed=3))
    assert [r["idea_id"] for r in a.records] == [r["idea_id"] for r in b.records]
    assert [r["decision"] for r in a.records] == [r["decision"] for r in b.records]


def test_c_refreshes_the_backlog_when_few_items_are_untried(tmp_path):
    eng, svc, _ = run(tmp_path, "C", n_slots=2, refresh_below=31)
    assert svc.models.roles().count("planner_refresh") >= 1 and len(eng.state["backlog"]) > 30


def test_c_drops_near_duplicate_ideas(tmp_path):
    a = {"title": "Raise matrix learning rate", "family": "optimizer",
         "spec": "Increase MATRIX_LR from 0.04 to 0.05 for faster early progress in training."}
    a2 = dict(a, spec=a["spec"].replace("0.05", "0.06"))
    b = {"title": "Add value embeddings", "family": "architecture",
         "spec": "Give every layer its own value embedding table to carry token identity."}
    eng, svc, _ = run(tmp_path, "C", {"planner_backlog": [{"items": [a, a2, b]}]}, n_slots=1)
    assert len(eng.state["backlog"]) == 2 and eng.state["dedup_dropped"] == 1


def test_c_widens_after_five_consecutive_rejections_and_writes_a_memo(tmp_path):
    models = FakeModelService(default_fx=(0.01,))                        # every candidate is worse
    eng, svc, _ = run(tmp_path, "C", models=models, n_slots=6)
    assert [r["decision"] for r in eng.records] == ["reject"] * 6
    assert svc.models.roles().count("planner_widen") == 1
    assert eng.state["consecutive_rejections"] == 1                      # reset by the widening, then one more reject
    assert svc.models.roles().count("analyst") == 1 and eng.state["memo"] == "notes 0"
    assert (eng.run_dir / "memo.md").read_text().strip() == "notes 0"
    assert "notes 0" in [c for c in svc.models.calls if c[0] == "coder"][-1][2]       # the memo reaches the coder


def test_c_pre_gpu_failures_leave_the_item_open_and_the_posterior_alone(tmp_path):
    eng, svc, _ = run(tmp_path, "C", {"coder": [abort_reply(), reply(0.01)]}, n_slots=1)
    ab, ok = eng.records
    assert ab["decision"] == "abort" and ab["charged"] is False and ab["job_id"] is None
    it = next(i for i in eng.state["backlog"] if i["id"] == ab["idea_id"])
    if ok["idea_id"] != ab["idea_id"]:      # the aborted idea is discounted, not scored: 1 + 3 votes + 1 prior
        assert (it["status"], it["times_tried"], it["alpha"] + it["beta"]) == ("open", 1, 5)
    assert eng.state["experiments"] == 1    # only the idea that ran counts as an experiment


def test_c_critic_vetoes_a_borderline_accept(tmp_path):
    veto = {"ok": False, "issues": ["suspicious"], "veto": True, "recomputed": {}}
    scripts = {"coder": [reply(-0.0085)], "critic": [veto]}
    eng, svc, _ = run(tmp_path, "C", scripts, n_slots=2)
    c, r = eng.records
    assert c["decision"] == "replicate" and c["z1"] == pytest.approx(0.0085 / ONE)
    assert r["z2"] == pytest.approx(0.0085 / TWO) and 1.5 <= r["z2"] < 2.0
    assert r["decision"] == "reject" and r["critic"] == {"ok": False, "issues": ["suspicious"], "veto": True}
    assert eng.state["incumbent"]["n"] == 24
    assert any(v["beta"] == 2 for v in eng.state["family_stats"].values())     # counted as a rejection


def test_c_critic_that_does_not_veto_lets_the_accept_through(tmp_path):
    eng, svc, _ = run(tmp_path, "C", {"coder": [reply(-0.0085)]}, n_slots=2)
    assert eng.records[1]["decision"] == "accept" and eng.records[1]["critic"]["veto"] is False
    b, svc_b, _ = run(tmp_path / "b", "B", {"base_step": [reply(-0.0085)]}, n_slots=2)
    assert b.records[1]["decision"] == "accept" and b.records[1]["critic"] is None and "critic" not in svc_b.models.roles()


def test_c_critic_reviews_a_lint_repair_before_gpu_time(tmp_path):
    scripts = {"coder": [violating_reply(-0.02)], "repairer": [remove_line_reply("import subprocess")]}
    eng, svc, _ = run(tmp_path, "C", scripts, n_slots=1)
    (r,) = eng.records
    assert r["critic"] == {"ok": True, "issues": [], "veto": False} and r["decision"] == "accept"
    assert svc.models.roles()[-3:] == ["coder", "repairer", "critic"] and len(svc.gpu.submits) == 1
    veto = {"ok": False, "issues": ["gaming"], "veto": True, "recomputed": {}}
    scripts = {"coder": [violating_reply(-0.02)], "repairer": [remove_line_reply("import subprocess")],
               "critic": [veto]}
    eng2, svc2, _ = run(tmp_path / "v", "C", scripts, n_slots=1)
    assert eng2.records[0]["decision"] == "reject" and eng2.records[0]["job_id"] is None
    assert eng2.records[0]["charged"] is False and eng2.records[0]["critic"]["veto"] is True
    assert eng2.records[1]["job_id"] is not None


def test_c_critic_reviews_every_nth_record(tmp_path):
    models = FakeModelService(default_fx=(0.01,))
    eng, svc, _ = run(tmp_path, "C", models=models, n_slots=3, critic_every=2)
    assert [r["critic"] is not None for r in eng.records] == [False, True, False]


def test_arms_a_to_c_ignore_routing_and_use_default_models(tmp_path):
    routing = {"coder": "qwen35-9b", "voter": "gemma4-12b", "critic": "qwen35-9b"}
    eng, svc, _ = run(tmp_path, "C", routing=routing, n_slots=2)
    used = {c[1] for c in svc.models.calls}
    assert used <= {"sonnet", "haiku", "opus"} and eng.state["designated_open_calls"] == 0


# ------------------------------------------------------------------ arm D

D_ROUTING = {   # the shape of loop/runs/calibration/routing_table.json (extra dict and list values are metadata)
    "coder_by_class": {"constant": "qwen35-9b", "schedule": "qwen35-9b", "structural": "sonnet"},
    "repairer": "sonnet", "planner_cheap": "qwen35-9b", "planner": "qwen35-9b", "analyst": "qwen35-9b",
    "voter": "gemma4-12b", "critic": "opus", "fallback": "sonnet", "qualified": {"qwen35-9b": ["constant"]},
    "notes": [], "thresholds": {"constant": 0.6}}
OPEN = ("qwen35-9b", "gemma4-12b")
CLASS = {"schedule": "schedule", "optimizer": "constant", "batch": "constant", "regularization": "constant",
         "architecture": "structural", "data": "structural", "other": "structural"}


def test_d_routes_roles_and_coder_by_edit_class_and_counts_served_models(tmp_path):
    models = FakeModelService(open_models=OPEN)
    eng, svc, _ = run(tmp_path, "D", models=models, routing=D_ROUTING, n_slots=6, gpu=FakeGPUService())
    coders = [c for c in models.calls if c[0] == "coder"]
    assert coders
    for _, model, user in coders:
        fam = re.search(r"Family: (\w+)", user).group(1)
        assert model == D_ROUTING["coder_by_class"][CLASS[fam]]
    assert {m for r, m, _ in models.calls if r == "voter"} == {"gemma4-12b"}
    s = eng.state
    assert s["served_counts"]["gemma4-12b"] == 90 and s["designated_open_calls"] >= 90
    assert s["open_served_calls"] == s["designated_open_calls"]                   # everything served by its open model
    assert s["served_counts"].get("opus", 0) == sum(1 for c in models.calls if c[0] == "critic")
    first_calls = eng.records[0]["model_calls"]
    assert all(c["requested_model"] == c["served_model"] == "gemma4-12b" for c in first_calls if c["role"] == "voter")


def test_d_keeps_the_initial_backlog_and_widening_on_sonnet_and_refreshes_on_the_cheap_planner(tmp_path):
    models = FakeModelService(open_models=OPEN, default_fx=(0.01,))
    eng, svc, _ = run(tmp_path, "D", models=models, routing=D_ROUTING, n_slots=6, refresh_below=31)
    by_role = {}
    for role, model, _ in models.calls:
        by_role.setdefault(role, set()).add(model)
    assert by_role["planner_backlog"] == {"sonnet"} and by_role["planner_widen"] == {"sonnet"}
    assert by_role["planner_refresh"] == {"qwen35-9b"} and by_role["analyst"] == {"qwen35-9b"}
    assert by_role.get("critic", {"opus"}) == {"opus"}


def test_d_counts_calls_that_fell_back_to_a_claude_model(tmp_path):
    models = FakeModelService(open_models=OPEN, serve={"gemma4-12b": "haiku"})   # the call layer hopped to the fallback
    eng, svc, _ = run(tmp_path, "D", models=models, routing=D_ROUTING, n_slots=2)
    s = eng.state
    assert s["served_counts"]["haiku"] == 90 and "gemma4-12b" not in s["served_counts"]
    assert s["designated_open_calls"] >= 90 and s["open_served_calls"] < s["designated_open_calls"]
    assert all(c["requested_model"] == "gemma4-12b" for c in eng.records[0]["model_calls"] if c["role"] == "voter")


def test_d_health_check_routes_to_the_fallback_and_logs_it(tmp_path):
    models = FakeModelService(open_models=OPEN, healthy={"qwen35-9b": False}, fallbacks={"qwen35-9b": "haiku"})
    eng, svc, _ = run(tmp_path, "D", models=models, routing=D_ROUTING, n_slots=3)
    assert "qwen35-9b" not in {m for _, m, _ in models.calls}
    assert [f["slot"] for f in eng.state["fallbacks"]] == [0, 1, 2]
    assert eng.state["fallbacks"][0] == {"slot": 0, "model": "qwen35-9b", "fallback": "haiku"}
    # designated for the open model, served by haiku
    assert eng.state["designated_open_calls"] > eng.state["open_served_calls"]


def test_d_without_health_support_skips_the_check(tmp_path):
    class NoHealth(FakeModelService):
        healthy = None

    models = NoHealth(open_models=OPEN)
    del NoHealth.healthy
    eng, svc, _ = run(tmp_path, "D", models=models, routing=D_ROUTING, n_slots=1)
    assert eng.state["fallbacks"] == []
