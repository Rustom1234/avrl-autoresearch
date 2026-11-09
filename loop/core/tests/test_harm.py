"""Harm control (P2_PREREG): with probability harm_rate the coder's applied edit is replaced by the incumbent plus one
comment line, decided by a hash of (rng_seed, slot, attempt). harm_rate 0 must change nothing."""
import hashlib
import json

import pytest

from arms import HARM_LINE
from fakes import FakeGPUService, FakeModelService, abort_reply, make_engine, remove_line_reply, reply, violating_reply


def hit(seed, slot, attempt, rate):
    """The decision rule from the P2 spec, written out independently of the engine."""
    h = int(hashlib.sha256(("%s:%s:%s" % (seed, slot, attempt)).encode()).hexdigest(), 16)
    return h % 1_000_000 < rate * 1_000_000


def run(tmp_path, arm="D", scripts=None, name="run", **over):
    eng, svc = make_engine(tmp_path / name, arm, FakeModelService(scripts), FakeGPUService(noise_sd=0), **over)
    eng.run()
    return eng, svc


def slot_files(eng):
    return {str(p.relative_to(eng.run_dir)): p.read_bytes() for p in sorted(eng.run_dir.rglob("*")) if p.is_file()
            and p.name not in ("state.json", "run_loop.log")}


def test_harm_rate_zero_changes_nothing(tmp_path):
    a, _ = run(tmp_path, name="unset", n_slots=5)
    b, _ = run(tmp_path, name="zero", n_slots=5, harm_rate=0.0)
    fa, fb = slot_files(a), slot_files(b)
    assert fa.keys() == fb.keys() and not any("harm" in k for k in fa)
    for k in fa:   # run_dir differs between the two runs, so compare file bytes (ledger rows hold no path)
        assert fa[k] == fb[k], k
    sa, sb = dict(a.state), dict(b.state)
    for s in (sa, sb):
        for k in ("incumbent", "started", "updated"):
            s.pop(k)
    assert sa == sb and a.state["harm_noops"] == [] == b.state["harm_noops"]


def test_every_edit_is_replaced_at_rate_one(tmp_path):
    eng, svc = run(tmp_path, n_slots=4, harm_rate=1.0)
    base = open(eng.cfg["base_train_py"]).read()
    noop = base + HARM_LINE + "\n"
    assert base.endswith("\n") and noop != base
    for k in range(4):
        d = eng.run_dir / "slots" / str(k)
        assert (d / "train.py").read_text() == noop
        h = json.loads((d / "harm.json").read_text())
        orig = (d / "harm_original_train.py").read_text()
        assert set(h) == {"slot", "attempt", "idea_id", "item_id", "coder_edit_sha256"}
        assert h["slot"] == k and h["attempt"] == 0 and h["idea_id"] == h["item_id"]
        assert h["coder_edit_sha256"] == hashlib.sha256(orig.encode()).hexdigest()
        assert orig != base and "# fx:" in orig                       # the coder's real edit is kept for the record
    assert [(n["slot"], n["attempt"]) for n in eng.state["harm_noops"]] == [(k, 0) for k in range(4)]
    assert all(set(n) == {"slot", "attempt", "idea_id"} for n in eng.state["harm_noops"])
    for r in eng.records:    # lint and the GPU run proceed as for any candidate, and a comment cannot win
        assert r["lint"]["ok"] is True and r["job_id"] and r["decision"] == "reject" and r["charged"]
    assert [i["status"] for i in eng.state["backlog"] if i["times_tried"]] == ["done"] * 4   # posterior updated
    fam = eng.state["family_stats"].values()                                                   # recorded as failed
    assert sum(v["beta"] - 1 for v in fam) == 4 and all(v["alpha"] == 1 for v in fam)


def test_replacement_is_a_valid_distinct_program_even_without_a_trailing_newline(tmp_path):
    eng, _ = make_engine(tmp_path, "D", FakeModelService(), FakeGPUService(noise_sd=0), harm_rate=1.0)
    src = "x = 1"
    out = eng._harm("y = 2\n", src, {"idea_id": "i001", "item_id": "i001"})
    assert out == "x = 1\n" + HARM_LINE + "\n" and out != src
    compile(out, "t", "exec")
    again = eng._harm("z = 3\n", out, {"idea_id": "i002", "item_id": "i002"})   # an accepted no-op can be the incumbent
    assert again == out + HARM_LINE + "\n" and again != out


def test_a_repeated_attempt_after_a_resume_counts_one_no_op(tmp_path):
    eng, _ = make_engine(tmp_path, "D", FakeModelService(), FakeGPUService(noise_sd=0), harm_rate=1.0)
    ctx = {"idea_id": "i001", "item_id": "i001"}
    eng._harm("y = 2\n", "x = 1\n", ctx)
    eng._harm("z = 3\n", "x = 1\n", ctx)                       # same (slot, attempt) again: the engine was killed and resumed
    assert [(n["slot"], n["attempt"]) for n in eng.state["harm_noops"]] == [(0, 0)]
    eng.state["attempt"] = 1                                     # the next attempt is a different one
    eng._harm("w = 4\n", "x = 1\n", ctx)
    assert [(n["slot"], n["attempt"]) for n in eng.state["harm_noops"]] == [(0, 0), (0, 1)]
    saved = json.loads((eng.run_dir / "state.json").read_text())
    assert saved["harm_noops"] == eng.state["harm_noops"]


def test_decision_uses_the_hash_and_about_half_at_rate_one_half(tmp_path):
    seed, rate = 7, 0.5
    expect = [hit(seed, k, 0, rate) for k in range(12)]
    eng, _ = run(tmp_path, n_slots=12, harm_rate=rate, rng_seed=seed)
    got = [(tmp_path / "run" / "run_D" / "slots" / str(k) / "harm.json").exists() for k in range(12)]
    assert got == expect and 0 < sum(got) < 12
    assert [n["slot"] for n in eng.state["harm_noops"]] == [k for k, g in enumerate(got) if g]
    n = sum(hit(seed, s, a, rate) for s in range(50) for a in range(40))
    assert 0.45 < n / 2000 < 0.55
    eng2, _ = run(tmp_path, name="again", n_slots=12, harm_rate=rate, rng_seed=seed)   # same seed, same hits
    assert eng2.state["harm_noops"] == eng.state["harm_noops"]


def test_repairs_and_crash_fixes_are_never_replaced(tmp_path):
    rate = 0.5
    seed = next(s for s in range(500) if not hit(s, 0, 0, rate) and hit(s, 1, 0, rate) and hit(s, 0, 1, rate))
    # slot 0: a candidate that fails lint and is repaired (the hash misses, and would hit again on attempt 1)
    scripts = {"coder": [violating_reply(0.0)], "repairer": [remove_line_reply("import subprocess")]}
    eng, svc = run(tmp_path, name="repair", scripts=scripts, n_slots=1, harm_rate=rate, rng_seed=seed)
    assert "repairer" in svc.models.roles() and eng.state["harm_noops"] == []
    assert "import subprocess" not in (eng.run_dir / "slots" / "0" / "train.py").read_text()
    # slot 0 crashes, slot 1 is its fix; the hash would hit at (1, 0) but a crash fix is not a new-idea attempt
    scripts = {"coder": [reply(0.0, "CRASH_ME")], "repairer": [reply(0.0, "fixed")]}
    eng, svc = run(tmp_path, name="crash", scripts=scripts, n_slots=2, harm_rate=rate, rng_seed=seed)
    assert [r["kind"] for r in eng.records][:2] == ["candidate", "crash_fix"]
    assert eng.state["harm_noops"] == [] and not (eng.run_dir / "slots" / "1" / "harm.json").exists()
    assert "# fixed" in (eng.run_dir / "slots" / "1" / "train.py").read_text()


def test_resumed_state_without_the_key_gets_it(tmp_path):
    eng, _ = run(tmp_path, n_slots=1)
    st = json.loads((eng.run_dir / "state.json").read_text())
    del st["harm_noops"]
    (eng.run_dir / "state.json").write_text(json.dumps(st))
    eng2, _ = make_engine(tmp_path / "run", "D", FakeModelService(), FakeGPUService(noise_sd=0), n_slots=1)
    assert eng2.state["harm_noops"] == []


def test_an_abort_is_not_replaced_and_the_next_attempt_is(tmp_path):
    eng, _ = run(tmp_path, scripts={"coder": [abort_reply()]}, n_slots=1, harm_rate=1.0)
    assert [r["decision"] for r in eng.records] == ["abort", "reject"]
    assert [(n["slot"], n["attempt"]) for n in eng.state["harm_noops"]] == [(0, 1)]
