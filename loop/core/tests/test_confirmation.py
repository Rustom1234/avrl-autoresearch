import json
from pathlib import Path

import pytest

import confirmation as C

LOOPS = [("A_01", "A", "done"), ("D_01", "D", "done"), ("B_01", "B", "stopped"), ("C_01", "C", "stopped"),
         ("A_02", "A", "running")]
IMP = {"A_01": 0.010, "D_01": 0.004, "B_01": 0.006, "C_01": 0.002}  # true sealed improvement used by the fake


def anchor_vals(wave):
    return 1.140 + 0.001 * wave, 1.150 + 0.001 * wave  # official, sealed (offset official minus sealed = -0.010)


class FakeClient:
    """Stands in for QueueClient: no ssh. fail(meta, seed, n) may return a status string, "NONE" (no result), or None."""

    def __init__(self, fail=None):
        self.submitted, self.jobs, self.outstanding, self.max_outstanding = [], {}, set(), 0
        self.fail, self.seen = fail or (lambda meta, seed, n: None), {}

    def submit(self, train_py, **kw):
        jid = "job%03d" % len(self.submitted)
        self.jobs[jid] = job = dict(kw, job_id=jid, train_py=str(train_py))
        self.submitted.append(job)
        self.outstanding.add(jid)
        self.max_outstanding = max(self.max_outstanding, len(self.outstanding))
        return jid

    def wait(self, job_ids, timeout_s, poll_interval=20):
        out = {}
        for jid in job_ids:
            self.outstanding.discard(jid)
            job = self.jobs[jid]
            meta, seed = job["meta"], job["seed"]
            key = (meta["wave"], meta["loop_id"], seed)
            self.seen[key] = n = self.seen.get(key, 0) + 1
            bad = self.fail(meta, seed, n)
            if bad == "NONE":
                out[jid] = None
            elif bad:
                out[jid] = {"job_id": jid, "status": bad, "official_val_bpb": None, "sealed_val_bpb": None, "gpu": 1}
            else:
                if meta["kind"] == "anchor":
                    official, sealed = anchor_vals(meta["wave"])
                else:
                    sealed = anchor_vals(meta["wave"])[1] - IMP[meta["loop_id"]] + 0.0005 * (seed - 1001)
                    official = sealed - 0.010
                out[jid] = {"job_id": jid, "status": "ok", "official_val_bpb": official, "sealed_val_bpb": sealed,
                            "gpu": len(self.submitted) % 4}
        return out


@pytest.fixture(autouse=True)
def no_network(monkeypatch):
    def boom():
        raise AssertionError("a test tried to build a real QueueClient")
    monkeypatch.setattr(C, "make_client", boom)


@pytest.fixture
def world(tmp_path):
    root = tmp_path / "root"
    loops = []
    for lid, arm, status in LOOPS:
        run_dir = root / lid
        run_dir.mkdir(parents=True)
        (run_dir / "incumbent.py").write_text("# %s\n" % lid)
        (run_dir / "state.json").write_text(json.dumps(
            {"loop_id": lid, "status": status, "incumbent": {"path": str(run_dir / "incumbent.py")}}))
        loops.append({"loop_id": lid, "arm": arm, "run_dir": str(run_dir)})
    return root, {"seed": 7, "loops": loops}


def rows_of(root):
    return C.read_jsonl(root / "confirmation" / "results.jsonl")


def test_collect_incumbents_only_done_or_stopped(world):
    root, plan = world
    found, skipped = C.collect_incumbents(plan)
    assert [f["loop_id"] for f in found] == ["A_01", "D_01", "B_01", "C_01"]
    assert skipped == ["A_02"]


def test_collect_incumbents_relative_path_and_missing_file(world):
    root, plan = world
    ss = root / "A_01" / "state.json"
    ss.write_text(json.dumps({"status": "done", "incumbent": {"path": "incumbent.py"}}))
    found, _ = C.collect_incumbents(plan)
    assert found[0]["train_py"] == str(root / "A_01" / "incumbent.py")
    (root / "A_01" / "incumbent.py").unlink()
    with pytest.raises(FileNotFoundError):
        C.collect_incumbents(plan)


def test_wave_composition(world):
    root, plan = world
    C.run_confirmation(root, plan, client=FakeClient())
    waves = json.loads((root / "confirmation" / "waves.json").read_text())
    assert len(waves) == 6  # 16 confirmations in groups of 3: five full waves and one with a single confirmation
    for w in waves[:-1]:
        assert sorted(j["kind"] for j in w["jobs"]) == ["anchor", "confirm", "confirm", "confirm"]
    assert sorted(j["kind"] for j in waves[-1]["jobs"]) == ["anchor", "confirm"]
    confirms = [(j["loop_id"], j["seed"]) for w in waves for j in w["jobs"] if j["kind"] == "confirm"]
    assert sorted(confirms) == sorted((lid, s) for lid in IMP for s in (1001, 1002, 1003, 1004))
    assert all(j["train_py"] == str(C.BASE_TRAIN) for w in waves for j in w["jobs"] if j["kind"] == "anchor")
    mixed = sum(1 for w in waves if len({j["arm"] for j in w["jobs"] if j["kind"] == "confirm"}) > 1)
    assert mixed >= 4  # arms are shuffled across waves, not run loop by loop


def test_waves_deterministic_for_a_seed(world):
    root, plan = world
    found, _ = C.collect_incumbents(plan)
    assert C.build_waves(found, 7) == C.build_waves(found, 7)
    assert C.build_waves(found, 7) != C.build_waves(found, 8)


def test_queue_usage_sealed_and_at_most_four_in_flight(world):
    root, plan = world
    client = FakeClient()
    C.run_confirmation(root, plan, client=client)
    assert len(client.submitted) == 22  # 16 confirmations + 6 wave anchors
    assert client.max_outstanding <= 4
    assert all(j["sealed"] is True for j in client.submitted)
    assert all("gpu_affinity" not in j for j in client.submitted)
    rows = rows_of(root)
    assert len(rows) == 22
    need = {"wave", "loop_id", "arm", "seed", "kind", "official_val_bpb", "sealed_val_bpb", "gpu", "status", "job_id"}
    assert all(need <= set(r) for r in rows)
    assert sum(r["kind"] == "anchor" for r in rows) == 6


def test_summary_arithmetic(world):
    root, plan = world
    summary = C.run_confirmation(root, plan, client=FakeClient())
    assert json.loads((root / "confirmation" / "summary.json").read_text()) == summary
    for lid, imp in IMP.items():
        d = summary["loops"][lid]
        assert d["n_ok"] == d["n_paired"] == d["n_runs"] == 4
        # sealed = anchor_sealed(wave) - imp + 0.0005 * (seed - 1001), so improvement = imp - 0.0005 * mean(0..3)
        assert d["improvement"] == pytest.approx(imp - 0.0005 * 1.5)
        assert d["improvement_official"] == pytest.approx(imp - 0.0005 * 1.5)
        rows = [r for r in rows_of(root) if r["loop_id"] == lid]
        assert d["mean_sealed"] == pytest.approx(sum(r["sealed_val_bpb"] for r in rows) / 4)
        assert d["mean_official"] == pytest.approx(d["mean_sealed"] - 0.010)
    assert summary["anchor"]["offset_official_minus_sealed"] == pytest.approx(-0.010)
    assert summary["anchor"]["n_ok"] == 6


def test_summarize_hand_computed():
    def row(wave, kind, lid, seed, official, sealed, status="ok"):
        return {"wave": wave, "kind": kind, "loop_id": lid, "arm": "A" if lid else None, "seed": seed,
                "official_val_bpb": official, "sealed_val_bpb": sealed, "gpu": 0, "status": status, "job_id": "j"}
    rows = [row(1, "anchor", None, 1001, 1.140, 1.150), row(1, "confirm", "X", 1001, 1.131, 1.140),
            row(2, "anchor", None, 1002, 1.145, 1.154), row(2, "confirm", "X", 1002, 1.140, 1.146),
            row(3, "anchor", None, 1003, None, None, status="crash"),  # wave 3 has no usable anchor
            row(3, "confirm", "X", 1003, 1.136, 1.143), row(3, "confirm", "X", 1004, None, None, status="timeout")]
    x = C.summarize(rows)["loops"]["X"]
    assert (x["n_runs"], x["n_ok"], x["n_paired"]) == (4, 3, 2)
    assert x["mean_sealed"] == pytest.approx((1.140 + 1.146 + 1.143) / 3)
    assert x["mean_official"] == pytest.approx((1.131 + 1.140 + 1.136) / 3)
    assert x["improvement"] == pytest.approx(((1.150 - 1.140) + (1.154 - 1.146)) / 2)  # 0.009
    assert x["improvement_official"] == pytest.approx(((1.140 - 1.131) + (1.145 - 1.140)) / 2)
    a = C.summarize(rows)["anchor"]
    assert (a["n_ok"], a["n_failed"]) == (2, 1)
    assert a["offset_official_minus_sealed"] == pytest.approx(((1.140 - 1.150) + (1.145 - 1.154)) / 2)  # -0.0095


@pytest.mark.parametrize("recorded, expected_new", [(0, 22), (8, 14), (6, 16), (22, 0)])
def test_resume_skips_recorded_jobs(world, recorded, expected_new):
    root, plan = world
    full = FakeClient()
    C.run_confirmation(root, plan, client=full)
    path = root / "confirmation" / "results.jsonl"
    lines = path.read_text().splitlines()
    path.write_text("".join(l + "\n" for l in lines[:recorded]))
    again = FakeClient()
    summary = C.run_confirmation(root, plan, client=again)
    assert len(again.submitted) == expected_new
    assert len(rows_of(root)) == 22  # nothing recorded twice
    assert len({(r["wave"], r["kind"], r["loop_id"], r["seed"]) for r in rows_of(root)}) == 22
    assert summary["loops"]["A_01"]["n_ok"] == 4


def test_resume_does_not_rebuild_waves_when_more_loops_finish(world):
    root, plan = world
    C.run_confirmation(root, plan, client=FakeClient())
    (root / "A_02" / "incumbent.py").write_text("# a02\n")
    (root / "A_02" / "state.json").write_text(json.dumps(
        {"status": "done", "incumbent": {"path": str(root / "A_02" / "incumbent.py")}}))
    again = FakeClient()
    summary = C.run_confirmation(root, plan, client=again)
    assert again.submitted == [] and "A_02" not in summary["loops"]  # the wave plan was frozen by the first call


def test_failed_job_is_retried_once_then_recorded(world):
    root, plan = world

    def fail(meta, seed, n):
        if (meta["loop_id"], seed) == ("A_01", 1001):
            return "crash" if n == 1 else None        # recovers on the retry
        if (meta["loop_id"], seed) == ("D_01", 1002):
            return "infra_fail"                       # fails both attempts
        if (meta["loop_id"], seed) == ("B_01", 1003):
            return "NONE" if n == 1 else None         # no result within the wave timeout, then fine
    client = FakeClient(fail)
    summary = C.run_confirmation(root, plan, client=client)
    assert len(client.submitted) == 22 + 3  # one resubmission per first failure
    assert client.max_outstanding <= 4
    rows = {(r["loop_id"], r["seed"]): r for r in rows_of(root) if r["kind"] == "confirm"}
    assert (rows[("A_01", 1001)]["status"], rows[("A_01", 1001)]["attempt"]) == ("ok", 2)
    assert (rows[("B_01", 1003)]["status"], rows[("B_01", 1003)]["attempt"]) == ("ok", 2)
    bad = rows[("D_01", 1002)]
    assert (bad["status"], bad["attempt"], bad["official_val_bpb"], bad["sealed_val_bpb"]) == ("infra_fail", 2, None, None)
    assert summary["loops"]["D_01"]["n_ok"] == 3 and summary["loops"]["D_01"]["n_runs"] == 4
    expect = [IMP["D_01"] - 0.0005 * (s - 1001) for s in (1001, 1003, 1004)]
    assert summary["loops"]["D_01"]["improvement"] == pytest.approx(sum(expect) / 3)


def test_nothing_to_confirm_exits(tmp_path):
    plan = {"seed": 1, "loops": [{"loop_id": "A_01", "arm": "A", "run_dir": str(tmp_path / "A_01")}]}
    with pytest.raises(SystemExit):
        C.run_confirmation(tmp_path, plan, client=FakeClient())
