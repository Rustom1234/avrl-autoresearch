"""The curation loop on fake models and a fake trainer: the z rule, the repair call, failures and resume."""
import json, sys
from pathlib import Path
from types import SimpleNamespace

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent))
import engine as E

GOOD = "```python\nimport os\n# v%d\ndef select(docs, target_sample):\n    return list(range(len(docs)))\n```"


class FakeModels:
    """Planner replies with idea k; the coder's replies come from a list (default: a valid file tagged with k)."""
    def __init__(self, coder=None, fail_planner=(), infra=0):
        self.coder, self.fail_planner, self.calls, self.k, self.infra = list(coder or []), set(fail_planner), [], 0, infra

    def is_open(self, name):
        return name == "cheap"

    def call(self, role, model, user, system=None, schema=None, extra=None, slot=0, **kw):
        self.calls.append((role, model, slot, extra))
        if self.infra:
            self.infra -= 1
            return SimpleNamespace(ok=False, obj=None, text="", cost_usd_nominal=0.0, served_model=None, error_class="infra")
        ok, text, obj = True, "", None
        if role == "planner":
            self.k += 1
            ok = self.k not in self.fail_planner
            obj = {"idea": "idea %d" % self.k, "why": "because"} if ok else None
        else:
            text = self.coder.pop(0) if self.coder else GOOD % self.k
        return SimpleNamespace(ok=ok, obj=obj, text=text, cost_usd_nominal=0.01, served_model=model,
                               error_class=None if ok else "bad_json")


class FakeRunner:
    """dev bpb by file version: the starting file 2.00, version k from `dev` (default 2.00); None is a failed run."""
    def __init__(self, dev=None, rep=None):
        self.dev, self.rep, self.log = dev or {}, rep or {}, []

    def __call__(self, path, seed, tag):
        self.log.append((Path(path).name, seed, tag))
        m = __import__("re").search(r"# v(\d+)", Path(path).read_text())
        k = int(m.group(1)) if m else 0
        v = (self.rep if seed == E.SEED_B else self.dev).get(k, self.dev.get(k, 2.00))
        if v is None:
            return None, "RuntimeError: boom"
        return {"dev_bpb": v, "sealed_bpb": v + 0.02, "n_selected": 5, "curator_tokens_before_cycle": E.T}, None


def loop(tmp_path, slots=3, models=None, runner=None):
    cfg = {"loop_id": "L", "run_dir": str(tmp_path), "cell": "X", "pool": "v2", "slots": slots, "seed": 1, "sigma": 0.02,
           "gpu_host": None, "routing": {"planner": "strong", "coder": "cheap"}}
    return E.Loop(cfg, models=models or FakeModels(), runner=runner or FakeRunner())


def ledger(tmp_path):
    return [json.loads(l) for l in open(tmp_path / "ledger.jsonl")]


def test_lint_and_code_extraction():
    ok = "import os\ndef select(docs, target_sample):\n    return [int(os.environ.get('CURATE_SEED', 0))]\n"
    assert E.lint(ok) is None
    assert "SyntaxError" in E.lint("def select(:")
    assert "must define" in E.lint("x = 1\n")
    for bad in ("open('a')", "import sys\nsys.argv[2]", "import subprocess", "np.load('x')", "os.path.join('a')"):
        assert "not allowed" in E.lint(ok + bad + "\n"), bad
    assert E.code_of("a\n```python\nx = 1\n```\nb\n```py\ny = 2\n```") == "y = 2\n"
    assert E.code_of("z = 3") == "z = 3\n"


def test_z_rule_accept_reject_and_grey_zone(tmp_path):
    # sigma 0.02: idea 1 at 1.95 (z 2.5) accept; idea 2 at 1.94 against 1.95 (z 0.5) reject;
    # idea 3 at 1.92 (z 1.5) grey: second seed 1.91 against incumbent 1.95, mean z 1.75, accept
    run = FakeRunner(dev={1: 1.95, 2: 1.94, 3: 1.92}, rep={1: 1.95, 3: 1.91})
    lp = loop(tmp_path, runner=run)
    final = lp.run()
    recs = ledger(tmp_path)
    assert [r["decision"] for r in recs] == ["accept", "reject", "accept"]
    assert recs[0]["z1"] == pytest.approx(2.5) and "z_mean" not in recs[0]
    assert recs[2]["z_mean"] == pytest.approx(1.75)
    assert final["n_accept"] == 2 and final["n_confirm"] == 2 and final["sealed_bpb"] == pytest.approx(1.94)
    assert final["model_usd"] == pytest.approx(0.06) and Path(final["incumbent"]).name == "s03_0.py"
    # the planner sees dev numbers and outcomes, never the sealed score
    assert all("sealed" not in h["outcome"] for h in lp.s["history"])
    # the open model gets the no-thinking switch, the Claude model does not
    assert all((e is not None) == (role == "coder") for role, _, _, e in lp.models.calls)


def test_grey_zone_reject(tmp_path):
    run = FakeRunner(dev={1: 1.97}, rep={1: 2.00})        # z 1.5, then mean z 0.75: reject
    loop(tmp_path, slots=1, runner=run).run()
    r = ledger(tmp_path)[0]
    assert r["decision"] == "reject" and r["z_mean"] == pytest.approx(0.75)


def test_repair_call_and_failures(tmp_path):
    bad = "```python\ndef select(docs, target_sample):\n    return open('x')\n```"
    models = FakeModels(coder=[bad, GOOD % 1, bad, bad], fail_planner={3})
    run = FakeRunner(dev={1: 1.90})
    lp = loop(tmp_path, models=models, runner=run)
    lp.run()
    recs = ledger(tmp_path)
    assert [r["decision"] for r in recs] == ["accept", "fail", "no_idea"]
    assert [c["role"] for c in recs[0]["calls"]] == ["planner", "coder", "coder"]      # one repair call
    assert "not allowed" in recs[1]["error"] and lp.s["slot"] == 3
    assert not any(name.startswith("s02") for name, _, _ in run.log)                   # a file that fails the lint never runs
    assert "FAILED" in lp.s["history"][1]["outcome"]


def test_crash_then_repair_runs_again(tmp_path):
    models = FakeModels(coder=[GOOD % 7, GOOD % 1])
    run = FakeRunner(dev={7: None, 1: 1.90})
    loop(tmp_path, slots=1, models=models, runner=run).run()
    r = ledger(tmp_path)[0]
    assert r["decision"] == "accept" and r["cand"].endswith("s01_1.py")


def test_resume_continues_from_state(tmp_path):
    run = FakeRunner(dev={1: 1.95})
    lp = loop(tmp_path, slots=2, runner=run)
    lp.inc_score(E.SEED_A)
    lp.run_slot()
    lp2 = loop(tmp_path, slots=2, runner=run)
    assert lp2.s["slot"] == 1 and Path(lp2.s["incumbent"]).name == "s01_0.py"
    lp2.models.k = 1
    lp2.run()
    assert len(ledger(tmp_path)) == 2


def test_provider_failure_is_waited_out(tmp_path):
    lp = loop(tmp_path, slots=1, models=FakeModels(infra=2), runner=FakeRunner(dev={1: 1.90}))
    waits = []
    lp.sleep = waits.append
    final = lp.run()
    assert ledger(tmp_path)[0]["decision"] == "accept" and waits == [E.INFRA_WAIT_S] * 2 and final["infra_failures"] == 2
