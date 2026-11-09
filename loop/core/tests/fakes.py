"""Fakes for the engine tests: scripted model service, synthetic GPU service, fake clock. No network, no sleeping.

Candidate files carry their own ground truth as comment lines: `# fx:<float>` adds that much to the true val_bpb
(cumulative over everything accepted so far), `# CRASH_ME` makes the fake worker crash, `# LEAK_ME` makes it report a
target leak. Edits are SEARCH/REPLACE blocks anchored on a line that never changes, so they always apply.
"""
import json
import random
import re
import string
from dataclasses import dataclass, field
from pathlib import Path

BASE = Path(__file__).resolve().parents[2] / "substrate" / "base" / "train_ampere.py"
ANCHOR = "WARMUP_RATIO = 0.0      # fraction of time budget for LR warmup"
FAMILIES = ("optimizer", "schedule", "architecture", "batch", "regularization", "data", "other")
DEFAULT_FX = (-0.003, 0.002, -0.011, 0.005, -0.001, -0.02, 0.004, -0.002)


def edit_text(fx=0.0, *markers, tag=""):
    """SEARCH/REPLACE text appending `# fx:` and marker comment lines after the anchor line."""
    lines = ["# fx:%r %s" % (fx, tag)] + ["# %s" % m for m in markers]
    return "<<<<<<< SEARCH\n%s\n=======\n%s\n%s\n>>>>>>> REPLACE\n" % (ANCHOR, ANCHOR, "\n".join(lines))


def reply(fx=0.0, *markers, desc=None, tag=""):
    return {"description": desc or "fx %r" % fx, "edits_text": edit_text(fx, *markers, tag=tag),
            "abort": False, "abort_reason": ""}


def bad_search_reply():
    return {"description": "bad", "abort": False, "abort_reason": "",
            "edits_text": "<<<<<<< SEARCH\nTHIS_LINE_DOES_NOT_EXIST = 1\n=======\nX = 2\n>>>>>>> REPLACE\n"}


def abort_reply():
    return {"description": "nothing worth a slot", "edits_text": "", "abort": True, "abort_reason": "no idea"}


def violating_reply(fx=0.0):
    """Adds `import subprocess` (a hard lint failure) plus an fx comment."""
    anchor = ANCHOR
    return {"description": "violating", "abort": False, "abort_reason": "",
            "edits_text": "<<<<<<< SEARCH\n%s\n=======\n%s\n# fx:%r\nimport subprocess\n>>>>>>> REPLACE\n"
                          % (anchor, anchor, fx)}


def remove_line_reply(line):
    return {"description": "repair", "abort": False, "abort_reason": "",
            "edits_text": "<<<<<<< SEARCH\n%s\n=======\n>>>>>>> REPLACE\n" % line}


class Infra:
    """Scripted infrastructure failure of a model call."""


class ModelFail:
    """Scripted non-infra model failure (schema never satisfied)."""


@dataclass
class FakeResult:
    obj: object = None
    text: str = ""
    ok: bool = True
    error: str = None
    error_class: str = None
    requested_model: str = ""
    served_model: str = ""
    usage: dict = field(default_factory=lambda: {"input": 1000, "output": 200, "cache_read": 0, "cache_write": 0})
    cost_usd_plan: float = 0.01
    cost_usd_nominal: float = 0.01
    latency_s: float = 1.5


def _rand_text(i, n=40):
    return "".join(random.Random(i).choices(string.ascii_lowercase, k=n))


class FakeModelService:
    """scripts: {role: [reply, ...]} consumed in order; a reply is a dict (the parsed object), Infra(), ModelFail(),
    or a callable(user) -> one of those. When a role's queue is empty a default generator answers."""

    def __init__(self, scripts=None, open_models=(), healthy=None, serve=None, fallbacks=None, default_fx=DEFAULT_FX):
        self.scripts = {k: list(v) for k, v in (scripts or {}).items()}
        self.calls = []                      # (role, model, user)
        self.open_models, self.health = set(open_models), healthy or {}
        self.serve, self.fallbacks, self.default_fx = serve or {}, fallbacks or {}, default_fx
        self.n = {}                          # per-role default counters
        self._item = 0

    # -- interface used by the engine
    def is_open(self, model):
        return model in self.open_models

    def healthy(self, model):
        return self.health.get(model, True)

    def fallback_of(self, model):
        return self.fallbacks.get(model)

    def call(self, role, model_name, user, system=None, schema=None, temperature=None, extra=None, slot=0):
        self.calls.append((role, model_name, user))
        q = self.scripts.get(role)
        item = q.pop(0) if q else self._default(role, user)
        if callable(item):
            item = item(user)
        served = self.serve.get(model_name, model_name)
        if isinstance(item, Infra):
            return FakeResult(ok=False, error="unreachable", error_class="infra", requested_model=model_name,
                              served_model=served, cost_usd_plan=0.0, cost_usd_nominal=0.0)
        if isinstance(item, ModelFail):
            return FakeResult(ok=False, error="schema", error_class="model", requested_model=model_name,
                              served_model=served)
        open_ = served in self.open_models
        return FakeResult(obj=item, text=json.dumps(item), requested_model=model_name, served_model=served,
                          cost_usd_plan=0.0 if open_ else 0.01, cost_usd_nominal=0.001 if open_ else 0.01)

    def sample_k(self, role, model_name, user, k, key, system=None, schema=None, temperature=None, extra=None,
                 slot=0):
        results = [self.call(role, model_name, user, slot=slot) for _ in range(k)]
        values = [r.obj[key] for r in results if r.ok]
        return {"results": results, "values": values, "agreement": 1.0, "modal_value": None, "n_ok": len(values)}

    # -- defaults
    def _count(self, role):
        self.n[role] = self.n.get(role, -1) + 1
        return self.n[role]

    def _default(self, role, user):
        i = self._count(role)
        if role in ("base_step", "coder"):
            return reply(self.default_fx[i % len(self.default_fx)], desc="default edit %d" % i, tag="d%d" % i)
        if role in ("planner", "planner_backlog", "planner_refresh", "planner_widen"):
            n = 30 if role in ("planner", "planner_backlog") else 10
            items = []
            for _ in range(n):
                self._item += 1
                items.append({"title": "idea %d" % self._item, "family": FAMILIES[self._item % len(FAMILIES)],
                              "spec": "change knob %s" % _rand_text(self._item)})
            return {"items": items}
        if role == "repairer":
            return reply(0.0, desc="default repair", tag="r%d" % i)
        if role == "voter":
            return {"vote": "no" if i % 3 == 0 else "yes", "reason": "fake"}
        if role == "analyst":
            return {"memo": "notes %d" % i}
        if role == "critic":
            return {"ok": True, "issues": [], "veto": False, "recomputed": {}}
        raise AssertionError("no scripted reply for role %r" % role)

    def roles(self):
        return [c[0] for c in self.calls]


class Killed(BaseException):
    """Raised by the fake GPU to simulate the engine process dying while it waits for a job."""


class FakeGPUService:
    """Synthetic worker. official = anchor_mean + gpu offset + sum of `# fx:` tags + N(0, noise_sd).
    bias: {submission_number: extra added to official}; patch: {submission_number: dict merged into the result};
    infra_on: submission numbers that come back as infra_fail; kill_on_wait: submission number whose wait() raises
    Killed once (then behaves normally). Submission numbers start at 1."""

    def __init__(self, anchor_mean=1.1436, offsets=None, noise_sd=0.006, seed=0, bias=None, patch=None,
                 infra_on=(), kill_on_wait=None, sealed_gap=0.002):
        self.anchor, self.offsets = anchor_mean, offsets or {}
        self.noise_sd, self.rng = noise_sd, random.Random(seed)
        self.bias, self.patch, self.infra_on = bias or {}, patch or {}, set(infra_on)
        self.kill_on_wait, self.sealed_gap = kill_on_wait, sealed_gap
        self.submits, self.jobs, self.waits = [], {}, []

    def submit(self, train_py_path, *, seed, sealed, tag, kind, gpu_affinity=None):
        n = len(self.submits) + 1
        src = Path(train_py_path).read_text()
        job_id = "job%03d" % n
        self.submits.append({"n": n, "path": str(train_py_path), "seed": seed, "sealed": sealed, "tag": tag,
                             "kind": kind, "job_id": job_id})
        gpu = (n - 1) % 4
        fx = sum(float(x) for x in re.findall(r"# fx:(\S+)", src))
        res = {"job_id": job_id, "status": "ok", "gpu": gpu, "seed": seed, "sealed": sealed, "tag": tag,
               "mismatch_flag": False, "target_leak_flag": False, "canary_ok": True, "harness_missing": False,
               "seed_patched": True, "seed_lines_matched": 2, "overbudget": False, "load_flagged": False,
               "harness": {"constants_ok": True, "prepare_code_hash_ok": True, "pipeline_code_hash_ok": True},
               "printed": {"training_seconds": 300.1, "total_seconds": 330.0, "peak_vram_mb": 22000.0,
                           "num_steps": 318, "num_params_M": 50.3},
               "log_tail": ["step 318 done", "---", "val_bpb: 1.0"], "wall_seconds": 400.0}
        noise = self.rng.gauss(0.0, self.noise_sd) if self.noise_sd else 0.0
        official = self.anchor + self.offsets.get(gpu, 0.0) + fx + noise + self.bias.get(n, 0.0)
        res["official_val_bpb"] = official
        res["sealed_val_bpb"] = official + self.sealed_gap if sealed else None
        if "CRASH_ME" in src:
            res.update(status="crash", official_val_bpb=None, sealed_val_bpb=None, exit_code=1,
                       log_tail=["step 10", "Traceback (most recent call last):",
                                 '  File "train.py", line 10, in <module>', "RuntimeError: boom"])
        if "LEAK_ME" in src:
            res.update(target_leak_flag=True, official_val_bpb=None, sealed_val_bpb=None)
        if n in self.infra_on:
            res = {"job_id": job_id, "status": "infra_fail", "official_val_bpb": None, "sealed_val_bpb": None}
        res.update(self.patch.get(n, {}))
        self.jobs[job_id] = res
        return job_id

    def wait(self, job_id, timeout_s):
        self.waits.append(job_id)
        if self.kill_on_wait is not None and job_id == "job%03d" % self.kill_on_wait:
            self.kill_on_wait = None
            raise Killed(job_id)
        return self.jobs.get(job_id)


class FakeClock:
    def __init__(self):
        self.t, self.sleeps = 1_800_000_000.0, []

    def now(self):
        self.t += 1.0
        return self.t

    def sleep(self, s):
        self.sleeps.append(s)
        self.t += s


@dataclass
class FakeServices:
    models: object
    gpu: object
    clock: object


def make_config(tmp_path, arm, **over):
    cfg = {"loop_id": "t_%s" % arm, "arm": arm, "run_dir": str(tmp_path / ("run_" + arm)), "n_slots": 6,
           "rng_seed": 7, "routing": {}, "sigma_used": 0.0067, "gpu_offsets": {}, "anchor": {"mean": 1.1436, "n": 24},
           "base_train_py": str(BASE), "stop_file": str(tmp_path / "STOP"), "dedup_mode": "ngram",
           "free_action_caps": {"slot": 2, "run": 40}, "critic_model": "opus", "voter_model": "haiku", "vote_k": 3,
           "timeout_s": 60, "infra_backoff_s": 5}
    cfg.update(over)
    return cfg


def make_engine(tmp_path, arm, models=None, gpu=None, clock=None, state=None, **over):
    from arms import LoopEngine
    cfg = make_config(tmp_path, arm, **over)
    svc = FakeServices(models or FakeModelService(), gpu or FakeGPUService(offsets=cfg["gpu_offsets"]),
                       clock or FakeClock())
    return LoopEngine(cfg, svc, state=state), svc
