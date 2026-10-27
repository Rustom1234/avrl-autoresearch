"""Thin interfaces the engine depends on, so tests can fake them (tests/fakes.py).

The engine touches the outside world only through Services(models, gpu, clock):

  models.call(role, model_name, user, system=None, schema=None, temperature=None, extra=None, slot=0) -> CallResult
  models.sample_k(role, model_name, user, k, key, system=None, schema=None, temperature=None, extra=None, slot=0) -> dict
  models.is_open(model_name) -> bool          open-weight (served by us) versus Claude
  models.healthy(model_name) -> bool          optional; the engine skips the health check when absent
  models.fallback_of(model_name) -> str|None  optional
  gpu.submit(train_py_path, *, seed, sealed, tag, kind, gpu_affinity=None) -> job_id
  gpu.wait(job_id, timeout_s) -> result dict, or None on timeout; an infra problem is {"status": "infra_fail", ...}
  clock.now() -> epoch seconds, clock.sleep(seconds)

The real classes import loop/models and loop/substrate lazily, so importing this module does nothing heavy.
"""
import inspect
import sys
import time
from dataclasses import dataclass
from pathlib import Path

LOOP = Path(__file__).resolve().parent.parent


def _path(sub):
    p = str(LOOP / sub)
    if p not in sys.path:
        sys.path.insert(0, p)


class Clock:
    def now(self):
        return time.time()

    def sleep(self, seconds):
        time.sleep(seconds)


class ModelService:
    """Wraps loop/models call() and sample_k(). Tag is f"{loop_id}:{role}:{slot}"; every attempt is logged by
    call() to <run_dir>/calls.jsonl."""

    def __init__(self, run_dir, loop_id, timeout_s=600, max_tokens=4096):
        _path("models")
        import call as call_mod
        import consistency
        self._m, self._c = call_mod, consistency
        self.loop_id, self.timeout_s, self.max_tokens = loop_id, timeout_s, max_tokens
        self.log_path = str(Path(run_dir) / "calls.jsonl")
        self.registry = call_mod.load_registry()

    def _kw(self, role, slot, system, schema, temperature, extra):
        return dict(system=system, schema=schema, temperature=temperature, extra=extra, max_tokens=self.max_tokens,
                    tag="%s:%s:%s" % (self.loop_id, role, slot), log_path=self.log_path, timeout_s=self.timeout_s)

    def _context_failure(self, model_name, e):
        return self._m.CallResult(obj=None, text="", ok=False, error="ContextTooLong: %s" % e, model_name=model_name,
                                  model_id="", provider="", requested_model=model_name, served_model=model_name,
                                  error_class="context")

    def call(self, role, model_name, user, system=None, schema=None, temperature=None, extra=None, slot=0):
        try:
            return self._m.call(model_name, user, **self._kw(role, slot, system, schema, temperature, extra))
        except self._m.ContextTooLong as e:
            return self._context_failure(model_name, e)

    def sample_k(self, role, model_name, user, k, key, system=None, schema=None, temperature=None, extra=None,
                 slot=0):
        try:
            return self._c.sample_k(model_name, user, k, key=key,
                                    **self._kw(role, slot, system, schema, temperature, extra))
        except self._m.ContextTooLong as e:
            return {"results": [self._context_failure(model_name, e)], "values": [], "agreement": 0.0,
                    "modal_value": None, "n_ok": 0}

    def is_open(self, model_name):
        return self.registry[model_name]["provider"] != "claude"

    def healthy(self, model_name):
        return self._m.healthy(model_name)

    def fallback_of(self, model_name):
        return self.registry[model_name].get("fallback")


class GPUService:
    """Wraps QueueClient. The candidate path is the copy the engine wrote under <run_dir>/slots/<k>/train.py.
    kind and gpu_affinity go to QueueClient.submit when it accepts them, and always into the job meta."""

    def __init__(self, queue_client=None, job_timeout_s=600, poll_interval=20, queue_kwargs=None):
        if queue_client is None:
            _path("substrate")
            from queue_client import QueueClient
            queue_client = QueueClient(**(queue_kwargs or {}))
        self.qc, self.job_timeout_s, self.poll_interval = queue_client, job_timeout_s, poll_interval
        params = inspect.signature(queue_client.submit).parameters
        self._optional = {k for k in ("kind", "gpu_affinity") if k in params}

    def submit(self, train_py_path, *, seed, sealed, tag, kind, gpu_affinity=None):
        extra = {k: v for k, v in (("kind", kind), ("gpu_affinity", gpu_affinity)) if k in self._optional}
        return self.qc.submit(str(train_py_path), seed=seed, sealed=sealed, timeout_s=self.job_timeout_s, tag=tag,
                              meta={"kind": kind, "gpu_affinity": gpu_affinity}, **extra)

    def wait(self, job_id, timeout_s):
        try:
            return self.qc.wait([job_id], timeout_s, poll_interval=self.poll_interval).get(job_id)
        except Exception as e:  # QueueError after repeated ssh failures: the job may still be fine, the engine retries
            return {"status": "infra_fail", "error": "%s: %s" % (type(e).__name__, e)}


@dataclass
class Services:
    models: object
    gpu: object
    clock: object
