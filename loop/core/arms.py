"""Loop engine: step logic of arms A to D (DESIGN.md sections 3, 5, 6, 8, 10; thresholds in PREREG.md).

The loop under study improves one file, train.py (a 300 second GPT training run), scored by val_bpb (validation bits
per byte, lower is better). LLM roles (planner, coder, voter, critic, analyst, repairer) propose and write the edits.
This module is the fixed code around them: it asks for an edit, applies and lints it, runs it on a GPU, decides
accept or reject by a fixed rule, and records every event.

Terms used in this file:
  arm        one loop configuration. A = base: one Sonnet call per step, keep the edit if its raw value is lower.
             B = A plus verification: linter, z-scored replicate-before-accept rule, abort action.
             C = B plus planning: idea backlog with Thompson sampling, voter, analyst memo, critic with veto.
             D = C plus routing: a routing table decides which model fills each role.
  incumbent  the best accepted train.py so far (<run_dir>/incumbent.py); every candidate is an edit of it.
  anchor     the unmodified baseline; its mean val_bpb is the first bar a candidate has to beat.
  slot       the unit of budget: one GPU run (candidate, replicate, audit or crash fix) or one charged event without
             a run (patch failure that used up the free actions, unrepaired lint failure, veto, abort over the cap).
             Every arm gets the same n_slots, so arms are compared at equal GPU time.
  free action  a retry inside a slot that is not charged, limited per slot and per run (free_action_caps).
  sealed     a validation shard the loop never trains or selects on; inside a loop only audit runs are scored on it.
  infra      a failure of the queue, the network or a model server. It is retried and never charged to the loop.

Entry points: LoopEngine(config, services).run() runs until n_slots are used. run_one_slot() does exactly one slot; a
candidate that needs a replicate, an audit or a crash fix leaves state["cand"] set and the next run_one_slot()
continues it. Everything the engine knows lives in state.json, written atomically after every event, so a killed
engine resumes from the file. Files under <run_dir>: state.json, ledger.jsonl (one record per event), incumbent.py,
memo.md and slots/<k>/ (train.py, result.json, reply.txt, lint.json, critic.json). See ENGINE.md for the schemas.
Started by run_loop.py; models, the GPU queue and the clock are reached only through services.py, so tests fake them.
"""
import difflib
import hashlib
import json
import os
import random
from datetime import datetime, timezone
from pathlib import Path
from statistics import fmean

import acceptance
import ledger
import prompts
import scheduler
from linter import lint
from patch import apply_with_retries
from sanitize_log import sanitize_tail

ARMS = ("A", "B", "C", "D")
# Worker result statuses that blame the infrastructure, not the candidate: the job is retried and no slot is charged.
INFRA_STATUSES = ("infra_fail", "cancelled", "error")
DEFAULT_MODELS = {"voter": "haiku", "critic": "opus"}   # every other role defaults to sonnet
# Config defaults; a loop's config.json overrides them (every key is described in ENGINE.md). timeout_s and
# infra_backoff_s are seconds, critic_every counts ledger records, memo_every counts decided experiments, and
# harm_rate is the probability that the harm control (see _harm) replaces a candidate edit.
DEFAULTS = {
    "free_action_caps": {"slot": 2, "run": 40}, "critic_model": "opus", "voter_model": "haiku", "vote_k": 3,
    "timeout_s": 7200, "candidate_seed": 42, "routing": {}, "backlog_size": 30, "refresh_items": 10,
    "refresh_below": 5, "critic_every": 8, "memo_every": 5, "stagnation_k": 5, "dedup_mode": "auto",
    "max_infra_retries": 20, "infra_backoff_s": 60, "voter_temperature": 0.8,
    "open_extra": {"chat_template_kwargs": {"enable_thinking": False}}, "stop_file": None, "harm_rate": 0.0,
}
HARM_LINE = "# candidate edit withheld"   # harm control (P2_PREREG): the one line a replaced edit adds to the incumbent


class EngineStop(Exception):
    """The engine cannot go on (stop file seen while waiting out an infra failure, or retries exhausted)."""


def _iso(t):
    """Epoch seconds to an ISO 8601 UTC string, the timestamp format of state.json and the ledger."""
    return datetime.fromtimestamp(t, timezone.utc).isoformat(timespec="seconds")


def _sha(text):
    """Hex SHA-256 of a source text, used to identify an incumbent or a candidate edit."""
    return hashlib.sha256(text.encode()).hexdigest()


def _write_atomic(path, text):
    """Write through a temp file and os.replace, so a kill mid-write never leaves a half-written file behind."""
    tmp = "%s.tmp" % path
    with open(tmp, "w") as f:
        f.write(text)
    os.replace(tmp, path)


class LoopEngine:
    """One loop run of one arm: holds the config, the services and the resumable state, and advances slot by slot.
    
    config: dict with the required keys checked by run_loop.load_config, plus any key of DEFAULTS.
    services: a services.Services(models, gpu, clock); tests pass fakes.
    state: optional state dict. By default the engine resumes from <run_dir>/state.json if it exists, else it starts
    fresh."""
    def __init__(self, config, services, state=None):
        self.cfg = {**DEFAULTS, **config}
        self.arm = self.cfg["arm"]
        if self.arm not in ARMS:
            raise ValueError("arm must be one of %s, got %r" % (ARMS, self.arm))
        self.svc, self.clock = services, services.clock
        self.run_dir = Path(self.cfg["run_dir"]).resolve()
        (self.run_dir / "slots").mkdir(parents=True, exist_ok=True)
        self.ledger = ledger.Ledger(self.run_dir / "ledger.jsonl")
        self.records = self.ledger.read()
        self.state_path = self.run_dir / "state.json"
        if state is None and self.state_path.exists():
            state = json.loads(self.state_path.read_text())
        self.state = state if state is not None else self._init_state()
        self.state.setdefault("harm_noops", [])   # a state.json written without the harm control lacks this key
        # The RNG state is saved with every event (see _save), so a resumed loop continues the same random sequence
        # instead of restarting it. JSON turns the internal tuple into a list, hence the tuple() below.
        self.rng = random.Random(self.cfg["rng_seed"])
        if self.state.get("rng_state"):
            v, internal, gauss = self.state["rng_state"]
            self.rng.setstate((v, tuple(internal), gauss))
        self.system = prompts.program_card(self.state["n_slots"])   # the system prompt shared by every role call
        # override: model name -> fallback for the current slot (arm D health check). cur_kind: kind of the slot in
        # progress, used to label infra events. stop_reason: why run() stopped early, read by run_loop.py.
        self.override, self.cur_kind, self._dedup, self.stop_reason = {}, "candidate", None, None
        self._save()

    # ------------------------------------------------------------------ state

    def _init_state(self):
        """Fresh state: the incumbent is the unmodified base train.py and its bar is the anchor mean over anchor n runs.
        The keys below are the state.json schema (ENGINE.md)."""
        cfg, now = self.cfg, _iso(self.clock.now())
        base = Path(cfg["base_train_py"]).read_text()
        inc_path = self.run_dir / "incumbent.py"
        _write_atomic(inc_path, base)
        return {
            "loop_id": cfg["loop_id"], "arm": self.arm, "slot_index": 0, "n_slots": cfg["n_slots"],
            "incumbent": {"path": str(inc_path), "sha256": _sha(base), "mean_adjusted": cfg["anchor"]["mean"],
                          "n": cfg["anchor"]["n"], "raw_values": [], "adjusted_values": [], "gpus": [],
                          "description": "baseline"},
            "free_actions_run": 0, "free_actions_slot": 0, "attempt": 0, "in_flight_job": None, "cand": None,
            "backlog": [], "backlog_created": False, "family_stats": {}, "consecutive_rejections": 0,
            "experiments": 0, "memo": "", "last_tail": "", "pending_calls": [],
            "rng_state": None, "claimed_best_official": None, "status": "running", "started": now, "updated": now,
            "served_counts": {}, "designated_open_calls": 0, "open_served_calls": 0, "fallbacks": [],
            "dedup_mode": None, "dedup_dropped": 0, "infra_events": 0, "harm_noops": []}

    def _save(self):
        """Stamp the state, capture the RNG state and write state.json atomically. Called after every event, which is
        what lets a killed engine resume at any point."""
        s = self.state
        s["updated"] = _iso(self.clock.now())
        v, internal, gauss = self.rng.getstate()
        s["rng_state"] = [v, list(internal), gauss]
        _write_atomic(self.state_path, json.dumps(s, indent=1, sort_keys=True))

    def _snap(self):
        """The incumbent's (mean, n), the form stored in ledger records as incumbent_before and incumbent_after."""
        return {"mean": self.state["incumbent"]["mean_adjusted"], "n": self.state["incumbent"]["n"]}

    def _stop_requested(self):
        """The stop file stops the loop now; if it holds an integer N the loop stops once slot_index >= N, which is
        how the controller makes the symmetric stop (same slot index for every loop in flight)."""
        sf = self.cfg["stop_file"]
        if not sf or not os.path.exists(sf):
            return False
        try:
            return self.state["slot_index"] >= int(Path(sf).read_text().strip())
        except ValueError:
            return True

    def _slot_dir(self, k=None):
        """Directory holding the files of slot k (default: the current slot), created on demand."""
        d = self.run_dir / "slots" / str(self.state["slot_index"] if k is None else k)
        d.mkdir(parents=True, exist_ok=True)
        return d

    def _inc_src(self):
        """Source text of the current incumbent train.py."""
        return Path(self.state["incumbent"]["path"]).read_text()

    def _table(self):
        """The results table shown to the models. It is generated by code from the ledger, never written by a model."""
        return ledger.results_table(self.records, self.state["incumbent"])

    # ------------------------------------------------------------------ ledger, charging, free actions

    def _log(self, kind, decision, ctx, *, before=None, **kw):
        """Append one ledger record for the current slot and attempt, and return it. The model calls made since the
        previous record (pending_calls) are attached to it, so the cost of every call lands on exactly one record.
        ctx: the idea or candidate dict (idea_id, description, edit_class). before: incumbent snapshot taken before a
        decision that may replace the incumbent."""
        s = self.state
        rec = ledger.new_record(
            slot=s["slot_index"], attempt=s["attempt"], kind=kind, decision=decision, idea_id=ctx.get("idea_id"),
            description=ctx.get("description"), edit_class=ctx.get("edit_class"), model_calls=s["pending_calls"],
            incumbent_before=before or self._snap(), incumbent_after=self._snap(), ts=_iso(self.clock.now()), **kw)
        s["pending_calls"] = []
        self.records.append(rec)
        self.ledger.append(rec)
        s["claimed_best_official"] = ledger.claimed_best(self.records)
        self._save()
        return rec

    def _charge(self):
        """Spend one slot: advance slot_index and reset the per-slot counters (free actions, attempt, in-flight job)."""
        s = self.state
        s["slot_index"] += 1
        s["free_actions_slot"] = s["attempt"] = 0
        s["in_flight_job"] = None
        self._save()

    def _take_free(self):
        """Use one free action if both caps allow it (per slot and per run). Returns False when none is left; the
        caller then charges the slot."""
        s, caps = self.state, self.cfg["free_action_caps"]
        if s["free_actions_slot"] >= caps["slot"] or s["free_actions_run"] >= caps["run"]:
            return False
        s["free_actions_slot"] += 1
        s["free_actions_run"] += 1
        self._save()
        return True

    def _infra(self, what):
        """Log an infra event (never charged, never touches posteriors), wait, and give up on a stop request."""
        s = self.state
        s["infra_events"] += 1
        self._log(self.cur_kind, "infra", {}, status="engine:infra:%s" % what, charged=False)
        if self._stop_requested():
            raise EngineStop("stop_file")
        self.clock.sleep(self.cfg["infra_backoff_s"])

    def _abandon(self, decision, reason, ctx, *, free, lint=None, critic=None):
        """End an attempt without a GPU run. With free=True the abandonment is a free action while the caps allow it,
        and the slot goes on with a new attempt; otherwise the slot is charged."""
        skipped = free and self._take_free()
        self._log("candidate", decision, ctx, status="engine:" + reason, lint=lint, critic=critic,
                  charged=not skipped)
        if ctx.get("item_id"):
            scheduler.note_untested(self._item(ctx["item_id"]))
        if skipped:
            self.state["attempt"] += 1
            self._save()
            return "skipped"
        self._charge()
        return "charged"

    # ------------------------------------------------------------------ model calls

    def _route(self, role, edit_class=None):
        """Model name for a role. Arms A to C: Sonnet for every role except voter and critic. Arm D reads the routing
        table: the coder is chosen by edit class (constant, schedule, structural), any other role by its own key, and
        a role the table does not name falls back to the A to C default."""
        cfg = self.cfg
        if self.arm == "D":
            r = cfg["routing"] or {}
            if role == "coder" and edit_class in r.get("coder_by_class", {}):
                return r["coder_by_class"][edit_class]
            # the initial backlog and widening calls stay on sonnet (DESIGN section 3); refreshes use the cheap planner
            for key in ("planner_refresh", "planner_cheap", "planner") if role == "planner_refresh" else (role,):
                if isinstance(r.get(key), str):
                    return r[key]
        return {"voter": cfg["voter_model"], "critic": cfg["critic_model"]}.get(role, "sonnet")

    def _note_call(self, role, designated, r):
        """Queue one model call (tokens, cost, latency, requested and served model) for the next ledger record.
        tokens_in counts fresh input plus cache reads and writes. Also counts how many calls designated for an
        open-weight model were really served by one: a fallback to a Claude model lowers that share, which is how a
        degraded arm D run is detected."""
        s, m = self.state, self.svc.models
        u = r.usage or {}
        s["pending_calls"].append({
            "role": role, "requested_model": r.requested_model or designated, "served_model": r.served_model,
            "cost_usd_plan": r.cost_usd_plan, "cost_usd_nominal": r.cost_usd_nominal,
            "tokens_in": u.get("input", 0) + u.get("cache_read", 0) + u.get("cache_write", 0),
            "tokens_out": u.get("output", 0), "latency_s": r.latency_s})
        if r.ok:
            s["served_counts"][r.served_model] = s["served_counts"].get(r.served_model, 0) + 1
            if m.is_open(designated):
                s["designated_open_calls"] += 1
                s["open_served_calls"] += int(m.is_open(r.served_model))
        self._save()

    def _call(self, role, user, *, edit_class=None, k=None, key=None):
        """One role call (or k self-consistency samples with key). Infra failures wait and repeat the same call."""
        designated = self._route(role, edit_class)
        model = self.override.get(designated, designated)   # the health check may have swapped in a fallback
        m, slot = self.svc.models, self.state["slot_index"]
        kw = dict(system=self.system, schema=prompts.SCHEMAS[role], slot=slot,
                  extra=self.cfg["open_extra"] if m.is_open(model) else None,
                  temperature=self.cfg["voter_temperature"] if role == "voter" else None)
        for _ in range(self.cfg["max_infra_retries"] + 1):
            out = m.sample_k(role, model, user, k, key, **kw) if k else m.call(role, model, user, **kw)
            results = out["results"] if k else [out]
            for r in results:
                self._note_call(role, designated, r)
            # Repeat only when every result is an infra failure. A reply that fails its schema is the model's own
            # failure and goes back to the caller, which spends a free action or the slot on it.
            if not all(not r.ok and r.error_class == "infra" for r in results):
                return out
            self._infra("model:%s" % role)
        raise EngineStop("model infra retries exhausted")

    def _health_check(self):
        """Arm D: an open model that fails its health check is replaced by its fallback for this slot (logged)."""
        self.override = {}
        m, s = self.svc.models, self.state
        if self.arm != "D" or not hasattr(m, "healthy"):
            return
        r = self.cfg["routing"] or {}
        names = {v for v in list(r.values()) + list(r.get("coder_by_class", {}).values()) if isinstance(v, str)}
        for name in sorted(names):
            if m.is_open(name) and not m.healthy(name):
                fb = (m.fallback_of(name) if hasattr(m, "fallback_of") else None) or "haiku"
                self.override[name] = fb
                s["fallbacks"].append({"slot": s["slot_index"], "model": name, "fallback": fb})
        self._save()

    # ------------------------------------------------------------------ patch application, lint, critic

    def _apply(self, src, edits_text):
        """(new_source, "") or (None, hint). Includes the compile() check and rejects a no-op edit."""
        new, rep = apply_with_retries(src, edits_text)
        if not rep["ok"]:
            return None, rep["retry_hint"] or rep["message"]
        try:
            compile(new, "train.py", "exec")
        except (SyntaxError, ValueError) as e:
            return None, "The edit applied but the result does not compile: %s. Fix it and resend." % e
        if new == src:
            return None, "The edit leaves train.py unchanged. Propose a change."
        return new, ""

    def _get_edit(self, make, src, allow_abort):
        """Call, apply, and retry while free actions last. make(prev, hint) -> CallResult.
        Returns (new_source or None, reply dict or None, why) with why in ok | abort | patch_fail | model_fail."""
        prev = hint = ""
        while True:
            r = make(prev, hint)
            obj = r.obj if r.ok and isinstance(r.obj, dict) else None
            if obj is None:
                why, prev, hint = "model_fail", "", "Your reply did not match the required JSON schema."
            elif allow_abort and obj.get("abort"):
                return None, obj, "abort"
            else:
                prev = obj.get("edits_text") or ""
                self._note_reply(prev)
                new, hint = self._apply(src, prev)
                if new is not None:
                    return new, obj, "ok"
                why = "patch_fail"
            if not self._take_free():
                return None, obj, why

    def _harm(self, new, inc_src, ctx):
        """Harm control (P2_PREREG). With probability harm_rate, decided by a hash of (loop seed, slot, attempt) and not
        by self.rng, the coder's applied edit is replaced by the incumbent plus one trailing comment line. The run,
        decision, ledger record and posterior update then go on as usual. Crash fixes and repairs never come here."""
        s, rate = self.state, self.cfg["harm_rate"]
        if not rate > 0:
            return new
        # A hash and not self.rng: the decision is identical when a killed attempt is repeated on resume, and it
        # takes no draw from the engine's RNG stream. The hash modulo 1e6 acts as a uniform draw on [0, 1e6).
        key = "%s:%s:%s" % (self.cfg["rng_seed"], s["slot_index"], s["attempt"])
        if int(hashlib.sha256(key.encode()).hexdigest(), 16) % 1_000_000 >= rate * 1_000_000:
            return new
        d = self._slot_dir()
        (d / "harm_original_train.py").write_text(new)
        (d / "harm.json").write_text(json.dumps(
            {"slot": s["slot_index"], "attempt": s["attempt"], "idea_id": ctx["idea_id"], "item_id": ctx["item_id"],
             "coder_edit_sha256": _sha(new)}))
        # a resume after a kill between this save and the in-flight save repeats the attempt: one entry per (slot, attempt)
        if not any(n["slot"] == s["slot_index"] and n["attempt"] == s["attempt"] for n in s["harm_noops"]):
            s["harm_noops"].append({"slot": s["slot_index"], "attempt": s["attempt"], "idea_id": ctx["idea_id"]})
        self._save()
        return inc_src + ("" if inc_src.endswith("\n") else "\n") + HARM_LINE + "\n"

    def _note_reply(self, text):
        """Append the raw edit text of a model reply to slots/<k>/reply.txt, one block per attempt."""
        with open(self._slot_dir() / "reply.txt", "a") as f:
            f.write(text.rstrip("\n") + "\n-----\n")

    def _diff(self, new_src):
        """Unified diff from incumbent to candidate (2 context lines), cut at 300 lines to bound the critic prompt."""
        d = list(difflib.unified_diff(self._inc_src().splitlines(), new_src.splitlines(), "incumbent", "candidate",
                                      lineterm="", n=2))
        return "\n".join(d[:300] + (["... (%d more lines)" % (len(d) - 300)] if len(d) > 300 else []))

    def _critic(self, new_src, numbers, reason):
        """Ask the critic to review a candidate. numbers: text summary of the run values. reason: why it was escalated.
        Returns {ok, issues, veto} (also saved as critic.json), or None when the call failed. A failed critic call
        never blocks a candidate; only an explicit veto does."""
        r = self._call("critic", prompts.critic(self._diff(new_src), numbers, reason))
        if not r.ok or not isinstance(r.obj, dict):
            return None
        (self._slot_dir() / "critic.json").write_text(json.dumps(r.obj, indent=1))
        return {"ok": bool(r.obj["ok"]), "issues": [str(i) for i in r.obj["issues"]], "veto": bool(r.obj["veto"])}

    # ------------------------------------------------------------------ backlog (arms C and D)

    def _item(self, item_id):
        """Backlog item with this id."""
        return next(i for i in self.state["backlog"] if i["id"] == item_id)

    def _dedupper(self):
        """The near-duplicate detector, built on first use (arms A and B never need it). Its mode is saved in the state."""
        if self._dedup is None:
            self._dedup = scheduler.Dedup(self.cfg["dedup_mode"])
            self.state["dedup_mode"] = self._dedup.mode
        return self._dedup

    def _plan(self, role, n, widen=False):
        """Ask a planner role for n new backlog items, drop near duplicates of anything already in the backlog, and
        seed the Beta prior of each kept item from vote_k yes/no voter samples (will this idea plausibly lower val_bpb).
        A failed planner call adds nothing."""
        s = self.state
        user = prompts.planner(self._inc_src(), n, self._table(), s["memo"], s["family_stats"], s["backlog"], widen)
        r = self._call(role, user)
        if not r.ok:
            return
        kept, dropped = scheduler.dedup_new(s["backlog"], r.obj["items"], self._dedupper())
        s["dedup_dropped"] += len(dropped)
        for c in kept[:n]:
            item = scheduler.make_item("i%03d" % (len(s["backlog"]) + 1), c["title"], c["spec"], c["family"])
            out = self._call("voter", prompts.voter(item), k=self.cfg["vote_k"], key="vote")
            item["alpha"], item["beta"] = scheduler.seed_prior(out["values"])
            s["backlog"].append(item)
        self._save()

    def _pick_item(self):
        """Refill the backlog if needed, then Thompson-sample the next idea (arms C and D). Refill cases: the first call
        (full backlog); stagnation, meaning stagnation_k rejections in a row (a widening call asks for different
        directions and the pick is uniform among the top scores); or fewer than refresh_below untried items.
        Returns an item dict, or None when nothing is open."""
        s, cfg = self.state, self.cfg
        widen = scheduler.should_widen(s["consecutive_rejections"], cfg["stagnation_k"])
        if not s["backlog_created"]:
            self._plan("planner_backlog", cfg["backlog_size"])
            s["backlog_created"] = True
        elif widen:
            self._plan("planner_widen", cfg["refresh_items"], widen=True)
        elif scheduler.untried_count(s["backlog"]) < cfg["refresh_below"]:
            self._plan("planner_refresh", cfg["refresh_items"])
        if widen:
            s["consecutive_rejections"] = 0
        self._save()
        return scheduler.thompson_pick(s["backlog"], self.rng, widen=widen)

    def _memo(self):
        """Ask the analyst for an updated memo: short running notes on what has worked, which the planner and the coder
        see in their prompts. Kept in the state and in memo.md."""
        s = self.state
        r = self._call("analyst", prompts.analyst(self._table(), s["memo"], s["family_stats"]))
        if r.ok:
            s["memo"] = r.obj["memo"]
            _write_atomic(self.run_dir / "memo.md", s["memo"] + "\n")

    def _finish(self, cand, accepted):
        """The experiment is decided: stagnation counter, posteriors (C, D) and memo cadence."""
        s = self.state
        s["consecutive_rejections"] = 0 if accepted else s["consecutive_rejections"] + 1
        s["experiments"] += 1
        s["cand"] = None
        if self.arm in ("C", "D"):
            if cand.get("item_id"):
                scheduler.update(self._item(cand["item_id"]), s["family_stats"], accepted)
            if scheduler.memo_due(s["experiments"], self.cfg["memo_every"]):
                self._memo()
        self._save()

    # ------------------------------------------------------------------ GPU runs

    def _stage(self, src):
        """Write the candidate source to slots/<k>/train.py, the file the GPU worker runs. Returns its path."""
        path = self._slot_dir() / "train.py"
        path.write_text(src)
        return str(path)

    def _run_gpu(self, kind, path, seed, sealed):
        """Record the job as in flight before submitting it, then drive it to a result. Saving first means a kill at any
        point resumes this same slot instead of losing or repeating it. sealed: also score the run on the sealed
        shard."""
        s = self.state
        if s["cand"]:
            s["cand"]["next"] = None
        s["in_flight_job"] = {"job_id": None, "kind": kind, "slot": s["slot_index"], "train_py": path,
                              "seed": seed, "sealed": sealed,
                              "tag": "%s:%s:%d" % (self.cfg["loop_id"], kind, s["slot_index"])}
        self._save()
        return self._drive_job()

    def _drive_job(self):
        """Submit (unless the in-flight record already holds a job id) and wait for a real result. Infra problems
        never charge a slot: a job that ended in infra_fail is resubmitted, a transport hiccup or a wait timeout
        keeps polling the same job."""
        s, gpu = self.state, self.svc.gpu
        fl = s["in_flight_job"]
        for _ in range(self.cfg["max_infra_retries"] + 1):
            try:
                if fl["job_id"] is None:
                    fl["job_id"] = gpu.submit(fl["train_py"], seed=fl["seed"], sealed=fl["sealed"], tag=fl["tag"],
                                              kind=fl["kind"], gpu_affinity=None)
                    self._save()
                res = gpu.wait(fl["job_id"], self.cfg["timeout_s"])
            except Exception as e:
                res = {"status": "infra_fail", "error": "%s: %s" % (type(e).__name__, e)}
            if res is not None and res.get("status") not in INFRA_STATUSES:
                return res
            if res is not None and res.get("job_id") is not None:
                fl["job_id"] = None   # the worker reported the job itself as failed: submit a new one
            self._infra("gpu:%s" % (res or {}).get("status", "wait_timeout"))
        raise EngineStop("gpu infra retries exhausted")

    def _on_result(self, kind, result):
        """Everything that follows a finished GPU run: flags, decision, ledger, charge, incumbent."""
        s, cfg = self.state, self.cfg
        cand, fl, before = s["cand"], s["in_flight_job"], self._snap()
        (self._slot_dir() / "result.json").write_text(json.dumps(result, indent=1, default=str))
        s["last_tail"] = sanitize_tail("\n".join(result.get("log_tail") or []), result)
        gpu, status = result.get("gpu"), result.get("status")
        run = dict(job_id=fl["job_id"], gpu=gpu, status=status, official_val_bpb=result.get("official_val_bpb"),
                   sealed_val_bpb=result.get("sealed_val_bpb"), lint=cand.get("lint"))

        # 1. Crash or timeout: charged. A first candidate gets one crash-fix slot; anything else ends the experiment.
        if status in ("crash", "timeout"):
            fixable = kind == "candidate" and not cand.get("is_fix")   # one fix attempt per candidate, in all arms
            cand["next"] = "crash_fix" if fixable else None
            self._log(kind, "crash", cand, before=before, critic=cand.get("critic"), **run)
            self._charge()
            if not fixable:
                self._finish(cand, False)
            return

        # 2. Integrity flags (target leak, hash mismatch, over budget, wrong seed, ...) reject the run whatever its
        # number says.
        flags = acceptance.integrity_flags(result)
        if fl["sealed"] and result.get("sealed_val_bpb") is None:
            flags.append("sealed_missing")
        if flags:
            crit = self._escalate_flagged(cand)
            self._log(kind, "invalid", cand, before=before, flags=flags, critic=crit, **run)
            self._charge()
            self._finish(cand, False)
            return

        # 3. Decision. Values are GPU-offset adjusted first: each GPU's frozen offset from the grand baseline mean is
        # subtracted, so a run is not favoured by landing on a GPU that reads low. z = (incumbent mean minus candidate
        # mean) / standard error with sigma_used as the noise sd; positive means the candidate is better.
        raw = result["official_val_bpb"]
        adj = acceptance.adjust(raw, gpu, cfg["gpu_offsets"])
        cand["runs"].append({"job_id": fl["job_id"], "kind": kind, "gpu": gpu, "seed": fl["seed"], "raw": raw,
                             "adjusted": adj, "sealed": result.get("sealed_val_bpb")})
        vals, inc, sigma = [r["adjusted"] for r in cand["runs"]], s["incumbent"], cfg["sigma_used"]
        # Arm A keeps the plain rule (raw value strictly below the incumbent's); its z is computed for the record only.
        if self.arm == "A" and len(vals) == 1:
            d, z = acceptance.decide_base(self._raw_bar(), raw), acceptance.z_score(
                inc["mean_adjusted"], inc["n"], vals, sigma)
        else:
            d, z = acceptance.decide_verified(inc, vals, sigma)
        if len(vals) == 1:
            cand["z1"] = z
        run.update(adjusted_val_bpb=adj, z1=cand["z1"], z2=None if len(vals) == 1 else z)
        # Tripwire: a gain too large to be plausible (in one step, or cumulative over the anchor) must first pass an
        # audit, one more run on a fresh seed that is also scored on the sealed shard.
        if d == "accept" and kind != "audit" and acceptance.tripwire(
                inc["mean_adjusted"], fmean(vals), cfg["anchor"]["mean"], sigma):
            d, tripped = "audit", ["tripwire"]
        else:
            tripped = []
        if d in ("replicate", "audit"):   # not decided yet: the next slot reruns this same candidate
            cand["next"] = d
            self._log(kind, d, cand, before=before, flags=tripped, critic=cand.get("critic"), **run)
            self._charge()
            return

        # 4. Critic escalation (arms C and D): audits, borderline accepts (accepted on replicated runs but below the
        # single-run bar) and every critic_every-th record. A veto turns an accept into a reject.
        crit = cand.get("critic")
        borderline = d == "accept" and len(vals) >= 2 and z < acceptance.Z_ACCEPT
        every = (len(self.records) + 1) % cfg["critic_every"] == 0   # + 1: this run's record is not appended yet
        if self.arm in ("C", "D") and crit is None and (kind == "audit" or borderline or every):
            numbers = ("adjusted values of the candidate runs: %s; incumbent mean %.4f over %d run(s); z = %.3f; "
                       "provisional decision: %s; integrity flags: none"
                       % ([round(v, 4) for v in vals], inc["mean_adjusted"], inc["n"], z, d))
            reason = "tripwire audit" if kind == "audit" else "borderline accept" if borderline else "periodic review"
            crit = self._critic(Path(cand["train_py"]).read_text(), numbers, reason)
        if d == "accept" and crit and crit["veto"]:
            d = "reject"
        if d == "accept":
            self._accept(cand)
        self._log(kind, d, cand, before=before, critic=crit, **run)
        self._charge()
        self._finish(cand, d == "accept")

    def _escalate_flagged(self, cand):
        """Arms C and D send a flagged run to the critic, once per candidate. The review is for the record: the run is
        rejected either way."""
        if self.arm not in ("C", "D") or cand.get("critic") is not None:
            return cand.get("critic")
        return self._critic(Path(cand["train_py"]).read_text(), "the run was rejected for integrity flags",
                            "integrity flag")

    def _raw_bar(self):
        """Arm A's bar: mean raw value of the incumbent's own runs, or the anchor mean while the incumbent is still the
        baseline (which has no runs of its own in this loop)."""
        inc = self.state["incumbent"]
        return fmean(inc["raw_values"]) if inc["raw_values"] else inc["mean_adjusted"]

    def _accept(self, cand):
        """Promote the candidate: its source becomes incumbent.py and its runs (mean, n, values, GPUs) the new bar."""
        s, runs = self.state, self.state["cand"]["runs"]
        src = Path(cand["train_py"]).read_text()
        _write_atomic(self.run_dir / "incumbent.py", src)
        s["incumbent"] = {"path": str(self.run_dir / "incumbent.py"), "sha256": _sha(src),
                          "mean_adjusted": fmean(r["adjusted"] for r in runs), "n": len(runs),
                          "raw_values": [r["raw"] for r in runs], "adjusted_values": [r["adjusted"] for r in runs],
                          "gpus": [r["gpu"] for r in runs], "description": cand["description"]}

    # ------------------------------------------------------------------ slots

    def run_one_slot(self):
        """Do exactly one slot; returns False when the slot budget is already used. Order: a job still in flight from
        before a kill is polled first; else the pending follow-up of the current candidate (crash fix, replicate or
        audit); else a new idea."""
        s = self.state
        if s["slot_index"] >= s["n_slots"]:
            return False
        if s["in_flight_job"]:   # resume: poll the job that was already submitted
            self.cur_kind = s["in_flight_job"]["kind"]
            self._on_result(self.cur_kind, self._drive_job())
            return True
        self._health_check()
        nxt = s["cand"]["next"] if s["cand"] else None
        if nxt == "crash_fix":
            self._crash_fix_slot()
        elif nxt in ("replicate", "audit"):
            self._followup_slot(nxt)
        else:
            self._new_idea_slot()
        return True

    def _followup_slot(self, kind):
        """Replicate or audit slot: rerun the same candidate file on a fresh seed drawn from the engine RNG. An audit is
        also scored on the sealed shard."""
        self.cur_kind = kind
        cand = self.state["cand"]
        path = self._stage(Path(cand["train_py"]).read_text())
        res = self._run_gpu(kind, path, self.rng.randrange(10_000, 2 ** 31 - 1), kind == "audit")
        self._on_result(kind, res)

    def _new_idea_slot(self):
        """Candidate slot. Attempts repeat while each one ends in a free action ("skipped"); the slot ends when an
        attempt reaches the GPU or is charged."""
        self.cur_kind = "candidate"
        while self._attempt() == "skipped":
            pass

    def _attempt(self):
        """One attempt at a new candidate. Steps: pick a backlog idea (arms C, D); get an edit from the coder, or from
        the single base-step caller in arms A and B; apply it; lint it (arms B to D, with one repair call); have the
        critic review a repaired edit (arms C, D); run it on a GPU. Returns "ran", "skipped" (a free action was used,
        try again inside the same slot) or "charged" (the slot was spent without a run)."""
        s, arm = self.state, self.arm
        inc_src = self._inc_src()
        allow_abort = arm != "A"   # declining to spend a slot is part of the verification stack, which arm A lacks
        ctx = {"idea_id": "c%d.%d" % (s["slot_index"], s["attempt"]), "description": "", "edit_class": None,
               "item_id": None}
        item = None
        if arm in ("C", "D"):
            item = self._pick_item()
            if item is None:
                return self._abandon("abort", "no_idea", ctx, free=False)
            ctx.update(idea_id=item["id"], description=item["title"], item_id=item["id"])
            fam_class = scheduler.FAMILY_CLASS[item["family"]]

        def make(prev, hint):
            if item:
                user = prompts.coder(inc_src, item, self._table(), s["memo"], s["family_stats"], allow_abort)
                role, cls = "coder", fam_class
            else:
                user, role, cls = prompts.base_step(inc_src, self._table(), s["last_tail"], allow_abort), "base_step", None
            return self._call(role, user + prompts.retry_suffix(prev, hint), edit_class=cls)

        new, reply, why = self._get_edit(make, inc_src, allow_abort)
        if reply and reply.get("description"):
            ctx["description"] = reply["description"]
        if why == "abort":
            return self._abandon("abort", "abort", ctx, free=True)
        if why != "ok":
            return self._abandon("invalid", why, ctx, free=False)
        new = self._harm(new, inc_src, ctx)

        lint_rec, repaired = None, False
        if arm == "A":
            try:
                ctx["edit_class"] = lint(new, inc_src).edit_class   # classification only, never gates arm A
            except Exception:
                pass
        else:
            lr = lint(new, inc_src)
            if not lr.ok:
                lint_rec = {"ok": False, "hard": lr.hard, "warnings": lr.warnings}
                if not self._take_free():
                    return self._abandon("lint_fail", "lint", ctx, free=False, lint=lint_rec)
                new2 = self._repair(new, lr.hard)
                repaired = True
                lr2 = lint(new2, inc_src) if new2 else None
                if lr2 is None or not lr2.ok:
                    if lr2 is not None:
                        lint_rec = {"ok": False, "hard": lr2.hard, "warnings": lr2.warnings}
                    return self._abandon("lint_fail", "lint_repair_failed", ctx, free=True, lint=lint_rec)
                new, lr = new2, lr2
            lint_rec = {"ok": True, "hard": [], "warnings": lr.warnings}
            ctx["edit_class"] = lr.edit_class
        crit = None
        if repaired and arm in ("C", "D"):
            crit = self._critic(new, "no GPU run yet; the candidate was repaired after a lint failure", "lint repair")
            if crit and crit["veto"]:
                return self._abandon("reject", "critic_veto", ctx, free=True, lint=lint_rec, critic=crit)

        path = self._stage(new)
        (self._slot_dir() / "lint.json").write_text(json.dumps(lint_rec))
        s["cand"] = {"idea_id": ctx["idea_id"], "description": ctx["description"], "edit_class": ctx["edit_class"],
                     "item_id": ctx["item_id"], "train_py": path, "lint": lint_rec, "critic": crit, "runs": [],
                     "next": None, "is_fix": False, "z1": None}
        # The first run of every candidate uses the same fixed seed; replicates and audits draw fresh ones.
        res = self._run_gpu("candidate", path, self.cfg["candidate_seed"], False)
        self._on_result("candidate", res)
        return "ran"

    def _repair(self, src, violations):
        """One repairer call to fix hard lint violations. Returns the repaired source, or None if the call or the edit
        failed."""
        r = self._call("repairer", prompts.repairer_lint(src, violations))
        if not r.ok or not isinstance(r.obj, dict):
            return None
        self._note_reply(r.obj.get("edits_text") or "")
        return self._apply(src, r.obj.get("edits_text") or "")[0]

    def _crash_fix_slot(self):
        """One fix attempt for a crashed candidate; it costs a slot whatever happens (all arms)."""
        s, arm = self.state, self.arm
        self.cur_kind = "crash_fix"
        cand = s["cand"]
        src = Path(cand["train_py"]).read_text()
        tail = s["last_tail"]
        new, reply, why = self._get_edit(
            lambda prev, hint: self._call("repairer", prompts.repairer_crash(src, tail) + prompts.retry_suffix(prev, hint)),
            src, False)
        if new is None:
            self._log("crash_fix", "invalid", cand, status="engine:" + why)
            self._charge()
            return self._finish(cand, False)
        lint_rec = None
        if arm != "A":
            lr = lint(new, src)
            lint_rec = {"ok": lr.ok, "hard": lr.hard, "warnings": lr.warnings}
            if not lr.ok:
                self._log("crash_fix", "lint_fail", cand, status="engine:lint", lint=lint_rec)
                self._charge()
                return self._finish(cand, False)
            cand["edit_class"] = lr.edit_class
        if reply.get("description"):
            cand["description"] = reply["description"]
        path = self._stage(new)
        cand.update(train_py=path, lint=lint_rec, critic=None, runs=[], is_fix=True, z1=None)
        res = self._run_gpu("crash_fix", path, self.cfg["candidate_seed"], False)
        self._on_result("crash_fix", res)

    # ------------------------------------------------------------------ driver

    def run(self):
        """Run slots until n_slots are used ("done"), or until the stop file asks to stop or infra retries run out
        ("stopped", with the cause in self.stop_reason). Returns the status string, which is also saved in
        state.json."""
        s = self.state
        s["status"], self.stop_reason = "running", None
        try:
            while s["slot_index"] < s["n_slots"]:
                if self._stop_requested():
                    s["status"], self.stop_reason = "stopped", "stop_file"
                    break
                self.run_one_slot()
            else:
                s["status"] = "done"
        except EngineStop as e:
            s["status"], self.stop_reason = "stopped", str(e)
        self._save()
        return s["status"]
