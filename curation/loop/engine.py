#!/usr/bin/env python3
"""One data-curation research loop: models propose and write changes to curate.py, a fixed trainer scores them.

    python curation/loop/engine.py --run-dir curation/runs/B1/SS_01 --planner fable --coder fable \
        --pool v2 --slots 12 --seed 1 --sigma 0.0169 [--gpu-host willow] [--resume]

Same shape as the thesis loop (loop/core/arms.py), cut down to what this task needs:
- two routed roles: the planner proposes one change in words, the coder writes the complete new curate.py (and gets
  one repair call if the file fails);
- each slot is one training run of the frozen trainer (curation/task/run.py) and a decision on dev bpb with the same
  z rule: z = (incumbent mean - candidate mean) / sigma; below 0.75 reject, at 2.0 or above accept, in between
  run both on a second seed and accept at 1.5 or above on the means;
- the models see dev bpb only. The sealed set is read once, after the last slot (two confirmation runs).
Model calls go through loop/models/call.py (imported, not edited) with this directory's models.json, which has no
fallbacks, and with separate lock and breaker files so that this loop never pauses a running thesis study.
Files in the run directory: config.json, state.json, ledger.jsonl (one line per slot), calls.jsonl (one line per
model call attempt), cand/ (every candidate file), runs/ (every trainer result), final.json.
"""
import argparse, fcntl, json, os, random, re, shlex, subprocess, sys, time
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]                       # active-audit/
TASK = HERE.parent / "task"
PY = str(HERE.parent / ".venv/bin/python")
SCRATCH = "/tmp/arloop_curation_%s" % os.environ.get("USER", "u")
os.environ.setdefault("AR_LOCK_DIR", SCRATCH)
os.environ.setdefault("AR_BREAKER_FILE", SCRATCH + "/claude_breaker.json")
os.environ.setdefault("AR_CLAUDE_MAX_CONCURRENT", "3")
sys.path[:0] = [str(ROOT / "loop" / "core"), str(ROOT / "loop" / "models")]
from services import ModelService            # noqa: E402  (adds loop/models to the path itself)
import call as call_mod                      # noqa: E402
call_mod.REGISTRY_PATH = HERE / "models.json"

SEED_A, SEED_B, CONFIRM_SEEDS = 11, 12, (1001, 1002)
Z_REJECT, Z_ACCEPT, Z_MEAN = 0.75, 2.0, 1.5
T = 9_830_400
N_SHOWN, SHOW_CHARS = 10, 300
INFRA_RETRIES, INFRA_WAIT_S = 30, 120
OPEN_EXTRA = {"chat_template_kwargs": {"enable_thinking": False}}
FORBIDDEN = re.compile(r"\bopen\s*\(|sys\.argv|listdir|scandir|\bglob\b|subprocess|socket|urllib|requests|pathlib|"
                       r"os\.walk|os\.system|os\.path|os\.getcwd|__file__|__import__|\beval\s*\(|\bexec\s*\(|np\.load|"
                       r"numpy\.load|loadtxt|fromfile|read_text|read_bytes|HIDDEN|ORACLE")

SYSTEM = """You are improving the training data of a small language model.

Setup. A fixed trainer trains a small GPT (6 layers, width 384) from scratch for exactly 300 steps on exactly \
T = 9,830,400 tokens, then measures bits per byte (bpb, lower is better) on held-out articles of the target domain. \
The trainer cannot be changed.

What can change: one Python file, curate.py, with a function select(docs, target_sample) that returns an ordered \
list of indices into docs (repeats allowed). Training reads the documents in that order and stops after T tokens; \
a list with fewer than T tokens is cycled.
- docs: a list of strings, a noisy pool of about 2T tokens. It mixes useful text with text from other domains, \
duplicates and damaged or junk documents. There are no labels.
- target_sample: a list of 200 strings from the target domain (they are not in the pool and not in the held-out set).

Rules for curate.py: CPU only, at most 180 seconds, no network and no file access. It may use the standard library, \
numpy, scikit-learn and tiktoken (encoding "gpt2"), and `from common import T`. Randomness must be seeded from the \
environment variable CURATE_SEED."""

PLAN_SCHEMA = {"type": "object", "properties": {"idea": {"type": "string"}, "why": {"type": "string"}},
               "required": ["idea", "why"]}


def code_of(text):
    """The Python file in a model reply: the last fenced block, or the whole reply if it has no fence."""
    blocks = re.findall(r"```(?:python|py)?[ \t]*\n(.*?)```", text or "", re.S)
    return (blocks[-1] if blocks else (text or "")).strip() + "\n"


def lint(src):
    """None if the file may run, else the reason."""
    try:
        compile(src, "curate.py", "exec")
    except SyntaxError as e:
        return "SyntaxError: %s (line %s)" % (e.msg, e.lineno)
    if not re.search(r"^def select\s*\(", src, re.M):
        return "curate.py must define select(docs, target_sample) at module level"
    m = FORBIDDEN.search(src)
    return "not allowed in curate.py (no file, process or network access): %r" % m.group(0) if m else None


class Loop:
    def __init__(self, cfg, models=None, runner=None):
        self.cfg, self.dir = cfg, Path(cfg["run_dir"])
        for d in ("cand", "runs"):
            (self.dir / d).mkdir(parents=True, exist_ok=True)
        self.models = models or ModelService(self.dir, cfg["loop_id"], timeout_s=900, max_tokens=4000)
        self.runner, self.sleep = runner or self.train, time.sleep
        pool = ROOT / "data" / "curation" / ("pool_%s" % cfg["pool"])
        self.docs = json.loads((pool / "docs.json").read_text())
        self.target = json.loads((pool / "target_sample.json").read_text())
        self.pool_tokens = json.loads((pool / "summary.json").read_text())["pool_tokens"]
        sf = self.dir / "state.json"
        self.s = json.loads(sf.read_text()) if sf.exists() else {
            "slot": 0, "incumbent": str(TASK / "curate.py"), "inc_scores": {}, "history": [], "usd": 0.0, "status": "new"}

    # ---------------------------------------------------------------- plumbing
    def save(self):
        tmp = self.dir / "state.json.tmp"
        tmp.write_text(json.dumps(self.s, indent=1))
        tmp.replace(self.dir / "state.json")

    def train(self, path, seed, tag):
        """One trainer run, here or on cfg["gpu_host"] over ssh (shared home). Returns (result dict or None, error)."""
        out = self.dir / "runs" / ("%s_s%d.json" % (tag, seed))
        if out.exists():
            return json.loads(out.read_text()), None
        cmd = [PY, str(TASK / "run.py"), "--curate", str(path), "--seed", str(seed), "--out", str(out),
               "--pool", self.cfg["pool"]]
        host = self.cfg.get("gpu_host")
        if host:
            cmd = ["ssh", "-o", "BatchMode=yes", host, " ".join(shlex.quote(c) for c in cmd)]
        os.makedirs(SCRATCH, exist_ok=True)
        with open("%s/gpu_%s.lock" % (SCRATCH, host or "local"), "w") as lock:
            fcntl.flock(lock, fcntl.LOCK_EX)       # one run at a time per GPU: two do not fit in 16 GB
            try:
                p = subprocess.run(cmd, capture_output=True, text=True, timeout=900)
            except subprocess.TimeoutExpired:
                return None, "the run did not finish in 900 seconds"
        if p.returncode != 0 or not out.exists():
            err = (p.stderr or "").strip()
            if "OutOfMemoryError" in err or "CUDA error" in err:   # the GPU's fault, never the candidate's: stop loudly
                raise RuntimeError("GPU failure on %s: %s" % (host or "local", err[-300:]))
            if "TimeoutExpired" in err:
                return None, "select() did not finish in 180 seconds"
            cut = err.rfind('Traceback (most recent call last):\n  File "%s' % (TASK / "run.py"))
            if cut > 0:                            # keep the curator's own traceback, drop the trainer wrapper's
                err = err[:cut].strip()
            return None, err[-1500:] or "run failed (exit %d)" % p.returncode
        return json.loads(out.read_text()), None

    def ask(self, role, user, schema=None):
        """One model call for a role, with the model this loop routes that role to. Returns the CallResult."""
        model = self.cfg["routing"][role]
        extra = OPEN_EXTRA if self.models.is_open(model) else None
        for attempt in range(INFRA_RETRIES + 1):   # a provider or server failure is waited out, not charged to the slot
            r = self.models.call(role, model, user, system=SYSTEM, schema=schema, extra=extra, slot=self.s["slot"])
            self.s["usd"] += r.cost_usd_nominal
            if r.ok or r.error_class != "infra" or attempt == INFRA_RETRIES:
                break
            self.s["infra_failures"] = self.s.get("infra_failures", 0) + 1
            self.save()
            self.sleep(INFRA_WAIT_S)
        self.slot_calls.append({"role": role, "model": r.served_model or model, "ok": r.ok, "usd": r.cost_usd_nominal,
                                "error_class": r.error_class})
        return r

    def inc_src(self):
        return Path(self.s["incumbent"]).read_text()

    def inc_score(self, seed):
        """Dev bpb of the incumbent on a seed, running it if that seed has not been run yet."""
        k = str(seed)
        if k not in self.s["inc_scores"]:
            res, err = self.runner(self.s["incumbent"], seed, "inc%02d" % self.s.get("inc_slot", 0))
            if res is None:
                raise RuntimeError("incumbent run failed: %s" % err)
            self.s["inc_scores"][k] = res["dev_bpb"]
            self.save()
        return self.s["inc_scores"][k]

    # ---------------------------------------------------------------- prompts
    def plan_prompt(self):
        rng = random.Random(self.cfg["seed"] * 1000 + self.s["slot"])
        shown = rng.sample(range(len(self.docs)), N_SHOWN)
        clip = lambda t: json.dumps(t[:SHOW_CHARS])
        lines = ["Pool: %d documents, %.2f T tokens in total." % (len(self.docs), self.pool_tokens / T), "",
                 "%d random documents of the pool (first %d characters of each):" % (N_SHOWN, SHOW_CHARS)]
        lines += ["- " + clip(self.docs[i]) for i in shown]
        lines += ["", "Two documents of target_sample (first %d characters of each):" % SHOW_CHARS]
        lines += ["- " + clip(t) for t in rng.sample(self.target, 2)]
        lines += ["", "Current best curate.py (dev bpb %.4f):" % self.inc_score(SEED_A), "```python", self.inc_src().rstrip(), "```", ""]
        if self.s["history"]:
            lines.append("Experiments so far in this run (dev bpb of one training run, lower is better):")
            lines += ["%d. %s -> %s" % (h["slot"], h["idea"], h["outcome"]) for h in self.s["history"]]
        else:
            lines.append("No experiments yet in this run.")
        lines += ["", "You have %d experiments left. Propose ONE next experiment: a specific change to curate.py that you "
                  "expect to lower dev bpb. Do not repeat an experiment that was rejected. Reply with JSON: "
                  '{"idea": "<the change, specific enough to implement>", "why": "<one sentence>"}'
                  % (self.cfg["slots"] - self.s["slot"])]
        return "\n".join(lines)

    def code_prompt(self, idea):
        return ("Current curate.py:\n```python\n%s```\n\nChange to implement:\n%s\n\nWrite the complete new curate.py. "
                "Reply with one Python code block and nothing else." % (self.inc_src(), idea))

    def fix_prompt(self, src, err):
        return ("This curate.py failed.\n```python\n%s```\n\nError:\n%s\n\nWrite the complete corrected curate.py. "
                "Reply with one Python code block and nothing else." % (src, err))

    # ---------------------------------------------------------------- one slot
    def candidate(self, idea):
        """Coder writes the file; one repair call if it does not pass the lint or the run fails.
        Returns (path, result, error)."""
        r = self.ask("coder", self.code_prompt(idea))
        if not r.ok:
            return None, None, "coder call failed (%s)" % r.error_class
        src, err, res, path = code_of(r.text), None, None, None
        for attempt in (0, 1):
            path = self.dir / "cand" / ("s%02d_%d.py" % (self.s["slot"], attempt))
            path.write_text(src)
            err = lint(src)
            if err is None:
                res, err = self.runner(path, SEED_A, "s%02d_%d" % (self.s["slot"], attempt))
            if err is None or attempt == 1:
                break
            r = self.ask("coder", self.fix_prompt(src, err))
            if not r.ok:
                return path, None, "repair call failed (%s); first error: %s" % (r.error_class, err[-300:])
            src = code_of(r.text)
        return path, res, err

    def run_slot(self):
        s = self.s
        self.slot_calls = []
        rec = {"slot": s["slot"] + 1, "t0": time.time(), "decision": None}
        r = self.ask("planner", self.plan_prompt(), schema=PLAN_SCHEMA)
        s["slot"] += 1
        idea = (r.obj or {}).get("idea") if r.ok else None
        if not idea:
            rec.update(idea=None, decision="no_idea", outcome="the planner call failed")
        else:
            rec["idea"], rec["why"] = idea.strip(), (r.obj.get("why") or "").strip()
            path, res, err = self.candidate(rec["idea"])
            rec["cand"] = str(path) if path else None
            if res is None:
                rec.update(decision="fail", error=err, outcome="FAILED to run (%s)" % " ".join((err or "").split())[-200:])
            else:
                inc_a = self.inc_score(SEED_A)
                z = (inc_a - res["dev_bpb"]) / self.cfg["sigma"]
                rec.update(dev=res["dev_bpb"], inc=inc_a, z1=z, docs=res["n_selected"],
                           tokens_over_T=res["curator_tokens_before_cycle"] / T)
                accept = z >= Z_ACCEPT
                if Z_REJECT <= z < Z_ACCEPT:           # grey zone: both on a second seed, decide on the means
                    res_b, err_b = self.runner(path, SEED_B, "s%02d_rep" % s["slot"])
                    if res_b is not None:
                        inc_m = (inc_a + self.inc_score(SEED_B)) / 2
                        rec["dev_rep"], rec["z_mean"] = res_b["dev_bpb"], (inc_m - (res["dev_bpb"] + res_b["dev_bpb"]) / 2) / self.cfg["sigma"]
                        accept = rec["z_mean"] >= Z_MEAN
                rec["decision"] = "accept" if accept else "reject"
                rec["outcome"] = "dev %.4f against %.4f for the best at the time, %s" % (
                    res["dev_bpb"], inc_a, "KEPT as the new best" if accept else "rejected")
                if accept:
                    s["incumbent"], s["inc_slot"] = str(path), s["slot"]
                    s["inc_scores"] = {str(SEED_A): res["dev_bpb"]}
                    if "dev_rep" in rec:
                        s["inc_scores"][str(SEED_B)] = rec["dev_rep"]
        rec.update(calls=self.slot_calls, usd_total=s["usd"], seconds=round(time.time() - rec.pop("t0"), 1))
        s["history"].append({"slot": rec["slot"], "idea": rec.get("idea") or "(no idea)", "outcome": rec["outcome"]})
        with open(self.dir / "ledger.jsonl", "a") as f:
            f.write(json.dumps(rec) + "\n")
        self.save()
        print("%s slot %d: %s  %s" % (self.cfg["loop_id"], rec["slot"], rec["decision"], rec["outcome"][:110]), flush=True)

    def run(self):
        s = self.s
        s["status"] = "running"
        self.inc_score(SEED_A)                       # the starting file on the candidate seed
        while s["slot"] < self.cfg["slots"]:
            self.run_slot()
        runs = [self.runner(s["incumbent"], seed, "confirm")[0] for seed in CONFIRM_SEEDS]
        ok = [r for r in runs if r]
        final = {"loop_id": self.cfg["loop_id"], "cell": self.cfg.get("cell"), "incumbent": s["incumbent"],
                 "n_confirm": len(ok), "model_usd": s["usd"], "infra_failures": s.get("infra_failures", 0),
                 "sealed_bpb": sum(r["sealed_bpb"] for r in ok) / len(ok) if ok else None,
                 "dev_bpb": sum(r["dev_bpb"] for r in ok) / len(ok) if ok else None,
                 "n_accept": sum(1 for l in open(self.dir / "ledger.jsonl") if json.loads(l)["decision"] == "accept")}
        (self.dir / "final.json").write_text(json.dumps(final, indent=1))
        s["status"] = "done"
        self.save()
        print("%s done: sealed %s, model dollars %.3f" % (self.cfg["loop_id"], final["sealed_bpb"], s["usd"]), flush=True)
        return final


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--run-dir", required=True)
    ap.add_argument("--planner", required=True)
    ap.add_argument("--coder", required=True)
    ap.add_argument("--pool", default="v2")
    ap.add_argument("--slots", type=int, default=12)
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--sigma", type=float, required=True, help="noise sd of dev bpb used by the z rule")
    ap.add_argument("--gpu-host", default=None, help="run the trainer on this host over ssh (default: here)")
    ap.add_argument("--cell", default=None)
    ap.add_argument("--resume", action="store_true")
    a = ap.parse_args(argv)
    run_dir = Path(a.run_dir).resolve()
    cf = run_dir / "config.json"
    if cf.exists():
        if not a.resume:
            raise SystemExit("%s exists; pass --resume to continue that loop" % cf)
        cfg = json.loads(cf.read_text())
    else:
        run_dir.mkdir(parents=True, exist_ok=True)
        cfg = {"loop_id": run_dir.name, "run_dir": str(run_dir), "cell": a.cell, "pool": a.pool, "slots": a.slots,
               "seed": a.seed, "sigma": a.sigma, "gpu_host": a.gpu_host,
               "routing": {"planner": a.planner, "coder": a.coder}}
        cf.write_text(json.dumps(cfg, indent=1))
    Loop(cfg).run()
    return 0


if __name__ == "__main__":
    sys.exit(main())
