#!/usr/bin/env python3
"""The CIFAR speedrun task for the loop engine (loop/core/arms.py), which is used unchanged.

The engine was written for the GPT task of P2 and P2B. It reaches the task through four things, and this file
replaces exactly those, in this process only (nothing under loop/core is edited):
  prompts   the program card and the role prompts            -> make_prompts()
  lint      the static check of a candidate train.py         -> lint()
  gpu       the service that runs a candidate and scores it  -> LabGPU (one lab machine per loop, harness.py)
  tripwire  the cumulative-gain audit threshold, an absolute number tuned to the GPT task; here it is set to
            5.5 noise sds, the same ratio as in P2B (0.03 against sigma 0.0055)
The score is a cost, lower is better (see harness.py); the engine keeps it in its fields named *_val_bpb.

Commands (from the repository root, with .venv/bin/python):
  python loop/cifar/task.py loop  --config CFG [--resume]     a loop driven by language models (engine arm D)
  python loop/cifar/task.py nloop --config CFG [--resume]     random search over the settings block, no model

Config: the engine's own fields (ENGINE.md) plus "host" (lab machine that runs this loop's jobs), "consts"
(calibration file written by calibrate.py) and optionally "runs" (timed runs per job, default 24).
"""
import argparse, ast, difflib, json, os, re, shlex, socket, subprocess, sys, time, types
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "core"))

import acceptance, arms, random_search, run_loop   # noqa: E402
import prompts as P0                               # noqa: E402
from linter import LintResult                      # noqa: E402
from services import Clock, ModelService, Services  # noqa: E402

PY = "/homes/rdubash/speedrun/.venv/bin/python"
BASE = HERE / "base" / "train.py"

# ------------------------------------------------------------------ prompts

EXAMPLE = """<<<<<<< SEARCH
LABEL_SMOOTHING = 0.09
=======
LABEL_SMOOTHING = 0.1
>>>>>>> REPLACE"""

CARD = """# autoresearch program card

Goal: lower the cost of train.py, a script that trains a small convolutional network on CIFAR-10 in a few seconds.
The cost combines training time and accuracy, lower is better:

    cost = ln(run time / %(t_ref).2f s) + (%(a_ref).2f - accuracy in percent) / %(k).2f

The starting script takes about %(t_ref).2f s per run and reaches about %(a_ref).2f percent, so its cost is 0. A
version that is 5 percent faster at the same accuracy costs about -0.05. A version that gains 0.1 point of accuracy
at the same time costs about %(tenth).3f. Training for more or fewer epochs moves along the script's own
time-accuracy curve and leaves the cost about unchanged, by construction. In results tables the cost is the
"official" column.

The starting script is the current record script of the CIFAR-10 94 percent speedrun. Its numeric settings are
already tuned, so retuning numbers alone is unlikely to help much.

What the script does: a fixed 2x2 whitening convolution initialised from the statistics of training patches, then
three blocks of two 3x3 convolutions each (widths 64, 256, 256) with max pooling, batch norm without learned scale
and SiLU, then a scaled linear head. Convolution filters are trained by Muon (orthogonalised momentum updates), the
biases and the head by Nesterov SGD, both with a linear decay to zero. Label smoothing. Augmentation: left-right flip
alternating by epoch, random 2 pixel translation, brightness and contrast jitter. Half precision, channels_last,
torch.compile. Prediction averages flipped and shifted views for the 25 percent least confident test images.

Setting. One NVIDIA RTX A4000 16 GB per job. A job compiles the script (not timed), then runs it %(runs)d times. Each
run is timed from the call of train() to the return of predict() and scored on held-out images whose labels the
script never sees. The cost uses the median time and the mean accuracy. The cost of one job has a noise sd of about
%(sigma).3f.

What you may change: train.py only, and anything in it: architecture, optimizer, schedules, data order, augmentation,
resolution, precision, the prediction function, the training loop. Install nothing.

Harness facts (enforced, stated so that no slot is wasted):
- train.py must keep defining make_model(), train(model, images, labels) and predict(model, images) with these
  arguments. images are uint8 tensors of shape (N, 32, 32, 3) on the GPU. predict returns logits of shape (N, 10).
- Every run must start from scratch. train() must reinitialise the model and may not reuse anything computed from
  data in an earlier run or in the untimed first call (which gets random data). Only compiled code carries over.
- predict() must not change the model.
- The harness owns the random seeds: do not seed any generator.
- No file or network access, and no imports beyond math, torch, torchvision.transforms and plain standard-library
  helpers (functools, itertools, collections, random, copy).
- A run longer than 2.5 times the starting script is rejected. A job is stopped after 15 minutes including
  compilation, or if it uses more than 9 GB of host memory.
- Simpler is better: a tiny gain that adds ugly code is not worth it.

Budget. You have {n_slots} slots in total. One slot is one job of about 5 minutes. A crash, a replicate run and an
audit run each use a slot. Rewriting an edit that failed to apply is free (at most 2 per slot).

Output format. Give edits to train.py as SEARCH/REPLACE blocks, aider style. Each SEARCH must match the current
file exactly once, line for line, including indentation. Blocks are applied in order. Example:

{example}

Reply as JSON with the fields requested in the task. Put the blocks, as text, in the field `edits_text`."""


def make_prompts(consts, runs):
    """A stand-in for loop/core/prompts.py: same names, the card and the task wording rewritten for this task."""
    m = types.ModuleType("cifar_prompts")
    m.__dict__.update({k: v for k, v in vars(P0).items() if not k.startswith("__")})
    card = CARD % dict(t_ref=consts["t_ref"], a_ref=consts["a_ref_val"], k=consts["k"], tenth=-0.1 / consts["k"],
                       runs=runs, sigma=consts["sigma_used"])

    def program_card(n_slots):
        return card.replace("{n_slots}", str(n_slots)).replace("{example}", EXAMPLE)

    def base_step(train_py, table, last_tail, allow_abort):
        return P0.base_step(train_py, table, last_tail, allow_abort).replace("lower val_bpb", "lower the cost")

    def planner(train_py, n_items, table, memo, family_stats, existing, widen=False):
        """The planner sees the whole file here (in P2 it saw only the settings block), because the ideas this task
        asks for are changes to the code, not to numbers."""
        kind = ("The search has stalled. Propose %d ideas from families that have not been tried recently, including "
                "at least a few bolder ones." if widen else "Propose %d distinct ideas.") % n_items
        return "\n\n".join([
            P0._file(train_py), "## Results so far\n%s" % table,
            "## Notes from earlier experiments\n%s\nFamily outcomes: %s" % (memo or "none yet", P0._fam(family_stats)),
            "## Ideas already in the backlog (do not repeat)\n%s" % P0._backlog_titles(existing),
            "## Task\n%s Each idea is one self-contained change to train.py that a coder can implement from one "
            "paragraph, without seeing the other ideas. Give each a short title, a spec paragraph, and a family from: "
            '%s. Reply as JSON: {"items": [{"title", "spec", "family"}]}.' % (kind, ", ".join(P0.FAMILIES))])

    def voter(item):
        return ("An idea for lowering the cost (training time and accuracy combined) of a tuned CIFAR-10 speedrun "
                "script that trains a small convolutional network to 94 percent in a few seconds on one GPU:\n"
                "Title: %s\nSpec: %s\nWill it plausibly lower the cost? Reply as JSON: "
                '{"vote": "yes" or "no", "reason": one sentence}.' % (item["title"], item["spec"]))

    def critic(diff, numbers, reason):
        return P0.critic(diff, numbers, reason).replace(
            "any attempt to change how val_bpb is measured, the data, the seed lines or the evaluation call",
            "any attempt to get around the measurement: work carried from one run to the next or done in the untimed "
            "first call, seeding, use of the test images for training, file access")

    m.program_card, m.base_step, m.planner, m.voter, m.critic = program_card, base_step, planner, voter, critic
    return m


# ------------------------------------------------------------------ static check

OK_IMPORTS = {"math", "torch", "torchvision", "functools", "itertools", "collections", "typing", "random",
              "dataclasses", "contextlib", "operator", "warnings", "copy"}
BAD_CALLS = {"open", "eval", "exec", "__import__", "globals", "input", "breakpoint", "compile"}
BAD_CHAINS = ("torch.load", "torch.save", "torch.hub", "torchvision.datasets", "torchvision.io")
SEED_ATTRS = {"manual_seed", "manual_seed_all", "seed"}
SIGS = {"make_model": 0, "train": 3, "predict": 2}
CONST_LINE = re.compile(r"^[A-Z][A-Z0-9_]* = ")


def _chain(node):
    parts = []
    while isinstance(node, ast.Attribute):
        parts.append(node.attr)
        node = node.value
    if isinstance(node, ast.Name):
        parts.append(node.id)
    return ".".join(reversed(parts))


def lint(source, base_source):
    """Static check of a candidate. Hard rules are the harness facts of the card that can be seen in the code."""
    changed = [l[2:] for l in difflib.ndiff(base_source.splitlines(), source.splitlines()) if l[:2] in ("+ ", "- ")]
    live = [l for l in changed if l.strip() and not l.strip().startswith("#")]
    cls = ("constant" if live and all(CONST_LINE.match(l) for l in live) else
           "schedule" if live and all(re.search(r"lr|momentum|schedule|warm", l, re.I) for l in live) else "structural")
    hard = []
    add = lambda rule, node, msg: hard.append({"rule": rule, "line": getattr(node, "lineno", None), "message": msg})
    try:
        tree = ast.parse(source)
    except SyntaxError as e:
        return LintResult(False, [{"rule": "syntax", "line": e.lineno, "message": str(e)}], [], len(changed), cls)
    defs = {n.name: n for n in tree.body if isinstance(n, ast.FunctionDef)}
    for name, nargs in SIGS.items():
        if name not in defs:
            add("interface", None, "train.py must define %s at module level" % name)
        elif len(defs[name].args.args) != nargs:
            add("interface", defs[name], "%s must take exactly %d argument(s)" % (name, nargs))
    for n in ast.walk(tree):
        if isinstance(n, (ast.Import, ast.ImportFrom)):
            mods = [a.name for a in n.names] if isinstance(n, ast.Import) else [n.module or ""]
            for mod in mods:
                if mod.split(".")[0] not in OK_IMPORTS:
                    add("import", n, "import of %s is not allowed" % mod)
                if mod.startswith(BAD_CHAINS):
                    add("import", n, "%s is not allowed" % mod)
        elif isinstance(n, ast.Call) and isinstance(n.func, ast.Name) and n.func.id in BAD_CALLS:
            add("call", n, "%s() is not allowed" % n.func.id)
        elif isinstance(n, ast.Attribute):
            c = _chain(n)
            if c.startswith(BAD_CHAINS):
                add("io", n, "%s is not allowed (no file or network access)" % c)
            if n.attr in SEED_ATTRS:
                add("seed", n, "the harness owns the random seeds; remove the call to %s" % n.attr)
        elif isinstance(n, ast.Constant) and isinstance(n.value, str) and re.search(r"\.pt\b|cifar10|/homes/", n.value):
            add("io", n, "no data files may be named in train.py")
    return LintResult(not hard, hard, [], len(changed), cls)


# ------------------------------------------------------------------ GPU service

class LabGPU:
    """Runs harness.py for one candidate on one lab machine. The job is started detached and found again through
    its result file, so a loop that is killed and resumed picks up the job it had in flight."""

    def __init__(self, host, jobs_dir, consts_path, runs=24, job_timeout_s=900):
        self.host, self.jobs, self.consts = host, Path(jobs_dir), str(consts_path)
        self.runs, self.job_timeout_s = runs, job_timeout_s
        self.jobs.mkdir(parents=True, exist_ok=True)

    def submit(self, train_py_path, *, seed, sealed, tag, kind, gpu_affinity=None):
        jid = "%s_%d" % (re.sub(r"[^A-Za-z0-9_.-]", "-", tag), time.time_ns() // 1000000)
        out = self.jobs / jid
        out.mkdir()
        cmd = [PY, str(HERE / "harness.py"), str(train_py_path), "--out", str(out), "--seed", str(seed), "--runs",
               str(self.runs), "--consts", self.consts, "--timeout", str(self.job_timeout_s), "--job-id", jid]
        cmd += ["--sealed"] if sealed else []
        (out / "cmd.json").write_text(json.dumps({"cmd": cmd, "host": self.host, "kind": kind, "t": time.time()}))
        log = str(out / "harness.log")
        if self.host == socket.gethostname().split(".")[0]:
            env = {k: v for k, v in os.environ.items() if k != "CUDA_VISIBLE_DEVICES"}   # the loop process has no GPU
            subprocess.Popen(cmd, stdout=open(log, "w"), stderr=subprocess.STDOUT, stdin=subprocess.DEVNULL,
                             start_new_session=True, cwd=str(HERE), env=env)
        else:
            remote = "cd %s && setsid nohup %s > %s 2>&1 < /dev/null" % (HERE, shlex.join(cmd), shlex.quote(log))
            subprocess.run(["ssh", "-f", "-o", "BatchMode=yes", "-o", "ConnectTimeout=10", self.host, remote],
                           check=True, timeout=40)
        return jid

    def wait(self, job_id, timeout_s):
        """The job's result dict; None when timeout_s passed without one; infra_fail when the job has stopped
        reporting (its machine went down), so that the engine submits it again without charging the loop."""
        out, end = self.jobs / job_id, time.time() + timeout_s
        t_sub = json.loads((out / "cmd.json").read_text())["t"]
        while True:
            if (out / "result.json").exists():
                time.sleep(1)
                return json.loads((out / "result.json").read_text())
            try:
                beat = float((out / "heartbeat").read_text().split()[0])
            except (OSError, ValueError, IndexError):
                beat = t_sub
            if time.time() - beat > 300:
                return {"status": "infra_fail", "job_id": job_id, "error": "no heartbeat for 300 s"}
            if time.time() > end:
                return None
            time.sleep(5)


# ------------------------------------------------------------------ random search (no model)

# name -> how one step moves it. "x": times or divided by 4/3; "1-x": the same on (1 - value); "+": plus or minus 1.
KNOBS = {"BATCH_SIZE": "x", "EPOCHS": "x", "WHITEN_BIAS_EPOCHS": "x", "BIAS_LR": "x", "HEAD_LR": "x",
         "WEIGHT_DECAY": "x", "SGD_MOMENTUM": "1-x", "MUON_LR": "x", "MUON_MOMENTUM": "1-x", "LABEL_SMOOTHING": "x",
         "BRIGHTNESS": "x", "CONTRAST": "x", "TRANSLATE": "+"}
MOVES = tuple((name, d) for name in KNOBS for d in ("down", "up"))
STEP = 4.0 / 3.0


def apply_move(src, name, direction):
    """One random-search move on a train.py source: (new_source, old, new), or None when it changes nothing."""
    old, f = random_search.read_value(src, name), (STEP if direction == "up" else 1 / STEP)
    kind = KNOBS[name]
    if kind == "+":
        new = min(4, max(0, int(old) + (1 if direction == "up" else -1)))
    elif kind == "1-x":
        new = float("%.4g" % (1 - (1 - old) / f))     # "up" moves the momentum up
    elif name == "BATCH_SIZE":
        new = int(round(old * f / 64.0)) * 64
    else:
        new = float("%.4g" % (old * f))
    if new == old:
        return None
    return random_search._set(src, name, new), old, new


# ------------------------------------------------------------------ entry

def install(cfg):
    """Point the unchanged engine at this task (this process only). Returns the GPU service."""
    consts = json.loads(Path(cfg["consts"]).read_text())
    runs = cfg.get("runs", 24)
    arms.prompts = make_prompts(consts, runs)
    arms.lint = random_search.lint = lint
    random_search.MOVES, random_search.apply_move = MOVES, apply_move
    acceptance.TRIPWIRE_CUMULATIVE = 5.5 * consts["sigma_used"]
    cfg.setdefault("base_train_py", str(BASE))
    cfg.setdefault("sigma_used", consts["sigma_used"])
    cfg.setdefault("gpu_offsets", consts["gpu_offsets"])
    cfg.setdefault("anchor", {"mean": 0.0, "n": consts["n_anchor"]})
    return LabGPU(cfg["host"], Path(cfg["run_dir"]) / "jobs", cfg["consts"], runs, cfg.get("job_timeout_s", 900))


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("cmd", choices=("loop", "nloop"))
    ap.add_argument("--config", required=True)
    ap.add_argument("--resume", action="store_true")
    a = ap.parse_args(argv)
    cfg = json.loads(Path(a.config).read_text())
    if isinstance(cfg.get("routing"), str):
        cfg["routing"] = json.loads(Path(cfg["routing"]).read_text())
    gpu = install(cfg)
    if (Path(cfg["run_dir"]) / "state.json").exists() and not a.resume:
        print("state.json exists; pass --resume", file=sys.stderr)
        return 2
    if a.cmd == "nloop":
        engine = random_search.RandomSearchEngine(cfg, Services(None, gpu, Clock()))
    else:
        models = ModelService(cfg["run_dir"], cfg["loop_id"], max_tokens=cfg.get("max_tokens", 8192))
        engine = arms.LoopEngine(cfg, Services(models, gpu, Clock()))
    status = engine.run()
    s = engine.state
    print("%s: %s after %d/%d slots, claimed best %s" % (cfg["loop_id"], status, s["slot_index"], s["n_slots"],
                                                          s["claimed_best_official"]))
    return 0 if status == "done" or engine.stop_reason == "stop_file" else 3


if __name__ == "__main__":
    sys.exit(main())
