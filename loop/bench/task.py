#!/usr/bin/env python3
"""The bench tasks (drug-property prediction, circle packing) for the unchanged loop engine (loop/core/arms.py).

Same approach as loop/cifar/task.py: in this process only, the engine's prompts, static check and job service are
replaced by the ones below. Jobs are CPU jobs on a lab machine (harness.py). The engine's audit tripwire is switched
off: it exists to catch implausible jumps on a tuned GPU task, while here large honest jumps from a simple starting
script are expected and the protection is the harness itself (scores are computed outside the candidate, and the
reported numbers come from held-out data or fresh seeds after the loop).

  python loop/bench/task.py loop  --config CFG [--resume]     language-model loop (engine arm D)
  python loop/bench/task.py nloop --config CFG [--resume]     random search over the Settings block, no model
Config: engine fields plus "task" (a key of tasks.json), "host", "consts" (calibration file).
"""
import argparse, ast, difflib, json, os, re, shlex, socket, subprocess, sys, time, types
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "core"))
sys.path.insert(0, str(HERE.parent / "cifar"))

import acceptance, arms, random_search   # noqa: E402
import prompts as P0                     # noqa: E402
from linter import LintResult            # noqa: E402
from services import Clock, ModelService, Services  # noqa: E402
import task as cifar_task                # noqa: E402  (only its LabGPU.wait and _chain are reused)

PY = str(HERE / ".venv/bin/python")
TASKS = json.loads((HERE / "tasks.json").read_text())
BASES = {"reg": "admet_reg.py", "clf": "admet_clf.py"}


def base_path(task):
    t = TASKS[task]
    return HERE / "base" / ("packing.py" if t["family"] == "packing" else BASES[t["kind"]])


# ------------------------------------------------------------------ prompts

TAIL = """
Budget. You have {n_slots} slots in total. One slot is one job. A crash and a replicate run each use a slot.
Rewriting an edit that failed to apply is free (at most 2 per slot).

Output format. Give edits to train.py as SEARCH/REPLACE blocks, aider style. Each SEARCH must match the current
file exactly once, line for line, including indentation. Blocks are applied in order. Example:

{example}

Reply as JSON with the fields requested in the task. Put the blocks, as text, in the field `edits_text`."""

ADMET = """# autoresearch program card

Goal: lower the loss of train.py, a script that predicts %(what)s of small molecules from their SMILES strings.

Data: %(n)d molecules with measured values, the public training file of a standard benchmark. loss = %(loss)s,
lower is better. In results tables the loss is the "official" column.

How a job scores train.py: 5-fold cross-validation, repeated with 3 different fold assignments (15 fits). Molecules
that share a ring skeleton (Murcko scaffold) always stay in the same fold, so every fold tests on unseen skeletons.
For each fold the harness calls fit_predict(train_smiles, train_y, test_smiles, seed) and scores the returned
predictions itself. The loss of a job is the mean over the 15 folds; it has a noise sd of about %(sigma).4f.
After the loop, the final script is trained on all the data and scored on held-out test molecules that no job sees.

The starting script: Morgan fingerprints (radius 2, 2048 bits) and a random forest. Its loss is about %(anchor).4f
and its job takes about %(secs)d s.

What you may change: train.py only, and anything in it: features, model family, ensembles, target transforms,
validation inside fit_predict.

Harness facts (enforced, stated so that no slot is wasted):
- train.py must keep defining fit_predict(train_smiles, train_y, test_smiles, seed). It returns one number per test
  molecule: %(ret)s.
- Available packages: numpy, scipy, pandas, scikit-learn, rdkit, lightgbm, xgboost, catboost. Nothing else. No
  downloads, no pretrained models, no file or network access, no GPU.
- A job has 8 CPU threads. It is stopped after 10 minutes (all 15 fits together) or above 6 GB of memory.
- Use the seed argument for all randomness.
- Simpler is better: a tiny gain that adds ugly code is not worth it.
"""

PACK = """# autoresearch program card

Goal: lower the loss of train.py, a script that packs %(n)d circles into the unit square so that the sum of their
radii is as large as possible. The circles may have different radii.

loss = minus the mean sum of radii over 3 runs with different seeds, lower is better. In results tables the loss is
the "official" column. The loss of a job has a noise sd of about %(sigma).4f.

The starting script: random restarts of SLSQP on the centres and radii until 80 percent of the time is used, then a
repair step that shrinks radii until the packing is valid. Its mean sum of radii is about %(score).4f. The best
published value for this problem is about %(ref).4f.

What you may change: train.py only, and anything in it.

Harness facts (enforced, stated so that no slot is wasted):
- train.py must keep defining solve(n, seed, time_limit), returning (centers, radii): arrays of shape (n, 2) and (n,).
- Each call of solve has time_limit = 30 s of CPU time on one thread. A call that uses more than 45 s voids the job.
- The harness checks the packing exactly in double precision: every circle inside [0, 1] x [0, 1] and no overlap,
  with a tolerance of 1e-10. An invalid packing is a crash.
- Available packages: numpy and scipy. Nothing else. No file or network access. Do not paste coordinates of a known
  packing: the script must compute its answer.
- Use the seed argument for all randomness.
- Simpler is better: a tiny gain that adds ugly code is not worth it.
"""
LOSS = {"mae": "mean absolute error", "auroc": "1 minus AUROC", "spearman": "1 minus the Spearman correlation"}
EXAMPLES = {"admet": ("N_TREES = 300", "N_TREES = 500"), "packing": ("MAX_ITER = 200", "MAX_ITER = 300")}


def make_prompts(task, c):
    """A stand-in for loop/core/prompts.py with the card and task wording of one bench task."""
    t = TASKS[task]
    m = types.ModuleType("bench_prompts")
    m.__dict__.update({k: v for k, v in vars(P0).items() if not k.startswith("__")})
    if t["family"] == "admet":
        card = ADMET % dict(what=t["what"], n=c["n_train"], loss=LOSS[t["metric"]], sigma=c["sigma_used"],
                            anchor=c["anchor"], secs=round(c["job_s"]),
                            ret="the probability of class 1" if t["kind"] == "clf" else "the predicted value")
        about = "the loss of a script that predicts %s of small molecules from SMILES (cross-validated %s)" % (
            t["what"], LOSS[t["metric"]])
    else:
        card = PACK % dict(n=t["n"], sigma=c["sigma_used"], score=-c["anchor"], ref=t["ref"])
        about = "the loss of a script that packs %d circles into the unit square (loss = minus the sum of radii)" % t["n"]
    a, b = EXAMPLES[t["family"]]
    example = "<<<<<<< SEARCH\n%s\n=======\n%s\n>>>>>>> REPLACE" % (a, b)

    def program_card(n_slots):
        return card + TAIL.replace("{n_slots}", str(n_slots)).replace("{example}", example)

    def base_step(train_py, table, last_tail, allow_abort):
        return P0.base_step(train_py, table, last_tail, allow_abort).replace("lower val_bpb", "lower the loss")

    def planner(train_py, n_items, table, memo, family_stats, existing, widen=False):
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
        return ("An idea for lowering %s:\nTitle: %s\nSpec: %s\nWill it plausibly lower the loss? Reply as JSON: "
                '{"vote": "yes" or "no", "reason": one sentence}.' % (about, item["title"], item["spec"]))

    def critic(diff, numbers, reason):
        return P0.critic(diff, numbers, reason).replace(
            "any attempt to change how val_bpb is measured, the data, the seed lines or the evaluation call",
            "any attempt to get around the measurement: file access, pasted answers, use of the test inputs' labels")

    m.program_card, m.base_step, m.planner, m.voter, m.critic = program_card, base_step, planner, voter, critic
    return m


# ------------------------------------------------------------------ static check

COMMON = {"math", "numpy", "scipy", "functools", "itertools", "collections", "typing", "random", "dataclasses",
          "warnings", "copy", "time", "heapq", "bisect"}
OK_IMPORTS = {"admet": COMMON | {"pandas", "sklearn", "rdkit", "lightgbm", "xgboost", "catboost", "re"},
              "packing": COMMON}
BAD_CALLS = {"open", "eval", "exec", "__import__", "globals", "input", "breakpoint", "compile"}
SIGS = {"admet": ("fit_predict", 4), "packing": ("solve", 3)}
CONST_LINE = re.compile(r"^[A-Z][A-Z0-9_]* = ")


def make_lint(family):
    def lint(source, base_source):
        changed = [l[2:] for l in difflib.ndiff(base_source.splitlines(), source.splitlines()) if l[:2] in ("+ ", "- ")]
        live = [l for l in changed if l.strip() and not l.strip().startswith("#")]
        cls = "constant" if live and all(CONST_LINE.match(l) for l in live) else "structural"
        hard = []
        add = lambda rule, node, msg: hard.append({"rule": rule, "line": getattr(node, "lineno", None), "message": msg})
        try:
            tree = ast.parse(source)
        except SyntaxError as e:
            return LintResult(False, [{"rule": "syntax", "line": e.lineno, "message": str(e)}], [], len(changed), cls)
        name, nargs = SIGS[family]
        defs = {n.name: n for n in tree.body if isinstance(n, ast.FunctionDef)}
        if name not in defs:
            add("interface", None, "train.py must define %s at module level" % name)
        elif len(defs[name].args.args) != nargs:
            add("interface", defs[name], "%s must take exactly %d arguments" % (name, nargs))
        for n in ast.walk(tree):
            if isinstance(n, (ast.Import, ast.ImportFrom)):
                for mod in ([a.name for a in n.names] if isinstance(n, ast.Import) else [n.module or ""]):
                    if mod.split(".")[0] not in OK_IMPORTS[family]:
                        add("import", n, "import of %s is not allowed" % mod)
            elif isinstance(n, ast.Call) and isinstance(n.func, ast.Name) and n.func.id in BAD_CALLS:
                add("call", n, "%s() is not allowed" % n.func.id)
            elif isinstance(n, ast.Attribute) and n.attr in ("read_csv", "read_pickle", "loadtxt", "load", "save",
                                                             "savetxt", "to_csv", "system", "popen"):
                add("io", n, "%s is not allowed (no file access)" % n.attr)
            elif isinstance(n, ast.Constant) and isinstance(n.value, str) and re.search(r"\.csv|admet_group|/homes/", n.value):
                add("io", n, "no data files may be named in train.py")
        return LintResult(not hard, hard, [], len(changed), cls)
    return lint


# ------------------------------------------------------------------ job service

class CpuJobs(cifar_task.LabGPU):
    """Starts harness.py for one candidate on a lab machine and finds the job again through its result file."""

    def __init__(self, host, jobs_dir, task, job_timeout_s=600):
        self.host, self.jobs, self.task, self.job_timeout_s = host, Path(jobs_dir), task, job_timeout_s
        self.jobs.mkdir(parents=True, exist_ok=True)

    def submit(self, train_py_path, *, seed, sealed, tag, kind, gpu_affinity=None):
        jid = "%s_%d" % (re.sub(r"[^A-Za-z0-9_.-]", "-", tag), time.time_ns() // 1000000)
        out = self.jobs / jid
        out.mkdir()
        cmd = [PY, str(HERE / "harness.py"), str(train_py_path), "--task", self.task, "--out", str(out), "--seed",
               str(seed), "--timeout", str(self.job_timeout_s), "--job-id", jid] + (["--sealed"] if sealed else [])
        (out / "cmd.json").write_text(json.dumps({"cmd": cmd, "host": self.host, "kind": kind, "t": time.time()}))
        log = str(out / "harness.log")
        if self.host == socket.gethostname().split(".")[0]:
            subprocess.Popen(cmd, stdout=open(log, "w"), stderr=subprocess.STDOUT, stdin=subprocess.DEVNULL,
                             start_new_session=True, cwd=str(HERE))
        else:
            remote = "cd %s && setsid nohup %s > %s 2>&1 < /dev/null" % (HERE, shlex.join(cmd), shlex.quote(log))
            subprocess.run(["ssh", "-f", "-o", "BatchMode=yes", "-o", "ConnectTimeout=10", self.host, remote],
                           check=True, timeout=40)
        return jid


# ------------------------------------------------------------------ random search (no model)

STEP = 4.0 / 3.0


def settings_of(src):
    """Names of the numeric constants in the '# Settings' block of a train.py."""
    names, on = [], False
    for ln in src.splitlines():
        if ln.startswith("# Settings"):
            on = True
        elif on and not ln.strip():
            break
        elif on and CONST_LINE.match(ln):
            names.append(ln.split(" = ")[0])
    return names


def apply_move(src, name, direction):
    """One random-search move: times or divided by 4/3 (integers rounded, moved by at least 1, kept at 1 or more)."""
    old, f = random_search.read_value(src, name), (STEP if direction == "up" else 1 / STEP)
    if isinstance(old, int):
        new = max(1, int(round(old * f)))
        if new == old:
            new = max(1, old + (1 if direction == "up" else -1))
    else:
        new = float("%.4g" % (old * f))
        if name in ("MAX_FEATURES", "TIME_FRACTION"):
            new = min(new, 1.0 if name == "MAX_FEATURES" else 0.95)
        if name == "INIT_LOW":
            new = min(new, 0.45)
    return None if new == old else (random_search._set(src, name, new), old, new)


# ------------------------------------------------------------------ entry

def install(cfg):
    task = cfg["task"]
    c = json.loads(Path(cfg["consts"]).read_text())[task]
    family = TASKS[task]["family"]
    arms.prompts = make_prompts(task, c)
    arms.lint = random_search.lint = make_lint(family)
    random_search.MOVES = tuple((n, d) for n in settings_of(base_path(task).read_text()) for d in ("down", "up"))
    random_search.apply_move = apply_move
    acceptance.TRIPWIRE_CUMULATIVE = acceptance.TRIPWIRE_SE = 1e12
    cfg.setdefault("base_train_py", str(base_path(task)))
    cfg.setdefault("sigma_used", c["sigma_used"])
    cfg.setdefault("gpu_offsets", {})
    cfg.setdefault("anchor", {"mean": c["anchor"], "n": c["n_anchor"]})
    return CpuJobs(cfg["host"], Path(cfg["run_dir"]) / "jobs", task, cfg.get("job_timeout_s", 600))


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("cmd", choices=("loop", "nloop"))
    ap.add_argument("--config", required=True)
    ap.add_argument("--resume", action="store_true")
    a = ap.parse_args(argv)
    cfg = json.loads(Path(a.config).read_text())
    jobs = install(cfg)
    if (Path(cfg["run_dir"]) / "state.json").exists() and not a.resume:
        print("state.json exists; pass --resume", file=sys.stderr)
        return 2
    if a.cmd == "nloop":
        engine = random_search.RandomSearchEngine(cfg, Services(None, jobs, Clock()))
    else:
        models = ModelService(cfg["run_dir"], cfg["loop_id"], max_tokens=cfg.get("max_tokens", 8192))
        engine = arms.LoopEngine(cfg, Services(models, jobs, Clock()))
    status = engine.run()
    s = engine.state
    print("%s: %s after %d/%d slots, claimed best %s" % (cfg["loop_id"], status, s["slot_index"], s["n_slots"],
                                                          s["claimed_best_official"]))
    return 0 if status == "done" or engine.stop_reason == "stop_file" else 3


if __name__ == "__main__":
    sys.exit(main())
