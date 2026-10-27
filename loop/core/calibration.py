"""Role-fitness calibration for the heterogeneous routing table (DESIGN.md section 9, PREREG H5).

    python -m loop.core.calibration --models sonnet haiku qwen35-9b gemma4-12b r1-qwen3-8b qwen35-4b \
        --out loop/runs/calibration            (run from /homes/rdubash/active-audit)

Every model gets the frozen specs in calib_specs.json wrapped in the production prompts (prompts.py: program card as the
system prompt, coder / repairer_lint / planner / analyst user prompts and schemas). Each reply is scored by code and
appended as one JSON line to <out>/results.jsonl (resumable). run() returns {"trials", "summary", "skipped"};
build_routing_table() turns that into the role -> model table. Nothing here judges training quality: coder trials measure
whether the edit applies, parses, lints, touches the requested symbols and leaves the forbidden ones alone.
See CALIBRATION.md for assumptions.
"""
import argparse
import ast
import difflib
import json
import math
import re
import sys
import threading
import time
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

HERE = Path(__file__).resolve().parent
for _p in (HERE, HERE.parent / "models"):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))
import call as call_mod  # noqa: E402  (loop/models/call.py)
import linter  # noqa: E402
import patch  # noqa: E402
import prompts  # noqa: E402

SPECS_PATH = HERE / "calib_specs.json"
BASE_PATH = HERE.parent / "substrate" / "base" / "train_ampere.py"
N_SLOTS = 20                              # PREREG N_SLOTS, only fills the program card
CODER_CLASSES = ("constant", "schedule", "structural")
ALL_CLASSES = CODER_CLASSES + ("repair", "planner", "analyst")
BARS = {"constant": 0.6, "schedule": 0.6, "structural": 0.75, "repair": 0.6, "planner": 0.6, "analyst": 0.6}
FALLBACK, CRITIC = "sonnet", "opus"
# Placeholder $/M tokens, used only to ORDER models (open models come from models.json nominal_price_per_m).
CLAUDE_PRICES = {"haiku": {"input": 1.0, "output": 5.0}, "sonnet": {"input": 3.0, "output": 15.0},
                 "opus": {"input": 15.0, "output": 75.0}}
FAMILIES = (("r1-", "deepseek"), ("deepseek", "deepseek"), ("qwen", "qwen"), ("gemma", "gemma"), ("gptoss", "gptoss"),
            ("haiku", "claude"), ("sonnet", "claude"), ("opus", "claude"))
OPEN_EXTRA = {"chat_template_kwargs": {"enable_thinking": False}}   # arms.py open_extra: production sends it to every open model
TIMEOUT_S, MAX_TOKENS = 600, 4096                                   # services.ModelService defaults
PLANNER_RESULT_DECIMALS = 4   # planner specs propose new values, so only result-like numbers (val_bpb, deltas) are checked
ANALYST_MAX_WORDS = 300       # the prompt asks for 150; the cap only catches runaway generation
MAX_CLAUDE_INFLIGHT = 4       # the call layer also enforces this with a flock semaphore
OPEN_WORKERS = 2


# ---------------------------------------------------------------- inputs

def render_table(inp):
    """Same layout as ledger.results_table, from an input's rows and incumbent."""
    lines = ["incumbent now: mean_adjusted %.4f over %d run(s)" % (inp["incumbent"]["mean_adjusted"], inp["incumbent"]["n"]),
             "", "| slot | description | official | delta vs incumbent | decision | runs | flags |", "|---|---|---|---|---|---|---|"]
    return "\n".join(lines + ["| %s | %s | %s | %s | %s | %s | %s |" % (
        r["slot"], r["description"], r["official"], r["delta"], r["decision"], r["runs"], r["flags"]) for r in inp["rows"]])


ANCHOR_TABLE = render_table({"incumbent": {"mean_adjusted": 1.0842, "n": 24}, "rows": []})


def load_specs(path=None):
    return json.loads(Path(path or SPECS_PATH).read_text())


def apply_edits(src, edits):
    for old, new in edits:
        assert src.count(old) == 1, f"edit anchor must be unique: {old!r}"
        src = src.replace(old, new)
    return src


def repair_inputs(base, spec):
    """(broken_source, lint hard violations) for a repair spec: the legitimate change plus the deliberate violation."""
    broken = apply_edits(apply_edits(base, spec["legit_edits"]), spec["violation_edits"])
    return broken, linter.lint(broken, base).hard


def tasks(specs, classes, base, limit=None):
    """[(class, spec, user_prompt, schema)] for the requested classes, built with the production prompts."""
    out = []
    for c in classes:
        if c in CODER_CLASSES:
            items = [(s, prompts.coder(base, {k: s[k] for k in ("title", "family", "spec")}, ANCHOR_TABLE, "", {}, False),
                      prompts.SCHEMAS["coder"]) for s in specs["coder"] if s["edit_class"] == c]
        elif c == "repair":
            items = []
            for s in specs["repair"]:
                broken, hard = repair_inputs(base, s)
                items.append((s, prompts.repairer_lint(broken, hard), prompts.SCHEMAS["repairer"]))
        elif c == "planner":
            items = [(s, prompts.planner(base, s["n_items"], render_table(s), s["memo"], {}, []), prompts.SCHEMAS["planner"])
                     for s in specs["planner"]]
        else:
            items = [(s, prompts.analyst(render_table(s), s["memo"], {}), prompts.SCHEMAS["analyst"]) for s in specs["analyst"]]
        out += [(c, *it) for it in items[:limit]]
    return out


# ---------------------------------------------------------------- scoring

def wilson_lower(k, n, z=1.96):
    """Lower end of the Wilson score interval for k successes in n trials (0.0 when n == 0)."""
    if n == 0:
        return 0.0
    p, d = k / n, 1 + z * z / n
    return max(0.0, (p + z * z / (2 * n) - z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n))) / d)


def _symbols(src):
    """{name: ast.dump} for top-level defs/classes/assignments and Class.method; comments and spacing are ignored."""
    out = {}

    def add(prefix, node):
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
            out[prefix + node.name] = ast.dump(node)
        elif isinstance(node, (ast.Assign, ast.AnnAssign)):
            for t in (node.targets if isinstance(node, ast.Assign) else [node.target]):
                if isinstance(t, ast.Name):
                    out[prefix + t.id] = ast.dump(node)
    for node in ast.parse(src).body:
        add("", node)
        if isinstance(node, ast.ClassDef):
            for sub in node.body:
                add(node.name + ".", sub)
    return out


def _rx(entry):
    """A plain identifier (optionally dotted) matches as a whole word, anything else is a regex."""
    return rf"\b{re.escape(entry)}\b" if re.fullmatch(r"[A-Za-z_][A-Za-z0-9_.]*", entry) else entry


def changed_text(ref, new):
    d = list(difflib.unified_diff(ref.splitlines(), new.splitlines(), n=0, lineterm=""))[2:]
    return "\n".join(x[1:] for x in d if x[:1] in "+-")


def score_edit(apply_src, ref, edits_text, spec):
    """Score SEARCH/REPLACE text. apply_src is what the edits apply to, ref is the clean base that lint and diffs use
    (equal for coder specs; the lint-broken file vs the base for repair specs, so a correct repair must restore ref's
    version of every must_not_touch symbol). Returns (reasons, lint_edit_class); no reasons means pass."""
    new, rep = patch.apply_text(apply_src, edits_text or "")
    if not rep["ok"]:
        return ["apply:" + (rep["reason"] or "failed")], None
    try:
        compile(new, "train.py", "exec")
    except (SyntaxError, ValueError) as e:
        return [f"parse:{type(e).__name__}"], None
    lint = linter.lint(new, ref)
    reasons = ["lint:" + h["rule"] for h in lint.hard]
    diff = changed_text(ref, new)
    if not diff:
        reasons.append("noop")
    for t in spec["target_symbols"]:
        if not re.search(_rx(t), diff, re.M):
            reasons.append("target:" + t)
    old_syms, new_syms = _symbols(ref), _symbols(new)
    for s in spec["must_not_touch"]:
        touched = (new_syms.get(s) != old_syms[s]) if s in old_syms else bool(re.search(_rx(s), diff, re.M))
        if touched:
            reasons.append("touched:" + s)
    for e in spec.get("expect", []):
        neg = e.startswith("!")
        if bool(re.search(e[1:] if neg else e, new, re.M)) == neg:
            reasons.append("expect:" + e)
    return reasons, lint.edit_class


_NUM = re.compile(r"(?<![\w.])\d+(?:\.(\d+))?(?!\w)")


def numbers(text, min_decimals=0):
    """Magnitudes of the standalone numbers in text with at least min_decimals decimals (sign dropped, list numbering
    at line starts ignored)."""
    text = re.sub(r"(?m)^\s*\(?\d+[.)]\s+", "", text)
    return {float(m.group(0)) for m in _NUM.finditer(text) if len(m.group(1) or "") >= min_decimals}


def check_numbers(output_text, prompt, min_decimals=0):
    """Numbers in output_text that the model was never shown."""
    return sorted(numbers(output_text, min_decimals) - numbers(prompt))


def _num_reason(bad):
    return ["numbers:" + ",".join(f"{x:g}" for x in bad)] if bad else []


def score_coder(obj, apply_src, ref, spec):
    if obj.get("abort"):
        return ["abort"], None
    return score_edit(apply_src, ref, obj.get("edits_text", ""), spec)


def score_planner(obj, inp, prompt):
    items = obj.get("items", []) if isinstance(obj, dict) else []
    reasons = []
    if len(items) != inp["n_items"]:
        reasons.append(f"count:{len(items)}!={inp['n_items']}")
    for k in ("title", "spec"):
        if any(not str(it.get(k, "")).strip() for it in items):
            reasons.append("empty:" + k)
    titles = [str(it.get("title", "")).strip().lower() for it in items]
    if len(set(titles)) != len(titles):
        reasons.append("duplicate:title")
    text = "\n".join(str(it.get(k, "")) for it in items for k in ("title", "spec"))
    return reasons + _num_reason(check_numbers(text, prompt, PLANNER_RESULT_DECIMALS))


def score_analyst(obj, prompt):
    memo = obj.get("memo") if isinstance(obj, dict) else None
    if not isinstance(memo, str) or not memo.strip():
        return ["empty:memo"]
    reasons = ["long:memo"] if len(memo.split()) > ANALYST_MAX_WORDS else []
    return reasons + _num_reason(check_numbers(memo, prompt))


def summarize(trials, bars=None):
    """{model: {class: {n, k, n_infra, rate, lb, bar, qualified, cost, latency, fails}}}. Infra trials are excluded from n."""
    bars = bars or BARS
    groups = {}
    for t in trials:
        groups.setdefault((t["model"], t["edit_class"]), []).append(t)
    out = {}
    for (m, c), ts in groups.items():
        real = [t for t in ts if not t.get("infra")]
        n, k = len(real), sum(1 for t in real if t["ok"])
        lb = wilson_lower(k, n)
        out.setdefault(m, {})[c] = {
            "n": n, "k": k, "n_infra": len(ts) - n, "rate": k / n if n else 0.0, "lb": lb, "bar": bars[c],
            "qualified": n > 0 and lb >= bars[c],
            "cost": sum(t["cost"] for t in real) / n if n else 0.0,
            "latency": sum(t["latency"] for t in real) / n if n else 0.0,
            "fails": dict(Counter(r if r.startswith("lint:") else r.split(":", 1)[0]
                                  for t in real if not t["ok"] for r in t["reasons"][:1]))}
    return out


# ---------------------------------------------------------------- routing table

def family(name):
    for key, fam in FAMILIES:
        if key in name.lower():
            return fam
    return name


def default_prices():
    reg = call_mod.load_registry()
    return {n: (e.get("nominal_price_per_m") or CLAUDE_PRICES.get(n)) for n, e in reg.items()}


def _price(p):
    return float(p) if not isinstance(p, dict) else float(p["input"]) + float(p["output"])


def build_routing_table(results, prices=None, out_dir=None):
    """results: the dict from run() (needs "summary"; "trials" for totals). prices: {model: $/M or {input, output}}.
    Cheapest qualifying model wins every role; sonnet is always a candidate; critic is opus; voter is the cheapest
    qualifying planner-class model of another family than planner_cheap. "planner" repeats planner_cheap because
    arms.py routes planner roles through that key. Every top-level string value is a model name (arms.py health-checks
    them all), so metadata lives in dict/list values. With out_dir, also writes routing_table.json, routing_table.md and
    summary.json there."""
    prices = prices or default_prices()
    summ = results["summary"]

    def cands(cls):
        qual = [m for m in summ if summ[m].get(cls, {}).get("qualified")]
        return qual + ([FALLBACK] if FALLBACK not in qual else [])

    def key(cls):
        return lambda m: (_price(prices[m]), summ.get(m, {}).get(cls, {}).get("cost", 0.0), m)

    def best(cls, pool=None):
        pool = cands(cls) if pool is None else pool
        return min(pool, key=key(cls)) if pool else None

    planner = best("planner")
    cross = [m for m in cands("planner") if family(m) != family(planner)]
    other = [m for m in cands("planner") if m != planner]
    voter = best("planner", cross) or best("planner", other) or planner
    coder = {c: best(c) for c in CODER_CLASSES}
    notes = [f"coder {c}: no cheaper model qualified, {FALLBACK} fallback" for c, m in coder.items() if m == FALLBACK]
    if not cross:
        notes.append("no qualifying planner-class model of another family: voter shares the planner's family or is the planner")
    table = {
        "coder_by_class": coder, "repairer": best("repair"), "planner_cheap": planner, "planner": planner,
        "analyst": best("analyst"), "voter": voter, "critic": CRITIC, "fallback": FALLBACK,
        "qualified": {m: sorted(c for c, v in cs.items() if v["qualified"]) for m, cs in summ.items()},
        "evidence": {m: {c: {x: round(v[x], 4) if isinstance(v[x], float) else v[x] for x in ("n", "k", "lb", "bar")}
                         for c, v in cs.items()} for m, cs in summ.items()},
        "notes": notes, "thresholds": BARS}
    trials = results.get("trials", [])
    table["calibration"] = {"cost_nominal": round(sum(t["cost"] for t in trials), 4),
                            "cost_plan": round(sum(t.get("cost_plan", 0.0) for t in trials), 4),
                            "latency_s": round(sum(t["latency"] for t in trials), 1), "n_trials": len(trials)}
    if out_dir:
        write_outputs(out_dir, results, table)
    return table


def routing_markdown(table, summary):
    roles = [("coder " + c, m) for c, m in table["coder_by_class"].items()] + [
        ("repairer", table["repairer"]), ("planner_cheap (also planner)", table["planner_cheap"]), ("analyst", table["analyst"]),
        ("voter", table["voter"]), ("critic", table["critic"])]
    lines = ["# Routing table", "", "| role | model |", "|---|---|"] + [f"| {r} | {m} |" for r, m in roles]
    lines += ["", "## Evidence (k/n passed, Wilson 95% lower bound vs bar; infra failures excluded)", "",
              "| model | class | k/n | lb | bar | qualified | mean cost | mean latency s | top failure |", "|---|---|---|---|---|---|---|---|---|"]
    for m in sorted(summary):
        for c in ALL_CLASSES:
            v = summary[m].get(c)
            if v:
                top = max(v["fails"], key=v["fails"].get) if v["fails"] else "-"
                lines.append(f"| {m} | {c} | {v['k']}/{v['n']} | {v['lb']:.3f} | {v['bar']} | {'yes' if v['qualified'] else 'no'} "
                             f"| {v['cost']:.4f} | {v['latency']:.1f} | {top} |")
    lines += [""] + [f"- {n}" for n in table["notes"]]
    c = table["calibration"]
    lines += [f"- calibration: {c['n_trials']} trials, nominal cost ${c['cost_nominal']}, plan cost ${c['cost_plan']}, "
              f"summed call latency {c['latency_s']} s"]
    return "\n".join(lines) + "\n"


# ---------------------------------------------------------------- running trials

def model_cfg(name, reg):
    """Claude as is; open models at temperature 0.7 (so the 2 samples differ) with the production open_extra."""
    if reg[name]["provider"] == "claude":
        return {"temperature": None, "extra": None, "max_tokens": MAX_TOKENS}
    return {"temperature": 0.7, "extra": OPEN_EXTRA, "max_tokens": MAX_TOKENS}


def run_trial(model, cls, spec, user, schema, k, base, cfg, call_fn, out_dir):
    t = {"model": model, "spec_id": spec["id"], "edit_class": cls, "sample": k, "ok": False, "infra": False,
         "reasons": [], "cost": 0.0, "cost_plan": 0.0, "latency": 0.0, "served_model": None, "lint_class": None}
    try:
        r = call_fn(model, user, system=prompts.program_card(N_SLOTS), schema=schema, temperature=cfg["temperature"],
                    max_tokens=cfg["max_tokens"], tag=f"calib:{model}:{spec['id']}:{k}", extra=cfg["extra"], attempts=1,
                    timeout_s=TIMEOUT_S, log_path=str(Path(out_dir) / "calls.jsonl"))
    except call_mod.ContextTooLong as e:
        return {**t, "infra": True, "reasons": [f"context:{str(e)[:80]}"]}
    t.update(cost=r.cost_usd_nominal, cost_plan=r.cost_usd_plan, latency=r.latency_s, served_model=r.served_model)
    d = Path(out_dir) / "replies" / model
    d.mkdir(parents=True, exist_ok=True)
    (d / f"{spec['id']}.{k}.txt").write_text(r.text or "")
    if r.served_model != model:
        return {**t, "infra": True, "reasons": [f"fallback:{r.served_model}"]}
    if not r.ok:
        infra = r.error_class in ("infra", "context")
        return {**t, "infra": infra, "reasons": [f"{'call' if infra else 'schema'}:{(r.error or '')[:100]}"]}
    if cls in CODER_CLASSES:
        t["reasons"], t["lint_class"] = score_coder(r.obj, base, base, spec)
    elif cls == "repair":
        t["reasons"], t["lint_class"] = score_coder(r.obj, repair_inputs(base, spec)[0], base, spec)
    elif cls == "planner":
        t["reasons"] = score_planner(r.obj, spec, user)
    else:
        t["reasons"] = score_analyst(r.obj, user)
    t["ok"] = not t["reasons"]
    return t


def load_trials(path):
    """Latest line per (model, spec_id, sample)."""
    latest = {}
    if Path(path).exists():
        for line in Path(path).read_text().splitlines():
            if line.strip():
                t = json.loads(line)
                latest[(t["model"], t["spec_id"], t["sample"])] = t
    return list(latest.values())


def run(models, classes=None, samples_per_spec=2, out_dir="loop/runs/calibration", *, call_fn=None, healthy_fn=None,
        specs=None, base=None, limit=None):
    """Run the calibration; resumable (finished non-infra trials in results.jsonl are skipped). Unhealthy open models are
    skipped and listed in "skipped". limit=N uses the first N specs of each class (plumbing checks)."""
    call_fn, healthy_fn = call_fn or call_mod.call, healthy_fn or call_mod.healthy
    classes = list(classes or ALL_CLASSES)
    specs = specs or load_specs()
    base = base if base is not None else BASE_PATH.read_text()
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    results_path, lock, reg = out / "results.jsonl", threading.Lock(), call_mod.load_registry()
    done = {(t["model"], t["spec_id"], t["sample"]) for t in load_trials(results_path) if not t.get("infra")}
    plan = tasks(specs, classes, base, limit)
    skipped = {m: "unhealthy" for m in models if not healthy_fn(m)}
    live = [m for m in models if m not in skipped]
    n_claude = sum(reg[m]["provider"] == "claude" for m in live) or 1

    def work(m, cls, spec, user, schema, k):
        t = run_trial(m, cls, spec, user, schema, k, base, model_cfg(m, reg), call_fn, out)
        t["ts"] = time.strftime("%Y-%m-%dT%H:%M:%S")
        with lock:
            with open(results_path, "a") as f:
                f.write(json.dumps(t) + "\n")

    futures, pools = [], []
    for m in live:
        w = max(1, MAX_CLAUDE_INFLIGHT // n_claude) if reg[m]["provider"] == "claude" else OPEN_WORKERS
        pool = ThreadPoolExecutor(w)
        pools.append(pool)
        futures += [pool.submit(work, m, cls, spec, user, schema, k)
                    for (cls, spec, user, schema) in plan for k in range(samples_per_spec)
                    if (m, spec["id"], k) not in done]
    try:
        for f in futures:
            f.result()
    finally:
        for p in pools:
            p.shutdown(cancel_futures=True)
    trials = load_trials(results_path)
    return {"trials": trials, "summary": summarize(trials), "skipped": skipped, "out_dir": str(out)}


def write_outputs(out_dir, results, table):
    out = Path(out_dir)
    (out / "routing_table.json").write_text(json.dumps(table, indent=1) + "\n")
    (out / "routing_table.md").write_text(routing_markdown(table, results["summary"]))
    (out / "summary.json").write_text(json.dumps(results["summary"], indent=1) + "\n")


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--models", nargs="+", required=True)
    ap.add_argument("--out", default="loop/runs/calibration")
    ap.add_argument("--classes", nargs="+", choices=ALL_CLASSES, default=list(ALL_CLASSES))
    ap.add_argument("--samples", type=int, default=2)
    ap.add_argument("--limit", type=int, default=None, help="first N specs per class (plumbing check)")
    ap.add_argument("--rebuild", action="store_true", help="no model calls: rebuild the table from <out>/results.jsonl")
    a = ap.parse_args(argv)
    if a.rebuild:
        Path(a.out).mkdir(parents=True, exist_ok=True)
        trials = load_trials(Path(a.out) / "results.jsonl")
        results = {"trials": trials, "summary": summarize(trials), "skipped": {}}
    else:
        results = run(a.models, a.classes, a.samples, a.out, limit=a.limit)
    table = build_routing_table(results, out_dir=a.out)
    print(routing_markdown(table, results["summary"]))
    if results["skipped"]:
        print("skipped (health check failed):", results["skipped"])


if __name__ == "__main__":
    main()
