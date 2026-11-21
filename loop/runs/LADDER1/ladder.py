"""Offline planner ladder (LADDER1): how often does each model put the TOTAL_BATCH_SIZE cut in its FIRST idea list?

Run from the repo root:  .venv/bin/python -m loop.runs.LADDER1.ladder validate | collect | score
(or  .venv/bin/python loop/runs/LADDER1/ladder.py <cmd>).

validate  score the stored first lists of the 60 P2F loops (36 Sonnet-planned, 24 Qwen-planned)
collect   12 first lists per model, written to lists.jsonl (resumable: finished ok (model, rep) pairs are skipped)
score     table from lists.jsonl, written to scores.json

The first call reproduced here is the engine's slot-0 call: arms.py _plan("planner_backlog", 30) ->
prompts.planner(base train.py, 30, empty results table, no memo, no family stats, empty backlog) sent with
system=program_card(20), schema=BACKLOG_SCHEMA, temperature=None (server default), max_tokens=4096, timeout 600 s,
attempts=3 (call() default), and for open models extra={"chat_template_kwargs": {"enable_thinking": False}}.
Check: the prompt built here is 3818 characters (user + system), the prompt_chars the P2F planner call logged.
"""
import json, re, sys, threading
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
OUT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))
from loop.core import prompts, ledger          # noqa: E402
from loop.models import call as C              # noqa: E402

MODELS = ["qwen35-4b", "qwen35-9b", "gemma4-12b", "haiku", "sonnet"]
REPS, N_ITEMS, N_SLOTS = 12, 30, 20
ANCHOR = {"mean_adjusted": 1.114583, "n": 16}   # loop/runs/P2F/anchor.json, the bar every P2 loop starts with
OPEN_EXTRA = {"chat_template_kwargs": {"enable_thinking": False}}   # arms.py default open_extra
BASE = (ROOT / "loop/substrate/base/train_ampere.py").read_text()
HYPER = set(re.findall(r"^([A-Z][A-Z0-9_]*) = ", prompts.hyper_block(BASE), re.M))   # the 15 settings the planner sees


def build():
    user = prompts.planner(BASE, N_ITEMS, ledger.results_table([], ANCHOR), "", {}, [], False)
    return user, prompts.program_card(N_SLOTS)


# ---------------------------------------------------------------- scorers
# No reusable rule exists in analysis_p2.py or P2F, so this one is new. An item has the batch cut if its title or spec
# proposes LOWERING the optimizer batch (TOTAL_BATCH_SIZE, "total/optimizer batch size", bare "batch size"), either by
# naming a target below 2**19 or by a lowering word aimed at it. DEVICE_BATCH_SIZE (the micro batch) does not count,
# and neither does an increase, a split into chunks, or a lowering word aimed at something else.
_BS = r"(?:TOTAL_BATCH_SIZE|(?<!DEVICE_)(?<!device )(?<!micro[- ])(?<!micro)(?<!per-device )batch[ _-]?size)"
_LOW = r"(?:reduc|lower|decreas|halv|cut|shrink)\w*"
_UP = r"(?:larger|bigger|increas|doubl|more|grow|raise)\w*"
_W = r"(?:(?!effective|per|split|divid)[\w'*^-]+\W+){0,2}?"


def _num(m):
    s = m.group(0).lower().replace(",", "").replace("_", "")
    mm = re.match(r"2\s*(?:\*\*|\^)\s*(\d+)", s)
    if mm:
        return 2 ** int(mm.group(1))
    mm = re.match(r"(\d+(?:\.\d+)?)\s*k\b", s)
    if mm:
        return float(mm.group(1)) * 1024 if "1024" in s else float(mm.group(1)) * 1000
    return int(s)


def has_cut(item):
    t = item["title"] + " | " + item["spec"]
    if "DEVICE_BATCH_SIZE" in t and "TOTAL_BATCH_SIZE" not in t:   # a micro-batch change, not the optimizer batch
        return False
    for m in re.finditer(r"TOTAL_BATCH_SIZE\W+(?:[\w'*^-]+\W+){0,4}?(?:to|=|of|at)\s*(2\s*(?:\*\*|\^)\s*\d+|\d[\d,_]{3,}|\d+(?:\.\d+)?\s*[kK]\b)", t):
        v = _num(re.search(r"2\s*(?:\*\*|\^)\s*\d+|\d[\d,_]{3,}|\d+(?:\.\d+)?\s*[kK]\b", m.group(1)))
        if v < 2 ** 19:
            return True
    for m in re.finditer(r"\b" + _LOW + r"\W+" + _W + _BS + r"(?!\W+(?:noise|variance))", t, re.I):      # "halve TOTAL_BATCH_SIZE", "lower the total batch size"
        if not re.search(_UP, m.group(0), re.I):
            return True
    for m in re.finditer(_BS + r"\W+" + _W + r"(?:by half|to half|halved|in half|by 50|by 2x)", t, re.I):
        return True
    return False


def names_tbs(item):
    """Loose upper bound: the item names TOTAL_BATCH_SIZE at all (any direction)."""
    return "TOTAL_BATCH_SIZE" in item["title"] + " " + item["spec"]


def actionable(item):
    """Names a setting of the hyperparameter block (an ALL_CAPS name that exists in train.py's block)."""
    return any(re.search(r"\b" + h + r"\b", item["title"] + " " + item["spec"]) for h in HYPER)


# ---------------------------------------------------------------- validate on stored P2F data
def stored_first_lists():
    """{loop: (planner model, first-list items)}. The first list = the first backlog items; its size is the number of
    voter calls logged right after planner_backlog:0 divided by vote_k=3 (items kept after dedup)."""
    out = {}
    for d in sorted((ROOT / "loop/runs/P2F").iterdir()):
        if not (d / "state.json").exists():
            continue
        rows = [json.loads(l) for l in open(d / "calls.jsonl")]
        tags = [r["tag"].split(":", 1)[1] for r in rows]
        i = tags.index("planner_backlog:0")
        j = i + 1
        while tags[j].startswith("voter"):
            j += 1
        n = (j - i - 1) // 3
        out[d.name] = (rows[i]["model_name"], json.load(open(d / "state.json"))["backlog"][:n])
    return out


def validate():
    D, res = stored_first_lists(), {}
    for grp, pick in (("sonnet", lambda m: m == "sonnet"), ("qwen35-4b", lambda m: m != "sonnet")):
        L = [v[1] for v in D.values() if pick(v[0])]
        its = [i for l in L for i in l]
        res[grp] = {"loops": len(L), "strict_cut": sum(any(map(has_cut, l)) for l in L),
                    "names_tbs": sum(any(map(names_tbs, l)) for l in L),
                    "actionable_share": round(sum(map(actionable, its)) / len(its), 3), "items": len(its)}
    print(json.dumps(res, indent=1))
    (OUT / "validation.json").write_text(json.dumps(res, indent=1))
    for k, (m, l) in D.items():
        if m != "sonnet":
            print(k, "strict" if any(map(has_cut, l)) else "-", "names" if any(map(names_tbs, l)) else "-",
                  [i["title"] for i in l if has_cut(i)])


# ---------------------------------------------------------------- collect
_lock, _claude = threading.Lock(), threading.Semaphore(2)


def one(model, rep):
    user, system = build()
    is_open = C.load_registry()[model]["provider"] != "claude"
    kw = dict(system=system, schema=prompts.SCHEMAS["planner_backlog"], temperature=None, max_tokens=4096,
              timeout_s=600, extra=OPEN_EXTRA if is_open else None, tag="LADDER1:planner_backlog:%s:%d" % (model, rep),
              log_path=str(OUT / "calls.jsonl"))
    if is_open:
        r = C.call(model, user, **kw)
    else:
        with _claude:
            r = C.call(model, user, **kw)
    rec = {"model": model, "rep": rep, "ok": bool(r.ok and r.served_model == model), "served_model": r.served_model,
           "error": r.error, "error_class": r.error_class,
           "items": r.obj["items"] if r.ok and isinstance(r.obj, dict) else None, "text": r.text, "usage": r.usage,
           "cost": {"plan": r.cost_usd_plan, "nominal": r.cost_usd_nominal}, "latency_s": r.latency_s,
           "attempts": r.attempts}
    with _lock, open(OUT / "lists.jsonl", "a") as f:
        f.write(json.dumps(rec) + "\n")
    print(model, rep, rec["ok"], len(rec["items"] or []), flush=True)


def collect(models):
    done = set()
    p = OUT / "lists.jsonl"
    if p.exists():
        done = {(r["model"], r["rep"]) for r in map(json.loads, open(p)) if r["ok"]}
    jobs = [(m, k) for m in models for k in range(REPS) if (m, k) not in done]
    with ThreadPoolExecutor(10) as ex:   # claude calls are capped at 2 by a semaphore; each open host gets its own threads
        list(ex.map(lambda a: one(*a), jobs))


# ---------------------------------------------------------------- score
def score():
    last = {}
    for r in map(json.loads, open(OUT / "lists.jsonl")):
        last[(r["model"], r["rep"])] = r          # a later retry of the same rep replaces an earlier failure
    reg, res = C.load_registry(), {}
    for m in MODELS:
        rs = [r for (mm, _), r in sorted(last.items()) if mm == m]
        good = [r for r in rs if r["ok"]]
        for r in good:
            r["items"] = r["items"][:N_ITEMS]      # the engine keeps kept[:n], n = 30
        its = [i for r in good for i in r["items"]]
        res[m] = {"lists": len(rs), "format_failures": len(rs) - len(good), "strict_cut": sum(any(map(has_cut, r["items"])) for r in good),
                  "names_tbs": sum(any(map(names_tbs, r["items"])) for r in good),
                  "cut_items_per_list": round(sum(map(has_cut, its)) / max(1, len(good)), 2),
                  "mean_items": round(len(its) / max(1, len(good)), 1),
                  "actionable_share": round(sum(map(actionable, its)) / max(1, len(its)), 3),
                  "nominal_usd_per_list": round(sum(r["cost"]["nominal"] for r in rs) / max(1, len(rs)), 4),
                  "list_price_per_m": reg[m]["nominal_price_per_m"]}
    (OUT / "scores.json").write_text(json.dumps(res, indent=1))
    print(json.dumps(res, indent=1))


if __name__ == "__main__":
    cmd = sys.argv[1]
    {"validate": validate, "collect": lambda: collect(sys.argv[2:] or MODELS), "score": score}[cmd]()
