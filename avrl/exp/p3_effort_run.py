"""P3 runner: one Haiku critic pass per AR3 item per reasoning-effort level.

Question (runs/P3/PREREG.md): does reasoning effort change how well Haiku's stated confidence separates right
from wrong critic verdicts, and at equal dollars is escalating to Sonnet better than thinking harder?

Uses exactly AR3's frozen SYSTEM prompt and SCHEMA (sha256 checked against runs/AR3/critic_hashes.json), the item
text as the user prompt, and loop/models/call.py call("haiku", ..., effort=level). Level "default" means no --effort
flag (effort=None, the CLI default). Items: the AR3 items whose final split (items.parquet logic, ar3_features.load_items)
is burn or test, which leaves out the 6 natural probes and the 23 controls adjudicated "natural" (not clean, so the
single-pass judging rule cannot score them).

Resumable: one row per finished (item, level) is appended to runs/P3/calls_<level>.jsonl; keys with a valid row are
skipped. A row is valid only if the call succeeded, was served by haiku (a fallback to sonnet is recorded, ok=false),
and the output has a problems list and an integer judgment_confidence.

  python avrl/exp/p3_effort_run.py --items day0 --levels default,low,medium,high --max-calls 80
  python avrl/exp/p3_effort_run.py --items all --max-calls 300 --dry-run

Real runs need AR_LOCK_DIR and AR_BREAKER_FILE in the environment (shared 4-call concurrency cap) and --max-calls.
A real run first moves to a fresh empty temp directory so the CLI loads no project-level instruction or memory files, and removes CLAUDE_EFFORT and
CLAUDE_CODE_EFFORT_LEVEL from the environment so an inherited override cannot set every level to the same effort.
"""
import argparse
import hashlib
import json
import os
import subprocess
import sys
import tempfile
import threading
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
from pathlib import Path

HERE = Path(__file__).resolve().parent
AVRL = HERE.parent
for _p in (HERE, AVRL.parent / "loop" / "models"):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))
import ar3_features as F  # noqa: E402
import ar3_run_critic as R  # noqa: E402  (frozen SYSTEM, SCHEMA_STR)
import call as call_mod  # noqa: E402  (loop/models/call.py)

AR3_DIR = AVRL / "runs" / "AR3"
P3_DIR = AVRL / "runs" / "P3"
LEVELS = ("default", "low", "medium", "high")  # "default" = no --effort flag
EFFORT_ENV_VARS = ("CLAUDE_EFFORT", "CLAUDE_CODE_EFFORT_LEVEL")
SCHEMA_OBJ = json.loads(R.SCHEMA_STR)
TIMEOUT_S = 600  # AR3 used 300; high effort may think longer and a timeout would censor the slow items
DAY0_ALLOC = {"control": 4, "N-big": 3, "N-mid": 4, "N-stale": 3, "V-body": 3, "V-coh": 3}  # 20 burn items
_tl = threading.local()  # CLI metadata of the last _run_claude on this thread (see install_capture)


# ---------------------------------------------------------------- inputs

def check_prompt_hashes(ar3_dir=AR3_DIR):
    """The imported SYSTEM and SCHEMA_STR must hash to the values recorded when AR3 ran."""
    rec = json.loads((Path(ar3_dir) / "critic_hashes.json").read_text())
    got = {"system_sha256": hashlib.sha256(R.SYSTEM.encode()).hexdigest(),
           "schema_sha256": hashlib.sha256(R.SCHEMA_STR.encode()).hexdigest()}
    for k, v in got.items():
        if rec.get(k) != v:
            raise RuntimeError(f"{k} mismatch: recorded {rec.get(k)}, imported {v}")
    return got


def level_effort(level):
    if level not in LEVELS:
        raise ValueError(f"level must be one of {LEVELS}, got {level!r}")
    return None if level == "default" else level


def _h(salt, item_id):
    return hashlib.sha256(f"{salt}|{item_id}".encode()).hexdigest()


def select_items(ar3_dir=AR3_DIR):
    """(items, counts): AR3 items with final split burn or test, in a fixed pseudo-random order.
    Raises if items.jsonl does not match its recorded sha256."""
    ar3_dir = Path(ar3_dir)
    want = (ar3_dir / "items.sha256").read_text().split()[0]
    got = hashlib.sha256((ar3_dir / "items.jsonl").read_bytes()).hexdigest()
    if want != got:
        raise RuntimeError(f"items.jsonl sha256 {got[:12]} != items.sha256 {want[:12]}")
    allit = F.load_items(ar3_dir)
    keep = [it for it in allit if it["split"] in ("burn", "test")]
    counts = {"all_items": len(allit),
              "probe_orig": sum(it["orig_split"] == "probe" for it in allit),
              "natural_controls_moved_to_probe": sum(it["truth"] == "natural" and it["orig_split"] != "probe" for it in allit),
              "burn": sum(it["split"] == "burn" for it in keep), "test": sum(it["split"] == "test" for it in keep),
              "selected": len(keep), "entries": len({it["entry_key"] for it in keep})}
    return sorted(keep, key=lambda it: _h("p3-order", it["item_id"])), counts


def day0_items(items):
    """The 20 fixed Day 0 items: DAY0_ALLOC per kind, burn split only, the first by sha256("p3-day0|item_id")."""
    out = []
    for kind, n in DAY0_ALLOC.items():
        pool = sorted((it for it in items if it["split"] == "burn" and it["kind"] == kind), key=lambda it: _h("p3-day0", it["item_id"]))
        if len(pool) < n:
            raise ValueError(f"burn split has {len(pool)} {kind} items, need {n}")
        out += pool[:n]
    return sorted(out, key=lambda it: _h("p3-order", it["item_id"]))


# ---------------------------------------------------------------- rows

def valid_output(so):
    """Schema-valid critic answer: a problems list and an integer judgment_confidence in 0..100."""
    c = so.get("judgment_confidence") if isinstance(so, dict) else None
    return isinstance(so, dict) and isinstance(so.get("problems"), list) and isinstance(c, int) and not isinstance(c, bool) and 0 <= c <= 100


def calls_path(run_dir, level):
    return Path(run_dir) / f"calls_{level}.jsonl"


def read_rows(run_dir, level):
    p = calls_path(run_dir, level)
    out = []
    if p.exists():
        for ln in p.read_text().splitlines():
            try:
                out.append(json.loads(ln))
            except json.JSONDecodeError:
                continue  # truncated last line of a killed run: that key is simply redone
    return out


def done_keys(run_dir, levels):
    return {(r["item_id"], lv) for lv in levels for r in read_rows(run_dir, lv) if r.get("ok") and valid_output(r.get("structured_output"))}


def make_row(item, level, r, cli_version=""):
    fallback = r.served_model != "haiku"
    meta = getattr(_tl, "last", None) or {}
    ids = meta.get("model_ids") or []
    return {"item_id": item["item_id"], "entry_key": item["entry_key"], "level": level, "split": item["split"],
            "ok": bool(r.ok) and not fallback and valid_output(r.obj), "call_ok": bool(r.ok), "fallback": fallback,
            "error": r.error, "error_class": r.error_class, "structured_output": r.obj, "usage": r.usage,
            "cost_usd": r.cost_usd_plan, "latency_s": round(r.latency_s, 3), "attempts": r.attempts,
            "served_model": r.served_model, "requested_model": r.requested_model,
            "model_id": ids[0] if len(ids) == 1 else None, "model_ids": ids, "argv_effort": meta.get("argv_effort"),
            "raw_usage": meta.get("raw_usage"), "cli_version": cli_version, "cwd": os.getcwd(),
            "ts": datetime.now(timezone.utc).isoformat(timespec="milliseconds")}


def enter_neutral_cwd():
    """chdir to a fresh empty temp dir so the CLI loads no project-level instruction or memory files; every call sees the same context."""
    d = tempfile.mkdtemp(prefix="p3_run_")
    os.chdir(d)
    return d


def install_capture():
    """Wrap call_mod._run_claude so each thread remembers the last CLI call's argv effort, modelUsage ids and raw usage."""
    orig = call_mod._run_claude

    def wrapped(argv, stdin_text, timeout):
        rc, out, err = orig(argv, stdin_text, timeout)
        meta = {"argv_effort": argv[argv.index("--effort") + 1] if "--effort" in argv else None, "model_ids": [], "raw_usage": None}
        try:
            d = json.loads(out)
            meta["model_ids"] = sorted((d.get("modelUsage") or {}).keys())
            meta["raw_usage"] = d.get("usage")
        except (ValueError, AttributeError):
            pass
        _tl.last = meta
        return rc, out, err

    call_mod._run_claude = wrapped
    return orig


# ---------------------------------------------------------------- sweep

def pending_keys(items, levels, run_dir):
    """(item, level) pairs without a valid row, item-major so a partial run covers whole items."""
    done = done_keys(run_dir, levels)
    return [(it, lv) for it in items for lv in levels if (it["item_id"], lv) not in done]


def run_sweep(items, levels, run_dir, max_calls, workers=4, call_fn=None, cli_version="", dry_run=False):
    """Call haiku once per pending (item, level), never more than max_calls call() invocations in this run.
    Work beyond the cap stays pending for the next invocation. Stops launching once the attempts reported by finished
    calls reach max_calls (retries count). Returns a summary dict."""
    call_fn = call_fn or call_mod.call
    run_dir = Path(run_dir)
    todo = pending_keys(items, levels, run_dir)
    n_pending = len(todo)
    todo = todo[:max_calls]
    summary = {"pending": n_pending, "planned": len(todo), "started": 0, "ok": 0, "invalid": 0, "fallback": 0,
               "attempts": 0, "cost_usd": 0.0, "dry_run": dry_run}
    if dry_run or not todo:
        return summary
    for v in EFFORT_ENV_VARS:  # an inherited override would silently set every level to the same effort
        os.environ.pop(v, None)
    run_dir.mkdir(parents=True, exist_ok=True)
    lock, errors = threading.Lock(), []

    def work(task):
        item, level = task
        with lock:
            if errors or summary["started"] >= max_calls or summary["attempts"] >= max_calls:
                return
            summary["started"] += 1
        try:
            _tl.last = None
            r = call_fn("haiku", item["text"], system=R.SYSTEM, schema=SCHEMA_OBJ, effort=level_effort(level), attempts=3,
                        timeout_s=TIMEOUT_S, tag=f"p3:{level}:{item['item_id']}", log_path=str(run_dir / "calllog.jsonl"))
            row = make_row(item, level, r, cli_version)
        except Exception as e:  # a crash is systemic (bad effort value, missing binary): stop launching and re-raise
            with lock:
                errors.append(e)
            return
        with lock:
            with open(calls_path(run_dir, level), "a") as f:
                f.write(json.dumps(row) + "\n")
            summary["attempts"] += r.attempts
            summary["cost_usd"] += r.cost_usd_plan
            summary["ok"] += row["ok"]
            summary["invalid"] += not row["ok"]
            summary["fallback"] += row["fallback"]

    with ThreadPoolExecutor(max_workers=workers) as ex:
        list(ex.map(work, todo))
    if errors:
        raise errors[0]
    return summary


# ---------------------------------------------------------------- cli

def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--levels", default=",".join(LEVELS), help="comma list of: " + ", ".join(LEVELS))
    ap.add_argument("--items", choices=["all", "day0"], default="all")
    ap.add_argument("--max-calls", type=int, help="hard cap on call() invocations in this run (required unless --dry-run)")
    ap.add_argument("--dry-run", action="store_true", help="count pending work, make no calls, write nothing")
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--run-dir", default=str(P3_DIR))
    args = ap.parse_args(argv)
    args.run_dir = str(Path(args.run_dir).resolve())
    levels = [lv.strip() for lv in args.levels.split(",") if lv.strip()]
    for lv in levels:
        level_effort(lv)
    if not args.dry_run:
        if args.max_calls is None or args.max_calls < 1:
            ap.error("--max-calls N (N >= 1) is required for a real run")
        missing = [v for v in ("AR_LOCK_DIR", "AR_BREAKER_FILE") if not os.environ.get(v)]
        if missing:
            ap.error("set " + " and ".join(missing) + " so every agent shares one concurrency cap")
    check_prompt_hashes()
    items, counts = select_items()
    if args.items == "day0":
        items = day0_items(items)
    print("items:", json.dumps({**counts, "this_run": len(items)}), flush=True)
    cli_version = "" if args.dry_run else subprocess.run(["claude", "--version"], capture_output=True, text=True).stdout.strip()
    if not args.dry_run:
        install_capture()
        enter_neutral_cwd()
    summary = run_sweep(items, levels, args.run_dir, args.max_calls if args.max_calls is not None else len(items) * len(levels),
                        workers=args.workers, cli_version=cli_version, dry_run=args.dry_run)
    print("summary:", json.dumps(summary), flush=True)
    return summary


if __name__ == "__main__":
    main()
