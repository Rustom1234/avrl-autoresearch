"""AR3 runner: collect critic passes (Haiku x5, Sonnet x3 per item) with `claude -p`.

Implements AR3_DESIGN.md section 5 with the frozen prompt and schema of section 4.
Reads runs/AR3/items.jsonl and runs/AR3/split.json (written by ar3_build_substrate.py).
Writes one JSONL line per call to runs/AR3/calls_{burn,test,probe}.jsonl (smoke: calls_smoke.jsonl).

Schedule: burn entries first (all their items), then test, then probe. Within an item the
pass order is H1 H2 H3 S1 S2 H4 H5 S3, so Haiku and Sonnet are interleaved and alias drift
is time-balanced. The runner is resumable: a (item, model, pass) key with a valid record is
never called again; failed records are retried on the next run.

Stops: total cost >= --budget (default $80, in-flight calls may overshoot by a few cents) and
failure rate above 5 percent of finished calls after retries (checked once >= 40 calls are done).
Progress lines print counts and cost only, never per-pass outcomes.

Smoke mode takes items from ar3_build_substrate.py (lazy import). Expected API of that module:
  render_entry(...)                    renders one stripped section as "L%03d| " lines (used internally)
  build_items_for_entries(header_ids)  list[dict]; one control item per ledger section whose header id is
                                       in header_ids, no pool rule applied; each dict has item_id, entry_key,
                                       header_id, text (and the other items.jsonl fields if convenient)
Use --smoke-items FILE.jsonl to bypass the import.

Self-test (no real calls): --dry-run fabricates outputs. Run it with --dir pointing away from runs/AR3.
"""
import argparse
import hashlib
import json
import random
import re
import subprocess
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "runs" / "AR3"

# ---- frozen critic task (AR3_DESIGN.md section 4) ----
SYSTEM = (
    "You review one entry from a research ledger, using only the entry itself. Each line is prefixed with its line number like L012|. "
    "Find every claim that the entry's own table, numbers or stated pass/kill thresholds do not support, including a stated verdict that "
    "contradicts the entry's own numbers, thresholds or header. Many entries are fully consistent; for those return an empty problems list. "
    "Do not flag style, rounding, terminology, forward-looking estimates, or claims about files, runs or code you cannot see. "
    "For each problem give the line number where the inconsistency is stated, a kind, the exact quote, and a one-sentence explanation. "
    "Also give judgment_confidence, 0 to 100, your confidence that your overall judgment (problems found, or none) is correct."
)
SCHEMA_STR = (
    '{"type":"object","properties":{"problems":{"type":"array","maxItems":5,"items":{"type":"object","properties":'
    '{"line":{"type":"integer"},"kind":{"type":"string","enum":["number","verdict","overclaim","other"]},'
    '"quote":{"type":"string"},"explanation":{"type":"string"}},"required":["line","kind","quote","explanation"]}},'
    '"judgment_confidence":{"type":"integer","minimum":0,"maximum":100}},"required":["problems","judgment_confidence"]}'
)
json.loads(SCHEMA_STR)  # fail at import if the constant is malformed
SYSTEM_SHA = hashlib.sha256(SYSTEM.encode()).hexdigest()
SCHEMA_SHA = hashlib.sha256(SCHEMA_STR.encode()).hexdigest()

# ---- schedule and limits (AR3_DESIGN.md section 5) ----
PASS_ORDER = [("haiku", 1), ("haiku", 2), ("haiku", 3), ("sonnet", 1), ("sonnet", 2), ("haiku", 4), ("haiku", 5), ("sonnet", 3)]
SPLIT_ORDER = ["burn", "test", "probe"]
MIN_VALID = {"haiku": 3, "sonnet": 2}
TIMEOUT_S = 300
FAIL_FRAC_ABORT = 0.05
MIN_CALLS_FOR_ABORT = 40
LOW_WORKERS = 6
SMOKE_HEADER_IDS = ["M01", "P1-judge-lr5-s2", "P1-judge-5090b-s0", "M06b"]  # the excluded entries (design 3.3)
OVERLOAD = re.compile(r"\b(429|529)\b|overload|rate.?limit", re.I)
BACKOFF_BASE = 5.0


def plabel(alias, idx):
    return f"{alias[0].upper()}{idx}"


def parse_passes(spec):
    want = []
    for tok in spec.split(","):
        tok = tok.strip().upper()
        pair = {"H": "haiku", "S": "sonnet"}.get(tok[:1]), int(tok[1:]) if tok[1:].isdigit() else 0
        if pair not in PASS_ORDER:
            raise SystemExit(f"bad pass {tok!r}")
        want.append(pair)
    return [p for p in PASS_ORDER if p in want]  # keep the frozen order


def cmd_for(alias, prompt):
    return ["claude", "-p", "--model", alias, "--no-session-persistence", "--tools", "", "--strict-mcp-config", "--setting-sources", "",
            "--output-format", "json", "--json-schema", SCHEMA_STR, "--system-prompt", SYSTEM, prompt]


def fake_call(alias, prompt, key, attempt, args):
    """Dry-run stand-in for `claude -p ... --output-format json`, same top-level shape."""
    h = int(hashlib.sha1(f"{key}|{attempt}".encode()).hexdigest(), 16)
    if (h % 10_000) / 10_000 < args.dry_fail_rate:
        return {"is_error": True, "result": "API Error: 429 overloaded, rate limit", "total_cost_usd": 0.0}
    rng = random.Random(int(hashlib.sha1(("o|" + str(key)).encode()).hexdigest(), 16) % 2**32)
    lines = prompt.splitlines()
    problems = []
    for _ in range(rng.choice([0, 0, 1, 1, 2])):
        i = rng.randrange(len(lines))
        problems.append({"line": i + 1, "kind": rng.choice(["number", "verdict", "overclaim", "other"]),
                         "quote": re.sub(r"^L\d+\| ?", "", lines[i])[:60], "explanation": "fabricated dry-run problem."})
    time.sleep(rng.random() * 0.02)
    cost = (0.012 if alias == "haiku" else 0.055) * (0.8 + 0.4 * rng.random())
    mid = {"haiku": "claude-haiku-fake", "sonnet": "claude-sonnet-fake"}[alias]
    return {"type": "result", "is_error": False, "duration_ms": int(1000 + 4000 * rng.random()), "total_cost_usd": cost,
            "structured_output": {"problems": problems, "judgment_confidence": rng.randint(40, 99)},
            "usage": {"input_tokens": 100 + len(prompt) // 4, "output_tokens": 60 + 80 * len(problems),
                      "cache_read_input_tokens": 0, "cache_creation_input_tokens": 0},
            "modelUsage": {mid: {"costUSD": cost}}}


def real_call(alias, prompt):
    """Returns (parsed json dict or None, error text)."""
    try:
        p = subprocess.run(cmd_for(alias, prompt), capture_output=True, text=True, timeout=TIMEOUT_S)
    except subprocess.TimeoutExpired:
        return None, "timeout"
    except (OSError, UnicodeDecodeError) as e:
        return None, f"{type(e).__name__}: {e}"
    try:
        d = json.loads(p.stdout)
        if isinstance(d, dict):
            return d, ""
        return None, "stdout json is not an object: " + p.stdout[-300:]
    except json.JSONDecodeError:
        return None, ("rc=%s stdout=%s stderr=%s" % (p.returncode, p.stdout[-300:], p.stderr[-300:]))


def valid(d):
    so = d.get("structured_output") if isinstance(d, dict) else None
    return isinstance(d, dict) and not d.get("is_error") and isinstance(so, dict) and isinstance(so.get("problems"), list)


class Limiter:
    """Adjustable concurrency cap (the thread pool stays at --workers; this gates actual calls)."""

    def __init__(self, n):
        self.limit, self.active, self.cv = n, 0, threading.Condition()

    def __enter__(self):
        with self.cv:
            while self.active >= self.limit:
                self.cv.wait()
            self.active += 1

    def __exit__(self, *exc):
        with self.cv:
            self.active -= 1
            self.cv.notify_all()

    def reduce(self, n):
        with self.cv:
            self.limit = min(self.limit, n)


class State:
    def __init__(self, workers, prior_cost, budget):
        self.lock = threading.Lock()
        self.limiter = Limiter(workers)
        self.workers, self.budget = workers, budget
        self.prior_cost, self.new_cost = prior_cost, 0.0
        self.done = self.nfail = 0
        self.stop = None
        self.overloads = []
        self.reduced = False

    def total(self):
        return self.prior_cost + self.new_cost

    def halt(self, why):
        with self.lock:
            if self.stop is None:
                self.stop = why
                print(f"STOP: {why}", flush=True)

    def note_overload(self):
        now = time.time()
        with self.lock:
            self.overloads = [t for t in self.overloads if now - t < 120] + [now]
            trip = len(self.overloads) >= 3 and not self.reduced and self.workers > LOW_WORKERS
            if trip:
                self.reduced = True
        if trip:
            self.limiter.reduce(LOW_WORKERS)
            print(f"repeated 429/overload: concurrency limited to {LOW_WORKERS}", flush=True)


def do_call(alias, prompt, key, args, st):
    """One scheduled call with retries. Returns the record fields (no ids)."""
    t_start, retry_cost, err, d = time.time(), 0.0, "", None
    for attempt in range(args.retries + 1):
        t0 = time.time()
        if args.dry_run:
            d, err = fake_call(alias, prompt, key, attempt, args), ""
        else:
            d, err = real_call(alias, prompt)
        wall = int((time.time() - t0) * 1000)
        if valid(d):
            break
        retry_cost += float((d or {}).get("total_cost_usd") or 0.0)
        err = err or str((d or {}).get("result") or "")[:300]
        if OVERLOAD.search(err):
            st.note_overload()
        if attempt < args.retries:
            time.sleep(min(120.0, BACKOFF_BASE * 2 ** attempt) * (0.5 + random.random()) * (0.002 if args.dry_run else 1.0))
    ok = valid(d)  # the successful attempt's cost is total_cost_usd; failed attempts' cost is retry_cost_usd
    u = (d or {}).get("usage") or {}
    return {
        "ok": ok,
        "structured_output": d.get("structured_output") if ok else None,
        "model_ids": sorted((d or {}).get("modelUsage") or {}),
        "duration_ms": (d or {}).get("duration_ms", wall),
        "total_cost_usd": float((d or {}).get("total_cost_usd") or 0.0) if ok else 0.0,
        "usage": {"input": u.get("input_tokens"), "output": u.get("output_tokens"),
                  "cache_read": u.get("cache_read_input_tokens"), "cache_write": u.get("cache_creation_input_tokens")},
        "ts": time.time(),
        "wall_ms": int((time.time() - t_start) * 1000),
        "attempts": attempt + 1,
        "retry_cost_usd": retry_cost,
        "error": None if ok else err[:300],
    }


def load_calls(path):
    out = []
    if path.exists():
        for ln in path.read_text().splitlines():
            if ln.strip():
                try:
                    out.append(json.loads(ln))
                except json.JSONDecodeError:  # truncated line from a killed run: that call is simply redone
                    print(f"warning: skipping unparseable line in {path.name}", flush=True)
    return out


def run_tasks(tasks, calls_path_of, args, prior_cost):
    """tasks: list of (split, item, alias, pass_idx). Returns the State."""
    st = State(args.workers, prior_cost, args.budget)
    files, flock = {}, threading.Lock()
    new_calls = [0]

    def work(t):
        split, item, alias, pidx = t
        if st.stop:
            return
        if st.total() >= st.budget:
            st.halt(f"hard stop, total cost ${st.total():.2f} reached budget ${st.budget:.2f}")
            return
        with st.limiter:
            if st.stop:
                return
            with st.lock:
                if args.max_calls and new_calls[0] >= args.max_calls:
                    return
                new_calls[0] += 1
            r = do_call(alias, item["text"], (item["item_id"], alias, pidx), args, st)
        rec = {"item_id": item["item_id"], "split": split, "model_alias": alias, "pass_idx": pidx, **r,
               "system_sha256": SYSTEM_SHA, "schema_sha256": SCHEMA_SHA}
        if args.dry_run:
            rec["dry_run"] = True
        with flock:
            if split not in files:
                fp = calls_path_of(split)
                glue = fp.exists() and fp.stat().st_size > 0 and fp.read_bytes()[-1:] != b"\n"
                files[split] = fp.open("a")
                if glue:  # previous run died mid-line: do not append onto the partial record
                    files[split].write("\n")
            files[split].write(json.dumps(rec) + "\n")
            files[split].flush()
        with st.lock:
            st.done += 1
            st.nfail += 0 if rec["ok"] else 1
            st.new_cost += rec["total_cost_usd"] + rec["retry_cost_usd"]
            done, nfail = st.done, st.nfail
            tally = done % 50 == 0
        if tally:
            print(f"calls {done} ok {done - nfail} fail {nfail} | new cost ${st.new_cost:.2f} | total ${st.total():.2f} | tasks {len(tasks)}", flush=True)
        if done >= MIN_CALLS_FOR_ABORT and nfail / done > FAIL_FRAC_ABORT:
            st.halt(f"abort, {nfail}/{done} calls failed after retries (> {FAIL_FRAC_ABORT:.0%})")

    with ThreadPoolExecutor(args.workers) as ex:
        list(ex.map(work, tasks))
    for f in files.values():
        f.close()
    print(f"finished: calls {st.done} ok {st.done - st.nfail} fail {st.nfail} | new cost ${st.new_cost:.2f} | total ${st.total():.2f}"
          + (f" | stopped: {st.stop}" if st.stop else ""), flush=True)
    return st


def build_tasks(items, split_of, splits, passes, valid_keys):
    """Burn entries first, then test, then probe; entries in file order; passes in the frozen order."""
    by_split = {s: {} for s in splits}
    for it in items:
        s = split_of(it)
        if s in by_split:
            by_split[s].setdefault(it["entry_key"], []).append(it)
    tasks = []
    for s in SPLIT_ORDER:
        for ek, its in by_split.get(s, {}).items():
            for it in its:
                for alias, pidx in passes:
                    if (it["item_id"], alias, pidx) not in valid_keys:
                        tasks.append((s, it, alias, pidx))
    return tasks


def write_invalid(out_dir, items, split_of):
    """Items with fewer than 3 valid Haiku or 2 valid Sonnet passes, among items whose 8 scheduled passes were all attempted.
    Always covers all splits and the full schedule, so a run restricted by --splits or --passes never hides or overwrites the rest."""
    passes, attempted, valid_n = PASS_ORDER, set(), {}
    for s in SPLIT_ORDER:
        for r in load_calls(out_dir / f"calls_{s}.jsonl"):
            attempted.add((r["item_id"], r["model_alias"], r["pass_idx"]))
            if r["ok"]:
                valid_n.setdefault(r["item_id"], set()).add((r["model_alias"], r["pass_idx"]))
    invalid, counts, incomplete = [], {}, []
    for it in items:
        if split_of(it) not in SPLIT_ORDER:
            continue
        iid = it["item_id"]
        if any((iid, a, p) not in attempted for a, p in passes):
            incomplete.append(iid)
            continue
        v = valid_n.get(iid, set())
        n = {a: sum(1 for x in v if x[0] == a) for a in MIN_VALID}
        counts[iid] = n
        if any(n[a] < MIN_VALID[a] for a in MIN_VALID):
            invalid.append(iid)
    (out_dir / "invalid_items.json").write_text(json.dumps({"invalid": invalid, "counts": counts, "incomplete": incomplete}, indent=1))
    print(f"invalid items {len(invalid)}, incomplete (not fully attempted) {len(incomplete)}, written to {out_dir / 'invalid_items.json'}", flush=True)


def load_smoke_items(path):
    if path:
        rows = [json.loads(ln) for ln in Path(path).read_text().splitlines() if ln.strip()]
    else:
        try:
            import ar3_build_substrate as sub
            rows = sub.build_items_for_entries(SMOKE_HEADER_IDS)
        except Exception as e:  # module not ready or API differs
            raise SystemExit(f"smoke needs ar3_build_substrate.build_items_for_entries(header_ids) -> list[dict] ({type(e).__name__}: {e}). "
                             "Pass --smoke-items FILE.jsonl to bypass.")
    for r in rows:
        r.setdefault("item_id", hashlib.sha1(r["text"].splitlines()[0].encode()).hexdigest()[:12])
        r.setdefault("entry_key", r["item_id"])
    return rows


def smoke_report(path, ids):
    recs = [r for r in load_calls(path) if r["item_id"] in ids]
    for m in ("haiku", "sonnet"):
        rm = [r for r in recs if r["model_alias"] == m]
        if not rm:
            continue
        okm = [r for r in rm if r["ok"]]
        print(f"smoke {m}: parsed {len(okm)}/{len(rm)}"
              f" | mean cost/call ${sum(r['total_cost_usd'] for r in okm) / max(1, len(okm)):.4f}"
              f" | mean latency {sum(r['duration_ms'] for r in okm) / max(1, len(okm)) / 1000:.1f}s"
              f" | model ids {sorted({i for r in okm for i in r['model_ids']})}", flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir", default=str(OUT), help="AR3 run dir holding items.jsonl, split.json and the calls files")
    ap.add_argument("--splits", default="burn,test,probe")
    ap.add_argument("--passes", default=",".join(plabel(*p) for p in PASS_ORDER))
    ap.add_argument("--workers", type=int, default=12)
    ap.add_argument("--budget", type=float, default=80.0, help="hard stop on total cost in USD")
    ap.add_argument("--retries", type=int, default=4)
    ap.add_argument("--max-calls", type=int, default=0, help="cap on new calls this run (0 = none)")
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--smoke-items", default=None)
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--dry-fail-rate", type=float, default=0.0)
    ap.add_argument("--allow-hash-mismatch", action="store_true")
    args = ap.parse_args()

    out_dir = Path(args.dir)
    if args.dry_run:
        if out_dir.resolve() == OUT.resolve():
            raise SystemExit("--dry-run refuses to write fake calls into the real runs/AR3 dir; pass --dir")
    out_dir.mkdir(parents=True, exist_ok=True)
    print(f"system prompt sha256 {SYSTEM_SHA}\nschema sha256        {SCHEMA_SHA}", flush=True)
    (out_dir / "critic_hashes.json").write_text(json.dumps({"system_sha256": SYSTEM_SHA, "schema_sha256": SCHEMA_SHA}, indent=1))
    passes = parse_passes(args.passes)
    splits = [s.strip() for s in args.splits.split(",") if s.strip()]
    if not splits or any(s not in SPLIT_ORDER for s in splits):
        raise SystemExit(f"--splits must be a comma list from {SPLIT_ORDER}")

    prior = 0.0
    for p in out_dir.glob("calls_*.jsonl"):
        for r in load_calls(p):
            prior += (r.get("total_cost_usd") or 0.0) + (r.get("retry_cost_usd") or 0.0)
            if not args.allow_hash_mismatch and (r.get("system_sha256"), r.get("schema_sha256")) != (SYSTEM_SHA, SCHEMA_SHA):
                raise SystemExit(f"{p.name} holds records with a different prompt or schema hash; refusing (use --allow-hash-mismatch)")

    if args.smoke:
        items = load_smoke_items(args.smoke_items)
        smoke_passes = [("haiku", 1), ("sonnet", 1)]
        path = out_dir / "calls_smoke.jsonl"
        done = {(r["item_id"], r["model_alias"], r["pass_idx"]) for r in load_calls(path) if r["ok"]}
        tasks = [("smoke", it, a, p) for it in items for a, p in smoke_passes if (it["item_id"], a, p) not in done]
        print(f"smoke: {len(items)} items, {len(tasks)} calls to make", flush=True)
        run_tasks(tasks, lambda s: path, args, prior)
        smoke_report(path, {it["item_id"] for it in items})
        return

    items = [json.loads(ln) for ln in (out_dir / "items.jsonl").read_text().splitlines() if ln.strip()]
    split_json = json.loads((out_dir / "split.json").read_text())

    def split_of(it):
        return split_json.get(it["entry_key"], it.get("split"))

    bad = [it["item_id"] for it in items if split_of(it) not in SPLIT_ORDER]
    if bad:
        raise SystemExit(f"{len(bad)} items have no valid split in split.json or items.jsonl (first: {bad[:3]})")
    valid_keys = set()
    for s in splits:
        valid_keys |= {(r["item_id"], r["model_alias"], r["pass_idx"]) for r in load_calls(out_dir / f"calls_{s}.jsonl") if r["ok"]}
    tasks = build_tasks(items, split_of, splits, passes, valid_keys)
    per = {s: sum(1 for t in tasks if t[0] == s) for s in splits}
    print(f"{len(items)} items, passes {[plabel(*p) for p in passes]}, remaining calls {len(tasks)} {per}, prior cost ${prior:.2f}", flush=True)
    st = run_tasks(tasks, lambda s: out_dir / f"calls_{s}.jsonl", args, prior)
    write_invalid(out_dir, items, split_of)
    if st.stop:
        sys.exit(3)


if __name__ == "__main__":
    main()
