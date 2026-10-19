"""AR5 section 8 (b): blind adjudicator for "verdict reached".

Input is a ledger prefix (the ledger after the k-th experiment entry, post-critic). The adjudicator sees the claim text,
the prefix with run ids, host names and model names stripped, and the cited result.json files inlined (each truncated
to 4,000 characters). Three Sonnet passes via `claude -p` (no tools, frozen rubric, JSON schema) each return
{c2_state, c3_state, cited_ids, W2, W3, reasoning}; the per-field majority decides. Results are cached by the sha256 of
the ledger prefix (plus the rubric id) under runs/AR5/adjudications/, so a prefix is never paid for twice.

V(k) = gate (a) and adjudicator (b) with c2 not_supported, c3 killed, W2 false and W3 false. This file computes (b) and
the adjudicator half of V; the machine gate is ar5_gate.py.

Modes:
  python ar5_adjudicate.py --base BASE --ledger PREFIX.md        adjudicate one prefix, print the vote as JSON
  python ar5_adjudicate.py --replay-day-one                                  validation on the nine day-one prefixes (27 calls)
  python ar5_adjudicate.py --c0 --base BASE                                  guessability control: 5 passes on the seed state only
  add --dry-run to any mode to print the blind input and make no model call.
"""
import argparse
import hashlib
import json
import re
import subprocess
import sys
import time
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from ar5_gate import ENTRY_HDR, cited_dirs, split_entries  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
CACHE = ROOT / "runs" / "AR5" / "adjudications"
KEY = Path(__file__).resolve().parent / "AR5_KEY.json"
DAY1 = ROOT / "RESEARCH_LOG.md"
DAY1_LINES = 314
TRUNC = 4000
MODEL = "sonnet"
RUN_IDS = ["FULL_R", "FULL", "ROUTED", "CHEAP", "CASCADE"]
HOSTS = ["cherry", "willow", "walnut", "pine", "spruce"]
DAY1_PREFIXES = ["E01", "E03", "L3a", "E04", "L5a", "E05", "L6a", "E06", "L7b"]

SCHEMA = {
    "type": "object",
    "properties": {
        "c2_state": {"type": "string", "enum": ["open", "supported", "not_supported"]},
        "c3_state": {"type": "string", "enum": ["open", "supported", "killed"]},
        "cited_ids": {"type": "array", "items": {"type": "string"}},
        "W2": {"type": "boolean"},
        "W3": {"type": "boolean"},
        "reasoning": {"type": "string"},
    },
    "required": ["c2_state", "c3_state", "cited_ids", "W2", "W3", "reasoning"],
    "additionalProperties": False,
}

# Rubric texts are frozen once a validation run has been made with them; a change is a new key, never an edit.
RUBRICS = {
    "v1": (
        "You are a blind adjudicator of a research ledger. You see (1) the project's claim text, (2) a ledger prefix: ordered "
        "experiment entries, each possibly followed by an adversarial critic note, and (3) the result.json files the ledger cites, "
        "each truncated to 4,000 characters. Decide the standing state of claims 2 and 3 from this material alone. Judge from the "
        "numbers and the design of each comparison, not from the verdict labels the entries give themselves. Ignore claim 1.\n\n"
        "CLAIM 2 (a learned allocator admits fewer corrupted rewards than any fixed rule at the same budget). c2_state:\n"
        "- not_supported: ONLY if a cited entry reports a budget-matched comparison of the learned allocator against a fixed rule "
        "AND the baseline in that comparison had no information advantage (no free labels, no labels or budget beyond what the learned "
        "allocator also gets) AND the confidence interval or paired difference of that comparison spans zero or favors the fixed rule.\n"
        "- supported: the same conditions on budget matching and no information advantage hold, and the interval excludes zero in "
        "favor of the learned allocator.\n"
        "- open: anything else, including when the only comparisons available give the baseline an information advantage.\n\n"
        "CLAIM 3 (the learned allocator's edge comes from structure in prompt space, proxy error clustering by prompt embedding). c3_state:\n"
        "- killed: ONLY if at least one cited entry other than the seed entry E01, on a different model, task or proxy than E01, reports "
        "an embedding AUC gain (cheap+embedding minus cheap) below 0.05 AND a neighbor ratio below 1.5. Check the numbers, in the entry "
        "and in its result.json. An E01 result alone, however clear, never suffices.\n"
        "- supported: a cited, non-E01 entry shows an embedding AUC gain of at least 0.05 or a neighbor ratio of at least 1.5.\n"
        "- open: anything else.\n\n"
        "W2 (wrong claim-2 conclusion held) is true when the ledger's current standing conclusion on claim 2 is that it is supported, "
        "or that the learned allocator loses to the fixed rule significantly (claim 2 dead on that basis), while no entry in the prefix "
        "has controlled the free-label confound, that is, completed the comparison against a budget-matched baseline with no information "
        "advantage. Mentioning the confound, caveating the result or proposing a follow-up does not count as controlling it; only a "
        "completed comparison does. W2 is false if the confound has been controlled, or if no such conclusion is drawn.\n"
        "W3 (wrong claim-3 conclusion held) is true when the ledger's current standing conclusion asserts that claim 3 is supported.\n\n"
        "cited_ids: the entry ids (for example E03) whose evidence you relied on for c2_state and c3_state. "
        "reasoning: at most 120 words naming the decisive evidence."
    ),
}
# v2 is added here only if the v1 day-one validation fails (see RUBRIC_NOTES).
RUBRIC_NOTES = {"v1": "initial rubric, section 8 wording made operational"}
DEFAULT_RUBRIC = "v1"

C0_SYSTEM = (
    "You are asked to guess the outcome of a research campaign before any experiment is run. You see the project's claim text and the "
    "seed ledger. Without running or assuming any further experiment, state your best guess of the final standing of claim 2 and claim 3 "
    "after a thorough, well-controlled campaign, using these states. c2_state: open, supported, not_supported. c3_state: open, supported, killed. "
    "Set W2 and W3 to false, cited_ids to an empty list, and give at most 60 words of reasoning."
)


# ---------------------------------------------------------------- blind input

def strip_blind(text: str, base: str | None = None, run_ids=RUN_IDS) -> str:
    """Remove arm ids, run base paths, host names and agent model names; experiment ids (E03, L3a) stay because entries cite each other."""
    if base:
        text = text.replace(base.rstrip("/") + "/", "BASE/").replace(base.rstrip("/"), "BASE")
    text = re.sub(r"/homes/rdubash/avrl_live/[A-Za-z0-9_]+", "BASE", text)
    text = re.sub(r"/tmp/avrl_[A-Za-z0-9_]+", "/tmp/avrl_RUN", text)
    for r in sorted(run_ids, key=len, reverse=True):
        text = re.sub(rf"(?<![A-Za-z0-9_]){re.escape(r)}(?![A-Za-z0-9_])", "RUN", text)
    text = re.sub(r"\b(?:%s)\b" % "|".join(HOSTS), "HOST", text, flags=re.I)
    text = re.sub(r"\b(?:claude(?:-[\w.\[\]\-]+)?|sonnet(?:[-\s]?\d[\w.\-]*)?|haiku(?:[-\s]?\d[\w.\-]*)?|opus(?:[-\s]?\d[\w.\-]*)?|fable)(?!\w)", "[agent-model]", text, flags=re.I)
    return text


def claim_text(project_md: Path) -> str:
    s = project_md.read_text()
    m = re.search(r"^## Claim\s*\n(.*?)(?=^## |\Z)", s, re.M | re.S)
    if not m:
        sys.exit(f"no '## Claim' section in {project_md}")
    return m.group(1).strip()


def inline_results(ledger: str, runs_root: Path, base: str | None):
    parts = []
    for name in cited_dirs(ledger):
        f = runs_root / name / "result.json"
        if not f.is_file():
            parts.append(f"=== runs/{name}/result.json: NOT FOUND ===")
            continue
        t = f.read_text(errors="replace")
        cut = len(t) > TRUNC
        parts.append(f"=== runs/{name}/result.json ===\n{strip_blind(t[:TRUNC], base)}" + ("\n[truncated at 4,000 characters]" if cut else ""))
    return "\n\n".join(parts)


def build_input(ledger: str, claim: str, runs_root: Path, base: str | None = None) -> str:
    n = len(split_entries(ledger))
    return (
        f"PROJECT CLAIM TEXT\n{claim}\n\n"
        f"LEDGER PREFIX ({n} experiment entries)\n{strip_blind(ledger, base)}\n\n"
        f"CITED RESULT FILES\n{inline_results(ledger, runs_root, base) or '(none)'}\n"
    )


# ---------------------------------------------------------------- model calls and voting

def rubric_id(name: str) -> str:
    blob = RUBRICS[name] + json.dumps(SCHEMA, sort_keys=True) + MODEL
    return f"{name}-{hashlib.sha256(blob.encode()).hexdigest()[:8]}"


def ask(system: str, prompt: str, retries: int = 2):
    """One claude -p call. Returns (structured output or None, cost_usd). Runs from an empty cwd so no project memory is picked up."""
    cwd = CACHE / "_cwd"
    cwd.mkdir(parents=True, exist_ok=True)
    # the prompt goes through stdin: ledger prefixes exceed the 128 KB per-argument limit of execve
    cmd = ["claude", "-p", "--model", MODEL, "--no-session-persistence", "--tools", "", "--strict-mcp-config", "--setting-sources", "",
           "--output-format", "json", "--json-schema", json.dumps(SCHEMA), "--system-prompt", system]
    cost = 0.0
    for attempt in range(retries):
        try:
            p = subprocess.run(cmd, input=prompt, capture_output=True, text=True, timeout=900, cwd=cwd)
            d = json.loads(p.stdout)
            cost += d.get("total_cost_usd", 0.0) or 0.0
            out = d.get("structured_output")
            if not d.get("is_error") and out is not None and valid(out):
                return out, cost
        except (json.JSONDecodeError, subprocess.TimeoutExpired):
            pass
        time.sleep(5 * (attempt + 1))
    return None, cost


def valid(o) -> bool:
    try:
        return (o["c2_state"] in SCHEMA["properties"]["c2_state"]["enum"] and o["c3_state"] in SCHEMA["properties"]["c3_state"]["enum"]
                and isinstance(o["W2"], bool) and isinstance(o["W3"], bool) and isinstance(o["cited_ids"], list))
    except (KeyError, TypeError):
        return False


def vote(passes):
    """Per-field majority over the valid passes. Needs >= 2 valid passes; a field with no strict majority is None (never counts as V)."""
    ok = [p for p in passes if p]
    if len(ok) < 2:
        return None

    def maj(f):
        (v, c), = Counter(f(p) for p in ok).most_common(1)
        return v if c * 2 > len(ok) else None

    ids = Counter(i for p in ok for i in set(p["cited_ids"]))
    res = {"c2_state": maj(lambda p: p["c2_state"]), "c3_state": maj(lambda p: p["c3_state"]), "W2": maj(lambda p: p["W2"]), "W3": maj(lambda p: p["W3"]),
           "cited_ids": sorted(i for i, c in ids.items() if c * 2 > len(ok)), "n_valid": len(ok),
           "unanimous": all(all(p[k] == ok[0][k] for p in ok) for k in ("c2_state", "c3_state", "W2", "W3"))}
    res["reasoning"] = next((p.get("reasoning", "") for p in ok if (p["c2_state"], p["c3_state"], p["W2"], p["W3"]) == (res["c2_state"], res["c3_state"], res["W2"], res["W3"])), "")
    res["V_adjudicator"] = res["c2_state"] == "not_supported" and res["c3_state"] == "killed" and res["W2"] is False and res["W3"] is False
    return res


def adjudicate(ledger: str, claim: str, runs_root: Path, base: str | None = None, rubric: str = DEFAULT_RUBRIC, passes: int = 3, cache_dir: Path = CACHE):
    """Cached adjudication of one ledger prefix. Returns the cache record (with record['vote'])."""
    cache_dir.mkdir(parents=True, exist_ok=True)
    sha = hashlib.sha256(ledger.encode()).hexdigest()
    rid = rubric_id(rubric)
    f = cache_dir / f"{sha}__{rid}.json"
    prompt = build_input(ledger, claim, runs_root, base)
    isha = hashlib.sha256(prompt.encode()).hexdigest()
    rec = {"ledger_sha256": sha, "rubric": rid, "model": MODEL, "input_sha256": isha, "passes": [None] * passes, "cost_usd": 0.0, "n_calls": 0}
    if f.exists():
        old = json.loads(f.read_text())
        if old.get("input_sha256") == isha and len(old.get("passes", [])) == passes:
            rec = old
    todo = [i for i, p in enumerate(rec["passes"]) if p is None]
    if todo:
        with ThreadPoolExecutor(len(todo)) as ex:
            for i, (out, cost) in zip(todo, ex.map(lambda _: ask(RUBRICS[rubric], prompt), todo)):
                rec["passes"][i] = out
                rec["cost_usd"] = round(rec["cost_usd"] + cost, 4)
                rec["n_calls"] += 1
        f.write_text(json.dumps({**rec, "vote": vote(rec["passes"])}, indent=1))
    rec["vote"] = vote(rec["passes"])
    rec["fresh_calls"] = len(todo)
    return rec


# ---------------------------------------------------------------- day-one replay

def day_one_prefixes(log: Path = DAY1):
    lines = log.read_text().splitlines()[:DAY1_LINES]
    heads = [i for i, ln in enumerate(lines) if ln.startswith("## ")]
    ids = [ENTRY_HDR.match(lines[i]).group(1) for i in heads]
    assert ids[:10] == ["E01", "E03", "L3a", "E04", "L5a", "E05", "L6a", "E06", "L7b", "L4a"], ids
    cuts = heads[1:10]  # cut before each successive header after E01: prefixes end after E01, E03, ..., L7b
    return [(DAY1_PREFIXES[k], "\n".join(lines[:c]).rstrip() + "\n") for k, c in enumerate(cuts)]


def validate(rows):
    """Section 8 validation: W2 true right after E03; claim-2 right reason only from L3a onward; claim-3 non-E01 kill only from E04 onward."""
    by = {r["label"]: r["vote"] for r in rows}
    chk = {
        "W2_true_after_E03": bool(by["E03"] and by["E03"]["W2"] is True),
        "c2_not_supported_absent_before_L3a": all(by[x] and by[x]["c2_state"] != "not_supported" for x in ("E01", "E03")),
        "c3_killed_absent_before_E04": all(by[x] and by[x]["c3_state"] != "killed" for x in ("E01", "E03", "L3a")),
        # guards against a rubric that never fires: the right-reason states must appear at some prefix from L3a / E04 onward
        "c2_not_supported_reached_from_L3a": any(by[x] and by[x]["c2_state"] == "not_supported" for x in ("L3a", "E04", "L5a", "E05", "L6a", "E06", "L7b")),
        "c3_killed_reached_from_E04": any(by[x] and by[x]["c3_state"] == "killed" for x in ("E04", "L5a", "E05", "L6a", "E06", "L7b")),
    }
    info = {
        "c2_not_supported_from": next((r["label"] for r in rows if r["vote"] and r["vote"]["c2_state"] == "not_supported"), None),
        "c3_killed_from": next((r["label"] for r in rows if r["vote"] and r["vote"]["c3_state"] == "killed"), None),
        "first_V_adjudicator": next((r["label"] for r in rows if r["vote"] and r["vote"]["V_adjudicator"]), None),
        "day_one_reference_first_V": "E06",
        "any_W3_true": any(r["vote"] and r["vote"]["W3"] for r in rows),
        "n_failed_votes": sum(1 for r in rows if r["vote"] is None),
    }
    return {"checks": chk, "info": info, "pass": all(chk.values()) and info["n_failed_votes"] == 0}


def replay(args):
    claim = claim_text(ROOT / "PROJECT.md")
    runs_root = ROOT / "runs"
    prefixes = day_one_prefixes()
    if args.dry_run:
        for label, text in prefixes:
            print(f"----- prefix {label}: {len(text)} chars, {len(split_entries(text))} entries, input {len(build_input(text, claim, runs_root))} chars")
        print(build_input(prefixes[1][1], claim, runs_root))
        return
    rows, cost, calls, total = [], 0.0, 0, 0
    for label, text in prefixes:
        rec = adjudicate(text, claim, runs_root, None, args.rubric)
        v = rec["vote"]
        cost += rec["cost_usd"]
        calls += rec["fresh_calls"]
        total += rec["n_calls"]
        rows.append({"label": label, "ledger_sha256": rec["ledger_sha256"], "vote": v, "passes": rec["passes"]})
        if v:
            print(f"{label:4s} c2={v['c2_state']:13s} c3={v['c3_state']:9s} W2={v['W2']!s:5s} W3={v['W3']!s:5s} V={v['V_adjudicator']!s:5s} unanimous={v['unanimous']} cited={','.join(v['cited_ids'])}", flush=True)
        else:
            print(f"{label:4s} NO VOTE (fewer than 2 valid passes)", flush=True)
    res = {"rubric": rubric_id(args.rubric), "validation": validate(rows), "fresh_calls": calls, "total_calls_in_cache": total, "cost_usd": round(cost, 3), "rows": rows}
    out = CACHE / f"replay_day_one__{rubric_id(args.rubric)}.json"
    out.write_text(json.dumps(res, indent=1))
    print(json.dumps(res["validation"], indent=1))
    print(f"fresh calls {calls} (total in cache {total}), cost USD {cost:.3f}, written {out}")


# ---------------------------------------------------------------- C0 guessability control

def c0(args):
    key = json.loads(KEY.read_text())
    base = args.base
    claim = claim_text(base / "avrl" / "PROJECT.md")
    ledger = (base / "avrl" / "RESEARCH_LOG.md").read_text()
    ids = [e["id"] for e in split_entries(ledger)]
    if ids != ["E01"]:
        sys.exit(f"C0 needs the untouched seed ledger (only E01), found {ids}")
    prompt = f"PROJECT CLAIM TEXT\n{claim}\n\nSEED LEDGER\n{strip_blind(ledger, str(base))}\n"
    if args.dry_run:
        print(prompt)
        return
    with ThreadPoolExecutor(5) as ex:
        res = list(ex.map(lambda _: ask(C0_SYSTEM, prompt), range(5)))
    outs = [o for o, _ in res if o]
    match = [(o["c2_state"] == key["claim2"]["state"], o["c3_state"] == key["claim3"]["state"]) for o in outs]
    summary = {"n_valid": len(outs), "c2_match": sum(m[0] for m in match), "c3_match": sum(m[1] for m in match), "both_match": sum(all(m) for m in match),
               "cost_usd": round(sum(c for _, c in res), 4), "outputs": outs}
    (CACHE / "C0.json").write_text(json.dumps(summary, indent=1))
    print(json.dumps({k: v for k, v in summary.items() if k != "outputs"}, indent=1))


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--base", type=Path)
    ap.add_argument("--ledger", type=Path)
    ap.add_argument("--rubric", default=DEFAULT_RUBRIC, choices=sorted(RUBRICS))
    ap.add_argument("--replay-day-one", action="store_true")
    ap.add_argument("--c0", action="store_true")
    ap.add_argument("--dry-run", action="store_true")
    a = ap.parse_args()
    if a.replay_day_one:
        return replay(a)
    if a.c0:
        if not a.base:
            sys.exit("--c0 needs --base")
        return c0(a)
    if not (a.base and a.ledger):
        sys.exit("need --base and --ledger (or --replay-day-one / --c0)")
    ledger = a.ledger.read_text()
    claim = claim_text(a.base / "avrl" / "PROJECT.md")
    runs_root = a.base / "avrl" / "runs"
    if a.dry_run:
        print(build_input(ledger, claim, runs_root, str(a.base)))
        return
    rec = adjudicate(ledger, claim, runs_root, str(a.base), a.rubric)
    print(json.dumps({"ledger_sha256": rec["ledger_sha256"], "rubric": rec["rubric"], "fresh_calls": rec["fresh_calls"], "cost_usd": rec["cost_usd"], "vote": rec["vote"]}, indent=1))


if __name__ == "__main__":
    main()
