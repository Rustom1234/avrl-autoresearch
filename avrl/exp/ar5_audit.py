"""AR5 section 9: independent audit of every ledger entry by a Sonnet auditor (claude -p, no tools), blind to arm.

For each experiment entry in a run's ledger the auditor sees: the entry text with any Critic note split off into a
separate field, the cited runs/<id>/result.json files (truncated), the last 80 lines of run.log, and the experiment
script if present. Taxonomy (frozen): V verdict not implied by the entry's own thresholds and raw numbers; N a table
number differs from raw beyond rounding; T a narrative number disagrees with table or raw; U unsupported qualitative
claim or an unflagged crash, truncation, leakage or confound; F a cited artifact is absent. MATERIAL = V or N or F.

Modes:
  --run BASE --ledger LEDGER --out OUT.jsonl      audit every experiment entry of one run (final, post-critic text,
                                                  with the Critic note shown separately so the auditor judges the
                                                  analyst's text and reports whether the critic note fixed it)
  --twins BASE --ledger LEDGER --out OUT.jsonl    plant one defect (number or verdict, from ar2_critic_eval) into a
                                                  10 percent sample of entries and audit the twins (auditor recall)
  --seeded --out OUT.jsonl                        validation on the original day-one ledger: the six seeded defects
                                                  listed in AR5_KEY.json plus six clean day-one entries as controls
Each output line: {run, id, variant, flags:{V,N,T,U,F}, material, problems:[{kind, quote, explanation}],
                   critic_fixed (for run mode), cost_usd, model_id}.
"""
import argparse
import json
import random
import re
import subprocess
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

HDR = re.compile(r"^## (.+?) (2026-\d\d-\d\d)(.*)$")
SCHEMA = {
    "type": "object",
    "properties": {
        "flags": {"type": "object", "properties": {k: {"type": "boolean"} for k in "VNTUF"}, "required": list("VNTUF")},
        "problems": {"type": "array", "maxItems": 8, "items": {"type": "object", "properties": {
            "kind": {"type": "string", "enum": list("VNTUF")}, "quote": {"type": "string"}, "explanation": {"type": "string"}},
            "required": ["kind", "quote", "explanation"]}},
        "critic_note_fixes_material_errors": {"type": "string", "enum": ["yes", "partly", "no", "no_note"]},
    },
    "required": ["flags", "problems", "critic_note_fixes_material_errors"],
}
SYSTEM = (
    "You audit one entry of a research ledger written by an automated analyst. You are given the entry text (with any "
    "Critic note separated out), the raw result files the entry cites, the tail of the run log, and the experiment script "
    "if present. Classify defects in the ANALYST'S TEXT with this frozen taxonomy: V the verdict is not implied by the "
    "entry's own pre-registered thresholds and the raw numbers; N a table number differs from the raw files beyond "
    "rounding; T a narrative number disagrees with the table or the raw files; U an unsupported qualitative claim, or a "
    "crash, truncation, leakage or confound visible in the logs or script that the entry does not flag; F a cited artifact "
    "is absent or unparseable. Set each flag true only when you can point to the specific text and the specific raw value or "
    "log line. Rounding at the printed precision is not a defect. For each problem give the kind, the exact quote from the "
    "analyst's text, and a one-sentence explanation naming the raw value or log line. Finally say whether the Critic note "
    "(if any) fixes the material errors you found: yes, partly, no, or no_note."
)


def sections(text):
    cur = None
    for line in text.splitlines():
        m = HDR.match(line)
        if m:
            if cur:
                yield cur
            cur = {"id": m.group(1).strip(), "header": line, "lines": []}
        elif cur is not None:
            cur["lines"].append(line)
    if cur:
        yield cur


def split_critic(lines):
    body, note, in_note = [], [], False
    for ln in lines:
        if not in_note and re.match(r"^\s*critic note", ln, re.I):
            in_note = True
        (note if in_note else body).append(ln)
    return "\n".join(body).strip(), "\n".join(note).strip()


def cited_dirs(text, runs_root):
    ids = set(re.findall(r"runs/([A-Za-z0-9][A-Za-z0-9_.\-]*)/", text))
    out = []
    for i in sorted(ids):
        d = runs_root / i
        if d.is_dir() and i != "p0":
            out.append(d)
    return out


def evidence(entry_text, base, entry_id):
    runs_root = base / "avrl" / "runs"
    dirs = cited_dirs(entry_text, runs_root)
    if not dirs and (runs_root / entry_id).is_dir():
        dirs = [runs_root / entry_id]
    parts = []
    for d in dirs[:4]:
        rj = d / "result.json"
        if rj.exists():
            parts.append(f"=== {d.name}/result.json ===\n{rj.read_text()[:6000]}")
        else:
            parts.append(f"=== {d.name}/result.json === ABSENT")
        rl = d / "run.log"
        if rl.exists():
            tail = "\n".join(rl.read_text(errors="replace").splitlines()[-80:])
            parts.append(f"=== {d.name}/run.log (last 80 lines) ===\n{tail[:5000]}")
        ec = d / "exit_code"
        if ec.exists():
            parts.append(f"=== {d.name}/exit_code === {ec.read_text().strip()}")
    for ext in (".py", ".sh"):
        s = base / "avrl" / "exp" / f"{entry_id}{ext}"
        if s.exists():
            parts.append(f"=== exp/{entry_id}{ext} ===\n{s.read_text()[:7000]}")
            break
    return "\n\n".join(parts) if parts else "(no cited run directory found)"


def ask(prompt):
    cmd = ["claude", "-p", "--model", "sonnet", "--no-session-persistence", "--tools", "", "--strict-mcp-config",
           "--setting-sources", "", "--output-format", "json", "--json-schema", json.dumps(SCHEMA), "--system-prompt", SYSTEM, prompt]
    for attempt in range(3):
        try:
            p = subprocess.run(cmd, capture_output=True, text=True, timeout=400)
            d = json.loads(p.stdout)
            if not d.get("is_error") and isinstance(d.get("structured_output"), dict):
                return d["structured_output"], d.get("total_cost_usd", 0.0), list((d.get("modelUsage") or {}).keys())
        except (subprocess.TimeoutExpired, json.JSONDecodeError):
            pass
        time.sleep(5 * (attempt + 1))
    return None, 0.0, []


def audit_one(run, entry_id, variant, body, note, ev):
    prompt = f"ANALYST TEXT (entry {entry_id}):\n{body}\n\nCRITIC NOTE (separate; may be empty):\n{note or '(none)'}\n\nRAW EVIDENCE:\n{ev}"
    out, cost, ids = ask(prompt)
    rec = {"run": run, "id": entry_id, "variant": variant, "cost_usd": cost, "model_id": ids[0] if ids else None}
    if out is None:
        rec.update({"flags": None, "material": None, "problems": [], "critic_fixed": None, "error": "call failed"})
        return rec
    f = out["flags"]
    rec.update({"flags": f, "material": bool(f["V"] or f["N"] or f["F"]), "problems": out["problems"], "critic_fixed": out["critic_note_fixes_material_errors"]})
    return rec


def experiment_entries(ledger):
    return [s for s in sections(Path(ledger).read_text()) if not s["id"].startswith(("E01", "Note", "AR"))]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run")
    ap.add_argument("--twins")
    ap.add_argument("--seeded", action="store_true")
    ap.add_argument("--ledger", type=Path)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--workers", type=int, default=6)
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()
    jobs = []
    if a.run or a.twins:
        base = Path(a.run or a.twins)
        run_id = base.name
        ents = experiment_entries(a.ledger)
        if a.twins:
            from ar2_critic_eval import plant_number, plant_verdict
            rng = random.Random(a.seed)
            sample = rng.sample(ents, max(2, round(0.1 * len(ents))))
            for s in sample:
                body, note = split_critic(s["lines"])
                full = s["header"] + "\n" + body
                pn = plant_number(full, rng)
                pv = plant_verdict(full)
                for kind, planted in (("number", pn), ("verdict", pv)):
                    if planted:
                        txt = planted[0].split("\n", 1)[1] if "\n" in planted[0] else planted[0]
                        jobs.append((run_id, s["id"], f"twin_{kind}", txt, "", evidence(full, base, s["id"].split()[0]), planted[1]))
                jobs.append((run_id, s["id"], "twin_control", body, "", evidence(full, base, s["id"].split()[0]), None))
        else:
            for s in ents:
                body, note = split_critic(s["lines"])
                full = s["header"] + "\n" + body
                jobs.append((run_id, s["id"], "final", body, note, evidence(full, base, s["id"].split()[0]), None))
    elif a.seeded:
        key = json.loads((Path(__file__).parent / "AR5_KEY.json").read_text())
        base = Path("/homes/rdubash/active-audit")
        text = (base / "avrl" / "RESEARCH_LOG.md").read_text()
        want = {"E04", "L5a", "E05", "L7b", "E03", "E06", "E08", "L3a", "L4a", "L6a"}
        for s in sections(text):
            hid = s["id"].split()[0]
            if hid in want and s["id"] in {x for x in want} | {s["id"]}:
                body, note = split_critic(s["lines"])
                full = s["header"] + "\n" + body
                jobs.append(("DAYONE", hid, "seeded" if hid in {"E04", "L5a", "E05", "L7b", "E03"} else "seeded_control", body, "", evidence(full, base, hid), None))
                want.discard(hid)
    print(f"{len(jobs)} audit calls", flush=True)
    with a.out.open("a") as f, ThreadPoolExecutor(a.workers) as ex:
        for rec, extra in zip(ex.map(lambda j: audit_one(*j[:6]), jobs), [j[6] for j in jobs]):
            if extra is not None:
                rec["plant"] = extra
            f.write(json.dumps(rec) + "\n")
            f.flush()
            print(f"{rec['run']} {rec['id'][:30]:30s} {rec['variant']:13s} material={rec.get('material')} flags={rec.get('flags')}", flush=True)


if __name__ == "__main__":
    main()
