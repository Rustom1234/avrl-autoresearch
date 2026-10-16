"""AR1: build the offline oracle for the planner-context ablation.

Parses avrl/RESEARCH_LOG.md into one record per experiment from the small-scale
campaign (2026-09-30 to 2026-10-03). Each record carries the id, the header
verdict, the hypothesis (what a planner may see before choosing), and the
result body with the analyst's own verdict and follow-ups stripped (what the
replayed "runner" returns). Writes avrl/runs/AR1/oracle.json.
"""
import json
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
LEDGER = ROOT / "RESEARCH_LOG.md"
OUT = ROOT / "runs" / "AR1" / "oracle.json"

HDR = re.compile(r"^## (.+?) (2026-\d\d-\d\d)(.*)$")
SKIP_PREFIX = ("Note", "PRODUCT", "DIAG", "H1", "E10p0", "M10 part", "E10e burn-in")
LAST_DATE = "2026-10-03"
STRIP_PARA = re.compile(r"^(Verdict|What it changes|Follow-ups?|Critic note|Thesis impact|Consequence)\b", re.I)


def verdict_of(rest: str) -> str:
    m = re.search(r"\b(pass|kill|inconclusive|tie|no_support)\b", rest.lower())
    return m.group(1) if m else "other"


def split_entries(text: str):
    cur = None
    for line in text.splitlines():
        m = HDR.match(line)
        if m:
            if cur:
                yield cur
            cur = {"id": m.group(1).strip(), "date": m.group(2), "rest": m.group(3).strip(), "lines": []}
        elif cur is not None:
            cur["lines"].append(line)
    if cur:
        yield cur


def strip_verdict(lines):
    """Drop paragraphs that state the verdict, interpretation or follow-ups."""
    out, skipping = [], False
    for ln in lines:
        if ln.strip() == "":
            skipping = False
            out.append(ln)
            continue
        if STRIP_PARA.match(ln.strip()) or (ln.strip().startswith("|") and False):
            skipping = True
        if re.match(r"^\s*\d+\.\s", ln) and skipping:
            continue
        if not skipping:
            out.append(ln)
    body = "\n".join(out)
    body = re.sub(r"\n{3,}", "\n\n", body).strip()
    return body


VERDICT_WORDS = re.compile(r"\b(pass|kill|killed|inconclusive|tie|no_support|fail|leaning \w+|\d\d:\d\d UTC|final)\b", re.I)


def hypothesis_of(lines, header_rest):
    """What a planner may see before choosing: the hypothesis or the setting, never a result."""
    for ln in lines:
        s = ln.strip()
        if s.lower().startswith("hypothesis"):
            return s.split(":", 1)[-1].strip()
    for ln in lines:
        s = ln.strip()
        if s.lower().startswith(("setting:", "pre-registered")):
            return VERDICT_WORDS.sub("", s)[:400]
    desc = VERDICT_WORDS.sub("", header_rest)
    desc = re.sub(r"\s+", " ", desc).strip(" (),:;")
    return desc[:400]


def main():
    text = LEDGER.read_text()
    recs = {}
    for e in split_entries(text):
        if e["date"] > LAST_DATE:
            continue
        if e["id"].startswith(SKIP_PREFIX) or "interim" in e["rest"].lower() or "interim" in e["id"].lower():
            continue
        v = verdict_of(e["rest"])
        if v == "other":
            continue
        rid = e["id"]
        recs[rid] = {
            "id": rid,
            "date": e["date"],
            "header_verdict": v,
            "header_rest": e["rest"],
            "hypothesis": hypothesis_of(e["lines"], e["rest"]),
            "result_body": strip_verdict(e["lines"]),
            "full_entry": "\n".join(e["lines"]).strip(),
        }
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(list(recs.values()), indent=1))
    print(f"{len(recs)} experiments -> {OUT}")
    for r in recs.values():
        print(f"{r['id']:28s} {r['date']} {r['header_verdict']:12s} body={len(r['result_body']):5d} hyp={r['hypothesis'][:70]}")


if __name__ == "__main__":
    sys.exit(main())
