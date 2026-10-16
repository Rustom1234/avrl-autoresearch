"""AR2: offline evaluation of the critic role, built from the ledger.

Takes the 52 small-scale ledger entries (avrl/runs/AR1/oracle.json, full_entry) and
plants one defect in each of two kinds, keeping the untouched entry as a control:
  number   a number in the narrative that also appears in the entry's table is changed
           by 20 to 60 percent, so the text no longer matches the table
  verdict  the verdict word in the Verdict paragraph is flipped (kill <-> pass,
           inconclusive -> pass), so the verdict no longer follows from the numbers
A critic model reads the entry alone (no raw files) and reports whether it is
internally consistent and, if not, where. Scored: detection rate per defect kind,
false-alarm rate on controls, by model. Writes avrl/runs/AR2/{rows.jsonl,result.json}.
"""
import argparse
import json
import random
import re
import subprocess
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
ORACLE = ROOT / "runs" / "AR1" / "oracle.json"
OUT = ROOT / "runs" / "AR2"

SCHEMA = {
    "type": "object",
    "properties": {
        "consistent": {"type": "boolean"},
        "problem": {"type": "string"},
        "quote": {"type": "string"},
    },
    "required": ["consistent", "problem", "quote"],
}
SYSTEM = (
    "You are the adversarial critic of a research ledger. You are given one ledger entry. Check that every number stated in the prose "
    "matches the entry's own table and that the verdict follows from the numbers and the stated pass and kill thresholds. "
    "You have no access to raw files; judge internal consistency only. If the entry is consistent, set consistent=true and leave problem and quote empty. "
    "If not, set consistent=false, describe the single most serious problem in one sentence, and copy the exact offending text into quote."
)
NUM = re.compile(r"(?<![\w.])(\d+\.\d+|\d{2,})(?![\w.])")
FLIP = {"kill": "pass", "pass": "kill", "inconclusive": "pass", "tie": "pass"}


def plant_number(entry: str, rng: random.Random):
    lines = entry.splitlines()
    table_nums = set()
    for ln in lines:
        if ln.strip().startswith("|"):
            table_nums.update(NUM.findall(ln))
    cands = []
    for i, ln in enumerate(lines):
        if ln.strip().startswith("|") or ln.lower().startswith(("files:", "run:")):
            continue
        for m in NUM.finditer(ln):
            if m.group(1) in table_nums and float(m.group(1)) != 0:
                cands.append((i, m))
    if not cands:
        return None
    i, m = rng.choice(cands)
    old = m.group(1)
    f = rng.choice([0.4, 0.6, 1.4, 1.6])
    new = f"{float(old) * f:.{len(old.split('.')[1]) if '.' in old else 0}f}"
    if new == old:
        return None
    lines[i] = lines[i][: m.start(1)] + new + lines[i][m.end(1):]
    return "\n".join(lines), {"line": i, "old": old, "new": new}


def plant_verdict(entry: str):
    lines = entry.splitlines()
    for i, ln in enumerate(lines):
        if ln.lower().startswith("verdict"):
            m = re.search(r"\b(kill|pass|inconclusive|tie)\b", ln, re.I)
            if m:
                new = FLIP[m.group(1).lower()]
                lines[i] = ln[: m.start(1)] + new + ln[m.end(1):]
                return "\n".join(lines), {"line": i, "old": m.group(1), "new": new}
    return None


def ask(model: str, entry: str):
    cmd = ["claude", "-p", "--model", model, "--no-session-persistence", "--tools", "", "--strict-mcp-config", "--setting-sources", "",
           "--output-format", "json", "--json-schema", json.dumps(SCHEMA), "--system-prompt", SYSTEM, "Ledger entry:\n\n" + entry]
    for attempt in range(3):
        p = subprocess.run(cmd, capture_output=True, text=True, timeout=600)
        try:
            d = json.loads(p.stdout)
            if not d.get("is_error") and d.get("structured_output") is not None:
                return d["structured_output"], d.get("total_cost_usd", 0.0)
        except json.JSONDecodeError:
            pass
        time.sleep(5 * (attempt + 1))
    return {"consistent": None, "problem": "call failed", "quote": ""}, 0.0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--models", nargs="+", default=["haiku", "sonnet"])
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--workers", type=int, default=6)
    a = ap.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    rng = random.Random(a.seed)
    recs = json.loads(ORACLE.read_text())
    items = []
    for r in recs:
        # drop the real critic's paragraphs so controls are the analyst's text as written
        paras = [p for p in r["full_entry"].split("\n\n") if not p.lower().startswith("critic note")]
        e = "\n\n".join(paras)
        if "|" not in e:
            continue
        items.append({"id": r["id"], "kind": "control", "entry": e, "plant": None})
        pn = plant_number(e, rng)
        if pn:
            items.append({"id": r["id"], "kind": "number", "entry": pn[0], "plant": pn[1]})
        pv = plant_verdict(e)
        if pv:
            items.append({"id": r["id"], "kind": "verdict", "entry": pv[0], "plant": pv[1]})
    jobs = [(m, it) for m in a.models for it in items]
    print(f"{len(items)} items ({sum(i['kind']=='control' for i in items)} control, {sum(i['kind']=='number' for i in items)} number, {sum(i['kind']=='verdict' for i in items)} verdict) x {len(a.models)} models = {len(jobs)} calls", flush=True)

    def run(job):
        m, it = job
        out, cost = ask(m, it["entry"])
        flagged = out.get("consistent") is False
        hit = None
        if it["kind"] == "number":
            hit = flagged and (it["plant"]["new"] in (out.get("quote") or "") or it["plant"]["new"] in (out.get("problem") or ""))
        elif it["kind"] == "verdict":
            txt = ((out.get("quote") or "") + " " + (out.get("problem") or "")).lower()
            hit = flagged and ("verdict" in txt or it["plant"]["new"] in txt)
        return {"model": m, "id": it["id"], "kind": it["kind"], "plant": it["plant"], "flagged": flagged, "hit": hit, "cost": cost, **{k: out.get(k) for k in ("problem", "quote", "consistent")}}

    rows = []
    with (OUT / "rows.jsonl").open("w") as f, ThreadPoolExecutor(a.workers) as ex:
        for row in ex.map(run, jobs):
            rows.append(row)
            f.write(json.dumps(row) + "\n")
            f.flush()
            print(f"{row['model']:7s} {row['id']:24s} {row['kind']:8s} flagged={row['flagged']} hit={row['hit']}", flush=True)

    summary = {}
    for m in a.models:
        rm = [r for r in rows if r["model"] == m and r["consistent"] is not None]
        s = {"calls": len(rm), "cost_usd": round(sum(r["cost"] for r in rm), 3)}
        for kind in ("control", "number", "verdict"):
            rk = [r for r in rm if r["kind"] == kind]
            if not rk:
                continue
            if kind == "control":
                s["false_alarm_rate"] = round(sum(r["flagged"] for r in rk) / len(rk), 3)
                s["controls"] = len(rk)
            else:
                s[f"{kind}_flagged_rate"] = round(sum(r["flagged"] for r in rk) / len(rk), 3)
                s[f"{kind}_located_rate"] = round(sum(bool(r["hit"]) for r in rk) / len(rk), 3)
                s[f"{kind}_n"] = len(rk)
        summary[m] = s
    (OUT / "result.json").write_text(json.dumps(summary, indent=1))
    print(json.dumps(summary, indent=1))
    (OUT / "done.flag").touch()


if __name__ == "__main__":
    main()
