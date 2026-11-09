"""AR5: experiments-to-verdict per run. Builds ledger prefixes at experiment boundaries, runs the machine gate and the
blind adjudicator on each prefix, and writes runs/AR5/grades_<run>.json with V(k), the first V, the settled k*, and flips.

A prefix k is the final (post-critic) ledger cut before the first header of experiment k+1, where experiments are
distinct ids in order of first appearance (pre-registration sections, result sections and critic-appended sections
with the same id belong to the same experiment). E01 is the seed and is experiment 0.

Usage: ar5_grade_runs.py --base /homes/rdubash/avrl_live/FULL --t0 1792365340 [--out runs/AR5/grades_FULL.json]
"""
import argparse
import json
import re
import subprocess
import sys
from pathlib import Path

HDR = re.compile(r"^## (\S+)")
EXP = Path(__file__).resolve().parent
PY = sys.executable


def prefixes(ledger_text):
    lines = ledger_text.splitlines()
    starts, ids = [], []
    for i, ln in enumerate(lines):
        m = HDR.match(ln)
        if m:
            hid = m.group(1)
            if hid.startswith("E01") or hid in ("Note", "AR3", "AR5"):
                continue
            if hid not in ids:
                ids.append(hid)
                starts.append(i)
    out = []
    for k in range(1, len(ids) + 1):
        end = starts[k] if k < len(ids) else len(lines)
        out.append((k, ids[k - 1], "\n".join(lines[:end]) + "\n"))
    return out


def run_json(cmd):
    p = subprocess.run(cmd, capture_output=True, text=True, timeout=1800)
    txt = p.stdout.strip()
    try:
        return json.loads(txt[txt.rindex("\n{") + 1:] if "\n{" in txt else txt)
    except (ValueError, json.JSONDecodeError):
        return {"error": (p.stdout[-600:] + p.stderr[-600:])}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", type=Path, required=True)
    ap.add_argument("--t0", required=True)
    ap.add_argument("--out", type=Path, default=None)
    a = ap.parse_args()
    run_id = a.base.name
    out = a.out or (EXP.parent / "runs" / "AR5" / f"grades_{run_id}.json")
    pdir = EXP.parent / "runs" / "AR5" / "prefixes" / run_id
    pdir.mkdir(parents=True, exist_ok=True)
    ledger = (a.base / "avrl" / "RESEARCH_LOG.md").read_text()
    rows = []
    for k, hid, text in prefixes(ledger):
        pf = pdir / f"prefix_{k:02d}.md"
        pf.write_text(text)
        gate = run_json([PY, str(EXP / "ar5_gate.py"), "--base", str(a.base), "--ledger", str(pf), "--t0", a.t0])
        adj = run_json([PY, str(EXP / "ar5_adjudicate.py"), "--base", str(a.base), "--ledger", str(pf)])
        vote = adj.get("vote") or {}
        gate_ok = bool(gate.get("gate", gate.get("gate_a", False))) if isinstance(gate, dict) else False
        v_adj = vote.get("V_adjudicator")
        if v_adj is None and vote:
            v_adj = vote.get("c2_state") == "not_supported" and vote.get("c3_state") == "killed" and not vote.get("W2") and not vote.get("W3")
        V = bool(gate_ok and v_adj)
        rows.append({"k": k, "id": hid, "gate": gate_ok, "gate_detail": {kk: vv for kk, vv in gate.items() if kk != "entries"} if isinstance(gate, dict) else gate,
                     "vote": vote, "W2": vote.get("W2"), "W3": vote.get("W3"), "V": V, "adj_cost_usd": adj.get("cost_usd"), "adj_error": adj.get("error")})
        print(f"{run_id} k={k:2d} {hid[:28]:28s} gate={gate_ok!s:5s} c2={vote.get('c2_state')} c3={vote.get('c3_state')} W2={vote.get('W2')} W3={vote.get('W3')} V={V}", flush=True)
    Vs = [r["V"] for r in rows]
    first_v = next((r["k"] for r in rows if r["V"]), None)
    settled = next((rows[i]["k"] for i in range(len(rows) - 1) if Vs[i] and Vs[i + 1]), None)
    flips_after_v = sum(1 for i in range(1, len(rows)) if Vs[i - 1] and not Vs[i])
    wrong_final = bool(rows and (rows[-1]["W2"] or rows[-1]["W3"] or rows[-1]["vote"].get("c2_state") == "supported" or rows[-1]["vote"].get("c3_state") == "supported"))
    ever_wrong = any((r["W2"] or r["W3"] or r["vote"].get("c2_state") == "supported" or r["vote"].get("c3_state") == "supported") for r in rows)
    res = {"run": run_id, "n_experiments": len(rows), "first_V": first_v, "settled_k": settled, "censored": settled is None,
           "flips_after_first_V": flips_after_v, "ever_wrong_claim_state": ever_wrong, "wrong_final_state": wrong_final,
           "final_vote": rows[-1]["vote"] if rows else None, "rows": rows,
           "adj_cost_usd_total": round(sum((r["adj_cost_usd"] or 0) for r in rows), 3)}
    out.write_text(json.dumps(res, indent=1))
    print(json.dumps({k: v for k, v in res.items() if k != "rows"}, indent=1))


if __name__ == "__main__":
    main()
