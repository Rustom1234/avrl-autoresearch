"""AR5 section 8 (a): machine gate for "verdict reached".

Given a run BASE and a ledger prefix file (a snapshot of RESEARCH_LOG.md taken after the
k-th experiment entry), list the experiment entries that qualify and the run dirs they cite.
An entry qualifies when it cites at least one runs/<id>/ directory under BASE/avrl/runs that
  - holds done.flag,
  - holds a result.json that parses as JSON, and
  - whose result.json mtime is strictly after T0 (seeded dirs keep their old mtime, so they never count).
The gate (a) is satisfied when at least two distinct non-E01 experiment entries qualify.

This file also holds the ledger parsing helpers shared by ar5_adjudicate.py and ar5_trace.py.

Usage:
  python ar5_gate.py --base /homes/rdubash/avrl_live/FULL --ledger BASE/snap/ledger_<epoch>_<sha8>.md --t0 <epoch>
T0 comes from --t0, else from the first token of BASE/T0. No model calls.
"""
import argparse
import json
import re
import sys
from pathlib import Path

CRITIC_START = re.compile(r"[\s*_>#`]*critic note", re.I)
ENTRY_HDR = re.compile(r"^## (\S+) (\d{4}-\d{2}-\d{2})\b(.*)$")
RUN_REF = re.compile(r"(?<![\w])runs/([A-Za-z0-9_][A-Za-z0-9_.\-]*)")
FILE_EXT = re.compile(r"\.[A-Za-z0-9]{1,8}$")


def split_entries(text: str):
    """Experiment entries of a ledger: '## <id> <YYYY-MM-DD> ...' sections, in order. Notes are skipped."""
    lines = text.splitlines()
    heads = [i for i, ln in enumerate(lines) if ln.startswith("## ")]
    out = []
    for n, i in enumerate(heads):
        m = ENTRY_HDR.match(lines[i])
        if not m or m.group(1).lower() == "note":
            continue
        j = heads[n + 1] if n + 1 < len(heads) else len(lines)
        body = "\n".join(lines[i:j]).rstrip() + "\n"
        main, critic = split_critic(body)
        out.append({"id": m.group(1), "date": m.group(2), "header": lines[i], "line": i + 1, "text": body, "main": main, "critic": critic})
    return out


def split_critic(entry_text: str):
    """(entry without critic note, critic note). The note starts at the first paragraph that begins 'Critic note' and runs to the end."""
    paras = entry_text.split("\n\n")
    for k, p in enumerate(paras):
        if CRITIC_START.match(p):
            return "\n\n".join(paras[:k]).rstrip() + "\n", "\n\n".join(paras[k:]).strip()
    return entry_text, ""


def cited_dirs(text: str):
    """Ordered unique run directory names cited as runs/<name> (file references like runs/sim_generic.py are dropped)."""
    seen = []
    for m in RUN_REF.finditer(text):
        name = m.group(1).rstrip(".")
        if FILE_EXT.search(name) or name in seen:
            continue
        seen.append(name)
    return seen


def check_dir(runs_root: Path, name: str, t0: float):
    d = runs_root / name
    rj = d / "result.json"
    info = {"dir": f"runs/{name}", "done_flag": (d / "done.flag").is_file(), "result_json": rj.is_file(), "parseable": False, "mtime": None, "after_t0": False}
    if rj.is_file():
        info["mtime"] = rj.stat().st_mtime
        info["after_t0"] = info["mtime"] > t0
        try:
            json.loads(rj.read_text())
            info["parseable"] = True
        except (ValueError, OSError):
            pass
    info["ok"] = info["done_flag"] and info["parseable"] and info["after_t0"]
    return info


def gate(base: Path, ledger_text: str, t0: float):
    runs_root = base / "avrl" / "runs"
    qualifying, rest = [], []
    for e in split_entries(ledger_text):
        dirs = [check_dir(runs_root, n, t0) for n in cited_dirs(e["main"])]
        rec = {"id": e["id"], "header": e["header"], "cited": dirs, "qualifying_dirs": [d["dir"] for d in dirs if d["ok"]]}
        (qualifying if rec["qualifying_dirs"] else rest).append(rec)
    distinct = sorted({r["id"] for r in qualifying if r["id"].upper() != "E01"})
    # stricter companion reading: the two entries must also rest on different primary dirs (own-named dir if it qualifies, else the last qualifying dir)
    prim = {}
    for r in qualifying:
        if r["id"].upper() != "E01":
            own = f"runs/{r['id']}"
            prim[r["id"]] = own if own in r["qualifying_dirs"] else r["qualifying_dirs"][-1]
    return {
        "base": str(base),
        "t0": t0,
        "n_entries": len(qualifying) + len(rest),
        "qualifying": qualifying,
        "non_qualifying": rest,
        "distinct_non_e01_qualifying": distinct,
        "gate": len(distinct) >= 2,
        "primary_dirs": prim,
        "gate_distinct_dirs": len(set(prim.values())) >= 2,
    }


def read_t0(base: Path, arg):
    if arg is not None:
        return float(arg)
    f = base / "T0"
    if f.is_file():
        tok = f.read_text().split()
        if tok:
            try:
                return float(tok[0])
            except ValueError:
                pass
        sys.exit(f"ar5_gate: {f} does not start with an epoch")
    sys.exit("ar5_gate: no T0 (pass --t0 EPOCH or put the epoch in BASE/T0)")


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--base", type=Path, required=True)
    ap.add_argument("--ledger", type=Path, required=True)
    ap.add_argument("--t0", default=None)
    a = ap.parse_args()
    res = gate(a.base, a.ledger.read_text(), read_t0(a.base, a.t0))
    print(json.dumps(res, indent=1))


if __name__ == "__main__":
    main()
