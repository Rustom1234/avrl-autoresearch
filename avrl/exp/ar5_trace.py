"""AR5 section 9 trace: does every number in a ledger entry appear in its cited result.json? No model calls.

For each experiment entry (critic note stripped off) every number in the prose and tables is looked up among the
numeric leaves of the result.json files in the runs/<id>/ directories the entry cites. Classes:
  traceable    some raw value equals the number within relative tolerance 0.5 percent, or rounds to it at the number's
               own last digit (percent-marked numbers are also compared against raw * 100)
  mismatched   no traceable value, but a raw value is a near miss: within 1.5 units of the number's last printed digit
               (decimals >= 1), within 2 percent for integers >= 100, or equal up to sign. The signature of a
               copied-wrong or edited number; coincidental near misses occur, so read the matched path
  untraceable  nothing close in the cited result.json files (timings, thresholds, counts read from other files, ...)
Untraceable is not an error by itself; ar5_audit.py judges those. Skipped as not numeric claims: the header line,
dates, clock times, paths, tokens glued to letters (E03, cell100, 1.5B), list markers.

Usage:
  python ar5_trace.py --ledger FILE (--base BASE | --runs-root DIR) [--ids E03,L3a] [--out trace.json]
"""
import argparse
import json
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from ar5_gate import cited_dirs, split_entries  # noqa: E402

REL_TOL = 0.005
NUM = re.compile(r"(?<![\w.])([-−]?(?:\d{1,3}(?:,\d{3})+|\d+)(?:\.\d+)?(?:[eE][-+]?\d+)?)(?![\w])(?!\.\d)")
PCT_AFTER = re.compile(r"\s*(?:%|percent\b|pp\b|points?\b)", re.I)
DATE = re.compile(r"\d{4}-\d{2}-\d{2}")
CLOCK = re.compile(r"\b\d{1,2}:\d{2}(?::\d{2})?\b")
LISTMARK = re.compile(r"^\s*\d+[.)]\s", re.M)


def leaves(obj, path=""):
    if isinstance(obj, bool):
        return
    if isinstance(obj, (int, float)):
        yield path, float(obj)
    elif isinstance(obj, dict):
        for k, v in obj.items():
            yield from leaves(v, f"{path}.{k}" if path else str(k))
    elif isinstance(obj, list):
        for i, v in enumerate(obj):
            yield from leaves(v, f"{path}[{i}]")


def numbers(text: str):
    lines = [ln for ln in text.splitlines() if not ln.startswith("## ")]
    t = "\n".join(lines)
    t = DATE.sub(" ", t)
    t = CLOCK.sub(" ", t)
    t = LISTMARK.sub("", t)
    t = re.sub(r"\S*[A-Za-z]\S*/\S*|\S*/\S*[A-Za-z]\S*", " ", t)  # path-like tokens
    for m in NUM.finditer(t):
        raw = m.group(1)
        s = raw.replace(",", "").replace("−", "-")
        try:
            x = float(s)
        except ValueError:
            continue
        dec = len(s.split("e")[0].split(".")[1]) if "." in s.split("e")[0] else 0
        pct = bool(PCT_AFTER.match(t, m.end()))
        yield {"text": raw, "value": x, "decimals": dec, "pct": pct, "context": t[max(0, m.start() - 40): m.end() + 25].replace("\n", " ")}


def agree(x: float, dec: int, v: float) -> bool:
    if v == x:
        return True
    if v != 0 and abs(x - v) / abs(v) <= REL_TOL:
        return True
    return round(v, dec) == round(x, dec)


def near(x: float, dec: int, v: float) -> float:
    """Closeness score of a near miss (smaller is closer) or None. Near = within 1.5 units of the number's last printed digit
    (decimals >= 1), within 2 percent for integers of 100 or more, or equal up to sign."""
    if x != 0 and agree(abs(x), dec, abs(v)) and (x < 0) != (v < 0):
        return 0.0
    if dec >= 1 and abs(x - v) <= 1.5 * 10 ** (-dec) + 1e-12:
        return abs(x - v) / 10 ** (-dec)
    if dec == 0 and abs(x) >= 100 and v != 0 and abs(x - v) / abs(v) <= 0.02:
        return abs(x - v) / abs(v) * 100
    return None


def classify(n, raws):
    x, dec = n["value"], n["decimals"]
    best = None
    for src, path, v in raws:
        for vv in [v] + ([v * 100.0] if n["pct"] else []):
            if agree(x, dec, vv):
                return "traceable", {"file": src, "path": path, "value": v}
            d = near(x, dec, vv)
            if d is not None and (best is None or d < best[0]):
                best = (d, {"file": src, "path": path, "value": v})
    if best:
        return "mismatched", best[1]
    return "untraceable", None


def trace_entry(e, runs_root: Path):
    raws, files = [], []
    for name in cited_dirs(e["main"]):
        rj = runs_root / name / "result.json"
        if not rj.is_file():
            continue
        try:
            data = json.loads(rj.read_text())
        except ValueError:
            continue
        files.append(f"runs/{name}/result.json")
        raws += [(f"runs/{name}/result.json", p, v) for p, v in leaves(data)]
    rows = []
    for n in numbers(e["main"]):
        cls, hit = classify(n, raws)
        rows.append({"text": n["text"], "class": cls, "match": hit, "context": n["context"]})
    c = {k: sum(r["class"] == k for r in rows) for k in ("traceable", "mismatched", "untraceable")}
    tot = len(rows)
    return {"id": e["id"], "result_files": files, "n_numbers": tot, **c, "frac_traceable": round(c["traceable"] / tot, 3) if tot else None, "numbers": rows}


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--ledger", type=Path, required=True)
    g = ap.add_mutually_exclusive_group(required=True)
    g.add_argument("--base", type=Path)
    g.add_argument("--runs-root", type=Path)
    ap.add_argument("--ids", default=None, help="comma-separated entry ids (default: all experiment entries)")
    ap.add_argument("--out", type=Path, default=None)
    ap.add_argument("--show", action="store_true", help="print mismatched numbers")
    a = ap.parse_args()
    runs_root = a.runs_root or a.base / "avrl" / "runs"
    want = set(a.ids.split(",")) if a.ids else None
    res = [trace_entry(e, runs_root) for e in split_entries(a.ledger.read_text()) if want is None or e["id"] in want]
    for r in res:
        print(f"{r['id']:10s} numbers={r['n_numbers']:3d} traceable={r['traceable']:3d} mismatched={r['mismatched']:2d} untraceable={r['untraceable']:3d} files={len(r['result_files'])}")
        if a.show:
            for x in r["numbers"]:
                if x["class"] == "mismatched":
                    print(f"    mismatched {x['text']!r} near {x['match']['value']} at {x['match']['file']}:{x['match']['path']} | {x['context']}")
    if a.out:
        a.out.write_text(json.dumps(res, indent=1))


if __name__ == "__main__":
    main()
