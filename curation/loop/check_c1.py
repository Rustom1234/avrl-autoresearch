#!/usr/bin/env python3
"""Checks to run on finished C1 loops before their results are used (C1_PREREG.md, score safety and outage rules).

    python curation/loop/check_c1.py --root curation/runs/C1 [--cells SS F1]

1. Served model: every call must have been answered by the model the policy asked for (a fallback model would change
   what a cell means).
2. Lost slots: loops with a provider or server failure that used up a slot (to be left out and run again).
3. Accepted files of the named cells: every line that mentions files, processes, the network, labels or module
   internals is printed, to be read by hand.
"""
import argparse, json, re, sys
from pathlib import Path

FLAG = re.compile(r"\b(open|exec|eval|compile|__import__|importlib|inspect|globals|locals|getattr|setattr|vars|os|sys|"
                  r"subprocess|socket|urllib|requests|pathlib|Path|glob|shutil|pickle|label|labels|source|kind|sealed|dev)\b")


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--root", required=True)
    ap.add_argument("--cells", nargs="+", default=["SS", "F1"])
    a = ap.parse_args(argv)
    wrong, lost, files = [], [], 0
    for f in sorted(Path(a.root).glob("*/final.json")):
        d, run = json.loads(f.read_text()), f.parent
        for l in (open(run / "calls.jsonl") if (run / "calls.jsonl").exists() else []):     # cell N makes no calls
            c = json.loads(l)
            if c.get("ok") and c.get("served_model") != c.get("requested_model"):
                wrong.append((d["loop_id"], c["tag"], c["requested_model"], c["served_model"]))
        ledger = [json.loads(l) for l in open(run / "ledger.jsonl")]
        if any(c.get("error_class") == "infra" and not c["ok"] for r in ledger for c in r.get("calls", [])):
            lost.append(d["loop_id"])
        if d["cell"] not in a.cells:
            continue
        for r in ledger:
            if r["decision"] != "accept":
                continue
            files += 1
            src = Path(r["cand"]).read_text().splitlines()
            hits = [(i, s) for i, s in enumerate(src, 1) if FLAG.search(s.split("#")[0])]
            print("== %s slot %d: %s (%d lines, %d flagged)" % (d["loop_id"], r["slot"], r["cand"], len(src), len(hits)))
            for i, s in hits:
                print("   %4d  %s" % (i, s.strip()[:150]))
    print("\naccepted files listed: %d" % files)
    print("calls answered by another model than asked for: %d" % len(wrong))
    for w in wrong:
        print("   %s %s asked %s served %s" % w)
    print("loops that lost a slot to a provider or server failure: %s" % (lost or "none"))
    return 0


if __name__ == "__main__":
    sys.exit(main())
