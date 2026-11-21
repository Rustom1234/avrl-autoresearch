#!/usr/bin/env python3
"""SR6: run trie.py unchanged, but with the tool server of iso/ (evaluate also reports which terms of a candidate
are right) and a system prompt that says so. The earlier candidates shown at the start of a step get the same term
check. Rules: loop/design/SR6_PREREG.md. Same arguments as trie.py.

    trie_iso.py run --problems problems_iso --root ../runs/SR6 --plans "haiku*4" --reps 2 --workers 12 --max-usd 2.5
    trie_iso.py prompt        print the changed system prompt and stop
"""
import json, os, runpy, subprocess, sys, tempfile, threading
from pathlib import Path

import session as S

HERE = Path(__file__).resolve().parent
EDITS = [
    ("evaluate(code) scores a candidate law.",
     "evaluate(code) scores a candidate law and checks its terms."),
    ("- The law is a short closed-form expression.",
     "- The term check. When a candidate is not exact, evaluate also tells you which of its terms are right. It tests\n"
     "  the terms that mix two inputs, and then the terms of each single input, each fitted together with free smooth\n"
     "  curves that stand in for the terms left out. So a term can be confirmed before the rest of the law is known.\n"
     "  For this, write the law as one return line that is a sum of terms.\n"
     "- The law is a short closed-form expression."),
    ("is still missing.",
     "is still missing. Use the term check: evaluate early, keep every part it reports RIGHT exactly as it is, and\n"
     "change only the part it reports NOT right."),
]
system = S.SYSTEM
for old, new in EDITS:
    assert system.count(old) == 1, old
    system = system.replace(old, new)

LOCAL = threading.local()
ENV1 = dict(os.environ, OMP_NUM_THREADS="1", OPENBLAS_NUM_THREADS="1", MKL_NUM_THREADS="1", NUMEXPR_NUM_THREADS="1")
_run_attempt, _user_prompt = S.run_attempt, S.user_prompt


def run_attempt(pdir, *args, **kw):
    LOCAL.train = str(Path(pdir).resolve() / "train.csv")
    return _run_attempt(pdir, *args, **kw)


def term_check(code, train):
    with tempfile.TemporaryDirectory() as tmp:
        f = Path(tmp) / "cand.py"
        f.write_text(code)
        try:
            p = subprocess.run([S.PY, "-I", str(HERE / "iso/term_check.py"), "--code", str(f), "--train", train],
                               capture_output=True, text=True, timeout=150, env=ENV1)
            return json.loads(p.stdout).get("text")
        except (subprocess.TimeoutExpired, ValueError):
            return None


def user_prompt(card, parents):
    """The prompt of session.py, with the term check added under each earlier candidate."""
    text = _user_prompt(card, parents)
    for p in parents or []:
        tc = term_check(p["code"], LOCAL.train)
        if tc and p["code"].strip() in text:
            text = text.replace(p["code"].strip(), p["code"].strip() + "\n" + tc, 1)
    return text


if __name__ == "__main__":
    if sys.argv[1:] == ["prompt"]:
        print(system.format(budget=14, args="t, P"))
        sys.exit(0)
    S.SYSTEM, S.run_attempt, S.user_prompt = system, run_attempt, user_prompt
    S.HERE = HERE / "iso"       # run_attempt takes tools_server.py from here; score takes fit.py (a link to ../fit.py)
    sys.argv[0] = str(HERE / "trie.py")
    runpy.run_path(str(HERE / "trie.py"), run_name="__main__")
