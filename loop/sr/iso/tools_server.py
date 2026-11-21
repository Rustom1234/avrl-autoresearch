#!/usr/bin/env python3
"""SR6 tool server: the two tools of ../tools_server.py. The only change: when a scored candidate is not exact,
evaluate also reports which of its terms are right (term_check.py). The term check is logged to parts.jsonl.

Environment, logging and limits are those of ../tools_server.py.
"""
import json, os, subprocess, sys, time
from pathlib import Path

from mcp.server.fastmcp import FastMCP

DIR = Path(os.environ["SR_DIR"])
BUDGET = int(os.environ.get("SR_BUDGET", "14"))
ISO = Path(__file__).resolve().parent
HERE = ISO.parent
PY = str(HERE / ".venv/bin/python")
ENV = dict(os.environ, OMP_NUM_THREADS="1", OPENBLAS_NUM_THREADS="1", MKL_NUM_THREADS="1", NUMEXPR_NUM_THREADS="1")
LIMIT = ["prlimit", "--as=%d" % (4 << 30), "--cpu=20"]
PRE = ("import numpy as np, pandas as pd\n"
       "df = pd.read_csv('train.csv')\n"
       "for _c in df.columns: globals()[_c] = df[_c].to_numpy()\n")
mcp = FastMCP("sr")


def used():
    f = DIR / "calls.jsonl"
    return sum(1 for _ in open(f)) if f.exists() else 0


def log(name, rec):
    with open(DIR / name, "a") as f:
        f.write(json.dumps(rec) + "\n")


def run(argv, stdin=None):
    try:
        p = subprocess.run(LIMIT + argv, input=stdin, capture_output=True, text=True, timeout=60, cwd=str(DIR), env=ENV)
        return p.stdout, p.stderr
    except subprocess.TimeoutExpired:
        return "", "stopped: the code ran longer than allowed"


def left(n):
    return "\n[tool calls left in this attempt: %d]" % max(0, BUDGET - n)


@mcp.tool()
def analyze(code: str) -> str:
    """Run Python code on the training data and return what it prints. numpy is np, pandas is pd, the table is df
    and every column is also a numpy array under its own name. One thread, 20 s of CPU. Counts as one tool call."""
    n = used()
    if n >= BUDGET:
        return "No tool calls left. Reply with one line that names your best law."
    out, err = run([PY, "-I", "-c", PRE + code])
    text = (out + ("\n[stderr]\n" + err[-1500:] if err.strip() else ""))[-4000:] or "(no output; use print)"
    log("calls.jsonl", {"t": time.time(), "tool": "analyze", "code": code, "out": text[:2000]})
    return text + left(n + 1)


@mcp.tool()
def evaluate(code: str) -> str:
    """Score a candidate law. `code` must define  def equation(<inputs in the order given in the task>, params)
    returning a numpy expression; params[0] .. params[9] are constants that are fitted for you by least squares
    (several starting points of order 1). Returns the training NMSE and the fitted constants. If the law is not
    exact, it also reports which of its terms are right. Counts as one call."""
    n = used()
    if n >= BUDGET:
        return "No tool calls left. Reply with one line that names your best law."
    cf = DIR / ("cand_%d.py" % n)
    cf.write_text(code)
    out, err = run([PY, "-I", str(HERE / "fit.py"), "--code", str(cf), "--train", str(DIR / "train.csv")])
    try:
        r = json.loads(out)
    except ValueError:
        r = {"ok": False, "error": "scoring failed: " + (err or out)[-300:]}
    log("calls.jsonl", {"t": time.time(), "tool": "evaluate", "code": code, "out": json.dumps(r)[:600]})
    if not r.get("ok"):
        return "Error: " + r.get("error", "unknown") + left(n + 1)
    log("evals.jsonl", {"call": n, "code": code, "train_nmse": r["train_nmse"], "params": r["params"]})
    text = "training NMSE = %.3e\nfitted params = %s" % (r["train_nmse"], ["%.5g" % x for x in r["params"]])
    if r["train_nmse"] > 1e-12:
        try:
            p = subprocess.run([PY, "-I", str(ISO / "term_check.py"), "--code", str(cf), "--train", str(DIR / "train.csv")],
                               capture_output=True, text=True, timeout=150, cwd=str(DIR), env=ENV)
            tc = json.loads(p.stdout)
        except (subprocess.TimeoutExpired, ValueError):
            tc = {"ok": False, "error": "term check did not finish"}
        log("parts.jsonl", dict(tc, call=n, train_nmse=r["train_nmse"]))
        if tc.get("text"):
            text += "\n" + tc["text"]
        else:
            text += ("\n(no term check for this candidate: write the law as one return line that is a sum of terms)")
    return text + left(n + 1)


if __name__ == "__main__":
    mcp.run()
