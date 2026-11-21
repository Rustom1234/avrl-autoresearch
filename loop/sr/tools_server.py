#!/usr/bin/env python3
"""The two tools a solver gets in one attempt, served to the claude command line over MCP (stdio).

    analyze(code)    run Python on the training data (numpy as np, pandas as pd, columns as arrays), returns the output
    evaluate(code)   fit the constants of a candidate law and return its training error

Environment: SR_DIR (attempt folder holding train.csv), SR_BUDGET (tool calls allowed in this attempt).
Every call is appended to SR_DIR/calls.jsonl; every scored candidate to SR_DIR/evals.jsonl. The attempt's result is
read from those files by the caller, so the solver does not need to submit anything.
All model-written code runs in a subprocess: one thread, 20 s of CPU, 4 GB of address space (quiet limits).
"""
import json, os, subprocess, sys, time
from pathlib import Path

from mcp.server.fastmcp import FastMCP

DIR = Path(os.environ["SR_DIR"])
BUDGET = int(os.environ.get("SR_BUDGET", "14"))
HERE = Path(__file__).resolve().parent
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
    (several starting points of order 1). Returns the training NMSE and the fitted constants. Counts as one call."""
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
    return ("training NMSE = %.3e\nfitted params = %s" % (r["train_nmse"], ["%.5g" % x for x in r["params"]])) + left(n + 1)


if __name__ == "__main__":
    mcp.run()
