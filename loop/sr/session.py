#!/usr/bin/env python3
"""One attempt at one problem by one model: a tool-using session run through the claude command line.

    session.py --problem loop/sr/problems/chem_L3_s0 --model haiku --out /some/folder [--budget 14]

The model gets the problem card and two tools (tools_server.py): analyze and evaluate, at most --budget calls.
Its best scored candidate (lowest training error) is then scored by us on the held-out splits. Returns one record:
cost, token use, tool calls, the candidate and its errors on val, test_id and test_ood.
Concurrency and rate-limit waiting use the loop's own model service (loop/models/call.py: slots and breaker).
"""
import argparse, json, os, re, shutil, subprocess, sys, time
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "models"))
import call as C  # noqa: E402  (slots, breaker, rate-limit patterns; nothing in it is changed)

PY = str(HERE / ".venv/bin/python")
AUTH = re.compile(r"login|authenticat|credential|unauthorized|\b401\b|invalid api key|oauth", re.I)
ENV1 = dict(os.environ, OMP_NUM_THREADS="1", OPENBLAS_NUM_THREADS="1", MKL_NUM_THREADS="1")

SYSTEM = """You are a scientist. Your job is to find the exact law behind a table of measurements.

How this works
- You have two tools. analyze(code) runs Python on the training data. evaluate(code) scores a candidate law.
- You have {budget} tool calls in total, so plan them.
- A candidate law is Python code that defines  def equation({args}, params)  and returns a numpy expression of the
  inputs. params[0] to params[9] are free constants. They are fitted for you from several starting points of order 1,
  so put a constant wherever you are not sure of a number. Only numpy (np) may be used inside equation: no data, no
  loops over rows, no fitting inside it.
- The data has no measurement noise. The right law reaches a training NMSE below 1e-12. An NMSE of 1e-4 or 1e-6
  means the structure of the law is still wrong, not that it is nearly right.
- The law is a short closed-form expression. It can contain terms that are standard for this kind of system and
  unusual ones (nested functions, non-integer powers, products of functions of different inputs).
- Your best scored candidate is recorded automatically. When you are done or out of calls, reply with one short line.

A method that tends to work: look at the data first (how the target changes with one input while the others are
nearly fixed); build the law term by term; study the residual of your best candidate against each input to see what
is still missing."""


def user_prompt(card, parents):
    L = [card["context"], "",
         "Target: %s = %s (%s)" % (card["target"]["name"], card["target"]["meaning"], card["target"]["unit"]),
         "Inputs, in this order:"]
    for i in card["inputs"]:
        L.append("  %s = %s (%s), measured between %g and %g" % (i["name"], i["meaning"], i["unit"], *i["train_range"]))
    L.append("Rows in the training table: %d. Columns: %s." % (
        card["n_train"], ", ".join([i["name"] for i in card["inputs"]] + [card["target"]["name"]])))
    if parents:
        L += ["", "Earlier attempts on this problem ended with these candidates (best first). None is the exact law.",
              "Start from what they show; do not repeat them."]
        for p in parents:
            L += ["", "training NMSE %.3e:" % p["train_nmse"], p["code"].strip()]
    L += ["", "Find the law."]
    return "\n".join(L)


def score(code_path, pdir, splits=("val", "test_id", "test_ood")):
    argv = ["prlimit", "--as=%d" % (4 << 30), "--cpu=120", PY, "-I", str(HERE / "fit.py"), "--code", str(code_path),
            "--train", str(pdir / "train.csv"), "--eval"] + ["%s=%s" % (s, pdir / (s + ".csv")) for s in splits]
    try:
        p = subprocess.run(argv, capture_output=True, text=True, timeout=300, env=ENV1)
        return json.loads(p.stdout)
    except (subprocess.TimeoutExpired, ValueError) as e:
        return {"ok": False, "error": "scoring failed: %s" % type(e).__name__}


def run_attempt(pdir, model, out, budget=14, parents=None, max_usd=4.0, timeout_s=1500):
    pdir, out = Path(pdir).resolve(), Path(out).resolve()
    out.mkdir(parents=True, exist_ok=True)
    card = json.loads((pdir / "card.json").read_text())
    shutil.copy(pdir / "train.csv", out / "train.csv")          # the only data the solver's code can see here
    cfg = out / "mcp.json"
    cfg.write_text(json.dumps({"mcpServers": {"sr": {"command": PY, "args": [str(HERE / "tools_server.py")],
                                                     "env": {"SR_DIR": str(out), "SR_BUDGET": str(budget)}}}}))
    system = SYSTEM.format(budget=budget, args=", ".join(i["name"] for i in card["inputs"]))
    argv = ["claude", "-p", "--model", model, "--no-session-persistence", "--tools", "", "--strict-mcp-config",
            "--mcp-config", str(cfg), "--allowedTools", "mcp__sr__analyze", "mcp__sr__evaluate",
            "--setting-sources", "", "--max-budget-usd", str(max_usd), "--output-format", "json",
            "--system-prompt", system]
    rec = {"problem": pdir.name, "model": model, "budget": budget, "n_parents": len(parents or []), "ok": False,
           "error": None, "error_class": None, "t_start": time.time()}
    t0 = time.time()
    try:
        with C._claude_slot():
            C._wait_breaker()
            p = subprocess.run(argv, input=user_prompt(card, parents), capture_output=True, text=True,
                               timeout=timeout_s, cwd=str(out))
        rc, so, se = p.returncode, p.stdout, p.stderr
    except subprocess.TimeoutExpired:
        rc, so, se = -1, "", "session timed out after %d s" % timeout_s
    rec["wall_s"] = time.time() - t0
    try:
        d = json.loads(so)
        assert isinstance(d, dict)
    except (ValueError, AssertionError):
        d = {}
    u = d.get("usage") or {}
    rec.update(cost_usd=float(d.get("total_cost_usd") or 0.0), num_turns=d.get("num_turns"),
               usage={"input": u.get("input_tokens", 0), "output": u.get("output_tokens", 0),
                      "cache_read": u.get("cache_read_input_tokens", 0),
                      "cache_write": u.get("cache_creation_input_tokens", 0)},
               models_used=sorted((d.get("modelUsage") or {}).keys()), reply=(d.get("result") or "")[-300:],
               stop=d.get("subtype") or d.get("stop_reason"))
    text = "%s %s %s" % (d.get("result") or "", se or "", d.get("api_error_status") or "")
    calls = out / "calls.jsonl"
    rec["n_calls"] = sum(1 for _ in open(calls)) if calls.exists() else 0
    failed = (not d) or rc != 0 or bool(d.get("is_error"))
    if failed and rec["n_calls"] < budget and d.get("subtype") != "error_max_budget_usd":
        # the session broke before using its calls: an infrastructure failure, not an attempt
        rec["error"] = ("rc=%s %s" % (rc, text))[:400]
        rec["error_class"] = ("auth" if AUTH.search(text) and not C._RATE_LIMIT.search(text) else
                              "rate" if C._RATE_LIMIT.search(text) or C._ACCOUNT_REFUSED.search(text) else "other")
        return rec
    ev = out / "evals.jsonl"
    cands = [json.loads(l) for l in open(ev)] if ev.exists() else []
    rec.update(ok=True, n_evals=len(cands), best=None, val=None, test_id=None, test_ood=None)
    if cands:
        b = min(cands, key=lambda c: c["train_nmse"])
        (out / "best.py").write_text(b["code"])
        s = score(out / "best.py", pdir)
        rec["best"] = {"code": b["code"], "train_nmse": b["train_nmse"], "call": b["call"]}
        if s.get("ok"):
            rec.update(val=s["evals"]["val"], test_id=s["evals"]["test_id"], test_ood=s["evals"]["test_ood"])
    return rec


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--problem", required=True)
    ap.add_argument("--model", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--budget", type=int, default=14)
    a = ap.parse_args()
    r = run_attempt(a.problem, a.model, a.out, a.budget)
    (Path(a.out) / "record.json").write_text(json.dumps(r, indent=1))
    print(json.dumps({k: v for k, v in r.items() if k not in ("best", "reply")}, indent=1))
    if r.get("best"):
        print(r["best"]["code"])
