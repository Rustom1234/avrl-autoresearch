#!/usr/bin/env python3
"""Control for S1: independent single-shot sampling with no loop (the IID baseline of Gideoni, Risi and Gal).

    python loop/shinka/iid.py --root loop/runs/S1_IID --model haiku --n 20 [--par 5]

Each sample: one model call that sees the same program card and starting program as the loops and returns a whole
new train.py; the file is scored once by the same harness. No sample sees another sample or any result.
"""
import argparse, json, re, subprocess, sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.argv, argv = sys.argv[:1], sys.argv
import importlib.util   # noqa: E402
_s = importlib.util.spec_from_file_location("shinka_task", HERE / "task.py")   # loads the bench task with the pack26s card
st = importlib.util.module_from_spec(_s)
_s.loader.exec_module(st)
ap = argparse.ArgumentParser()
ap.add_argument("--root", required=True)
ap.add_argument("--model", required=True)
ap.add_argument("--n", type=int, default=20)
ap.add_argument("--par", type=int, default=5)
a = ap.parse_args(argv[1:])
c = json.loads((HERE / "consts.json").read_text())["pack26s"]
card = st.bt.PACK % dict(n=26, sigma=c["sigma_used"], score=-c["anchor"], ref=2.635983)
prompt = (card + "\n## Current train.py\n```python\n" + (HERE / "initial_shinka.py").read_text() + "\n```\n\n## Task\n"
          "Write the best complete replacement for train.py that you can. You get one attempt and will not see its "
          "result. Reply with the whole file in a single ```python code block and nothing else.")
root = Path(a.root)


def one(i):
    d = root / ("%s_%02d" % (a.model, i))
    if (d / "result.json").exists():
        return json.load(open(d / "result.json"))
    d.mkdir(parents=True, exist_ok=True)
    p = subprocess.run(["claude", "-p", "--model", a.model, "--output-format", "json"], input=prompt, capture_output=True, text=True, timeout=1800)
    try:
        out = json.loads(p.stdout)
    except ValueError:
        out = {"result": "", "error": p.stdout[-300:] + p.stderr[-300:]}
    (d / "call.json").write_text(json.dumps({k: out.get(k) for k in ("total_cost_usd", "usage", "is_error", "error", "duration_ms")}))
    m = re.findall(r"```python\n(.*?)```", out.get("result") or "", re.S)
    if not m:
        (d / "result.json").write_text(json.dumps({"status": "no_code", "cost": out.get("total_cost_usd")}))
        return {"status": "no_code"}
    (d / "train.py").write_text(max(m, key=len))
    subprocess.run([st.bt.PY, str(HERE / "harness.py"), str(d / "train.py"), "--out", str(d), "--timeout", "300"], timeout=400)
    r = json.load(open(d / "result.json"))
    r["cost"] = out.get("total_cost_usd")
    (d / "result.json").write_text(json.dumps(r))
    return r


with ThreadPoolExecutor(a.par) as ex:
    res = list(ex.map(one, range(a.n)))
sums = sorted((r.get("sum") or 0.0 for r in res), reverse=True)
print(a.model, "samples", len(res), "ok", sum(r.get("status") == "ok" for r in res), "best %.7f" % sums[0], "at or above 2.635977:", sum(s >= 2.635977 for s in sums),
      "at or above 2.6359:", sum(s >= 2.6359 for s in sums), "median %.5f" % sums[len(sums) // 2], "cost USD %.2f" % sum(r.get("cost") or 0 for r in res))
print("all sums:", " ".join("%.5f" % s for s in sums))
