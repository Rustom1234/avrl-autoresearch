"""L14b code check (CPU only): the copied readout must reproduce E03 on the E03 pool.

Runs avrl/exp/l14b_claim2_readout.py on avrl/runs/p0/feats.parquet (the E03 pool) with the E03 arguments, into
avrl/runs/L14b/code_check_E03/, then compares with avrl/runs/E03/result.json. Pass: pool counts equal; caught mean and admitted
equal for every arm and budget; the 200 group-bootstrap draws of D(ts_online) and D(greedy_online) and their 95% intervals equal,
all to 1e-9. Writes code_check_E03/compare.json and exits 1 on any mismatch.

Usage: /homes/rdubash/active-audit/.venv/bin/python avrl/exp/l14b_code_check.py
"""
import os
import sys
import json
import subprocess

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(os.path.dirname(HERE))
READOUT = os.path.join(HERE, "l14b_claim2_readout.py")
POOL = os.path.join(ROOT, "avrl", "runs", "p0", "feats.parquet")
E03 = os.path.join(ROOT, "avrl", "runs", "E03", "result.json")
OUT = os.path.join(ROOT, "avrl", "runs", "L14b", "code_check_E03")
TOL = 1e-9

os.makedirs(OUT, exist_ok=True)
cmd = [sys.executable, READOUT, "--feats", POOL, "--cell", "cell100", "--features", "cheap+emb", "--budgets", "0.10,0.25,0.50",
       "--seeds", "8", "--boot", "200", "--boot_budget", "0.25", "--out", OUT]
run = subprocess.run(cmd, capture_output=True, text=True)
open(os.path.join(OUT, "run.log"), "w").write(run.stdout + run.stderr)
if run.returncode != 0:
    print(f"readout copy exited {run.returncode}; see {os.path.join(OUT, 'run.log')}")
    sys.exit(1)

ref = json.load(open(E03))
got = json.load(open(os.path.join(OUT, "result.json")))
res = {"pool_counts_equal": {k: ref["pool"][k] == got["pool"][k] for k in ["n_sim", "n_buckets", "total_errors"]}}
diffs = {}
for bf in ref["point"]:
    for arm in ref["point"][bf]:
        for key in ["caught_mean", "admitted"]:
            diffs[f"point/{bf}/{arm}/{key}"] = abs(ref["point"][bf][arm][key] - got["point"][bf][arm][key])
for arm in ["ts_online", "greedy_online"]:
    a = np.asarray(ref["D_bootstrap_draws"][arm], float)
    b = np.asarray(got["D_bootstrap_group_draws"][arm], float)
    diffs[f"group_draws/{arm}/max_abs"] = float(np.max(np.abs(a - b))) if a.shape == b.shape else float("inf")
    diffs[f"group_ci/{arm}"] = float(np.max(np.abs(np.asarray(ref["D_bootstrap"][arm]["ci95"]) - np.asarray(got["D_bootstrap_group"][arm]["ci95"]))))
    diffs[f"D_point/{arm}"] = max(abs(ref["D_point"][bf][arm] - got["D_point"][bf][arm]) for bf in ref["D_point"])
res["max_abs_diff"] = {k: float(v) for k, v in diffs.items()}
res["n_group_draws"] = {"E03": len(ref["D_bootstrap_draws"]["ts_online"]), "copy": len(got["D_bootstrap_group_draws"]["ts_online"])}
res["passed"] = bool(all(res["pool_counts_equal"].values()) and max(diffs.values()) <= TOL)
res["E03_verdict"] = ref["verdict"]
res["copy_verdict"] = got["verdict"]
json.dump(res, open(os.path.join(OUT, "compare.json"), "w"), indent=1)
print(json.dumps(res, indent=1))
print("CODE CHECK PASSED" if res["passed"] else "CODE CHECK FAILED", flush=True)
sys.exit(0 if res["passed"] else 1)
