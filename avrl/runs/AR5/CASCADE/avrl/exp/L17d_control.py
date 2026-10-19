"""L17d control (CPU only): the copy must reproduce avrl/runs/L14b/result.json on every arm it shares with L14b.

Compares, to 1e-9: the budget, caught mean, caught sd, admitted and n_runs of the six L14b arms at 10, 25 and 50%; the point D
of ts_online and greedy_online; the pool counts and budget docs; and the 200 cluster and 200 group bootstrap draws and CIs of
D(ts_online) and D(greedy_online) at 25%. The new static arms and the L17d metrics are not compared here (they are checked by
avrl/exp/L17d_check.py). Also gates on the L14b reference values: static caught 938.0 at 25%, threshold admitted 0.519 at 25%,
and D(ts_online) at 25% = -0.077. Writes avrl/runs/L17d/control.json and exits 1 on any mismatch or failed gate.
"""
import json
import sys

CASCADE = "/homes/rdubash/avrl_live/CASCADE"
L14B = f"{CASCADE}/avrl/runs/L14b/result.json"
L17D = f"{CASCADE}/avrl/runs/L17d/result.json"
OUT = f"{CASCADE}/avrl/runs/L17d/control.json"
TOL = 1e-9
ARMS = ["uniform", "static", "threshold", "threshold_input_order", "greedy_online", "ts_online"]
BUDGETS = ["0.1", "0.25", "0.5"]
KEYS = ["budget", "caught_mean", "caught_sd", "admitted", "n_runs"]

a = json.load(open(L14B))
b = json.load(open(L17D))
checked, bad = 0, []


def cmp(name, x, y):
    global checked
    checked += 1
    if abs(float(x) - float(y)) > TOL:
        bad.append({"name": name, "L14b": x, "L17d": y})


for bf in BUDGETS:
    for arm in ARMS:
        for k in KEYS:
            cmp(f"point/{bf}/{arm}/{k}", a["point"][bf][arm][k], b["point"][bf][arm][k])
    for arm in ["ts_online", "greedy_online"]:
        cmp(f"D_point/{bf}/{arm}", a["D_point"][bf][arm], b["D_point"][bf][arm])
    cmp(f"budget_docs/{bf}", a["pool"]["budget_docs"][bf], b["pool"]["budget_docs"][bf])

for k in ["n_sim", "n_buckets", "total_errors"]:
    cmp(f"pool/{k}", a["pool"][k], b["pool"][k])

for arm in ["ts_online", "greedy_online"]:
    for name in ["D_bootstrap_cluster_draws", "D_bootstrap_group_draws"]:
        xs, ys = a[name][arm], b[name][arm]
        cmp(f"{name}/{arm}/length", len(xs), len(ys))
        for i, (x, y) in enumerate(zip(xs, ys)):
            cmp(f"{name}/{arm}/{i}", x, y)
    for name in ["D_bootstrap_cluster", "D_bootstrap_group"]:
        for i, (x, y) in enumerate(zip(a[name][arm]["ci95"], b[name][arm]["ci95"])):
            cmp(f"{name}/{arm}/ci95/{i}", x, y)

thr_adm = b["point"]["0.25"]["threshold"]["admitted"]
static_caught = b["point"]["0.25"]["static"]["caught_mean"]
d_ts = b["D_point"]["0.25"]["ts_online"]
gate = round(thr_adm, 3) == 0.519 and static_caught == 938.0 and round(d_ts, 3) == -0.077

out = {
    "exp": "L17d control",
    "compared": checked,
    "tolerance": TOL,
    "mismatches": len(bad),
    "mismatch_examples": bad[:20],
    "static_caught_25": static_caught,
    "threshold_admitted_25": thr_adm,
    "D_ts_online_25": d_ts,
    "gate_938_0_and_0.519_and_-0.077": gate,
    "passed": len(bad) == 0 and gate,
}
with open(OUT, "w") as f:
    json.dump(out, f, indent=1)
print(json.dumps({k: out[k] for k in ["compared", "mismatches", "gate_938_0_and_0.519_and_-0.077", "passed"]}))
sys.exit(0 if out["passed"] else 1)
