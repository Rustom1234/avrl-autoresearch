#!/usr/bin/env bash
# L10a: bootstrap CI on greedy_online vs static_matched (and two more comparisons) for E06 and E08.
# Runs avrl/exp/L10a_bootstrap_e06_e08.py on each point, then combines both into runs/L10a/result.json.
# CPU-only. Run from /homes/rdubash/active-audit.
set -euo pipefail
cd /homes/rdubash/active-audit

PY=.venv/bin/python
OUT=avrl/runs/L10a
mkdir -p "$OUT"

CHEAP6="agreement,n_distinct,mean_tokens,std_tokens,any_truncated,log_qlen"
CHEAP8="${CHEAP6},frac_syntax_error,frac_exception"

echo "=== L10a E06 bootstrap (cheap6, cell100, n_boot=300) ==="
$PY avrl/exp/L10a_bootstrap_e06_e08.py \
  --feats avrl/runs/E06/feats.parquet \
  --cheap "$CHEAP6" \
  --group_col cell100 --n_boot 300 --seed 0 \
  --out avrl/runs/E06/bootstrap.json

echo "=== L10a E08 bootstrap (cheap8, cell100, n_boot=300) ==="
$PY avrl/exp/L10a_bootstrap_e06_e08.py \
  --feats avrl/runs/E08/feats.parquet \
  --cheap "$CHEAP8" \
  --group_col cell100 --n_boot 300 --seed 0 \
  --out avrl/runs/E08/bootstrap.json

echo "=== L10a combine ==="
$PY - <<'EOF'
import json

def load(p):
    return json.load(open(p))

e06 = load("avrl/runs/E06/bootstrap.json")
e08 = load("avrl/runs/E08/bootstrap.json")

def find(d, bf, a, b):
    for c in d["comparisons"]:
        if c["budget_frac"] == bf and c["policy_a"] == a and c["policy_b"] == b:
            return c
    return None

# pass_threshold: CI on the 25% relative gain (greedy_online vs static_matched) excludes zero on the
# positive side for either E06 or E08.
# kill_threshold: CI includes zero at every budget (10/25/50%) for both E06 and E08.
budgets = [0.10, 0.25, 0.50]
per_point = {}
for name, d in (("E06", e06), ("E08", e08)):
    rows = {}
    for bf in budgets:
        c = find(d, bf, "greedy_online", "static_matched")
        lo, hi = c["rel_improve_ci95"]
        rows[bf] = {"rel_improve_point": c["rel_improve_point"], "rel_improve_ci95": [lo, hi],
                    "hits_a_greedy_online": c["hits_a"], "hits_b_static_matched": c["hits_b"],
                    "excludes_zero_positive": lo is not None and lo > 0}
    per_point[name] = rows

ci25_e06 = per_point["E06"][0.25]["rel_improve_ci95"]
ci25_e08 = per_point["E08"][0.25]["rel_improve_ci95"]
pass_e06 = per_point["E06"][0.25]["excludes_zero_positive"]
pass_e08 = per_point["E08"][0.25]["excludes_zero_positive"]
all_include_zero = all(not per_point[n][bf]["excludes_zero_positive"] for n in ("E06", "E08") for bf in budgets)

if pass_e06 or pass_e08:
    verdict = "PASS"
elif all_include_zero:
    verdict = "KILL"
else:
    verdict = "INCONCLUSIVE"

out = {
    "id": "L10a",
    "hypothesis": "greedy_online vs static_matched point estimates on E06/E08 (sim_cell100_cheap.json) are a real effect, not resampling noise",
    "pass_threshold": "CI on the 25% relative gain (greedy_online vs static_matched) excludes zero on the positive side for either E06 or E08",
    "kill_threshold": "CI includes zero at every budget (10/25/50%) for both E06 and E08",
    "per_point_25pct_budget": {"E06": per_point["E06"][0.25], "E08": per_point["E08"][0.25]},
    "per_point_all_budgets": per_point,
    "ci25_e06": ci25_e06,
    "ci25_e08": ci25_e08,
    "verdict": verdict,
    "e06_bootstrap_file": "avrl/runs/E06/bootstrap.json",
    "e08_bootstrap_file": "avrl/runs/E08/bootstrap.json",
    "e06_n_groups": e06["n_groups"],
    "e08_n_groups": e08["n_groups"],
    "e06_runtime_sec": e06["runtime_sec"],
    "e08_runtime_sec": e08["runtime_sec"],
}
json.dump(out, open("avrl/runs/L10a/result.json", "w"), indent=1)
print(json.dumps(out, indent=1))
EOF

touch avrl/runs/L10a/done.flag
echo "L10a done"
