#!/usr/bin/env bash
# L3f CPU runner (step 4). Local shell only: no GPU, no gpu_run.sh, no ssh, no pip.
# Runs the unchanged E03 script on the fresh pool, then the L3f aggregate (exp/L3f.py).
# Log: runs/L3f/run.log. Outputs: runs/L3f/e03/result.json, runs/L3f/result.json, then exit_code and done.flag.
OUT=/homes/rdubash/avrl_live/CHEAP/avrl/runs/L3f
PY=/homes/rdubash/active-audit/.venv/bin/python
mkdir -p "$OUT"
rm -f "$OUT/done.flag" "$OUT/exit_code"
timeout 2700 "$PY" -B /homes/rdubash/avrl_live/CHEAP/avrl/exp/e03_real_allocators.py \
  --pool "$OUT/pool/feats.parquet" --budgets 0.25 --seeds 8 --boot 100 --boot_budget 0.25 --boot_seeds 4 \
  --out "$OUT/e03" > "$OUT/run.log" 2>&1
rc=$?
timeout 600 "$PY" -B /homes/rdubash/avrl_live/CHEAP/avrl/exp/L3f.py >> "$OUT/run.log" 2>&1
arc=$?
[ "$rc" -eq 0 ] && rc=$arc
echo "$rc" > "$OUT/exit_code"
touch "$OUT/done.flag"
exit "$rc"
