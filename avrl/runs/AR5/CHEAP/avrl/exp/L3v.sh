#!/usr/bin/env bash
# L3v runner. CPU only, local shell: no GPU, no gpu_run.sh, no ssh, no pip.
# Runs the L3s design (avrl/exp/l3v_null_calibration.py) on the 20 null pools, then the aggregate.
# Per pool: runs/L3v/null_k/run.log and result.json. Top level: exit_code and done.flag (written last).
BASE=/homes/rdubash/avrl_live/CHEAP/avrl
PY=/homes/rdubash/active-audit/.venv/bin/python
OUT=$BASE/runs/L3v
mkdir -p "$OUT"
rm -f "$OUT/done.flag" "$OUT/exit_code" "$OUT/pool_exits.log"
rc=0
for k in $(seq 0 19); do
  d="$OUT/null_$k"
  mkdir -p "$d"
  timeout 2700 "$PY" -B "$BASE/exp/l3v_null_calibration.py" \
    --pool "$OUT/pools/null_$k.parquet" --budgets 0.25 --seeds 8 --boot 200 --boot_budget 0.25 --boot_seeds 4 \
    --out "$d" > "$d/run.log" 2>&1
  r=$?
  echo "null_$k exit $r" >> "$OUT/pool_exits.log"
  # exit 3 is a recorded INVALID_CONTROL verdict; any other non-zero exit (crash, or 124 at the 45-minute cap) is a failed pool
  if [ $r -ne 0 ] && [ $r -ne 3 ]; then rc=1; fi
done
timeout 900 "$PY" -B "$BASE/exp/l3v_aggregate.py" > "$OUT/aggregate.log" 2>&1
ra=$?
if [ $ra -ne 0 ]; then rc=1; fi
echo "$rc" > "$OUT/exit_code"
touch "$OUT/done.flag"
exit $rc
