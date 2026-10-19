#!/usr/bin/env bash
# L3z runner (evaluation). Launch ONLY through bin/gpu_run.sh, which enforces the 45-minute cap and writes
# runs/L3z/exit_code and runs/L3z/done.flag. Refuses to run unless the pilot status is OK (pre-registered abort rule).
# Per pool: runs/L3z/planted_k/run.log and result.json (POOL_SEED 40000 + k, k = 0..19, strength from the pilot).
# Aggregate: runs/L3z/result.json, runs/L3z/summary.json, runs/L3z/aggregate.log.
BASE=/homes/rdubash/avrl_live/CHEAP/avrl
PY=/homes/rdubash/active-audit/.venv/bin/python
OUT=$BASE/runs/L3z
STRENGTH=$("$PY" -B -c 'import json,sys; r=json.load(open(sys.argv[1])); assert r["status"]=="OK", r["status"]; print(r["chosen_strength"])' "$OUT/pilot/result.json") || { echo "pilot status is not OK; nothing launched" >&2; exit 1; }
echo "planted strength s=$STRENGTH (from $OUT/pilot/result.json)"
mkdir -p "$OUT"
rm -f "$OUT/pool_exits.log"
rc=0
for k in $(seq 0 19); do
  d="$OUT/planted_$k"
  mkdir -p "$d"
  timeout 2700 "$PY" -B "$BASE/exp/l3z_planted_power.py" --pool_seed $((40000 + k)) --strength "$STRENGTH" \
    --budgets 0.25 --seeds 8 --boot 200 --boot_budget 0.25 --boot_seeds 4 --out "$d" > "$d/run.log" 2>&1
  r=$?
  echo "planted_$k exit $r" >> "$OUT/pool_exits.log"
  # exit 3 is a recorded INVALID_CONTROL verdict; any other non-zero exit (crash, or 124 at the 45-minute cap) is a failed pool
  if [ $r -ne 0 ] && [ $r -ne 3 ]; then rc=1; fi
done
timeout 900 "$PY" -B "$BASE/exp/l3z_aggregate.py" > "$OUT/aggregate.log" 2>&1
ra=$?
if [ $ra -ne 0 ]; then rc=1; fi
exit $rc
