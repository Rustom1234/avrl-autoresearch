#!/bin/bash
# L3a: CPU replay only (no GPU, no gpu_run.sh). Order: L1B pool first, because its reproduction gate (Gate B) must pass
# before the fresh-pool verdict is read. Each pool writes its own run.log, result.json and table.md; the summary writes
# runs/L3a/result.json. exit_code and done.flag are written last.
PY=/homes/rdubash/active-audit/.venv/bin/python
ROOT=/homes/rdubash/avrl_live/ROUTED
EXP=$ROOT/avrl/exp
RUN=$ROOT/avrl/runs/L3a
mkdir -p "$RUN/fresh" "$RUN/l1b_pool"
rm -f "$RUN/done.flag" "$RUN/exit_code"

$PY -I "$EXP/l3a_floor_sensitivity.py" --feats "$ROOT/avrl/runs/p0/feats.parquet" \
    --out "$RUN/l1b_pool/replay" --result_dir "$RUN/l1b_pool" > "$RUN/l1b_pool/run.log" 2>&1
c1=$?
c2=0
if [ $c1 -eq 0 ]; then
  $PY -I "$EXP/l3a_floor_sensitivity.py" --feats "$ROOT/avrl/runs/L2b/feats.parquet" \
      --out "$RUN/fresh/replay" --result_dir "$RUN/fresh" > "$RUN/fresh/run.log" 2>&1
  c2=$?
fi
$PY -I "$EXP/l3a_summary.py" --fresh "$RUN/fresh/result.json" --l1b "$RUN/l1b_pool/result.json" \
    --out "$RUN/result.json" > "$RUN/summary.log" 2>&1
c3=$?
code=0
for c in $c1 $c2 $c3; do [ "$code" -eq 0 ] && code=$c; done
echo "$code" > "$RUN/exit_code"
cat "$RUN/l1b_pool/run.log" "$RUN/fresh/run.log" "$RUN/summary.log" 2>/dev/null > "$RUN/run.log"
touch "$RUN/done.flag"
echo "L3a finished: l1b_pool exit $c1, fresh exit $c2, summary exit $c3"
