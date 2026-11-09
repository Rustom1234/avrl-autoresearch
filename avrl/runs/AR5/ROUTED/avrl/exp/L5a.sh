#!/bin/bash
# L5a runbook. CPU only, runs directly in the shell (no GPU job, nothing launched through bin/gpu_run.sh).
# Frozen-table test: config B (stratum agreement x n_distinct x any_truncated, k0 50, 10%) with the table fitted once on
# the p0 pool and never refit. Ranks the L2b and L4b sim docs by the frozen score and compares with plain fixed.
# Writes runs/L5a/run.log, result.json, table.md, exit_code and done.flag.
R=/homes/rdubash/avrl_live/ROUTED
RUN=$R/avrl/runs/L5a
PY=/homes/rdubash/active-audit/.venv/bin/python

mkdir -p "$RUN"
rm -f "$RUN/done.flag" "$RUN/exit_code"
"$PY" -I "$R/avrl/exp/l5a_frozen_table.py" \
  --fit_feats "$R/avrl/runs/p0/feats.parquet" \
  --eval_feats "$R/avrl/runs/L2b/feats.parquet" "$R/avrl/runs/L4b/feats.parquet" \
  --budget 0.10 --k0 50 --floor 40 --splits 0-7 --tie_seeds 8 --out "$RUN" > "$RUN/run.log" 2>&1
echo $? > "$RUN/exit_code"
touch "$RUN/done.flag"
cat "$RUN/exit_code"
