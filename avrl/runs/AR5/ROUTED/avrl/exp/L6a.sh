#!/bin/bash
# L6a runbook. CPU only, runs directly in the shell (no GPU job, nothing launched through bin/gpu_run.sh).
# Frozen config B (stratum agreement x n_distinct x any_truncated, k0 50, 10%) on the 1,319 registered L4b test rows
# (pids 0 to 1318). Judged tables: the 3,000-label table fitted on all p0 prompts, and 20 frozen 200-label tables.
# Context only: 1,473 remaining-train rows and 64 random 1,319-prompt halves of L4b.
# Writes runs/L6a/run.log, result.json, table.md, exit_code and done.flag.
R=/homes/rdubash/avrl_live/ROUTED
RUN=$R/avrl/runs/L6a
PY=/homes/rdubash/active-audit/.venv/bin/python

mkdir -p "$RUN"
rm -f "$RUN/done.flag" "$RUN/exit_code"
cd "$R" && "$PY" -I "$R/avrl/exp/l6a_test_rows.py" \
  --subsets 20 --subset_size 200 --splits 0-7 --tie_seeds 8 --eval_pid_max 1318 --out "$RUN" > "$RUN/run.log" 2>&1
echo $? > "$RUN/exit_code"
touch "$RUN/done.flag"
cat "$RUN/exit_code"
