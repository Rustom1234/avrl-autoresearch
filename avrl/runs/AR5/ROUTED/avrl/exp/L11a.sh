#!/bin/bash
# L11a runbook. CPU only, runs directly in the shell (no GPU job, nothing launched through bin/gpu_run.sh, no ssh, no pip).
# L10a table-pooled posterior (frozen 200-label config B table as prior counts, k0 50, greedy online, scores refit every
# R = 40 verified labels, no logistic model) on the 1,319 registered L4b test rows (pids 0 to 1318).
# 10% budget, 20 p0 subsets, split seeds 0 to 7, 8 tie-break seeds. Comparator: the frozen 200-label table.
# Writes runs/L11a/run.log, result.json, table.md, exit_code and done.flag.
R=/homes/rdubash/avrl_live/ROUTED
RUN=$R/avrl/runs/L11a
PY=/homes/rdubash/active-audit/.venv/bin/python

mkdir -p "$RUN"
rm -f "$RUN/done.flag" "$RUN/exit_code"
cd "$R" && "$PY" -I avrl/exp/l11a_table_pooled_test_rows.py \
  --fit_feats avrl/runs/p0/feats.parquet --eval_feats avrl/runs/L4b/feats.parquet --eval_pid_max 1318 \
  --budget 0.10 --k0 50 --subsets 20 --subset_size 200 --splits 0-7 --tie_seeds 8 --out avrl/runs/L11a > "$RUN/run.log" 2>&1
echo $? > "$RUN/exit_code"
touch "$RUN/done.flag"
cat "$RUN/exit_code"
