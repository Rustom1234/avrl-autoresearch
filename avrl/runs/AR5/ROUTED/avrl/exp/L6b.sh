#!/bin/bash
# L6b runbook. CPU only, runs directly in the shell (no GPU job, nothing launched through bin/gpu_run.sh).
# Frozen config B (stratum agreement x n_distinct x any_truncated, k0 50, 10%) with the table fitted on 200 p0 prompts,
# 20 subsets, each frozen and never refit. Evaluated on L2b and L4b against plain fixed on the same splits and seeds.
# Writes runs/L6b/run.log, result.json, table.md, exit_code and done.flag.
R=/homes/rdubash/avrl_live/ROUTED
RUN=$R/avrl/runs/L6b
PY=/homes/rdubash/active-audit/.venv/bin/python

mkdir -p "$RUN"
rm -f "$RUN/done.flag" "$RUN/exit_code"
"$PY" -I "$R/avrl/exp/l6b_frozen_label_budget.py" \
  --fit_feats "$R/avrl/runs/p0/feats.parquet" \
  --eval_feats "$R/avrl/runs/L2b/feats.parquet" "$R/avrl/runs/L4b/feats.parquet" \
  --budget 0.10 --k0 50 --subsets 20 --subset_size 200 --splits 0-7 --tie_seeds 8 --out "$RUN" > "$RUN/run.log" 2>&1
echo $? > "$RUN/exit_code"
touch "$RUN/done.flag"
cat "$RUN/exit_code"
