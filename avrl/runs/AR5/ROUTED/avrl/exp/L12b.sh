#!/bin/bash
# L12b runbook. CPU only, runs directly in the shell (no GPU job, nothing launched through bin/gpu_run.sh, no ssh, no pip).
# Does the frozen 200-label table's stratum ordering (36 p0 strata: agreement x n_distinct x any_truncated) transfer
# across the two L4b row sets: registered test rows (pid <= 1318) and remaining-train rows (pid > 1318)?
# Primary: Spearman rho of per-stratum proxy_error rates, pass above 0.8, kill below 0.5, 1,000-draw bootstrap CI.
R=/homes/rdubash/avrl_live/ROUTED
RUN=$R/avrl/runs/L12b
PY=/homes/rdubash/active-audit/.venv/bin/python
mkdir -p "$RUN"; rm -f "$RUN/done.flag" "$RUN/exit_code"
"$PY" -I "$R/avrl/exp/l12b_stratum_transfer.py" \
  --p0 "$R/avrl/runs/p0/feats.parquet" --l4b "$R/avrl/runs/L4b/feats.parquet" \
  --test_pid_max 1318 --boot 1000 --out "$RUN" > "$RUN/run.log" 2>&1
echo $? > "$RUN/exit_code"; touch "$RUN/done.flag"
cat "$RUN/exit_code"
