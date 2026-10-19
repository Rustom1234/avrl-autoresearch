#!/bin/bash
# L18a runbook. CPU only, runs directly in the local shell (no GPU job, nothing launched through bin/gpu_run.sh, no ssh, no pip).
# Learned arm greedy_strata: the L9b greedy with the scorer features CHEAP plus one 0/1 indicator per p0 stratum cell
# (36 cells), against the 3,000-label frozen table (L5a), 10% budget, pools L2b and L4b, split seeds 0 to 7, 8 tie-break seeds each.
# Then the independent recomputation from the saved hit counts (avrl/exp/l18a_check.py).
R=/homes/rdubash/avrl_live/ROUTED
RUN=$R/avrl/runs/L18a
PY=/homes/rdubash/active-audit/.venv/bin/python

mkdir -p "$RUN"; rm -f "$RUN/exit_code" "$RUN/done.flag" "$RUN/indep_check_exit_code"
"$PY" -I "$R/avrl/exp/l18a_strata_logistic_3000.py" --fit_feats "$R/avrl/runs/p0/feats.parquet" \
  --eval_feats "$R/avrl/runs/L2b/feats.parquet" "$R/avrl/runs/L4b/feats.parquet" --budget 0.10 --k0 50 \
  --splits 0-7 --tie_seeds 8 --out "$RUN" > "$RUN/run.log" 2>&1
echo $? > "$RUN/exit_code"
"$PY" -I "$R/avrl/exp/l18a_check.py" --result "$RUN/result.json" --out "$RUN" > "$RUN/indep_check.log" 2>&1
echo $? > "$RUN/indep_check_exit_code"
touch "$RUN/done.flag"
cat "$RUN/exit_code"
