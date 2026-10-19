#!/usr/bin/env bash
# L3c launcher. CPU only, runs directly in the shell (no GPU, so no gpu_run.sh).
# Writes avrl/runs/L3c/run.log, exit_code and done.flag; result.json is written by the script itself.
cd /homes/rdubash/avrl_live/CASCADE || exit 1
mkdir -p avrl/runs/L3c
/homes/rdubash/active-audit/.venv/bin/python avrl/exp/l3c_crossfit_ranker.py --feats avrl/runs/p0/feats.parquet --out avrl/runs/L3c > avrl/runs/L3c/run.log 2>&1
echo $? > avrl/runs/L3c/exit_code
touch avrl/runs/L3c/done.flag
cat avrl/runs/L3c/exit_code
