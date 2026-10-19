#!/bin/bash
# L2a full run (CPU only): 8 (label set, split) jobs in parallel, then the summary. Launch: bin/gpu_run.sh runs/L2a 'bash exp/L2a.sh'
# Does NOT write done.flag or exit_code: bin/gpu_run.sh's wrapper writes both.
cd /homes/rdubash/avrl_live/FULL_R/avrl && mkdir -p runs/L2a
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
PY=/homes/rdubash/active-audit/.venv/bin/python
CJ=agreement,n_distinct,mean_tokens,std_tokens,any_truncated,log_qlen,judge_top,judge_margin,judge_mean,judge_std
(for s in 0 1 2 3; do echo "Qwen2.5-3B-Instruct $s"; echo "Qwen2.5-7B-Instruct-AWQ $s"; done) | \
  xargs -P 8 -L1 bash -c "$PY exp/l2a_sim_alloc.py runs/E04/feats_\$0.parquet judge_error cell100 $CJ runs/L2a/sim_\$0_split\$1.json \$1 > runs/L2a/sim_\$0_split\$1.log 2>&1"
rc=$?
if [ $rc -eq 0 ]; then $PY exp/l2a_summarize.py runs/L2a > runs/L2a/summary.log 2>&1; rc=$?; fi
exit $rc
