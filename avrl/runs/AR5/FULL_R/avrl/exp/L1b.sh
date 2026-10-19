#!/bin/bash
# L1b full run (CPU only): 12 (label set, split) jobs in parallel, then the summary. Usage: bash exp/L1b.sh
cd /homes/rdubash/avrl_live/FULL_R/avrl && mkdir -p runs/L1b
rm -f runs/L1b/done.flag runs/L1b/exit_code
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
PY=/homes/rdubash/active-audit/.venv/bin/python
CH=agreement,n_distinct,mean_tokens,std_tokens,any_truncated,log_qlen; CJ=$CH,judge_top,judge_margin,judge_mean,judge_std
(for s in 0 1 2 3; do echo "E01 proxy_error $CH $s"; echo "Qwen2.5-3B-Instruct judge_error $CJ $s"; echo "Qwen2.5-7B-Instruct-AWQ judge_error $CJ $s"; done) | \
  xargs -P 12 -L1 bash -c "$PY exp/l1b_sim_alloc.py runs/E04/feats_\$0.parquet \$1 cell100 \$2 runs/L1b/sim_\$0_split\$3.json \$3 > runs/L1b/sim_\$0_split\$3.log 2>&1"
rc=$?
if [ $rc -eq 0 ]; then $PY exp/l1b_summarize.py runs/L1b > runs/L1b/summary.log 2>&1; rc=$?; fi
echo $rc > runs/L1b/exit_code; touch runs/L1b/done.flag
