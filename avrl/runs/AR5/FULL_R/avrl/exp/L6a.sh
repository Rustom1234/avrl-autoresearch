#!/bin/bash
# L6a full run (CPU only): 64 sweep jobs (2 judges x per-step B in 62, 125 x order seeds 0..15), then the summary (regression gate vs runs/L5a on order seeds 0..7 lives in the summarizer; exit 3 = void).
# Launch: bin/gpu_run.sh runs/L6a 'bash exp/L6a.sh'. Does NOT write done.flag or exit_code: bin/gpu_run.sh's wrapper writes both.
cd /homes/rdubash/avrl_live/FULL_R/avrl && mkdir -p runs/L6a
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
PY=/homes/rdubash/active-audit/.venv/bin/python
CJ=agreement,n_distinct,mean_tokens,std_tokens,any_truncated,log_qlen,judge_top,judge_margin,judge_mean,judge_std
J3=Qwen2.5-3B-Instruct; J7=Qwen2.5-7B-Instruct-AWQ
(for j in $J3 $J7; do for B in 62 125; do for o in $(seq 0 15); do echo "$j $B $o"; done; done; done) | \
  xargs -P 24 -L1 bash -c "$PY exp/l6a_amort.py runs/E04/feats_\$0.parquet judge_error $CJ \$1 \$2 runs/L6a/amort_\$0_B\$1_o\$2.json > runs/L6a/amort_\$0_B\$1_o\$2.log 2>&1"
rc=$?
$PY exp/l6a_summarize.py runs/L6a runs/L5a > runs/L6a/summary.log 2>&1; src=$?
[ $rc -ne 0 ] && exit $rc
exit $src
