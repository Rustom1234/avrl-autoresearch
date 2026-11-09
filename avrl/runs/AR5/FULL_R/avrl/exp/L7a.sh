#!/bin/bash
# L7a full run (CPU only): 192 jobs (direction 1 = 3B then 7B-AWQ, direction 2 = 7B-AWQ then 3B; per-step B in 62, 125; order seeds 0..15; modes drift, control_exact, control_z), then the summary
# (regression gate vs runs/L6a on order seeds 0..7 lives in the summarizer; exit 3 = void).
# Launch: bin/gpu_run.sh runs/L7a 'bash exp/L7a.sh'. Does NOT write done.flag or exit_code: bin/gpu_run.sh's wrapper writes both.
cd /homes/rdubash/avrl_live/FULL_R/avrl && mkdir -p runs/L7a
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
export PY=/homes/rdubash/active-audit/.venv/bin/python
export CJ=agreement,n_distinct,mean_tokens,std_tokens,any_truncated,log_qlen,judge_top,judge_margin,judge_mean,judge_std
export F3=runs/E04/feats_Qwen2.5-3B-Instruct.parquet F7=runs/E04/feats_Qwen2.5-7B-Instruct-AWQ.parquet
(for B in 125 62; do for d in 1 2; do for m in drift control_exact control_z; do for o in $(seq 0 15); do echo "$d $B $o $m"; done; done; done; done) | \
  xargs -P 24 -L1 bash -c 'if [ $0 = 1 ]; then FA=$F3; FB=$F7; else FA=$F7; FB=$F3; fi; [ $3 = drift ] || FB=same
    $PY exp/l7a_drift.py $FA $FB judge_error $CJ $1 $2 $3 runs/L7a/l7a_d$0_B$1_o$2_$3.json > runs/L7a/l7a_d$0_B$1_o$2_$3.log 2>&1'
rc=$?
$PY exp/l7a_summarize.py runs/L7a runs/L6a > runs/L7a/summary.log 2>&1; src=$?
[ $rc -ne 0 ] && exit $rc
exit $src
