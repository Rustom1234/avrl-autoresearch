#!/bin/bash
# L8a full run (CPU only): 128 jobs (direction 1 = 3B then 7B-AWQ, direction 2 = 7B-AWQ then 3B; per-step B in 62, 125; order seeds 0..15; mode flip = relation flip f = -1 on judge_top / judge_mean z of every post-switch row,
# mode gate = f = +1, which must reproduce runs/L7a drift exactly), then the summary (regression gate vs runs/L7a lives in the summarizer; exit 3 = void).
# Launch: bin/gpu_run.sh runs/L8a 'bash exp/L8a.sh'. Does NOT write done.flag or exit_code: bin/gpu_run.sh's wrapper writes both.
cd /homes/rdubash/avrl_live/FULL_R/avrl && mkdir -p runs/L8a
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
export PY=/homes/rdubash/active-audit/.venv/bin/python
export CJ=agreement,n_distinct,mean_tokens,std_tokens,any_truncated,log_qlen,judge_top,judge_margin,judge_mean,judge_std
export F3=runs/E04/feats_Qwen2.5-3B-Instruct.parquet F7=runs/E04/feats_Qwen2.5-7B-Instruct-AWQ.parquet
(for B in 125 62; do for d in 1 2; do for m in flip gate; do for o in $(seq 0 15); do echo "$d $B $o $m"; done; done; done; done) | \
  xargs -P 24 -L1 bash -c 'if [ $0 = 1 ]; then FA=$F3; FB=$F7; else FA=$F7; FB=$F3; fi; if [ $3 = flip ]; then F=-1; else F=1; fi
    $PY exp/l8a_flip.py $FA $FB judge_error $CJ $1 $2 drift $F runs/L8a/l8a_d$0_B$1_o$2_$3.json > runs/L8a/l8a_d$0_B$1_o$2_$3.log 2>&1'
rc=$?
$PY exp/l8a_summarize.py runs/L8a runs/L7a > runs/L8a/summary.log 2>&1; src=$?
[ $rc -ne 0 ] && exit $rc
exit $src
