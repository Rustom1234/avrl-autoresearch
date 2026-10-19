#!/bin/bash
# L9a full run (CPU only), four 64-job stages on the grid 2 judges x per-step B in 62, 125 x seeds 0..15, then the summary:
#   hz_*     iid-with-replacement streams, H = 60 batches (the verdict data)
#   gate_*   mode disjoint12 (the L7b layout), compared exactly with runs/L7b in the summarizer (regression gate; exit 3 = void)
#   nr_*     mode iid_nr, the pre-registered memorisation remedy (carry_plain's paid prompts removed from later candidate pools; the stream ends when fewer than 250 unpaid prompts remain)
#   hz120_*  the iid streams to H = 120 (extension for the ratio-versus-H slope, never enters the verdict; the 45-minute cap was judged safe from a 31 s H = 60 timing job)
# summarizer exit 4 = memorisation control void (verdict still written).
# Launch: bin/gpu_run.sh avrl/runs/L9a 'bash exp/L9a.sh'. Does NOT write done.flag or exit_code: bin/gpu_run.sh's wrapper writes both.
cd /homes/rdubash/avrl_live/FULL_R/avrl && mkdir -p runs/L9a
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
PY=/homes/rdubash/active-audit/.venv/bin/python
CJ=agreement,n_distinct,mean_tokens,std_tokens,any_truncated,log_qlen,judge_top,judge_margin,judge_mean,judge_std
J3=Qwen2.5-3B-Instruct; J7=Qwen2.5-7B-Instruct-AWQ
grid() { for j in $J3 $J7; do for B in 62 125; do for s in $(seq 0 15); do echo "$j $B $s"; done; done; done; }
grid | xargs -P 24 -L1 bash -c "$PY exp/l9a_horizon.py runs/E04/feats_\$0.parquet judge_error $CJ \$1 \$2 iid 60 runs/L9a/hz_\$0_B\$1_s\$2.json > runs/L9a/hz_\$0_B\$1_s\$2.log 2>&1"
rc1=$?
grid | xargs -P 24 -L1 bash -c "$PY exp/l9a_horizon.py runs/E04/feats_\$0.parquet judge_error $CJ \$1 \$2 disjoint12 12 runs/L9a/gate_\$0_B\$1_o\$2.json > runs/L9a/gate_\$0_B\$1_o\$2.log 2>&1"
rc2=$?
grid | xargs -P 24 -L1 bash -c "$PY exp/l9a_horizon.py runs/E04/feats_\$0.parquet judge_error $CJ \$1 \$2 iid_nr 60 runs/L9a/nr_\$0_B\$1_s\$2.json > runs/L9a/nr_\$0_B\$1_s\$2.log 2>&1"
rc3=$?
grid | xargs -P 24 -L1 bash -c "$PY exp/l9a_horizon.py runs/E04/feats_\$0.parquet judge_error $CJ \$1 \$2 iid 120 runs/L9a/hz120_\$0_B\$1_s\$2.json > runs/L9a/hz120_\$0_B\$1_s\$2.log 2>&1"
rc4=$?
$PY exp/l9a_summarize.py runs/L9a runs/L7b > runs/L9a/summary.log 2>&1; src=$?
[ $rc1 -ne 0 ] && exit $rc1
[ $rc2 -ne 0 ] && exit $rc2
[ $rc3 -ne 0 ] && exit $rc3
[ $rc4 -ne 0 ] && exit $rc4
exit $src
