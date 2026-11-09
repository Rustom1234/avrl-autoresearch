#!/bin/bash
# L2a pool bootstrap of the 10% budget (CPU only, no GPU). 8 sim_warm_boot.py jobs (xargs -P 6), then l2a_table.py. done.flag + exit_code like the L1a driver.
set -uo pipefail
cd /homes/rdubash/avrl_live/FULL/avrl
export PY=/homes/rdubash/active-audit/.venv/bin/python OUT=runs/L2a-pool-bootstrap-10pct E21=runs/E21-lowT-confident-errors REF=runs/L1a-warm-start-replay
export CHEAP=agreement,n_distinct,mean_tokens,std_tokens,any_truncated,log_qlen OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
mkdir -p $OUT; rm -f $OUT/done.flag $OUT/exit_code $OUT/boot_*.json $OUT/boot_*.log $OUT/jobs_failed
for spec in "0.25 0.1 cell100" "0.25 0.25 cell100" "0.5 0.1 cell100" "0.5 0.25 cell100" "1.0 0.1 cell100" "1.0 0.25 cell100" "0.25 0.1 cell30" "0.5 0.1 cell30"; do echo $spec; done |
  xargs -P 6 -L 1 bash -c 'BOOT_B=200 BOOT_SEEDS=4 BOOT_FRAC=$1 REF_JSON=$REF/sim_T$0_$2_cheap.json $PY exp/sim_warm_boot.py $E21/feats_T$0.parquet proxy_error $2 $CHEAP \
    $OUT/boot_T$0_${2}_f$1.json > $OUT/boot_T$0_${2}_f$1.log 2>&1 || echo "FAILED T=$0 frac=$1 cell=$2" >> $OUT/jobs_failed'
if [ -s $OUT/jobs_failed ]; then cat $OUT/jobs_failed; echo 1 > $OUT/exit_code; touch $OUT/done.flag; exit 1; fi
$PY exp/l2a_table.py $OUT > $OUT/table.log 2>&1 || { echo "table failed"; tail -20 $OUT/table.log; echo 1 > $OUT/exit_code; touch $OUT/done.flag; exit 1; }
echo 0 > $OUT/exit_code; touch $OUT/done.flag; tail -3 $OUT/table.log
