#!/bin/bash
# L7a-tiebreak driver (CPU only, no GPU). Replay jobs (main mode + cross-fit mode, 25% and 10%, 3 pools + rm K=1,840 sensitivity, plus unmodified j3 at 10% as the reproduction reference), then the finalizer.
# Writes runs/L7a-tiebreak/{rm,judge3b,L3b-judge-proxy,rm_sens}/{boot_tb_f*.json,boot_xfit_f*.json}, _ref/<pool>/boot_j3_f0.1.json, result.json, table.txt, exit_code, done.flag (last).
cd /homes/rdubash/avrl_live/FULL/avrl || exit 9
OUT=runs/L7a-tiebreak; PY=/homes/rdubash/active-audit/.venv/bin/python; PAR=${1:-16}; rm -f $OUT/done.flag $OUT/exit_code
FEATS=s_maj,s_oth,s_diff,mean_tokens,std_tokens,any_truncated,log_qlen,agreement,n_distinct
declare -A SRC=([rm]=runs/L5c-proxies/rm [judge3b]=runs/L5c-proxies/judge3b [L3b-judge-proxy]=runs/L3b-judge-proxy [rm_sens]=runs/L5c-proxies/rm); JOBS=$OUT/jobs.txt; : > $JOBS
for N in rm judge3b L3b-judge-proxy rm_sens; do
  mkdir -p $OUT/$N $OUT/_ref/$N; ENVX=""; LAB=judge_error; [ $N = rm_sens ] && { ENVX="ACCEPT_COL=accept_sens"; LAB=judge_error_sens; }
  for F in 0.25 0.1; do
    echo "$ENVX OMP_NUM_THREADS=2 BOOT_B=200 BOOT_SEEDS=4 BOOT_FRAC=$F $PY exp/sim_warm_boot_j4.py ${SRC[$N]}/feats_J.parquet $LAB cell100 $FEATS $OUT/$N/boot_tb_f$F.json > $OUT/$N/boot_f$F.log 2>&1" >> $JOBS
    echo "XFIT=1 $ENVX OMP_NUM_THREADS=2 BOOT_B=200 BOOT_SEEDS=4 BOOT_FRAC=$F $PY exp/sim_warm_boot_j4.py ${SRC[$N]}/feats_J.parquet $LAB cell100 $FEATS $OUT/$N/boot_xfit_f$F.json > $OUT/$N/xfit_f$F.log 2>&1" >> $JOBS
  done
  echo "$ENVX OMP_NUM_THREADS=2 BOOT_B=200 BOOT_SEEDS=4 BOOT_FRAC=0.1 $PY exp/sim_warm_boot_j3.py ${SRC[$N]}/feats_J.parquet $LAB cell100 $FEATS $OUT/_ref/$N/boot_j3_f0.1.json > $OUT/_ref/$N/boot_j3_f0.1.log 2>&1" >> $JOBS
done
RC=0; xargs -P $PAR -I{} bash -c '{}' < $JOBS || RC=$?; echo "replay stage rc=$RC"
for f in $OUT/*/boot_*f*.log $OUT/*/xfit_f*.log $OUT/_ref/*/*.log; do grep -l "GATE FAILED\|Traceback" $f; done
$PY exp/l7a_finalize.py --root $OUT > $OUT/finalize.log 2>&1 || RC=$?; echo "finalize rc=$RC"
echo $RC > $OUT/exit_code; touch $OUT/done.flag
