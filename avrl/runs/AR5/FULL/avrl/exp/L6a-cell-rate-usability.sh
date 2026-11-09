#!/bin/bash
# L6a-cell-rate-usability driver (CPU only, no GPU). CV (3 pools, rm also at K=1,840) in parallel, then the replay (3 pools + rm sensitivity) in parallel, then the finalizer.
# Writes runs/L6a-cell-rate-usability/{rm,judge3b,L3b-judge-proxy}/{result_cv.json,cv_table.txt,boot_cellrerank_f0.25.json}, result.json, table.txt, exit_code, done.flag (last).
cd /homes/rdubash/avrl_live/FULL/avrl || exit 9
OUT=runs/L6a-cell-rate-usability; PY=/homes/rdubash/active-audit/.venv/bin/python; mkdir -p $OUT/rm $OUT/judge3b $OUT/L3b-judge-proxy; rm -f $OUT/done.flag $OUT/exit_code
FEATS=s_maj,s_oth,s_diff,mean_tokens,std_tokens,any_truncated,log_qlen,agreement,n_distinct
declare -A SRC=([rm]=runs/L5c-proxies/rm [judge3b]=runs/L5c-proxies/judge3b [L3b-judge-proxy]=runs/L3b-judge-proxy); RC=0
pids=(); for N in rm judge3b L3b-judge-proxy; do S=""; [ $N = rm ] && S="--sens"; OMP_NUM_THREADS=4 $PY exp/l6a_cell_rate.py --pool ${SRC[$N]} --out $OUT/$N $S > $OUT/$N/cv.log 2>&1 & pids+=($!); done
for p in "${pids[@]}"; do wait $p || RC=$?; done; echo "cv stage rc=$RC"
pids=(); for N in rm judge3b L3b-judge-proxy; do OMP_NUM_THREADS=2 BOOT_B=200 BOOT_SEEDS=4 BOOT_FRAC=0.25 $PY exp/sim_warm_boot_j3.py ${SRC[$N]}/feats_J.parquet judge_error cell100 $FEATS $OUT/$N/boot_cellrerank_f0.25.json > $OUT/$N/boot.log 2>&1 & pids+=($!); done
ACCEPT_COL=accept_sens OMP_NUM_THREADS=2 BOOT_B=200 BOOT_SEEDS=4 BOOT_FRAC=0.25 $PY exp/sim_warm_boot_j3.py ${SRC[rm]}/feats_J.parquet judge_error_sens cell100 $FEATS $OUT/rm/boot_cellrerank_sens_f0.25.json > $OUT/rm/boot_sens.log 2>&1 & pids+=($!)
for p in "${pids[@]}"; do wait $p || RC=$?; done; echo "replay stage rc=$RC"
$PY exp/l6a_finalize.py --root $OUT > $OUT/finalize.log 2>&1 || RC=$?; echo "finalize rc=$RC"
echo $RC > $OUT/exit_code; touch $OUT/done.flag
