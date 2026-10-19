#!/bin/bash
# L7a smoke driver (CPU only): synthetic-label worlds on the three real pools' non-label columns, then main + cross-fit replay per pool, then the finalizer per world in --smoke mode.
# usage: L7a-smoke.sh <B> <parallel>   worlds: null_s0..null_s5, planted_soth_s0, planted_sdiff_s0 ; output runs/L7a-tiebreak/_smoke/<world>/<pool>/
cd /homes/rdubash/avrl_live/FULL/avrl || exit 9
WORLDS=${WORLDS:-"null_s0 null_s1 null_s2 null_s3 null_s4 null_s5 planted_soth_s0 planted_sdiff_s0"}; B=${1:-200}; PAR=${2:-14}; PY=/homes/rdubash/active-audit/.venv/bin/python; S=runs/L7a-tiebreak/_smoke; mkdir -p $S
FEATS=s_maj,s_oth,s_diff,mean_tokens,std_tokens,any_truncated,log_qlen,agreement,n_distinct
declare -A SRC=([rm]=runs/L5c-proxies/rm [judge3b]=runs/L5c-proxies/judge3b [L3b-judge-proxy]=runs/L3b-judge-proxy); IDX=([rm]=0 [judge3b]=1 [L3b-judge-proxy]=2)
JOBS=$S/jobs_$(echo $WORLDS | tr ' ' '_' | cut -c1-40).txt; : > $JOBS
for W in $WORLDS; do
  V=${W%_s*}; WS=${W##*_s}
  for N in rm judge3b L3b-judge-proxy; do
    D=$S/$W/$N; mkdir -p $D; $PY exp/l7a_smoke_synth.py ${SRC[$N]}/feats_J.parquet $D $V $((WS * 10 + ${IDX[$N]})) > $D/synth.log 2>&1 || { echo "synth failed $W $N"; exit 8; }
    echo "OMP_NUM_THREADS=2 BOOT_B=$B BOOT_SEEDS=4 BOOT_FRAC=0.25 $PY exp/sim_warm_boot_j4.py $D/feats_J.parquet judge_error cell100 $FEATS $D/boot_tb_f0.25.json > $D/boot_tb.log 2>&1" >> $JOBS
    echo "XFIT=1 OMP_NUM_THREADS=2 BOOT_B=$B BOOT_SEEDS=4 BOOT_FRAC=0.25 $PY exp/sim_warm_boot_j4.py $D/feats_J.parquet judge_error cell100 $FEATS $D/boot_xfit_f0.25.json > $D/boot_xfit.log 2>&1" >> $JOBS
  done
done
xargs -P $PAR -I{} bash -c '{}' < $JOBS; echo "replay stage done rc=$?"
for W in $WORLDS; do $PY exp/l7a_finalize.py --root $S/$W --smoke --fracs 0.25 > $S/$W/finalize.log 2>&1; echo "finalize $W rc=$?"; done
