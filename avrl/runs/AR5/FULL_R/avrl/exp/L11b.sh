#!/bin/bash
# L11b full run (CPU only, launched through bin/gpu_run.sh, which writes exit_code and done.flag; this script must not):
#   1 join feats to the GSM8K gold answers, augmented table copies, structure statistics + R^2 + cheap_r AUC gate (exit 3 = VOID)
#   2 allocation replays on the augmented tables: 2 rungs x {cell100, cell30} x {cheap_r, cheap_r + 3 oracle, cheap_r + 2 oracle (no gold magnitude)} x splits 0..3 = 48 jobs, xargs -P 24
#   3 regression gate: no-oracle replay vs runs/L1a files, 0 per-seed differences (exit 3 = VOID)
#   4 summary (result.json, summary.md)
# Launch: cd /homes/rdubash/avrl_live/FULL_R/avrl && mkdir -p runs/L11b && /homes/rdubash/avrl_live/FULL_R/bin/gpu_run.sh runs/L11b 'bash exp/L11b.sh'
cd /homes/rdubash/avrl_live/FULL_R/avrl && mkdir -p runs/L11b/gate
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
PY=/homes/rdubash/active-audit/.venv/bin/python
CR=judge_logit,log_tokens,truncated,has_decimal,is_negative,log_abs_ans,empty_answer,log_qlen
CO=$CR,n_steps,n_numbers_q,log_abs_gold; CN=$CR,n_steps,n_numbers_q
t0=$(date +%s)

$PY exp/l11b_oracle.py runs/L1a runs/p0/feats.parquet 3B_q_corrupted 7B_q_corrupted > runs/L11b/oracle.log 2>&1; rc=$?
cat runs/L11b/oracle.log
[ $rc -ne 0 ] && { echo "ORACLE STEP OR AUC GATE FAILED rc=$rc: run is VOID"; exit 3; }
echo "oracle done $(( $(date +%s) - t0 ))s"

job() { tag=$1; lay=$2; fs=$3; sp=$4; cols=$CR; [ "$fs" = orc ] && cols=$CO; [ "$fs" = orc_ng ] && cols=$CN
  $PY exp/l1a_sim_alloc.py runs/L11b/tab_$tag.parquet corrupted $lay $cols runs/L11b/sim_${tag}_${fs}_${lay}_s$sp.json $sp > runs/L11b/sim_${tag}_${fs}_${lay}_s$sp.log 2>&1; }
export -f job; export PY CR CO CN
for tag in 3B_q_corrupted 7B_q_corrupted; do for lay in cell100 cell30; do for fs in cheap orc orc_ng; do for sp in 0 1 2 3; do echo "$tag $lay $fs $sp"; done; done; done; done \
  | xargs -P 24 -L1 bash -c 'job $0 $1 $2 $3'
rc=$?
echo "replay done rc=$rc $(( $(date +%s) - t0 ))s"
[ $rc -ne 0 ] && { echo "REPLAY JOBS FAILED rc=$rc"; exit 5; }
ls runs/L11b/sim_*.json | wc -l

$PY exp/l11b_gate_check.py || { echo "REGRESSION GATE FAILED: run is VOID"; exit 3; }
$PY exp/l11b_summarize.py > runs/L11b/summary.log 2>&1; src=$?
cat runs/L11b/summary.log
echo "summary rc=$src $(( $(date +%s) - t0 ))s"
exit $src
