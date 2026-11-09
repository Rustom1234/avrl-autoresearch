#!/bin/bash
# L1a full run (CPU only, launched through bin/gpu_run.sh, which writes exit_code and done.flag; this script must not):
#   1 build the six rollout-level tables (exit 3 if the build gate fails: >= 200 positives, judge AUC reproduction)
#   2 regression gate: group mode on runs/E04/feats_{E01,3B,7B-AWQ}.parquet, cell100, cheap features, split seed 0, 0 per-seed differences against runs/E04/sim_*_cell100_cheap.json (exit 3 = VOID)
#   3 structure statistics (6 tables in parallel)
#   4 allocation replay: 6 tables x {cheap_r, cheap_r + 16 PCs} x {cell100 split 0..3, cell30 split 0, pid split 0} = 72 jobs, xargs -P 24, largest tables first
#   5 summary (result.json, summary.md)
# Launch: cd /homes/rdubash/avrl_live/FULL_R/avrl && /homes/rdubash/avrl_live/FULL_R/bin/gpu_run.sh runs/L1a 'bash exp/L1a.sh'
cd /homes/rdubash/avrl_live/FULL_R/avrl && mkdir -p runs/L1a/gate
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
PY=/homes/rdubash/active-audit/.venv/bin/python
CH=agreement,n_distinct,mean_tokens,std_tokens,any_truncated,log_qlen; CJ=$CH,judge_top,judge_margin,judge_mean,judge_std
CR=judge_logit,log_tokens,truncated,has_decimal,is_negative,log_abs_ans,empty_answer,log_qlen
CP=$CR,pc0,pc1,pc2,pc3,pc4,pc5,pc6,pc7,pc8,pc9,pc10,pc11,pc12,pc13,pc14,pc15
t0=$(date +%s)

$PY exp/l1a_build.py --rollouts runs/p0/rollouts.parquet --feats runs/p0/feats.parquet --judges runs/E04 --out runs/L1a > runs/L1a/build.log 2>&1 || { echo "BUILD GATE/ERROR rc=$?"; cat runs/L1a/build.log; exit 3; }
echo "build done $(( $(date +%s) - t0 ))s"

$PY exp/l1a_sim_alloc.py runs/E04/feats_E01.parquet proxy_error cell100 $CH runs/L1a/gate/g_E01.json 0 > runs/L1a/gate/g_E01.log 2>&1 &
$PY exp/l1a_sim_alloc.py runs/E04/feats_Qwen2.5-3B-Instruct.parquet judge_error cell100 $CJ runs/L1a/gate/g_3B.json 0 > runs/L1a/gate/g_3B.log 2>&1 &
$PY exp/l1a_sim_alloc.py runs/E04/feats_Qwen2.5-7B-Instruct-AWQ.parquet judge_error cell100 $CJ runs/L1a/gate/g_7B.json 0 > runs/L1a/gate/g_7B.log 2>&1 &
wait
$PY exp/l1a_gate_check.py || { echo "REGRESSION GATE FAILED: run is VOID"; exit 3; }
echo "gate done $(( $(date +%s) - t0 ))s"

$PY exp/l1a_structure.py runs/L1a runs/p0/feats.parquet > runs/L1a/structure.log 2>&1 || { echo "STRUCTURE FAILED"; tail -30 runs/L1a/structure.log; exit 4; }
echo "structure done $(( $(date +%s) - t0 ))s"

job() { tag=$1; lab=$2; lay=$3; fs=$4; sp=$5; cols=$CR; [ "$fs" = pc ] && cols=$CP
  $PY exp/l1a_sim_alloc.py runs/L1a/tab_$tag.parquet $lab $lay $cols runs/L1a/sim_${tag}_${fs}_${lay}_s$sp.json $sp > runs/L1a/sim_${tag}_${fs}_${lay}_s$sp.log 2>&1; }
export -f job; export PY CR CP
jobs_list() {
  for tl in 3B_q_judgewrong:judge_wrong 7B_q_judgewrong:judge_wrong 7B_p05_judgewrong:judge_wrong 3B_q_corrupted:corrupted 7B_q_corrupted:corrupted 7B_p05_corrupted:corrupted; do
    tag=${tl%%:*}; lab=${tl##*:}
    for fs in cheap pc; do
      for sp in 0 1 2 3; do echo "$tag $lab cell100 $fs $sp"; done
      echo "$tag $lab cell30 $fs 0"; echo "$tag $lab pid $fs 0"
    done
  done
}
jobs_list | xargs -P 24 -L1 bash -c 'job $0 $1 $2 $3 $4'
rc=$?
echo "replay done rc=$rc $(( $(date +%s) - t0 ))s"
[ $rc -ne 0 ] && { echo "REPLAY JOBS FAILED rc=$rc"; exit 5; }
ls runs/L1a/sim_*.json | wc -l

$PY exp/l1a_summarize.py runs/L1a > runs/L1a/summary.log 2>&1; src=$?
echo "summary rc=$src $(( $(date +%s) - t0 ))s"
exit $src
