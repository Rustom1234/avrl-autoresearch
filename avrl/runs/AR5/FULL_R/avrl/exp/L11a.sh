#!/bin/bash
# L11a (CPU only, launched through bin/gpu_run.sh, which writes exit_code and done.flag; this script must not):
#   1 regression gate: target 0, split 0, cheap_r, cell100 on the four L1a tables vs runs/L1a/sim_*_cheap_cell100_s0.json, 0 per-seed differences (exit 3 = VOID)
#   2 grid: 4 tables x target n_sim {4500, 2250, 1125} x split 0..3 x subsample 0..3 (192 jobs) + the full-scale reference (target 0, split 0..3, 16 jobs), all cell30 + cheap_r,
#     the first (largest) job is timed alone, the rest run with xargs -P 24, largest first
#   3 summary (result.json, summary.md)
# Launch: cd /homes/rdubash/avrl_live/FULL_R/avrl && mkdir -p runs/L11a && /homes/rdubash/avrl_live/FULL_R/bin/gpu_run.sh runs/L11a 'bash exp/L11a.sh'
cd /homes/rdubash/avrl_live/FULL_R/avrl && mkdir -p runs/L11a/gate
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
PY=/homes/rdubash/active-audit/.venv/bin/python
CR=judge_logit,log_tokens,truncated,has_decimal,is_negative,log_abs_ans,empty_answer,log_qlen
t0=$(date +%s)

for tag in 3B_q_corrupted 7B_q_corrupted 3B_q_judgewrong 7B_q_judgewrong; do
  lab=corrupted; [[ $tag == *judgewrong ]] && lab=judge_wrong
  $PY exp/l11a_sim_alloc.py runs/L1a/tab_$tag.parquet $lab cell100 $CR runs/L11a/gate/g_$tag.json 0 0 0 > runs/L11a/gate/g_$tag.log 2>&1 &
done
wait
$PY exp/l11a_gate_check.py || { echo "REGRESSION GATE FAILED: run is VOID"; exit 3; }
echo "gate done $(( $(date +%s) - t0 ))s"

job() { tag=$1; lab=$2; tgt=$3; sp=$4; sub=$5; o=runs/L11a/sim_${tag}_n${tgt}_sp${sp}_sub${sub}
  $PY exp/l11a_sim_alloc.py runs/L1a/tab_$tag.parquet $lab cell30 $CR $o.json $sp $tgt $sub > $o.log 2>&1; }
export -f job; export PY CR
jobs_list() {
  for tgt in 0 4500 2250 1125; do
    for tl in 3B_q_judgewrong:judge_wrong 7B_q_judgewrong:judge_wrong 3B_q_corrupted:corrupted 7B_q_corrupted:corrupted; do
      tag=${tl%%:*}; lab=${tl##*:}
      for sp in 0 1 2 3; do
        if [ $tgt = 0 ]; then echo "$tag $lab $tgt $sp 0"; else for sub in 0 1 2 3; do echo "$tag $lab $tgt $sp $sub"; done; fi
      done
    done
  done
}
jobs_list > runs/L11a/jobs.txt; echo "jobs $(wc -l < runs/L11a/jobs.txt)"
t1=$(date +%s); head -1 runs/L11a/jobs.txt | xargs -L1 bash -c 'job $0 $1 $2 $3 $4'; rc1=$?
echo "first job (largest: $(head -1 runs/L11a/jobs.txt)) rc=$rc1 took $(( $(date +%s) - t1 ))s"
[ $rc1 -ne 0 ] && { echo "FIRST JOB FAILED"; exit 5; }
tail -n +2 runs/L11a/jobs.txt | xargs -P 24 -L1 bash -c 'job $0 $1 $2 $3 $4'
rc=$?
echo "grid done rc=$rc $(( $(date +%s) - t0 ))s"
[ $rc -ne 0 ] && { echo "GRID JOBS FAILED rc=$rc"; exit 5; }
echo "sim json: $(ls runs/L11a/sim_*.json | wc -l)"

$PY exp/l11a_summarize.py runs/L11a > runs/L11a/summary.log 2>&1; src=$?
echo "summary rc=$src $(( $(date +%s) - t0 ))s"
tail -5 runs/L11a/summary.log
exit $src
