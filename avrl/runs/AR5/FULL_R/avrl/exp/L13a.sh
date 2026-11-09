#!/bin/bash
# L13a (CPU only, launched through bin/gpu_run.sh, which writes exit_code and done.flag; this script must not):
#   1 gate 1 (regression): l13a_sim_alloc.py with the warm arms skipped (nowarm flag), corrupted label, 3B and 7B-AWQ, target 2250 and target 0, split 0, subsample 0, cell30, cheap_r
#     vs runs/L12c/sim_*_n{2250,0}_sp0_sub0.json, 0 per-seed differences in every arm (exit 3 = VOID)
#     gate 2 (identity): same script with the norefit flag, target 2250, split 0, subsample 0, both rungs: warm_online_n == recal_free_n per policy seed at all three budgets (exit 3 = VOID)
#   2 grid: corrupted label only, rungs 3B and 7B-AWQ, target n_sim {4500, 2250, 1125} x split 0..3 x subsample 0..3 (96 jobs) + the full-scale reference (target 0, split 0..3, 8 jobs), all cell30 + cheap_r,
#     policies incl. warm_online_{100,200,400}; the first (largest) job is timed alone, the rest run with xargs -P 24, largest first
#   3 post gate (all pre-existing arms of the grid equal L12c, warm arms present; exit 3 = VOID, summary is still written) and summary (result.json, summary.md)
# Launch: cd /homes/rdubash/avrl_live/FULL_R/avrl && mkdir -p runs/L13a && /homes/rdubash/avrl_live/FULL_R/bin/gpu_run.sh runs/L13a 'bash exp/L13a.sh'
cd /homes/rdubash/avrl_live/FULL_R/avrl && mkdir -p runs/L13a/gate
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
PY=/homes/rdubash/active-audit/.venv/bin/python
CR=judge_logit,log_tokens,truncated,has_decimal,is_negative,log_abs_ans,empty_answer,log_qlen
t0=$(date +%s)

for tag in 3B_q_corrupted 7B_q_corrupted; do
  for tgt in 2250 0; do
    $PY exp/l13a_sim_alloc.py runs/L1a/tab_$tag.parquet corrupted cell30 $CR runs/L13a/gate/g_${tag}_n$tgt.json 0 $tgt 0 nowarm > runs/L13a/gate/g_${tag}_n$tgt.log 2>&1 &
  done
  $PY exp/l13a_sim_alloc.py runs/L1a/tab_$tag.parquet corrupted cell30 $CR runs/L13a/gate/i_${tag}_n2250.json 0 2250 0 norefit > runs/L13a/gate/i_${tag}_n2250.log 2>&1 &
done
wait
$PY exp/l13a_gate_check.py pre || { echo "GATE 1 (REGRESSION) FAILED: run is VOID"; exit 3; }
$PY exp/l13a_gate_check.py ident || { echo "GATE 2 (IDENTITY) FAILED: run is VOID"; exit 3; }
echo "gates done $(( $(date +%s) - t0 ))s"

job() { tag=$1; lab=$2; tgt=$3; sp=$4; sub=$5; o=runs/L13a/sim_${tag}_n${tgt}_sp${sp}_sub${sub}
  $PY exp/l13a_sim_alloc.py runs/L1a/tab_$tag.parquet $lab cell30 $CR $o.json $sp $tgt $sub > $o.log 2>&1; }
export -f job; export PY CR
jobs_list() {
  for tgt in 0 4500 2250 1125; do
    for tl in 3B_q_corrupted:corrupted 7B_q_corrupted:corrupted; do
      tag=${tl%%:*}; lab=${tl##*:}
      for sp in 0 1 2 3; do
        if [ $tgt = 0 ]; then echo "$tag $lab $tgt $sp 0"; else for sub in 0 1 2 3; do echo "$tag $lab $tgt $sp $sub"; done; fi
      done
    done
  done
}
jobs_list > runs/L13a/jobs.txt; echo "jobs $(wc -l < runs/L13a/jobs.txt)"
t1=$(date +%s); head -1 runs/L13a/jobs.txt | xargs -L1 bash -c 'job $0 $1 $2 $3 $4'; rc1=$?
echo "first job (largest: $(head -1 runs/L13a/jobs.txt)) rc=$rc1 took $(( $(date +%s) - t1 ))s"
[ $rc1 -ne 0 ] && { echo "FIRST JOB FAILED"; exit 5; }
tail -n +2 runs/L13a/jobs.txt | xargs -P 24 -L1 bash -c 'job $0 $1 $2 $3 $4'
rc=$?
echo "grid done rc=$rc $(( $(date +%s) - t0 ))s"
[ $rc -ne 0 ] && { echo "GRID JOBS FAILED rc=$rc"; exit 5; }
echo "sim json: $(ls runs/L13a/sim_*.json | wc -l)"

$PY exp/l13a_gate_check.py post; prc=$?
echo "post gate rc=$prc"
$PY exp/l13a_summarize.py runs/L13a > runs/L13a/summary.log 2>&1; src=$?
echo "summary rc=$src $(( $(date +%s) - t0 ))s"
tail -5 runs/L13a/summary.log
[ $prc -ne 0 ] && { echo "POST GATE FAILED: run is VOID"; exit 3; }
exit $src
