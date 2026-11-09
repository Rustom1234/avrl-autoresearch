#!/bin/bash
# L15a (CPU only, launched through bin/gpu_run.sh, which writes exit_code and done.flag; this script must not):
#   1 gate 1 (regression): l15a_sim_alloc.py with STALE=none, refits on, corrupted label, 3B and 7B-AWQ, target 2250, split 0, subsample 0, cell30, cheap_r
#     vs runs/L13a/sim_*_n2250_sp0_sub0.json, 0 per-seed differences in uniform, static, greedy_online, recal_free_200, warm_online_200 (exit 3 = VOID)
#     gate 2 (identity): same script with the norefit flag, all four STALE variants (none, flip25, flip50, lowlen), both rungs: warm_online_200 == recal_free_200 per policy seed at all three budgets (exit 3 = VOID)
#   2 grid: corrupted label only, rungs 3B and 7B-AWQ, STALE in {flip25, flip50, lowlen, none} x target n_sim 2250 x split 0..3 x subsample 0..3 = 128 jobs, all cell30 + cheap_r, K = 1, 8 policy seeds;
#     arms uniform, static, greedy_online (cold), recal_free_200, warm_online_200, warm_online_200_w25; the first job is timed alone (3B flip25, the pre-registered primary), the rest run with xargs -P 24
#   3 post gate (128 files present; none files == L13a on all five L13a arms; uniform/static/greedy_online == L13a in every variant; exit 3 = VOID, summary is still written) and summary (result.json, summary.md)
# Launch: cd /homes/rdubash/avrl_live/FULL_R/avrl && mkdir -p runs/L15a && /homes/rdubash/avrl_live/FULL_R/bin/gpu_run.sh runs/L15a 'bash exp/L15a.sh'
cd /homes/rdubash/avrl_live/FULL_R/avrl && mkdir -p runs/L15a/gate
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
PY=/homes/rdubash/active-audit/.venv/bin/python
CR=judge_logit,log_tokens,truncated,has_decimal,is_negative,log_abs_ans,empty_answer,log_qlen
t0=$(date +%s)

for tag in 3B_q_corrupted 7B_q_corrupted; do
  $PY exp/l15a_sim_alloc.py runs/L1a/tab_$tag.parquet corrupted cell30 $CR runs/L15a/gate/g_${tag}_none.json 0 2250 0 refit none > runs/L15a/gate/g_${tag}_none.log 2>&1 &
  for st in none flip25 flip50 lowlen; do
    $PY exp/l15a_sim_alloc.py runs/L1a/tab_$tag.parquet corrupted cell30 $CR runs/L15a/gate/i_${tag}_${st}.json 0 2250 0 norefit $st > runs/L15a/gate/i_${tag}_${st}.log 2>&1 &
  done
done
wait
$PY exp/l15a_gate_check.py pre || { echo "GATE 1 (REGRESSION) FAILED: run is VOID"; exit 3; }
$PY exp/l15a_gate_check.py ident || { echo "GATE 2 (IDENTITY) FAILED: run is VOID"; exit 3; }
echo "gates done $(( $(date +%s) - t0 ))s"

job() { tag=$1; st=$2; sp=$3; sub=$4; o=runs/L15a/sim_${tag}_${st}_n2250_sp${sp}_sub${sub}
  $PY exp/l15a_sim_alloc.py runs/L1a/tab_$tag.parquet corrupted cell30 $CR $o.json $sp 2250 $sub refit $st > $o.log 2>&1; }
export -f job; export PY CR
jobs_list() {
  for st in flip25 flip50 lowlen none; do
    for tag in 3B_q_corrupted 7B_q_corrupted; do
      for sp in 0 1 2 3; do for sub in 0 1 2 3; do echo "$tag $st $sp $sub"; done; done
    done
  done
}
jobs_list > runs/L15a/jobs.txt; echo "jobs $(wc -l < runs/L15a/jobs.txt)"
t1=$(date +%s); head -1 runs/L15a/jobs.txt | xargs -L1 bash -c 'job $0 $1 $2 $3'; rc1=$?
echo "first job ($(head -1 runs/L15a/jobs.txt)) rc=$rc1 took $(( $(date +%s) - t1 ))s"
[ $rc1 -ne 0 ] && { echo "FIRST JOB FAILED"; exit 5; }
tail -n +2 runs/L15a/jobs.txt | xargs -P 24 -L1 bash -c 'job $0 $1 $2 $3'
rc=$?
echo "grid done rc=$rc $(( $(date +%s) - t0 ))s"
[ $rc -ne 0 ] && { echo "GRID JOBS FAILED rc=$rc"; exit 5; }
echo "sim json: $(ls runs/L15a/sim_*.json | wc -l)"

$PY exp/l15a_gate_check.py post; prc=$?
echo "post gate rc=$prc"
$PY exp/l15a_summarize.py runs/L15a > runs/L15a/summary.log 2>&1; src=$?
echo "summary rc=$src $(( $(date +%s) - t0 ))s"
tail -5 runs/L15a/summary.log
[ $prc -ne 0 ] && { echo "POST GATE FAILED: run is VOID"; exit 3; }
exit $src
