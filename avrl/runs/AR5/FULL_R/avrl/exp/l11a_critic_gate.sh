#!/bin/bash
# L11a critic check (CPU only, launched through bin/gpu_run.sh runs/L11a_critic 'bash exp/l11a_critic_gate.sh'; this script must not write done.flag or exit_code).
# Exercises the NEW subsample block of exp/l11a_sim_alloc.py at fraction 1.0 (target 1,000,000 keeps every pid) for all four L1a tables and split seeds 0 to 3, cell100,
# so the output must reproduce runs/L1a/sim_<tag>_cheap_cell100_s<split>.json per-seed (the literal backlog gate: "a subsample fraction of 1.0 reproduces runs/L1a/sim_*_cheap_cell100_s*.json").
cd /homes/rdubash/avrl_live/FULL_R/avrl && mkdir -p runs/L11a_critic
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
PY=/homes/rdubash/active-audit/.venv/bin/python
CR=judge_logit,log_tokens,truncated,has_decimal,is_negative,log_abs_ans,empty_answer,log_qlen
job() { tag=$1; lab=$2; sp=$3; o=runs/L11a_critic/g_${tag}_s${sp}
  $PY exp/l11a_sim_alloc.py runs/L1a/tab_$tag.parquet $lab cell100 $CR $o.json $sp 1000000 0 > $o.log 2>&1; }
export -f job; export PY CR
for tl in 3B_q_judgewrong:judge_wrong 7B_q_judgewrong:judge_wrong 3B_q_corrupted:corrupted 7B_q_corrupted:corrupted; do
  for sp in 0 1 2 3; do echo "${tl%%:*} ${tl##*:} $sp"; done
done | xargs -P 16 -L1 bash -c 'job $0 $1 $2'
echo "critic gate jobs rc=$?"
ls runs/L11a_critic/g_*.json | wc -l
