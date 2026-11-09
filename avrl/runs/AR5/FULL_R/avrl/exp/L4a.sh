#!/bin/bash
# L4a full run (CPU only): regression gate at m = 500, subset 0 vs runs/L2b (2 jobs), lam = inf sanity check (4 jobs), then 96 sweep jobs
# (2 judges x m in 250, 500, 1000 x subset seeds 0..15), then the summary.
# Launch: bin/gpu_run.sh runs/L4a 'bash exp/L4a.sh'. Does NOT write done.flag or exit_code: bin/gpu_run.sh's wrapper writes both.
cd /homes/rdubash/avrl_live/FULL_R/avrl && mkdir -p runs/L4a/gate runs/L4a/sanity
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
PY=/homes/rdubash/active-audit/.venv/bin/python
CJ=agreement,n_distinct,mean_tokens,std_tokens,any_truncated,log_qlen,judge_top,judge_margin,judge_mean,judge_std
J3=Qwen2.5-3B-Instruct; J7=Qwen2.5-7B-Instruct-AWQ
# ---- regression gate: m = 500, subset seed 0, both judge labels, vs runs/L2b
$PY exp/l4a_sim_alloc.py runs/E04/feats_$J3.parquet judge_error $CJ 500 0 runs/L4a/gate/g_3B.json > runs/L4a/gate/g_3B.log 2>&1 &
$PY exp/l4a_sim_alloc.py runs/E04/feats_$J7.parquet judge_error $CJ 500 0 runs/L4a/gate/g_7B.json > runs/L4a/gate/g_7B.log 2>&1 &
wait
$PY exp/l4a_gate_check.py runs/L4a/gate runs/L2b > runs/L4a/gate/gate_check.log 2>&1; grc=$?
if [ $grc -ne 0 ] || ! grep -q '"passed": true' runs/L4a/gate/gate_check.json 2>/dev/null; then
  echo "REGRESSION GATE FAILED: run is void, sweep skipped"; $PY exp/l4a_summarize.py runs/L4a > runs/L4a/summary.log 2>&1; exit 3
fi
# ---- sanity: run_offset with lam = inf (w = 0) and n0 in 0, 5 must reproduce thr_agreement per-seed hits (m in 250, 500; subset 0; both judges)
for j in $J3 $J7; do for m in 250 500; do
  L4A_SANITY=runs/L4a/sanity/san_${j}_m${m}_s0.json $PY exp/l4a_sim_alloc.py runs/E04/feats_$j.parquet judge_error $CJ $m 0 > runs/L4a/sanity/san_${j}_m${m}_s0.log 2>&1 &
done; done
wait
# ---- sweep: judge x m x subset seed
(for j in $J3 $J7; do for m in 250 500 1000; do for s in $(seq 0 15); do echo "$j $m $s"; done; done; done) | \
  xargs -P 24 -L1 bash -c "$PY exp/l4a_sim_alloc.py runs/E04/feats_\$0.parquet judge_error $CJ \$1 \$2 runs/L4a/sim_\$0_m\$1_s\$2.json > runs/L4a/sim_\$0_m\$1_s\$2.log 2>&1"
rc=$?
if [ $rc -eq 0 ]; then $PY exp/l4a_summarize.py runs/L4a > runs/L4a/summary.log 2>&1; rc=$?; fi
exit $rc
