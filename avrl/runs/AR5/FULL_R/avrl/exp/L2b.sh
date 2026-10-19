#!/bin/bash
# L2b full run (CPU only): m = 0 regression gate vs runs/L2a (2 jobs), then 96 sweep jobs (2 judges x m in 250, 500, 1000 x subset seeds 0..15), then the summary.
# Launch: bin/gpu_run.sh runs/L2b 'bash exp/L2b.sh'. Does NOT write done.flag or exit_code: bin/gpu_run.sh's wrapper writes both.
cd /homes/rdubash/avrl_live/FULL_R/avrl && mkdir -p runs/L2b/gate
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
PY=/homes/rdubash/active-audit/.venv/bin/python
CJ=agreement,n_distinct,mean_tokens,std_tokens,any_truncated,log_qlen,judge_top,judge_margin,judge_mean,judge_std
J3=Qwen2.5-3B-Instruct; J7=Qwen2.5-7B-Instruct-AWQ
# ---- regression gate: m = 0, split seed 0, both judge labels
$PY exp/l2b_sim_alloc.py runs/E04/feats_$J3.parquet judge_error $CJ 0 0 runs/L2b/gate/g_3B.json > runs/L2b/gate/g_3B.log 2>&1 &
$PY exp/l2b_sim_alloc.py runs/E04/feats_$J7.parquet judge_error $CJ 0 0 runs/L2b/gate/g_7B.json > runs/L2b/gate/g_7B.log 2>&1 &
wait
$PY exp/l2b_gate_check.py runs/L2b/gate runs/L2a > runs/L2b/gate/gate_check.log 2>&1; grc=$?
if [ $grc -ne 0 ] || ! grep -q '"passed": true' runs/L2b/gate/gate_check.json 2>/dev/null; then
  echo "REGRESSION GATE FAILED: run is void, sweep skipped"; $PY exp/l2b_summarize.py runs/L2b > runs/L2b/summary.log 2>&1; exit 3
fi
# ---- sweep: judge x m x subset seed
(for j in $J3 $J7; do for m in 250 500 1000; do for s in $(seq 0 15); do echo "$j $m $s"; done; done; done) | \
  xargs -P 24 -L1 bash -c "$PY exp/l2b_sim_alloc.py runs/E04/feats_\$0.parquet judge_error $CJ \$1 \$2 runs/L2b/sim_\$0_m\$1_s\$2.json > runs/L2b/sim_\$0_m\$1_s\$2.log 2>&1"
rc=$?
if [ $rc -eq 0 ]; then $PY exp/l2b_summarize.py runs/L2b > runs/L2b/summary.log 2>&1; rc=$?; fi
exit $rc
