#!/bin/bash
# L5a full run (CPU only): regression gate vs runs/L4a (4 gate jobs + gate check), then 32 sweep jobs (2 judges x per-step B in 62, 125 x order seeds 0..7), then the summary.
# Launch: bin/gpu_run.sh runs/L5a 'bash exp/L5a.sh'. Does NOT write done.flag or exit_code: bin/gpu_run.sh's wrapper writes both.
cd /homes/rdubash/avrl_live/FULL_R/avrl && mkdir -p runs/L5a/gate
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
PY=/homes/rdubash/active-audit/.venv/bin/python
CJ=agreement,n_distinct,mean_tokens,std_tokens,any_truncated,log_qlen,judge_top,judge_margin,judge_mean,judge_std
J3=Qwen2.5-3B-Instruct; J7=Qwen2.5-7B-Instruct-AWQ
# ---- regression gate: L4a prompt set (m = 250, subset seed 0), both judges, both per-step budgets, vs runs/L4a (4 jobs)
for j in $J3 $J7; do for B in 62 125; do
  $PY exp/l5a_carry.py --gate runs/E04/feats_$j.parquet judge_error $CJ $B runs/L5a/gate/g_${j}_B${B}.json > runs/L5a/gate/g_${j}_B${B}.log 2>&1 &
done; done
wait
$PY exp/l5a_gate_check.py runs/L5a/gate runs/L4a > runs/L5a/gate/gate_check.log 2>&1; grc=$?
if [ $grc -ne 0 ] || ! grep -q '"passed": true' runs/L5a/gate/gate_check.json 2>/dev/null; then
  echo "REGRESSION GATE FAILED: run is void, sweep skipped"; $PY exp/l5a_summarize.py runs/L5a > runs/L5a/summary.log 2>&1; exit 3
fi
# ---- sweep: judge x per-step budget x order seed
(for j in $J3 $J7; do for B in 62 125; do for o in $(seq 0 7); do echo "$j $B $o"; done; done; done) | \
  xargs -P 24 -L1 bash -c "$PY exp/l5a_carry.py runs/E04/feats_\$0.parquet judge_error $CJ \$1 \$2 runs/L5a/carry_\$0_B\$1_o\$2.json > runs/L5a/carry_\$0_B\$1_o\$2.log 2>&1"
rc=$?
if [ $rc -eq 0 ]; then $PY exp/l5a_summarize.py runs/L5a > runs/L5a/summary.log 2>&1; rc=$?; fi
exit $rc
