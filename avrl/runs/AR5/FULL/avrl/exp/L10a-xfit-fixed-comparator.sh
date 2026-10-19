#!/bin/bash
# L10a-xfit-fixed-comparator driver. Launched ONLY through bin/gpu_run.sh (CPU only, but long running: the helper gives logging, the 45 min cap and done.flag). Registered rules and bins: docstring of exp/l10a_boot_xfit_fixed.py.
# Stage 1 reproduction gate: main mode (XFIT unset) of the new script on the 4 pools at 25%, must equal the stored boot_tb_f0.25.json (exp/l10a_gate_repro.py). Stage 2: 24 XFIT=2 jobs (4 pools x 6 budgets). Stage 3: exp/l10a_finalize.py.
# Env overrides (smoke tests on synthetic null labels only): R (run dir), POOLBASE (holds <pool>/pool/<proxy>/feats_J.parquet), BOOT_B, SMOKE=1 (skips stage 1, finalize --smoke).
F=/homes/rdubash/avrl_live/FULL; A=$F/avrl; PY=/homes/rdubash/active-audit/.venv/bin/python; cd $A || exit 9
R=${R:-$A/runs/L10a-xfit-fixed-comparator}; POOLBASE=${POOLBASE:-$A/runs}; B=${BOOT_B:-200}
FEATS=s_maj,s_oth,s_diff,mean_tokens,std_tokens,any_truncated,log_qlen,agreement,n_distinct   # identical to $FEATS in exp/L9c-weak-policy-tiebreak.sh
POOLS="L9c-weak-policy-tiebreak L8c-A-fresh-tiebreak"; FRACS="0.1 0.15 0.2 0.25 0.3 0.4"
frozen() { if [ "$1" = q3_4b ]; then echo agree_acc_sdiff; else echo agree_acc_soth; fi; }
echo "start $(date -u +%FT%TZ) host $(hostname -s) nproc $(nproc)"; uptime; LOAD=$(cut -d' ' -f1 /proc/loadavg); P=16; if awk "BEGIN{exit !($LOAD > 16)}"; then P=8; fi; echo "load1 $LOAD, parallel jobs $P"
sha256sum $A/exp/l10a_boot_xfit_fixed.py $A/exp/l10a_finalize.py $A/exp/l10a_gate_repro.py
mkdir -p $R
if [ -z "$SMOKE" ]; then
  GJ=$R/gate_repro_jobs.txt; : > $GJ; rm -f $R/gate_repro/gate.json
  for pool in $POOLS; do for p in q3_4b rm; do mkdir -p $R/gate_repro/$pool/$p
    echo "MIN_CELL=1 FORCE_BEST7=$(frozen $p) OMP_NUM_THREADS=1 BOOT_B=200 BOOT_SEEDS=4 BOOT_FRAC=0.25 $PY exp/l10a_boot_xfit_fixed.py $POOLBASE/$pool/pool/$p/feats_J.parquet judge_error cell100 $FEATS $R/gate_repro/$pool/$p/boot_tb_f0.25.json > $R/gate_repro/$pool/$p/boot_f0.25.log 2>&1" >> $GJ
  done; done
  xargs -P 4 -I{} bash -c '{}' < $GJ; echo "stage 1 jobs done $(date -u +%FT%TZ)"
  $PY exp/l10a_gate_repro.py $R > $R/gate_repro/gate.log 2>&1; GRC=$?; tail -3 $R/gate_repro/gate.log
  if [ $GRC -ne 0 ]; then echo "REPRODUCTION GATE FAILED (exit $GRC): the copy changed old behaviour, fan-out not run"; $PY -c "import json;json.dump({'status':'FAILED','failures':['reproduction gate mismatch, see gate_repro/gate.log']},open('$R/result.json','w'),indent=1)"; exit 5; fi
fi
JOBS=$R/xfit_jobs.txt; : > $JOBS
for pool in $POOLS; do for p in q3_4b rm; do mkdir -p $R/$pool/$p; for f in $FRACS; do
  echo "XFIT=2 MIN_CELL=1 FORCE_BEST7=$(frozen $p) OMP_NUM_THREADS=1 BOOT_B=$B BOOT_SEEDS=4 BOOT_FRAC=$f $PY exp/l10a_boot_xfit_fixed.py $POOLBASE/$pool/pool/$p/feats_J.parquet judge_error cell100 $FEATS $R/$pool/$p/xfit_f$f.json > $R/$pool/$p/xfit_f$f.log 2>&1" >> $JOBS
done; done; done
wc -l $JOBS; xargs -P $P -I{} bash -c '{}' < $JOBS; echo "stage 2 jobs done $(date -u +%FT%TZ)"
grep -l "GATE FAILED\|Traceback" $R/*/*/xfit_f*.log
if [ -n "$SMOKE" ]; then $PY exp/l10a_finalize.py --run_dir $R --n_boot $B --smoke > $R/final.log 2>&1; else $PY exp/l10a_finalize.py --run_dir $R --n_boot $B > $R/final.log 2>&1; fi; FRC=$?
tail -5 $R/final.log; echo "finalize exit $FRC, end $(date -u +%FT%TZ)"; exit $FRC
