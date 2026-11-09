#!/bin/bash
# L1a warm-start replay, CPU only. 12 sim jobs (xargs -P 6), table + verdict, optional pool bootstrap on the triggered headline cell, table again.
set -uo pipefail
cd /homes/rdubash/avrl_live/FULL/avrl
PY=/homes/rdubash/active-audit/.venv/bin/python
E21=runs/E21-lowT-confident-errors; OUT=runs/L1a-warm-start-replay; mkdir -p $OUT; rm -f $OUT/done.flag $OUT/exit_code
CHEAP=agreement,n_distinct,mean_tokens,std_tokens,any_truncated,log_qlen
EMBC=$(seq -s, -f "pc%g" 0 15)
for T in 0.25 0.5 1.0; do for C in cell30 cell100; do for F in cheap cheap_emb; do echo "$T $C $F"; done; done; done |
  xargs -P 6 -L 1 bash -c 'T=$0; C=$1; F=$2; COLS='$CHEAP'; [ "$F" = cheap_emb ] && COLS='$CHEAP,$EMBC';
    '$PY' exp/sim_warm.py '$E21'/feats_T$T.parquet proxy_error $C $COLS '$OUT'/sim_T${T}_${C}_${F}.json > '$OUT'/sim_T${T}_${C}_${F}.log 2>&1 || echo "FAILED $T $C $F" >&2'
$PY exp/l1a_table.py $OUT > $OUT/table.log 2>&1 || { echo "table failed"; tail -20 $OUT/table.log; echo 1 > $OUT/exit_code; touch $OUT/done.flag; exit 1; }
BCS=$($PY -c "import json; print(' '.join(json.load(open('$OUT/result.json'))['bootstrap_trigger']['bootstrap_cells']))")
for BC in $BCS; do
  T=${BC#T}; T=${T%%_*}; REST=${BC#T${T}_}; C=${REST%%_*}; F=${REST#${C}_}; COLS=$CHEAP; [ "$F" = cheap_emb ] && COLS=$CHEAP,$EMBC
  echo "bootstrap on $BC (T=$T C=$C F=$F)"
  BOOT_B=50 BOOT_SEEDS=2 $PY exp/sim_warm.py $E21/feats_T$T.parquet proxy_error $C $COLS $OUT/boot_$BC.json > $OUT/boot_$BC.log 2>&1 || echo "bootstrap FAILED $BC" >&2
done
if [ -n "$BCS" ]; then $PY exp/l1a_table.py $OUT > $OUT/table.log 2>&1 || { echo "table (2) failed"; tail -20 $OUT/table.log; echo 1 > $OUT/exit_code; touch $OUT/done.flag; exit 1; }; fi
echo 0 > $OUT/exit_code; touch $OUT/done.flag; tail -4 $OUT/table.log
