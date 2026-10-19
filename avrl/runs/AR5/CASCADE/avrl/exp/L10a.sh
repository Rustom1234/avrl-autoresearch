#!/usr/bin/env bash
# L10a (CPU, local shell, no GPU, no ssh): oracle-ceiling replay for claim 2. Pre-registered in avrl/RESEARCH_LOG.md
# ("L10a pre-registration"). Runs the input checks, then the primary (E01 proxy_error added to the learned features)
# and a control (feature omitted, must reproduce the E10 replay) in parallel, each under a 45 min cap.
# Writes avrl/runs/L10a/run.log, exit_code and done.flag. Start with:
#   nohup bash /homes/rdubash/avrl_live/CASCADE/avrl/exp/L10a.sh > /dev/null 2>&1 < /dev/null &
cd /homes/rdubash/avrl_live/CASCADE || exit 1
PY=/homes/rdubash/active-audit/.venv/bin/python
OUT=avrl/runs/L10a
COMMON="--feats avrl/runs/E10/feats.parquet --cell cell100 --features cheap+emb --budgets 0.10,0.25,0.50 --seeds 8 --boot 200 --boot_budget 0.25 --workers 8"
mkdir -p "$OUT/control_no_e01"
rm -f "$OUT/exit_code" "$OUT/done.flag"
$PY avrl/exp/l10a_checks.py > "$OUT/run.log" 2>&1
CHECK=$?
timeout -k 30 2700 $PY avrl/exp/l10a_oracle_ceiling.py $COMMON --out "$OUT/control_no_e01" > "$OUT/control_no_e01/run.log" 2>&1 &
P_CTRL=$!
timeout -k 30 2700 $PY avrl/exp/l10a_oracle_ceiling.py $COMMON --e01_feats avrl/runs/p0/feats.parquet --out "$OUT" >> "$OUT/run.log" 2>&1 &
P_MAIN=$!
wait $P_CTRL; C_CTRL=$?
wait $P_MAIN; C_MAIN=$?
echo "checks=$CHECK control=$C_CTRL primary=$C_MAIN" > "$OUT/exit_codes.txt"
echo $(( CHECK | C_CTRL | C_MAIN )) > "$OUT/exit_code"
touch "$OUT/done.flag"
