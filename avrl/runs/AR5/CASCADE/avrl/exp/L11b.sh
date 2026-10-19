#!/usr/bin/env bash
# L11b (CPU, local shell, no GPU, no ssh): low-dimensional online learner and labeled static ranker with per-seed refit.
# Pre-registered in avrl/RESEARCH_LOG.md ("L11b pre-registration"). Runs five invocations of avrl/exp/l11b_low_dim_ceiling.py
# in parallel (control, primary, ceiling, and two reference invocations), then avrl/exp/l11b_assemble.py, which checks
# the control against L10a and writes avrl/runs/L11b/result.json. Writes avrl/runs/L11b/run.log, exit_code and done.flag.
# Start with:
#   nohup bash /homes/rdubash/avrl_live/CASCADE/avrl/exp/L11b.sh > /dev/null 2>&1 < /dev/null &
# or run it in the foreground (about 1 minute).
cd /homes/rdubash/avrl_live/CASCADE || exit 1
PY=/homes/rdubash/active-audit/.venv/bin/python
export PYTHONDONTWRITEBYTECODE=1
OUT=avrl/runs/L11b
COMMON="--feats avrl/runs/E10/feats.parquet --cell cell100 --budgets 0.10,0.25,0.50 --seeds 8 --boot 200 --boot_budget 0.25 --workers 8"
E01="--e01_feats avrl/runs/p0/feats.parquet"
mkdir -p "$OUT/control_cheap_emb" "$OUT/primary" "$OUT/ceiling" "$OUT/ref_agree_nd" "$OUT/ref_cheap_e01"
rm -f "$OUT/done.flag" "$OUT/exit_code"

run() {  # run <name> <extra args...>: one invocation, own log and exit code
  local name=$1; shift
  timeout -k 30 2700 $PY avrl/exp/l11b_low_dim_ceiling.py $COMMON "$@" --out "$OUT/$name" > "$OUT/$name/run.log" 2>&1
  echo $? > "$OUT/$name/exit_code"
}

run control_cheap_emb --features cheap+emb &
run primary --features cheap &
run ceiling --features cheap+emb $E01 &
run ref_agree_nd --features agree_nd &
run ref_cheap_e01 --features cheap $E01 &
wait

$PY avrl/exp/l11b_assemble.py > "$OUT/assemble.log" 2>&1
ASM=$?

{
  for n in control_cheap_emb primary ceiling ref_agree_nd ref_cheap_e01; do
    echo "== $n (exit $(cat "$OUT/$n/exit_code"))"
    cat "$OUT/$n/run.log"
    echo
  done
  echo "== assemble (exit $ASM)"
  cat "$OUT/assemble.log"
} > "$OUT/run.log"

CODE=$ASM
for n in control_cheap_emb primary ceiling ref_agree_nd ref_cheap_e01; do
  c=$(cat "$OUT/$n/exit_code"); [ "$c" -ne 0 ] && CODE=$c
done
echo "$CODE" > "$OUT/exit_code"
touch "$OUT/done.flag"
