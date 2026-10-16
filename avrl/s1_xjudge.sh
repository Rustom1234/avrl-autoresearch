#!/bin/bash
# One unit of the cross-family judge check, meant for a 16 GB card.
#   bash avrl/s1_xjudge.sh gen <arm>                     arm = base or an S1 run name; writes responses.jsonl
#   bash avrl/s1_xjudge.sh judge <set> <tag> <model_id>  scores <set>/responses.jsonl with one judge
set -u
cd "$HOME/active-audit" || exit 1
X=avrl/runs/s1_xjudge
PY=.venv/bin/python
POLICY_QUANT=${POLICY_QUANT:-8bit}
JUDGE_QUANT=${JUDGE_QUANT:-4bit}
case "$1" in
  gen)
    a=$2; mkdir -p $X/$a
    [ -f $X/$a/gen.done ] && { echo "skip gen $a"; exit 0; }
    if [ "$a" = base ]; then ad=""; else ad="--adapter avrl/runs/hannah_s1/$a/final"; fi
    echo "=== gen $a on $(hostname -s) $(date -u)"
    $PY avrl/exp/S1_base_eval.py --stage gen $ad --policy_quant $POLICY_QUANT --out $X/$a > $X/$a/gen.log 2>&1 \
      && echo "=== gen $a done $(date -u)" || echo "GEN FAILED $a on $(hostname -s)";;
  judge)
    set=$2; tag=$3; model=$4; out=$X/$set/$tag
    [ -f $out/judge.done ] && { echo "skip judge $set/$tag"; exit 0; }
    [ -f $X/$set/responses.jsonl ] || { echo "JUDGE FAILED $set/$tag: no responses"; exit 1; }
    mkdir -p $out
    echo "=== judge $set/$tag on $(hostname -s) $(date -u)"
    $PY avrl/exp/S1_base_eval.py --stage judge --judge_model "$model" --judge_quant $JUDGE_QUANT --judge_batch_size 2 \
      --responses $X/$set/responses.jsonl --out $out > $out/judge.log 2>&1 \
      && echo "=== judge $set/$tag done $(date -u): $(grep -o '"judge_score_mean": [0-9.]*' $out/result.json)" \
      || echo "JUDGE FAILED $set/$tag on $(hostname -s)";;
  *) echo "usage: gen <arm> | judge <set> <tag> <model_id>"; exit 1;;
esac
