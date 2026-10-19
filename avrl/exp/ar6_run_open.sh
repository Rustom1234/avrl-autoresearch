#!/bin/bash
# AR6 block 2: open-weights cheap tier served by vLLM (OpenAI-compatible). Seed-major order; skips finished runs.
# Usage: ar6_run_open.sh <endpoint> <served-model-name> <tag> [parallel]
cd "$(dirname "$0")/.."
PY=/homes/rdubash/yes/bin/python; EP=$1; M=$2; TAG=$3; PAR=${4:-6}
for s in ${SEEDS:-0 1 2 3 4}; do
  for p in SONNET HAIKU CONSIST3 VERBAL STALL STRONG_EARLY RANDOM25; do
    [ -f runs/AR6/${TAG}_${p}_s${s}/result.json ] || echo "$p $s"
  done
done | xargs -P $PAR -n 2 bash -c '$0 exp/ar6_open_loop.py --policy $1 --seed $2 --max-iters 20 --endpoint '"$EP"' --cheap-model '"$M"' --tag '"$TAG"' > runs/AR6/'"$TAG"'_${1}_s${2}.log 2>&1' $PY
touch runs/AR6/${TAG}.done
