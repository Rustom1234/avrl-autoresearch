#!/bin/bash
# AR6 block 1: Haiku as the cheap tier. Seed-major order so partial completion is balanced across policies;
# skips runs that already have a result.json.
cd "$(dirname "$0")/.."
PY=/homes/rdubash/yes/bin/python
for s in 0 1 2 3 4 5 6 7 8 9; do
  for p in SONNET HAIKU CONSIST3 VERBAL STALL STRONG_EARLY RANDOM25; do
    [ -f runs/AR6/${p}_s${s}/result.json ] || echo "$p $s"
  done
done | xargs -P 10 -n 2 bash -c '$0 exp/ar6_route_loop.py --policy $1 --seed $2 --max-iters 20 > runs/AR6/${1}_s${2}.log 2>&1' $PY
touch runs/AR6/block1.done
