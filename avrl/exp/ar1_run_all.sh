#!/bin/bash
# AR1: 3 arms x 5 seeds, five runs at a time. Writes runs/AR1/<arm>_s<seed>/ and runs/AR1/all.done.
cd "$(dirname "$0")/.."
PY=../.venv/bin/python
for arm in A B C; do for s in 0 1 2 3 4; do echo "$arm $s"; done; done | \
  xargs -P 5 -n 2 bash -c '$0 exp/ar1_loop.py --arm $1 --seed $2 --max-iters 20 > runs/AR1/${1}_s${2}.log 2>&1' $PY
touch runs/AR1/all.done
