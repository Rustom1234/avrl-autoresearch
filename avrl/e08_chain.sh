#!/bin/bash
cd /homes/rdubash/active-audit; export CPATH=$HOME/.local/include/python3.12
mkdir -p avrl/runs/E08
timeout 2700 .venv-judge/bin/python avrl/exp/E08_mbpp.py --out avrl/runs/E08/ \
  > avrl/runs/E08/rollouts.log 2>&1
echo "rollouts exit $?" >> avrl/runs/E08/done.flag
.venv/bin/python avrl/exp/E08_analyze.py \
  > avrl/runs/E08/analyze.log 2>&1
echo "analyze exit $?" >> avrl/runs/E08/done.flag
echo "E08 done $(date -u)" >> avrl/runs/E08/done.flag
