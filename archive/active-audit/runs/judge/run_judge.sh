#!/bin/bash
# waits for the hyp scoring chain to release the GPU, then runs the judge with user-local Python headers for Triton
cd /homes/rdubash/active-audit
export CPATH=$HOME/.local/include/python3.12
while pgrep -f 'runs/hyp/chain.sh' >/dev/null; do sleep 30; done
source .venv-judge/bin/activate
python aa/judge_label.py --model models/Qwen3-4B --n 20000 --seed 0 --out runs/judge/labels_20k.parquet > runs/judge/label_20k.log 2>&1
echo "judge exit $?" >> runs/judge/done.flag
