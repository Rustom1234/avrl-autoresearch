#!/bin/bash
# sequential: featurize (RAM-heavy, CPU) -> density analysis -> judge labels (GPU)
cd /homes/rdubash/active-audit
source .venv/bin/activate
python runs/density/featurize_all.py > runs/density/featurize.log 2>&1 && python runs/density/analyze.py > runs/density/analyze.log 2>&1
echo "density exit $?" >> runs/chain_night.log
deactivate
source .venv-judge/bin/activate
python aa/judge_label.py --model models/Qwen3-4B --n 20000 --seed 0 --out runs/judge/labels_20k.parquet > runs/judge/label_20k.log 2>&1
echo "judge exit $?" >> runs/chain_night.log
