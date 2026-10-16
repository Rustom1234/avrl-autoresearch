#!/bin/bash
# after the influence test: judge the whole domain-stratified hyp sample so all labels align on the same docs
cd /homes/rdubash/active-audit
export CPATH=$HOME/.local/include/python3.12
while [ ! -f runs/infl/done.flag ] || ! grep -q 'analyze exit' runs/infl/done.flag; do sleep 30; done
source .venv-judge/bin/activate
python aa/judge_label.py --model models/Qwen3-4B --ids runs/hyp/scores.parquet --out runs/judge/labels_hyp.parquet > runs/judge/label_hyp.log 2>&1
echo "judge_hyp exit $?" >> runs/judge/done_hyp.flag
