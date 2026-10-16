#!/bin/bash
# E10b seed 1 part 2 on goldberg: threshold then random arms (25 percent budget), after the proxy and
# full arms (avrl/e10b_goldberg.sh) finish. Gives two seeds for all four E10b arms with hannah's seed 0.
cd /homes/rdubash/active-audit; export CPATH=$HOME/.local/include/python3.12 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
F=avrl/runs/e10b_goldberg.flag; M=/homes/rdubash/active-audit/models/OLMo-2-0425-1B-Instruct
CUR=/homes/rdubash/active-audit/avrl/runs/E10p0/instruct_solvable_task_ids.json
until grep -q "e10b goldberg done" $F 2>/dev/null; do sleep 120; done
while [ "$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits)" -gt 2000 ] || pgrep -f 'p1_judge_grpo|p1_grpo|p1_code_grpo' >/dev/null; do sleep 120; done
for arm in threshold random; do
  echo "=== $arm s1 start $(hostname) $(date -u)" >> $F
  CUDA_VISIBLE_DEVICES=0 timeout 6h .venv-rl/bin/python avrl/p1_code_grpo.py --arm $arm --no_vllm --budget 0.25 --steps 300 --lr 1e-5 --seed 1 --proxy visible --curriculum $CUR --model $M --out avrl/runs/e10b_${arm}_s1 > avrl/runs/e10b_${arm}_s1.log 2>&1
  echo "$arm s1 exit $? $(date -u) $(grep -E 'trained .* steps' avrl/runs/e10b_${arm}_s1.log | tail -1)" >> $F
done
echo "e10b goldberg part2 done $(date -u)" >> $F
