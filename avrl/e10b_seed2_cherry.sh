#!/bin/bash
# E10b seed 2 on cherry (A4000): threshold then random arms at 25 percent budget, same config as goldberg seed 1.
cd /homes/rdubash/active-audit; export CPATH=$HOME/.local/include/python3.12 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
F=avrl/runs/e10b_s2.flag; M=/homes/rdubash/active-audit/models/OLMo-2-0425-1B-Instruct
CUR=/homes/rdubash/active-audit/avrl/runs/E10p0/instruct_solvable_task_ids.json
while [ "$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits)" -gt 2000 ] || pgrep -f 'p1_judge_grpo|p1_grpo|p1_code_grpo' >/dev/null; do sleep 120; done
for arm in threshold random; do
  echo "=== $arm s2 start $(hostname) $(date -u)" >> $F
  CUDA_VISIBLE_DEVICES=0 timeout 8h .venv-rl/bin/python avrl/p1_code_grpo.py --arm $arm --no_vllm --budget 0.25 --steps 300 --lr 1e-5 --seed 2 --proxy visible --curriculum $CUR --model $M --out avrl/runs/e10b_${arm}_s2 > avrl/runs/e10b_${arm}_s2.log 2>&1
  echo "$arm s2 exit $? $(date -u) $(grep -E 'trained .* steps' avrl/runs/e10b_${arm}_s2.log | tail -1)" >> $F
done
echo "e10b s2 done $(date -u)" >> $F
