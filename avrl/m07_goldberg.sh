#!/bin/bash
# M07: greedy self-training allocator in training, E01 setting (GSM8K, 1.5B), vs threshold and bandit-flavoured learned. Runs on goldberg after the 5090 open-ended run.
cd /homes/rdubash/active-audit; export CPATH=$HOME/.local/include/python3.12 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
F=avrl/runs/m07.flag; M=/homes/rdubash/active-audit/models/Qwen2.5-1.5B-Instruct
while ! grep -qE 'p1_judge_5090 done|SMOKE FAILED' avrl/runs/p1_judge_5090.flag 2>/dev/null || [ "$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits)" -gt 2000 ] || pgrep -f 'p1_judge_grpo|p1_grpo|p1_code_grpo' >/dev/null; do sleep 120; done
for seed in 0 1; do for arm in greedy threshold learned; do
  echo "=== $arm s$seed start $(hostname) $(date -u)" >> $F
  CUDA_VISIBLE_DEVICES=0 timeout 4h .venv-rl/bin/python avrl/p1_grpo.py --arm $arm --no_vllm --budget 0.25 --steps 100 --prompts_per_step 8 --G 8 --max_completion_length 512 --eval_every 25 --eval_n 300 --ckpt_every 50 --seed $seed --lr 1e-5 --model $M --out avrl/runs/m07_${arm}_s$seed > avrl/runs/m07_${arm}_s$seed.log 2>&1
  echo "$arm s$seed exit $? $(date -u) $(grep -E 'trained .* steps' avrl/runs/m07_${arm}_s$seed.log | tail -1)" >> $F
done; done
echo "m07 done $(date -u)" >> $F
