#!/bin/bash
# Phase 1 on goldberg (RTX A4000 16GB, HF generation): the two decisive arms, sequential, half batch.
# threshold (fixed rule: verify the lowest-agreement groups) vs learned (online allocator), 25% budget, 100 steps.
cd /homes/rdubash/active-audit; export CPATH=$HOME/.local/include/python3.12
# wait for the autoresearch loop to finish (loop_done.flag, written when its workflow reports) and for the GPU to be idle
while [ ! -f avrl/runs/loop_done.flag ] || [ "$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits)" -gt 2000 ] || pgrep -f 'avrl/exp/' >/dev/null; do sleep 60; done
for arm in threshold learned; do
  echo "=== $arm start $(date -u)" >> avrl/runs/p1_goldberg_s1.flag
  CUDA_VISIBLE_DEVICES=0 timeout 5h .venv-rl/bin/python avrl/p1_grpo.py --arm $arm --no_vllm --budget 0.25 --steps 100 --prompts_per_step 8 --G 8 --eval_every 25 --eval_n 300 --ckpt_every 50 --seed 1 --out avrl/runs/p1_${arm}_s1 --model models/Qwen2.5-1.5B-Instruct > avrl/runs/p1_${arm}_s1.log 2>&1
  echo "$arm exit $? $(date -u) $(grep -E 'trained .* steps' avrl/runs/p1_${arm}_s1.log | tail -1)" >> avrl/runs/p1_goldberg_s1.flag
done
echo "p1_goldberg_s1 done $(date -u)" >> avrl/runs/p1_goldberg_s1.flag
