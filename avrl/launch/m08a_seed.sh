#!/bin/bash
# Phase 1 v2 (judge-reward, E09 task) on goldberg: smoke first, then threshold and learned arms if smoke passes.
cd /homes/rdubash/active-audit; export CPATH=$HOME/.local/include/python3.12 FLASHINFER_DISABLE_VERSION_CHECK=1 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
SEED=${1:?seed}
F=avrl/runs/m08a_s${SEED}.flag
while [ "$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits)" -gt 2000 ] || pgrep -f "p1_code_grpo|p1_judge_grpo" >/dev/null; do sleep 60; done
M=/homes/rdubash/active-audit/models/Qwen2.5-1.5B-Instruct; P=avrl/data/e09b_prompts.parquet
for arm in greedy; do
  echo "=== $arm start $(date -u)" >> $F
  CUDA_VISIBLE_DEVICES=0 timeout 6h .venv-rl/bin/python avrl/p1_judge_grpo.py --arm $arm --no_vllm --budget 0.25 --steps 100 --prompts_per_step 8 --G 6 --max_completion_length 320 --eval_every 25 --eval_n 200 --ckpt_every 50 --seed $SEED --lr 1e-5 --judge 4bit-7b --judge_model /homes/rdubash/active-audit/models/Qwen2.5-3B-Instruct --judge_device cuda --model $M --data $P --out avrl/runs/p1_judge_${arm}_lr5_s${SEED} > avrl/runs/p1_judge_${arm}_lr5_s${SEED}.log 2>&1
  echo "$arm exit $? $(date -u) $(grep -E 'trained .* steps' avrl/runs/p1_judge_${arm}_lr5_s${SEED}.log | tail -1)" >> $F
done
echo "m08a_s${SEED} done $(date -u)" >> $F
