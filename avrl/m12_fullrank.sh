#!/bin/bash
# M12: full-verification arm with rank-shaped judge rewards (top half of each group by judge score), to
# separate label quality from reward shape on the open-ended 3B-judge task. Usage: bash avrl/m12_fullrank.sh <seed>
# Gated on the harness-ready flag and a free GPU (runs after M11 on the same box).
SEED=${1:?seed}
cd /homes/rdubash/active-audit; export CPATH=$HOME/.local/include/python3.12 FLASHINFER_DISABLE_VERSION_CHECK=1 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
F=avrl/runs/m12_s${SEED}.flag
M=/homes/rdubash/active-audit/models/Qwen2.5-1.5B-Instruct; P=avrl/data/e09b_prompts.parquet
until [ -f avrl/runs/m12_harness_ready.flag ]; do sleep 60; done
while [ "$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits)" -gt 2000 ] || pgrep -f "p1_code_grpo|p1_judge_grpo|p1_grpo" >/dev/null; do sleep 60; done
echo "=== full_rank s$SEED start $(hostname) $(date -u)" >> $F
CUDA_VISIBLE_DEVICES=0 timeout 6h .venv-rl/bin/python avrl/p1_judge_grpo.py --arm full --gold_reward_shape rank --no_vllm --budget 0.25 --steps 100 --prompts_per_step 8 --G 6 --max_completion_length 320 --eval_every 25 --eval_n 200 --ckpt_every 50 --seed $SEED --lr 1e-5 --judge 4bit-7b --judge_model /homes/rdubash/active-audit/models/Qwen2.5-3B-Instruct --judge_device cuda --model $M --data $P --out avrl/runs/p1_judge_fullrank_lr5_s${SEED} > avrl/runs/p1_judge_fullrank_lr5_s${SEED}.log 2>&1
echo "full_rank s$SEED exit $? $(date -u) $(grep -E 'trained .* steps' avrl/runs/p1_judge_fullrank_lr5_s${SEED}.log | tail -1)" >> $F
echo "m12 s$SEED done $(date -u)" >> $F
