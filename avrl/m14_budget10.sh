#!/bin/bash
# M14: vote-agreement threshold at 10 percent budget, GSM8K 1.5B, M07 configuration (dose-response for claim 1).
# Usage: bash avrl/m14_budget10.sh <seed>   (spruce 0, cherry 1; seeds 2 and 3 OOM on A4000s)
S=${1:?seed}
cd /homes/rdubash/active-audit; export CPATH=$HOME/.local/include/python3.12 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
F=avrl/runs/m14_s$S.flag; M=/homes/rdubash/active-audit/models/Qwen2.5-1.5B-Instruct
while [ "$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits)" -gt 2000 ] || pgrep -f 'p1_judge_grpo|p1_grpo|p1_code_grpo' >/dev/null; do sleep 120; done
echo "=== threshold b0.10 s$S start $(hostname) $(date -u)" >> $F
CUDA_VISIBLE_DEVICES=0 timeout 5h .venv-rl/bin/python avrl/p1_grpo.py --arm threshold --no_vllm --budget 0.10 --steps 100 --prompts_per_step 8 --G 8 --max_completion_length 512 --eval_every 25 --eval_n 300 --ckpt_every 50 --seed $S --lr 1e-5 --model $M --out avrl/runs/m14_threshold_b10_s$S > avrl/runs/m14_threshold_b10_s$S.log 2>&1
echo "threshold b0.10 s$S exit $? $(date -u) $(grep -E 'trained .* steps' avrl/runs/m14_threshold_b10_s$S.log | tail -1)" >> $F
echo "m14 s$S done $(date -u)" >> $F
