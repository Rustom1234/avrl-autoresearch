#!/bin/bash
# M10: full-verification and proxy-only reference arms for the GSM8K 1.5B setting (M07 config).
# Usage: bash avrl/m10_refs.sh <seed>   (spruce runs seed 0, cherry runs seed 1)
# Gives claim 1 its exact bound (full) and the proxy-only drift curve for the canary premise (M04).
S=${1:?seed}
cd /homes/rdubash/active-audit; export CPATH=$HOME/.local/include/python3.12 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
F=avrl/runs/m10_s$S.flag; M=/homes/rdubash/active-audit/models/Qwen2.5-1.5B-Instruct
while [ "$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits)" -gt 2000 ] || pgrep -f 'p1_judge_grpo|p1_grpo|p1_code_grpo' >/dev/null; do sleep 120; done
for arm in full proxy; do
  echo "=== $arm s$S start $(hostname) $(date -u)" >> $F
  CUDA_VISIBLE_DEVICES=0 timeout 5h .venv-rl/bin/python avrl/p1_grpo.py --arm $arm --no_vllm --budget 0.25 --steps 100 --prompts_per_step 8 --G 8 --max_completion_length 512 --eval_every 25 --eval_n 300 --ckpt_every 50 --seed $S --lr 1e-5 --model $M --out avrl/runs/m10_${arm}_s$S > avrl/runs/m10_${arm}_s$S.log 2>&1
  echo "$arm s$S exit $? $(date -u) $(grep -E 'trained .* steps' avrl/runs/m10_${arm}_s$S.log | tail -1)" >> $F
done
echo "m10 s$S done $(date -u)" >> $F
