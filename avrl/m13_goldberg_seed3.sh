#!/bin/bash
# Third seed of the open-ended full (absolute reward) and consistency-threshold arms on goldberg (32 GB),
# because seeds 2 and 3 of this configuration OOM on the 16 GB cards. Runs after E10e.
cd /homes/rdubash/active-audit; export CPATH=$HOME/.local/include/python3.12 FLASHINFER_DISABLE_VERSION_CHECK=1 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
F=avrl/runs/m13.flag; M=/homes/rdubash/active-audit/models/Qwen2.5-1.5B-Instruct; P=avrl/data/e09b_prompts.parquet
until grep -q "e10e done" avrl/runs/e10b_goldberg.flag 2>/dev/null; do sleep 120; done
while [ "$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits)" -gt 2000 ] || pgrep -f 'p1_judge_grpo|p1_grpo|p1_code_grpo' >/dev/null; do sleep 120; done
COMMON="--no_vllm --budget 0.25 --steps 100 --prompts_per_step 8 --G 6 --max_completion_length 320 --eval_every 25 --eval_n 200 --ckpt_every 50 --seed 3 --lr 1e-5 --judge 4bit-7b --judge_model /homes/rdubash/active-audit/models/Qwen2.5-3B-Instruct --judge_device cuda --model $M --data $P"
for spec in "full:" "threshold:--threshold_feature consistency"; do arm=${spec%%:*}; extra=${spec#*:}; name=$([ "$arm" = threshold ] && echo thrcons || echo full)
  echo "=== $name s3 start $(hostname) $(date -u)" >> $F
  CUDA_VISIBLE_DEVICES=0 timeout 4h .venv-rl/bin/python avrl/p1_judge_grpo.py --arm $arm $extra $COMMON --out avrl/runs/p1_judge_${name}_lr5_s3 > avrl/runs/p1_judge_${name}_lr5_s3.log 2>&1
  echo "$name s3 exit $? $(date -u) $(grep -E 'trained .* steps' avrl/runs/p1_judge_${name}_lr5_s3.log | tail -1)" >> $F
done
echo "m13 done $(date -u)" >> $F
