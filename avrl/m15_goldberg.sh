#!/bin/bash
# M15: two more paired comparisons of the greedy self-trainer vs the consistency threshold in the low-error
# regime (open-ended task, 4-bit 7B judge, 150 steps, same configuration as the 5090b panel), seeds 2 and 3,
# with the fixed evaluation (--eval_seed 0, full-length eval responses).
cd /homes/rdubash/active-audit; export CPATH=$HOME/.local/include/python3.12 FLASHINFER_DISABLE_VERSION_CHECK=1 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
F=avrl/runs/m15.flag; M=/homes/rdubash/active-audit/models/Qwen2.5-1.5B-Instruct; P=avrl/data/e09b_prompts.parquet
while [ "$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits)" -gt 2000 ] || pgrep -f 'p1_judge_grpo|p1_grpo|p1_code_grpo|reeval_adapters' >/dev/null; do sleep 120; done
COMMON="--no_vllm --budget 0.25 --steps 150 --prompts_per_step 8 --G 8 --max_completion_length 384 --eval_every 25 --eval_n 200 --ckpt_every 50 --lr 1e-5 --judge 4bit-7b --judge_model /homes/rdubash/active-audit/models/Qwen2.5-7B-Instruct --judge_device cuda --model $M --data $P --eval_seed 0"
for seed in 2 3; do for spec in "greedy:" "threshold:--threshold_feature consistency"; do arm=${spec%%:*}; extra=${spec#*:}; name=$([ "$arm" = threshold ] && echo thrcons || echo greedy)
  echo "=== $name s$seed start $(hostname) $(date -u)" >> $F
  rm -rf avrl/runs/p1_judge5090b_${name}_s${seed}
  CUDA_VISIBLE_DEVICES=0 timeout 4h .venv-rl/bin/python avrl/p1_judge_grpo.py --arm $arm $extra $COMMON --seed $seed --out avrl/runs/p1_judge5090b_${name}_s${seed} > avrl/runs/p1_judge5090b_${name}_s${seed}.log 2>&1
  echo "$name s$seed exit $? $(date -u) $(grep -E 'trained .* steps' avrl/runs/p1_judge5090b_${name}_s${seed}.log | tail -1)" >> $F
done; done
echo "m15 done $(date -u)" >> $F
