#!/bin/bash
# Phase 1 v2 (judge-reward, E09 task) on goldberg: smoke first, then threshold and learned arms if smoke passes.
cd /homes/rdubash/active-audit; export CPATH=$HOME/.local/include/python3.12 FLASHINFER_DISABLE_VERSION_CHECK=1 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
F=avrl/runs/p1_judge_goldberg_s1.flag
while ! grep -qE "E09d done|ledger exit" avrl/runs/E09d/done.flag 2>/dev/null || [ "$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits)" -gt 2000 ]; do sleep 60; done
M=/homes/rdubash/active-audit/models/Qwen2.5-1.5B-Instruct; P=avrl/data/e09b_prompts.parquet
echo "=== smoke start $(date -u)" >> $F
CUDA_VISIBLE_DEVICES=0 timeout 40m .venv-rl/bin/python avrl/p1_judge_grpo.py --arm learned --smoke --no_vllm --judge 4bit-7b --judge_model /homes/rdubash/active-audit/models/Qwen2.5-3B-Instruct --judge_device cuda --model $M --data $P --out avrl/runs/smoke_judge_learned_s1 > avrl/runs/smoke_judge_learned_s1.log 2>&1
rc=$?; echo "smoke exit $rc $(grep -E 'trained .* steps|peak_gpu' avrl/runs/smoke_judge_learned_s1.log | tr '\n' ' ')" >> $F
if [ $rc -ne 0 ]; then echo "SMOKE FAILED, arms not launched" >> $F; exit 1; fi
for arm in threshold learned; do
  echo "=== $arm start $(date -u)" >> $F
  CUDA_VISIBLE_DEVICES=0 timeout 6h .venv-rl/bin/python avrl/p1_judge_grpo.py --arm $arm --no_vllm --budget 0.25 --steps 100 --prompts_per_step 8 --G 6 --max_completion_length 320 --eval_every 25 --eval_n 200 --ckpt_every 50 --seed 1 --judge 4bit-7b --judge_model /homes/rdubash/active-audit/models/Qwen2.5-3B-Instruct --judge_device cuda --model $M --data $P --out avrl/runs/p1_judge_${arm}_s1 > avrl/runs/p1_judge_${arm}_s1.log 2>&1
  echo "$arm exit $? $(date -u) $(grep -E 'trained .* steps' avrl/runs/p1_judge_${arm}_s1.log | tail -1)" >> $F
done
echo "p1_judge_s1 done $(date -u)" >> $F
