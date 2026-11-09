#!/bin/bash
# Powered open-ended comparison, rerun with the 4-bit 7B judge (the bf16 judge beside training hit OOM at step 145 of 150), G=8, 150 steps, lr 1e-5.
# Runs ON goldberg (launch via ssh goldberg). Waits for the E10 smoke/evals there to finish and the GPU to be idle.
cd /homes/rdubash/active-audit; export CPATH=$HOME/.local/include/python3.12 FLASHINFER_DISABLE_VERSION_CHECK=1 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
F=avrl/runs/p1_judge_5090b.flag; M=/homes/rdubash/active-audit/models/Qwen2.5-1.5B-Instruct; P=avrl/data/e09b_prompts.parquet
while ! grep -qE 'p1_judge_lr5_s2 done|exit [1-9]' avrl/runs/p1_judge_goldberg_lr5_s2.flag 2>/dev/null || [ "$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits)" -gt 2000 ] || pgrep -f 'p1_code_grpo|p1_judge_grpo' >/dev/null; do sleep 60; done
echo "=== smoke start $(hostname) $(date -u)" >> $F
CUDA_VISIBLE_DEVICES=0 timeout 40m .venv-rl/bin/python avrl/p1_judge_grpo.py --arm learned --smoke --no_vllm --judge 4bit-7b --judge_device cuda --model $M --data $P --out avrl/runs/smoke_judge_5090b > avrl/runs/smoke_judge_5090b.log 2>&1
rc=$?; echo "smoke exit $rc $(grep -E 'trained .* steps|peak_gpu' avrl/runs/smoke_judge_5090b.log | tr '\n' ' ')" >> $F
if [ $rc -ne 0 ]; then echo "SMOKE FAILED" >> $F; exit 1; fi
for arm in threshold learned; do
  echo "=== $arm start $(date -u)" >> $F
  CUDA_VISIBLE_DEVICES=0 timeout 8h .venv-rl/bin/python avrl/p1_judge_grpo.py --arm $arm --no_vllm --budget 0.25 --steps 150 --prompts_per_step 8 --G 8 --max_completion_length 384 --eval_every 25 --eval_n 200 --ckpt_every 50 --seed 0 --lr 1e-5 --judge 4bit-7b --judge_device cuda --model $M --data $P --out avrl/runs/p1_judge5090b_$arm > avrl/runs/p1_judge5090b_$arm.log 2>&1
  echo "$arm exit $? $(date -u) $(grep -E 'trained .* steps' avrl/runs/p1_judge5090b_$arm.log | tail -1)" >> $F
done
echo "p1_judge_5090b done $(date -u)" >> $F
