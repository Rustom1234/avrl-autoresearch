#!/bin/bash
# Memory and step-time probe for the scale-up: Qwen2.5-7B-Instruct policy (LoRA r32, gradient checkpointing)
# on GPU 0, bf16 Qwen2.5-7B-Instruct judge on GPU 1, Skywork 0.6B reward model on GPU 0. 4 real steps at
# 8 prompts x 8 completions x 384 tokens, one evaluation of 50 prompts, then prints peak GPU memory.
#SBATCH --job-name=smoke14b4
#SBATCH --partition=gpu
#SBATCH -N1
#SBATCH -n1
#SBATCH -c 16
#SBATCH --time=01:30:00
#SBATCH --output=%x.%j.log
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 HF_HOME=/nfs/home/rustom/.cache/huggingface
cd ~/avrl; mkdir -p runs; PY=~/avrl/.venv/bin/python
nvidia-smi --query-gpu=index,memory.used --format=csv,noheader
rm -rf runs/smoke14b4
echo "=== start $(date -u)"
CUDA_VISIBLE_DEVICES=0,1 $PY p1_judge_grpo.py --arm threshold --threshold_feature consistency --no_vllm --gradient_checkpointing --policy_single_gpu --micro_batch 4 --budget 0.25 --steps 2 --prompts_per_step 4 --G 8 --max_completion_length 384 --eval_every 2 --eval_n 24 --ckpt_every 1000 --seed 0 --eval_seed 0 --lr 1e-5 --judge bf16-7b --judge_model ~/avrl/models/Qwen2.5-14B-Instruct --judge_device cuda:1 --judge_batch_size 2 --judge_eval_batch_size 2 --rm_model ~/avrl/models/Skywork-Reward-V2-Qwen3-0.6B --model ~/avrl/models/Qwen2.5-7B-Instruct --data ~/avrl/data/e09d_prompts.parquet --out runs/smoke14b4 > runs/smoke14b4.log 2>&1
echo "exit $? $(date -u)"
grep -E "trained .* steps|peak_gpu|eval_max_new|Error|Traceback" runs/smoke14b4.log | tail -6
nvidia-smi --query-gpu=index,memory.used --format=csv,noheader
echo "smoke14b4 done $(date -u)" > runs/smoke14b4.flag
