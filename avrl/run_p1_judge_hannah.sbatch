#!/bin/bash
# Hannah (4x A100 40GB) launcher for p1_judge_grpo.py (E09, open-ended/judge-verified phase 1).
# Mirrors ~/avrl/run_p1_all.sbatch's structure (smoke test gate, then four arms on four GPUs).
#
# Differences from p1_grpo.py's job, driven by what's actually on hannah today (checked via
# `ssh hannah` read-only ls/import, 2026-10-01): ~/avrl/.venv has no vllm and no bitsandbytes, so
# this always passes --no_vllm (forces TRL's HF generation path, same as p1_grpo.py would without
# vLLM) and --judge bf16-7b (the 4-bit bitsandbytes judge is goldberg-only).
#
# PREREQUISITE, NOT DONE BY THIS SCRIPT: as of 2026-10-01, ~/avrl/models/ on hannah only has
# Qwen2.5-1.5B-Instruct, and ~/avrl/data/ only has the GSM8K parquets. Before this script can run,
# someone needs to copy to hannah: avrl/models/Qwen2.5-7B-Instruct, avrl/models/Skywork-Reward-
# V2-Qwen3-0.6B, and avrl/data/e09_prompts.parquet (all currently only on goldberg). This script
# does not scp them (no go-ahead was given to push data to another machine; flagged in the task
# report instead).
#SBATCH --job-name=p1_judge_all
#SBATCH --partition=gpu
#SBATCH -N1
#SBATCH -n1
#SBATCH -c 16
#SBATCH --time=08:00:00
#SBATCH --output=%x.%j.log
cd ~/avrl; mkdir -p runs; PY=~/avrl/.venv/bin/python
RM=~/avrl/models/Skywork-Reward-V2-Qwen3-0.6B
JUDGE=~/avrl/models/Qwen2.5-7B-Instruct
# --rm_model/--judge_model are passed explicitly: p1_judge_grpo.py's own defaults point at
# <active-audit>/models/... (goldberg's layout, a sibling of avrl/), but on hannah the models
# live under ~/avrl/models/ instead (same convention p1_grpo.py's own --model default already
# uses there) -- see p1_judge_grpo.py's ROOT/DATA_ROOT comment.
JUDGE_FLAGS="--judge bf16-7b --judge_device cuda --no_vllm --rm_model $RM --judge_model $JUDGE"
nvidia-smi --query-gpu=index,memory.used --format=csv,noheader
echo "=== smoke test (learned arm) $(date -u)"
CUDA_VISIBLE_DEVICES=0 timeout 25m $PY p1_judge_grpo.py --arm learned --smoke $JUDGE_FLAGS --out runs/smoke_judge_learned > runs/smoke_judge_learned.log 2>&1
rc=$?; echo "smoke exit $rc"; tail -5 runs/smoke_judge_learned.log
if [ $rc -ne 0 ] || ! ls runs/smoke_judge_learned/*.jsonl >/dev/null 2>&1; then echo "SMOKE FAILED, not launching arms"; echo "smoke failed $rc" > runs/p1_judge_status.flag; exit 1; fi
echo "=== launching four arms $(date -u)"
for spec in "0 full" "1 random" "2 threshold" "3 learned"; do set -- $spec
  CUDA_VISIBLE_DEVICES=$1 $PY p1_judge_grpo.py --arm $2 --budget 0.25 --steps 150 --seed 0 $JUDGE_FLAGS --out runs/p1_judge_$2 > runs/p1_judge_$2.log 2>&1 &
done
wait; echo "arms done $(date -u)" > runs/p1_judge_status.flag
