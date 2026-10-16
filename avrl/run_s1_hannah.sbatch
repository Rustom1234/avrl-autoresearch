#!/bin/bash
# S1 (scale-up): Qwen2.5-7B-Instruct policy, Skywork 0.6B reward model as the cheap grader, Qwen2.5-14B-Instruct
# bf16 judge as the expensive grader, open-ended task (e09d prompts). Two arms per job, each arm on two GPUs
# (policy pinned to the first, judge on the second). Jobs: JOB=a proxy + consistency threshold (seed 0);
# JOB=b full (seed 0) + proxy at learning rate 3e-5 (drift probe); JOB=c consistency threshold + full (seed 1).
# Generation runs at 2 prompts x 8 completions per step (a 4 x 8 batch overflowed the policy card at step 338 on a long
# prompt batch), the reward model sits on the judge card. Submit only through launch_s1_hannah.sh, which chains the jobs with dependencies; the singleton rule below
# also stops two of these from sharing the node.
#SBATCH --job-name=s1_scale
#SBATCH --partition=gpu
#SBATCH -N1
#SBATCH -n1
#SBATCH -c 32
#SBATCH --time=3-00:00:00
#SBATCH --dependency=singleton
#SBATCH --output=%x.%j.log
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 HF_HOME=/nfs/home/rustom/.cache/huggingface
cd ~/avrl; mkdir -p runs; PY=$HOME/avrl/.venv/bin/python
JOB=${JOB:?a, b, c or d}; STEPS=${STEPS:?steps}
F=runs/s1_${JOB}_status.flag
JUDGE=$HOME/avrl/models/Qwen2.5-14B-Instruct
echo "job $JOB  steps $STEPS  judge $JUDGE  $(date -u)"
COMMON="--no_vllm --gradient_checkpointing --policy_single_gpu --micro_batch 4 --budget 0.25 --steps $STEPS --prompts_per_step 2 --G 8 --max_completion_length 384 --eval_every 200 --eval_n 200 --ckpt_every 200 --eval_seed 0 --max_hours 68 --judge bf16-7b --judge_model $JUDGE --judge_device cuda:1 --rm_device cuda:1 --judge_batch_size 2 --judge_eval_batch_size 2 --rm_model $HOME/avrl/models/Skywork-Reward-V2-Qwen3-0.6B --model $HOME/avrl/models/Qwen2.5-7B-Instruct --data $HOME/avrl/data/e09d_prompts.parquet"
# spec fields: gpus name arm seed lr extra
case $JOB in
  a) SPECS=("0,1 proxy_s0 proxy 0 1e-5" "2,3 thrcons_s0 threshold 0 1e-5 --threshold_feature=consistency");;
  b) SPECS=("0,1 full_s0 full 0 1e-5" "2,3 proxy_lr3_s0 proxy 0 3e-5");;
  c) SPECS=("0,1 thrcons_s1 threshold 1 1e-5 --threshold_feature=consistency" "2,3 full_s1 full 1 1e-5");;
  d) SPECS=("0,1 fullrank_s0 full 0 1e-5 --gold_reward_shape=rank" "2,3 thrconsrank_s0 threshold 0 1e-5 --threshold_feature=consistency --gold_reward_shape=rank");;
  *) echo "unknown JOB $JOB"; exit 1;;
esac
nvidia-smi --query-gpu=index,memory.used --format=csv,noheader
echo "=== launching $(date -u)"
PIDS=(); NAMES=()
for spec in "${SPECS[@]}"; do set -- $spec
  gpus=$1; name=$2; arm=$3; seed=$4; lr=$5; shift 5; extra="$*"
  rm -rf runs/s1_$name
  CUDA_VISIBLE_DEVICES=$gpus $PY p1_judge_grpo.py --arm $arm $extra --seed $seed --lr $lr $COMMON --out runs/s1_$name > runs/s1_$name.log 2>&1 &
  PIDS+=($!); NAMES+=($name)
done
RC=""
for i in "${!PIDS[@]}"; do wait ${PIDS[$i]}; rc=$?; RC="$RC ${NAMES[$i]}=$rc"; done
echo "=== finished $(date -u) exit codes:$RC"
for n in "${NAMES[@]}"; do echo "$n: $(grep -E 'trained .* steps|peak_gpu' runs/s1_$n.log | tail -2 | tr '\n' ' ')"; done
echo "arms done $(date -u) exit codes:$RC" > $F
