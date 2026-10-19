#!/usr/bin/env bash
# E04 GPU step: judge scoring of the 24,000 p0 rollouts with Qwen2.5-0.5B-Instruct (vLLM, .venv-judge).
# Started through bin/gpu_run.sh, which writes avrl/runs/E04/run.log, exit_code and done.flag. Start with:
#   /homes/rdubash/avrl_live/CASCADE/bin/gpu_run.sh /homes/rdubash/avrl_live/CASCADE/avrl/runs/E04 'bash /homes/rdubash/avrl_live/CASCADE/avrl/exp/E04.sh'
cd /homes/rdubash/avrl_live/CASCADE || exit 1
mkdir -p avrl/runs/E04
exec /homes/rdubash/active-audit/.venv-judge/bin/python avrl/exp/e04_judge_score.py \
  --rollouts avrl/runs/p0/rollouts.parquet --groups avrl/runs/p0/groups.parquet \
  --judge /homes/rdubash/avrl_live/CASCADE/models/Qwen2.5-0.5B-Instruct --out avrl/runs/E04
