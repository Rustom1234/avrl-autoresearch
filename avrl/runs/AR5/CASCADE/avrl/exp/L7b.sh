#!/usr/bin/env bash
# L7b GPU step: E04 protocol with a 4B judge (Qwen3-4B, thinking disabled) on the 24,000 p0 GSM8K rollouts (vLLM, .venv-judge).
# Started through bin/gpu_run.sh, which writes avrl/runs/L7b/run.log, exit_code and done.flag. Start with:
#   /homes/rdubash/avrl_live/CASCADE/bin/gpu_run.sh /homes/rdubash/avrl_live/CASCADE/avrl/runs/L7b 'bash /homes/rdubash/avrl_live/CASCADE/avrl/exp/L7b.sh'
cd /homes/rdubash/avrl_live/CASCADE || exit 1
mkdir -p avrl/runs/L7b
exec /homes/rdubash/active-audit/.venv-judge/bin/python avrl/exp/l7b_judge_score.py \
  --rollouts avrl/runs/p0/rollouts.parquet --groups avrl/runs/p0/groups.parquet \
  --judge /homes/rdubash/avrl_live/CASCADE/models/Qwen3-4B --out avrl/runs/L7b
