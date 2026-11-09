#!/usr/bin/env bash
# E06 GPU step: fresh GSM8K rollouts from Qwen2.5-0.5B-Instruct (same family as the E01 policy, Qwen2.5-1.5B-Instruct, at 0.5B).
# Same 3,000 train prompts as E01 (sample seed 0), G 8, max_tokens 512. p0_rollouts.py is unchanged.
# Started through bin/gpu_run.sh, which writes avrl/runs/E06/run.log, exit_code and done.flag. Start with:
#   /homes/rdubash/avrl_live/CASCADE/bin/gpu_run.sh /homes/rdubash/avrl_live/CASCADE/avrl/runs/E06 'bash /homes/rdubash/avrl_live/CASCADE/avrl/exp/E06.sh'
cd /homes/rdubash/avrl_live/CASCADE || exit 1
mkdir -p avrl/runs/E06
exec /homes/rdubash/active-audit/.venv-judge/bin/python avrl/p0_rollouts.py \
  --model /homes/rdubash/active-audit/models/Qwen2.5-0.5B-Instruct \
  --n_prompts 3000 --G 8 --max_tokens 512 --seed 0 --out avrl/runs/E06
