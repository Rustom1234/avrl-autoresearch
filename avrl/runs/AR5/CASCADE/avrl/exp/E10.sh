#!/usr/bin/env bash
# E10 GPU step: fresh GSM8K rollouts from OLMo-2-0425-1B-Instruct (a different family, substituting for Llama-3.2-3B,
# which is not on disk). Same 3,000 train prompts as E01 (sample seed 0), G 8, max_tokens 512. p0_rollouts.py is unchanged.
# Started through bin/gpu_run.sh, which writes avrl/runs/E10/run.log, exit_code and done.flag. Start with:
#   /homes/rdubash/avrl_live/CASCADE/bin/gpu_run.sh /homes/rdubash/avrl_live/CASCADE/avrl/runs/E10 'bash /homes/rdubash/avrl_live/CASCADE/avrl/exp/E10.sh'
cd /homes/rdubash/avrl_live/CASCADE || exit 1
mkdir -p avrl/runs/E10
exec /homes/rdubash/active-audit/.venv-judge/bin/python avrl/p0_rollouts.py \
  --model /homes/rdubash/active-audit/models/OLMo-2-0425-1B-Instruct \
  --n_prompts 3000 --G 8 --max_tokens 512 --seed 0 --out avrl/runs/E10
