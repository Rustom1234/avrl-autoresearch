#!/usr/bin/env bash
# L13a GPU step (copy of E06.sh, all 7,473 train prompts): fresh GSM8K rollouts from Qwen2.5-0.5B-Instruct (same family as the E01 policy, Qwen2.5-1.5B-Instruct, at 0.5B).
# Same policy and rollout settings as E06; prompts are all 7,473 train prompts (sample seed 0, a full permutation), G 8, max_tokens 512. p0_rollouts.py is unchanged.
# Started through bin/gpu_run.sh, which writes avrl/runs/L13a/run.log, exit_code and done.flag. Start with:
#   /homes/rdubash/avrl_live/CASCADE/bin/gpu_run.sh /homes/rdubash/avrl_live/CASCADE/avrl/runs/L13a 'bash /homes/rdubash/avrl_live/CASCADE/avrl/exp/L13a.sh'
cd /homes/rdubash/avrl_live/CASCADE || exit 1
mkdir -p avrl/runs/L13a
exec /homes/rdubash/active-audit/.venv-judge/bin/python avrl/p0_rollouts.py \
  --model /homes/rdubash/active-audit/models/Qwen2.5-0.5B-Instruct \
  --n_prompts 7473 --G 8 --max_tokens 512 --seed 0 --out avrl/runs/L13a
