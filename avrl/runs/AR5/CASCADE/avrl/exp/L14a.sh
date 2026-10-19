#!/usr/bin/env bash
# L14a GPU step (copy of L13a.sh, all 1,319 GSM8K test prompts): fresh rollouts from Qwen2.5-0.5B-Instruct, the same policy as L13a.
# Same policy and rollout settings as L13a: G 8, max_tokens 512, sample seed 0. The script is avrl/exp/l14a_rollouts_test.py, a copy of avrl/p0_rollouts.py (p0 unchanged).
# Started through bin/gpu_run.sh, which writes avrl/runs/L14a/run.log, exit_code and done.flag. Start with:
#   /homes/rdubash/avrl_live/CASCADE/bin/gpu_run.sh /homes/rdubash/avrl_live/CASCADE/avrl/runs/L14a 'bash /homes/rdubash/avrl_live/CASCADE/avrl/exp/L14a.sh'
cd /homes/rdubash/avrl_live/CASCADE || exit 1
mkdir -p avrl/runs/L14a
exec /homes/rdubash/active-audit/.venv-judge/bin/python avrl/exp/l14a_rollouts_test.py \
  --model /homes/rdubash/active-audit/models/Qwen2.5-0.5B-Instruct \
  --n_prompts 1319 --G 8 --max_tokens 512 --seed 0 --out avrl/runs/L14a
