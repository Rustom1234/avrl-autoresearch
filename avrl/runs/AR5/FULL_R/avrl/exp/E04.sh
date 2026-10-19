#!/bin/bash
# E04 launcher record. GPU step (3 judge rungs, each its own process so GPU memory is freed between them; a failed rung does not stop later ones):
#   bin/gpu_run.sh avrl/runs/E04 "$(cat avrl/exp/E04.sh | sed -n '/^GPU_CMD=/,/^EOF2/p')"   (see the exact command in avrl/runs/E04/cmd.sh)
# CPU step: cd avrl && /homes/rdubash/active-audit/.venv/bin/python exp/e04_analyze.py --feats runs/p0/feats.parquet --judges runs/E04 --out runs/E04
B=/homes/rdubash/avrl_live/FULL_R
for J in $B/models/Qwen2.5-0.5B-Instruct /homes/rdubash/active-audit/models/Qwen2.5-3B-Instruct /homes/rdubash/active-audit/models/Qwen2.5-7B-Instruct-AWQ; do
  /homes/rdubash/active-audit/.venv-judge/bin/python $B/avrl/exp/p0_judge_score.py --rollouts $B/avrl/runs/p0/rollouts.parquet --groups $B/avrl/runs/p0/groups.parquet --judge $J --out $B/avrl/runs/E04
  echo "RUNG_EXIT $J $?"
done
