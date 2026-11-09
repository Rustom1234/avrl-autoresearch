#!/usr/bin/env bash
# L5c launcher. Batch candidate sets (16 unverified groups per step, the phase-1 step shape), CPU only, long-running,
# so it is started through bin/gpu_run.sh, which writes avrl/runs/L5c/run.log, exit_code and done.flag.
# The script writes avrl/runs/L5c/result.json itself.
# Start with:
#   /homes/rdubash/avrl_live/CASCADE/bin/gpu_run.sh /homes/rdubash/avrl_live/CASCADE/avrl/runs/L5c 'bash /homes/rdubash/avrl_live/CASCADE/avrl/exp/L5c.sh'
cd /homes/rdubash/avrl_live/CASCADE || exit 1
mkdir -p avrl/runs/L5c
exec /homes/rdubash/active-audit/.venv/bin/python avrl/exp/l5c_batch_learned_class.py \
  --feats avrl/runs/p0/feats.parquet --batch 16 --budgets 0.10,0.25,0.50 \
  --seeds 8 --boot 200 --boot_budget 0.25 --workers 8 --out avrl/runs/L5c
