#!/usr/bin/env bash
# L5a launcher. CPU only, but long-running (about 20 min), so it is started through bin/gpu_run.sh, which writes
# avrl/runs/L5a/run.log, exit_code and done.flag. The script writes avrl/runs/L5a/result.json itself.
# Start with:
#   /homes/rdubash/avrl_live/CASCADE/bin/gpu_run.sh /homes/rdubash/avrl_live/CASCADE/avrl/runs/L5a 'bash /homes/rdubash/avrl_live/CASCADE/avrl/exp/L5a.sh'
cd /homes/rdubash/avrl_live/CASCADE || exit 1
mkdir -p avrl/runs/L5a
exec /homes/rdubash/active-audit/.venv/bin/python avrl/exp/l5a_learned_class.py \
  --feats avrl/runs/p0/feats.parquet --cell cell100 --features cheap+emb --budgets 0.10,0.25,0.50 \
  --seeds 8 --boot 200 --boot_budget 0.25 --workers 8 --out avrl/runs/L5a
