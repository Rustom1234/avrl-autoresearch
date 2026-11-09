#!/usr/bin/env bash
# L3b runner. CPU only, local shell: no GPU, no gpu_run.sh, no ssh, no pip.
# Log goes to runs/L3b/run.log; done.flag is written when the command finishes (exit code in exit_code).
OUT=/homes/rdubash/avrl_live/CHEAP/avrl/runs/L3b
mkdir -p "$OUT"
rm -f "$OUT/done.flag" "$OUT/exit_code"
timeout 2700 /homes/rdubash/active-audit/.venv/bin/python -B \
  /homes/rdubash/avrl_live/CHEAP/avrl/exp/l3b_scaled_regression.py \
  --pool /homes/rdubash/avrl_live/CHEAP/avrl/runs/p0/feats.parquet \
  --budgets 0.25 --seeds 8 --boot 100 --boot_budget 0.25 --boot_seeds 4 \
  --out "$OUT" > "$OUT/run.log" 2>&1
rc=$?
echo "$rc" > "$OUT/exit_code"
touch "$OUT/done.flag"
exit $rc
