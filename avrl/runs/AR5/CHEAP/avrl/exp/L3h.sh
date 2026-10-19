#!/usr/bin/env bash
# L3h runner. CPU only, local shell: no GPU, no gpu_run.sh, no ssh, no pip.
# Log goes to runs/L3h/run.log; done.flag is written when the command finishes (exit code in exit_code).
OUT=/homes/rdubash/avrl_live/CHEAP/avrl/runs/L3h
mkdir -p "$OUT"
rm -f "$OUT/done.flag" "$OUT/exit_code"
timeout 2700 /homes/rdubash/active-audit/.venv/bin/python -B \
  /homes/rdubash/avrl_live/CHEAP/avrl/exp/l3h_tiebreak_threshold.py \
  --pool /homes/rdubash/avrl_live/CHEAP/avrl/runs/p0/feats.parquet \
  --seeds 8 --boot_a 100 --boot_b 200 --boot_seeds 4 \
  --out "$OUT" > "$OUT/run.log" 2>&1
rc=$?
echo "$rc" > "$OUT/exit_code"
touch "$OUT/done.flag"
exit $rc
