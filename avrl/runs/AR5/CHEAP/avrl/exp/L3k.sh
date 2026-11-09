#!/usr/bin/env bash
# L3k runner. CPU only, local shell: no GPU, no gpu_run.sh, no ssh, no pip.
# Log goes to runs/L3k/run.log; done.flag is written when the command finishes (exit code in exit_code).
OUT=/homes/rdubash/avrl_live/CHEAP/avrl/runs/L3k
mkdir -p "$OUT"
rm -f "$OUT/done.flag" "$OUT/exit_code"
timeout 2700 /homes/rdubash/active-audit/.venv/bin/python -B \
  /homes/rdubash/avrl_live/CHEAP/avrl/exp/l3k_offline_ranking.py \
  --pool /homes/rdubash/avrl_live/CHEAP/avrl/runs/p0/feats.parquet \
  --out "$OUT" > "$OUT/run.log" 2>&1
rc=$?
echo "$rc" > "$OUT/exit_code"
touch "$OUT/done.flag"
exit $rc
