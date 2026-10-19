#!/bin/bash
# L10b runbook. CPU only, runs directly in the shell (no GPU job, nothing launched through bin/gpu_run.sh, no ssh, no pip).
# Matched-precision control for L9a: eval rows take the 160-label fold table posterior (one random fold per eval row, from
# RandomState(400 + i)), p0 rows keep their out-of-fold posterior, and the frozen comparator stays the 200-label subset table.
# 10% budget, 20 p0 subsets, split seeds 0 to 7, 8 tie seeds. One pool per call (each call stays under the 10 minute shell limit):
# L2b, then L4b, then combine (with the L9a comparison) and the independent check.
R=/homes/rdubash/avrl_live/ROUTED
RUN=$R/avrl/runs/L10b
PY=/homes/rdubash/active-audit/.venv/bin/python

case "$1" in
  L2b|L4b)
    mkdir -p "$RUN/$1"; rm -f "$RUN/$1/exit_code"
    "$PY" -I "$R/avrl/exp/l10b_matched_precision.py" --fit_feats "$R/avrl/runs/p0/feats.parquet" \
      --eval_feats "$R/avrl/runs/$1/feats.parquet" --budget 0.10 --k0 50 --subsets 20 --subset_size 200 \
      --splits 0-7 --tie_seeds 8 --out "$RUN/$1" > "$RUN/$1/run.log" 2>&1
    echo $? > "$RUN/$1/exit_code"; cat "$RUN/$1/exit_code" ;;
  combine)
    rm -f "$RUN/done.flag"; LOG="$RUN/run.log"; : > "$LOG"
    for p in L2b L4b; do echo "== pool $p (exit $(cat "$RUN/$p/exit_code"))" >> "$LOG"; cat "$RUN/$p/run.log" >> "$LOG"; done
    echo "== combine (avrl/exp/l10b_combine.py)" >> "$LOG"
    "$PY" -I "$R/avrl/exp/l10b_combine.py" --pool_dirs "$RUN/L2b" "$RUN/L4b" --out "$RUN" >> "$LOG" 2>&1
    echo $? > "$RUN/exit_code"
    echo "== independent check (avrl/exp/l10b_check.py)" >> "$LOG"
    "$PY" -I "$R/avrl/exp/l10b_check.py" --result "$RUN/result.json" >> "$LOG" 2>&1
    echo $? > "$RUN/check_exit_code"; touch "$RUN/done.flag"
    echo "combine exit $(cat "$RUN/exit_code"), check exit $(cat "$RUN/check_exit_code")" ;;
  *) echo "usage: $0 L2b|L4b|combine" >&2; exit 2 ;;
esac
