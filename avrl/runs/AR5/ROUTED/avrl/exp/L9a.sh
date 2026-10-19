#!/bin/bash
# L9a runbook. CPU only, runs directly in the shell (no GPU job, nothing launched through bin/gpu_run.sh, no ssh).
# Learned arm table_refit: greedy online allocator started from the frozen 200-label config B table, refit on the 200 p0
# labels (out-of-fold table posterior) plus verified eval labels. 10% budget, 20 p0 subsets, split seeds 0 to 7, 8 tie seeds.
# One pool per call (each call stays under the 10 minute shell limit): L2b, L4b, then combine.
R=/homes/rdubash/avrl_live/ROUTED
RUN=$R/avrl/runs/L9a
PY=/homes/rdubash/active-audit/.venv/bin/python

case "$1" in
  L2b|L4b)
    mkdir -p "$RUN/$1"; rm -f "$RUN/$1/exit_code"
    "$PY" -I "$R/avrl/exp/l9a_table_init_learned.py" --fit_feats "$R/avrl/runs/p0/feats.parquet" \
      --eval_feats "$R/avrl/runs/$1/feats.parquet" --budget 0.10 --k0 50 --subsets 20 --subset_size 200 \
      --splits 0-7 --tie_seeds 8 --out "$RUN/$1" > "$RUN/$1/run.log" 2>&1
    echo $? > "$RUN/$1/exit_code"; cat "$RUN/$1/exit_code" ;;
  combine)
    rm -f "$RUN/done.flag"; LOG="$RUN/run.log"; : > "$LOG"
    for p in L2b L4b; do echo "== pool $p (exit $(cat "$RUN/$p/exit_code"))" >> "$LOG"; cat "$RUN/$p/run.log" >> "$LOG"; done
    echo "== combine (avrl/exp/l9a_combine.py)" >> "$LOG"
    "$PY" -I "$R/avrl/exp/l9a_combine.py" --pool_dirs "$RUN/L2b" "$RUN/L4b" --out "$RUN" >> "$LOG" 2>&1
    echo $? > "$RUN/exit_code"; touch "$RUN/done.flag"; cat "$RUN/exit_code" ;;
  *) echo "usage: $0 L2b|L4b|combine" >&2; exit 2 ;;
esac
