#!/bin/bash
# L13a runbook. CPU only, runs directly in the shell (no GPU job, nothing launched through bin/gpu_run.sh, no ssh, no pip).
# Offset-ridge refit anchored at the matched 160-label fold table (avrl/exp/l13a_shrunk_refit.py, copied from L10b).
# 10% budget, 20 p0 subsets, split seeds 0 to 7, 8 tie-break seeds. One pool per call (each under the 10 minute shell limit):
# L2b, L4b, then the L4b pid cuts (registered test rows pid <= 1318, remaining train rows pid > 1318), then combine and the
# independent replay.
R=/homes/rdubash/avrl_live/ROUTED
RUN=$R/avrl/runs/L13a
PY=/homes/rdubash/active-audit/.venv/bin/python
COMMON="--fit_feats $R/avrl/runs/p0/feats.parquet --budget 0.10 --k0 50 --subsets 20 --subset_size 200 --splits 0-7 --tie_seeds 8"

case "$1" in
  L2b|L4b)
    mkdir -p "$RUN/$1"; rm -f "$RUN/$1/exit_code"
    "$PY" -I "$R/avrl/exp/l13a_shrunk_refit.py" $COMMON --eval_feats "$R/avrl/runs/$1/feats.parquet" --pool "$1" \
      --out "$RUN/$1" > "$RUN/$1/run.log" 2>&1
    echo $? > "$RUN/$1/exit_code"; cat "$RUN/$1/exit_code" ;;
  L4b_pid_le1318)
    mkdir -p "$RUN/$1"; rm -f "$RUN/$1/exit_code"
    "$PY" -I "$R/avrl/exp/l13a_shrunk_refit.py" $COMMON --eval_feats "$R/avrl/runs/L4b/feats.parquet" --pool L4b \
      --eval_pid_max 1318 --out "$RUN/$1" > "$RUN/$1/run.log" 2>&1
    echo $? > "$RUN/$1/exit_code"; cat "$RUN/$1/exit_code" ;;
  L4b_pid_gt1318)
    mkdir -p "$RUN/$1"; rm -f "$RUN/$1/exit_code"
    "$PY" -I "$R/avrl/exp/l13a_shrunk_refit.py" $COMMON --eval_feats "$R/avrl/runs/L4b/feats.parquet" --pool L4b \
      --eval_pid_min 1319 --out "$RUN/$1" > "$RUN/$1/run.log" 2>&1
    echo $? > "$RUN/$1/exit_code"; cat "$RUN/$1/exit_code" ;;
  combine)
    rm -f "$RUN/done.flag"; LOG="$RUN/run.log"; : > "$LOG"
    for p in L2b L4b L4b_pid_le1318 L4b_pid_gt1318; do echo "== $p (exit $(cat "$RUN/$p/exit_code"))" >> "$LOG"; cat "$RUN/$p/run.log" >> "$LOG"; done
    echo "== combine (avrl/exp/l13a_combine.py)" >> "$LOG"
    "$PY" -I "$R/avrl/exp/l13a_combine.py" --pool_dirs "$RUN/L2b" "$RUN/L4b" --cut_dirs "$RUN/L4b_pid_le1318" "$RUN/L4b_pid_gt1318" \
      --out "$RUN" >> "$LOG" 2>&1
    echo $? > "$RUN/exit_code"
    echo "== independent replay (avrl/exp/l13a_critic_indep.py)" >> "$LOG"
    "$PY" -I "$R/avrl/exp/l13a_critic_indep.py" --out "$RUN/critic" >> "$LOG" 2>&1
    echo $? > "$RUN/check_exit_code"; touch "$RUN/done.flag"
    echo "combine exit $(cat "$RUN/exit_code"), replay exit $(cat "$RUN/check_exit_code")" ;;
  *) echo "usage: $0 L2b|L4b|L4b_pid_le1318|L4b_pid_gt1318|combine" >&2; exit 2 ;;
esac
