#!/bin/bash
# L4b runbook: third GSM8K rollout pool (1,319 test rows plus the 1,473 train rows left after the L1b and L2b draws).
# Configs A (k0 2, stratum agreement x n_distinct, 5%) and B (k0 50, stratum agreement x n_distinct x any_truncated, 10%)
# are fixed from L1b. Refit floor 40 is the verdict, floor 10 is reference. No retuning. Steps, in order:
#   L4b.sh repro    CPU only. Reproduction gate on the L1B pool (runs/p0): the copy must reproduce runs/L3a/l1b_pool at
#                   refit floor 40 (A +1.30 pp, B +4.64 pp, splits 0 to 7). Writes runs/L4b/replay/repro_l1b_pool.json.
#   L4b.sh launch   checks the GPU has under 2048 MiB used, then starts the one GPU job through bin/gpu_run.sh: fresh
#                   rollouts (vLLM), then the feature columns. Poll $RUN/done.flag in-band; exit_code must be 0.
#   L4b.sh replay   CPU only, after the GPU job exits 0. Runs avrl/exp/l4b_replay.py (repro gate, then the verdict and
#                   floor 10 reference on the L4b pool) and avrl/exp/l4b_critic_cuts.py (secondary, 64 cuts).
#                   Writes runs/L4b/result.json, table.md, replay_exit_code and replay_done.flag.
set -euo pipefail
R=/homes/rdubash/avrl_live/ROUTED
RUN=$R/avrl/runs/L4b
PY_JUDGE=/homes/rdubash/active-audit/.venv-judge/bin/python
PY=/homes/rdubash/active-audit/.venv/bin/python

case "${1:-}" in
  repro)
    mkdir -p "$RUN/replay"
    "$PY" -I "$R/avrl/exp/l4b_replay.py" --feats "$R/avrl/runs/p0/feats.parquet" --out "$RUN/replay" --result_dir "$RUN" --repro_only
    ;;
  launch)
    used=$("$R/bin/gpu_status.sh" | grep -oE '^[0-9]+ MiB' | awk '{print $1}')
    if [ -z "$used" ] || [ "$used" -ge 2048 ]; then echo "GPU not free (used: ${used:-unknown} MiB), wait" >&2; exit 1; fi
    "$R/bin/gpu_run.sh" "$RUN" "$PY_JUDGE $R/avrl/exp/l4b_rollouts_pool3.py --G 8 --seed 1 --out $RUN && $PY $R/avrl/exp/l2b_feats.py --run_dir $RUN"
    ;;
  replay)
    if [ ! -f "$RUN/done.flag" ] || [ "$(cat "$RUN/exit_code" 2>/dev/null)" != "0" ]; then
      echo "GPU job has not finished with exit 0 (see $RUN/run.log)" >&2; exit 1
    fi
    mkdir -p "$RUN/replay"
    set +e
    "$PY" -I "$R/avrl/exp/l4b_replay.py" --feats "$RUN/feats.parquet" --out "$RUN/replay" --result_dir "$RUN" > "$RUN/replay/replay.log" 2>&1
    c1=$?
    c2=0
    if [ $c1 -eq 0 ]; then
      "$PY" -I "$R/avrl/exp/l4b_critic_cuts.py" --feats "$RUN/feats.parquet" --result_dir "$RUN" > "$RUN/replay/cuts.log" 2>&1
      c2=$?
    fi
    set -e
    echo "$(( c1 != 0 ? c1 : c2 ))" > "$RUN/replay_exit_code"
    touch "$RUN/replay_done.flag"
    cat "$RUN/replay/replay.log" "$RUN/replay/cuts.log" 2>/dev/null
    echo "L4b replay finished: replay exit $c1, cuts exit $c2"
    ;;
  *)
    echo "usage: $0 repro|launch|replay" >&2; exit 2
    ;;
esac
