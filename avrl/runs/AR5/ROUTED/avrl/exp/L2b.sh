#!/bin/bash
# L2b runbook: fresh GSM8K pool replication of the L1b label-free gains (configs fixed from L1b, no retuning).
# Two steps, in order. The commands are the ones that produced avrl/runs/L2b/.
#   L2b.sh launch  checks that the GPU has under 2 GB used, then starts the one GPU job through bin/gpu_run.sh:
#                  fresh rollouts (vLLM), then the feature columns. Poll $RUN/done.flag in-band; exit_code must be 0.
#   L2b.sh replay  CPU only. Runs avrl/exp/l2b_replay.py, which writes replay/, result.json and table.md.
# Note: the unchanged L1b script (avrl/exp/l1b_tuned_threshold.py) exits 3 at its G01 reproduction gate on this pool,
# because that gate compares against the E01 pool. Its output is kept in replay/unchanged_run.log for the record.
set -euo pipefail
R=/homes/rdubash/avrl_live/ROUTED
RUN=$R/avrl/runs/L2b
PY_JUDGE=/homes/rdubash/active-audit/.venv-judge/bin/python
PY=/homes/rdubash/active-audit/.venv/bin/python

case "${1:-}" in
  launch)
    used=$("$R/bin/gpu_status.sh" | grep -oE '^[0-9]+ MiB' | awk '{print $1}')
    if [ -z "$used" ] || [ "$used" -ge 2048 ]; then echo "GPU not free (used: ${used:-unknown} MiB), wait" >&2; exit 1; fi
    "$R/bin/gpu_run.sh" "$RUN" "$PY_JUDGE $R/avrl/exp/l2b_rollouts_fresh.py --n_prompts 3000 --G 8 --seed 1 --out $RUN && $PY $R/avrl/exp/l2b_feats.py --run_dir $RUN"
    ;;
  replay)
    if [ ! -f "$RUN/done.flag" ] || [ "$(cat "$RUN/exit_code" 2>/dev/null)" != "0" ]; then
      echo "GPU job has not finished with exit 0 (see $RUN/run.log)" >&2; exit 1
    fi
    "$PY" -I "$R/avrl/exp/l2b_replay.py"
    ;;
  *)
    echo "usage: $0 launch|replay" >&2; exit 2
    ;;
esac
