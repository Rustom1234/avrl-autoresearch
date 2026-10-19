#!/bin/bash
# L19c runbook: GSM8K rollout replicate on the L4b pool (pids 0 to 1318 are the registered test rows, pids 1319 to 2791
# the remaining train rows), new sampling seed 2. Registered in avrl/RESEARCH_LOG.md before launch.
# The rollout and feature scripts are the unchanged L4b and L2b ones. The analysis is the L6a copy l19c_test_replicate.py.
#   L19c.sh launch    checks the GPU has under 2048 MiB used, then starts the one GPU job through bin/gpu_run.sh: fresh
#                     rollouts (vLLM, l4b_rollouts_pool3.py --G 8 --seed 2), then the feature columns (l2b_feats.py).
#                     Poll runs/L19c/done.flag in-band; exit_code must be 0.
#   L19c.sh analyze   CPU only, after the GPU job exits 0. Runs l19c_test_replicate.py into runs/L19c/analysis, copies
#                     result.json and table.md to runs/L19c/, then runs the critic check l19c_check.py into runs/L19c/critic.
#                     Writes runs/L19c/analysis_exit_code and runs/L19c/analysis_done.flag.
set -euo pipefail
R=/homes/rdubash/avrl_live/ROUTED
RUN=$R/avrl/runs/L19c
PY_JUDGE=/homes/rdubash/active-audit/.venv-judge/bin/python
PY=/homes/rdubash/active-audit/.venv/bin/python

case "${1:-}" in
  launch)
    used=$("$R/bin/gpu_status.sh" | grep -oE '^[0-9]+ MiB' | awk '{print $1}')
    if [ -z "$used" ] || [ "$used" -ge 2048 ]; then echo "GPU not free (used: ${used:-unknown} MiB), wait" >&2; exit 1; fi
    "$R/bin/gpu_run.sh" "$RUN" "$PY_JUDGE $R/avrl/exp/l4b_rollouts_pool3.py --G 8 --seed 2 --out $RUN && $PY $R/avrl/exp/l2b_feats.py --run_dir $RUN"
    ;;
  analyze)
    if [ ! -f "$RUN/done.flag" ] || [ "$(cat "$RUN/exit_code" 2>/dev/null)" != "0" ]; then
      echo "GPU job has not finished with exit 0 (see $RUN/run.log)" >&2; exit 1
    fi
    mkdir -p "$RUN/analysis" "$RUN/critic"
    set +e
    "$PY" -I "$R/avrl/exp/l19c_test_replicate.py" --fit_feats "$R/avrl/runs/p0/feats.parquet" --eval_feats "$RUN/feats.parquet" \
      --eval_pid_max 1318 --budget 0.10 --k0 50 --subsets 20 --subset_size 200 --splits 0-7 --tie_seeds 8 \
      --out "$RUN/analysis" > "$RUN/analysis/run.log" 2>&1
    c1=$?
    c2=0
    if [ $c1 -eq 0 ]; then
      cp "$RUN/analysis/result.json" "$RUN/result.json"; cp "$RUN/analysis/table.md" "$RUN/table.md"
      "$PY" -I "$R/avrl/exp/l19c_check.py" --fit_feats "$R/avrl/runs/p0/feats.parquet" --eval_feats "$RUN/feats.parquet" \
        --analysis "$RUN/analysis/result.json" --out "$RUN/critic" > "$RUN/critic/run.log" 2>&1
      c2=$?
    fi
    set -e
    echo "$(( c1 != 0 ? c1 : c2 ))" > "$RUN/analysis_exit_code"
    touch "$RUN/analysis_done.flag"
    cat "$RUN/analysis/run.log" 2>/dev/null
    echo "L19c analysis finished: analysis exit $c1, critic exit $c2"
    ;;
  *)
    echo "usage: $0 launch|analyze" >&2; exit 2
    ;;
esac
