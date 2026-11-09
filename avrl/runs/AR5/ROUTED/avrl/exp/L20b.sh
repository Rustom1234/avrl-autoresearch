#!/bin/bash
# L20b runbook: fresh fit-pool and test-pool GSM8K rollouts with sampling seeds 100 and 200, then the frozen 3,000-label
# table scored on the 1,319 registered test rows. Registered in avrl/RESEARCH_LOG.md before launch.
#   L20b.sh launch    checks the GPU has under 2048 MiB used, then starts the one GPU job through bin/gpu_run.sh:
#                     l20b_fit_rollouts.py (the 3,000 p0 prompts, --seed 100), then l20b_test_rollouts.py (the 1,319 test rows,
#                     --seed 200). Poll runs/L20b/done.flag in-band; exit_code must be 0.
#   L20b.sh analyze   CPU only, after the GPU job exits 0: l20b_feats.py (cell columns attached by question text, no KMeans),
#                     then l20b_replicate.py into runs/L20b/analysis, then the critic l20b_check.py into runs/L20b/critic.
#                     result.json and table.md are copied to runs/L20b/ only if the critic agrees with the analysis.
#                     Writes runs/L20b/analysis_exit_code and analysis_done.flag.
set -euo pipefail
R=/homes/rdubash/avrl_live/ROUTED
RUN=$R/avrl/runs/L20b
PY_JUDGE=/homes/rdubash/active-audit/.venv-judge/bin/python
PY=/homes/rdubash/active-audit/.venv/bin/python

case "${1:-}" in
  launch)
    used=$("$R/bin/gpu_status.sh" | grep -oE '^[0-9]+ MiB' | awk '{print $1}')
    if [ -z "$used" ] || [ "$used" -ge 2048 ]; then echo "GPU not free (used: ${used:-unknown} MiB), wait" >&2; exit 1; fi
    "$R/bin/gpu_run.sh" "$RUN" "$PY_JUDGE $R/avrl/exp/l20b_fit_rollouts.py --G 8 --seed 100 --out $RUN/fit && $PY_JUDGE $R/avrl/exp/l20b_test_rollouts.py --G 8 --seed 200 --out $RUN/test"
    ;;
  analyze)
    if [ ! -f "$RUN/done.flag" ] || [ "$(cat "$RUN/exit_code" 2>/dev/null)" != "0" ]; then
      echo "GPU job has not finished with exit 0 (see $RUN/run.log)" >&2; exit 1
    fi
    mkdir -p "$RUN/analysis" "$RUN/critic"
    set +e
    "$PY" -I "$R/avrl/exp/l20b_feats.py" > "$RUN/feats.log" 2>&1
    c0=$?
    c1=1; c2=1
    if [ $c0 -eq 0 ]; then
      "$PY" -I "$R/avrl/exp/l20b_replicate.py" --fit_feats "$RUN/fit/feats.parquet" --eval_feats "$RUN/test/feats.parquet" \
        --eval_pid_max 1318 --budget 0.10 --k0 50 --subsets 20 --subset_size 200 --splits 0-7 --tie_seeds 8 \
        --out "$RUN/analysis" > "$RUN/analysis/run.log" 2>&1
      c1=$?
    fi
    if [ $c0 -eq 0 ] && [ $c1 -eq 0 ]; then
      "$PY" -I "$R/avrl/exp/l20b_check.py" --fit_feats "$RUN/fit/feats.parquet" --eval_feats "$RUN/test/feats.parquet" \
        --analysis "$RUN/analysis/result.json" --fit_rollouts "$RUN/fit/rollouts.parquet" \
        --test_rollouts "$RUN/test/rollouts.parquet" --out "$RUN/critic" > "$RUN/critic/run.log" 2>&1
      c2=$?
    fi
    if [ $c0 -eq 0 ] && [ $c1 -eq 0 ] && [ $c2 -eq 0 ]; then
      cp "$RUN/analysis/result.json" "$RUN/result.json"; cp "$RUN/analysis/table.md" "$RUN/table.md"
    fi
    set -e
    worst=$c2; if [ $c1 -ne 0 ]; then worst=$c1; fi; if [ $c0 -ne 0 ]; then worst=$c0; fi
    echo "$worst" > "$RUN/analysis_exit_code"
    touch "$RUN/analysis_done.flag"
    cat "$RUN/feats.log" 2>/dev/null
    cat "$RUN/analysis/run.log" 2>/dev/null
    cat "$RUN/critic/run.log" 2>/dev/null
    echo "L20b analysis finished: feats exit $c0, analysis exit $c1, critic exit $c2"
    ;;
  *)
    echo "usage: $0 launch|analyze" >&2; exit 2
    ;;
esac
