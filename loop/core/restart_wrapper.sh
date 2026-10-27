#!/usr/bin/env bash
# Keep `controller.py run --resume` alive until the run completes.
#   nohup loop/core/restart_wrapper.sh /path/to/root/plan.json > /path/to/root/wrapper.out 2>&1 &
# Extra arguments go to `controller.py run` (for example --max-inflight 6 --window-hours 24).
# Exit 0 when the controller exits 0 (run complete), 2 on a usage error (also from the controller), 1 if another
# wrapper holds the lock. Any other controller exit code means a crash: sleep and resume.
# Env: PY (python, default the active-audit venv), SLEEP_S (default 30).
PLAN=${1:?usage: restart_wrapper.sh PLAN_JSON [controller run args]}
shift
ROOT=$(dirname "$PLAN")
HERE=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
PY=${PY:-/homes/rdubash/active-audit/.venv/bin/python}

exec 9>"$ROOT/wrapper.lock"
flock -n 9 || { echo "restart_wrapper already running for $ROOT" >&2; exit 1; }

while true; do
  "$PY" "$HERE/controller.py" run --plan "$PLAN" --resume "$@" 9>&-
  rc=$?
  if [ "$rc" -eq 0 ] || [ "$rc" -eq 2 ]; then
    echo "$(date -u +%FT%TZ) wrapper: controller exited rc=$rc, wrapper done" >> "$ROOT/controller.log"
    exit "$rc"
  fi
  echo "$(date -u +%FT%TZ) wrapper: controller exited rc=$rc, resuming in ${SLEEP_S:-30} s" >> "$ROOT/controller.log"
  sleep "${SLEEP_S:-30}"
done
