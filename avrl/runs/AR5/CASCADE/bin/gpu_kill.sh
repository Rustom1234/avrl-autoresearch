#!/bin/bash
# Usage: bin/gpu_kill.sh <run_dir>
# Kills the run's process group on HOST, waits 10 s, then any leftover python from this run tree, then prints GPU state.
set -uo pipefail
BASE=$(cd "$(dirname "$0")/.." && pwd)
HOST=$(head -1 "$BASE/HOST")
on_host() { if [ "$HOST" = "$(hostname -s)" ]; then bash -c "$1"; else ssh -o BatchMode=yes "$HOST" "$1"; fi; }
[ $# -ge 1 ] || { echo "usage: $0 <run_dir>" >&2; exit 2; }
RUN_DIR=$(realpath -m "$1")
PGID=$(on_host "cat '$RUN_DIR/pgid' 2>/dev/null" | tr -d ' \n' || true)
case "$PGID" in ''|*[!0-9]*|0|1) echo "no valid pgid in $RUN_DIR/pgid, skipping group kill"; PGID="" ;; esac
on_host "if [ -n '$PGID' ]; then kill -- -$PGID 2>/dev/null && echo killed group $PGID || echo group $PGID already gone; fi; sleep 10; pkill -u rdubash -f '$BASE/avrl/.*[.]py'; nvidia-smi --query-gpu=memory.used,memory.total --format=csv,noheader; nvidia-smi --query-compute-apps=pid,used_memory --format=csv,noheader"
exit 0
