#!/bin/bash
# Usage: bin/gpu_run.sh <run_dir> <command...>
# Launches the command detached on this run's GPU host (local if HOST is this machine, else ssh), 45 min cap.
# Files in run_dir: cmd.sh, wrap.sh, run.log (stdout+stderr), pgid, exit_code, done.flag (written last).
set -uo pipefail
BASE=$(cd "$(dirname "$0")/.." && pwd)
HOST=$(head -1 "$BASE/HOST")
RUN_ID=$(basename "$BASE")
on_host() { if [ "$HOST" = "$(hostname -s)" ]; then bash -c "$1"; else ssh -o BatchMode=yes "$HOST" "$1"; fi; }
[ $# -ge 2 ] || { echo "usage: $0 <run_dir> <command...>" >&2; exit 2; }
RUN_DIR=$(realpath -m "$1"); shift
mkdir -p "$RUN_DIR"
rm -f "$RUN_DIR/exit_code" "$RUN_DIR/done.flag" "$RUN_DIR/pgid"
printf '%s\n' "$*" > "$RUN_DIR/cmd.sh"
cat > "$RUN_DIR/wrap.sh" <<'WRAP'
#!/bin/bash
# Runs as session leader (setsid). The timeout is inside so exit_code and done.flag are written even on timeout (124).
D=$1
timeout -k 30 2700 bash -c 'ps -o pgid= -p $$ | tr -d " " > "$0/pgid"; exec bash "$0/cmd.sh"' "$D"
echo $? > "$D/exit_code"
touch "$D/done.flag"
WRAP
LAUNCH="cd '$BASE/avrl' && export CPATH=\"\$HOME/.local/include/python3.12\" HF_HUB_OFFLINE=1 VLLM_CACHE_ROOT=/tmp/avrl_$RUN_ID/vllm TRITON_CACHE_DIR=/tmp/avrl_$RUN_ID/triton && mkdir -p /tmp/avrl_$RUN_ID/vllm /tmp/avrl_$RUN_ID/triton; setsid nohup bash '$RUN_DIR/wrap.sh' '$RUN_DIR' > '$RUN_DIR/run.log' 2>&1 < /dev/null &"
on_host "$LAUNCH" || { echo "launch failed on $HOST" >&2; exit 1; }
PGID=""
for _ in $(seq 1 10); do   # read on HOST itself: the NFS client here may not see the new file for a while
  PGID=$(on_host "cat '$RUN_DIR/pgid' 2>/dev/null" || true)
  [ -n "$PGID" ] && break
  sleep 1
done
echo "$RUN_DIR"
echo "launched host=$HOST pgid=${PGID:-unknown}"
