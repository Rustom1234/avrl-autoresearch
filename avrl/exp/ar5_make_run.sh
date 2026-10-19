#!/bin/bash
# AR5 step 4: build one run tree /homes/rdubash/avrl_live/<run_id>/ (BASE) from the hash-manifested seed.
# Usage: exp/ar5_make_run.sh <run_id> <host>
# Writes only under BASE (plus the pre-warm lock file in /homes/rdubash/avrl_live). No GPU job is started.
set -euo pipefail

RUN_ID=${1:?usage: ar5_make_run.sh <run_id> <host>}
HOST=${2:?usage: ar5_make_run.sh <run_id> <host>}
LIVE=/homes/rdubash/avrl_live
SEED=$LIVE/_seed_day1
BASE=$LIVE/$RUN_ID
GATE='AR1|AR2|AR3|AR5|S1 |Where the claims stand|M0[1-9]|L9[a-d]|bootstrap\.json|KILL_REVERSED|goldberg|hannah|5090|Priority note|settled'

[ -d "$SEED/avrl" ] || { echo "seed missing: run exp/ar5_build_seed.sh first" >&2; exit 1; }
[ ! -e "$BASE" ] || { echo "refusing to overwrite existing $BASE" >&2; exit 1; }

# ---- tree ----
cp -a "$SEED" "$BASE"
sed -i "s|{BASE}|$BASE|g" "$BASE/lit/avrl/EXPERIMENT_BACKLOG.md"
if grep -q '{BASE}' "$BASE/lit/avrl/EXPERIMENT_BACKLOG.md"; then echo "unsubstituted {BASE} left" >&2; exit 1; fi
echo "$HOST" > "$BASE/HOST"
mkdir -p "$BASE/bin" "$BASE/snap" "$BASE/events"

# ---- helpers (identical text in every run, parameterized only by BASE/HOST) ----
cat > "$BASE/bin/gpu_run.sh" <<'EOF'
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
EOF

cat > "$BASE/bin/gpu_kill.sh" <<'EOF'
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
EOF

cat > "$BASE/bin/gpu_status.sh" <<'EOF'
#!/bin/bash
# Usage: bin/gpu_status.sh
# Prints GPU memory used and compute apps on this run's GPU host.
set -uo pipefail
BASE=$(cd "$(dirname "$0")/.." && pwd)
HOST=$(head -1 "$BASE/HOST")
on_host() { if [ "$HOST" = "$(hostname -s)" ]; then bash -c "$1"; else ssh -o BatchMode=yes "$HOST" "$1"; fi; }
on_host "echo host \$(hostname -s); nvidia-smi --query-gpu=memory.used,memory.total --format=csv,noheader; echo compute apps:; nvidia-smi --query-compute-apps=pid,used_memory --format=csv,noheader"
EOF
chmod +x "$BASE/bin/"*.sh

# ---- freeze the code and seed results, manifest ----
cd "$BASE"
FROZEN=$( { printf '%s\n' avrl/p0_rollouts.py avrl/p0_analyze.py avrl/allocators.py runs/sim_generic.py; find avrl/runs/p0 -type f; } | LC_ALL=C sort )
echo "$FROZEN" | xargs chmod a-w
echo "$FROZEN" | xargs sha256sum > RUN_MANIFEST.sha256
echo "== RUN_MANIFEST.sha256: $(wc -l < RUN_MANIFEST.sha256) files"

# ---- leak gate on the tree (BASE paths stripped before matching) ----
hits=$(grep -rInE "$GATE" "$BASE" | sed "s|$BASE||g" | grep -E "$GATE" || true)
if [ -n "$hits" ]; then echo "$hits"; echo "LEAK GATE FAILED for $RUN_ID" >&2; exit 1; fi
echo "== leak gate clean for $RUN_ID"

# ---- sequential pre-warm on HOST (no GPU); flock keeps parallel make_run calls sequential ----
on_host() { if [ "$HOST" = "$(hostname -s)" ]; then bash -c "$1"; else ssh -o BatchMode=yes "$HOST" "$1"; fi; }
exec 9>"$LIVE/.prewarm.lock"
flock 9
t0=$(date +%s.%N)
on_host "cd /homes/rdubash/active-audit && .venv-judge/bin/python -c 'import vllm' && .venv/bin/python -c 'import sentence_transformers, sklearn' && ls models/Qwen2.5-1.5B-Instruct >/dev/null"
t1=$(date +%s.%N)
echo "prewarm host=$HOST seconds=$(awk -v a="$t0" -v b="$t1" 'BEGIN{printf "%.1f", b-a}')" | tee "$BASE/snap/prewarm.txt"
echo "ready: $BASE"
