#!/bin/bash
#SBATCH --partition=gpu
#SBATCH -N 1
#SBATCH -n 1
#SBATCH -c 32
#SBATCH --time=2-20:00:00
#SBATCH --requeue
#SBATCH --job-name=arloop
#SBATCH --output=/home/rustom/arloop/logs/workers.%j.log
#
# Starts and supervises the GPU workers of the autoresearch job queue (worker.py; protocol in README.md).
# A worker claims queued training jobs for one GPU; this script is the only thing Slurm runs.
#
# Layout: one worker per GPU (0..3), each pinned to its own 8 cores, plus a supervisor loop. The logs dir must
# exist before sbatch (see deploy_hint.md). Stop gracefully with `touch $ROOT/STOP`: the supervisor stops
# restarting, workers finish their current job and exit. Every setting below can be overridden from the
# environment (the offline tests do that); on the cluster just run `sbatch start_workers.sbatch`.
#
# Choosing GPUs:
# GPU_LIST (default: 0..NGPU-1) picks the GPUs, e.g. GPU_LIST="0 1". A second job that adds GPUs to a queue
# already served by a first job sets FIRST_JOB=0 so that it neither deletes STOP nor requeues the first job's
# running jobs, e.g. `FIRST_JOB=0 GPU_LIST="2 3" sbatch -c 16 start_workers.sbatch`. Slurm gives no device
# isolation on this node, so a preflight skips (and logs) any listed GPU that already holds more than
# MAX_USED_MIB of memory, and the supervisor never restarts a skipped GPU. PREFLIGHT=0 turns the check off.
#
# End of the Slurm job:
# When Slurm ends the job (time limit or scancel) it signals only this script; the setsid workers are outside its
# process group and proctrack/pgid does not follow them, so they could outlive the job and, with a later job on the
# same GPU, run two time-budgeted trainers side by side. The TERM trap below ends the workers this job started
# (worker.py turns SIGTERM into a clean exit that kills its trainer); their running jobs go back to pending when the next
# job starts with FIRST_JOB=1 or when their heartbeats go stale.

# ---- settings (each can be overridden from the environment) ----
ROOT=${ROOT:-/home/rustom/arloop}
BASE=${BASE:-/home/rustom/autoresearch_base}
PY=${PY:-$BASE/.venv/bin/python}
NGPU=${NGPU:-4}
CORES_PER=${CORES_PER:-8}        # worker g runs on cores g*8 .. g*8+7
INTERVAL=${INTERVAL:-60}         # supervisor period, seconds
STALE_S=${STALE_S:-180}          # heartbeat age that gets a worker killed and restarted
POLL=${POLL:-5}
WORKER_EXTRA=${WORKER_EXTRA:-}   # extra worker.py arguments (e.g. --cache-src ...)
FIRST_JOB=${FIRST_JOB:-1}        # 0: another job already serves this queue
GPU_LIST=${GPU_LIST:-$(seq -s' ' 0 $(( NGPU - 1 )))}; GPU_LIST=${GPU_LIST//,/ }
PREFLIGHT=${PREFLIGHT:-1}
MAX_USED_MIB=${MAX_USED_MIB:-2000}

# ---- queue directories, and a clean slate if this is the only job serving the queue ----
mkdir -p "$ROOT"/{code,logs,inbox,queue/pending,queue/running,queue/done,results,workers}
if [ "$FIRST_JOB" = 1 ]; then
  rm -f "$ROOT/STOP"
  # A previous (killed or time-limited) Slurm job may have left jobs in queue/running: nothing is alive yet.
  "$PY" "$ROOT/code/worker.py" --requeue-all-running --root "$ROOT"
fi

# Preflight: keep only the listed GPUs that are not already in use. If nvidia-smi cannot be queried, keep them all.
preflight_log() { echo "$(date '+%F %T') preflight: $*" | tee -a "$ROOT/logs/preflight.log"; }
if [ "$PREFLIGHT" = 1 ] && SMI=$(nvidia-smi --query-gpu=index,memory.used --format=csv,noheader 2>/dev/null); then
  KEEP=""
  for g in $GPU_LIST; do
    used=$(echo "$SMI" | awk -F', *' -v g="$g" '$1 == g { gsub(/[^0-9]/, "", $2); print $2 }')
    if [ -z "$used" ]; then
      preflight_log "gpu$g not reported by nvidia-smi, skipped"
    elif [ "$used" -gt "$MAX_USED_MIB" ]; then
      preflight_log "gpu$g has ${used} MiB in use (limit $MAX_USED_MIB), skipped"
    else
      KEEP="$KEEP $g"
    fi
  done
  GPU_LIST=${KEEP# }
elif [ "$PREFLIGHT" = 1 ]; then
  preflight_log "nvidia-smi unavailable, GPUs not checked"
fi
if [ -z "$GPU_LIST" ]; then
  preflight_log "no usable GPU left, no worker started"
  exit 1
fi

# ---- workers ----
start_worker() {
  local g=$1 lo=$(( $1 * CORES_PER )) hi=$(( $1 * CORES_PER + CORES_PER - 1 ))
  # setsid: the worker leads its own process group, so the supervisor can kill the group
  setsid nohup taskset -c "$lo-$hi" "$PY" "$ROOT/code/worker.py" --gpu "$g" --root "$ROOT" \
    --base "$BASE" --python "$PY" --poll "$POLL" $WORKER_EXTRA >> "$ROOT/logs/worker_gpu$g.log" 2>&1 &
  printf -v "WPID_$g" %s "$!"   # the pid is also the process group id of the worker
}

# SIGTERM (then SIGKILL after 15 s, inside Slurm's KillWait of 30 s) to the worker groups this job started. A pid is only
# signalled while its command line still names that GPU's worker, so a reused pid is left alone.
alive() { grep -qs '^State:[[:space:]]*[^Z]' "/proc/$1/status"; }
stop_workers() {
  local g v p list="" n
  for g in $GPU_LIST; do
    v="WPID_$g"; p=${!v:-}
    if [ -n "$p" ] && tr '\0' ' ' < "/proc/$p/cmdline" 2>/dev/null | grep -q -- "--gpu $g "; then
      kill -TERM -- "-$p" 2>/dev/null; list="$list $p"
    fi
  done
  for n in $(seq 1 15); do
    v=""; for p in $list; do alive "$p" && v=1; done
    [ -z "$v" ] && break
    sleep 1
  done
  for p in $list; do kill -KILL -- "-$p" 2>/dev/null; done
  echo "$(date '+%F %T') stop_workers: signalled workers${list:- none}" >> "$ROOT/logs/supervisor.log"
}
trap 'kill "$SLEEP_PID" 2>/dev/null; stop_workers; exit 143' TERM INT HUP
SLEEP_PID=

GPUS=${GPU_LIST// /,}   # the supervisor watches (and restarts) only the GPUs that were started
STARTED=$(date +%s)
for g in $GPU_LIST; do start_worker "$g"; done

# Supervisor: kill and restart workers whose heartbeat is older than STALE_S (their trainer's process
# group dies with them), and move stuck running jobs back to pending (retries+1, max 2, then infra_fail).
while [ ! -e "$ROOT/STOP" ]; do
  sleep "$INTERVAL" & SLEEP_PID=$!   # in the background so that the trap runs at once, not after the sleep
  wait "$SLEEP_PID"
  [ -e "$ROOT/STOP" ] && break
  for g in $("$PY" "$ROOT/code/worker.py" --supervise --root "$ROOT" --gpus "$GPUS" \
               --stale-s "$STALE_S" --started-at "$STARTED" 2>> "$ROOT/logs/supervisor.log"); do
    echo "$(date '+%F %T') restarting worker gpu$g" >> "$ROOT/logs/supervisor.log"
    start_worker "$g"
  done
done
wait
