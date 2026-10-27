#!/bin/bash
#SBATCH --partition=gpu
#SBATCH -N 1
#SBATCH -n 1
#SBATCH -c 32
#SBATCH --time=2-20:00:00
#SBATCH --requeue
#SBATCH --job-name=arloop
#SBATCH --output=/home/rustom/arloop/logs/workers.%j.log
# One worker per GPU (0..3), each pinned to its own 8 cores, plus a supervisor loop. The logs dir must
# exist before sbatch (see deploy_hint.md). Stop gracefully with `touch $ROOT/STOP`: the supervisor stops
# restarting, workers finish their current job and exit. Every setting below can be overridden from the
# environment (the offline tests do that); on the cluster just run `sbatch start_workers.sbatch`.

ROOT=${ROOT:-/home/rustom/arloop}
BASE=${BASE:-/home/rustom/autoresearch_base}
PY=${PY:-$BASE/.venv/bin/python}
NGPU=${NGPU:-4}
CORES_PER=${CORES_PER:-8}        # worker g runs on cores g*8 .. g*8+7
INTERVAL=${INTERVAL:-60}         # supervisor period, seconds
STALE_S=${STALE_S:-180}          # heartbeat age that gets a worker killed and restarted
POLL=${POLL:-5}
WORKER_EXTRA=${WORKER_EXTRA:-}   # extra worker.py arguments (e.g. --cache-src ...)

mkdir -p "$ROOT"/{code,logs,inbox,queue/pending,queue/running,queue/done,results,workers}
rm -f "$ROOT/STOP"

# A previous (killed or time-limited) Slurm job may have left jobs in queue/running: nothing is alive yet.
"$PY" "$ROOT/code/worker.py" --requeue-all-running --root "$ROOT"

start_worker() {
  local g=$1 lo=$(( $1 * CORES_PER )) hi=$(( $1 * CORES_PER + CORES_PER - 1 ))
  # setsid: the worker leads its own process group, so the supervisor can kill the group
  setsid nohup taskset -c "$lo-$hi" "$PY" "$ROOT/code/worker.py" --gpu "$g" --root "$ROOT" \
    --base "$BASE" --python "$PY" --poll "$POLL" $WORKER_EXTRA >> "$ROOT/logs/worker_gpu$g.log" 2>&1 &
}

GPUS=$(seq -s, 0 $(( NGPU - 1 )))
STARTED=$(date +%s)
for g in $(seq 0 $(( NGPU - 1 ))); do start_worker "$g"; done

# Supervisor: kill and restart workers whose heartbeat is older than STALE_S (their trainer's process
# group dies with them), and move stuck running jobs back to pending (retries+1, max 2, then infra_fail).
while [ ! -e "$ROOT/STOP" ]; do
  sleep "$INTERVAL"
  [ -e "$ROOT/STOP" ] && break
  for g in $("$PY" "$ROOT/code/worker.py" --supervise --root "$ROOT" --gpus "$GPUS" \
               --stale-s "$STALE_S" --started-at "$STARTED" 2>> "$ROOT/logs/supervisor.log"); do
    echo "$(date '+%F %T') restarting worker gpu$g" >> "$ROOT/logs/supervisor.log"
    start_worker "$g"
  done
done
wait
