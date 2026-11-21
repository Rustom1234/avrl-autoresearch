#!/usr/bin/env bash
# Cell N of P2B (random search, no language model): 12 loops on the whole hannah n7 node, after the P2B blocks.
# Run from spruce:
#   cd /homes/rdubash/active-audit && nohup setsid loop/runs/P2B_N/launch_n.sh > loop/runs/P2B_N/launch.out 2>&1 < /dev/null &
# What it does, in order:
#   1. waits until the P2B launcher has finished its confirmation block and stopped its workers; exits without
#      doing anything if that launcher ended any other way;
#   2. writes the cell N plan (12 loops, 20 slots, the P2B anchor);
#   3. submits ONE whole-node worker job (1 day) and waits for four fresh workers. It kills nothing;
#   4. runs the controller (8 loops in flight), then the confirmation block, then tells the workers to stop.
# No model is called and no vLLM server is needed. Stop: kill this script, then
#   pkill -u rdubash -f "random_searc[h]"; ssh rustom@hannah.haverford.edu scancel -n arloopP2BN
set -u
cd /homes/rdubash/active-audit
PY=.venv/bin/python
RS="$PY -m loop.core.random_search"
MAIN=loop/runs/P2B
ROOT=loop/runs/P2B_N
H="ssh -o BatchMode=yes -o ConnectTimeout=20 rustom@hannah.haverford.edu"
log() { echo "$(date '+%F %T') $*"; }

# 1. wait for the main run (up to 4 days)
for i in $(seq 1 5760); do
  grep -q "workers told to stop" $MAIN/launch.out 2>/dev/null && break
  if ! pgrep -u rdubash -f "launch_p2[b].sh" > /dev/null; then
    sleep 5
    grep -q "workers told to stop" $MAIN/launch.out 2>/dev/null && break
    log "ABORT: the P2B launcher is gone and did not finish normally; cell N not started"; exit 1
  fi
  sleep 60
done
grep -q "workers told to stop" $MAIN/launch.out || { log "ABORT: P2B did not finish in time"; exit 1; }
[ -f $MAIN/confirmation/summary.json ] || { log "ABORT: P2B has no confirmation summary"; exit 1; }
if pgrep -u rdubash -f "restart_wrapper.s[h]|controller.py ru[n]|loop.core.run_loo[p]|random_searc[h]" > /dev/null; then
  log "ABORT: loop or controller processes still running"; exit 1
fi
log "P2B finished; starting cell N"

# 2. plan
[ -f $ROOT/plan.json ] || $RS plan --root $ROOT --anchor $MAIN/anchor.json --seed 3 || { log "ABORT: plan failed"; exit 1; }

# 3. worker job
sleep 120   # let the old workers exit and their heartbeats go stale
JOB=$($H "squeue -u rustom -h -n arloopP2BN -o %i" | head -1)
if [ -z "$JOB" ]; then
  JOB=$($H 'cd /home/rustom/arloop && sbatch --parsable --time=1-00:00:00 -c 32 --job-name=arloopP2BN start_workers.sbatch')
  log "submitted worker job $JOB"
fi
[ -n "$JOB" ] || { log "ABORT: no worker job"; exit 1; }
echo "$JOB" > $ROOT/worker_job.txt
n=0
for i in $(seq 1 4320); do
  st=$($H "squeue -j $JOB -h -o %T" 2>/dev/null)
  if [ "$st" = "RUNNING" ]; then
    n=$($PY -c "import sys; sys.path.insert(0,'loop/substrate'); from queue_client import QueueClient; s=QueueClient().status(); print(sum(1 for h in (s.get('workers') or {}).values() if h.get('age_s', 1e9) < 60))" 2>/dev/null)
    [ "${n:-0}" -ge 4 ] && break
  elif [ -z "$st" ]; then
    log "ABORT: worker job $JOB is gone"; exit 1
  fi
  [ $((i % 30)) -eq 1 ] && log "waiting for worker job $JOB (state ${st:-?}, fresh workers ${n:-0})"
  sleep 60
done
log "worker job $JOB running with ${n:-0} fresh workers"

# 4. loops, confirmation, stop
for try in $(seq 1 40); do
  $RS controller run --plan $ROOT/plan.json --max-inflight 8 --window-hours 20 --resume >> $ROOT/controller.out 2>&1
  rc=$?
  [ "$rc" -eq 0 ] || [ "$rc" -eq 2 ] && break
  log "controller exited rc=$rc, resuming in 30 s"; sleep 30
done
log "controller exited rc=$rc"
$RS controller confirm --plan $ROOT/plan.json; log "confirmation exited rc=$?"
$PY -c "import sys; sys.path.insert(0,'loop/substrate'); from queue_client import QueueClient; QueueClient().stop_workers()"
log "workers told to stop; cell N loops and confirmation are in $ROOT"
