#!/usr/bin/env bash
# One-command launch of run P2 on the whole hannah n7 node (option 2, chosen by the author on 2026-11-06).
# Run from spruce:
#   cd /homes/rdubash/active-audit && nohup loop/runs/P2/launch_p2.sh > loop/runs/P2/launch.out 2>&1 &
# What it does, in order (every step logs to loop/runs/P2/launch.out):
#   1. refuses to start if any of our loop or controller processes are already running;
#   2. restarts the two open-model servers (walnut qwen35-4b, pine gemma4-12b) and waits until both answer;
#   3. submits ONE whole-node worker job (-c 32, 3 days). It kills nothing: Slurm starts it when the jobs now on
#      n7 finish on their own; other users' later jobs queue behind it;
#   4. waits until the job runs and four worker heartbeats are fresh;
#   5. runs the 16 launch sentinels and freezes anchor.json (zero offsets, sigma = 1.15 x sd);
#   6. starts the controller (restart wrapper, 8 loops in flight, stop by hour 61.5);
#   7. after 60 minutes, stops the loops (not the workers) if any loop failed or no loop got past slot 0;
#   8. when the loops finish: confirmation block, analysis, then STOP for the workers and the vLLM servers.
# Emergency stop of everything: see loop/design/P2_RUNBOOK.md, "Stop and clean up".
set -u
cd /homes/rdubash/active-audit
PY=.venv/bin/python
C=loop/core/controller.py
ROOT=loop/runs/P2
H="ssh -o BatchMode=yes -o ConnectTimeout=20 rustom@hannah.haverford.edu"
A=/homes/rdubash/active-audit
log() { echo "$(date '+%F %T') $*"; }

# 1. no second instance
if pgrep -u rdubash -f 'restart_wrapper.sh|controller.py run|loop.core.run_loop' > /dev/null; then
  log "ABORT: loop or controller processes already running"; exit 1
fi
[ -f $ROOT/plan.json ] || { log "ABORT: no plan"; exit 1; }
[ -f $ROOT/anchor.json ] && { log "ABORT: anchor.json exists already (refusing to redo sentinels)"; exit 1; }
[ "$(sha256sum loop/design/P2_PREREG.md | cut -d' ' -f1)" = "$(cut -d' ' -f1 loop/design/P2_PREREG.sha256)" ] || \
  { log "ABORT: P2_PREREG.md does not match its frozen hash"; exit 1; }

# 2. open-model servers
ENV="cd $A && export FLASHINFER_DISABLE_VERSION_CHECK=1 PATH=$A/.venv-rl/bin:\$PATH HF_HUB_OFFLINE=1"
ssh -o BatchMode=yes walnut 'pkill -u rdubash -f "api_serve[r]"' ; ssh -o BatchMode=yes pine 'pkill -u rdubash -f "api_serve[r]"'
sleep 5
timeout 20 ssh -o BatchMode=yes walnut "$ENV && setsid nohup $A/.venv-rl/bin/python -m vllm.entrypoints.openai.api_server --host 0.0.0.0 --model cyankiwi/Qwen3.5-4B-AWQ-4bit --max-model-len 32768 --served-model-name qwen35-4b --reasoning-parser qwen3 --gpu-memory-utilization 0.9 > /homes/rdubash/vllm_logs/walnut_qwen35_4b_p2.log 2>&1 < /dev/null &" || true
timeout 20 ssh -o BatchMode=yes pine "$ENV && setsid nohup $A/.venv-rl/bin/python -m vllm.entrypoints.openai.api_server --host 0.0.0.0 --model google/gemma-4-12B-it-qat-w4a16-ct --max-model-len 16384 --served-model-name gemma4-12b --max-num-seqs 4 > /homes/rdubash/vllm_logs/pine_gemma4_p2.log 2>&1 < /dev/null &" || true
for i in $(seq 1 90); do
  w=$(curl -s -m 5 walnut:8000/v1/models | grep -c qwen35-4b); p=$(curl -s -m 5 pine:8000/v1/models | grep -c gemma4-12b)
  [ "$w" = 1 ] && [ "$p" = 1 ] && break
  sleep 20
done
[ "$w" = 1 ] && [ "$p" = 1 ] || { log "ABORT: open-model servers did not come up (walnut=$w pine=$p)"; exit 1; }
log "open-model servers up"

# 3. worker job (skipped if one of ours is already queued or running)
JOB=$($H "squeue -u rustom -h -n arloopP2 -o %i" | head -1)
if [ -z "$JOB" ]; then
  JOB=$($H 'cd /home/rustom/arloop && sbatch --parsable --time=3-00:00:00 -c 32 --job-name=arloopP2 start_workers.sbatch')
  log "submitted worker job $JOB"
else
  log "worker job $JOB already present"
fi
echo "$JOB" > $ROOT/worker_job.txt

# 4. wait for the job and 4 fresh heartbeats (up to 3 days of queueing)
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
log "worker job $JOB running with ${n:-0} fresh workers"; $H 'cat /home/rustom/arloop/logs/preflight.log 2>/dev/null | tail -5'

# 5. launch sentinels and anchor
$PY $C launch-sentinels --root $ROOT --n 16 || { log "ABORT: launch sentinels failed"; exit 1; }
$PY -m loop.core.anchor_p2 --root $ROOT --min-n 14 || { log "ABORT: anchor failed"; exit 1; }
log "anchor: $(cat $ROOT/anchor.json | head -c 300)"

# 6. controller
export AR_LOCK_DIR=$A/loop/runs/P2/claude_locks AR_BREAKER_FILE=$A/loop/runs/P2/claude_breaker
mkdir -p $AR_LOCK_DIR
nohup loop/core/restart_wrapper.sh $ROOT/plan.json --max-inflight 8 --window-hours 66 > $ROOT/wrapper.out 2>&1 &
WPID=$!
log "controller wrapper started, pid $WPID"

# 7. early check after 60 minutes
sleep 3600
BAD=$($PY - <<'EOF'
import json, pathlib
root = pathlib.Path("loop/runs/P2")
st = json.loads((root / "controller_state.json").read_text()).get("loops", {})
failed = [k for k, v in st.items() if v.get("status") == "failed"]
progressed = [k for k, v in st.items() if (v.get("slot_index") or 0) >= 1]
print("failed=%s progressed=%d" % (",".join(failed) or "none", len(progressed)) if failed or not progressed else "")
EOF
)
if [ -n "$BAD" ]; then
  log "EARLY CHECK FAILED ($BAD): stopping the loops, workers keep running"
  pkill -u rdubash -f restart_wrapper.sh; pkill -u rdubash -f 'controller.py run'; pkill -u rdubash -f loop.core.run_loop
  exit 1
fi
log "early check passed"

# 8. wait for the loops, then confirmation and analysis, then stop workers and servers
wait $WPID; rc=$?
log "controller wrapper exited rc=$rc"
$PY $C confirm --plan $ROOT/plan.json; log "confirmation exited rc=$?"
$PY -m loop.core.analysis_p2 --run $ROOT; log "analysis exited rc=$?"
$PY -c "import sys; sys.path.insert(0,'loop/substrate'); from queue_client import QueueClient; QueueClient().stop_workers()"
ssh -o BatchMode=yes walnut 'pkill -u rdubash -f "api_serve[r]"'; ssh -o BatchMode=yes pine 'pkill -u rdubash -f "api_serve[r]"'
log "workers told to stop, vLLM servers stopped; results in $ROOT/analysis/report.md"
