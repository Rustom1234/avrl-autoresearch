# Runbook for run R1 (production, 24 hours)

All commands from the repository root on spruce. PY=.venv/bin/python, C=loop/core/controller.py, ROOT=loop/runs/R1.

## Preconditions (check, do not skip)
1. Workers alive on n7: `$PY -c "import sys; sys.path.insert(0,'loop/substrate'); from queue_client import QueueClient; print(QueueClient().status())"` shows 4 workers with fresh heartbeats; Slurm job `arloop` RUNNING (`ssh rustom@hannah squeue -u rustom`). Deployed worker code must match loop/substrate/ (sha256 of worker.py and harness_tail.py); pinned_hashes.json on hannah includes data_files.
2. Open servers up: cherry qwen35-9b, walnut qwen35-4b, pine gemma4-12b, willow r1-qwen3-8b (`curl host:8000/v1/models`). Restart recipe: loop/runs/servers/SERVERS.md.
3. Calibration finished: loop/runs/calibration/routing_table.json exists; every top-level string is a live model name. The D configs point at this path.
4. anchor.json frozen from LIVE sentinels only (not the sbatch noise job, see loop/runs/T1/STRESS.md): fields anchor_mean, n, gpu_offsets, sigma_used. sigma_used stays 0.006736 (PREREG). gpu_offsets are zero unless the one-way ANOVA across GPUs on the live sentinels has p < 0.05, in which case use per-GPU mean minus grand mean.
5. PREREG.md Deviations section updated with the anchor change before launch.
6. Integration T2 showed accept, reject and replicate decisions working with correct flags (loop/runs/T2).
7. No T1 or T2 loop processes still running (`pgrep -f loop.core.run_loop`). Spend baseline noted.

## Launch
    nohup loop/core/restart_wrapper.sh $ROOT/plan.json --max-inflight 6 --window-hours 24 > $ROOT/wrapper.out 2>&1 &
    $PY $C status --plan $ROOT/plan.json

## Monitor (every hour or so)
- `$PY $C status --plan $ROOT/plan.json`; `tail -n 30 $ROOT/controller.log`.
- Queue: workers heartbeats and pending counts (precondition 1 command).
- Servers: the four /v1/models probes; if one is down, restart it (SERVERS.md) and note the time in PREREG Deviations; D loops fall back to Haiku meanwhile and record served fractions.
- Spend: `$PY -m loop.models.cost <loop>/calls.jsonl`; ceiling 60M Claude tokens is enforced by the controller.
- Sentinel sd versus 0.005857 (analysis does this at the end; eyeball sentinels.jsonl meanwhile).

## After the loops (controller exits, about hour 19.5 to 20.5)
    $PY $C confirm --plan $ROOT/plan.json
    $PY -m loop.core.analysis --run $ROOT --out $ROOT/analysis
Then: stop workers (`QueueClient().stop_workers()`), `scancel` the arloop job, stop the vLLM servers, and write up (README section, ledger entries).

## Known limits to state in the write-up
- Arm A is a stateless one-shot control, not Karpathy's persistent agent session.
- Harness runs in the candidate's process: defends against accidents and opportunism, not a deliberate adversary.
- The sbatch noise job and the worker environment differ by about 0.03 bpb for reasons not isolated; all production numbers come from the worker environment.
- Confirmation uses the same-wave anchor; the headline is a bundle contrast (A vs D) and is likely to be inconclusive at 8 vs 8.
