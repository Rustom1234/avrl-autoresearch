# Runbook for run P2 (planner x coder factorial plus harm control, about 40 to 72 hours)

**One-command launch (whole node, chosen 2026-11-06):** `nohup loop/runs/P2/launch_p2.sh > loop/runs/P2/launch.out 2>&1 &` from the repository root on spruce. It restarts the open-model servers, submits one 32-core, 3-day worker job (it waits for the jobs already on n7 to finish and kills nothing), runs the launch sentinels and anchor, starts the controller (8 loops in flight), stops the loops if the first hour shows a failure, and at the end runs confirmation and analysis and stops the workers and servers. The final design has 60 loops (12 H), so the 54-loop and 6-H figures below are superseded by loop/design/P2_PREREG.md.


All commands from the repository root on spruce unless a command says ssh. The experiment is defined in `loop/design/P2_PREREG.md`; this file is only the command sequence. Variables used below:

    PY=.venv/bin/python; C=loop/core/controller.py; ROOT=loop/runs/P2
    H="ssh -o BatchMode=yes rustom@hannah"

Shape of the run: 54 loops of 20 slots (SS, ST, QS, QT x 12, H x 6), 1,080 loop jobs, 24 sentinels during the run (one per 45 completed slots), 16 launch sentinels, and a confirmation block of 144 jobs (2 seeds per incumbent plus one anchor per wave of 3). Cells differ only in the routing table under `$ROOT/routing/`; H runs the SS table with `harm_rate` 0.5.

## Preconditions (check, do not skip)
1. **Spec frozen.** `loop/design/P2_PREREG.md` has its [P1] sections filled and its hash is written to `avrl/RESEARCH_LOG.md`. Do not launch before it.
2. **Tests green** (three invocations, because the three `conftest.py` files collide when pytest collects them together):

       PYTHONDONTWRITEBYTECODE=1 $PY -m pytest -q -p no:cacheprovider loop/core/tests
       PYTHONDONTWRITEBYTECODE=1 $PY -m pytest -q -p no:cacheprovider loop/models/tests
       PYTHONDONTWRITEBYTECODE=1 $PY -m pytest -q -p no:cacheprovider loop/substrate/tests

3. **Both open servers healthy.** Only walnut (qwen35-4b) and pine (gemma4-12b) are used. cherry and willow are not needed.

       curl -s http://walnut:8000/v1/models | grep -o '"id":"[^"]*"'     # qwen35-4b
       curl -s http://pine:8000/v1/models   | grep -o '"id":"[^"]*"'     # gemma4-12b
       $PY -c "import sys; sys.path.insert(0,'loop/models'); import call; print(call.healthy('qwen35-4b'), call.healthy('gemma4-12b'))"   # True True

   Restart recipes (vLLM 0.30 in `.venv-rl`, logs in `~/vllm_logs/`; the lab pool shares /homes, so the same paths work on both hosts):

       A=$PWD; ENV="cd $A && export FLASHINFER_DISABLE_VERSION_CHECK=1 PATH=$A/.venv-rl/bin:\$PATH HF_HUB_OFFLINE=1"
       ssh -o BatchMode=yes walnut "$ENV && setsid nohup $A/.venv-rl/bin/python -m vllm.entrypoints.openai.api_server --host 0.0.0.0 --model cyankiwi/Qwen3.5-4B-AWQ-4bit --max-model-len 32768 --served-model-name qwen35-4b --reasoning-parser qwen3 --gpu-memory-utilization 0.9 > ~/vllm_logs/walnut_qwen35_4b_p2.log 2>&1 < /dev/null & echo walnut pid \$!"
       ssh -o BatchMode=yes pine "$ENV && setsid nohup $A/.venv-rl/bin/python -m vllm.entrypoints.openai.api_server --host 0.0.0.0 --model google/gemma-4-12B-it-qat-w4a16-ct --max-model-len 16384 --served-model-name gemma4-12b --max-num-seqs 4 > ~/vllm_logs/pine_gemma4_p2.log 2>&1 < /dev/null & echo pine pid \$!"

   Stop one: `ssh -o BatchMode=yes <host> 'pkill -u rdubash -f "api_serve[r]"'`. Loading takes 2 to 6 minutes; wait for `/v1/models` before going on. If either server is down while the loops run, its calls fall back to Haiku for that slot (logged in `state.json` `fallbacks`), which contaminates the cell: keep both up.
4. **Deployed worker code matches the repo.** The hashes must be identical on both sides (the sbatch changed, so copy it first, see Workers):

       sha256sum loop/substrate/worker.py loop/substrate/harness_tail.py loop/substrate/start_workers.sbatch
       $H 'cd /home/rustom/arloop && sha256sum code/worker.py code/harness_tail.py start_workers.sbatch'
       $H 'sha256sum /home/rustom/autoresearch_base/prepare.py'     # 4f2ba9cbb8ba8c4a3d35be405a913e2f3be3af9aea103ed52ef7b2a662058150

   Known good as of Nov 5: worker.py 4e918ce64808dd16f9c38cff4bdc9eee7f01ca13b0d383d9cb7871e55b1ca77e, harness_tail.py 7437d4925979952bac48c32d13f66c489cc90576f8d458c2cbf841b50d5ddbf3. `/home/rustom/autoresearch_base/pinned_hashes.json` must contain `data_files`.
5. **No stray processes of ours.** Kill anything left over before starting (never run a second instance beside an old one), then confirm:

       pkill -u rdubash -f restart_wrapper.sh; pkill -u rdubash -f 'controller.py run'; pkill -u rdubash -f loop.core.run_loop
       pgrep -u rdubash -af 'restart_wrapper|controller.py|loop.core.run_loop'        # prints nothing
       $H 'squeue -u rustom; ls /home/rustom/arloop/queue/pending /home/rustom/arloop/queue/running | head'   # no arloop job, queues empty
       test ! -e $ROOT/plan.json && echo "ROOT is fresh"

6. **GPUs 2 and 3 stay untouched while another user's jobs are on them.** Slurm gives no device isolation on n7 (no gres, no cgroups), so protection is our own discipline: workers are started only on GPUs 0 and 1, and the sbatch preflight refuses any listed GPU that already holds more than 2000 MiB (it writes `/home/rustom/arloop/logs/preflight.log` and skips that GPU; the supervisor never restarts a skipped GPU). Before adding GPUs 2 and 3 later, look at `$H 'squeue -w n7 -o "%i %u %j %C %M"'` and read the preflight log after the job starts. If a skipped GPU is still skipped, leave it.
7. **CPU cores.** The worker job asks for 16 cores (`-c 16`, 8 per worker). As of Nov 5, another user's jobs held all 32 cores on n7, so the job will wait in the queue until two of them end (`squeue` shows reason Resources). Plan for that wait.
8. **Budget.** Claude list price for the run is about 45 USD; the controller stops the run at 60M Claude tokens.
9. **Optional, 3 minutes: do `setsid` processes outlive a Slurm job here?** The Slurm configuration (`proctrack/pgid`) suggests yes, but this was never tested. The sbatch now terminates its own workers on SIGTERM either way, so this only tells you whether that trap is needed. Recipe, not run yet; if the quoting fights you, put the three job lines in a file and `sbatch` the file:

       $H 'cd /home/rustom/arloop/logs && sbatch --time=00:02:00 --job-name=orphantest --output=orphantest.%j.log --wrap="setsid nohup sleep 900 > /dev/null 2>&1 & echo \$! > /home/rustom/arloop/logs/orphantest.pid; sleep 600"'
       # after the job has hit its 2 minute limit (about 3 minutes):
       $H 'ps -p $(cat /home/rustom/arloop/logs/orphantest.pid) -o pid,etime,cmd'     # a row: the sleep outlived the job
       $H 'kill $(cat /home/rustom/arloop/logs/orphantest.pid) 2>/dev/null; rm -f /home/rustom/arloop/logs/orphantest.pid'

## Plan
Writes `$ROOT/plan.json`, the four routing tables and 54 loop configs. Refuses to overwrite. Seed 2 (any value other than 1, so the loop seeds, the in-run sentinel seeds and the confirmation shuffle differ from R1). The 16 launch sentinels do not depend on this seed: `launch-sentinels` derives them from a fixed tag, so they reuse R1's 16 launch seeds.

    $PY -m loop.core.plan_p2 --root $ROOT --seed 2
    cat $ROOT/routing/SS.json $ROOT/routing/QT.json

If the [P1] requalification says Haiku fails structural edits or repairs, add `--structural-model sonnet --repairer-model sonnet` (T cells then differ from S only on the constant and schedule classes, or not at all for the repairer). The tables are read once per process start and again on every `--resume`: never edit them after launch.

Check the layout printed by the command: 12 blocks, each with SS, ST, QS, QT once, and an H in 6 of them.

**Decide the harm control size before freezing the PREREG.** With the default 6 H loops the KILL gate has limited power. Rough Monte Carlo (a one-off script, not in the repo; 3,000 trials per cell): draw SS loops from the 11 R1 C and D loops (7 reached 2^17, 64 percent), model a harmed loop as 1.0, 1.5 or 2.0 times slower with half of the reaching loops losing the batch idea (a tried idea is never proposed again), and run the one-sided log-rank test at 0.05 against 12 SS loops:

| SS reach rate | H slowdown | power, 6 H loops | power, 12 H loops |
|---|---|---|---|
| 0.64 (R1) | 1.0 | 0.32 | 0.47 |
| 0.64 | 1.5 | 0.46 | 0.66 |
| 0.64 | 2.0 | 0.48 | 0.68 |
| 0.85 | 1.0 | 0.52 | 0.69 |
| 0.85 | 1.5 | 0.68 | 0.86 |
| 0.85 | 2.0 | 0.73 | 0.89 |

With no harm the test fires in 4 percent of runs (the nominal 5). A second check with the real hash schedule (uneven hits per loop) gave 0.19 to 0.74 for 6 H loops. So a KILL with 6 H loops is more likely a missed detection than a blind substrate, and the run would then end as a cost breakdown after about 145 A100-hours. `--harm-blocks 12` puts one H loop in every block: 60 loops, 1,200 loop jobs, about 139 more jobs in total (120 loop jobs, 16 confirmation jobs, 3 sentinels) and about 16 more A100-hours (about 8 hours more on 2 GPUs). It is not the default because P2_PREREG fixes 6 and its budget lines would have to change with it; that is decided before freezing. A balanced (every other attempt) harm schedule would not help much: what decides a loop is whether the one attempt that proposes the batch change is a hit, and that stays a 50 percent coin flip per loop either way.

## Workers (job A, GPUs 0 and 1)
Copy the patched sbatch, then submit. Command-line options override the `#SBATCH` lines: ask for 3 days (the cluster maximum; the default 2 days 20 hours is too short for two GPUs) and 16 cores.

    scp -o BatchMode=yes loop/substrate/start_workers.sbatch rustom@hannah:/home/rustom/arloop/
    $H 'cd /home/rustom/arloop && GPU_LIST="0 1" sbatch --time=3-00:00:00 -c 16 --job-name=arloop01 start_workers.sbatch'
    $H 'squeue -u rustom; tail -n 5 /home/rustom/arloop/logs/preflight.log /home/rustom/arloop/logs/worker_gpu0.log /home/rustom/arloop/logs/worker_gpu1.log'

Wait for the job to run. Then check the queue sees two fresh workers (gpu2 and gpu3 heartbeats left over from R1 are stale and expected):

    $PY -c "import sys; sys.path.insert(0,'loop/substrate'); from queue_client import QueueClient; print(QueueClient().status())"

`preflight.log` should be empty or only name GPUs you expected to skip. If GPU 0 or 1 was skipped, stop and sort that out before going on (a one-GPU run takes twice as long).

## Launch sentinels and anchor
16 baseline runs of the unmodified script, four at a time (about 4 waves of 15 minutes on two GPUs). Rerunning tops up to 16 good runs.

    $PY $C launch-sentinels --root $ROOT --n 16
    $PY -m loop.core.anchor_p2 --root $ROOT --min-n 14
    cat $ROOT/anchor.json | head -12; grep sigma_used $ROOT/plan.json

Expect an anchor mean near 1.115 and an sd near 0.004 (R1: 1.114885 and 0.004125). `gpu_offsets` are zero for every GPU seen by design. `sigma_used` is 1.15 x sd and is written into `plan.json` and `anchor.json`; loops read the anchor at start, so it must exist before the launch. It is frozen: the command refuses to run again.

## Launch
`--window-hours 120` is only an upper bound (the controller stops at hour 115.5 and exits earlier when every loop is finished); the 24 hour default would stop the run at hour 19.5.

    nohup loop/core/restart_wrapper.sh $ROOT/plan.json --max-inflight 6 --window-hours 120 > $ROOT/wrapper.out 2>&1 &
    $PY $C status --plan $ROOT/plan.json

Six loops in flight on two workers means each slot waits behind a few jobs; that is expected (R1 ran six on four).

## Monitor (every hour or so)
- `$PY $C status --plan $ROOT/plan.json` and `tail -n 30 $ROOT/controller.log`. Every row says arm D; the cell is in the loop id.
- Cell health, one line per loop (slot, status, fallback events, open-served over designated open calls, harm no-ops):

      $PY - <<'EOF'
      import glob, json
      for p in sorted(glob.glob("loop/runs/P2/*/state.json")):
          s = json.load(open(p))
          print(s["loop_id"], s["slot_index"], s["status"], "fallbacks", len(s["fallbacks"]), "open", s["open_served_calls"], "/", s["designated_open_calls"], "harm", len(s.get("harm_noops", [])))
      EOF

  A cell with fallbacks or an open-served fraction well below 1 means a server was down. H loops should show about half of their new-idea attempts as harm no-ops.
- Servers: the two `/v1/models` probes from the preconditions.
- Queue and workers: the `QueueClient().status()` command above; `logs/supervisor.log` on hannah for restarts; `$H squeue -u rustom` for the time left of the job.
- Spend: `$PY -m loop.models.cost $ROOT/SS_01/calls.jsonl` per loop, `tail -n 3 $ROOT/spend.jsonl`. The 60M token ceiling is enforced by the controller (symmetric stop).
- Sentinel drift: `$PY $C status` prints the live sentinel sd against the old noise-job value (0.005857) because P2's `anchor.json` has no `sigma_residual_noise`. Ignore its "OUTSIDE 25 percent: replay B, C, D offline" remark (that is about R1) and compare the live sd with `sd` in `anchor.json` by eye (expect about 0.004).
- **Job time limit.** Job A ends after 3 days. On two GPUs the run needs about 72 hours, so it may not finish. At the limit (and on `scancel`) Slurm sends SIGTERM to the sbatch script; the script's trap then terminates the worker groups it started (15 s grace, `stop_workers: signalled workers ...` in `logs/supervisor.log`) and exits 143, so the `setsid` workers do not outlive the job. The trainer job running on each GPU at that moment is lost (under 15 minutes of GPU work per GPU) and goes back to pending as described below. To lose nothing, touch STOP at about hour 71.5, wait for the workers to exit, and resubmit (a `FIRST_JOB=1` start clears STOP); STOP is global, so do this only when no GPUs 2 and 3 job is live. A harder kill (SIGKILL, node loss) skips the trap, which is why the heartbeat check below stays. Do not submit a second job on top of a live one: a start with `FIRST_JOB=1` requeues the live job's running work. When job A ends, check that its workers are really gone (heartbeats in the `QueueClient().status()` output go stale) and resubmit the same command at once, except when the GPUs 2 and 3 job (below) is still running: then add `FIRST_JOB=0`, because `FIRST_JOB=1` would requeue that job's running work too; with `FIRST_JOB=0` the new supervisor requeues the dead job's stuck jobs within a few minutes once their heartbeats are stale. If the ended job's heartbeats stay fresh for more than 3 minutes, its `setsid` workers outlived the job (`proctrack/pgid` does not track them): do not start workers on those GPUs again while they run, because two workers on one GPU slow both trainers inside the fixed 300 s budget and corrupt their val_bpb. A submitted job just waits in `queue/pending` while no worker runs (the engine waits up to 2 hours per poll, 20 polls), so a gap of an hour or two costs time only; jobs that were running when the old job died are requeued by the new start.
- **Adding GPUs 2 and 3.** Only if the other user's jobs on n7 have ended and the squeue check from precondition 6 is clean. The second job must not touch the first job's STOP file or running jobs:

      $H 'cd /home/rustom/arloop && FIRST_JOB=0 GPU_LIST="2 3" sbatch --time=3-00:00:00 -c 16 --job-name=arloop23 start_workers.sbatch'

  Then read `logs/preflight.log` again. If GPU 2 or 3 is reported in use, it is skipped and no worker starts on it. The STOP file is global: touching it stops both jobs. Per-GPU analysis (below) shows how the cells were spread over GPUs; adding GPUs mid-run is a time confound that the blocked layout only partly absorbs.

## After the loops (the wrapper log says `wrapper done`)
Confirmation runs every final incumbent twice (seeds 1001 and 1002) plus a same-wave anchor, all sealed, from `confirm_seeds` in `plan.json`: 108 + 36 = 144 jobs, about 9 hours on two GPUs. It is resumable; if it dies, run it again.

    $PY $C confirm --plan $ROOT/plan.json
    $PY -m loop.core.analysis_p2 --run $ROOT
    less $ROOT/analysis/report.md

`analysis_p2` writes `report.md`, `loops.csv` and `results.json` to `$ROOT/analysis/`. The report opens with the PASS, KILL or PIVOT block from P2_PREREG (KILL has precedence, then PIVOT, then PASS; NO VERDICT when none applies). The slot margins of the decision use the Kaplan-Meier median of slots to 2^17 (a median that is not reached counts as +inf, so it can trigger PIVOT but never PASS); the mean is printed beside it. The report also lists the per-GPU breakdown, served fractions (including a planner column from the planner_backlog, planner_widen and planner_refresh calls, with the loops whose planner calls were not all served by the designated model), the load-flagged jobs per cell and GPU with a sensitivity verdict on the loops that have none, and the GPU-inclusive dollar totals at 0.5, 1, 2 and 4 USD per A100-hour. `loop.core.analysis` (the R1 analysis) does not apply to a factorial and is not run.

Rehearsal of the analysis code on R1 (meaningless numbers, only checks that the pipeline runs): `$PY -m loop.core.analysis_p2 --run loop/runs/R1 --out /tmp/p2_r1_smoke --cell-map C=SS,D=ST`.

## Stop and clean up
1. Stop the workers gracefully and wait for them to exit (a bare `scancel` also ends them, through the sbatch trap, but it kills the running trainer jobs):

       $PY -c "import sys; sys.path.insert(0,'loop/substrate'); from queue_client import QueueClient; QueueClient().stop_workers()"
       $H 'squeue -u rustom; tail -n 3 /home/rustom/arloop/logs/worker_gpu0.log'      # wait for "STOP file found, exiting"

2. When the job has ended on its own, or `scancel` if it has not within a few minutes: `$H 'scancel -u rustom'`.
3. Stop both vLLM servers (commands in precondition 3) and confirm the GPUs are free (`nvidia-smi` on walnut and pine).
4. `pgrep -u rdubash -af 'restart_wrapper|controller.py|loop.core.run_loop'` prints nothing.
5. Write up.

Emergency stop of the whole experiment (loops included): `pkill -u rdubash -f restart_wrapper.sh; pkill -u rdubash -f 'controller.py run'; pkill -u rdubash -f loop.core.run_loop`. A later `restart_wrapper.sh ... --resume` continues from the state files.

## Known limits to state in the write-up
- The workers share n7 with another user's jobs; Slurm does not isolate GPUs there. The preflight only looks at job start, and R1's per-GPU offsets did not hold, so GPU is a blocking factor (see the per-GPU section), not a corrected quantity.
- With two GPUs the run needs a resubmission of the worker job; the gap only delays the loops, but it is lost GPU time and a time confound across blocks.
- The harm control replaces the coder's real edit by the incumbent plus one comment line (`# candidate edit withheld`) with probability 0.5 per new-idea attempt, decided by a hash of (loop seed, slot, attempt). The real edit is kept in `slots/<k>/harm_original_train.py`. Crash fixes and repairs are never replaced.
- Stall, time to 2^17 and "backed" accepted changes are the definitions at the top of `loop/core/analysis_p2.py`. "Backed" means the accept rested on at least two runs with different seeds; it stands in for "survives a fresh seed" because intermediate incumbents have no confirmation run of their own. Times are slots used (slot index plus one for an event; a loop that stopped after k slots is censored at k). The stall rate counts unfinished loops with the incumbent they had, so a cell with many stopped or failed loops overstates its stall; the report shows the unfinished count per cell.
- **The KILL gate has limited power** (table under Plan). Read a KILL with 6 H loops together with the observed H and SS numbers in the harm control section, not as proof that the substrate cannot see a wasteful coder.
- ST is "R1 arm D as run", not an exact copy: R1's table sent `planner_refresh` to qwen35-4b (`planner_cheap`), ST sends it to Sonnet. This had no effect in R1 because no refresh ever fired.
- A worker that was load-flagged (another process on its GPU before or after the job) can have a slower, noisier trainer inside the fixed 300 s budget. The report counts such jobs per cell and GPU and gives the verdict again without the loops that have one; a different sensitivity verdict means the GPU sharing may have moved the result.
- Sealed gain is expected to be inconclusive at 24 loops per side.
