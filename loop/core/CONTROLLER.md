# Run controller and confirmation block

`controller.py` runs the 22 loop runs of the experiment (DESIGN.md sections 2, 11, 12, 14; PREREG.md): it launches
`python -m loop.core.run_loop` subprocesses, watches their `state.json`, restarts crashes, feeds sentinel baselines
through the queue, applies the symmetric stop, and runs the confirmation block (`confirmation.py`). It contains no
loop logic. Python 3.12, stdlib only; the venv is `.venv` at the repository root.

Tests, from the repository root (no network, no hannah, no GPU; the loop engine and `QueueClient` are replaced by fakes):

    .venv/bin/python -m pytest loop/core/tests/test_controller.py loop/core/tests/test_confirmation.py -q

## Real launch, in order

Everything runs from the repository root on spruce (the control host). `ROOT` is the controller directory, for example
`loop/runs/main1`. Workers on hannah must be up before step 2 (see `loop/substrate/README.md`).

    PY=.venv/bin/python; C=loop/core/controller.py; ROOT=loop/runs/main1

1. Plan. Writes `plan.json` and one `config.json` per loop. Refuses to overwrite an existing plan.
   `base.json` is optional: its keys are copied into every loop config (for example `n_slots`,
   `spend_ceiling_tokens`; loop-specific keys win).

       $PY $C plan --root $ROOT --seed 1 --base-config base.json

2. Launch sentinels. Eight runs of `base/train_ampere.py`, four at a time, so they see the same four-wide load as the
   loops. Writes `launch_sentinels.jsonl`. Rerunning tops it up to 8 good runs.

       $PY $C launch-sentinels --root $ROOT --n 8

3. Anchor. Reads `loop/runs/noise/sigma_table.txt` plus `launch_sentinels.jsonl`, prints and writes `anchor.json`
   (anchor mean, n, per-GPU offsets, sigma_used). Refuses to overwrite: it is frozen. Check the printed numbers against
   PREREG (16 noise runs alone give 1.143598, offsets -0.00306 / +0.00474 / +0.00166 / -0.00334, sigma_used 0.006736).

       $PY $C anchor --root $ROOT

4. Run, under the restart wrapper. The wrapper runs `controller.py run --plan ... --resume` until it exits 0, sleeping
   30 s between crashes, and holds `$ROOT/wrapper.lock` so a second wrapper cannot start. The first start and every
   restart are the same command, because `run --resume` starts fresh when no `controller_state.json` exists.

       nohup loop/core/restart_wrapper.sh $ROOT/plan.json --max-inflight 6 --window-hours 24 > $ROOT/wrapper.out 2>&1 &

   Watch it:

       $PY $C status --plan $ROOT/plan.json
       tail -f $ROOT/controller.log

   Stop everything (wrapper first, or it restarts the controller; loops are separate sessions and keep running unless
   killed too; they can be resumed later with the wrapper command above):

       pkill -u "$USER" -f restart_wrapper.sh; pkill -u "$USER" -f 'controller.py run'
       pkill -u "$USER" -f loop.core.run_loop     # only if the loops should die as well

5. Confirm, after `run` has exited (about hour 19.5 to 20.5; the wrapper log says `wrapper done`):

       $PY $C confirm --plan $ROOT/plan.json

   Resumable: if it dies (for example the queue is unreachable), run it again and finished jobs are skipped.

## What `run` does

- Reads `plan.json`; refuses to start without `anchor.json`; takes `$ROOT/controller.lock` (exit 3 if held).
- Launches loops in plan order (block order, shuffled inside a block by the plan seed), at most `--max-inflight` (6) at
  once, each as a new session with stdout in `<run_dir>/run_loop.log`.
- Every `--poll-s` (30 s) it reads each loop's `state.json`. A loop that exited 0 with state `done` or `stopped` is
  finished. Anything else is a crash: restart with `--resume`, at most 3 times, then the loop is marked `failed`.
- One sentinel baseline job (`base/train_ampere.py`, fresh seed, tag `sentinel`, official shard only) is submitted for
  every 12 completed loop slots summed over all loops (total slot_index // 12). Results go to `sentinels.jsonl`;
  failed sentinels are recorded, not retried. A queue error on submit or poll is logged and retried next cycle.
- Symmetric stop at hour `window_hours - 4.5` (19.5 for 24) or when spend is over the ceiling: every running loop gets
  its stop file (`<run_dir>/STOP_AT`) holding its own current slot index, loops not yet started are `skipped`, no new
  loops or sentinels start, and the controller waits for the loops. Loops still alive 1 hour later are SIGTERMed and
  marked `failed`.
- Every 30 minutes it sums Claude tokens (all types) over every loop's `calls.jsonl`, appends a row to `spend.jsonl`,
  and applies the same stop if the sum is over `spend_ceiling_tokens` (60 million unless the base config says
  otherwise).
- `controller_state.json` is rewritten atomically every cycle. After a controller crash or `kill -9`, `run --resume`
  keeps the original window start, does not relaunch finished loops, and adopts loops that are still alive by finding
  the process whose argv has `--config <that loop's config.json>`. A loop recorded as running with no live process is
  restarted with `--resume`. `run` without `--resume` refuses to start over an existing state file.
- Exits 0 when every loop is `done`, `stopped`, `failed` or `skipped` and the outstanding sentinels are recorded (or 30
  minutes have passed). Exit 2 is a usage error (the wrapper does not retry it). Any other code is a crash.

## The confirmation block

`confirm` takes every loop whose `state.json` says `done` or `stopped` and has an `incumbent.path`, and builds waves
once (frozen in `confirmation/waves.json`): the 4 runs (seeds 1001 to 1004) of every incumbent are shuffled across arms
with `random.Random(plan seed)`, cut into groups of 3, and each group gets one anchor run of `base/train_ampere.py`
(anchor seed cycles 1001 to 1004 by wave). A wave is submitted whole with `sealed=True` and awaited before the next, so
at most 4 jobs are pending or running and a wave's anchor sees the same load as its runs. A job that fails (status not
`ok`, missing values, or no result within 40 minutes) is submitted once more; a second failure is recorded as is.

`confirmation/summary.json`, per loop: `mean_official`, `mean_sealed` over its successful runs; `improvement` = mean over
its runs of (same-wave anchor sealed minus incumbent sealed), and `improvement_official` the same on the official shard;
`n_ok`, `n_paired`. Runs whose own job failed, or whose wave anchor failed, drop out of the means or the improvement
respectively. Overall: `anchor.offset_official_minus_sealed`, the mean of the anchors' official minus sealed.
No GPU-offset adjustment is applied; `results.jsonl` keeps `gpu` per row for analysis.

## File layout

    $ROOT/
      plan.json                  blocks, loops (id, arm, block, index, run_dir, config, stop_file), seed, n_slots, ceiling
      anchor.json                frozen anchor numbers (step 3)
      launch_sentinels.jsonl     the 8 launch baseline runs (step 2)
      controller_state.json      window start, per-loop status/restarts/pid/slot_index, sentinel counters, stop info
      controller.log             timestamped log of every launch, restart, stop, sentinel; the wrapper appends here too
      controller.lock, wrapper.lock, wrapper.out
      sentinels.jsonl            live sentinel baselines: k, job_id, seed, status, official_val_bpb, gpu, slots_at_submit
      spend.jsonl                {t, tokens, ceiling, over} per spend check
      <loop_id>/                 A_01 ... D_08; run_dir of one loop run
        config.json              written by `plan`
        state.json, ledger.jsonl, calls.jsonl, incumbent.py, slots/   written by run_loop
        STOP_AT                  written by the controller at the symmetric stop
        run_loop.log             stdout and stderr of the loop process
      confirmation/
        waves.json, results.jsonl, summary.json

Loop config keys set by `plan`: `loop_id, arm, block, rng_seed, run_dir, stop_file, anchor_file, routing, n_slots,
spend_ceiling_tokens`, plus any base-config keys. `rng_seed` is `seed * 1000 + position in the plan`, unique per loop.
`routing` is the path of `loop/runs/calibration/routing_table.json` for arm D and null for A, B, C.

## Assumptions

1. `state.json.slot_index` is the number of completed slots (it starts at 0); the stop file "N" means stop once
   `slot_index >= N`. Sentinels are counted from it. `state.json` fields read: `status`, `slot_index`, `n_slots`,
   `updated`, `incumbent.path` (absolute, or relative to the run dir), `incumbent.mean_adjusted`, `incumbent.n`.
2. `<run_dir>/calls.jsonl` is each loop's model-call log (DESIGN section 14), rows shaped like `loop/models/call.py`
   writes them. Spend counts only `provider == "claude"`, via `loop.models.budget.tokens_used` (the same shape is summed
   locally if that module cannot be imported). Open-model tokens are free and not counted.
3. `run_loop` is launched with the controller's own interpreter from the repository root (so `loop` resolves as
   a namespace package) and exits 0 on both `done` and `stopped`. `--resume` is passed on a restart only if
   `state.json` already exists in the run dir; a loop that crashed before writing one is started fresh.
4. Sentinel and launch-sentinel jobs use `sealed=False`, `timeout_s=900`, no `gpu_affinity`, and seeds from a
   deterministic RNG in 100000 to 999999. `QueueClient.submit` currently has no `gpu_affinity` argument; the controller
   never passes it.
5. The queue (`QueueClient()`, default host `rustom@hannah`) is built lazily, only when a sentinel or confirmation job is
   needed. Its calls block the controller for the duration of the ssh retries; the loops keep running meanwhile.
6. Wave and sentinel jobs submitted just before a controller or confirm crash may be submitted again on resume (a few
   duplicate GPU runs at most); recorded results are never duplicated.

## Not implemented as written in DESIGN.md

- Symmetric stop "at the same slot index for all arms in flight" (section 11): loops are at different slot indices at
  hour 19.5, so each running loop is stopped at its own current slot index (PREREG wording: "the current slot index of
  every in-flight loop"). The cut is simultaneous and arm-blind, not equal in slot count.
- Spend check "via budget.over_ceiling": `over_ceiling(log_path, ceiling)` takes one log and the ceiling is across all
  loops, so the controller sums `tokens_used` over all `calls.jsonl` and compares with the ceiling itself.
- The account usage cross-check at hours 1 and 4 (section 11) is not done by the controller.
- "One process running loop coroutines" (section 11) is implemented as one controller process supervising loop
  subprocesses, as the run_loop contract requires.
- Per-loop fresh seeds from "a per-loop RNG seeded by the loop index" (section 6) are drawn by run_loop; the controller
  only hands it `rng_seed`.
- 22 incumbents give 88 confirmation runs in 30 waves (the last wave has one confirmation run and its anchor), so 30
  wave anchors rather than the 29 in section 2's count.
- Added beyond DESIGN.md: the `launch-sentinels` command (without it nothing produces the launch baseline file
  that `anchor` reads).
