# Loop engine (arms A to D)

Implements DESIGN.md sections 3, 5, 6, 8, 10 and the PREREG decision rules. One `LoopEngine` runs one loop run
(20 slots). The controller (separate module) owns blocks, concurrency, sentinels, the symmetric stop and the
confirmation block; it talks to the engine through a config file, `state.json`, `ledger.jsonl` and the stop file.

| file | what |
|---|---|
| `acceptance.py` | pure: `adjust`, `z_score`, `decide_verified`, `decide_base`, `tripwire`, `integrity_flags` |
| `ledger.py` | `new_record`, `Ledger` (append-only JSONL), `results_table`, `claimed_best` |
| `scheduler.py` | backlog items, `seed_prior`, `thompson_pick`, `update`, `Dedup`, stagnation and memo helpers |
| `prompts.py` | the program card, role prompts, JSON schemas (`SCHEMAS[role]`) |
| `services.py` | `ModelService`, `GPUService`, `Clock`, `Services` (the only things the engine touches outside the run dir) |
| `arms.py` | `LoopEngine` |
| `run_loop.py` | CLI |
| `tests/test_engine_*.py`, `tests/fakes.py` | about 120 tests on fakes, no network, no hannah, no sleeping (about 8 s) |

## Run a loop

    /homes/rdubash/active-audit/.venv/bin/python /homes/rdubash/active-audit/loop/core/run_loop.py --config cfg.json
    ... --config cfg.json --resume          # after a kill: reads <run_dir>/state.json, polls the in-flight job

Exit code 0 = done (n_slots used) or stopped by the stop file (`state.json` `status: stopped`; this is the controller's
normal symmetric stop, so it is not a failure). 3 = an infra failure outlasted `max_infra_retries` (`status: stopped`,
nothing lost; the controller treats it as a crash and resumes). 2 = usage error (state.json exists and `--resume`
was not given, or config fields missing). The controller starts loops with
`python -m loop.core.run_loop --config CFG [--resume]` from `/homes/rdubash/active-audit`. To stop a loop cleanly create the file named by `stop_file`; the engine finishes the
slot it is in (a pending replicate or audit stays in `state.cand`) and exits. If the file holds an integer N the loop
stops once `slot_index >= N` instead (the controller's symmetric stop: write the same N to every loop's stop file).
Remove the file and `--resume` to go on.
The engine never kills a GPU job.

Run the tests: `/homes/rdubash/active-audit/.venv/bin/python -m pytest -q /homes/rdubash/active-audit/loop/core/tests/test_engine_*.py`

### Config (JSON)

Required: `loop_id`, `arm` (A|B|C|D), `run_dir`, `n_slots`, `rng_seed`, `sigma_used` (0.006736 in PREREG), `gpu_offsets`
({gpu: offset}, int or string keys), `anchor` ({mean, n}). `base_train_py` defaults to
`loop/substrate/base/train_ampere.py` and is copied to `run_dir/incumbent.py`. `run_loop.load_config` also accepts the
controller's shape: `anchor_file` (anchor.json with `anchor_mean, n, gpu_offsets, sigma_used`) in place of `anchor`,
`gpu_offsets` and `sigma_used`, and `routing` as the path of `routing_table.json` (or null).
Optional (defaults in `arms.DEFAULTS`): `stop_file` (null), `routing` (arm D only: role to model, plus
`coder_by_class` {constant|schedule|structural: model}), `critic_model` (opus), `voter_model` (haiku), `vote_k` (3),
`free_action_caps` ({slot: 2, run: 40}), `timeout_s` (7200, how long to wait for one GPU result per poll round),
`candidate_seed` (42), `backlog_size` (30), `refresh_items` (10), `refresh_below` (5), `critic_every` (8),
`memo_every` (5), `stagnation_k` (5), `dedup_mode` (auto|ngram|embedding), `max_infra_retries` (20),
`infra_backoff_s` (60), `voter_temperature` (0.8), `open_extra` (sent to open models, default disables thinking),
`job_timeout_s` (600, worker-side limit) and `queue` ({host, root} for `QueueClient`).
Roles: `base_step planner_backlog planner_refresh planner_widen voter coder repairer analyst critic`. Only arm D reads
`routing`; A, B, C use sonnet everywhere except voter (`voter_model`) and critic (`critic_model`). In D: coder =
`routing.coder_by_class[class]` (class from the item's family), `planner_refresh` = `planner_refresh`, else
`planner_cheap`, else `planner`; `planner_backlog` and `planner_widen` are sonnet unless the table names them
(DESIGN section 3: Sonnet for the initial backlog and every widening call; `routing_table.json` repeats the cheap
planner under the key `planner`, which is therefore not used for those two); every other role is `routing[role]`,
falling back to `voter_model` / `critic_model` / sonnet when the table lacks it. Every top-level string in the table
is health-checked each slot (non-open models pass).

## What a slot is

A slot is one GPU run (kind candidate, replicate, audit or crash_fix) or one charged event with no run. `run_one_slot()`
does one slot. A candidate that needs a replicate, an audit or a crash fix stays in `state.cand` with `next` set and the
following `run_one_slot()` continues it. Invariant: the number of ledger records with `charged: true` equals
`state.slot_index`.

Per-slot flow (A: no lint, no abort, no replicate; B adds lint, abort, z rule; C adds backlog and critic; D adds routing):

1. D only: health check of every open model in the routing table; a failed one is replaced by its registry fallback for
   this slot (logged in `state.fallbacks`: slot, model, fallback; those calls still count as designated open calls).
2. C, D: refresh the backlog if needed (initial 30 items, refresh when under 5 untried, widening call after 5
   consecutive rejections), 3 votes per new item (`sample_k`, key `vote`) seed the Beta prior (1 + yes, 1 + no), dedup
   (cosine 0.90 with all-MiniLM-L6-v2 when it loads offline, else character 3-gram Jaccard 0.6; the mode used is in
   `state.dedup_mode`), Thompson pick (score = Beta sample / (1 + times_tried); uniform over the top 5 when widening).
3. One edit call (`base_step` in A and B, `coder` in C and D). Patch application = `patch.apply_with_retries` plus
   `compile()`; a failure is retried with the applier's hint as a free action.
4. B, C, D: `linter.lint(new, incumbent)`; on a hard failure one free `repairer` call, a second failure skips the idea
   (free action). C, D: the critic reviews a repaired file before it gets GPU time (veto skips the idea).
5. Submit (seed 42 for a first run, a fresh seed from the loop RNG for replicate and audit, `sealed` only for audit),
   wait, then decide:
   crash or timeout: record `crash`, charge, next slot is the one crash fix (all arms), a second crash is final;
   integrity flags: `invalid`, charge, reject (C, D: critic reviews, no effect on the decision);
   arm A: `decide_base` on the raw value; arms B, C, D: `decide_verified` on adjusted values;
   an accept (any arm) whose mean adjusted gain trips the tripwire first becomes `audit`; after the audit the rule is
   z over all runs of the candidate >= 1.5 and the audit run clean (no flags, sealed value present);
   C, D: critic on audit, on a borderline accept (replicated, z < 2.0), and on every `critic_every`-th record; a veto
   turns an accept into a reject.
6. Accept writes `incumbent.py`, sets `incumbent` to the runs behind the candidate (n = number of runs), resets
   `consecutive_rejections`. Every decided experiment updates `family_stats` and the item (C, D) and, every 5th
   experiment, calls the analyst for a memo (`memo.md`).

Free actions (patch retry, lint repair, abort, skipped idea): 2 per slot and 40 per run. When one is needed and a cap is
reached, the attempt ends and the slot is charged (record with `charged: true`, no run). So a patch that fails three
times in a slot is a charged `invalid` record with `status: engine:patch_fail`.

Infra failures (call error_class `infra` after the call layer's own fallback chain, GPU result `infra_fail`, `error` or
`cancelled`, a failed submit, a wait that returned nothing) write a record with `decision: infra, charged: false`, wait
`infra_backoff_s` through `clock.sleep`, and repeat the same step. A job that ended in infra_fail is resubmitted; a wait
timeout or transport error keeps polling the same job. No posterior, counter or slot changes. After `max_infra_retries`
repeats, or when the stop file appears, the loop stops (`status: stopped`).

## ledger.jsonl record

Field names are the shared schema; one addition (`charged`).

    slot, attempt, kind, idea_id, description, edit_class, model_calls[], job_id, gpu, status, official_val_bpb,
    adjusted_val_bpb, sealed_val_bpb, flags[], z1, z2, decision, incumbent_before{mean,n}, incumbent_after{mean,n},
    lint{ok,hard,warnings}|null, critic{ok,issues,veto}|null, charged, ts

- `kind`: candidate | replicate | audit | crash_fix. `decision`: accept | reject | replicate | audit | invalid | crash |
  lint_fail | infra | abort. On a candidate record `replicate` or `audit` means "that run is scheduled next"; the final
  accept or reject is on the later record. `invalid` = integrity flags, or (job_id null) a patch that never applied.
  `reject` with job_id null = the critic vetoed a repaired file before GPU time; `reject` with `critic.veto` true on a
  run = the critic turned an accept into a reject.
- `charged` (added): whether this record used up a slot. `slot` is the slot being worked on; `attempt` counts ideas
  tried inside that slot (0 for the first, +1 after a free skip).
- `idea_id`: the backlog id (C, D) or `c<slot>.<attempt>` (A, B); replicate, audit and crash_fix share it with their
  candidate. Group records by this to rebuild an experiment.
- `flags`: integrity flags from `acceptance.integrity_flags` (mismatch, target_leak, canary_failed, harness_missing,
  constants, hash, overbudget, invalid_seed, no_official, plus sealed_missing on an audit) and the single flag
  `tripwire` on the record whose accept was turned into an audit. Nothing else is ever put in `flags`, so "every flag
  except tripwire is an integrity flag" (what analysis.py assumes) holds.
- `status`: the worker's status for a run (ok, crash, timeout, ...). For a record without a job it is the engine's
  reason, prefixed `engine:`: patch_fail, model_fail, lint, lint_repair_failed, critic_veto, abort, no_idea,
  infra:model:<role>, infra:gpu:<status>.
- `model_calls`: every model call made since the previous record (planning, votes, memo, failed infra attempts
  included), so summing over records gives total spend. `tokens_in` = input + cache read + cache write.
- `z1` is the first run's z and is repeated on every later record of the candidate; `z2` (n_c >= 2) is set only on
  the replicate or audit record. Arm A also records z1 (descriptive; its rule is `decide_base`).
- `edit_class` comes from the linter's classification against the incumbent (arm A runs the classifier only, never
  gates). `lint.hard` and `lint.warnings` are the linter's lists of {rule, line, message}.
- Numbers are copied from result dicts by code. `description` is the model's own sentence and is never parsed.

## state.json

Required keys (shared): `loop_id, arm, slot_index, n_slots, incumbent{path, sha256, mean_adjusted, n, raw_values,
adjusted_values, gpus, description}, free_actions_run, free_actions_slot, in_flight_job, backlog, family_stats{family:
{alpha, beta}}, consecutive_rejections, memo, rng_state, claimed_best_official, status (running|done|stopped), started,
updated, served_counts{model: n}, designated_open_calls, open_served_calls`.

Additions the engine needs to resume: `attempt`; `cand` (the candidate being evaluated: idea_id, description,
edit_class, item_id, train_py, lint, critic, runs[{job_id, kind, gpu, seed, raw, adjusted, sealed}], next
(replicate|audit|crash_fix|null), is_fix, z1); `in_flight_job` is {job_id, kind, slot, train_py, seed, sealed, tag} and
`job_id` is null between the write that precedes submission and the write that records the id; `backlog_created`;
`experiments`; `last_tail`; `pending_calls` (flushed into the next record); `fallbacks`;
`dedup_mode`, `dedup_dropped`; `infra_events`. `claimed_best_official` = mean raw official of the runs behind the final
incumbent (`ledger.claimed_best`), the "claimed official" of the selection gap. Items in `backlog`: {id, title, spec,
family, alpha, beta, times_tried, status open|done|dropped}.

`served_counts`, `designated_open_calls` and `open_served_calls` count successful calls only. A call is designated open
when the routing table (before any health substitution) names an open-weight model; it is served open when the
`served_model` of the result is open too. Degraded = `open_served_calls / designated_open_calls < 0.8`.

Writes: `state.json.tmp` then `os.replace`, after every model call, ledger append, submission and charge. The ledger
append and the state write are two files; a kill between them can duplicate or lose the last record (resume does not
deduplicate). Run dir: `state.json, ledger.jsonl, incumbent.py, memo.md, calls.jsonl (model layer),
slots/<k>/{reply.txt (every edit reply of the slot, appended), train.py, lint.json, result.json, critic.json}`.

## Assumptions and choices where DESIGN.md is silent

1. A replicate or audit is its own slot, so one candidate can span 2 or 3 slots; `n_slots` counts GPU runs. A
   candidate whose replicate, audit or crash fix would fall beyond the last slot stays undecided: the incumbent is
   unchanged and no posterior moves.
2. The tripwire is checked on the mean adjusted value of the candidate's runs, and only when the rule would accept and
   no audit has run yet; it applies to arm A too (decision on raw values, tripwire on adjusted ones).
3. Audit acceptance = `decide_verified` over all runs of the candidate (n >= 2): z >= 1.5. A very large first-run gain
   can therefore carry a weak audit; the rule is as written in DESIGN section 5.
4. The initial first run uses seed 42 (the harness seed lines, explicitly), replicate and audit a fresh seed drawn
   from `random.Random(rng_seed)` in `[10000, 2^31)`; the same RNG drives Thompson sampling, state holds its state.
5. Linting is against the incumbent, not the original baseline, so `edit_class` and the diff-size warning describe this
   step. The crash fix is not linted for repair (a failing lint ends it) and gets no critic.
6. "Third free action charges the slot" is implemented as: the attempt ends and the slot is charged (no GPU run).
7. Crash fix uses role `repairer` in every arm (so arms A and B call sonnet) and sees `sanitize_tail(log, result)` (raw
   last 25 lines for crash or timeout). The fixed file is a new candidate (kind crash_fix, first-run seed) judged by
   the arm's normal rule; a crash fix is never itself fixed.
8. Votes and planning calls are not slots; their cost rides on the next ledger record. Voting uses temperature 0.8
   (vLLM models; Claude ignores it).
9. Coder routing in D picks the edit class from the item's family (schedule -> schedule; optimizer, batch,
   regularization -> constant; architecture, data, other -> structural) because the measured class exists only after
   the edit. Replace `scheduler.FAMILY_CLASS` if calibration says otherwise.
10. `update()` records an evaluated outcome on the item and its family. Ideas that never reached the GPU stay open
    with a discount and are dropped after 3 such attempts; they never change a posterior. Family stats are shown to
    the coder, planner and analyst but do not feed the sampler (the spec samples the item posterior only).
11. Memo cadence counts decided experiments (accept, reject, invalid, final crash), not slots.
12. The critic on every 8th record runs whatever the decision; its veto matters only for an accept. A failed critic call
    is ignored. `recomputed` is kept in `slots/<k>/critic.json`, never used.
13. `harness_missing` is also raised when `result.harness` holds an `error` key (what the worker writes); `canary_ok`,
    `target_leak_flag` and the hash flags are read from the top level first, then from `result.harness`.
14. `GPUService.submit` passes `kind` and `gpu_affinity` to `QueueClient.submit` only if its signature has them (it does
    not today); both are always in the job `meta`. The engine always passes `gpu_affinity=None`.
15. `patch.apply_with_retries` is used as a one-shot applier plus hint; the retry loop and its budget live in the engine.
16. Kill-and-resume: a kill between writing `in_flight_job` (job_id null) and the id being returned by `submit` makes
    the resume submit again (an orphan job is possible); after the id is stored the resume only polls.
