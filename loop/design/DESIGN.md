# Design v2: a verified, scheduled, heterogeneously routed autoresearch loop on Karpathy's nanochat substrate

Status: v2 after the adversarial review (25 findings, 5 blockers) and the 16-run noise job. Thresholds are in PREREG.md. v1 is in git history of this file. Pre-launch amendments (anchor and sigma from 16 live worker sentinels, GPU offsets from the same, frozen routing table) are listed in PREREG.md and override the numbers quoted below where they differ.

## 1. Question

At equal GPU slots on karpathy/autoresearch (commit 228791f, metric val_bpb after 300 s of training, train.py the only editable file), does a loop built from the practices the literature supports (deterministic verification, noise-aware replicate-before-accept, sealed confirmation, a statistical backlog scheduler, and routing roles to the cheapest model measured good enough) reach a better held-out metric than a stateless one-shot strong-model control, and what does each layer cost in model dollars?

Wording constraint from the review: arm A is a stateless, tool-less, one-shot control that follows program.md's keep rule with the same prompts and output format as the other arms. It is not Karpathy's persistent tool-using session, and the write-up must not call it that.

Hardware and baseline: hannah n7, four A100 PCIe 40 GB, offline, shared NFS home. Ampere adaptation: DEVICE_BATCH_SIZE 64 (128 overflows). Flash-attention 3 from kernels-community runs on sm80 unchanged. Noise job (16 runs, four at a time): mean 1.1436, pooled sd 0.0063, same-seed sd 0.0070, fresh-seed sd 0.0032, residual sd after per-GPU means 0.0059, range 0.020. Per-GPU step counts differ (GPU1 about 5 steps slower, about +0.005 bpb). Community single-change effects on H100 are mostly 0.0003 to 0.003; batch halving is 0.007 to 0.012. Consequence: a single run cannot resolve most real effects, and a keep-if-lower rule will mostly keep noise. The design treats that as the thing to measure.

## 2. Units and budget

- Slot: one GPU run of a candidate train.py (about 7.5 min including the harness evaluator). Replicates and crashes consume slots. Infrastructure failures (worker death, server outage, rate limit) do not.
- Loop run: one arm instance with N_SLOTS = 20 slots, ending with a final incumbent train.py. A loop run also ends early at the symmetric stop (section 12) or at hour 19.5 of the window; truncated runs are analysed intention-to-treat with the incumbent at stop and slots completed as a covariate.
- Loop runs: A x 8, D x 8, B x 3, C x 3 = 22, run as rolling randomized blocks (section 11).
- Anchor: the unmodified Ampere baseline. Anchor value = mean of the 16 noise-job runs plus 8 live sentinel runs made at launch under four-wide load (n = 24). GPU offsets = per-GPU mean minus grand mean over those 24 runs, frozen at launch and subtracted from every run value used in a decision. Sentinel baseline runs continue through the queue at one per 12 loop slots; their sd is the live sigma check (PREREG).
- Confirmation block: after the last loop run ends, every final incumbent is run 4 times (seeds 1001 to 1004, identical for all arms) in randomized four-wide waves that mix arms, with one anchor run per wave, each run evaluated on the official val shard and the sealed shard 06541. Improvement of an incumbent = same-wave anchor sealed value minus incumbent sealed value, averaged over its 4 runs. Reserve hours 19.5 to 23.5 for this block.
- GPU time: 440 loop slots + 37 sentinels + 24 anchor runs + 88 confirmations + 29 wave anchors = about 618 runs, about 77 GPU-hours, about 19.5 hours wall at four-wide. Buffer about 4 hours.

## 3. Arms

Everything below the agent layer is identical across arms: worker, harness-owned metric, receipts, tripwires, seed control, anchor, confirmation, cost accounting, program card, patch application and the parse or syntax retry budget.

A. BASE (control). One strong model (Sonnet) per iteration, stateless: it sees the program card, the current train.py, the code-generated results table, and the last run's tail, and returns SEARCH/REPLACE blocks plus a description. One run per candidate. Keep if the raw official val_bpb is strictly lower than the incumbent's raw value (initial bar = anchor mean, a stated deviation from program.md, which would use a single baseline run). A crash gets one fix attempt that costs a slot. No linter, no replicates, no backlog, no critic.

B. VERIFIED = A plus the verification stack, same single Sonnet caller: linter before GPU time (section 7) with one free repair call per lint failure; z-scored replicate-before-accept rule (section 8); graded ledger records written by code; abort action.

C. SCHEDULED = B plus the planning layer: self-generated backlog (30 items at start, refresh under 5 untried), Beta posterior per item seeded by three cheap votes (Haiku), Thompson sampling with score p / (1 + times_tried), embedding dedup (all-MiniLM-L6-v2, cosine 0.90), stagnation widening after 5 consecutive rejections, memo every 5 experiments, Opus critic on escalation (lint repair, integrity flag, borderline accept with z2 under 2, every 8th record) with veto.

D. ROUTED = C plus heterogeneous routing by measured role fitness (section 9): coder is the cheapest qualifying model per edit class with Sonnet for structural edits unless an open model qualifies at the higher bar; repair escalates one tier; planner is Sonnet for the initial backlog and every widening call, the calibrated cheap planner for refreshes; voters are three samples from a different model family than the planner (Gemma 4 12B if it qualifies, else DeepSeek-R1-Qwen3-8B); analyst is the calibrated open reasoning model; critic is Opus. Fallback chains are logged; a D run with under 80 percent of designated open-model calls actually served by open models is marked degraded and analysed as a sensitivity case.

Contrasts: A vs D is the single confirmatory test (bundle of all layers). A vs B isolates verification, B vs C planning, C vs D routing; these are descriptive (estimate and 90 percent interval) because 3 runs cannot reach significance. Model-dollar cost versus A is uncontrolled and reported, not claimed.

## 4. Harness-owned metric (all arms)

Each run executes the pinned prepare.py (sha256 checked at worker start) plus the candidate train.py plus an appended harness tail, with a per-job nonce and result path passed by the worker. The tail:
- checks the pinned constants (TIME_BUDGET, EVAL_TOKENS, MAX_SEQ_LEN, VOCAB_SIZE, VAL_FILENAME, DATA_DIR and TOKENIZER_DIR suffixes) and the bytecode hashes of evaluate_bpb, make_dataloader, _document_batches, get_token_bytes, list_parquet_files, Tokenizer.encode, and the Tokenizer class identity;
- hashes token_bytes.pt, tokenizer.pkl and the val parquet actually read, against pins taken from the first smoke run;
- scores the official metric itself from raw logits: `model(x)` with no targets, cross-entropy computed in the tail, token_bytes loaded by the tail, byte and token totals asserted against EVAL_TOKENS, fixed eval batch 64 with OOM fallback to 32 and 16; the pinned evaluate_bpb value is kept as a secondary number;
- runs a known-answer canary (zero-logit stand-in model must score exactly log2(8192) bits per token) to detect tampering with Tensor.item, math or F;
- runs the target-leak probe (loss from model(x, y) versus cross-entropy from model(x) on one batch; difference over 0.01 flags);
- records torch.initial_seed and compares it with the job seed;
- when a sealed directory is given, repeats the scoring on the sealed shard;
- writes the receipt (nonce, sha256 of train.py, parameter hash and count, shard hash, eval batch used, timings).
The worker records the printed val_bpb separately, measures startup-plus-training wall time externally (tail start minus launch; over 450 s is overbudget), records load average and foreign GPU processes before and after, and raises integrity flags: printed versus official mismatch over 1e-4, any hash or constant check failed, canary failed, harness result missing or nonce wrong or nonzero exit, target leak, invalid seed (seed lines not matched or initial_seed mismatch on a seeded job), overbudget. An integrity flag auto-rejects the candidate in every arm and the slot is charged. The agent's printed number is never used for a decision.

Limits stated in the write-up: the tail runs inside the candidate's process, so this defends against accidents and opportunism, not a deliberate adversary with environment access; the sealed shard detects validation-specific overfit, not gaming in general.

## 5. Tripwires (all arms, mechanical)

- Single-step gain over the incumbent larger than 5 standard errors (5 x sigma_used x sqrt(2)), or cumulative gain over the anchor larger than 0.03 within a loop run, forces an audit before acceptance: one extra run of the candidate with sealed evaluation and all probes, on a fresh seed, charged as a slot. Accept only if the audit run is clean and z against the incumbent is at least 1.5 using both runs. In C and D the Opus critic also reviews the diff. Tripwire counts per arm are a secondary outcome.
- Any integrity flag: reject, charge the slot, record.

## 6. Seeds (all arms)

The two seed lines are owned by the harness. The program card says so for every arm. A replicate, audit or confirmation job is invalid (status invalid_seed, charged to the slot, candidate rejected) unless both seed lines were matched and the tail confirms torch.initial_seed equals the job seed. Fresh seeds are drawn by the controller from a per-loop RNG seeded by the loop index.

## 7. Static linter (arms B, C, D)

As implemented in loop/core/linter.py: parse; forbidden imports (subprocess, socket, requests, urllib, http, shutil, signal, ctypes, importlib, multiprocessing, sys, pickle, inspect, builtins, pathlib, io, glob, runpy); forbidden calls (open, exec, eval, bare compile, setattr, delattr, globals, locals, vars, __import__, breakpoint, input, getattr on guarded modules or with non-literal names); os limited to os.environ and os.path with AR_ environment keys forbidden; protected names imported from prepare and module roots; attribute assignment on module roots except torch.backends and torch._dynamo.config; attribute denylist (DATA_DIR, VAL_FILENAME, CACHE_DIR, TOKENIZER_DIR, list_parquet_files, _document_batches, get_token_bytes and the other prepare internals); dunder access; "val" split and denylisted string literals including literal concatenation; the prepare import line and an evaluate_bpb call at module level; both seed lines unchanged; GPT.forward signature unchanged; no try/except around the eval or the final print block; file under 64 KB. Warnings: diff over 300 lines, new imports, DEVICE_BATCH_SIZE not dividing 10240, new training-mode branches. One free repair call per lint failure (cap in section 12); a second failure skips the idea.

Arm A gets none of this except what the harness enforces for every arm (section 4 to 6). The program card is identical across arms and states the harness facts (seed lines owned by the harness, forbidden actions, what gets a run rejected) without exhortation.

## 8. Acceptance rule (arms B, C, D)

All values GPU-offset adjusted. sigma_used is fixed in PREREG (1.15 x residual sd of the noise job = 0.0067). z1 = (incumbent_mean minus candidate) / (sigma_used x sqrt(1/n_c + 1/n_i)) with n_c = 1 and n_i the number of runs behind the incumbent (24 for the initial anchor).
- z1 < 0.75: reject.
- z1 >= 2.0: accept; the incumbent becomes the candidate with n_i = 1 (its one run).
- otherwise: replicate once with a fresh seed; z2 computed with n_c = 2; accept if z2 >= 1.5, else reject.
Expected under the noise measured: about half as many replicate slots and bad accepts as the v1 rule at equal expected gain (review simulation). PREREG states the expected sign: verified arms may show equal or smaller claimed gain than A while showing fewer bad accepts and a smaller selection gap.
Arm A: keep if raw candidate < raw incumbent bar, no replicates (program.md).

## 9. Calibration and routing table (before launch, arm D)

Models: sonnet, haiku, qwen35-9b (cherry), gemma4-12b (pine), r1-qwen3-8b (willow), qwen35-4b (walnut); gpt-oss-20b does not fit the 16 GB cards and is excluded. Coder: 8 edit specs per class (constant, schedule, structural) written from the base train.py, 2 samples each per model (temperature where available), scored by apply-clean, parse, lint pass and target-symbol changed; qualify when the Wilson 95 percent lower bound of the pass rate is at least 0.6 (structural: 0.75). Repairer: 6 lint-failing files, same scoring. Planner-cheap and analyst: 6 inputs each, scored by schema validity, item count, and no numbers absent from the input (checked by code). Voters: no ground truth; the family rule applies. The routing table is frozen in models.json and runs/calibration/routing_table.json before block 1; calibration cost and latency are recorded. Open-model servers are health-checked at the start of every D slot; a failed check routes to the fallback and is logged.

## 10. Prompts, context, and free actions

One program card for all arms: program.md adapted to this hardware (A100 40 GB, batch 64 baseline, 300 s budget, same metric), the SEARCH/REPLACE output format, the harness facts, and the slot budget. Current train.py in full. Results table generated by code from result.json fields (experiment, description, official val_bpb, adjusted delta, decision, runs, flags). Last run tail: for a clean run the code-built summary; for a crash the last 25 raw lines including the traceback, control characters stripped, lines cut at 300 chars. C and D add the backlog with posteriors and family statistics and the memo. Patch application failures (parse, no match, multiple match, syntax) get the same retry budget in every arm: 2 free retries per slot attempt with the applier's hint. Free actions (abort, retries, lint repairs, skipped ideas) are capped at 2 per slot attempt and 40 per loop run; a third per slot charges the slot. Model calls per slot are logged as a cost covariate.

## 11. Scheduling and operations

- Rolling randomized blocks: 8 blocks, each with one A and one D; blocks 1, 3, 5 add a B and blocks 2, 4, 6 add a C; order within a block from a seeded RNG. Up to 6 loop runs in flight so at least 4 GPU jobs are always queued. A block's loops start as slots free up.
- Controller: one process on spruce running loop coroutines; every loop writes state.json atomically after every event (incumbent path and hash, slot index, ledger, backlog and posteriors, RNG state, in-flight job id written before submission). A resume command polls in-flight jobs instead of resubmitting. The controller runs under nohup with a restart wrapper and a lock file; a kill-and-resume test is part of the stress tests.
- Workers: four per node pinned to 8 cores each, data staged to node-local disk with HOME override, supervisor in the Slurm job restarts dead workers and requeues stale running jobs (retry cap 2), queue client can requeue from the login side, job requeue enabled.
- Claude calls: at most 4 concurrent headless processes (flock semaphore); rate-limit signatures trip a shared circuit breaker so every loop pauses together with jittered backoff; infrastructure failures never update posteriors or the ledger and never charge a slot.
- Spend: token ceiling from calls.jsonl cross-checked against ccusage at hours 1 and 4; the stop is symmetric (same slot index for all arms in flight).
- Prompt-injection surface: raw crash tails are shown only for crashes; summary numbers come from result.json.

## 12. Metrics and inference (thresholds in PREREG.md)

Primary: sealed improvement over the same-wave anchor of the final incumbent, mean of 4 confirmation runs, per loop run. Confirmatory test: A vs D, two-sided exact permutation on the difference of means, with a pre-registered margin. Rung contrasts A vs B, B vs C, C vs D: estimate with 90 percent bootstrap interval, no p-values. Secondaries (at most 6, Benjamini-Hochberg q under 0.10, exploratory): selection gap gap_sel = claimed official minus confirmed official (same-wave anchor adjusted); shard gap gap_shard = confirmed official minus confirmed sealed minus the anchor's own official-minus-sealed offset (integrity metric); accepts per loop run; integrity and tripwire counts; model dollars (plan and nominal) per loop run; D's served fraction and cost per accepted change. Extra seeds, if time remains, are decided at hour 16 blind to outcomes and added equally to A and D; everything is analysed once.

## 13. Related systems and what is different

AutoScientists (arXiv 2605.28655) gates promotion at 2 sigma with a second-seed confirmation and critiques proposals before GPU time on this substrate; it does not separate verification from planning, does not route roles by measured fitness, and reports no sealed holdout or same-load confirmation. Rehearse (2607.27687) ranks ideas with an LLM judge and finds the judge decays late in a run; the scheduler here uses cheap-vote consistency plus outcome feedback. Ferreira et al. (2603.24647) show classical hyperparameter optimizers beat LLM agents in a fixed space; a numeric operator is future work. Karpathy's discussions 32 and 43 record a seed change kept as an improvement and effects that did not replicate; the noise job here quantifies why.

## 14. Interfaces for the build (what each module produces)

- loop/substrate: worker.py, harness_tail.py, queue_client.py (QueueClient.submit returns job_id; job JSON fields: job_id, kind, train_py, seed, sealed, timeout_s, priority, tag, meta, gpu_affinity). result.json fields used by the loop: status (ok, crash, timeout, error, overbudget, invalid_seed, infra_fail, cancelled), official_val_bpb, sealed_val_bpb, gpu, printed{...}, harness{...}, mismatch_flag, target_leak_flag, canary_ok, harness_missing, seed_patched, seed_lines_matched, overbudget, load_flagged, eval_batch_size_used, wall_seconds, log_tail.
- loop/models: call(model_name, user, system, schema, temperature, max_tokens, tag, extra) -> CallResult(ok, obj, text, error, error_class, requested_model, served_model, usage, cost_usd_plan, cost_usd_nominal, latency_s); consistency.sample_k; healthy(); budget.tokens_used; models.json with fallback chains.
- loop/core: patch.apply_text and apply_with_retries; linter.lint(new, base) -> LintResult(ok, hard, warnings, diff_lines_changed, edit_class); sanitize_log.sanitize_tail(log, result).
- loop/core (to build): acceptance.py (z rules, GPU offsets, anchor), ledger.py (records and results table), scheduler.py (backlog, Beta posteriors, Thompson, dedup, stagnation, memo cadence), prompts.py (program card and role prompts), arms.py (A to D step logic), controller.py (blocks, concurrency, state.json, resume, sentinels, symmetric stop, confirmation block, spend check), calibration.py (section 9), analysis.py (section 12, figures).
- Run directory: loop/runs/<RUN_TAG>/<arm>_<idx>/{state.json, ledger.jsonl, incumbent.py, slots/<k>/{reply.txt, train.py, lint.json, result.json}, calls.jsonl, memo.md}; controller-level: plan.json, sentinels.jsonl, anchor.json, confirmation/, spend.jsonl.
