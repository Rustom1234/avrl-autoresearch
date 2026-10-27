# History of the nanochat loop experiment (AR7)

A record of what was done, in order, with pointers. Dates are October 2026. Numbers quoted here are the ones in the ledger entry AR7 in avrl/RESEARCH_LOG.md and in loop/runs/R1/analysis/report.md.

## 1. Question and settings

Build the strongest autonomous research loop the literature supports, run it on someone else's real metric, and measure what each layer buys. Settings fixed before design: substrate karpathy/autoresearch (nanochat, val_bpb after 300 s), evidence by a control versus full-loop comparison with grouped intermediate rungs, Sonnet as the strong tier with Opus only as critic and adjudicator, a 24 hour run window, open models wherever a calibrated one is good enough.

## 2. Research before design

- Seven-angle literature sweep (lit/avrl/AUTORESEARCH_LIT_SYNTHESIS.md): thirteen design decisions, three next experiments, evidence quality flags. Addendum on Epi (2610.24944).
- Substrate research: repo mechanics at commit 228791f, the four metric-gaming paths in autoresearch issue 599, community results and effect sizes on H100 (most single changes 0.0003 to 0.003, batch halving 0.007 to 0.012), open-model candidates that fit 16 GB cards.
- Hardware facts: hannah n7 has four A100 40 GB, no internet, broken Slurm GPU accounting; the H100 example peaks at 45 GB so DEVICE_BATCH_SIZE drops to 64; flash-attention 3 from kernels-community runs on sm80 offline once cached.

## 3. Measurements before design was frozen

- Smoke runs: baseline 1.1303 at 317 steps (sbatch, three-wide).
- Noise job (loop/runs/noise): 16 baseline runs four-wide, same seed sd 0.0070, fresh seed sd 0.0032, pooled 0.0063, range 0.020.
- Later: 16 live sentinels through the production worker (loop/runs/R1/launch_sentinels.jsonl): mean 1.114885, sd 0.004125, with a significant per-GPU pattern (F(3,12) = 8.58). Worker runs score about 0.03 better than sbatch runs at identical data and code; cause environmental, not isolated (loop/runs/T1/STRESS.md).

## 4. Design, review, pre-registration

- loop/design/DESIGN.md v1, then an adversarial review by three skeptics (25 findings, 5 blockers), then v2.
- loop/design/PREREG.md: hypotheses H1 to H5, six secondaries with Benjamini-Hochberg, validity rules, pre-launch amendments (live anchor and sigma, GPU offsets, frozen routing table, the T1 bug), deviations.
- loop/design/RUNBOOK.md: preconditions, launch, monitoring, wrap-up.

## 5. Build (all under loop/, tests green before launch)

- substrate/: worker.py (queue, node-local staging, affinity, load logging, supervision), harness_tail.py (harness-owned scoring from raw logits, pinned code and data hashes, known-answer check, target-leak probe, seed binding, nonce receipt, external time check), queue_client.py, start_workers.sbatch. 116 tests.
- models/: call.py (Claude headless and vLLM, schema output, fallback chains, rate-limit breaker, concurrency slots), consistency.py, cost.py, budget.py, models.json. 111 tests.
- core/: linter.py, patch.py, sanitize_log.py, acceptance.py, ledger.py, scheduler.py, prompts.py, services.py, arms.py, run_loop.py, controller.py, confirmation.py, calibration.py, analysis.py. 621 tests.
- Servers: vLLM 0.30 on cherry (Qwen3.5-9B), walnut (Qwen3.5-4B), willow (DeepSeek-R1-Qwen3-8B), pine (Gemma 4 12B); gpt-oss-20b excluded (does not fit 16 GB). loop/runs/servers/SERVERS.md.

## 6. Verification before launch

- Smoke run through the worker: harness score equals the printed score and the pinned evaluator to 4e-9; canary ok; all hashes ok; data file hashes pinned.
- Stress tests (loop/runs/T1/STRESS.md): loop kill and resume, worker kill and supervisor restart, server outage and fallback, adversarial candidate (optimizer skipped, fake printed score 0.5; harness scored 3.2158, rejected).
- Integration T1 found an engine bug (every clean result flagged harness_missing); fixed with a regression test. Integration T2 showed accept, reject, tripwire audit, and sealed audit-and-accept working live.
- Calibration (loop/runs/calibration): 504 trials, six models, $2.89; routing table frozen.

## 7. The run (loop/runs/R1)

- Launched 02:51 EDT Oct 21; 22 loop runs (A 8, D 8, B 3, C 3), 20 slots each, six in flight, 36 sentinels; loops finished 16:46 EDT with 440 slots and no integrity flags, no worker restarts, no server outages.
- Confirmation block 18:27 to 22:23 EDT (started 100 minutes late, watcher self-match bug): 88 fresh-seed re-runs plus 30 same-wave anchors on official and sealed shards.
- Analysis as pre-registered: loop/runs/R1/analysis/report.md and six figures.

## 8. Results

Sealed improvement over the same-wave anchor: A 0.0633, B 0.0629, C 0.0607, D 0.0595. H1 inconclusive (D minus A minus 0.0038, 90 percent interval minus 0.0084 to plus 0.0009, p 0.18), as predicted. Model dollars per run A 0.78, B 1.06, C 1.44, D 0.21; D 9x cheaper per accepted change than C, noninferior at 0.01; both cost contrasts significant after correction. The control's keep-if-lower rule proposed 7.1 keeps per run of which 5.0 failed a fresh-seed audit; the verified arms rejected those first. Shard gap under 0.0003 everywhere. Frozen GPU offsets did not hold during the run. Total project spend about $222 plan equivalent, of which the run's model calls about $15.

## 9. Follow-ups considered worthwhile

1. Rerun the control without the uniform tripwire so the optimism hypotheses get a real test.
2. Start all arms from the batch 2^17 and depth 6 baseline so only small effects remain.
3. Zero-offset replay of decisions and a rule for GPU affinity.
4. Harder routing test: open models on structural edits with a repair budget; Opus as strong tier; live planner routing.
5. Branch-and-revert gated by the replicate rule; cross-run lessons memory.
6. A second substrate (cheap CPU benchmark).
7. Recover the control's would-have-claimed scores from its ledger.
8. Score saved weights in a fresh process.
