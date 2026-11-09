# P2 preregistration: which loop role earns the expensive model (planner x coder factorial)

Status: FINAL TEXT, frozen when its sha256 is written to avrl/RESEARCH_LOG.md (see the end of this file). It must be frozen before the first P2 job, including the launch sentinels. Changes after freezing go to the Deviations section with a time and a reason.

## Question

R1 routed the coder, voter and repair roles but never the planner; every planner call went to Sonnet. The routing table was built from per-role tryouts, and five models scored 12 of 12 on the planner tryout, which checks only the reply format, not plan quality. P2 asks two things:
1. Which role's model tier changes the loop's outcome: the planner (the idea lists) or the coder (the edits)?
2. Does a per-role tryout predict what happens inside the loop? Both cheap tiers in this design passed their tryouts, so any role where the cheap tier hurts is a role the tryout missed.

## Substrate (unchanged from R1)

Karpathy autoresearch at commit 228791f. The training budget is 300 s, and train.py is the only editable file. The official metric is val_bpb on pinned shard 06542; the sealed held-out shard is 06541. The harness, worker, linter and z-rule (reject below 0.75, accept at 2.0 or above, replicate in between and accept at 1.5 on the mean) are the same code and versions as R1, checked by sha256 against loop/substrate. Each loop has 20 slots.

## Cells

Every cell runs the arm D engine: a backlog of 30 ideas with Thompson sampling, the linter with repair, the critic every 8 records and on borderline and audit cases, an analyst memo every 5 experiments, and the z-rule. Only the routing table differs. The tables are written by loop/core/plan_p2.py into the run's routing/ directory and frozen there.

| Cell | Planner (backlog, widen, refresh) | Coder | Repairer | Voter | Analyst | Critic | Loops |
|---|---|---|---|---|---|---|---|
| SS | sonnet | sonnet | sonnet | gemma4-12b | sonnet | opus | 12 |
| ST | sonnet | table: constant qwen35-4b, schedule gemma4-12b, structural haiku | haiku | gemma4-12b | sonnet | opus | 12 |
| QS | qwen35-4b | sonnet | sonnet | gemma4-12b | sonnet | opus | 12 |
| QT | qwen35-4b | table, as ST | haiku | gemma4-12b | sonnet | opus | 12 |
| H (harm control) | as SS, but on half of the coder's new proposals the edit is replaced by a comment-only no-op that reruns the incumbent | | | | | | 12 |

- **ST is R1's arm D configuration as run.** R1's table would have sent the planner top-up (planner_refresh) to qwen35-4b, but no top-up ever fired in R1; ST sends it to Sonnet so that the planner factor is clean.
- **The voter is held at gemma4-12b in every cell,** so the planner factor varies only the idea lists.
- **The repairer follows the coder factor,** because repairs are edits.
- **The coder table was requalified on held-out items before freezing** (loop/runs/calibration_requal/RESULT.md): Haiku passed 30 of 30 new structural items (Wilson lower bound 0.886 against a bar of 0.75) and 20 of 20 new repair items (0.839 against 0.6), so the table is unchanged.
- **Fallback.** When an open server fails its health check, calls fall back to Haiku for that slot, as in R1, and are logged.
- **Viability check done before freezing.** Live calls to qwen35-4b with the real 30-item backlog prompt returned 29, 30 and 27 valid items (the engine keeps what it receives), a 10-item widen call returned 10, and qwen35-4b and gemma4-12b coder edits applied, compiled and passed lint.

**Harm control mechanics.** For each new-idea attempt, a deterministic hash of (loop seed, slot, attempt) decides with probability 0.5 whether the coder's applied edit is replaced by the incumbent plus one trailing comment line. The run, decision, ledger record and planner feedback then proceed normally, so a good idea is recorded as tested and failed. Crash fixes and repairs are never replaced. Each replacement is logged in the slot directory (harm.json, plus the withheld edit) and in state.json (harm_noops).

## Schedule and blocking

- **Sentinels and anchor.** 16 live sentinel runs of the unmodified base script come first. anchor_mean is their mean, and sigma_used is 1.15 times their sd (R1's rule), written by loop/core/anchor_p2.py. gpu_offsets are zero: R1's frozen offsets did not hold, and its 52 sentinels show no GPU effect (F(3,48) = 0.82, p = 0.49). GPU is reported as a blocking factor instead.
- **Randomized blocks.** 12 blocks, each holding one loop of each of SS, ST, QS, QT and H, in random order (plan seed 2). Loops launch in block order. The block order and seeds are fixed in plan.json before launch.
- **GPUs.** Workers run on hannah n7 A100s that no other job is using; the worker script refuses a GPU that already holds more than 2000 MiB. The GPU of every job is recorded, and jobs flagged as sharing their GPU with a foreign process are listed, with a sensitivity verdict on the loops that have none.
- **Sentinels during the run.** One sentinel per 45 completed slots (about 27 in total) tracks drift.

## Outcomes, in order of priority

1. **Stall**: the final incumbent's TOTAL_BATCH_SIZE is above 2^17, parsed from the incumbent train.py (last module-level assignment wins). In R1 this flags 4 of 22 loops (C_03, D_01, D_03, D_05), and it separates held-out gain cleanly: 0.047 to 0.056 for those four, against 0.060 to 0.066 for the other 18. All four took six layers early and never reached the smaller batch. Validated on the R1 burn-in half before freezing (loop/design/P1_PREREG.md). Loops that stop early are counted with the incumbent they had, so an early stop caused by a treatment cannot hide.
2. **Slots to 2^17**: slots used (slot index + 1) up to the first accepted change whose source has TOTAL_BATCH_SIZE at most 2^17, censored at the loop's finished slots (20) if never reached. Groups are compared by log-rank test and by the Kaplan-Meier median; a median that is not reached counts as infinite. Secondary: slots used up to the last incumbent change, with a loop that never changes censored.
3. **Dollars per backed accepted change**: model dollars (Claude at CLI list price, open models at loop/models/open_prices.json) divided by accepted changes that rest on at least two runs with different seeds (replicate or audit path). Intermediate incumbents get no confirmation run, so this is the available fresh-seed check. Also reported GPU-inclusive at 0.5, 1, 2 and 4 USD per A100-hour, because GPU time was 76 to 95 percent of R1's loop dollars.
4. **Sealed held-out gain** (secondary): the final incumbent of every loop is rerun twice (seeds 1001 and 1002) in the confirmation block, scored on the sealed shard. Gain is the mean of all confirmation anchors minus the loop's mean; the paired same-wave version is reported too. R1 showed the paired version adds anchor noise.

## Analysis

- **Main effects.** Planner (Q against S) and coder (T against S), each over the 48 factorial loops (24 against 24), for outcomes 1 to 3. Interaction is descriptive.
- **Tests.** Stall: Fisher exact test. Slots: log-rank and Kaplan-Meier medians. Cost: permutation test on log dollars. Sealed gain: permutation test, two-sided, with 90 percent intervals.
- **Harm control.** H against SS on slots to 2^17 (log-rank, one-sided, alpha 0.05) and on stall.
- **Manipulation checks.** The planner served fraction per loop (the share of planner calls served by the designated model; Q loops below 1.0 are named), open-model served fractions, health-check fallbacks, and harm no-ops per H loop.
- **Code.** loop/core/analysis_p2.py; margins are the constants at its top.

## Decision rules

Margins: stall +0.25; median slots to 2^17 +5 slots. The slot margin comes from the P1 burn-in: arm D's slot endpoints had an sd of about 6 slots, so a 24 against 24 comparison has a 90 percent half-width of about 3 slots, and a margin of 3 would fail half the time with no true delay.

For each role (planner Q against S, coder T against S):
- **cheap tier hurts** if its effect on stall exceeds +0.25 or its effect on median slots to 2^17 exceeds +5;
- **cheap tier holds** if both effects are below their margins;
- **unclear** otherwise (for example, an effect that cannot be computed).

Run verdicts, with precedence KILL, then PIVOT, then CODER HURTS, then PASS, and NO VERDICT when none applies:
- **KILL.** The harm control is not detected (H not slower than SS at one-sided p < 0.05). The factorial result is then reported only as a cost breakdown with the per-role numbers shown as description. Power note: by bootstrap of the R1 times, 12 H loops against 12 SS loops detect the harm in about 47 to 89 percent of runs (32 to 73 percent with 6). KILL therefore means the harm was not detected, not that the substrate is blind.
- **PIVOT.** The cheap planner hurts. Headline: the 12-of-12 planner tryout missed the role that matters.
- **CODER HURTS.** The cheap coder table hurts and the cheap planner does not. Headline: the coder tryouts, which Haiku passed on held-out items, missed what matters inside the loop. R1 already hints at this (routed loops reached 2^17 later), but there the planner was fixed and acceptance rules differed between arms.
- **PASS.** Harm detected, both roles hold, and QT is at least 5 times cheaper than SS in model dollars per backed change. Claim: on this substrate the per-role tryouts were a valid routing signal for both roles.

## Budget

- **Jobs.** 60 loops x 20 slots = 1,200 jobs, plus 16 launch sentinels, about 27 interleaved sentinels, and about 160 confirmation jobs (2 per loop plus anchors): about 1,400 jobs.
- **GPU time.** About 0.117 A100-hours per job, or about 165 A100-hours: about 82 hours on 2 GPUs or about 41 hours on 4. The worker job has a 3-day cap, so a 2-GPU run needs one resubmission (P2_RUNBOOK.md).
- **Claude.** About $50 at list price on the Max plan. The controller enforces a hard ceiling of 60M Claude tokens.
- **No real money.**

## Deviations

1. **2026-11-07, model provider outage; 28 loops rerun.** From 14:19 UTC to about 16:10 UTC every Claude call failed with an account-level API refusal (HTTP 403) by the model provider. The call layer counted these as model failures, so the engine charged slots for them. 28 loops have at least one refused call: SS, ST and H 07 to 12, and QS and QT 08 to 12. The other 32 loops had no refused call and are kept as run. Actions, chosen by the outage alone (which loops had a refused call), not by any outcome, and without changing the design. The interim all-60 analysis below had already been viewed when the outage was found:
   - loop/models/call.py now treats this account-level refusal (HTTP 403 or its refusal message) like a rate limit: the loop backs off for up to about 2 hours instead of charging slots. Unit tests cover it. Calls that succeed behave exactly as before.
   - The 28 affected loops are rerun from scratch with identical configs (same seeds, routing tables, harm rate and anchor) in loop/runs/P2R, built by loop/runs/P2R/make_rerun_plan.py. Their damaged first attempts stay in loop/runs/P2 for the record and are excluded from the analysis.
   - The confirmation block and the analysis run once, on the 32 kept loops plus the 28 reruns. An interim run of the analysis on all 60 loops (loop/runs/P2/analysis_preconfirm) included the damaged loops and is void.
   - Reruns happen about a day after the kept loops, on the same node and code, so time is a possible confound; the report will compare the kept and rerun loops of each cell.
