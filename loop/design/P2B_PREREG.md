# P2B preregistration: replication of P2 with a routing endpoint (gap recovered minus extra cost)

Status: FINAL TEXT, frozen when its sha256 is written to loop/design/P2B_PREREG.sha256. It must be frozen before the first P2B job, including the launch sentinels. Changes after freezing go to the Deviations section with a time and a reason.

## Why this study

P2 (60 loops) found a large planner effect and no detectable coder effect, but its preregistered verdict is KILL: the damaged loop (H) was not detected on the registered endpoint (slots to 2^17, p = 0.22), because that endpoint had too few events. On sealed held-out gain H did lose (-0.022 against SS, p = 0.005), but that was not the registered test. P2B reruns the key cells on fresh seeds with the endpoint and the validity check fixed in advance on sealed held-out gain. It asks:
1. Does a fixed routing table (strong planner, cheap coder) beat the straight line between "all cheap" and "all strong"?
2. Do P2's two leads replicate: the planner needs the strong model, the coder does not?
3. Secondary: how do a first-list rule, random routing tables and a middle-rung planner score on the same endpoint?

## Substrate (unchanged from P2)

Karpathy autoresearch at commit 228791f, 300 s training budget, train.py the only editable file, official metric val_bpb on shard 06542, sealed shard 06541, same harness, worker, linter and z rule (reject below 0.75, accept at 2.0 or above, replicate in between and accept at 1.5 on the mean), checked by sha256 against loop/substrate. 20 slots per loop. Engine: arm D, as in P2. The loop/ code is not edited while P2B runs, except new files that running code does not import.

## Cells

Routing tables are written by loop/core/plan_p2b.py (which reuses loop/core/plan_p2.py) and frozen in the run directory. In every cell the voter is gemma4-12b, the analyst is sonnet and the critic is opus, as in P2. "Cheap table" means P2's coder table: constant edits qwen35-4b, schedule edits gemma4-12b, structural edits haiku, repairer haiku.

| Cell | Planner | Coder and repairer | Role | Loops |
|---|---|---|---|---|
| SS | sonnet | sonnet | strong anchor ("all strong") | 12 |
| QT | qwen35-4b | cheap table | cheap anchor ("all cheap") | 12 |
| ST | sonnet | cheap table | the fixed routing table (primary policy) | 12 |
| H | as SS, with half of the coder's new proposals replaced by a no-op (P2's harm mechanics, rate 0.5) | | damaged loop (validity check) | 12 |
| F | sonnet for the initial backlog only, qwen35-4b for every later planner call | cheap table | first-list rule | 12 |
| R | for each routable role key (each planner call type, each coder edit class, repairer) a fair coin picks the SS model or the QT model, drawn once per loop from the plan seed and the block (nine keys; two of the five planner keys are unused fallbacks in this engine, so seven coins matter) | | random routing tables | 12 |
| M | haiku for every planner call | cheap table | middle-rung planner | 12 |

- **Anchors are named for the routed roles only.** "All cheap" still uses sonnet as analyst and opus as critic, as every cell does. The endpoint compares policies for the planner, coder and repairer.
- **M and the offline ladder.** Before freezing, five models each wrote 12 first idea lists from the real backlog prompt with no GPU runs (loop/runs/LADDER1/RESULT.md). Rule fixed before the ladder ran: the cheapest model other than sonnet whose first lists contain the batch-size cut in at least 10 of 12 becomes the M planner; if none qualifies there is no M cell. Outcome: haiku qualifies (12 of 12) and is the only qualifier; qwen35-4b, qwen35-9b and gemma4-12b had the cut in 4, 5 and 4 of 12 lists; sonnet 12 of 12. So M runs with haiku as planner. The ladder scorer is stricter than the hand count behind the "10 of 24" figure in the P2 report (it gives 5 of 24 on the stored P2 lists and 36 of 36 for sonnet).
- **QS is dropped.** P2 measured it (strong coder with weak planner); it is not needed for the endpoint.

## Schedule and blocking

- **Sentinels and anchor.** 16 launch sentinels of the unmodified base script; anchor_mean is their mean and sigma_used is 1.15 times their sd; gpu_offsets are zero (as P2).
- **Randomized blocks.** 12 blocks, each holding one loop of every cell in random order, plan seed 3 (P2 used 2). Order and seeds are fixed in plan.json before launch.
- **GPUs.** One whole-node Slurm job on hannah n7 (4 A100s), submitted only when no other job is on the node. Sentinels every 45 completed slots.
- **Confirmation.** Every loop's final incumbent is rerun twice (seeds 1001 and 1002) on the sealed shard, as in P2.

## Outcome and endpoint

**Outcome: sealed held-out gain** of each loop (mean of all confirmation anchors minus the loop's confirmation mean; the "global" version in P2's analysis). The loop never sees the sealed shard.

**Endpoint for a policy P (a cell other than the anchors):**
- gap recovered = (gain(P) - gain(QT)) / (gain(SS) - gain(QT))
- extra cost = (cost(P) - cost(QT)) / (cost(SS) - cost(QT))
- advantage = gap recovered - extra cost

Gains and costs are cell means. Cost is model dollars per loop (Claude at CLI list price, open models at loop/models/open_prices.json). Every loop is charged the same GPU budget (20 slots), so GPU dollars cancel; P2 showed that cheap loops use fewer GPU hours only because more of their edits fail before reaching a GPU. The total-dollar version (GPU at 2 USD per A100-hour as used) is reported as context only. Reading: 0 is the straight line between the anchors (what mixing the two anchors at random would give), 1 is the strong loop's gain at the cheap loop's price. The ratio follows RouteLLM's "performance gap recovered"; subtracting the cost fraction is this project's own addition.

**Intervals.** Block bootstrap over the 12 blocks, 10,000 draws, 90 percent percentile interval. Draws where gain(SS) - gain(QT) is at most 0.004 (one noise sd) are counted and reported.

## Decision rules

Tests on gain differences are sign-flip permutation tests on within-block differences (exact, 4,096 sign patterns), one-sided where a direction is stated.

**Gates (checked first; either failure gives the verdict KILL and nothing confirmatory is claimed):**
- **G1, damaged loop detected.** gain(H) is below gain(SS), one-sided p < 0.05. On P2's own loops this test gives p = 0.004 and detects the harm in about 95 percent of block resamples; that estimate is optimistic because it reuses the data that suggested the test.
- **G2, the anchors differ.** gain(SS) is above gain(QT), one-sided p < 0.05, and fewer than 5 percent of bootstrap draws have a gap at or below 0.004.

**Confirmatory claims (in this fixed order; each is tested only if the one before it holds):**
- **C1, the fixed table beats the line.** The 90 percent interval of ST's advantage lies above 0. (P2, post hoc: 0.96, interval 0.80 to 1.15.)
- **C2, planner effect replicates.** gain(ST) is above gain(QT), one-sided p < 0.05. (P2: +0.044.)
- **C3, cheap coder is non-inferior.** The 90 percent bootstrap interval of gain(ST) - gain(SS) lies above -0.010 bpb. The margin is 2.5 noise sds and about a quarter of P2's anchor gap. (P2: +0.0015, interval -0.005 to +0.009.)
- **C4, first-list rule beats the line.** The 90 percent interval of F's advantage lies above 0.

**Verdicts.** KILL if a gate fails. REPLICATED if C1, C2 and C3 hold. PARTIAL if C1 holds and C2 or C3 does not (named). NOT REPLICATED if the gates pass and C1 fails. C4 is reported as held or not held beside the verdict.

**Descriptive only (no verdict):** advantage of H (expected below 0), of R (prediction: interval contains 0) and of M; stall, slots to 2^17 and dollars per backed change as in P2; served fractions, fallbacks and harm no-ops as manipulation checks; a regression of gain on R's coin flips; the P2 and P2B estimates side by side and pooled.

**No-LLM baseline (cell N, descriptive, run after the blocks on the same node and anchor if the worker window allows; its absence does not affect the verdict).** 12 loops of random search with the same z rule and 20 slots. Each slot draws uniformly one of eight settings (TOTAL_BATCH_SIZE, EMBEDDING_LR, UNEMBEDDING_LR, MATRIX_LR, SCALAR_LR, WEIGHT_DECAY, DEPTH, WARMDOWN_RATIO) and a direction (down or up), applied to the current incumbent: times 0.5 or times 2 for the first six, minus 2 or plus 2 for DEPTH, minus 0.2 or plus 0.2 for WARMDOWN_RATIO (clipped to 0.1 and 0.9). A move already rejected on the same incumbent is redrawn. Model cost is zero. Its advantage is computed like any policy. If N recovers most of the gap, the task is too simple to rank loop designs, and that is reported as a limit.

## Budget

- **Jobs.** 84 loops x 20 slots, 16 launch sentinels, interleaved sentinels and 2 confirmation runs per loop plus anchors. P2 observed 15.8 to 19.9 GPU jobs per loop at 0.098 to 0.114 A100-hours per job.
- **GPU time.** About 185 A100-hours, about 46 hours on 4 GPUs, inside one 3-day worker job. If the job ends early, the remaining loops run in a second job with identical configs, in block order.
- **Claude.** About 60 USD at list price on the Max plan, no real money. The controller enforces a hard ceiling of 60M Claude tokens.
- **Outage rule (fixed now).** A loop with any account-level provider refusal is rerun from scratch with an identical config; the choice is made by the outage alone.

## Deviations

None yet.
