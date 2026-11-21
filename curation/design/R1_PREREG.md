# R1 preregistration: a router learned from the C1 loops, tested live on a new pool

Status: FROZEN (see the last line). Changes after the freeze go in the Deviations section only.

## Why this study

C1 (`curation/design/C1_PREREG.md`, result in `curation/RESULT_C1.md`) ran 48 curation loops under 11 routing policies and stored every routed loop slot by slot. Its own plan said what comes next: train a router on those loops and test it live on a pool it has not seen, against the two anchors and the best simple rule. R1 is that test.

**What R1 can and cannot show.** C1's verdict was SHALLOW: its validity check failed (a loop that throws away half its coder files was not clearly worse than a normal one), so 12 slots are more than this task needs. C1's rules say what follows from that: the router experiment is a **demonstration of the pipeline** (ledger to trained router to live loop to the same endpoint), not a confirmed claim about routing inside research loops. R1 has no damaged cell of its own, so it inherits that label whatever its numbers are. Loop-level routing claims rest on P2B.

**What C1 found, for reference.** Strong planner with cheap coder (SC) reached the all-strong gain in 4 of 4 loops at a quarter of the cost: gap recovered 1.01, extra cost 0.24, advantage 0.77 (0.61 to 0.92). So the best simple rule from C1 is SC.

## Task, loop and models

Unchanged from C1 and run by the same frozen files (hashes in the last line): 12 slots; the planner proposes, the coder writes (one repair call); one training run on seed 11; the z rule on dev bpb with sigma 0.007266 (reject below 0.75, accept at 2.0, in between rerun both on seed 12 and accept at 1.5); final file scored on the sealed set with seeds 1001 and 1002. Strong = Fable, cheap = Gemma 12B served locally. The launcher is C1's (`launch_c1.py run`), reading `curation/runs/R1/plan.json`.

## Test pool: v2b

- **What it is.** The v2 recipe drawn again with seed 1 (v2 is seed 0), built by `curation/task/build_pool_v2b.py` into `data/curation/pool_v2b`: 20,734 documents, 2.003 T tokens, same shares by kind as v2. 242 documents (1.2 percent) also occur in v2. Dev and sealed text are different from v2's.
- **Why not v3 (a deviation from C1's "What follows" line, which names v3).** On pool v3 the raw file on seed 11 scores dev 1.9192 against a mean of about 1.973 over the other seeds, 7 noise sds off. Every loop scores the raw file on seed 11 as its start value, so on v3 the start value, the accept rule and the router's "gain so far" would all be wrong by that amount. On v2b the raw file on seed 11 scores 1.9405, inside the range of the other seeds.
- **Baselines on v2b (run before the freeze, no loop involved).** Raw file, 10 runs (seeds 0 to 7, 11, 12): dev mean 1.9519, sealed mean 1.9877. Label oracle (`oracle_wiki`, seeds 0 and 1): dev 1.8030, sealed 1.8438, a sealed gain of about 0.13 to 0.14 bpb, the same headroom as v2 (0.129). Hand-written heuristic (seeds 0 and 1): dev 1.9291, sealed 1.9660.
- **Noise, stated plainly.** The 10 raw runs have a sealed sd of 0.0231, three times v2's 0.0072. One run causes most of it (seed 2, dev 2.0102); without it the sd is 0.0120. This is a property of the trainer, not of v2b: a given file on a given seed is reproducible, but now and then a seed lands 0.03 to 0.07 bpb away from the others, in either direction. It was seen on all three pools (v2: the raw file on seed 12 is 0.034 worse than on seed 11 in all 10 C1 loops that ran both; v3: seed 11; v2b: seed 2), and in C1's final scores (in 2 of 48 loops the two confirmation seeds differ by 0.05 and 0.07). Consequences, fixed here before the run:
  - sigma stays 0.007266. It is part of the method being carried over, the router's state is counted in it, and re-fitting it on the test pool would be tuning on the test pool. The accept rule may therefore let a lucky run through more often than intended; this hits every cell the same way.
  - G2's threshold is computed by the frozen analysis from the 10 raw runs, so it is 2 x 0.0231 = 0.0462 bpb, stricter than C1's 0.0144.
  - a four-seed sensitivity check is added (below).

## The router

**Training data.** The 36 undamaged model-routed C1 loops whose two models are Fable and Gemma (SS, CC, SC, CS, F1, E3, UA: 4 each; RR: 8), 432 slots. HH, N and DH are left out. All of it is on pool v2.

**Trainer.** `curation/loop/train_router2.py`. The state a router sees before a slot is the gain so far, cut into levels, and the slots left. C1 loops end at one of a few gain levels and the final gain depends on the level, hardly on the path. So the trainer (1) values each level by the mean final dev gain of the loops that ended there, as a share of the anchor gap, (2) charges each slot its dollars as a share of the anchor dollar gap, (3) estimates from the records, for each level and each of the four model pairs, the chance that a slot moves the loop to a higher level, and (4) works backwards from the last slot to the best pair for every level and number of slots left. The reward is the one C1's plan named: gain as a share of the anchor gap minus dollars as a share of the anchor cost gap. Built in: at the top level the model sees nothing more to gain, so the table goes all cheap there.

**How the trainer and its two settings were chosen. All of it used C1 data only, none of it v2b. All of it was done after looking at C1's results, so it is a search over router designs and is disclosed as one.**
1. The first trainer (`train_router.py`, per-slot rewards) was dropped because it failed a held-out-cell check: fitted without the loops of a cell, it ranked the all-strong policy first and strong planner with cheap coder last, the reverse of what the loops show.
2. Setting one, pooling. The state also contains "slots since the last accept". A state can use its own records or the records of its whole gain level. With own records (at least 4) the table said: strong planner in slot 1, all cheap in slots 2 and 3, strong planner again in slot 4. The SC ledgers show that is an artifact: Fable's first plans fail in Gemma's hands, Fable then rewrites the plan to be simpler, and that lands in slot 3 or 4. The slots without an accept are the planner adapting, not a sign to stop paying. I chose level-only pooling, which cannot produce that pattern and estimates a third as many numbers.
3. Setting two, the gain levels. C1's state had three (cuts at 2 and 9 noise sds). With those, both tables went all cheap at 9 sds. The C1 trajectories show that is too early: of the 7 loops that first passed 9 sds below 14, 4 later rose to 16 to 19 sds (SS_03, SS_04, SC_02, SC_04), every one of them with a strong planner still in place. A table that stops at 9 would have cut those loops off at about half their final gain. I added one cut at 14 sds so that the model can see this. `train_router.decide` reads the cuts from the table (a small change to `train_router.py`; old tables behave as before).
4. The held-out-cell check does not settle either choice: mean error 0.21 to 0.24 across pooling settings with three levels, 0.23 to 0.26 with four (`curation/runs/R1/router_check.txt`). The check is weak in absolute terms too: errors of about 0.2 on values that range from 0.23 to 0.94, and the all-strong policy is overvalued when held out (about 0.6 against 0.23). The two settings rest on the reading of the ledgers and trajectories above, not on the check.
5. The three designs gave three different tables (below, and the two discarded ones in items 2 and 3). The learned table is **not stable** to reasonable modelling choices with 36 loops. That is a finding of its own and goes in the write-up whatever R1 shows.

**The two tables (frozen files, hashes below). Neither depends on the slots left.**
- **LA, learned from all 36 loops** (`curation/runs/R1/router_la.json`): all strong while the gain is under 9 sds; strong planner with cheap coder from 9 to 14; all cheap from 14 sds on.
- **LB, learned from the 8 random-routing loops only** (`curation/runs/R1/router_lb.json`): all strong under 2 sds; strong planner with cheap coder from 2 to 9; all strong from 9 to 14; all cheap from 14 on. C1's plan said "the RR loops first"; LB is that router. RR loops are the only ones whose model choices were not set by a rule, which is the data a router would have in practice. It is thin: at the 9 to 14 level three of the four model pairs had fewer than 4 records and were estimated from all levels together.

In words, both tables are "pay for strong models until the loop is at the top, then stop", a stricter version of C1's UA cell (which stopped at the first accepted change and reached the strong level in 2 of 4 loops). Neither is SC.

## Cells

Four blocks, five loops each, 20 loops. Loop n of a cell uses seed n.

| Cell | Policy | What it is |
|---|---|---|
| SS | strong plans and codes, every slot | strong anchor |
| CC | cheap plans and codes, every slot | cheap anchor |
| SC | strong plans, cheap codes | best simple rule from C1 |
| LA | table learned from all C1 loops | learned router |
| LB | table learned from the random-routing loops | learned router, random data only |

The loop list and its order (shuffled within each block, seed 20261111) are in `curation/runs/R1/plan.json`. Loops run on spruce, willow and cherry, three at a time per machine, all model calls from spruce, as in C1.

## Outcome and endpoint

As in C1, computed by C1's frozen analysis through `curation/loop/analysis_r1.py`:
- sealed gain of a loop = mean sealed bpb of the raw pool (1.9877, the 10 runs above) minus the mean sealed bpb of the loop's final file on seeds 1001 and 1002;
- dollars of a loop = model dollars from its call log, same prices as C1;
- gap recovered = (gain P - gain CC) / (gain SS - gain CC); extra cost = the same ratio on dollars; **advantage = gap recovered - extra cost**;
- intervals: bootstrap over loops within cells, anchors resampled too, 10,000 resamples, seed 20261110, 95 percent. With 4 loops per cell they are coarse and somewhat too narrow.

## Decision rules

1. **G2, the anchors differ on the new pool.** gain(SS) - gain(CC) at least 0.0462 bpb and one-sided exact permutation p at or below 0.05. If G2 fails: verdict **NO GAP**, no advantage is computed, cells are only described.
2. **R1.** LA has advantage above 0 (lower end of its interval above 0).
3. **R2.** LB has advantage above 0 (lower end of its interval above 0).
4. **R3.** LA beats the best simple rule: advantage(LA) - advantage(SC) has the lower end of its interval above 0.
5. Descriptive, no pass or fail: LB against SC and LA against LB (same difference with interval); loops per cell that reach the strong level (sealed gain at least 85 percent of the SS cell mean); what the tables did in each loop (slots with a strong planner, slots with a strong coder, the first slot that started at 14 sds or more); the budget curve.

Reading, fixed now: R1 and R2 holding means "a router trained offline on the ledger of another pool beats mixing whole strong and cheap loops on a new pool". R3 holding would mean the learned router adds something over the simple rule; R3 not holding with LA close to SC means the router relearned the simple rule plus a stopping point, which is the expected outcome on a shallow task. All of it is under the "demonstration" label above.

**Sensitivity check (because of the noise note).** After the run, every loop's final file is also scored on seeds 1003 and 1004 (`curation/loop/rescore_finals.py`), and the table, R1, R2 and R3 are recomputed with four-seed means (`analysis_r1.py --rescored`). The two-seed result is the main one. If a rule holds on two seeds and not on four, or the reverse, the write-up says so and calls that rule unsettled.

## Predictions, written before any R1 loop (Nov 11 2026)

- G2 passes (gap about 0.10 bpb against the 0.046 threshold).
- SC repeats: at the strong level in at least 3 of 4 loops, about 2.5 USD per loop, advantage about 0.75.
- LA: at the strong level in at least 3 of 4 loops; 1 to 4.5 USD per loop depending on how fast the loop reaches 9 sds (about 0.9 USD per all-strong slot), mean about 2.5; advantage about 0.7. R1 holds.
- LB: at the strong level in at least 3 of 4 loops, 1 to 4 USD per loop, advantage about 0.7. R2 holds.
- R3 does not hold: LA and SC come out about equal (difference near 0, interval includes 0). On a task where one good strong-planner step makes most of the gain, a learned stopping rule has little room to beat "strong planner throughout".
- Known ways this can go wrong: (a) the cuts (2, 9, 14 sds) come from v2; if a good file on v2b sits just under 14 sds, the tables keep paying to the end, which costs dollars but not gain; (b) a lucky seed-11 run can push a loop over 14 sds with a mediocre file, and both tables then stop paying, which costs gain (the trial loop below had a first accept at 6.3 sds on seed 11 whose dev score on the two confirmation seeds was 3.4 sds); (c) the noise note: one confirmation seed 0.05 off moves a loop's gain by a quarter of the anchor gap, and with 4 loops per cell that moves a cell.

## Outage and exclusion rules

As in C1. A provider or server failure is waited out (up to 30 tries, 2 minutes apart) and does not use up the slot; if a slot is still lost, the loop is left out and run again from scratch with the same configuration, and the lost run is kept and listed. A loop stopped by a machine problem is resumed with `--resume`; every resume is listed. No loop is dropped for its result. No cell, block or slot count changes after the freeze.

Before results are used (`check_c1.py --root curation/runs/R1 --cells SS LA LB`): every call must have been answered by the model the policy asked for; every accepted file of an SS, LA or LB loop is read by hand at the flagged lines, and a file that reads the hidden labels voids its loop.

## Budget

About 75 USD at list price (SS about 41, SC about 10, LA about 10, LB about 10, CC under 1), charged to the subscription and not billed. About 4 hours on three A4000s. The rescoring adds 40 training runs, about half an hour.

## What was looked at before the freeze

- The baselines on v2b listed above (raw, label oracle, heuristic). No loop.
- One trial loop of one slot on v2b, to check that the frozen loop reads a table from the new trainer and runs on the new pool: seed 99 (not used in R1), run in a scratch folder, not part of R1, with an earlier three-level table whose first slot is strong planner with cheap coder. The slot was accepted with dev 1.8951 against a start of 1.9407 (6.3 sds); on the two confirmation seeds the same file scored dev 1.9163 (3.4 sds) and sealed 1.9537. That is one more look at v2b than the baselines, and it is disclosed for that reason. The third cut was added after this trial but for the reason given above (C1 trajectories), not because of it; the changed decision code is covered by a unit test (`test_router2.py`), not by a second live trial.
- Post hoc on C1, not on v2b: the final files of C1 loops CC_04 and DH_02 were scored on two more seeds because their two confirmation seeds disagreed. Result in `curation/RESULT_C1.md`; it does not change C1's verdict.

## Deviations

1. **Nov 11 2026, 11:01 EST, during the run: memory cap on candidate code.** `curation/task/curate_runner.py` (frozen hash 086d7d09821c) was replaced by a version that sets a 16 GB address-space limit before a candidate's `select()` runs (new hash 676dad5cfa8b; two added lines, nothing else changed). Reason: the department's system administrator reported that nine of my processes had been killed by the kernel for running willow, cherry and spruce out of memory (62 GB machines) between Nov 10 22:05 and Nov 11 07:53. All of them were model-written candidate files in the pilot and in C1 (C0 F1_01 slot 7; C1 RR_05 slot 9, SC_03 slot 3, F1_04 slot 6, UA_04 slot 3), most of them calling `np.dot` on a sparse matrix, which makes numpy copy the matrix until memory runs out. The runner had a time limit but no memory limit. This is a shared-machine safety problem, so the change was made at once and not held until the end of the run.
   - What changes for a loop: a candidate that needs more than 16 GB now fails in seconds with a MemoryError that the repair call can read. Before, it was killed by the kernel after about a minute and the loop saw "died with SIGKILL". In both cases the candidate fails; the error text differs.
   - Check that the cap does not touch working candidates: all 48 final files of C1 run under the capped runner (largest peak address space 5.9 GB, largest resident memory 2.1 GB). Four of the five killed candidates now stop with a MemoryError within 13 seconds; the fifth finishes under the cap.
   - State of R1 at the change: 81 slots done across the first 9 loops (CC_01 12, CC_02 11, LA_01 9, LB_01 8, LB_02 11, SC_01 7, SC_02 8, SS_01 6, SS_02 9), no loop finished, and no candidate in R1 had been killed for memory, so no R1 slot so far would have gone differently. The loops were not stopped or restarted: the runner is started fresh for every training run, so the cap applies to every run from 11:01 on, in every cell alike.
   - C1 is not rerun. In C1 each of those kills was scored as a failed candidate, which is what the cap would also have produced.

2. **Nov 11 2026, 11:20 EST, during the run: machine-level memory guard.** The administrator's limit is 50 GB per machine for all of my processes together. A new script, `curation/task/mem_guard.py` (hash d1ebb4d5e7c6), runs on spruce, willow and cherry. Every second it adds up the memory of all my processes on the machine; above 40 GB (or when the machine has under 6 GB available and a candidate holds more than 2 GB) it kills the running candidate script and its child processes, nothing else. It covers the one case the 16 GB cap does not: a candidate that starts several processes. No frozen file is changed. A killed candidate is scored as a failed candidate, as before; every kill is logged in `curation/runs/mem_guard_<host>.log` and will be listed in the write-up.

Frozen: 2026-11-11 09:44 EST, before any R1 loop was started. First 12 hex digits of each file's SHA-256. Unchanged from C1: `curation/loop/engine.py` db6d551665f5; `curation/loop/routed.py` 20d7ee9b3e47; `curation/loop/menu.py` 4dc7778a1d0e; `curation/loop/launch_c1.py` 97cced4ce832; `curation/loop/analysis_c1.py` 661a659e7f6c; `curation/loop/table.py` 05f722518e85; `curation/task/run.py` eca4e260e2c3; `curation/task/common.py` 118f17466a9d; `curation/task/curate_runner.py` 086d7d09821c; `curation/task/curate.py` 4e4223e181fc; New or changed for R1: `curation/loop/train_router.py` 68d0f4f8b9eb; `curation/loop/train_router2.py` 2adb97e41786; `curation/loop/analysis_r1.py` e1036a3aae23; `curation/loop/rescore_finals.py` 16bbeabce455; `curation/loop/check_c1.py` b3908be4dfbf; `curation/task/run_recipes.py` 083b70237fc4; `curation/task/build_pool_v2b.py` 269b7c785467; `curation/runs/R1/plan.json` 23f7c0c5062c; `curation/runs/R1/router_la.json` 1325a5935b2a; `curation/runs/R1/router_lb.json` d6e89ec0843f; Pool files under `data/curation/pool_v2b/`: `dev_tokens.npy` 0d04d50dd013; `docs.json` 481baff9c9b6; `labels_HIDDEN.json` b2e3af5f3540; `pool_offsets.npy` 35f8e2baf2d9; `pool_tokens.npy` 8ee3a7daffd8; `sealed_tokens.npy` a69e5a1055b7; `summary.json` 01079a4fe308; `target_sample.json` c038f0ce34f1.
