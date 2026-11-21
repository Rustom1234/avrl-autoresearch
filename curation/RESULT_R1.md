# R1 result: a router learned from the C1 loops, tested live on a new pool (Nov 11 2026)

Plan, tables, cells, endpoint and decision rules: `curation/design/R1_PREREG.md`, frozen Nov 11 09:44 EST before any R1 loop ran. This file reads the result against that plan. Numbers come from the frozen script (`curation/loop/analysis_r1.py`, output in `curation/runs/R1/analysis.json`).

**Label, as fixed in the plan: demonstration.** C1's verdict was SHALLOW (12 slots are more than this task needs), so nothing here is a claim about routing inside research loops in general. It shows what a table learned offline from one pool's records does on another pool.

## Verdict

| Rule | Result on the two preregistered seeds | On four seeds |
|---|---|---|
| G2, anchors differ on the new pool | SS gain 0.1213, CC gain 0.0182; gap 0.1031 bpb against a threshold of 0.0462; permutation p = 0.014 (smallest possible). **PASS** | gap 0.0958, p = 0.029. Pass |
| R1, LA advantage above 0 | 0.76 (0.31, 1.86). **HOLDS** | 0.80 (0.33, 2.21). Holds |
| R2, LB advantage above 0 | 0.96 (0.64, 2.04). **HOLDS** | 0.98 (0.64, 2.35). Holds |
| R3, LA beats the simple rule SC | LA minus SC = +0.14 (-0.38, +0.74). **DOES NOT HOLD** | +0.14 (-0.39, +0.77). Does not hold |

The two-seed and four-seed results agree on every rule, so no rule is unsettled.

In the plan's words: R1 and R2 holding means "a router trained offline on the ledger of another pool beats mixing whole strong and cheap loops on a new pool". R3 not holding with LA close to SC means "the router relearned the simple rule plus a stopping point, which is the expected outcome on a shallow task".

## The run

- 20 loops (4 blocks of 5), all finished with exit 0. None left out, none resumed, no provider or server wait used up a slot.
- Started Nov 11 09:44 EST, finished 14:20 EST: 4 hours 36 minutes on spruce, willow and cherry. The rescoring on seeds 1003 and 1004 (40 training runs) took 31 more minutes.
- Model spend at list price: 58.28 USD (SS 33.56, SC 10.75, LA 9.88, LB 4.04, CC 0.04). The budget was about 75. Charged to the subscription, not billed.

## Result table

Gain = sealed bpb of the raw pool (1.9877) minus sealed bpb of the loop's final file (mean of the two confirmation seeds). Dollars = model dollars per loop. Means over 4 loops.

| Cell | Policy | Gain | sd | USD | Accepts | Gap recovered | Extra cost | Advantage (95 percent) | Loops at the strong level |
|---|---|---|---|---|---|---|---|---|---|
| SS | Fable plans and codes | 0.1213 | 0.0504 | 8.39 | 2.2 | (anchor) | | | 3 of 4 |
| LB | table learned from the 8 random-routing loops | 0.1292 | 0.0168 | 1.01 | 2.0 | 1.08 | 0.12 | 0.96 (0.64, 2.04) | 4 of 4 |
| LA | table learned from all 36 loops | 0.1271 | 0.0178 | 2.47 | 2.2 | 1.06 | 0.29 | 0.76 (0.31, 1.86) | 3 of 4 |
| SC | Fable plans, Gemma codes | 0.1153 | 0.0474 | 2.69 | 2.2 | 0.94 | 0.32 | 0.62 (0.16, 1.62) | 3 of 4 |
| CC | Gemma plans and codes | 0.0182 | 0.0147 | 0.01 | 0.8 | (anchor) | | | 0 of 4 |

"Strong level" is the plan's cut: a sealed gain of at least 85 percent of the SS cell mean (0.1031 on two seeds, 0.1003 on four). On four seeds LA is at 4 of 4, because LA_01 sits at 0.1022 on two seeds and 0.1045 on four; the other cells do not change.

How to read one row, LB. The all-strong loop gains 0.1213 and the all-cheap loop 0.0182, so the gap to recover is 0.1031. LB gains 0.1292, which is 0.1110 above the cheap loop: 0.1110 / 0.1031 = 1.08, the whole gap and a little more. LB costs 1.01 USD against 0.01 and 8.39 for the anchors: (1.01 - 0.01) / (8.39 - 0.01) = 0.12, an eighth of the extra cost. Advantage = 1.08 - 0.12 = 0.96. A value of 0 would mean "no better than running some all-strong loops and some all-cheap loops"; 1 would mean "the strong loop's gain at the cheap loop's price".

Differences between cells (advantage, 95 percent interval): LA minus SC +0.14 (-0.38, +0.74); LB minus SC +0.34 (-0.03, +0.88); LA minus LB -0.20 (-0.63, +0.16). None excludes 0. Only LA minus SC was a named rule.

The intervals are a bootstrap over 4 loops per cell and are coarse. Their upper ends are above 1 because the SS cell has one loop that stayed low (SS_04, below); a resample that draws it more than once shrinks the anchor gap and inflates every ratio. Read the lower ends, which is what the rules use.

## Per loop

| Loop | Gain, two seeds | Gain, four seeds | USD | Accepted changes (slot: dev gain in noise sds) | Slots with strong planner / strong coder |
|---|---|---|---|---|---|
| SS_01 | 0.1520 | 0.1494 | 9.37 | 1: 17.5, 12: 20.4 | 12 / 12 |
| SS_02 | 0.1500 | 0.1474 | 7.15 | 1: 7.9, 2: 16.7, 6: 19.1 | 12 / 12 |
| SS_03 | 0.1370 | 0.1355 | 8.50 | 1: 16.3, 6: 18.0 | 12 / 12 |
| SS_04 | 0.0463 | 0.0400 | 8.55 | 1: 0.9, 2: 4.0 | 12 / 12 |
| LA_01 | 0.1022 | 0.1045 | 5.44 | 1: 0.9, 2: 3.3, 8: 14.5 | 8 / 8 |
| LA_02 | 0.1348 | 0.1357 | 2.40 | 2: 4.5, 4: 13.5, 6: 17.3 | 6 / 4 |
| LA_03 | 0.1278 | 0.1258 | 0.95 | 2: 16.7 | 2 / 2 |
| LA_04 | 0.1436 | 0.1423 | 1.10 | 1: 2.8, 2: 17.4 | 2 / 2 |
| LB_01 | 0.1432 | 0.1403 | 0.47 | 1: 3.8, 2: 17.3 | 2 / 1 |
| LB_02 | 0.1352 | 0.1309 | 0.52 | 1: 17.7 | 1 / 1 |
| LB_03 | 0.1049 | 0.1049 | 0.53 | 1: 14.1 | 1 / 1 |
| LB_04 | 0.1337 | 0.1329 | 2.53 | 1: 2.2, 3: 5.6, 4: 13.6, 7: 16.8 | 7 / 4 |
| SC_01 | 0.1419 | 0.1415 | 2.59 | 1: 1.5, 2: 13.2, 5: 17.4, 6: 18.9 | 12 / 0 |
| SC_02 | 0.0443 | 0.0499 | 2.59 | 6: 3.9, 10: 6.0 | 12 / 0 |
| SC_03 | 0.1392 | 0.1394 | 2.89 | 1: 18.3 | 12 / 0 |
| SC_04 | 0.1358 | 0.1332 | 2.67 | 4: 6.8, 5: 16.5 | 12 / 0 |
| CC_01 to CC_04 | 0.0347, 0.0210, -0.0011, 0.0183 | 0.0402, 0.0222, 0.0079, 0.0188 | 0.01 each | at most one, 1.0 to 4.8 | 0 / 0 |

Seed noise was small this time: no loop moves by more than 0.01 bpb between two and four seeds.

## What the tables did

Both tables did exactly one thing: **pay for strong models until the loop's best file is at the top level (14 noise sds of dev gain), then switch to all cheap for the rest of the loop.**

- In all 8 learned-table loops every accepted change came from a slot with a strong planner. After the switch to all cheap, Gemma accepted nothing in any loop (0 accepts in 67 all-cheap slots).
- The dollars follow from how fast the loop reached the top. LB_02 and LB_03 got there in slot 1 and cost about 0.5 USD. LA_03, LA_04 and LB_01 got there in slot 2 (about 0.5 to 1.1 USD). LA_02 and LB_04 took 6 and 7 slots (about 2.5 USD). LA_01 took 8 all-strong slots (5.44 USD).
- The rule has no spending cap. A loop that never reaches the top pays for strong models in all 12 slots, which is the all-strong price. SS_04 is that loop: all strong, stuck at 4 sds for 12 slots, 8.55 USD. LA would have behaved the same way in its place. The saving comes from stopping after success, not from being cheaper on the way up.
- The middle levels did little. LA's "strong planner, cheap coder from 9 to 14 sds" was used in 2 slots (LA_02 slots 5 and 6). LB's "strong planner, cheap coder from 2 to 9 sds" was used in 4 slots (LB_01 slot 2, LB_04 slots 2 to 4), and 3 of those 4 slots produced an accepted change.

## Three things to keep in mind when reading the table

1. **LB being cheaper than LA is mostly first-slot luck, not a property of the tables.** Slot 1 is all strong in SS, LA and LB alike, 12 loops with the same policy. Four of them reached the top in slot 1: SS_01, SS_03, LB_02, LB_03. None of the LA loops did. That alone accounts for most of the 1.01 against 2.47 USD. The safer statement pools the two tables: 8 loops under "strong until the top, then cheap", 7 of 8 at the strong level on two seeds (8 of 8 on four), 1.74 USD per loop against 8.39 for all strong.
2. **Stopping at the top leaves a little gain behind.** The three SS loops that reached the top kept paying and kept improving (accepts at slots 6, 6 and 12); they end at 0.137 to 0.152. The eight learned-table loops end at 0.102 to 0.144, mean 0.128. That is about 0.02 bpb for roughly 6 more USD per loop. The advantage metric counts this trade as not worth it; someone who only cares about the last 0.02 would not use these tables.
3. **The top-level cut can be passed by a lucky reading.** The tables read the loop's level from the decision seeds. LA_01 and LB_03 crossed 14 sds on those (14.5 and 14.1) with files that scored about 13 sds on the confirmation seeds, and both tables stopped paying there. They are the two lowest learned-table loops (0.1022 and 0.1049). This is failure mode (b) in the plan's predictions, in mild form.

## The new pool is a little harder than the training pool

On pool v2 (C1) all 4 SS loops and all 4 SC loops reached the top. On v2b one SS loop (SS_04) and one SC loop (SC_02) stayed at 4 to 6 sds for all 12 slots, each of them with a strong planner in every slot. In both the planner kept building on a first idea that did not work. This is why the SS and SC cells have large sds here (0.05) and why SC's advantage is 0.62 and not the 0.77 of C1. With 4 loops per cell, one stuck loop in LA or LB would have changed their rows by as much.

## Budget curve

Mean dev gain of the current best after slot k:

| Cell | 1 | 2 | 4 | 8 | 12 |
|---|---|---|---|---|---|
| SS | 0.0773 | 0.0991 | 0.0991 | 0.1067 | 0.1119 |
| LB | 0.0687 | 0.0932 | 0.1139 | 0.1197 | 0.1197 |
| LA | 0.0067 | 0.0762 | 0.0925 | 0.1195 | 0.1195 |
| SC | 0.0360 | 0.0573 | 0.0697 | 0.1045 | 0.1083 |
| CC | 0.0000 | 0.0000 | 0.0000 | 0.0042 | 0.0129 |

LA and LB are flat after slot 8: by then every loop had switched to all cheap.

## Predictions against the outcome

| Prediction (written before the run) | Outcome |
|---|---|
| G2 passes, gap about 0.10 bpb | Pass, gap 0.1031. Right |
| SC repeats: strong level in at least 3 of 4, about 2.5 USD, advantage about 0.75 | 3 of 4, 2.69 USD, advantage 0.62. Right on the first two, lower on advantage because of SC_02 |
| LA: at least 3 of 4, 1 to 4.5 USD per loop, mean about 2.5, advantage about 0.7. R1 holds | 3 of 4, 0.95 to 5.44 USD, mean 2.47, advantage 0.76. R1 holds. Right (one loop above the dollar range) |
| LB: at least 3 of 4, 1 to 4 USD per loop, advantage about 0.7. R2 holds | 4 of 4, 0.47 to 2.53 USD, advantage 0.96. R2 holds. Right on the rule; cheaper than predicted, for the luck reason above |
| R3 does not hold: LA and SC about equal | +0.14 (-0.38, +0.74). Right |
| Failure mode (a): a good file just under 14 sds keeps the table paying | Seen: LA_02 and LB_04 sat at 13.5 and 13.6 sds for two and three slots before a further accept |
| Failure mode (b): a lucky reading stops the table early | Seen in mild form: LA_01, LB_03 |
| Failure mode (c): one confirmation seed far off | Not seen: four-seed gains within 0.01 of two-seed gains |

## Post-run checks

`check_c1.py --root curation/runs/R1 --cells SS LA LB`:
- Calls answered by another model than the policy asked for: 0.
- Loops that lost a slot to a provider or server failure: none.
- All 26 accepted files of the SS, LA and LB loops read at the flagged lines. The flags are `import os` (to read the seed from the environment), regular expressions, `argsort`, k-means cluster labels and progress lines printed to stderr. No file reads the hidden labels, the dev or sealed tokens, or anything outside the two inputs it is given.
- Slots where the candidate failed to run: LA 12, SC 7, LB 1, SS 0, CC 0. All but one are ordinary code errors in Gemma-written files (syntax errors, index errors, wrong attribute names); one ran past the 180 second limit. None was a memory failure.

## Deviations from the plan

Both are recorded in the plan's Deviations section with times and hashes. Neither changed a loop's outcome.

1. **Memory cap on candidate code (11:01 EST, during the run).** The department's system administrator reported that model-written candidate files in the pilot and in C1 had run willow, cherry and spruce out of memory. `curation/task/curate_runner.py` now sets a 16 GB address-space limit before a candidate's `select()` runs. At the change, 81 of the 240 slots were done (in the first 9 loops) and no R1 candidate had been killed for memory. In the 159 slots after the change, and in the 40 rescoring runs, **no candidate hit the cap** (0 MemoryError in the ledgers).
2. **Machine-level memory guard (11:20 EST).** `curation/task/mem_guard.py` ran on the three machines and would have killed a candidate if my total memory on a machine passed 40 GB. **It never acted** (0 kill lines, 0 warnings in `curation/runs/mem_guard_<host>.log`). Peak total memory while it ran: spruce 14.9 GB (this includes the P2B loop processes), cherry 2.5 GB, willow 1.9 GB.

Since neither limit was ever reached, loops before and after 11:01 ran under the same effective conditions.

## What can and cannot be said

Can be said:
- A table fitted offline to the records of 36 loops on one pool, and frozen, was run live on a second pool it had never seen and beat mixing all-strong and all-cheap loops there (R1, R2), at 12 to 29 percent of the extra cost of the all-strong loop and with the same gain.
- What it learned is a stopping rule: pay for strong models until the loop is at the top, then stop. The gain levels it was cut on (2, 9, 14 sds, set on the first pool) carried over to the second pool.
- A table fitted on only the 8 random-routing loops (LB) did as well as the one fitted on all 36.

Cannot be said:
- That the learned table beats the simple rule "strong planner, cheap coder" (R3 does not hold; LB minus SC is +0.34 with an interval that just includes 0, and it was not a named rule).
- That LB is better than LA (see first-slot luck).
- That learned routing helps on tasks where a loop needs many slots. This task is solved in one or two good steps, which is the easiest case for a stopping rule. That question stays with P2B.
- That the table is *the* table. As disclosed in the plan, the trainer and its two settings were chosen after looking at C1, and three reasonable designs gave three different tables from the same 36 loops. One of those (stop at 9 sds) would have cut loops off at about half their final gain by C1's own trajectories. The result here is for the table that was frozen, not for "a learned router" as a class.
- Anything fine-grained: 4 loops per cell, one stuck loop moves a cell by a quarter of the gap.

## Next

- P2B (the replication on the original task) is the study that can carry a loop-level routing claim. Its analysis runs when it ends (by Nov 13).
- For any further study on this task: add memory guidance to the coder prompt (stay sparse, use float32, state the memory budget), so that fewer candidates are wasted on memory errors now that the cap turns them into failures.
- A spending cap for the stopping rule (for example, stop paying after k strong slots without reaching the top) is the obvious next table to test; SS_04 and LA_01 show where it would matter.
