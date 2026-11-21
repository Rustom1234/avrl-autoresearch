# P2B result: the routing table replicates, the cheap coder claim misses by a hair, and random search with no model does as well

Written Nov 13 2026 from `loop/runs/P2B/analysis/report.md` (the registered analysis, `loop/core/analysis_p2b.py`,
unchanged) and from `loop/core/posthoc_p2b.py` (post hoc, descriptive). Plan: `loop/design/P2B_PREREG.md`, frozen
before the first job. Cell N (random search with no model) finished Nov 13 and is reported in its own section below.

## What was run

An automated research loop improves a 5 minute language-model training script (Karpathy's autoresearch task, scored
in bits per byte on held-out text, lower is better). Three roles call a model: the **planner** writes and refreshes
the list of ideas, the **coder** turns one idea into an edit, the **repairer** fixes crashes. A **routing table**
says which model serves which role. 84 loops ran on hannah n7, 12 per cell, 20 GPU slots per loop, in 12 randomized
blocks.

| Cell | Planner | Coder and repairer | Role in the study |
|---|---|---|---|
| SS | sonnet | sonnet | "all strong" anchor |
| QT | qwen 4B | cheap table (qwen 4B, gemma 12B, haiku by edit class) | "all cheap" anchor |
| ST | sonnet | cheap table | **the fixed routing table** |
| F | sonnet for the first idea list only, then qwen 4B | cheap table | first-list rule |
| M | haiku | cheap table | middle-rung planner |
| R | a coin per role picks sonnet or the cheap model | | random routing tables |
| H | as SS, half of the coder's proposals replaced by a no-op | | damaged loop (validity check) |

**Outcome:** sealed gain = how much lower the final script scores than the unmodified script on a held-out shard the
loop never sees. **Cost:** model dollars per loop at list price.

**Endpoint, with ST as the worked example.** ST gains 0.0507 and costs 0.26 USD per loop. The anchors: QT gains
0.0127 for 0.15 USD, SS gains 0.0526 for 1.35 USD.

- gap recovered = (0.0507 - 0.0127) / (0.0526 - 0.0127) = 0.95: ST gets 95 percent of what the strong loop adds.
- extra cost = (0.26 - 0.15) / (1.35 - 0.15) = 0.09: it pays 9 percent of what the strong loop adds in cost.
- advantage = 0.95 - 0.09 = 0.87. Zero means "no better than mixing the two anchors at random"; 1 means "the strong
  loop's gain at the cheap loop's price".

## Verdict (registered rules): PARTIAL, C3 not held

| Check | Rule | Result | |
|---|---|---|---|
| Gate G1 | damaged loop gains less than SS, p < 0.05 | p = 0.035 | held |
| Gate G2 | SS gains more than QT, p < 0.05 | p = 0.0005, no bootstrap draw without a gap | held |
| C1 | interval of ST's advantage above 0 | 0.87 (0.67 to 1.15) | **held** |
| C2 | ST gains more than QT, p < 0.05 | p = 0.0005 | **held** |
| C3 | interval of gain(ST) minus gain(SS) above -0.010 | -0.0019 (-0.0109 to +0.0081) | **not held**, by 0.0009 |
| C4 | interval of F's advantage above 0 | 0.98 (0.69 to 1.32) | not tested (the order stops at C3) |

All 12 blocks are complete. Intervals are 90 percent block-bootstrap intervals.

What can be said, and what cannot:

- **The fixed table beats the straight line, twice.** P2 gave 0.96 (0.80 to 1.16) post hoc; P2B gives 0.87 on fresh
  seeds with the endpoint fixed in advance. Pooled: 0.91 (0.78 to 1.08).
- **The planner needs the strong model** (C2, replicated).
- **"The cheap coder is as good as the strong coder" is not established.** The measured difference is small
  (-0.0019, against a gap of 0.040 between the anchors), but the interval reaches -0.0109 and the margin was -0.010. A loss of up to 0.011 bpb from the cheap
  coder cannot be excluded with 12 blocks. This is reported as a miss. It is not rounded into a pass.
- **F is descriptive only.** Its advantage (0.98) is the highest in the study, but the ordered testing stopped
  before it.

## All policies

| Policy | Sealed gain | Model USD per loop | Gap recovered | Extra cost | Advantage |
|---|---|---|---|---|---|
| SS (anchor) | 0.0526 | 1.35 | 1 | 1 | 0 |
| ST | 0.0507 | 0.26 | 0.95 | 0.09 | 0.87 (0.67 to 1.15) |
| F | 0.0532 | 0.20 | 1.02 | 0.04 | 0.98 (0.69 to 1.32) |
| R | 0.0406 | 0.93 | 0.70 | 0.64 | 0.06 (-0.29 to 0.41) |
| M | 0.0211 | 0.18 | 0.21 | 0.02 | 0.19 (-0.12 to 0.41) |
| H | 0.0314 | 1.49 | 0.47 | 1.11 | -0.64 (-1.15 to -0.16) |
| QT (anchor) | 0.0127 | 0.15 | 0 | 0 | 0 |

- **R, random tables: on the line**, as predicted in the plan (interval contains 0). Spending on the strong model at
  random buys gain at exactly the anchors' exchange rate. Routing only pays when the strong model goes to the right
  role. The per-coin regression (7 coins, 12 loops) is too noisy to rank the roles.
- **H, damaged loop: below the line**, as expected. The endpoint notices a loop that wastes strong calls.
- **M, haiku as planner: a fifth of the gap.** This contradicts the offline ladder, which had qualified haiku
  (explained below).

## Post hoc: where the gain comes from (descriptive, not in the plan)

`.venv/bin/python -m loop.core.posthoc_p2b --run loop/runs/P2B`

**1. Two settings of the final script explain 94 percent of the variance of gain across 83 loops.** The cell a loop
belongs to explains 42 percent. (R_04's final batch size is not readable from its script; it is left out here.)

| Final total batch | Depth below 8 | Loops | Mean sealed gain |
|---|---|---|---|
| 2^19 (start) | no | 18 | 0.003 |
| 2^19 | yes | 5 | 0.020 |
| 2^18 | no | 14 | 0.027 |
| 2^18 | yes | 23 | 0.053 |
| 327,680 | no | 1 | 0.024 |
| 2^17 | yes | 8 | 0.058 |
| 2^17 | no | 14 | 0.063 |

So a loop's result is almost entirely "did it cut the batch size, once or twice, and did it make the model
shallower". Everything else the 1,680 slots tried moved the score by noise.

**2. The planner effect is whether those ideas are on the list, and where.** Each entry: loops whose list holds the
idea; median position on the list; loops that tried it; loops that kept it. Ideas are matched by keyword, so counts
are approximate.

| Cell | batch to 2^18 | batch to 2^17 | shallower model |
|---|---|---|---|
| SS | 12; 1; 8; 7 | 12; 2; 7; 6 | 12; 14; 11; 7 |
| ST | 12; 1; 11; 8 | 12; 2; 6; 2 | 12; 14.5; 12; 10 |
| F | 12; 1; 10; 4 | 12; 2; 9; 8 | 12; 15; 12; 6 |
| R | 10; 1; 8; 6 | 7; 2; 5; 2 | 10; 15; 9; 5 |
| H | 12; 1; 11; 5 | 12; 2; 9; 3 | 12; 14; 11; 3 |
| M | 12; 13.5; 10; 7 | 1; 48; 0; 0 | 7; 17; 4; 3 |
| QT | 5; 10; 1; 1 | 0; -; 0; 0 | 4; 10; 1; 0 |

Sonnet writes "batch to 2^18" and "batch to 2^17" as ideas 1 and 2 of every first list: 48 of 48 loops in SS, ST, F
and H. The lists are not copies (all 12 lists of a cell differ further down), they just all open with the same two
moves. Haiku writes one halving, at position 11 to 23 behind a dozen optimizer ideas, and a second cut in 1 of 12
lists (position 48, never tried). Qwen usually writes neither. In R the planner is sonnet only when its coin says
so, which is why its counts sit in between.

**3. Why M failed although haiku passed the ladder.** The ladder rule asked whether a first list contains a
batch-size cut. Haiku's lists do, 12 of 12, in the ladder and in the live loops. But the gain needs more than one
halving: 2^18 alone is worth 0.027, while 2^17, or 2^18 with a shallower model, is worth 0.053 to 0.063. Haiku's
lists stop at the first halving, 7 of 12 M loops took it, none went further on batch size and 3 went shallower.
The ladder measured presence of one idea; the outcome depends on depth and order of the list.

The ladder's own lists, written before the run, already showed this (same keyword match, scored after the fact):

| Model | Lists of 12 with a 2^18 cut | with a 2^17 cut | with both among the first 5 ideas | Median position of the 2^18 cut |
|---|---|---|---|---|
| qwen 4B | 2 | 0 | 0 | 10 |
| qwen 9B | 5 | 2 | 0 | 9 |
| gemma 12B | 4 | 0 | 0 | 16.5 |
| haiku | 12 | 0 | 0 | 13.5 |
| sonnet | 12 | 12 | 12 | 1 |

Under the rule "both cuts among the first 5 ideas" no model but sonnet qualifies and there would have been no M
cell. The lesson for an offline planner screen: score the top of the list against every known good move, not the
whole list against one. (This rule was written after the outcome, so it is a proposal for the next screen, not a
test.)

**4. Why F matches SS.** F uses sonnet for one call per loop (the first list, about 0.05 USD) and qwen for every
later planner call. Its lists carry the same first two ideas, so it reaches the same settings. The later planner
calls of SS and ST add nothing measurable on this task.

**5. ST against SS.** ST ended at 2^17 in 2 loops, SS in 6, yet their gains differ by 0.002, because ST more often
took the other route to the same level: a shallower model (11 of 12 loops, against 7 for SS), mostly on top of 2^18.
After 2^18 plus a shallower model, the second cut is a small step (z of 0.5 to 2.2) and often fails the accept rule;
a direct jump from 2^19 to 2^17 scores z near 11 and passes easily. So the two routes end within 0.01 of each other,
and which one a loop takes depends on the order of tries.

## Cell N: random search with no language model (descriptive, in the plan)

12 loops with the same accept rule and the same 20 slots, but no planner and no coder. Each slot draws one of eight
settings of the script at random (total batch size, four learning rates, weight decay, depth, warmdown ratio) and a
direction (down or up), and tries that one step. The eight settings were fixed in the plan before the run.

| | Sealed gain | Gap recovered | Advantage (model dollars) | Advantage with GPU dollars |
|---|---|---|---|---|
| N, random search | 0.0485 | 0.90 (0.71 to 1.14) | 1.02 (0.84 to 1.27) | 0.55 (0.36 to 0.81) |
| ST, routing table | 0.0507 | 0.95 (0.75 to 1.24) | 0.87 (0.67 to 1.15) | 0.67 (0.49 to 0.91) |
| SS, all strong | 0.0526 | 1 | 0 | 0 |

- N gains far more than the all-cheap loop (p = 0.0002) and is not distinguishable from the all-strong loop
  (gain(SS) above gain(N): one-sided p = 0.23).
- Its advantage in model dollars is above 1 because it pays for no planner and no coder at all, so its extra cost
  is negative.
- Where N's loops ended, with the same pattern as the model-driven loops: 3 loops at batch 2^17 (mean gain 0.064),
  5 at 2^18 with depth 6 (0.052), 4 at 2^19 with depth 6 (0.032).

**The plan's sentence applies: N recovers most of the gap, so this task is too simple to rank loop designs.** A loop
that tries one random step on eight named settings finds the two moves about as often as Sonnet's plan does.

What that does and does not take away:

- It does not undo the registered comparisons. Among loops that use models, strong planner plus cheap coder beats
  the line, and a weak planner loses most of the gain.
- It does change what they mean. The weak planners (Qwen, Haiku) do worse than random search over eight settings:
  0.013 and 0.021 against 0.049. So on this task the cheap planner is not "a weaker researcher", it is a worse
  proposal source than a short fixed menu, and the strong planner's value is that it rebuilds that menu from its
  own knowledge.
- The menu is not free of knowledge either: a person who knew the script chose the eight settings, and two of them
  are the two that matter. N shows the knowledge fits in a one-line menu. It does not show that no knowledge is
  needed.
- N ran in its own session after the blocks (not inside them) and is scored against that session's own anchor runs.

## What this means for the thesis

1. **The routing claim holds on this task and replicates:** strong planner plus cheap coder gets about 95 percent
   of the strong loop's gain for about 9 percent of its extra model cost.
2. **A cheaper rule does as well, descriptively:** pay for the strong model once, for the first idea list.
3. **Random routing earns nothing over the line**, so the result is about which role gets the strong model, not about
   mixing models.
4. **The honest limit:** on this task "planning" is close to "knowing two moves". The strong planner's value is that
   its first list starts with them. That is why one strong call is enough, and why the middle-rung planner fails.
   Cell N confirms it: random search over eight named settings recovers 0.90 of the gap with no model, so the task
   is too simple to rank loop designs, and that is reported as the limit the plan named.
5. A routing study that should say something about planning beyond one list needs a task where the good moves are
   not known in advance to the strong model.

## Limits

One task, one family of models, 12 blocks. Costs are list prices, not money spent. C3 missed its margin, so the
coder claim rests on "no detectable difference". Sections under "Post hoc" were chosen after seeing the data.

## Run notes

- Main run: Nov 10 20:00 to Nov 12 17:00 EST, 84 loops, 1,680 slots, 37 sentinels, one loop restart (ST_05, resumed
  by the controller as designed). No outage reruns.
- The plan's Deviations section still says "None yet" and its hash is frozen. Deviations to add by hand when the
  study closes: cell N runs in a second whole-node job (see `P2B_CELL_N_NOTES.md`).
- Cell N: Slurm job 82993, Nov 13 00:28 to 14:22 EST. The 12 loops finished at 07:35. spruce (which hosts the
  controller) rebooted at 08:31 during the confirmation block, after 24 of its 32 jobs. The block was resumed at
  14:07 on the same worker job with the same command; recorded jobs were skipped and the 8 missing ones were run
  (32 of 32 ok). The node sat idle in between. Analysis:
  `.venv/bin/python -m loop.core.analysis_p2b --run loop/runs/P2B --pool-csv loop/runs/P2F/analysis/loops.csv --n-run loop/runs/P2B_N`
