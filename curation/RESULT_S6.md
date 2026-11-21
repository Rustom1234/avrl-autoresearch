# S6 result: does memory of past strong runs replace the strong planner? (Nov 11 2026)

Plan, cells, rules and predictions: `curation/design/S6_PREREG.md`, written before any S6 loop ran. Numbers:
`curation/runs/S6/analysis.json` (from `curation/loop/analysis_s6.py`).

**Label, as fixed in the plan: demonstration.** The task is shallow (C1), the test pool comes from the same recipe as
the pool the memory was built on, and every cell has 4 loops.

## Verdict

| Rule | Result |
|---|---|
| T1, training helps: PW gains more than P0, one-sided permutation p at most 0.05 | PW minus P0 = +0.0366 bpb (-0.0050, +0.0832), p = 0.114. **DOES NOT HOLD** |
| T2, the trained planner replaces the strong one: at least 3 of 4 PW loops at the strong level | 1 of 4. **DOES NOT HOLD** |

In the plan's words, T1 not holding means "48 examples are not enough, or the coder is the limit". The plan also said:
if K is at the strong level, say so first. It is.

## What each form of memory did

Gain = sealed bpb of the raw pool (1.9877) minus sealed bpb of the final file, mean of the two confirmation seeds.
Strong level = gain of at least 0.1031 (85 percent of the SS mean). SS, CC and SC are the R1 loops (same pool, engine
and seeds, run earlier the same day, not concurrent).

| Cell | What it is | Gain | USD per loop | At the strong level | Gap recovered | Advantage (95 percent) |
|---|---|---|---|---|---|---|
| K | old strong final file, run unchanged, no model | 0.1336 | 0 | 7 of 8 | 1.12 | (no loop, see text) |
| SS | Fable plans and codes (R1) | 0.1213 | 8.39 | 3 of 4 | (anchor) | |
| SC | Fable plans, Gemma codes (R1) | 0.1153 | 2.69 | 3 of 4 | 0.94 | 0.62 (0.17, 1.57) |
| PM | Qwen 4B plans with 3 old ideas in the prompt | 0.0749 | 0.01 | 1 of 4 | 0.55 | 0.55 (0.21, 1.29) |
| PW | Qwen 4B with a LoRA trained on 48 old ideas | 0.0565 | 0.04 | 1 of 4 | 0.37 | 0.37 (-0.04, 1.00) |
| P0 | plain Qwen 4B plans | 0.0199 | 0.01 | 0 of 4 | 0.02 | 0.02 (-0.15, 0.16) |
| CC | Gemma plans and codes (R1) | 0.0182 | 0.01 | 0 of 4 | (anchor) | |

Loop gains: P0 0.0220, 0.0069, 0.0271, 0.0234. PM 0.0606, 0.1442, 0.0378, 0.0569. PW 0.1217, 0.0403, 0.0638, 0.0002.

How to read the K row. Eight files that strong loops ended with on the old pool were run on the new pool with no model
call at all. Their mean gain is 0.1336. A fresh all-strong loop on the new pool gained 0.1213 and cost 8.39 USD. So for
a second job of the same family, the stored file does at least as well as paying for the strong model again.

Three readings, in order of how much the data supports them:

1. **Reusing the old answer beats every routing policy on this task (K, 7 of 8 at the strong level, zero cost).** This
   is the plan's warning coming true: the test pool is a new draw from the same recipe, so the job repeats. The lesson
   for a routing table is a first row before any model is chosen: "has a strong loop already solved this job family?
   Then run its file."
2. **Old ideas in the prompt move the cheap planner about half the way (PM).** PM minus P0 = +0.0550 bpb
   (+0.0228, +0.1022), permutation p = 0.014, the smallest value 4 against 4 can give. All four PM loops are above all
   four P0 loops. No rule was set for this comparison, so it is descriptive. PM reaches the strong level in 1 of 4
   loops, so it does not replace the strong planner.
3. **Training the ideas into the weights did not beat putting them in the prompt (PW minus PM = -0.0184,
   (-0.0772, +0.0387)).** PW is above P0 on average but one PW loop gained nothing, and the rule T1 fails.

## Why the trained planner fell short (looked at after the result, not a tested claim)

The trained planner did learn to write like the strong one: its ideas have a median length of 1,951 characters
(plain Qwen: 305; Fable in R1: about 1,700). But the cheap coder could not turn them into working code:

| Planner (coder is Gemma 12B in all rows) | Slots that failed to run or scored far worse than the raw pool (dev above 2.2) |
|---|---|
| Fable (R1 cell SC) | 8 of 48 |
| plain Qwen 4B (P0) | 9 of 48 |
| Qwen 4B with memory in the prompt (PM) | 12 of 48 |
| trained Qwen 4B (PW) | 26 of 48 |

With real Fable plans of the same length the same coder breaks 8 of 48 slots. With the imitation plans it breaks 26.
So the adapter copied the form of a strong plan (long, many steps) without the part that makes it implementable. This
fits C1 and R1: the planner is the role that needs the strong model, and 48 examples do not transfer that.

## Predictions against outcomes

| Prediction | Outcome |
|---|---|
| K: at least 6 of 8 at the strong level | 7 of 8. Right |
| P0: mean gain under 0.03 | 0.0199. Right |
| PM: 2 of 4 at the strong level | 1 of 4. Too high |
| PW: 2 of 4 at the strong level | 1 of 4. Too high |
| T1 holds (55 percent) | Does not hold. Wrong |
| T2 does not hold (65 percent) | Does not hold. Right |
| PW minus PM: no clear difference | No clear difference. Right |

## The run

- K: 16 training runs (8 files, seeds 1001 and 1002), 15:15 to 15:35 EST.
- Training: one run, 40 minutes on one A4000 (cherry), 48 examples, 5 epochs. Loss on the 3 held-out loops: 2.20
  before, 1.09 after (lowest 1.04 after epoch 3; the final adapter was used, as planned). Deviation 1 in the plan: the
  first start stopped with a GPU out-of-memory error before any weight update; the loss code was changed to use less
  GPU memory (same loss, same data, same settings).
- Loops: 12 loops, all exit 0, none left out, none resumed, no server wait. 16:11 to 19:45 EST on spruce and willow.
- Model spend: 0.24 USD at the nominal price of the local models. No Fable call was made in S6.
- Shared machines: peak of my total memory per host from the guard logs: spruce 14.9 GB (P2B included), willow 2.6 GB,
  cherry 9.8 GB. The department limit is 50 GB. The guards stopped nothing. The model server and the guards were
  stopped at 19:52.

## Limits

- Same recipe for the old and the new pool. This measures reuse inside one job family. Nothing here says the stored
  file, the memory text or the adapter help on a different job.
- 4 loops per cell; the PW interval includes zero and includes PM. Anchors are from R1, not concurrent.
- One training setup (rank 16, 5 epochs, 48 examples). A different setup or more data could give another answer; none
  was tried, as planned.
- The memory text is 3 ideas chosen by one rule. Other choices were not tried.
- The table of broken slots was made after seeing the result.

## What it means for the thesis

C1 and R1 said: pay for the strong model in the planner role, and stop paying once the gain is in. S6 adds the cheapest
rule of all for repeated jobs: keep what the strong loop found. As a stored file it is worth the whole strong gain at
zero cost; as three lines of text in the prompt it is worth about half; trained into a small model with 48 examples it
is worth less than the text. A routing policy for a job that repeats should check its memory before it picks a model.

## Addendum: the demo on a different recipe (exploratory, run after the result, no rule set)

`curation/studio.py` is the small product demo: given a pool, it screens the library of old strong files on one seed,
confirms the best on a second seed and prints the route. It was run on pool v3, which is **not** the same recipe as the
pool the files were written for (v3 adds three kinds of damaged Wikipedia text and has less clean Wikipedia).
Output: `curation/runs/studio/v3.out`.

- All 8 old files beat the raw v3 pool on the screening seed: dev bpb 0.099 to 0.111 below raw (raw 1.9551).
- Best file (SS_03): sealed bpb 0.1063 below raw on seed 0 and 0.1618 on seed 1. Seed 1 is a noisy raw run, so read the
  seed 0 number. For scale, the best hand-built mix that uses the hidden labels reaches 0.0986 below raw on seed 0.
- Route printed: reuse the stored file, 0 USD of model calls, 9 training runs for the check (about 10 minutes).

This weakens the first limit above a little: the stored files also carry over to one pool with a different recipe.
It is one pool, two seeds and no strong loop was run on v3 to compare with, so it is not a result of the study.
