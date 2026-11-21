# S1: the loop engine on ShinkaEvolve's circle-packing task, and a no-loop control

Nov 17 2026. Plan frozen before launch: `loop/design/S1_PREREG.md`. Loops: `loop/runs/S1` (9 loops, 150 slots,
Nov 16 16:17 to Nov 17 03:02 EST). Control: `loop/runs/S1_IID` (`loop/shinka/iid.py`, run after the loops, not in
the frozen plan). Tables: `loop/shinka/progress.py`.

## The question

ShinkaEvolve reports the best known packing of 26 circles in the unit square (sum of radii 2.635983) after about
150 generations from a hand-placed start that scores 0.96. Can our loop match that with fewer evaluations and
dollars under the same conditions (their starting program, their validity check at zero tolerance, 5 minutes per
evaluation, their domain hints in the prompt)?

## Run health

All 9 loops ran 150 slots. No failed model call, no authentication failure, no memory event. Every top packing was
checked again in exact rational arithmetic (zero tolerance). No winning script contains stored coordinates.

## Loops

| Cell (planner, coder) | Loops reaching 2.635977 | Evaluations to reach it | Model USD when reached | Total USD per loop (median) |
|---|---|---|---|---|
| Haiku, Haiku | 3 of 3 | 14, 16, 19 | 0.15, 0.18, 0.25 | 1.79 |
| Sonnet, Haiku | 3 of 3 | 9, 14, 37 | 0.14, 0.15, 0.80 | 2.92 |
| Sonnet, Sonnet | 3 of 3 | 2, 9, 9 | 0.14, 0.49, 0.52 | 13.85 |

Best sum over all loops: 2.6359830849, the same packing as ShinkaEvolve's. A tie with the best known value, not a
record. Between 76 and 95 percent of each cell's evaluations came after the top packing had been found.

Frozen claims: M match HOLDS (3 of 3 in every cell). C cheaper HOLDS (median cost when reached 0.15 to 0.49 USD,
bar 6). R routing FAILS by the letter: the median best sum of the Sonnet-planner, Haiku-coder cell is 3e-9 below
all-Sonnet (the same packing with a different safety margin on the radii), at 2.92 against 13.85 USD. The rule
should have compared against a level, not against the other cell's last digits.

## Control: one model call, no loop

Each sample is a single call that sees the same card and starting program and returns a whole new script, scored
once. No sample sees a result or another sample.

| Model | Samples | Reach 2.635977 | Median sum | Model USD per sample (as reported by the CLI) |
|---|---|---|---|---|
| Haiku | 20 | 5 (25%) | 2.6314 | about 0.01 |
| Sonnet | 12 | 6 (50%) | 2.6359 | about 0.08 |

So a single Haiku call matches the published 150-generation result one time in four, and about four independent
calls (a few cents) are expected to reach it. The loops needed 14 to 19 evaluations with Haiku. On this task the
loop is not what finds the answer, and it is slower than independent sampling.

## What it means

1. **The benchmark is saturated for current models.** Under matched conditions the best known packing is a
   single-call result for both models. The comparison with ShinkaEvolve's reported budget says the models got
   better; it says nothing in favour of our loop. This agrees with Gideoni, Risi and Gal, who found simple sampling
   competitive with code evolution, and goes further: with 2026 models one sample is often enough.
2. **Which search to run depends on the chance that one fresh sample succeeds.** Here that chance is 25 to 50
   percent, so independent restarts beat step-by-step refinement. A loop design question ("refine the incumbent or
   draw a fresh program?") that our engine never asks turns out to matter more than the model routing.
3. **Stopping matters most for cost.** Every cell spent at least three quarters of its evaluations after success.
4. **Model choice:** Haiku everywhere matched Sonnet everywhere at an eighth of the total cost; Sonnet got there in
   fewer evaluations (median 9 against 16).

## Limits

- The control was run after seeing the loops and is not preregistered. Its calls go through the command-line
  client with its default settings, not the loop's model service, so cost per sample is approximate.
- ShinkaEvolve's numbers are from their paper with 2025 models; nothing of theirs was rerun here.
- 3 loops per cell; 20 and 12 control samples.
- The models may simply know the standard method for this problem (the hints name it). That is the point of
  finding 1, and it means this task cannot be used to compare loop designs.
