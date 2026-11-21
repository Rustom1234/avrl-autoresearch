# SR3 result: the "out of ideas" signal does not work on this task

2026-11-19. Rules: `loop/design/SR3_PREREG.md` (written before the run). Code: `loop/sr/novelty_test.py`, run once.
Output: `loop/runs/SR2/novelty_test.txt`. Stored sessions only, no model session, 11 seconds, 44 MB.

## Short version

Both written claims fail, so by the kill rule the certificate is dropped and no replay was built.

## What was tested

Idea: with an exact checker, a chain can only solve by proposing something it has not tried. So the rate of new
proposals (Good-Turing estimate) should bound the chance of solving, and a chain that stops producing new things
should be a chain that will not solve. Two definitions of "new": new fitted functions among the candidates sent
to the checker, and new building blocks in the code of all tool calls.

## Result

E1, signal (117 chains of levels 2, 6, 3 that were unsolved after 4 steps; 43 solved by step 12):

| Signal at step 4 | AUC alone (0.5 = no information) |
|---|---|
| Best error so far | 0.77 |
| Good-Turing estimate (new functions) | 0.56 |
| Breadth (distinct building blocks) | 0.57, in the direction "broader chains solve less" |
| Recent novelty (blocks new in the last two steps) | 0.52 |

Fitted leaving one problem out: best error alone 0.753; best error plus the three signals 0.740; difference
-0.013 (95% interval -0.039 to +0.013). Fail.

E2, certificate (2,317 chain steps, all levels): the bound has median 0.77 and is below 0.5 on only 8% of steps,
so it says almost nothing. Where it was tight (below 0.25: 7 steps) the next step solved 2 times of 7, above the
bound of 0.09. Fail.

E3, level 4 (descriptive, 4 chains solved after step 12): lower error at step 12 goes with solving (AUC 0.84),
and so does narrower breadth (0.83). The novelty signals do not (0.56, 0.76 in the direction "less novelty").

## Why it fails

- Haiku sends only 2.7 candidates per step to the checker and nearly every one is a new function. The stream of
  proposals never dries up, so "out of ideas" does not happen in a way that can be counted.
- Trying more kinds of building blocks does not help. If anything, chains that solve are the focused ones.
- The plain error of the best candidate remains the best early signal we have (AUC about 0.75 to 0.78).

## Prior art

See the review in `loop/design/SR3_PREREG.md`. The "parts law" is not new as an idea (Ord 2025; MAKER 2025; Large
Language Monkeys 2024). Every component of the certificate exists (Good-Turing on LLM samples, a missing-mass
stopping rule for LLM sampling, the fuzzing bound, stagnation handoff); only the combination was not found, and
it does not work here.
