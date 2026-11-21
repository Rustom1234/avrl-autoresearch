# SR5 result: unknown parts of a law hide each other from the cheap model

2026-11-19. Rules and guesses: `loop/design/SR5_PREREG.md` (written 15:51, before any session). Code:
`loop/sr/advice2.py`. Run: `loop/runs/SR5` (hinted problems in `loop/sr/problems_advice2`), 15:52 to 19:04.
Haiku only: 240 chains, 584 sessions, 5.94 USD nominal, no failed session, no Fable or Sonnet session.
Cells 000, 001, 110 (level 3) and 0000, 0010, 0011, 1100, 1111 (level 4) are reused from SR4 (same day, same
code, same hint wording).

## Words

A law is a sum of terms. Level 3 has three: S (a simple factor), CS (a common factor times a simple factor) and
U (an unusual factor). Level 4 has a fourth, US (unusual times simple). A hint cell tells Haiku the exact form of
some terms and nothing about the others. f(cell) is the share of fresh Haiku chains (12 problems, 2 chains each,
up to 4 steps) that reach validation error at most 1e-10.

## Level 3, all eight cells

| Told | Missing | Bits told (of 13.8) | Solved | Share | Guess written before |
|---|---|---|---|---|---|
| nothing | S, CS, U | 0 | 1 of 24 | 0.04 | (SR4) |
| S | CS, U | 3.6 | 3 of 24 | 0.12 | 0.10 |
| CS | S, U | 6.2 | 5 of 24 | 0.21 | 0.20 |
| U | S, CS | 4.0 | 15 of 24 | 0.62 | (SR4) |
| S, CS | U | 9.8 | 11 of 24 | 0.46 | (SR4) |
| S, U | CS | 7.6 | 22 of 24 | 0.92 | 0.80 |
| CS, U | S | 10.2 | 24 of 24 | 1.00 | 0.95 |
| all | none | 13.8 | 24 of 24 | 1.00 | 1.00 |
| wrong form for U | all | 0 | 0 of 24 | 0.00 | 0.03 |

Bit values are for a typical problem; they vary a little with the input count.

## Level 4

| Told | Missing | Solved | Share | Guess |
|---|---|---|---|---|
| nothing | all four | 0 of 8 | 0.00 | (SR4) |
| U | S, CS, US | 0 of 8 | 0.00 | (SR4) |
| S, CS | U, US | 1 of 24 | 0.04 | (SR4) |
| U, US | S, CS | 10 of 24 | 0.42 | (SR4) |
| S, CS, U | US | 11 of 24 | 0.46 | 0.20 |
| S, CS, US | U | 17 of 24 | 0.71 | 0.45 |
| S, U, US | CS | 20 of 24 | 0.83 | 0.75 |
| CS, U, US | S | 24 of 24 | 1.00 | 0.95 |
| all | none | 8 of 8 | 1.00 | (SR4) |

## Claims fixed in advance

- **F1, hiding at level 3: PASSED.** If each missing term were found on its own, the share for a cell would be
  the product of the shares with only one term missing. Mean of observed minus this product over the four cells
  with two or three missing terms: -0.304 (95% -0.490 to -0.142, resampling the 12 problems). Predicted against
  observed: missing CS and U 0.42 against 0.12; missing S and U 0.46 against 0.21; missing S and CS 0.92 against
  0.62; missing all 0.42 against 0.04.
- **F2, the product term is the bottleneck at level 4: FAILED.** f(US missing) minus f(U missing) = -0.250
  (95% -0.583 to +0.083). The direction is as guessed, the interval includes 0.
- **F3, hiding at level 4: PASSED.** f(U and US missing) minus f(U missing) times f(US missing) = -0.283
  (95% -0.514 to -0.104): 0.32 predicted, 0.04 observed.

Chances written before the run: F1 0.6, F2 0.5, F3 0.4.

## What each term is worth to be told (level 3, descriptive)

Shapley values (the average gain from being told a term, over all orders of telling): S +0.12, CS +0.20,
U +0.64. They sum to 0.96, the gap between no hint and the full hint. The U hint is 4 of the 13.8 bits and
carries two thirds of the gain. The bit count of a hint under the generator's odds does not predict its value.

## Stricter check, added after the run (not in the rules file)

The product rule lets every missing term use the whole budget. A fairer rule for a solver that finds the terms
one after another: the efforts add. effort(missing set) = base + sum over the missing terms of (effort with only
that term missing - base), and the prediction is the share of chain combinations of the same problem whose
summed effort fits the budget (4 steps, or 56 tool calls). `advice2.py seq`.

| Level | Missing | Observed | Product | Efforts add, steps | Efforts add, tool calls |
|---|---|---|---|---|---|
| 3 | S, CS | 0.62 | 0.92 | 0.92 | 0.92 |
| 3 | S, U | 0.21 | 0.46 | 0.46 | 0.44 |
| 3 | CS, U | 0.12 | 0.46 | 0.44 | 0.44 |
| 3 | S, CS, U | 0.04 | 0.46 | 0.44 | 0.39 |
| 4 | S, CS | 0.42 | 0.83 | 0.81 | 0.79 |
| 4 | U, US | 0.04 | 0.35 | 0.23 | 0.12 |
| 4 | S, CS, US (4 problems) | 0.00 | 0.38 | 0.31 | 0.12 |
| 4 | all (4 problems) | 0.00 | 0.38 | 0.16 | 0.00 |

Here the product is taken per problem, so it differs a little from the pooled product of F1. Mean of observed
minus predicted under "efforts add, tool calls": level 3 -0.294 (95% -0.464 to -0.141); level 4 (the two cells
with 12 problems) -0.229 (95% -0.354 to -0.104). So the shortfall is not explained by two searches sharing one
budget. Every cell with two or more missing terms on 12 problems is below every prediction (6 of 6).

## Wrong hint (descriptive)

With a wrong form for U, 0 of 24 chains solve. The final formula still contains the wrong form in 21 of 24
chains, 91% of all formulas sent to the checker contain it, and only 1 chain ever tries the true form
(`advice2.py wrong`). Haiku does not test the hint; it blames the other terms. That models follow wrong hints is
known (see below).

## Could a strong model supply the hint cheaply? (stored sessions, no new session)

In the 12 stored Fable first attempts at level 3 (`loop/runs/SR1/attempts`, mean 0.89 USD, mean 13.3 tool calls,
5 solved), the first formula sent to the checker that contains the true unusual form comes at tool call 9, 10,
12, 13, 13 in the five solved sessions, and at call 13 of 14 in two unsolved ones; never in the other five. No
session has it by call 7. Haiku: 7 of 144 sessions ever have it, 0 solved. Sonnet: 4 of 48, 0 solved. The
detector reads formulas sent to the checker, not the model's reasoning, so "has it" means "tested it".
Reading: Fable reaches the hard term only at the end of a full session and then solves by itself. A short Fable
visit would very likely not supply the hint. The hint is cheap to use and expensive to make.

## Known work, and what a search did and did not find

Five web searches on 2026-11-19 (not a full review):

- Hints in equation discovery with language models: LaSR (Grayeli et al., NeurIPS 2024, arXiv 2409.09359) adds
  vague hints from the chapter title and reports faster solving. One on or off condition, no term-level hints.
- Models follow wrong hints: AssistedDS (arXiv 2506.13992), Unspoken Hints (arXiv 2509.26041). The wrong-hint
  cell repeats a known pattern in a new task.
- Full factorial designs with interaction effects for agent parts: arXiv 2605.05716 (five scaffold components,
  all 32 subsets). The design is a standard tool (Fisher), as are Shapley values (1953).
- Solving the parts but not the whole: the compositionality gap (Press et al., arXiv 2210.03350), for question
  answering.
- Finding one term at a time from residuals in symbolic regression: Luo et al. 2017 (arXiv 1705.08061), AI
  Feynman 2020 (separability), RISR (arXiv 2610.11387), L2G-SR (arXiv 2610.20685).
- Not found: a term-level hint table for an equation-discovery agent that tests whether the missing parts are
  found independently. This is "not found in five searches", not a novelty claim.

## Limits

- One model (Haiku), one synthetic task family from my own generator, 12 problems per level, 2 chains per cell
  and problem, budget of 4 steps. No second model, no outside benchmark.
- Some cells come from SR4 and some from SR5 (same day, same code).
- The hint tells how many terms the law has, so Haiku knows how many are missing.
- The "efforts add" check and the wrong-hint scan were added after seeing the results.
- The stored-session scan uses text patterns (see Deviations in the rules file).
