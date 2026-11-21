# SR1 result: a cost-per-success index for model routing, on equation discovery

Run and written 2026-11-19 by the coding agent after Rustom's instruction of 02:14. Rules written beforehand:
`loop/design/SR1_PREREG.md` (gate, claims C1 to C5, three deviations). Code: `loop/sr/`. Data: `loop/runs/SR0` (gate),
`loop/runs/SR1` (main bank, replay outputs `replay_final.txt`, `replay_whatif.txt`), `loop/runs/SR0C` and `SR2`
(chains). Nothing here is reviewed by Rustom yet.

## Why this task

On the drug-property and packing tasks (BG1, E1, B1, B2) Haiku alone gets about 80 percent of the way to the best
published result and no model finds real gains at the stall, so there is nothing to route. This task was chosen so
that (a) a stronger model can do what a cheaper one cannot, (b) the score is exact and cannot depend on run time or
machine load, (c) nothing can be recalled from training data, (d) a test costs seconds of CPU.

## The task

A problem is a table of noise-free measurements made from a hidden law (2 to 4 terms), generated fresh by
`loop/sr/gen.py`. One attempt is one session of one model with two tools (run Python on the data; score a candidate
law), at most 14 tool calls. Success: the candidate's error on unseen validation rows is at most 1e-10 (the right
law reaches about 1e-23; a wrong 10-constant fit reaches about 1e-5). The same candidate always gets the same score.
Five difficulty levels, four settings, three seeds: 60 problems. Stored attempts per problem: Haiku 12, Sonnet 4,
Fable 1, all fresh (no attempt sees another). 1,020 sessions, 0 failed. Held-out check (same threshold outside the
training ranges): of the validation successes, Haiku 182 of 194, Sonnet 72 of 75, Fable 49 of 49 also pass there.

## What the models can do (main bank, successes per attempt)

| Level, easiest first | Haiku, 0.0075 USD | Sonnet, 0.074 USD | Fable, 0.47 USD |
|---|---|---|---|
| 1 | 133 of 144 | 45 of 48 | 12 of 12 |
| 5 | 59 of 144 | 25 of 48 | 12 of 12 |
| 2 | 1 of 144 | 3 of 48 | 10 of 12 |
| 6 | 1 of 144 | 2 of 48 | 10 of 12 |
| 3 | 0 of 144 | 0 of 48 | 5 of 12 |

The gate run (seed 0) showed the same pattern beforehand. This is the first task in the program where a stronger
model does what a cheaper one cannot. Problems of one level differ a lot: on level 5 Haiku solves between 0 and 10
of its 12 attempts depending on the problem.

## The rule

Always make the attempt with the lowest expected dollars per success, rho = price / P(success), over all open
problems and models. P(success) comes from memory: the problem is assumed to behave like one of the problems of
other seeds, each weighted by how well it explains the failures seen so far on this problem and how close those
failures got. Spending in order of rho is the greedy answer to the knapsack "most problems solved for B dollars".

Known parts: the reservation-price index (Weitzman 1979), the greedy cost-per-success schedule and its factor 4
guarantee (Streeter and Golovin 2008; Gergatsouli and Tzamos 2023), greedy knapsack, cost of pass as a measure
(arXiv 2504.13359). Model choice by bandits inside program search exists (ShinkaEvolve, AB-MCTS); one model per
workload by multiple-choice knapsack exists (arXiv 2608.29560). Not found in my searches: repeated attempts by
several models across a set of research problems under one budget, with beliefs learned from other problems.

## Result at real prices

Every policy is replayed on the same stored attempts (100 random orders). Area = share of the 60 problems solved,
averaged over all budgets from 0 to 28.48 USD (one Fable attempt on every problem). Last column: area of the rule
minus area of that policy, 200 resamples of the problems, 95% interval.

| Policy | Area | USD to solve half | Rule minus policy |
|---|---|---|---|
| Always Haiku | 0.408 | never | +0.169 (+0.119 to +0.217) |
| Always Sonnet | 0.370 | never | +0.200 (+0.134 to +0.277) |
| Always Fable | 0.392 | 17.52 | +0.177 (+0.079 to +0.262) |
| Cascade, 1 try each | 0.526 | 11.48 | +0.046 (-0.003 to +0.097) |
| Cascade, 3 cheap tries then Fable | 0.572 | 8.64 | +0.001 (-0.044 to +0.049) |
| Cascade, every model used up in turn | 0.484 | 13.77 | +0.094 (+0.038 to +0.146) |
| Cascade, schedule tuned on other problems | 0.559 | 9.27 | +0.017 (-0.031 to +0.065) |
| Best fixed schedule in hindsight (a bound) | 0.579 | 8.02 | -0.014 (-0.054 to +0.023) |
| Bandit over models with cost blend | 0.476 | 12.59 | +0.101 (+0.050 to +0.148) |
| Thompson sampling, no cost | 0.454 | 14.24 | +0.124 (+0.057 to +0.192) |
| Thompson sampling, cost aware | 0.485 | 13.64 | +0.093 (+0.041 to +0.145) |
| The rule with separate beliefs per model | 0.484 | 13.77 | +0.094 (+0.038 to +0.146) |
| The rule with memory of failure counts only | 0.546 | 9.77 | +0.032 (-0.004 to +0.069) |
| **The rule** | **0.577** | **8.92** | |
| Clairvoyant, knows each problem's success shares | 0.714 | 2.11 | -0.133 (-0.190 to -0.077) |

Verdicts by the rules written beforehand:

| Claim | Verdict |
|---|---|
| C1 the rule beats every single model | **pass** |
| C2 the rule beats every fixed cascade | **fail**: it ties "3 cheap tries then Fable" and the schedule tuned on other problems |
| C3 the rule beats the bandit and cost-aware Thompson | **pass** |
| C4 how close Haiku got predicts that a stronger model solves the problem | **fail**: AUC 0.49 (0.28 to 0.70) |
| C5 memory beats separate beliefs and failure counts only | **half**: against separate beliefs yes; against failure counts only +0.032 with an interval that touches zero |

In words: learning across models is what matters (0.577 against 0.484). After Haiku fails, memory says Sonnet is
poor value on these problems and Fable is not, so the rule skips the middle model; the bandit and Thompson routers
and "use everything up" spend on Sonnet first. A simple schedule that happens to skip Sonnet does as well.

## Post hoc findings (not in the written rules; each needs a fresh test)

1. What the closeness signal does predict: whether Haiku itself solves the problem on another try. Median error of
   a problem's failed Haiku attempts against "Haiku solves it on some other attempt": AUC 0.79 (0.62 to 0.93).
   Failed attempts sit near 1e-5 on problems Haiku can solve and near 1e-3 on problems it cannot.
2. Prices changed, same stored attempts (30 orders, no intervals):

   | Prices | Rule | 3 cheap then Fable | All used up in turn | Thompson, cost aware | Bandit | Schedule re-tuned on other problems | Best in hindsight |
   |---|---|---|---|---|---|---|---|
   | Real | 0.577 | 0.575 | 0.486 | 0.485 | 0.481 | 0.559 | 0.581 |
   | Fable 5 times dearer | 0.632 | 0.593 | 0.634 | 0.632 | 0.613 | 0.609 | 0.644 |
   | Fable 5 times cheaper | 0.496 | 0.486 | 0.369 | 0.367 | 0.367 | 0.500 | 0.508 |
   | Haiku at Sonnet's price | 0.494 | 0.393 | 0.321 | 0.362 | 0.373 | 0.477 | 0.477 |
   | All models one price | 0.395 | 0.131 | 0.131 | 0.137 | 0.258 | 0.395 | 0.395 |

   The rule is within 0.012 of the best fixed schedule in hindsight in every setting without retuning. Every
   hand-written schedule and both bandit-style routers fail badly in at least one setting. Only a schedule
   re-tuned by grid search for each price setting keeps up; the rule ties or beats it in four of five, by at most
   0.02. The rule's mix of attempts moves with the prices (real: 475 Haiku, 53 Sonnet, 34 Fable per run; dear
   Fable: 133 Sonnet; one price: Fable only).

## Chains: attempts that build on each other (exploratory, no rules written beforehand)

Superseded by `loop/RESULT_SR2.md`: four chains per problem, 12 steps, all 60 problems, rules written beforehand
(`loop/design/SR2_PREREG.md`). The numbers below are the first half of that data.

`loop/sr/trie.py`, runs `loop/runs/SR0C` (12 gate problems) and `loop/runs/SR2` (the 48 main problems of levels 5,
2, 6, 3; two Haiku chains of up to 8 steps per problem). Each step is one Haiku session that is shown the three
best candidates of the earlier steps. 549 steps, 0 failed.

Share solved within k steps of a chain, against k fresh Haiku attempts on the same problems:

| Level | Chained, k = 1 | 2 | 4 | 8 | Fresh, k = 1 | 2 | 4 | 8 | 12 | Mean cost of a chain |
|---|---|---|---|---|---|---|---|---|---|---|
| 5 | 0.38 | 0.71 | 0.96 | 0.96 | 0.42 | 0.50 | 0.83 | 0.92 | 0.92 | 0.018 USD |
| 2 | 0.00 | 0.08 | 0.42 | 0.62 | 0.00 | 0.00 | 0.00 | 0.00 | 0.08 | 0.057 USD |
| 6 | 0.00 | 0.08 | 0.17 | 0.33 | 0.00 | 0.00 | 0.00 | 0.00 | 0.08 | 0.067 USD |
| 3 | 0.00 | 0.00 | 0.00 | 0.04 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | 0.083 USD |

On level 2 a Haiku chain solves 62 percent of the time for about 6 cents; twelve independent Haiku attempts (9
cents) solve 8 percent of the problems; one Fable attempt (47 cents) solves 83 percent. Per solved problem that is
about 0.09 USD for the chain against 0.57 USD for Fable.

Whole plans, all 60 problems worked to the end (mean of two runs per problem; Fable = the stored fresh attempt):

| Plan | Solved of 60 | Total USD | USD per solved problem | Fable sessions |
|---|---|---|---|---|
| Always Fable | 49.0 | 28.48 | 0.581 | 60.0 |
| 3 fresh Haiku tries, then Fable | 49.0 | 26.44 | 0.540 | 39.5 |
| Haiku chain up to 4 steps, then Fable | 50.0 | 21.76 | 0.435 | 29.5 |
| Haiku chain up to 8 steps, then Fable | 50.0 | 20.54 | 0.411 | 24.5 |
| Haiku chain up to 8 steps, no Fable | 35.5 | 2.76 | 0.078 | 0 |
| 6 fresh Haiku tries, no Fable | 23.5 | 2.17 | 0.092 | 0 |

Fable shown the candidates of a stalled 6-step Haiku chain (9 problems of seed 1 where the chain had not solved):
solved 8 of 9 at a mean of 0.79 USD; the stored fresh Fable attempt on the same 9 problems solved 4. The four
problems that differ all favour the seeded attempt (sign test p = 0.125). Nine problems, one draw each: a lead,
not a result.

What this means for routing: in a loop where attempts build on each other, the cheap model is the workhorse and
"after the cheap model fails, call the strong one" is the wrong rule. The decisions that matter are how long to
keep refining with the cheap model, when its chain has stalled, and whether to hand its candidates to the strong
model. That is the refine, redraw, escalate or stop problem of the Nov 18 index plan, now on a task where the
answer is not obvious.

## Limits

- One task family, made by us: random laws in a scientific setting, not real science. Claude models only.
- Attempts do not build on each other here. The chain runs below show that this matters.
- One stored Fable attempt per problem, so no policy can try Fable twice.
- 60 problems: differences below about 0.05 in area cannot be resolved.
- Dollars are nominal (what the command-line client reports).
- The gap to the clairvoyant bound (0.577 against 0.714) is knowledge of each problem's type and of single draws;
  only the first part is attainable.
- The published benchmark of this kind (LLM-SRBench) is not yet run: its data needs a HuggingFace login.

## Mistakes and fixes during the night

- The first version of the replay judged each problem with a memory that left out only itself. That makes hard
  problems look slightly better and decides ties the wrong way. Fixed before any main result: one memory per test
  fold, random problem order per run, ties broken by fewest tries.
- No kill, no memory error, no failed session. Highest load on spruce about 7 of 32.

## Cost

Nominal dollars: gate 19.75 (Haiku 0.59, Sonnet 6.07, Fable 13.09); main bank 51.69 (Haiku 5.43, Sonnet 17.78,
Fable 28.48); chain pilot 4.57; chain run on the main problems see below. About 1,300 sessions before the chain run.
