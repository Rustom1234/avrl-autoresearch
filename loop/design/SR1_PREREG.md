# SR1: a cost-per-success index for model routing, tested on equation discovery

Written 2026-11-19, about 02:50, by the coding agent after Rustom's instruction of 02:14 ("find the right tasks, and
then implement an algorithm like we've discussed, the knapsack, the index theory"). He has not reviewed it; it is his
to audit or void. What I had seen when writing: 4 smoke attempts, a self-test of the replay code on made-up data,
and the first 19 of 108 gate attempts (success counts only: level 1 solved by Haiku and Sonnet, level 2 not yet).

## Why a new task

BG1, E1, B1 and B2 (drug-property models, circle packing): Haiku alone goes about 80 percent of the way to the best
published result in 40 steps, and at the stall no model finds real gains. There is nothing to route. A routing test
needs a task where (a) a stronger model can do what a cheaper one cannot, (b) the score is exact and does not depend
on run time or machine load, (c) the answer cannot be recalled from training data, (d) one test costs seconds of CPU.

## Task: find a hidden law from data, with tools

- A problem is a table of noise-free measurements generated from a hidden closed-form law (2 to 4 terms; standard
  terms and unusual ones). Fresh laws from `loop/sr/gen.py`, so none can be in any model's training data. Four
  difficulty levels, four scientific settings. Same recipe as LSR-Synth of LLM-SRBench (Shojaee et al., ICML 2025),
  whose data sits behind a login; it becomes an outside anchor when Rustom provides access.
- One attempt = one session of one model with two tools (run Python on the training data; score a candidate law),
  at most 14 tool calls. The session's best candidate (lowest training error) is its result.
- Success = the candidate's error (NMSE) on the validation split is at most 1e-10. Constants are fitted by least
  squares from 8 fixed starting points: the same candidate always gets the same score. Held-out check: the same
  threshold outside the training ranges (test_ood). The solver never sees validation or test data.
- Published evidence that the model matters on this kind of task: SR-Scientist (ICLR 2026) reports 32 to 64 percent
  solved on LSR-Synth depending on the model, when the model works as an agent with tools.

## Gate (run SR0: seed 0, 16 problems). Rules fixed now

- G1 Same candidate scored twice gives the same number. Checked before the gate (identical to all digits).
- G2 Range: at some level at least two models have a per-attempt success share between 0.05 and 0.95.
- G3 Gap: at some level a stronger model's success share is at least 0.2 above a cheaper model's.
- G4 Reach: at levels 2 to 4 together, the best model solves at least 25 percent of its attempts.
G2 or G4 fails: change the difficulty levels once and rerun the gate. A second failure: stop and report.
G3 fails: the task does not separate the models; report it, do not build the comparison on it.
The gate has few attempts per cell, so it is a filter, not a result. The main bank measures the same shares again.

## Main bank (run SR1)

- Problems: seeds 1, 2, 3, all four levels and settings: 48 problems. Easy problems stay in: a router must find out
  by itself that the cheap model is enough.
- Attempts stored per problem: Haiku 6, Sonnet 3, Fable 1. Every attempt is fresh (no attempt sees another).
- The seed 0 problems are used for the gate and as memory only; they are never scored.

## Policies (replayed on the same stored attempts, `loop/sr/policy.py`)

- The rule: always make the attempt with the lowest expected dollars per success, rho = cost / P(success). Beliefs
  come from memory: the problem is assumed to behave like one of the problems of other seeds, each weighted by how
  well it explains the failures seen so far and how close the failed attempts got (their training error).
  Spending in order of rho is the greedy solution of the knapsack "most problems solved for B dollars".
- Comparisons: each model alone; three fixed cascades (1 try each; 3 cheap tries then the strongest; every model
  used up in turn); a bandit over models with a cost blend (the style of ShinkaEvolve); Thompson sampling without
  cost (the arm choice of AB-MCTS) and with cost; the rule with separate beliefs per model; the rule with memory of
  failure counts only; a clairvoyant reference that knows each problem's true success shares.
- Worlds: 100 random orders of the stored attempts. Problem order is random per world and the same for all policies.

## Metric and claims, fixed now

Primary metric: area = share of problems solved, averaged over all budgets from 0 to B_max, where B_max is the mean
cost of one Fable attempt times the number of problems. Uncertainty: 200 resamples of the problems, 95% interval of
the difference in area.

| Claim | Passes if | My guess |
|---|---|---|
| C1 routing | the rule's area is above every single model's, each interval excluding 0 | 0.8 |
| C2 against common practice | the rule's area is above each of the three cascades, each interval excluding 0 | 0.5 |
| C3 against bandits | the rule's area is above the cost-blend bandit and cost-aware Thompson, intervals excluding 0 | 0.35 |
| C4 signal | among problems where every Haiku attempt failed, the best Haiku training error predicts whether Sonnet or Fable solves the problem: AUC at least 0.70, interval above 0.5 | 0.5 |
| C5 memory | the rule beats the same rule with separate beliefs and with failure counts only, intervals excluding 0 | 0.4 |

Secondary, reported whatever they show: dollars to solve half of the problems; the rule's attempts by model; share
of validation successes that also pass outside the training range; the price (rho) at which each budget runs out.

Kill rules. C1 fails: routing adds nothing here, report it. C2 and C3 fail but C1 passes: the rule is no better than
simple cheap-first rules on this task; report that, the task and the measured shares stand alone. C4 fails: drop
the "how close" signal from every claim.

## Known limits, stated before the data

- Stage 1 only: attempts do not build on each other. Refining an earlier candidate (and handing a cheap model's
  candidate to a strong model) is the next stage and needs live runs.
- At most 6, 3 and 1 stored attempts per problem. A policy cannot try Fable twice on one problem.
- One task family, made by us; random laws in a scientific setting, not real science. Claude models only.
- Dollars are nominal (what the command-line client reports), one price per model.
- 48 problems: only large differences can show.
- The parts of the rule are known: the reservation-price index (Weitzman 1979), the greedy cost-per-success
  schedule and its factor 4 guarantee (Streeter and Golovin 2008; Gergatsouli and Tzamos 2023), greedy knapsack.
  Bandits over models inside program search exist (ShinkaEvolve, AB-MCTS). What would be ours: the setting
  (repeated attempts by several models across a portfolio of research problems under one budget), the memory-based
  joint belief, and the measured value of "how close the cheap model got" as a routing signal.

## Safeguards

Sessions run on spruce only, at most 6 at a time, each tool call one thread with 20 s of CPU and 4 GB of address
space (quiet limits). The builder waits while the machine's load is above 16. No job's result depends on run time.
A login failure stops the run; a session that breaks is logged and not tried again; dollar cap per build.

## Deviations

1. 2026-11-19, 02:47, before any main-bank attempt. Stored attempts per problem are Haiku 12, Sonnet 4, Fable 1
   (was 6, 3, 1). Reason: the gate's price data (Haiku 0.009, Sonnet 0.08, Fable 0.8 USD per attempt). With a
   90-fold price gap the question "how long to keep trying the cheap model before paying for the strong one" needs
   longer cheap streams than 6. Two competitors were added for the same reason: the fixed schedule (h cheap tries,
   s middle tries, then the strongest) with the best area in hindsight, and the same schedule tuned on other folds.
   C2 now also requires beating the schedule tuned on other folds; the hindsight schedule is reported as a bound.
2. 2026-11-19, 02:47, from the first 51 gate attempts. Levels 3 and 4 were solved by no model (0 of 22 attempts),
   level 2 by Fable 3 of 3, Sonnet 1 of 4, Haiku 0 of 5; level 1 by Haiku 3 of 5, Sonnet 5 of 5. Two levels were
   added between them (5: between 1 and 2; 6: between 2 and 3) and sent through the gate. The main set will be
   chosen from the gate results only: levels where some model succeeds, plus one level nobody solves as the
   hopeless tail (a router must learn to stop). Level 4 is dropped. To save time the main bank started at 02:48 with
   what does not depend on that choice: Haiku attempts for levels 1, 2, 5, 6 and 3 (cheap, about 6 USD nominal for
   all), and Sonnet and Fable attempts for levels 1 and 2. Sonnet and Fable attempts for the other levels wait for
   the gate. No main-bank result is read before the choice of levels is written here.
3. 2026-11-19, 02:53. Gate outcome and choice of levels, from 75 of 156 gate attempts (all 12 Fable attempts in;
   no main-bank result read). Successes per attempt, levels in order of difficulty:

   | Level | Haiku | Sonnet | Fable |
   |---|---|---|---|
   | 1 | 5 of 7 | 7 of 7 | not run |
   | 5 | 1 of 3 | 0 of 3 | pending |
   | 2 | 0 of 6 | 1 of 6 | 3 of 4 |
   | 6 | 1 of 3 | 1 of 3 | pending |
   | 3 | 0 of 7 | 0 of 6 | 1 of 4 |
   | 4 | 0 of 6 | 0 of 6 | 0 of 4 |

   G2 passes (level 2: Sonnet 0.17 and Fable 0.75 are both inside 0.05 to 0.95, and the attempts still to come
   cannot move Sonnet out of that range). G3 passes (level 2: Fable 0.75 against Haiku at most 0.5 whatever the
   remaining attempts give). G4 passes (Fable solves 4 of 12 at levels 2 to 4). The gate has 3 to 7 attempts per
   cell: it is a filter, not a result.
   Main set: levels 1, 5, 2, 6 and 3, seeds 1 to 3: 60 problems. Level 3 turned out not to be hopeless (Fable 1 of
   4), so it stays as the hard end; level 4 stays dropped as written in deviation 2, so the main set has no level
   that nobody solves. B_max and all claims are computed on these 60 problems.
