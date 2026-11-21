# SR2 result: an index for "keep refining, start over, pay for a stronger model, or stop"

2026-11-19. Rules written beforehand: `loop/design/SR2_PREREG.md` (07:00, before the second half of the chain data
was read). Code: `loop/sr/chain_policy.py`, `loop/sr/trie.py`. Data: `loop/runs/SR2` (chains), `loop/runs/SR1`
(fresh Sonnet and Fable attempts). Output: `loop/runs/SR2/replay_real.txt`, `replay_whatif.txt`. The replay was
run once on the real data; nothing was tuned afterwards. Rustom has not reviewed this.

## Short version

- Letting Haiku build on its own candidates changes the picture of SR1. Twelve fresh Haiku tries solved 8 percent
  of the level 2 problems; one Haiku chain of 12 steps solves 75 percent of the time for 6.5 cents. One Fable
  attempt solves 83 percent for 47 cents.
- Haiku chains alone solve 48 of the 60 problems for 9.05 USD. One Fable attempt on every problem solves 49 for
  28.48 USD. Everything stored together solves 55.
- The index rule is the best policy that does not use hindsight (area 0.792), but it is not separable from the
  simple plan "one Haiku chain of up to 12 steps, then Fable" (0.780), from a plan tuned on other problems
  (0.771), or from Haiku chains with no strong model at all (0.750). Four of the six claims fail.
- Why: at real prices there is no routing decision on this task. One more Haiku step costs about 1 cent and
  solves about 5 percent of the time even on the hardest level (about 0.25 USD per success). A Fable attempt costs
  about 1 USD per success. The rule works that out and calls Fable only after about 40 Haiku steps, which is what
  the best fixed plan does too.
- What holds up: a stalled chain can be seen in its error (AUC 0.78), the rule needs no tuning when prices change,
  and the rule gets to half of the problems for 20 percent fewer dollars than any fixed plan.

## What was run

1,468 Haiku chain steps (4 chains of up to 12 steps on each of the 60 problems), 0 failed sessions, 14.46 USD
nominal. A step is one Haiku session with the two tools of SR1 (14 tool calls) that is shown the three best
candidates of the earlier steps of its chain. Fresh attempts are those of SR1: 4 Sonnet and 1 Fable per problem.

## What chains do (all four chains, 12 steps)

Share of chains solved within k steps:

| Level | k = 1 | 2 | 4 | 8 | 12 | Mean USD of a chain | Sonnet, per fresh attempt | Fable, per fresh attempt |
|---|---|---|---|---|---|---|---|---|
| 1 | 0.94 | 1.00 | 1.00 | 1.00 | 1.00 | 0.004 | 45 of 48 | 12 of 12 |
| 5 | 0.38 | 0.69 | 0.92 | 0.98 | 0.98 | 0.020 | 25 of 48 | 12 of 12 |
| 2 | 0.00 | 0.12 | 0.40 | 0.69 | 0.75 | 0.065 | 3 of 48 | 10 of 12 |
| 6 | 0.00 | 0.08 | 0.17 | 0.40 | 0.50 | 0.093 | 2 of 48 | 10 of 12 |
| 3 | 0.00 | 0.00 | 0.00 | 0.06 | 0.21 | 0.119 | 0 of 48 | 5 of 12 |

For comparison, fresh Haiku attempts in SR1 (12 per problem, none sees another): level 2, 1 of 144; level 6, 1 of
144; level 3, 0 of 144.

Chance that step k solves when the chain was unsolved before it (solved / chains that took the step):

| Level | 1 | 2 | 3 | 4 | 5 | 6 | 7 | 8 | 9 | 10 | 11 | 12 |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 1 | 45/48 | 3/3 | | | | | | | | | | |
| 5 | 18/48 | 15/30 | 6/15 | 5/9 | 2/4 | 1/2 | 0/1 | 0/1 | 0/1 | 0/1 | 0/1 | 0/1 |
| 2 | 0/48 | 6/48 | 8/42 | 5/34 | 5/29 | 3/24 | 2/21 | 4/19 | 1/15 | 2/14 | 0/12 | 0/12 |
| 6 | 0/48 | 4/48 | 2/44 | 2/42 | 3/40 | 3/37 | 2/34 | 3/32 | 1/29 | 1/28 | 1/27 | 2/26 |
| 3 | 0/48 | 0/48 | 0/48 | 0/48 | 1/48 | 0/47 | 1/47 | 1/46 | 1/45 | 2/44 | 3/42 | 1/39 |

Chains on levels 6 and 3 are still solving at steps 9 to 12. The limit of 12 steps cut them off; they had not
stalled.

Problems solved by at least one of the four chains, by the Fable attempt, by anything stored: level 1, 12, 12, 12;
level 5, 12, 12, 12; level 2, 10, 10, 12; level 6, 10, 10, 12; level 3, 4, 5, 7. Seven problems are solved by
Fable and by no Haiku chain; six by a Haiku chain and not by Fable; five by nothing.

Whole plans worked to the end (mean over 100 random orders of the stored sessions):

| Plan | Solved of 60 | Total USD | USD per solved problem | Fable sessions |
|---|---|---|---|---|
| Always Fable | 49.0 | 28.48 | 0.581 | 60.0 |
| 3 fresh Haiku tries, then Fable (best plan of SR1) | 49.0 | 26.75 | 0.546 | 40.5 |
| One Haiku chain up to 4 steps, then Fable | 49.8 | 22.19 | 0.446 | 30.1 |
| One Haiku chain up to 8 steps, then Fable | 51.4 | 18.98 | 0.369 | 22.5 |
| One Haiku chain up to 12 steps, then Fable | 53.0 | 16.95 | 0.320 | 18.7 |
| Four Haiku chains up to 12 steps, then Fable | 55.0 | 18.34 | 0.333 | 12.0 |
| One Haiku chain up to 12 steps, no Fable | 41.3 | 3.61 | 0.087 | 0 |
| Four Haiku chains up to 12 steps, no Fable | 48.0 | 9.05 | 0.188 | 0 |

## The rule

Known theory: the index of Dumitriu, Tetali and Winkler (2003), a variant of the Gittins index, for the game
"several Markov processes, pay to advance one at a time, lowest expected cost until one reaches its target".
Grade of an option = the lowest (expected dollars) / (chance of success) over all horizons; always take the option
with the lowest grade. Across problems under one budget this is the greedy knapsack order. The chances and
dollars are estimated from a memory of the 40 problems of the other two seeds (details in the prereg). The
theorem assumes known chances and independent processes; here both are approximations, so the rule is a
heuristic built on the theorem.

## Replay at real prices

60 problems in 3 portfolios (one per seed), 100 worlds. Largest budget 28.48 USD (one Fable attempt per problem).
Area = share of problems solved averaged over all budgets up to the largest. Last column: area of the rule minus
area of the policy, 95% interval from 200 resamples of the problems.

| Policy | Solved at 0.57 USD | 1.42 | 2.85 | 5.70 | 11.39 | 28.48 | Area | USD to solve half | Rule minus policy |
|---|---|---|---|---|---|---|---|---|---|
| Always Fable | 1.0 | 2.4 | 4.9 | 9.8 | 19.5 | 49.0 | 0.401 | 17.01 | +0.386 (+0.298 to +0.477) |
| Haiku chains only (4 chains of 12) | 17.7 | 28.2 | 37.6 | 44.4 | 48.0 | 48.0 | 0.750 | 1.43 | +0.032 (-0.025 to +0.076) |
| 3 fresh Haiku tries, then Fable | 16.7 | 19.6 | 21.0 | 24.4 | 30.8 | 49.0 | 0.564 | 9.31 | +0.222 (+0.128 to +0.311) |
| Haiku chain up to 4 steps, then Fable | 17.7 | 28.2 | 30.7 | 33.7 | 38.6 | 49.8 | 0.674 | 1.93 | +0.111 (+0.044 to +0.188) |
| Haiku chain up to 8 steps, then Fable | 17.7 | 28.2 | 37.2 | 39.7 | 44.2 | 51.4 | 0.743 | 1.43 | +0.041 (-0.009 to +0.098) |
| Haiku chain up to 12 steps, then Fable | 17.7 | 28.2 | 37.6 | 42.7 | 47.2 | 53.0 | 0.780 | 1.43 | +0.005 (-0.037 to +0.051) |
| Plan tuned on the memory problems | 17.7 | 28.2 | 37.4 | 42.5 | 46.3 | 53.0 | 0.771 | 1.43 | +0.018 (-0.011 to +0.066) |
| Best plan in hindsight (a bound) | 17.7 | 28.2 | 37.4 | 44.5 | 50.4 | 55.0 | 0.814 | 1.43 | -0.026 (-0.070 to +0.008) |
| Bandit (Thompson, cost aware) | 13.0 | 21.1 | 27.0 | 33.8 | 40.8 | 53.9 | 0.686 | 3.24 | +0.098 (+0.059 to +0.141) |
| Rule without look-ahead | 21.7 | 29.7 | 36.0 | 43.1 | 45.0 | 55.0 | 0.779 | 1.34 | +0.011 (-0.023 to +0.049) |
| Rule without the error signal | 17.2 | 24.7 | 32.1 | 40.5 | 46.0 | 55.0 | 0.761 | 2.14 | +0.028 (+0.001 to +0.073) |
| **The rule** | 21.8 | 31.0 | 36.2 | 43.9 | 48.0 | 55.0 | **0.792** | **1.14** | |
| Reference: knows the problem | 29.1 | 33.7 | 38.9 | 44.7 | 52.1 | 55.0 | 0.827 | 0.54 | -0.036 (-0.093 to +0.020) |

The plans tuned on memory were "4 chains of up to 8 or 12 steps, then Fable" in every portfolio; the best plans in
hindsight were of the same kind. Inside the largest budget the rule takes 910 Haiku steps, 21 Sonnet attempts and
14 Fable attempts per world.

## What-if prices (same stored sessions)

| Setting | Always Fable | Haiku chains only | Plan tuned on memory | Best plan in hindsight | Rule without the error signal | The rule | Reference |
|---|---|---|---|---|---|---|---|
| Real prices | 0.401 | 0.750 | 0.771 | 0.814 | 0.761 | 0.792 | 0.827 |
| Fable 5 times dearer | 0.401 | 0.792 | 0.837 | 0.879 | 0.869 | 0.869 | 0.877 |
| Fable 5 times cheaper | 0.401 | 0.562 | 0.569 | 0.590 | 0.522 | 0.600 | 0.722 |
| Haiku 10 times dearer (about Sonnet's price) | 0.401 | 0.429 | 0.483 | 0.487 | 0.464 | 0.502 | 0.700 |

Rule minus the plan tuned on memory: +0.018 (-0.011 to +0.066), +0.030 (-0.011 to +0.097), +0.033 (-0.011 to
+0.079), +0.022 (-0.008 to +0.058). Rule minus the best plan in hindsight: -0.023, -0.010, +0.011, +0.015 (the
resample intervals all include 0). With Haiku 10 times dearer the rule takes 14 Haiku steps, 77 Sonnet attempts and 30 Fable
attempts per world: it drops the chains by itself. In the two settings where prices are closer together the gap
to the reference is large (0.12 and 0.20): knowing the problem's type before spending anything would be worth a
lot there, and the rule has no way to know it (it sees nothing of a problem before the first step).

## Verdicts on the written claims

| Claim | Verdict | Numbers |
|---|---|---|
| D1 the rule beats Fable alone, chains alone and the bandit | fail | beats Fable alone (+0.386) and the bandit (+0.098); not separable from Haiku chains alone (+0.032, -0.025 to +0.076) |
| D2 the rule beats every fixed plan without hindsight | fail | beats "3 fresh tries, then Fable" and "chain of 4, then Fable"; ties chain of 8 (+0.041), chain of 12 (+0.005) and the tuned plan (+0.018) |
| D3 looking ahead helps | fail | +0.011 (-0.023 to +0.049) |
| D4 the error signal helps | pass, narrowly | +0.028 (+0.001 to +0.073); what-ifs +0.000, +0.081, +0.041 |
| D5 a stall can be seen in the error | pass | AUC 0.78 (0.69 to 0.86); inside levels 2, 6, 3: 0.87, 0.69, 0.71 |
| D6 within 0.02 of the best plan in hindsight at four price settings | fail by 0.003 | -0.023 (real prices), -0.010, +0.011, +0.015 |

By the kill rules written beforehand: D1 fails, so the rule adds nothing measurable over the cheap model's chains
on this bank; D2 and D6 fail, so a fixed plan does the job here and the chain measurements stand alone. The
numbers themselves say tie, with the rule ahead of the tuned plan by 0.02 to 0.03 in all four price settings and
never separable from it with 60 problems.

## What the rule does (looked at after the replay)

| Level | Steps per problem | Solved by a Haiku step | Solved by Fable | Unsolved | Share of problems where Fable is called | Haiku steps before Fable | USD per problem |
|---|---|---|---|---|---|---|---|
| 1 | 1.1 | 1.00 | 0 | 0 | 0 | | 0.004 |
| 5 | 2.4 | 1.00 | 0 | 0 | 0 | | 0.020 |
| 2 | 13.1 | 0.83 | 0.17 | 0 | 0.17 | 37.0 | 0.230 |
| 6 | 18.1 | 0.83 | 0.17 | 0 | 0.17 | 42.0 | 0.266 |
| 3 | 43.9 | 0.25 | 0.33 | 0.42 | 0.83 | 40.8 | 1.294 |

The grade of a Haiku step stays between 0.03 and 0.2 USD per success; the grade of a Fable attempt is about 1 USD
per success. So the rule keeps refining and restarting with Haiku until the stored chains are used up, and only
then pays for Fable. Its gain over the fixed plans is at small budgets (it works on the problems that look close
first: 31.0 solved at 1.42 USD against 28.2), and it is lost again in the middle of the range.

## What this means

1. The gap between models measured per single attempt (level 2: Haiku 1 of 144, Fable 10 of 12) mostly closes when
   the cheap model iterates on its own candidates (chain of 12: 0.75 against 0.83), at about a seventh of the
   price. Independent retries do not do this (12 fresh tries: 0.08); building on candidates does.
2. With a price gap of about 48 between a Haiku step and a Fable attempt, the strong model is the last resort on
   this task, not a routing choice. Any sensible policy finds that, so policies differ little: the reference that
   knows each problem's type is only 0.035 ahead of the rule.
3. This is the same pattern as the earlier tasks (drug-property models, circle packing, CIFAR): a cheap model in a
   loop gets most of the way. Here it is measured more sharply, with an exact score and a clear single-attempt
   gap that the loop then removes.
4. The index rule is a sound default: across SR1 and SR2 and all price settings it was never separably worse than
   the best plan found in hindsight, without tuning. It is not a result by itself: simple plans tie it.

## Follow-up probe: level 4 (exploratory, after the result)

Question: is there a difficulty where the cheap chain stops working? Level 4 was dropped from the main set because
no model solved it in single attempts at the gate (Haiku 0 of 6, Sonnet 0 of 6, Fable 0 of 4). Probe: 4 Haiku
chains of up to 12 steps on the 12 level 4 problems (`loop/runs/SR2L4`, 576 steps, 6.07 USD nominal, no failed
session). My written guess was at most 2 of 48 chains solved.

| | Level 3 (main set) | Level 4 (probe) |
|---|---|---|
| Chains solved within 4, 8, 12 steps | 0, 3, 10 of 48 | 0, 0, 0 of 48 |
| Problems with a solving chain | 4 of 12 | 0 of 12 |
| Median best error of unsolved chains after 1, 4, 12 steps (log10) | -2.8, -4.1, -4.8 | -2.6, -3.7, -4.4 |
| Best error reached by any unsolved chain (log10) | -9.9 | -7.1 |
| Solved by setting (biology, chemistry, materials, physics) | 0, 5, 4, 1 of 12 each | 0, 0, 0, 0 |

Level 4 best errors by setting after 12 steps (log10, best chain and median chain): biology -7.1 and -5.0,
chemistry -5.8 and -4.5, materials -5.3 and -4.6, physics -2.7 and -2.5. The target is -10.

Reading: the cheap chain has a reach. It covers levels 1 and 5, most of levels 2 and 6, a fifth of level 3 and
none of level 4 within 12 steps. The errors keep falling slowly at level 4, but no chain finds the exact form.
What this does not show: whether any strong option solves level 4. Single Fable attempts did not (0 of 4 at the
gate) and no Sonnet or Fable chain has been run. If nothing solves level 4, the right decision there is to stop
spending, which the grade rule already expresses (the grade goes to infinity). If a Sonnet or Fable chain does
solve it, level 4 is the one place on this task where a routing decision exists.

### Handoff to a strong model at level 4 (8 sessions, exploratory)

On the 4 level 4 problems of seed 1, chain 0 after its 12 Haiku steps was handed once to Sonnet and once to Fable
(each sees the chain's three best candidates and gets 14 tool calls). My written guess: Sonnet 0 of 4, Fable 1 of
4.

| Problem | Best Haiku error after 12 steps | Haiku step 13 | Sonnet handoff | Fable handoff | Fable USD |
|---|---|---|---|---|---|
| biology | 3.0e-06 | 1.3e-07 | 1.4e-07 | **2.4e-23 (solved)** | 1.76 |
| chemistry | 1.2e-03 | 2.1e-03 | 2.9e-03 | **2.2e-23 (solved)** | 1.48 |
| materials | 2.2e-05 | 7.1e-05 | 4.3e-04 | 5.8e-05 | 1.78 |
| physics | 1.6e-03 | 5.6e-03 | 2.0e-02 | **2.8e-23 (solved)** | 1.44 |

(Best Haiku error is on the training data; the other columns are validation error. Target 1e-10.)

Fable solved 3 of 4, exactly (the error outside the training range is also about 1e-23, so it found the law, not
a fit). Sonnet did no better than one more Haiku step. Cost: Sonnet 0.47, Fable 6.45, total 6.91 USD nominal,
which is over the 5 USD cap I had set (the cap is checked when a session ends and the four Fable sessions ran at
the same time; a Fable session at level 4 costs 1.4 to 1.8 USD, three to four times a Fable attempt on the main
set).

Reading, with 4 problems and one chain each: level 4 is a place where the cheap chain fails (0 of 48 chains) and
a strong model succeeds, so a real decision exists there: when to stop the cheap chain and pay about 1.6 USD for
Fable. Per success that is about 2.2 USD by handoff. For the Haiku chain, 0 of 48 only says its rate is below
about 6 percent per chain, so its cost per success is above about 2 USD: not yet separated from Fable on cost.
Not measured: Fable alone at level 4 without a Haiku chain first (at the gate, on other problems: 0 of 4), so it
is open whether the chain's candidates help Fable or Fable alone is enough.

### The same Haiku chains continued to 24 steps (exploratory)

561 more Haiku steps, 5.95 USD nominal, no failed session. My written guess: at most 3 of 48 chains solved, none
in physics.

- Solved within 12, 16, 20, 24 steps: 0, 0, 2, 4 of 48 chains. The guess of at most 3 missed by one; none in
  physics held.
- The 4 solves are on 2 of the 12 problems: three chains on the biology problem of seed 1 (steps 19, 20, 21) and
  one on the chemistry problem of seed 2 (step 21).
- The typical chain does not move: median best error (log10) -4.4 after 12 steps and -4.5 after 24. Best error
  of any chain in physics after 24 steps: 3e-4.
- Cost: 12.02 USD for the 48 chains of 24 steps, so about 3.0 USD per solved chain.

Level 4 on the 4 problems of seed 1, side by side:

| Problem | 4 Haiku chains of 24 steps (about 1 USD) | One Fable handoff after 12 Haiku steps (about 1.7 USD) |
|---|---|---|
| biology | solved (3 of 4 chains) | solved |
| chemistry | not solved | solved |
| materials | not solved | not solved |
| physics | not solved (best 3e-4) | solved |

Reading: at level 4 the two options cost about the same per success (Haiku chain about 3.0 USD, Fable handoff
about 2.3 USD including the 12 Haiku steps) but they solve different problems: Haiku's solves sit on a few
problems, and Fable reaches problems where 96 Haiku steps did not. This is the first place in the program where
"which model, and when to switch" is a real decision and not settled by price alone. It rests on 4 handoff
sessions, so it is a lead, not a result.

## Limits

- Half of the chain data was seen before the rules were written (stated in the prereg).
- The strong models get fresh single attempts only. A Fable chain, or Fable shown a Haiku chain's candidates, is
  not measured (9 problems tried earlier: 8 solved against 4 fresh, a lead only). At about 0.8 USD per such
  session it would have to beat a Haiku step's 0.25 USD per success, which looks unlikely to change the order of
  spending, but it could raise the number solved at the top end.
- Chains were cut at 12 steps while still solving on levels 6 and 3.
- One task family made by us, Claude models only, nominal dollars, 60 problems (differences under about 0.05 in
  area cannot be resolved).
- The index is known theory (2003). Related uses: reservation-price rules for LLM cascades (arXiv 2606.07392;
  Pandora's AI Model Routing Box, NeurIPS 2026), a Gittins index as a stopping rule in LLM tree search (arXiv
  2510.20272), a model of retry against clean restart (arXiv 2605.08563). I found no use of the Markov-process
  index for refine, restart and escalate decisions across research problems; that is an absence in my searches,
  not a proof, and here it would be an application of a known index, with a tie as its result.

## Cost

Nominal dollars for SR2: Haiku chains 14.46 (1,468 steps); 9 Fable sessions shown a chain's candidates 7.13
(exploratory, before the prereg); level 4 probes 18.93 (1,137 Haiku steps
12.02; 4 Sonnet sessions 0.47; 4 Fable sessions 6.45, which put that run 1.91 over its 5 USD cap). Replay: no
model sessions. No kill, no memory error,
no failed session; highest load on spruce about 7 of 32 threads during chain collection.
