# SR4 pilot: how many bits of advice does the cheap model need?

Written 2026-11-19, 11:22, by the coding agent, before any hinted session was run. Haiku only.

## Idea

The gap between a cheap and a strong model is usually stated in accuracy points. Here it can be stated in bits:
how much information about the answer must be handed to the cheap model before it finishes the job. In this task
that can be counted exactly, because a hidden law is built from known pools of factors (`loop/sr/gen.py`): naming
a term's structure takes log2(number of choices) bits. A level 4 law holds 20 to 25 bits of structure in all.

Known, not ours: a strong model sending short, length-limited advice to a weak model that does all the work
(Redwood Research, "Untrusted advice for AI control"; "Ask Your Advisor", arXiv 2610.13870; the advisor tool of
commercial APIs); graded hints built from privileged information ("Graders should cheat", arXiv 2502.10961);
rate-distortion theory and advice complexity as mathematics. Not found in my searches (an absence, not a proof):
advice measured in exact bits against a known generator, a curve of success against bits, and the comparison of
equal bits about different parts of the answer.

## Pilot

The 4 level 4 problems of seed 1. For each, six versions of the problem text (`loop/sr/advice.py`), each run as 2
fresh Haiku chains of up to 4 steps (`loop/runs/SR4`, cap 4 USD nominal):

| Rung | Hint | Bits |
|---|---|---|
| G | only the grammar: a sum of 4 terms, each a constant times one or two factors | 0 |
| K | G plus the kinds of the two unusual factors, not which input or which term | 5 |
| U1 | G plus the exact form of one unusual term | 4 to 4.6 |
| U2 | G plus the exact forms of both terms that contain an unusual factor | 10.6 to 12.8 |
| E | G plus the exact forms of the two ordinary terms | 9.8 to 11.9 |
| ALL | G plus the exact forms of all four terms | 20.3 to 24.7 |

Baseline without any hint (stored): 0 of 16 chains solved within 4 steps, 0 of 16 within 12, 3 of 16 within 24.

My guesses, chains solved within 4 steps out of 8: G 0, K 2, U1 1, U2 4, E 1, ALL 8.

What would make this worth a real experiment: U2 clearly above E (the same number of bits is worth more when it
is about the parts the cheap model cannot find by itself), and a steep rise somewhere between 5 and 13 bits.
What would kill it: ALL does not solve (the pipeline or the idea is broken), or success rises only at ALL (the
cheap model needs the whole answer, so advice is no cheaper than doing the work).

## Pilot result (11:50, 1.88 USD nominal, 192 Haiku steps, no failed session)

Chains solved within 4 steps, out of 8: G 0, K 0, U1 0, U2 4 (on 3 of 4 problems), E 0, ALL 8 (all at step 1).
Guesses were 0, 2, 1, 4, 1, 8. Both conditions for a real experiment are met: about 11 bits on the hard terms
give 4 of 8, about 10 bits on the ordinary terms give 0 of 8, and nothing happens up to 5 bits.

## Stage 1 (added 11:55, before running; Haiku only, cap 5 USD nominal)

Question: does "equal bits are not equal" hold beyond 4 problems, and at a second level? Runs, each cell 2 fresh
Haiku chains of up to 4 steps:
- level 4, the 8 problems of seeds 2 and 3: rungs U2 and E;
- level 3, all 12 problems: rungs G, U2 and E. A level 3 law has one term with an unusual factor, so U2 there is
  that single term (4 to 4.6 bits) and E is the two ordinary terms (9.8 to 11.9 bits): the control gets more bits.

Claim S1, fixed now: at each level the share of chains solved within 4 steps is higher under U2 than under E,
with the 95% interval of the mean difference per problem (resampling problems) above 0. Level 4 uses all 12
problems (pilot included).
My guesses, share of chains solved within 4 steps: level 4 U2 0.40, E 0.05; level 3 U2 0.60, E 0.15, G 0.05.

## Stage 1 result (12:38, 3.98 USD nominal, Haiku only, no failed session)

Share of chains solved within 4 steps (24 chains per cell; guesses in brackets):
- level 4: U2 0.42 (0.40), E 0.04 (0.05). U2 minus E per problem +0.375 (95% +0.167 to +0.625), ahead on 6 problems,
  behind on 0. Claim S1 passes at level 4.
- level 3: U2 0.63 (0.60), E 0.46 (0.15), G 0.04 (0.05). U2 minus E per problem +0.167 (95% -0.167 to +0.500).
  Claim S1 fails at level 3: the ordinary-terms hint helped far more than I guessed.
Write-up: `loop/RESULT_SR4.md`.

## Deviations

None.
