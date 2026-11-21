# SR3: does an "out of ideas" signal from the cheap model's own proposals work?

Written 2026-11-19, about 11:05, by the coding agent, before computing any of the statistics below. Rustom asked
for a deep prior-art review and a check that the idea works before anything is built. He has ruled out Fable
sessions. This test uses stored sessions only: no model session of any kind.

## Prior-art review (18 searches, Nov 19)

Idea A, the "parts law" (one try needs all n parts at once, q^n; a chain keeps found parts, so the loop removes
the power). Verdict: not new as an idea.
- Exponential decay of one-shot success with task length: Ord 2025, "Is there a half-life for the success rates
  of AI agents?" (arXiv 2505.05115).
- Per-step error correction turns the exponential into near-linear cost, and small models become the most cost
  effective: MAKER, "Solving a million-step LLM task with zero errors" (arXiv 2511.09030).
- A weaker model with more samples and an exact checker beats a stronger model's single sample: "Large Language
  Monkeys" (arXiv 2407.21787). Loops with a verifier converge: the 4/delta bound (arXiv 2512.02080).
What is left for us: a measurement of this in a discovery loop with exact scoring, and its consequence for
routing. That is an explanation and a measurement, not a new theory.

Idea B, the Good-Turing "out of ideas" certificate. Verdict: every component exists; the combination was not
found.
- Good-Turing on LLM samples: McCabe et al., ICLR 2026 (arXiv 2509.14478); SHADE (arXiv 2604.19162).
- A missing-mass stopping rule for sampling an LLM: Conformal Prediction Beyond the Seen, NeurIPS 2025 (arXiv
  2506.05497): query until the missing mass stops falling.
- "No success so far, so the Good-Turing estimate bounds the chance that the next sample succeeds": STADS,
  Boehme 2018 (fuzzing), with a known weakness when samples depend on each other (Lee and Boehme, ICSE 2026).
- Hand off from a cheap to a strong model when the cheap one stagnates: AgentCollab; a Bayesian optimal-stopping
  escalation preprint; SWE-Router (reads the partial trajectory).
- Budget across problems between resampling and rerouting with a verifier: "Resample or Reroute?" (arXiv
  2607.08665); SPLIT (arXiv 2610.23255).
Not found: the unseen-proposal rate used as a bound on a cheap loop's chance of success inside a budget rule
across research problems. If it works it is an application that recombines known pieces. It must be described
that way.

## Test (stored Haiku chains of `loop/runs/SR2` and `loop/runs/SR2L4`)

Every tool call of every chain step is stored (`work/calls.jsonl`). Two definitions of "something new", fixed now:
- Functions: every candidate passed to `evaluate`. Two candidates are the same function when their fitted training
  error agrees to 2 decimals of log10 (the fit is deterministic). Good-Turing estimate at step j: candidates whose
  function was seen exactly once, divided by all candidates of steps 1 to j. With an exact checker and no success
  so far this is an upper bound on the chance that the next candidate solves.
- Ideas: building blocks in the code of all tool calls (a numpy math function with the inputs inside it, a ratio
  with an input in the denominator, a power of an input). Breadth = distinct blocks so far. Recent novelty =
  blocks first seen in the last two steps.

Claims:
| Claim | Passes if |
|---|---|
| E1 signal | among chains of levels 2, 6, 3 unsolved after 4 steps, predicting "solved by step 12": a logistic model on (best error, Good-Turing estimate, breadth, recent novelty), fitted leaving one problem out, has AUC at least 0.05 above the same model on best error alone, with the 95% interval of the difference (resampling problems) above 0 |
| E2 certificate | the Good-Turing bound is below 0.5 for at least half of the chain steps (otherwise it says nothing), and in no bin of the bound is the observed solve rate of the next step above the bound |
| E3 reach (descriptive, 4 positives) | at step 12 on level 4, the signals rank the 4 chains that later solved above the rest |

My guesses: E1 passes 0.25; E2 passes 0.2 (I expect most candidates to be new functions, so the bound is near 1).

Kill rule: if E1 and E2 both fail, the certificate is dropped and no replay is built. If E1 passes, the rule of
SR2 is replayed with the signal added, at real prices and the three what-if prices, same metric as SR2.

## Deviations

None. Run once on 2026-11-19 at 11:09. E1 fail, E2 fail; the kill rule was applied (no replay built).
Result: `loop/RESULT_SR3.md`.
