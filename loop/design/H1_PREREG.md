# H1: on tasks a settings search cannot solve, where does a small open model fall behind, and which role fixes it?

Written Nov 15 2026, before any loop of this run started. Frozen by `H1_PREREG.sha256`.

## Why

BG1 (`RESULT_BG1.md`) found two task families that work as tests of a loop (a strong loop makes held-out progress,
a settings search does not) and found no gap between Haiku and Sonnet on them. So routing between those two models
has nothing to recover. The open question for the thesis is one step down: is there a gap between Haiku and a small
open model that runs on a lab GPU, and if so, is it the planner or the coder that has to be paid for? On the GPT
task the answer was the planner (P2, P2B).

## Tasks, loop, scoring

Exactly BG1: the six tasks, `loop/bench/` unchanged, 40 slots per loop, held-out scoring by `loop/bench/finish.py`,
progress = (start - final) / (start - reference). References stay as frozen in `tasks.json`, although the best
published value for 32 circles is 2.93957 and not 2.937 (found Nov 15); this rescales progress on that task for
every cell alike.

## Cells

Two letters: planner, then coder (the repairer goes with the coder). G = Gemma 12B (4-bit, served on pine, 16k
token window), H = Haiku. Analyst (Sonnet), critic (Opus) and voter (Haiku) are the same in every cell, as in BG1,
so GG is "cheap planner and coder", not "no Claude model at all".

| Cell | Planner | Coder | Loops |
|---|---|---|---|
| GG | Gemma | Gemma | 14, new |
| HG | Haiku | Gemma | 14, new |
| GH | Gemma | Haiku | 14, new |
| HH | Haiku | Haiku | 14, taken from `loop/runs/BG1_R` |
| N | none (settings search) | | 14, taken from `loop/runs/BG1_R` |

Drug tasks 2 loops per cell and task, packing 3. Plan: `loop/bench/make_plan_h1.py --seed 31`, order shuffled,
three loops at a time per lab machine (BG1 ran two; the jobs are not scored on wall time). Run directory
`loop/runs/H1`.

## Checks, per family

Costs are model dollars at list price (Gemma at the nominal 0.15 / 0.60 USD per million tokens of `models.json`).

| Check | Question | Holds when |
|---|---|---|
| G gap | Does the cheap pair trail Haiku? | mean progress(GG) <= two thirds of mean progress(HH), and HH above GG by a permutation test inside tasks, p < 0.05 |
| P planner | Does paying for the planner alone close it? | gap recovered by HG >= 0.5 |
| C coder | Does paying for the coder alone close it? | gap recovered by GH >= 0.5 |

gap recovered by X = (progress(X) - progress(GG)) / (progress(HH) - progress(GG)); extra cost of X is the same ratio
on model dollars; the thesis metric is gap recovered minus extra cost. Each with a bootstrap interval over loops
inside tasks. P and C are read only where G holds. If G fails in a family, the result for that family is "a 12B open
model on a lab GPU matches Haiku here", and P and C are reported as descriptive numbers.

Also reported without a pass rule: share of slots lost to code that does not run, per coder; accepted ideas.

## The idea-gap score, tested blind this time

`loop/k1/predict_h1.py` (run Nov 15 before any Gemma loop existed; lists in `loop/k1/h1_lists.jsonl`) gives the gap
between Haiku's top five ideas and anything Gemma lists:

| Task | Haiku against Gemma | For comparison, Sonnet against Haiku (K1) |
|---|---|---|
| bbb | 1.62 | 2.81 |
| pack26 | 1.67 | 1.79 |
| clearance | 1.72 | 2.83 |
| caco2 | 1.75 | 2.59 |
| lipophilicity | 2.21 | 1.98 |
| pack32 | 2.24 | 1.89 |

All six are below every Sonnet-against-Haiku score of the drug tasks, where Haiku lost nothing to Sonnet. So the
score says Gemma knows the moves. Predictions frozen here:

1. **Planner swap is cheap.** In at least 5 of 6 tasks, mean progress(GH) >= two thirds of mean progress(HH).
2. **The gap, if any, is on the coder side.** Pooled over the six tasks, gap recovered by GH > gap recovered by HG
   (read only where G holds in at least one family).
3. **Ordering (weak, the six scores are close together).** Spearman correlation between the score and
   progress(HH) - progress(GH) over the six tasks is positive.

The score counts as supported if 1 and 2 hold. If GH falls far below HH on most tasks, the score is refuted for
this pair of models, and that is reported.

My probabilities: G holds, drug 0.6, packing 0.7. Prediction 1: 0.6. Prediction 2: 0.6. Prediction 3: 0.55.

## Limits known in advance

- HH and N loops are from BG1_R (Nov 14 and 15, two loops per machine); the new cells run on Nov 15 and 16 with
  three per machine. Not randomised against each other.
- 2 or 3 loops per cell and task.
- One open model, one quantisation, one context window. Scripts that outgrow the 16k window fail for Gemma; that
  is part of what "cheap coder" means here and is counted, not excused.
- Analyst, critic and voter are Claude models in every cell.
- I have seen BG1's results and the pilot below.

## Shared machines

CPU jobs on willow, walnut, pine, cherry, spruce (16 GB address-space limit per job, quiet watchdog at 6 GB, peak
of BG1's jobs under 1 GB by the starting scripts); at most three loops per machine, each with one job of at most 8
threads. The Gemma server holds pine's GPU (14.6 GB of GPU memory, 7 GB host memory in use on pine with it
running). No automatic retry of a job stopped for memory. A watcher stops the run if the Claude login fails.

## Pilot

Before freezing, six loops of 4 slots each (caco2 and 26 circles, one loop per new cell, `loop/runs/H1_PILOT`,
on pine only) tested the plumbing. Not part of H1; nothing was changed after it. For the record, since I saw it
before freezing: all Gemma calls were served by Gemma, prompts stayed under 3,200 tokens, no Claude call failed.
With Gemma as coder, 14 of 16 slots produced nothing usable (8 edits that could not be applied, 6 crashes) and 2
were plain rejects; with Haiku as coder and Gemma as planner, all 8 slots ran and one edit was accepted on caco2.
This points the same way as predictions 1 and 2, which were written from the idea-gap scores before the pilot ran;
my probabilities above were set after it.

## Deviations

None yet.
