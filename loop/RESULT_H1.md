# H1: Gemma against Haiku, by role

Nov 15 2026. Plan frozen before launch: `loop/design/H1_PREREG.md`. Loops: `loop/runs/H1` (42 new loops, 40 slots
each, launched 12:09 EST, finished 19:13). Anchors HH and N: `loop/runs/BG1_R`. Analysis: `loop/bench/analyze_h1.py`,
output `loop/runs/H1/analysis.txt`. No deviations.

## The question

BG1 showed Haiku does as well as Sonnet on these tasks, so there was nothing to route between them. H1 moves one
step down: Gemma 12B, an open model served on one lab GPU, against Haiku. If the all-Gemma loop trails the all-Haiku
loop, which role do you have to pay for to get the gap back: the planner (who lists the ideas) or the coder (who
writes the edit)?

Cells are named planner then coder. GG = Gemma in both. HG = Haiku plans, Gemma codes. GH = Gemma plans, Haiku
codes. HH = Haiku in both. Analyst, critic and voter are the same Claude models everywhere.

Metric. Progress is the held-out score: 0 = the starting script, 1 = the reference (leaderboard first place, or the
packing reference). Gap recovered by a cell X = (progress(X) - progress(GG)) / (progress(HH) - progress(GG)).
Example, packing: GG 0.059, HH 0.774, HG 0.529 gives (0.529 - 0.059) / (0.774 - 0.059) = 0.66, so the Haiku planner
alone gets back two thirds of what all-Haiku adds. Extra cost share is the same ratio on model dollars, and the
thesis metric is gap recovered minus extra cost share.

## Run health

All 42 loops ran 40 charged slots. No authentication failure, no memory event. 155 model calls failed, all of them
Gemma as coder or repairer returning text that was not valid JSON; that is model behaviour and is counted, not
excused. Held-out scoring finished for all 42 scripts.

## Results

Mean held-out progress (loops per cell: 2 per drug task, 3 per packing task):

| Task | N (no model) | GG | HG | GH | HH | SS |
|---|---|---|---|---|---|---|
| bbb | 0.069 | 0.674 | 0.577 | 0.711 | 0.722 | 0.822 |
| caco2 | 0.076 | 0.583 | 0.738 | 0.634 | 0.954 | 0.858 |
| clearance | 0.601 | 0.457 | 0.605 | 0.671 | 0.597 | 0.742 |
| lipophilicity | 0.155 | 0.509 | 0.555 | 0.737 | 0.942 | 0.854 |
| 26 circles | 0.000 | 0.000 | 0.205 | 0.561 | 0.728 | 0.561 |
| 32 circles | 0.000 | 0.118 | 0.854 | 0.745 | 0.819 | 0.551 |

| Family | | GG | HG | GH | HH |
|---|---|---|---|---|---|
| Drug tasks | progress | 0.556 | 0.619 | 0.688 | 0.804 |
| | model USD per loop | 0.26 | 0.30 | 0.35 | 0.45 |
| Packing | progress | 0.059 | 0.529 | 0.653 | 0.774 |
| | model USD per loop | 0.26 | 0.27 | 0.39 | 0.49 |

Frozen checks:

| Family | G: all-Gemma trails Haiku | P: planner alone recovers half | C: coder alone recovers half |
|---|---|---|---|
| Drug tasks | FAILS by a hair: GG / HH = 0.69, bar 0.67 (the difference itself is clear, p = 0.004) | not read (descriptive: 0.25, interval -0.30 to 0.59) | not read (descriptive: 0.53, interval 0.29 to 0.68) |
| Packing | HOLDS: GG / HH = 0.08, p = 0.003 | HOLDS: 0.66 (0.36 to 1.01) | HOLDS: 0.83 (0.59 to 1.14) |

Thesis metric in packing: Haiku planner with Gemma coder 0.65 (interval 0.27 to 1.05), because it recovers 0.66
of the gap for 1% of the extra cost. Gemma planner with Haiku coder 0.29 (-0.03 to 0.70): it recovers 0.83 but
costs 54% of the extra dollars. In the drug tasks both are near zero (0.04 and 0.06).

Idea-gap predictions (frozen from the scores before any Gemma loop ran):

| Prediction | Result |
|---|---|
| 1. With Gemma planning and Haiku coding, progress is at least two thirds of all-Haiku in 5 of 6 tasks | HOLDS, exactly 5 of 6 (caco2 misses: 0.634 against 0.954) |
| 2. Pooled, the coder swap recovers more than the planner swap | HOLDS: 0.71 against 0.49 |
| 3. The score orders the tasks (Spearman with HH minus GH positive) | HOLDS weakly: 0.37 over six tasks |

By the frozen rule the score counts as supported for this pair of models.

## What it means

1. **Packing gives the first clean routing-by-role result of the thesis.** A settings search gets 0 there, all-Gemma
   gets 0.06, all-Haiku 0.77. Paying for one Haiku role gets most of it back: planner alone 0.53, coder alone 0.65.
2. **Per dollar the planner is the role to buy.** The planner makes few calls, so the Haiku planner adds one cent
   per loop; the Haiku coder adds 13 cents. This is the same shape as P2B (strong planner, cheap coder), now on a
   task where random search finds nothing.
3. **In absolute terms the coder matters more,** as predicted: Gemma's code is unusable in about half of its slots
   (invalid edit or crash in 46 to 55% of drug slots and 58 to 63% of packing slots, against 1 to 7% for Haiku).
4. **Drug tasks: Gemma alone already gets 69% of Haiku's progress.** The known recipes are within its reach, so
   there is less to recover and neither swap pays for itself.
5. **Haiku refuses many of Gemma's ideas.** With Gemma planning, the Haiku coder declined 354 ideas before any run
   (duplicates and ideas it judged harmful); these cost model calls but no slots.
6. Two loops beat the rounded packing reference for 32 circles (2.93896 with HG, 2.93822 with GH). The published
   best is 2.93957, so neither is a record.

## Limits

- 2 or 3 loops per cell and task. The two swaps do not differ significantly from each other (p = 0.12 drug, 0.24
  packing), and the packing planner result rests on 6 loops, two of which found nothing on 26 circles.
- Drug check G missed its bar by 0.02; with other loops it could fall either way.
- Model dollars are small next to the fixed roles (Sonnet analyst, Opus critic), and Gemma is priced at a nominal
  list rate; GPU time for serving it is not counted.
- HH and N anchors ran a day earlier with two loops per machine.
- The planner and coder numbers were predicted after seeing BG1 and a 6-loop pilot.
- One open model, one quantisation, a 16k window.
- Added Nov 16 (from E1): rescoring unchanged packing scripts moves single progress values by roughly 0.1 to 0.2,
  because the packing script is time-budgeted and depends on machine load. The contrast between all-Gemma and the
  routed cells is far larger than that; the difference between the planner swap and the coder swap is not.
