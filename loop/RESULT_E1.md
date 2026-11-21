# E1: do loops find more between slot 40 and slot 80?

Nov 16 2026. Plan frozen before launch: `loop/design/E1_PREREG.md`. Loops: `loop/runs/E1` (the 28 Sonnet and Haiku
loops of BG1, continued with full state from slot 40 to slot 80; launched Nov 15 12:09 EST, finished Nov 16 10:30).
Analysis: `loop/bench/analyze_e1.py`, output `loop/runs/E1/analysis.txt`. No deviations.

## The question

At 40 slots Haiku matched Sonnet, and the accepted ideas were standard practice that both models know. One reading:
the strong model only shows its value later, once those remembered moves are used up. E1 doubles the loop length to
test that.

Metric: extra progress of a loop = held-out progress at slot 80 minus at slot 40 (0 = starting script, 1 =
reference). A loop that accepts nothing after slot 40 has extra progress 0. Next to it, the gain the loop itself
reports for those slots, on the same scale. Example: bbb_SS_2 was at 0.827 at slot 40 and 0.845 at slot 80, extra
0.019, while its own numbers claimed 0.160.

## Run health

All 28 loops reached slot 80. No failed model call of any kind, no memory event. The two Haiku loops on
lipophilicity spent their 20 late candidate slots on timeouts: their scripts at slot 40 already ran close to the
600 second job limit, so every heavier idea ran out of time.

## Results

Only 8 of 28 loops accepted anything in 40 more slots (1,120 slots in all, 8 accepts).

| Check | Question | Result |
|---|---|---|
| X more | Is there anything left to find? Needs mean extra > 0.05 | FAILS: mean extra 0.006 (p = 0.11) |
| S strong | Does Sonnet find more of it? | FAILS: Sonnet 0.012, Haiku 0.000 (p = 0.09) |
| F phantom | Are late gains mostly noise? Needs in-loop gain at least twice the held-out gain | HOLDS: loops reported 0.035, held out 0.006 |

| Cell | Progress at 40 | Progress at 80 | Gain the loops reported |
|---|---|---|---|
| Sonnet | 0.706 | 0.718 | 0.032 |
| Haiku | 0.791 | 0.791 | 0.038 |

The eight loops that changed:

| Loop | Extra, held out | Reported in-loop |
|---|---|---|
| bbb, Sonnet 1 | +0.108 | +0.145 |
| lipophilicity, Sonnet 1 | +0.041 | +0.039 |
| clearance, Haiku 1 | +0.026 | +0.055 |
| bbb, Sonnet 2 | +0.019 | +0.160 |
| 26 circles, Haiku 3 | +0.014 | +0.401 |
| caco2, Sonnet 1 | +0.013 | +0.048 |
| caco2, Sonnet 2 | -0.012 | +0.050 |
| clearance, Haiku 2 | -0.042 | +0.077 |

The late accepted ideas are still standard practice: one more blend member (kNN on Tanimoto similarity, an MLP, a
Bayesian linear model), a linear prior on logP with the trees on the residual, more fingerprint types, molecule
standardisation. Nothing new in kind.

My predictions were X 0.4, S 0.15, F 0.6; all three came out on the likelier side.

## What it means

1. **The loops plateau by slot 40.** Doubling the budget added 0.006 of progress on average. For these tasks 40
   slots is enough, and the BG1 and H1 comparisons were not cut short.
2. **Sonnet does not pull ahead later.** Five of the eight late changes were Sonnet's, and the one sizeable gain
   (+0.108 on bbb) was Sonnet's, but the difference is not significant and Haiku still leads at slot 80 (0.791
   against 0.718). The "strong model pays off once recall is exhausted" reading is not supported here.
3. **Late in a loop, its own numbers are mostly luck.** Reported gain was about six times the held-out gain, and two
   of eight accepted edits made the held-out score worse. This is the phantom-gain effect of `loop/null/RESULT_NULL.md`
   seen inside real loops: once true gains are small, the accept rule mostly picks up noise.

## A measurement note found here

Packing scripts that did not change between slot 40 and slot 80 were scored again and did not get the same
held-out number: for example 26 circles, Sonnet 1 went from 0.516 to 0.710 with an identical script. The packing
script uses a fixed time budget, so its result depends on how busy the machine is, and the progress scale spans
only 0.008 to 0.012 in sum of radii. So single packing progress values carry an error of roughly 0.1 to 0.2. This
does not touch the checks above (unchanged loops count as 0 by the frozen rule). It does widen the uncertainty of
the packing numbers of BG1 and H1 beyond what their bootstrap intervals show; the large contrasts there (random
search and all-Gemma near 0 against 0.5 to 0.8) are far outside it, the finer ones (planner swap against coder
swap) are not.

## Limits

- Same tasks and models as BG1; 2 or 3 loops per cell and task.
- Continued loops ran one per machine, a day after their first halves.
- The X test uses a sign-flip test in place of the t test named in the plan (the data are mostly exact zeros); the
  t statistic is 1.30 and gives the same verdict.
