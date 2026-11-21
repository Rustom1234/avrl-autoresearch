# K1: can the value of the strong model on a task be predicted from idea lists alone?

Written Nov 14 2026, while the BG1 loops were still running and before any BG1 ledger, score or confirmation
result was opened. Frozen by `K1_PREREG.sha256`.

## The idea

Four experiments so far say the same thing: routing to the strong model pays when the strong model knows a good
move that the cheap model does not, and pays nothing when the move is common knowledge. If that is the mechanism,
it can be measured before a loop is run, with a few planner calls and no experiment: ask both planners for their
first idea list several times and measure how far the strong planner's leading ideas lie outside what the cheap
planner proposes.

Searches on Nov 14 found routing by predicted correctness, by hidden-state probes and by disagreement between a weak
and a strong answer to one query (for example arXiv 2610.22245). They did not find a task-level prediction of the
strong model's value from idea coverage, made before any experiment. That is the claim to novelty, as far as those
searches go.

## The score

`loop/k1/gap.py`. Every idea (title and spec) is embedded with a small sentence-embedding model. For each of the
first 5 ideas of each strong list: d_cheap = distance to the nearest idea in the cheap planner's lists, d_strong =
distance to the nearest idea in the other lists of the strong planner (pools of equal size).

    gap = mean d_cheap / mean d_strong

Worked example: if the strong planner's leading ideas sit 0.18 away from the nearest Haiku idea and 0.05 away from
the nearest idea in another Sonnet list, the gap is 0.18 / 0.05 = 3.6: Sonnet keeps leading with something Haiku
does not say. A gap near 1 means Haiku covers Sonnet's leading ideas as well as Sonnet repeats itself.

## Calibration on the two tasks whose outcome is known (looked at before writing the rule)

| Task | Lists | gap (Sonnet against Haiku) | Known outcome |
|---|---|---|---|
| GPT training (P2B) | 12 per model | 3.69 (against the Qwen and Gemma planners: 4.8 to 5.8; Sonnet against itself: 1.07) | weak planners got 0.2 to 0.4 of the strong planner's gain |
| CIFAR speedrun (CG1) | 6 per model | 1.67 | Haiku did at least as well as Sonnet |

Rule fixed from these two points: threshold 2.48 (their geometric mean). gap above 2.48 predicts that the cheap
model trails the strong one on that task; gap below predicts that it does not.

## Predictions for the six BG1 tasks

Lists: 3 per model and task, written outside any loop with the same prompt a loop's first planner call gets
(`loop/k1/lists.py`, output `loop/k1/bg1_lists.jsonl`, scores `loop/k1/bg1_gap.json`).

| Task | gap | Prediction: does Haiku trail Sonnet? |
|---|---|---|
| clearance_hepatocyte_az | 2.83 | yes |
| bbb_martins | 2.81 | yes |
| caco2_wang | 2.59 | yes |
| lipophilicity_astrazeneca | 1.98 | no |
| pack32 | 1.89 | no |
| pack26 | 1.79 | no |

All six lie between the two calibration points, so these are weak predictions, and the three "yes" are close to the
threshold.

## Test (after BG1's confirmation)

Outcome per task: mean progress(SS) minus mean progress(HH), from `loop/bench/finish.py` (held-out scores).
"Haiku trails" means mean progress(HH) <= two thirds of mean progress(SS), the same rule as BG1's check D, applied
per task; a task where neither cell makes progress above 0.05 is scored "no".

1. Direction: number of the six yes or no predictions that are right. Chance is 3. Reported with the exact
   binomial probability of doing at least that well.
2. Ordering: Spearman correlation between gap and the outcome over the six tasks, with a one-sided permutation p.
3. Read as support only if at least 5 of 6 are right or the correlation is at least 0.77 (p about 0.05 for six
   points). Otherwise the score, as built, does not predict, and that is reported.

Predictions (probability): at least 5 of 6 right, 0.3; correlation positive, 0.6.

## Limits known in advance

- Two calibration points and six test tasks. This can refute the score or keep it alive; it cannot establish it.
- Embedding distance sees wording as well as content.
- 2 or 3 loops per cell and task make the outcome itself noisy.
- I saw the BG1 pilot before this (4-slot loops on caco2 and 26 circles: Sonnet accepted edits on both, Haiku one on
  caco2 and none on 26 circles). The score and the rule were not adjusted to it; the pilot points the other way for
  26 circles.
- Calibration lists for the GPT task came from planners that saw only the settings block; all others saw the file.

## Deviations

1. **Void first scoring seen (Nov 14, written 18:40 EST).** BG1's first scoring was run on loops damaged by a login
   expiry (BG1_PREREG.md, deviation 1): 23 of the 28 model loops lost some or all of their model calls. I saw that
   void table, and scored these predictions against it (2 of 6, correlation -0.26), before noticing the failure.
   That scoring is void, since most of what it compared was loops with no model. The predictions, the threshold and
   the test above are unchanged and will be scored once against the rerun (`loop/runs/BG1_R`). The test is no
   longer blind in the strict sense: I have seen held-out scores of 5 clean model loops and of the cut-short
   attempts of 10 others.
