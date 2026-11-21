# CG1 result: the CIFAR speedrun is not a settings task, but it does not rank loop designs either

Nov 14 2026. Plan: `loop/design/CG1_PREREG.md` (frozen before the run). Code: `loop/cifar/`. Numbers:
`loop/runs/CG1/report.txt`, `loop/runs/CG1/confirmation/summary.json`.

## Why this was run

Cell N of P2B showed that on the GPT task a random search over eight settings does as well as the all-Sonnet loop,
so that task cannot rank loop designs. CG1 asks whether a harder task can: the record script of the CIFAR-10
94 percent speedrun, whose settings are already tuned. It is a gate on the task, not a routing result.

## What was run

Same loop engine as P2 and P2B (planner, coder, repairer, accept rule, 20 slots). One slot = one job on a lab
A4000: 24 timed training runs of the candidate script.

Score, lower is better: cost = ln(run time / 7.60 s) + (94.17 - accuracy) / 1.38. The baseline costs 0. A gain of
0.05 means "worth a 5 percent speedup at equal accuracy". Example: a script that runs in 6.84 s (ln 0.9 = -0.105)
and loses 0.07 points of accuracy (+0.05) has a gain of 0.055.

| Cell | Who proposes the edits | Loops |
|---|---|---|
| N | random search over the script's 13 settings, no model | 12 |
| SS | Sonnet as planner, coder and repairer | 6 |
| HH | Haiku as planner, coder and repairer | 6 |

All 24 loops finished (Nov 13 19:09 to Nov 14 02:24 EST). Then every final script was scored twice on held-out
images the loops never saw (the sealed half of the test set), next to 10 fresh baseline jobs.

## Result

| Cell | Mean held-out gain | 95% interval | Mean gain the loop saw itself | Model cost per loop |
|---|---|---|---|---|
| N (random search) | 0.000 | -0.008 to 0.008 | 0.018 | 0 USD |
| SS (Sonnet) | 0.019 | 0.011 to 0.027 | 0.091 | 1.49 USD |
| HH (Haiku) | 0.034 | 0.015 to 0.053 | 0.108 | 0.28 USD |

| Check | Question | Rule | Value | Verdict |
|---|---|---|---|---|
| A headroom | Does a strong loop find a real gain? | gain(SS) at least 0.05 | 0.019 (above 0 with p = 0.005) | **fail** |
| B menu | Does a settings search get it too? | gain(N) at most a third of gain(SS) | ratio 0.01, p = 0.006 | **pass** |
| C one trick | Is the gain one known move? | largest idea at most 60 percent | 61 percent | **fail** |
| D gap | Does the cheap model trail the strong one? | gain(HH) at most two thirds of gain(SS) | ratio 1.82 | **fail** |

Predictions written beforehand: A 0.75, B 0.85, C 0.5, D 0.5. A went against the prediction.

**Outcome by the plan's table: A fails, so no routing study is built on this task as it stands.**

## What happened inside the loops

Every model loop accepted one edit (HH_06 two). All accepted edits:

| Idea | Sonnet loops | Haiku loops |
|---|---|---|
| Train at lower resolution (24 or 28 pixels) for the first 20 to 50 percent of steps | 3 of 6 | 6 of 6 |
| Stop the Muon helper from being recompiled at every step (a real speed bug in the record script) | 2 of 6 | 0 of 6 |
| No augmentation in the last epoch | 1 of 6 | 0 of 6 |

- **Settings search finds nothing here.** 240 random moves, 5 accepted, held-out gain 0.000. This part worked: the
  task is not a short menu.
- **The model loops mostly found one known trick.** Low resolution early in training is a published move for this
  benchmark. Sonnet also proposed it in every loop; in three of them its version was rejected.
- **Haiku did at least as well as Sonnet,** at a fifth of the model cost. On this task there is nothing for a
  router to recover.
- **The one thing only Sonnet found** is the recompilation bug, in 2 of 6 loops. That is a real finding about the
  record script, and the only sign of a strong-model advantage. Two loops are too few to claim it.
- **The loops' own numbers were four to five times too high.** Loops saw gains near 0.10; held out, the same
  scripts gain 0.02 to 0.03. The low-resolution scripts are about 10 percent faster but lose about 0.13 points of
  held-out accuracy, which the score prices at 0.09. An edit is accepted when its validation accuracy happens to
  come out high, so accepted edits look better than they are. A simulation of the accept rule with no true effect at all gives a
  loop-reported gain of 0.018 and an accept in 41 percent of loops; cell N showed 0.018 and 5 of 12. On the GPT task
  of P2B, where real gains were about 8 noise sds, there was no such inflation (loops reported 0.037 on average,
  held out 0.039). So the inflation is a property of gains near the noise level, and there the in-loop numbers are
  not evidence.

## What this means for the thesis

1. Two tasks have now failed to rank loop designs, for different reasons. The GPT task was too easy (a menu wins).
   The CIFAR task resists the menu, but 20-slot loops by either model mostly retrieve a known trick, the gains are
   small next to the noise, and the cheap model is as good as the strong one.
2. So the routing result of P2 and P2B stands only with its limit: the strong planner's value there was knowing the
   right move in advance. CG1 adds that when the move is common knowledge, the cheap model knows it too.
3. Neither loop produced a new idea. On both tasks the proposals were recall of known practice. A task that tests
   ideas needs (a) good moves that are not in the models' memory and (b) a score with little noise, so that small
   real gains are kept and lucky ones are not.

## Limits

One task, one starting script, 6 loops per model cell, 20 slots per loop. Each held-out gain rests on 2 jobs per
script against 10 baseline jobs (job noise sd 0.03), so single loops are uncertain by about 0.02; cell means are
better. Idea labels in check C are mine, from the diffs. Check C failed by one point (61 against 60 percent). Longer
loops or more runs per job might show more; this gate does not test that.

## Run notes

- Jobs: 543 scored, 21 candidate crashes (18 of them in Haiku loops), 92 "GPU busy" reports. All 92 were my own
  confirmation jobs starting within a second of the previous one leaving the GPU; none was inside a loop and none
  ran a candidate. Fixed and recorded as deviation 1 in the plan.
- Shared machines: nobody else was logged in on the five lab machines. No kill and no memory error. Peak host
  memory of any job: 2.0 GB. hannah was not used.
