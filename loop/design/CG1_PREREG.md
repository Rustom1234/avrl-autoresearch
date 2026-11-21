# CG1: is the CIFAR speedrun a task that can rank loop designs?

Written Nov 13 2026, before any loop of this run started. Frozen by `CG1_PREREG.sha256`.

## Why

Cell N of P2B (`loop/RESULT_P2B.md`) showed that on the GPT task a random search over eight settings, with no
language model, recovers 0.90 of the gap between the all-cheap and the all-strong loop. That task rewards knowing one
trick (cut the batch size), so it cannot tell a good loop design from a short menu. A routing study needs a task
where (1) retuning settings does not help, (2) code-level ideas do, (3) the gain is not one known trick, and (4) the
strong model does better than the cheap one. CG1 tests a candidate task for these four properties before any routing
study is run on it. It is a gate, not a routing result.

## The task

Starting file: the current record script of the CIFAR-10 94 percent speedrun, rewritten only so that a harness can
time it and score it (`loop/cifar/base/train.py`; same model, optimizer, schedule, augmentation and prediction code).
Its settings are tuned already. In the Nov 10 speedrun lab, batch size, width and freezing changes did nothing on
this script, and two structural changes together made it 21.7 percent faster (`~/speedrun/RESULT.md`).

One job (`loop/cifar/harness.py`) = one lab RTX A4000, alone on its GPU: compile (not timed), then 24 timed runs.
The harness times train() plus predict() itself and computes accuracy itself from the returned logits. The 10,000
test images are split once into a validation half (the score loops see) and a sealed half (audits and confirmation).

Score, lower is better:

    cost = ln(median run time / T_REF) + (A_REF - mean accuracy in percent) / K

T_REF and A_REF are the baseline's time and accuracy. K is how many points of accuracy the baseline gains per unit
of ln(time) when its epochs are scaled (measured at 0.9, 1.0 and 1.1 times the epochs). Worked example with round
numbers: if K is 1.5, a script that is 10 percent faster (ln 0.9 = -0.105) and 0.03 points less accurate
(0.03 / 1.5 = +0.02) costs -0.085: it is worth about an 8.5 percent speedup at equal accuracy. Training for more or
fewer epochs moves along the baseline's own curve and changes the cost by about zero, so the epochs setting cannot
buy a gain.

All constants (T_REF per machine, A_REF, K, the noise sd of one job) come from 25 calibration jobs run before this
file was frozen and are stored in `loop/cifar/consts.json` (sha256 in the freeze file). See "Calibration" below.

## The loop

The engine of P2 and P2B (`loop/core/arms.py`, arm D), unchanged: a planner writes an idea list, a coder implements
one idea per slot, a repairer fixes crashes and static-check failures, an accept rule keeps an edit when
z >= 2 on one job or z >= 1.5 on two, 20 slots per loop. Task-specific parts live in `loop/cifar/task.py`:

- The program card describes this task, its score and its rules.
- The planner sees the whole file (in P2 it saw only the settings block), because the ideas wanted are code changes.
- A static check enforces what can be seen in code: the three entry functions, no file access, no seeding, a short
  import list.
- The cumulative-gain audit threshold is 5.5 noise sds, the same ratio as in P2B.
- Each loop runs all its jobs on one lab machine; machine time offsets are frozen in consts.json.

Cells:

| Cell | Proposals come from | Loops |
|---|---|---|
| N | random search: each slot moves one of the 13 named settings of the script up or down by a factor 4/3 (momenta on 1 minus the value, translation by 1 pixel); no model | 12 |
| SS | Sonnet as planner, coder and repairer | 6 |
| HH | Haiku as planner, coder and repairer | 6 |

In SS and HH the analyst is Sonnet, the critic Opus and the voter Haiku. 6 blocks, each with two N loops, one SS and
one HH in shuffled order (`loop/cifar/make_plan.py --seed 11 --blocks 6 --cells N,SS,N,HH`), run on five lab
machines, one loop per machine at a time.

The cheap tier is Haiku alone, not the open-weight models of P2B: those are served from lab GPUs, and here the lab
GPUs are the stopwatches.

## Outcome

After all loops: the confirmation block (`loop/cifar/confirm.py`). Every final script that differs from the baseline
gets 2 fresh jobs on 2 machines, and the baseline gets 2 jobs on each machine in the same session, all scored on the
sealed half. gain of a loop = mean baseline sealed cost minus mean sealed cost of its final script. A loop that
accepted nothing has gain 0. A gain of 0.05 means "worth a 5 percent speedup at equal accuracy".

## Checks and rules

| Check | Question | Passes when |
|---|---|---|
| A headroom | Does a strong loop find a real gain in 20 slots? | mean gain(SS) >= 0.05, and gain(SS) > 0 by a one-sided one-sample t test, p < 0.05 |
| B menu | Does a settings search with no ideas get it too? | mean gain(N) <= one third of mean gain(SS), and gain(SS) > gain(N) by a one-sided permutation test, p < 0.05 |
| C one trick | Is the gain one known move? | over the SS loops, no single idea accounts for more than 60 percent of the summed in-loop gain of accepted edits |
| D gap | Is there something for a router to recover? | mean gain(HH) <= two thirds of mean gain(SS) |

For C, I label each accepted SS edit by its idea (for example "train at lower resolution early") from its diff, and
list every label with its share in the result. The in-loop gain of an accepted edit is the drop of the incumbent's
mean adjusted cost at that accept.

What follows:

| Result | Action |
|---|---|
| A fails | The task gives too little in 20 slots. Report. No routing study on this task as built. |
| B fails | The task is again too simple. Rejected. Report. |
| C fails, A and B pass | Add the dominant idea to the starting script, recalibrate, and run the gate again from there (one time). |
| D fails, A to C pass | The task is hard enough, but Haiku is as good as Sonnet on it. Report that as the result; a routing study would need a weaker cheap tier. |
| all pass | Build and preregister the routing study on this task (cells SS, HH, ST, TS, F, N). |

Predictions (probability the check passes): A 0.75, B 0.85, C 0.5, D 0.5.

## Integrity

- The harness, not the candidate, times runs and computes accuracy; the candidate never receives test labels.
- A job is not counted (and is rerun at no charge) when another process is on its GPU or when a fixed benchmark
  workload runs more than 10 percent off that machine's calibrated speed.
- A first run that differs from the rest of its job (time by 30 percent, accuracy by 1.5 points) flags work carried
  between runs; the run is rejected.
- A run longer than 2.5 times the baseline is rejected.
- Selection inside loops uses the validation half only; every reported gain is from the sealed half, fresh seeds.

## Shared machines

Every job runs under a watchdog inside the harness that ends it quietly (as an ordinary crash result) at 9 GB of
host memory; measured peak of the baseline job is about 2 GB. One job per GPU, one compile thread. No automatic
retry of a job stopped for memory. Jobs run only on willow, walnut, pine, cherry and spruce.

## Limits known in advance

- One task, one starting script, lab GPUs (the times are A4000 times, not record times).
- The linear cost is exact only near the baseline's accuracy.
- Progressive resizing is public knowledge for this benchmark; check C exists for that reason.
- 6 loops per model cell: enough for a gate, not for a routing claim.
- The random search steps (factor 4/3) and its 13 settings were chosen by me; a different menu could do better.

## Calibration

25 jobs on Nov 13 (5 per machine: 3 baseline, 1 at 0.9 and 1 at 1.1 times the epochs), 24 runs each, files in
`loop/cifar/calib/jobs/`, computed by `loop/cifar/calibrate.py`.

| Quantity | Value |
|---|---|
| Baseline median run time, per machine | cherry 7.72 s, pine 7.60 s, spruce 7.55 s, walnut 7.57 s, willow 7.54 s |
| T_REF (geometric mean) | 7.596 s |
| Baseline accuracy, validation half / sealed half / all 10,000 | 94.17 / 93.87 / 94.02 percent |
| Accuracy at 0.9 / 1.0 / 1.1 times the epochs (all 10,000) | 93.86 / 94.02 / 94.13 percent |
| K | 1.382 points of accuracy per unit of ln(time) |
| sd of one baseline job's cost (15 jobs) | 0.0224 |
| Noise sd used by the accept rule (1.15 times that, the P2B rule) | 0.0258 |
| Accuracy sd of one run (validation half) | about 0.17 points |
| Peak host memory of a job | 1.9 GB |
| Wall time of a job | about 3.8 minutes |

With these numbers, an edit worth a 5 percent speedup has z = 1.9 on its first job (so it is run a second time), and
one worth 10 percent has z = 3.8. The benchmark workload varied by up to 6.4 percent between calibration jobs on an
idle machine, so its tolerance is set to 10 percent: it catches a shared GPU, not small drifts.

## Pilot

Before freezing, two short loops (one N, one SS, 3 slots each, `loop/runs/CG1_PILOT`) were run to test the plumbing.
They are not part of CG1. What they showed and what was changed because of them:

- The loop process must not hold the GPU of the machine it runs on (the engine's duplicate-idea check loads a small
  embedding model). Loop processes now run without a GPU.
- Everything else ran as designed. For the record, since I saw it before freezing: the SS pilot's first idea list
  was 30 mostly small changes (prediction shortcuts, widths, schedule shapes), with no resolution change in it, and
  one edit was accepted (no augmentation in the last epoch, in-loop cost -0.046 over two jobs). The N pilot rejected
  its moves. No threshold above was changed after seeing this.

## Deviations

1. Nov 14, after all 24 loops had finished: in the confirmation block, jobs queue back to back on a machine, and a
   job that started within a second of the previous one leaving saw that job still on the GPU and reported "GPU
   busy" (92 such reports, all of my own jobs, none inside a loop; nobody else was on the machines). They were
   resubmitted as designed, but both jobs of HH_06 ran out of resubmissions. `harness.py` now waits up to 30 s for
   the GPU to clear before reporting it busy, and the confirmation command was run again, which submitted only the
   two missing HH_06 jobs. No score is affected: a "GPU busy" job never runs the candidate.
