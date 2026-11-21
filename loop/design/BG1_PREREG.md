# BG1: two more candidate tasks, taken from what other autoresearch papers used

Written Nov 14 2026, before any loop of this run started. Frozen by `BG1_PREREG.sha256`.

## Why

Two tasks have failed to rank loop designs: the GPT task (a random menu matched the strong loop, `RESULT_P2B.md`)
and the CIFAR speedrun (both models retrieved one known trick, held-out gains near 0.02, `RESULT_CG1.md`).
Rustom asked to try other hard tests and to look at what the autoresearch papers used and why AutoScientists did
well.

What the papers used (read Nov 14):

- **AutoScientists** (arXiv 2605.28655). Headline: BioML-Bench, 24 biomedical ML tasks built from scratch (9 of
  them drug-property prediction from the TDC and Polaris collections), hidden test labels, score = percentile on
  human leaderboards: 74 against 66 for the single-agent loop. Teams of 9 Sonnet agents, 4 hours and on the order
  of 75 to 100 experiments per task, about 166 USD per task against 38. Most numbers are single runs; the ablations
  are one run each. On the GPT task its gain is 0.0047 bpb over 93 experiments, read from the loop's own validation
  numbers.
- **AlphaEvolve, ShinkaEvolve, OpenEvolve.** Program search on problems with an exact score, such as packing 26
  circles in a unit square to maximise the sum of radii (best published about 2.636).
- **MLE-bench, MLGym, RE-Bench, AlgoTune.** Kaggle-style and research-engineering tasks; larger than fits here.

Two things are copied: the task type (start from a simple script, score on held-out data or exactly, measure against
what people have reached) and longer loops (40 slots instead of 20). BG1 is again a gate on the tasks, not a routing
result.

## Tasks

Family 1, drug-property prediction (the TDC ADMET benchmark group, public data, the benchmark's own scaffold split):

| Task | Molecules (train / test) | Metric | Leaderboard first place |
|---|---|---|---|
| caco2_wang | 728 / 182 | MAE | 0.256 |
| lipophilicity_astrazeneca | 3360 / 840 | MAE | 0.456 |
| bbb_martins | 1624 / 406 | AUROC | 0.924 |
| clearance_hepatocyte_az | 970 / 243 | Spearman | 0.536 |

The starting script is Morgan fingerprints with a random forest. In a loop a job scores a script by 5-fold
cross-validation over whole ring skeletons, repeated 3 times, inside the training file. The test file is used only
after the loop.

Family 2, circle packing: 26 circles and 32 circles in the unit square, maximise the sum of radii. The starting
script is random restarts of a standard constrained optimiser for the whole time budget (30 CPU seconds per run),
which already reaches 2.628 and 2.925. Best published: about 2.636 and 2.937. The packing is checked exactly.

Losses (lower is better): MAE, 1 minus AUROC, 1 minus Spearman, minus the sum of radii. Baseline loss, its noise sd
over 10 seeds and the job time per task are in `loop/bench/consts.json` (calibration of Nov 14, 60 jobs).

## The loop

The engine of P2, P2B and CG1, unchanged, 40 slots per loop. Task-specific parts are in `loop/bench/task.py`: the
program card of each task, a static check (allowed packages, no file access, the entry function), CPU jobs on the
lab machines. The planner sees the whole file. The audit tripwire is off, because large honest jumps from a simple
start are expected here; the protection is that scores are computed outside the candidate and every reported number
comes from held-out data or fresh seeds after the loop.

Cells, for every task: N = random search over the five named settings of the starting script (times or divided by
4/3), no model; SS = Sonnet as planner, coder and repairer; HH = Haiku in those roles. Analyst Sonnet, critic Opus,
voter Haiku. Drug tasks: 2 loops per cell and task (8 per cell). Packing: 3 loops per cell and task (6 per cell).
42 loops in all, order shuffled (`make_plan.py --seed 21`), two loops at a time per lab machine.

## Outcome

After the loops (`loop/bench/finish.py`): every final script and every starting script gets one sealed job. Drug
tasks: train on the whole training file with 5 seeds, score on the benchmark's test molecules, mean of the 5.
Packing: 5 fresh seeds, the same for every script.

    progress = (sealed loss of the start - sealed loss of the final script) / (sealed loss of the start - reference loss)

The reference is the leaderboard's first place, or the best published packing. 0 means no better than the start, 1
means level with the reference. Example: caco2 start MAE 0.413, final 0.300, reference 0.256 gives
(0.413 - 0.300) / (0.413 - 0.256) = 0.72. A loop that accepted nothing has progress 0.

## Checks, per family

| Check | Question | Passes when |
|---|---|---|
| A headroom | Does a strong loop get somewhere? | mean progress(SS) >= 0.25, and above 0 by a one-sided t test, p < 0.05 |
| B menu | Does a settings search get it too? | mean progress(N) <= one third of mean progress(SS), and SS above N by a permutation test inside tasks, p < 0.05 |
| D gap | Does the cheap model trail the strong one? | mean progress(HH) <= two thirds of mean progress(SS) |

Described, with no pass rule: which ideas were accepted and whether they are standard practice; the leaderboard rank
of each final drug-task script; whether any packing result beats the best published value.

A family where A, B and D all pass becomes the task of the routing study (to be preregistered separately). A family
where D fails has no room for routing between Haiku and Sonnet; that is reported as the result. If A or B fails the
family is dropped.

Predictions (probability of passing). Drug tasks: A 0.85, B 0.7, D 0.3. Packing: A 0.5, B 0.8, D 0.4.

## Limits known in advance

- The good recipes for these drug tasks (gradient boosting on several fingerprint types plus descriptors) are public,
  and no pretrained models are allowed here, so the top of the leaderboard may be out of reach and what the loops
  find may again be recall.
- Small test sets (182 to 840 molecules): single test scores are noisy, so tasks are pooled.
- The packing start is strong, so progress is measured over a small interval (0.008 and 0.012).
- 2 or 3 loops per cell and task.
- No web access and one agent per role, unlike the 9-agent teams of AutoScientists.

## Shared machines

CPU only, on willow, walnut, pine, cherry and spruce; hannah and the GPUs are not used. At most two loops per
machine, each with one job of at most 8 threads at a time. A job has a 16 GB address-space limit (a runaway
allocation fails inside Python) and a watchdog that ends it quietly at 6 GB; measured peak of the starting scripts
is 0.4 GB. No automatic retry of a job stopped for memory.

## Pilot

Before freezing, six short loops (4 slots each; caco2_wang and 26 circles, one loop per cell,
`loop/runs/BG1_PILOT`) were run to test the plumbing. They are not part of BG1 and nothing had to be changed after
them. For the record, since I saw it before freezing: in 3 or 4 slots, the Sonnet loop on caco2 accepted two edits
(a blend of a random forest with a kernel SVR, then a LightGBM member on RDKit descriptors; cross-validated MAE 0.44
to 0.38), the Sonnet loop on 26 circles accepted one (perturbing the best packing after the restarts, 2.628 to
2.633), the Haiku loops accepted nothing yet, and random search accepted two small moves on caco2 (0.44 to 0.43) and
none on packing. No rule above was changed after seeing this.

## Deviations

1. **Login expiry, 23 loops rerun (Nov 14, written 18:35 EST, before any rerun loop finished).** The command-line
   login used for all Claude model calls expired at 12:38 EST ("this login reached its time limit"). From then until
   the launcher ended at 13:59 every model call failed: 13 loops ran all 40 slots with no model reply (40 aborts
   each) and 10 more lost between 36 and 261 calls and filled their remaining slots with "invalid". The launcher
   had no check for this. `finish.py` was run on this state at 18:24 before the failure was noticed; its output
   (`loop/runs/BG1/finish.log`) is void and its checks are not reported as results.
   Rule, applied mechanically and not by outcome: every loop whose `calls.jsonl` contains an authentication failure
   is rerun whole from scratch with the same config; every other loop is kept. That keeps 19 loops (all 14 N loops,
   which use no model, and bbb_HH_2, caco_HH_2, clear_HH_2, p26_SS_2, p26_SS_3) and reruns 23. The rerun lives in
   `loop/runs/BG1_R` (same `plan.json`, kept loops linked in, launched 18:30 EST with the frozen launcher); the
   checks are computed there. The first attempts stay in `loop/runs/BG1` and are not used.
   What I had seen when deciding: the void report, including the held-out scores of the first, cut-short attempts
   of the 10 partly hit loops. No rule, threshold or config was changed. Because the rerun set is fixed by the
   failure and not by score, this does not select on outcome, but the rerun loops start later than planned and the
   kept and rerun loops ran at different times of day.
