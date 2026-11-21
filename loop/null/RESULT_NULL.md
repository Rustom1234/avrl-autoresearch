# No-effect pools: how much gain does a loop report when nothing changed?

Nov 15 2026. Data: `loop/runs/GPT_NULL/pool.jsonl` (96 jobs), `loop/runs/CG1_NULL/pool.jsonl` (160 jobs).
Replay script: `loop/null/replay.py`. Outputs: `replay.txt` in each run directory. Not preregistered; descriptive.

## The question

A research loop keeps an edit when its score beats the current best. Scores are noisy: the same script, run twice,
gives two different numbers. So a loop can keep edits that do nothing and report a gain that is only luck. I call
that a phantom gain. Earlier (`RESULT_CG1.md`) I estimated it by simulating bell-curve noise. Here it is measured
with real runs.

## Method

1. Run the unchanged starting script many times with different seeds. That is the pool. Every number in it comes
   from the same code, so the true gain between any two is zero.
2. Replay a loop on the pool: each "candidate edit" is just another draw from the pool. Apply the loop's accept
   rule. Whatever gain the loop reports at the end is phantom by construction.
3. Repeat 20,000 times and average.

Worked example. Starting bar 1.1406 bpb. Draws 1.1432, 1.1390, 1.1301. A rule that keeps any run below the best so
far keeps the second and the third and reports a gain of 1.1406 - 1.1301 = 0.0105 bpb, from three runs of one
unchanged script.

Three accept rules are replayed:

| Rule | What it does |
|---|---|
| ours | the engine of this thesis: reject below 0.75 noise units, accept at 2.0, in between run it again and accept at a mean of 1.5; 20 or 40 experiments |
| strict | keep whenever one run beats the best so far (the original autoresearch loop); 100 experiments |
| two-seed | keep when one run beats the bar by 2 noise units and a second run also beats it; 93 experiments |

A "noise unit" (sigma) is the loop's own estimate of the run-to-run sd.

## Pool 1: GPT task, hannah n7

96 runs of the unchanged baseline `train.py` (device batch 64, 300 s budget, four A100s, 24 runs each), all clean.
Mean 1.14057 bpb, sd 0.00697, range 1.1269 to 1.1656. The noise is lopsided: the worst run is 3.6 sds above the
mean, the best only 2.0 below. Per GPU the means are 1.1390, 1.1412, 1.1384, 1.1437.

| Rule | Experiments | Phantom gain (bpb) | Edits kept | Loops that keep something |
|---|---|---|---|---|
| ours, sigma from the pool (0.0080) | 20 | 0.0011 | 0.08 | 8% |
| ours, sigma from the pool | 40 | 0.0018 | 0.12 | 12% |
| strict | 100 | 0.0132 | 3.7 | 98% |
| two-seed, sigma from the pool | 93 | 0.0043 | 0.30 | 29% |
| two-seed, sigma set to 0.001 | 93 | 0.0090 | 1.65 | 86% |
| ours, sigma set to 0.001 | 40 | 0.0116 | 2.36 | 100% |

Reading:

- With the noise measured honestly, our rule reports almost nothing from nothing on this task (0.001 to 0.002 bpb).
  P2B's real gains were about 0.037, twenty times larger. That agrees with the earlier check that P2B's in-loop and
  held-out gains matched.
- The strict keep-if-better rule reports 0.013 bpb from nothing in 100 experiments, in 98% of loops.
- Bell-curve noise overstates the problem here. Because good outliers are rare, the real pool gives a quarter
  to three quarters of the phantom gain that a Gaussian of the same sd gives.
- The sigma matters more than the rule. If a loop believes its noise is 0.001 while the real sd is 0.007, even a
  two-seed 2-sigma rule reports 0.009 bpb from nothing.

On AutoScientists: its GPT-task gain is 0.0047 bpb over 93 experiments, read from in-loop numbers, with a two-seed
2-sigma rule and a stated sigma of 0.001. On our pool, a correctly calibrated two-seed rule produces 0.0043 from
nothing in 29% of loops, and the same rule with sigma 0.001 produces 0.0090. This does not show their gain is
noise: their hardware (two H100s) and settings differ from ours, their noise may really be near 0.001, and they
publish no run logs or final script to check. What it shows is that a gain of that size cannot be told from luck
without a held-out rerun, on a setup with our noise level.

## Pool 2: CIFAR speedrun, four lab GPUs

160 runs of the unchanged starting script (40 each on willow, walnut, cherry, spruce), all clean. Score sd 0.0266
(the loops used sigma 0.0258). Host means differ: cherry +0.014, walnut -0.009, against a run sd of 0.027.

| Rule | Experiments | Phantom gain (score) | Edits kept | Loops that keep something |
|---|---|---|---|---|
| ours | 20 | 0.026 | 0.46 | 46% |
| ours | 40 | 0.037 | 0.66 | 66% |
| strict | 100 | 0.063 | 3.9 | 99% |
| two-seed | 93 | 0.029 | 0.61 | 57% |

With the sealed (held-out) score of the same jobs as the pool: 0.021 and 41% at 20 experiments.

Reading:

- The CG1 random-search cell, which had no model and whose held-out gain was 0.000, reported 0.018 in-loop and
  kept something in 5 of 12 loops (42%). The replay on real noise gives 0.021 to 0.026 and 41 to 46%. The earlier
  bell-curve simulation gave 0.018 and 41%. So measured noise confirms it: that cell's in-loop gain is what pure
  luck produces.
- Here real noise and bell-curve noise agree (0.96 against 1.01 sds), unlike the GPT task.
- Machines are not interchangeable: a candidate that happens to land on walnut against a bar set on cherry gets
  0.02 for free, close to the whole phantom gain. Loops should score candidate and bar on the same machine.

## What this adds to the thesis

1. The phantom-gain claim now rests on 256 real runs on two tasks, not on a simulation.
2. Whether a loop's own numbers can be trusted depends on gain size over noise, and the noise must be measured, not
   assumed: wrong sigma turns a careful rule into a lucky one.
3. A practical rule for any routing comparison: report held-out reruns, and score on one machine.

## Limits

- GPT pool: batch 64 on A100s, which is our configuration, not that of any other paper. 96 runs.
- The replay assumes candidates are as noisy as the baseline and ignores crashes.
- Replays are of accept rules alone, with no model in the loop.
