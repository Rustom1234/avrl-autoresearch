# E1: beyond recall. Does the strong model pull ahead when the loop runs twice as long?

Written Nov 15 2026, before any continued loop started. Frozen by `E1_PREREG.sha256`.

## Why

In BG1 Haiku matched Sonnet over 40 slots, and what both accepted was known practice. One reading: 40 slots only
cover the moves both models remember, and a strong model would show its value after those are used up. E1 tests
that reading directly.

## What is run

The 28 model loops of BG1 (`loop/runs/BG1_R`: 14 Sonnet, 14 Haiku, six tasks) are continued from slot 40 to slot 80
with everything they had: script, ledger, idea backlog, analyst memo. `loop/bench/make_ext.py` copies each loop
directory to `loop/runs/E1` and raises its slot budget; the frozen launcher resumes it. Same models, same engine,
same jobs. One continued loop at a time per lab machine (the H1 run uses the machines at the same time).

## Outcome

`loop/bench/finish.py --root loop/runs/E1` scores the script at slot 80 exactly as BG1 scored the script at slot 40.
extra progress of a loop = progress at slot 80 minus its progress at slot 40 (both held out). A loop that accepts
nothing after slot 40 has extra progress 0. Next to it, the gain the loop itself reports for slots 41 to 80, on the
same progress scale: (in-loop bar at slot 40 minus in-loop bar at slot 80) / (held-out start minus reference).

## Checks, pooled over the six tasks

| Check | Question | Holds when |
|---|---|---|
| X more | Is there anything left to find? | mean extra progress over all 28 loops > 0.05 and above 0 by a one-sided t test, p < 0.05 |
| S strong | Does Sonnet find more of it? | mean extra(SS) > mean extra(HH), permutation test inside tasks, p < 0.05 |
| F phantom | Are late gains mostly noise? | mean in-loop reported gain of slots 41 to 80 is at least twice the mean held-out extra progress |

Described without a rule: the ideas accepted after slot 40 and whether they are still standard practice.

Predictions (probability that the check holds): X 0.4, S 0.15, F 0.6.

## Limits known in advance

- Loops score candidates on one fixed seed, so a script kept for a lucky score on that seed sets a bar that is
  too hard for the second half. This is how the engine has always worked; it applies to both cells.
- 14 loops per cell. The 5 loops that BG1 kept from its first attempt and the 23 it reran are continued alike.
- I know every loop's held-out progress at slot 40.
- A smoke test (`loop/runs/E1_SMOKE`, one packing loop continued for 2 slots on pine, both rejected) checked that
  resuming works. It is not part of E1; the E1 copy of that loop starts again from slot 40.

## Shared machines

CPU jobs as in BG1 (16 GB address-space limit, quiet watchdog at 6 GB). Together with H1 at most four loops per
machine, each with one job of at most 8 threads. A watcher stops the run if the Claude login fails.

## Deviations

None yet.
