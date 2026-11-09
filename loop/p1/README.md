# loop/p1: re-accounting of run R1

`loop/p1` is a re-accounting study of run R1, the first of the November 2026 follow-up studies. It asks what run R1 really cost and what it really found, using only files already on disk: no GPU, no model calls, no Slurm.

The rules are written down in `loop/design/P1_PREREG.md` before the test half of the loops is touched.

## What it computes

For every loop and every arm, on one half of the loops:

1. Cost reconciliation: GPU jobs, GPU hours (measured wall seconds), Claude dollars, open-model dollars, totals at 0.5, 1, 2 and 4 USD per A100-hour, dollars per accepted change, and the GPU price at which arm D costs the same as arm A (or C).
2. Plateau: the slot of the last incumbent change, observed if at least 5 charged slots follow it, otherwise censored at 20. Cost up to that slot. Kaplan-Meier median per arm.
3. Reach 2^17: the first slot whose incumbent has `TOTAL_BATCH_SIZE <= 2**17`, and the cost to get there.
4. Stall: the final incumbent has `TOTAL_BATCH_SIZE > 2**17`, shown next to the final sealed improvement (paired and global-anchor versions).
5. Halting rules: stop after k quiet slots, the same counted from the first change, or stop at a fixed cap. For each loop: where it stops, what share of the dollars it has spent, and how much worse the incumbent is than the final one. The rule with the lowest dollar share among those keeping at least 80% of loops within 0.004 is selected.
6. Noise replay: how many "keep if lower" wins and failed fresh-seed audits pure run-to-run noise predicts for arm A after it reached 2^17, compared with what R1 logged. Then a simulated frontier of four acceptance rules.
7. A one-way ANOVA of the 52 sentinel runs across GPUs.

## The split

Inside each arm, sort the loop ids. Even positions (0, 2, 4, ...) are burn-in, odd positions are test.

- Burn-in: A_01 A_03 A_05 A_07 B_01 B_03 C_01 C_03 D_01 D_03 D_05 D_07
- Test: A_02 A_04 A_06 A_08 B_02 C_02 D_02 D_04 D_06 D_08

## Running it

From the repository root:

```
.venv/bin/python loop/p1/reaccount.py --half burnin
```

Output goes to `loop/runs/R1/p1/burnin/` as `results.json` (every number) and `report.md` (tables in plain language). The run is seeded and takes about two seconds.

`--half test` and `--half all` are refused unless the environment variable `P1_ALLOW_TEST=1` is set, and they also need the frozen halting rule, for example `--rule after_first:5`. The test loops are never opened by a burn-in run. Do not set the variable until the preregistration is frozen and hashed. `results.json` records the sha256 of the code that produced it.

Options: `--sims` (simulation runs, default 20000), `--boot` (bootstrap resamples, default 10000), `--seed` (default 1), `--run` (the run directory), `--out`.

## Tests

From the repository root:

```
PYTHONDONTWRITEBYTECODE=1 .venv/bin/python -m pytest -q -p no:cacheprovider loop/p1/tests
```

They use tiny made-up ledgers and a small made-up run directory. They cover the train.py parser, plateau and censoring, the halting rules (against an independent window version), cost math, the split guard, Kaplan-Meier, the bootstrap, the sentinel ANOVA and both simulations.

## Things to know

- Slots are counted from 0. A censored loop is recorded at slot 20, one past the last real slot.
- The last model call of two loops (the analyst call after slot 19) is in `state.json` but not in the ledger. It is added to slot 19.
- No GPU offsets are used. The sentinels show no GPU effect (F(3,48) = 0.815, p = 0.49), so every value is the raw official val_bpb.
- The noise sd comes from the sentinel runs, which use the unmodified baseline. The converged arm A runs are less noisy (sd about 0.0023 against 0.0038), so the report adds a third noise row at that sd as a check.
- The comparison of arm A with arm D also differs in acceptance rule and scheduler, not only in model tier. The clean contrast is C against D, which has few loops.
