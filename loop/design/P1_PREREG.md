# P1 preregistration: re-accounting of run R1

Status: FROZEN 2026-11-05 23:40 EST (2026-11-06T04:40Z), before any test-half run. Freezing means: this file is hashed, and the hash plus the code fingerprint below are written to avrl/RESEARCH_LOG.md before the test half is run. Changes after freezing go to the Deviations section with a time and a reason.

Code fingerprint when this draft was written: sha256 of `loop/p1/reaccount.py` = `418bfc38c98c96d5343d41ac081935010af2b88126cc740a3437ca2e34ab61f8` (updated at freezing, see "Changes at freezing"). `results.json` records the hash of the code that made it. If the code changes before freezing, rerun the burn-in, check that every rule below still reads the same, and update this line.

## Question

Run R1 (22 loops, 4 arms, 20 slots each) was summarized as "the routed arm is about 6 times cheaper per run at equal held-out gain". P1 re-counts R1 with GPU time included and measures cost to the final result, not cost per fixed 20-slot loop. It asks four things:

1. What do the loops cost in total, and where does the routed arm save or lose money?
2. How many slots does each arm need before its incumbent stops changing (cost to plateau), and is A (Sonnet, keep-if-lower) different from D (routed)?
3. Can a plain stopping rule halt a loop early without losing the result?
4. Does run-to-run noise alone explain R1's audit failures in the converged regime, or is a live second-seed study needed?

## Data and prior exposure

All data are files in `loop/runs/R1`. P1 makes no GPU job, no Slurm job and no Claude call.

Prior exposure, stated plainly: before the split was applied, the planning memo (`lit/avrl/RESEARCH_DIRECTIONS_2026-11.md`), the scoping specs and the data dictionary prepared for this task already contain exploratory R1 numbers computed on all 22 loops (last-change slots, final batch sizes, a halting replay, audit counts). The split limits the damage but does not remove it. This draft was written with code run only on the burn-in half; the test loops' ledgers and outcomes were not opened, and every choice below was made from the burn-in output. The test half is therefore not fully blind. The memo already flags the same limit (its Caveats section).

## The split (fixed)

Within each arm, sort the loop ids. Even positions (0, 2, 4, ...) are burn-in, odd positions are test.

- Burn-in (12): A_01 A_03 A_05 A_07 B_01 B_03 C_01 C_03 D_01 D_03 D_05 D_07
- Test (10): A_02 A_04 A_06 A_08 B_02 C_02 D_02 D_04 D_06 D_08

`reaccount.py --half test` and `--half all` refuse to run unless `P1_ALLOW_TEST=1` is set and a frozen halting rule is given with `--rule`. A burn-in run never opens a test loop (checked by a unit test).

## Definitions

**Slot.** A charged slot, numbered 0 to 19. Every R1 loop has exactly 20 charged slots. A censored loop is recorded at slot 20.

**Incumbent change.** An experiment is a candidate row with the replicate, audit and crash_fix rows that follow it (a new experiment starts at kind candidate or crash_fix; infra rows are skipped). It is an incumbent change if its last row has decision accept. Its slot is the slot of that last row. The incumbent after slot s includes any change made at slot s.

**Raw official mean of the incumbent.** The mean of `official_val_bpb` over the runs behind the accepted experiment (rows with a job id and an official value). Before the first change it is the mean of the 16 launch sentinels (1.114885). No GPU offset is used anywhere: the frozen offsets are probably noise (see item 7).

**Batch size and depth.** Read with Python `ast` from the `train.py` of the accept row's slot (the last module-level assignment to `TOTAL_BATCH_SIZE`, `DEPTH`, `ASPECT_RATIO`, `HEAD_DIM` wins; constants and `** * + - // <<` are evaluated). The final incumbent is read from `incumbent.py`. Each parse is cross-checked against the run's printed depth and tokens per step; a mismatch stops the program. The baseline has `TOTAL_BATCH_SIZE = 2**19`, read from `loop/substrate/base/train_ampere.py`.

### 1. Cost reconciliation

- **GPU jobs:** ledger rows with a job id.
- **GPU hours:** sum of measured `wall_seconds` in `slots/<slot>/result.json`, divided by 3600 (the job id in the file must match the ledger).
- **Claude dollars:** `cost_usd_plan` of every model call of a Claude model, on every ledger row (charged or not). Calls ride on the row after them. The analyst call after slot 19 (in `state.json` `pending_calls`; present only in D_01 and D_06) is added to slot 19.
- **Open-model dollars:** `tokens_in x input + tokens_out x output` per million tokens, at the prices in `loop/models/open_prices.json`, for calls served by a model listed there.
- **Model dollars** = Claude + open. **Total at price p** = model dollars + GPU hours x p, for p in {0.5, 1, 2, 4} USD per A100-hour. The reference price is 2.
- **Accepted changes per loop** = number of incumbent changes. **Dollars per accepted change** for an arm = sum of its loops' totals divided by the sum of its changes.
- **Crossover price:** the p where arm D's mean total per loop equals arm A's (and C's). Reported only if it is positive; otherwise the report says who is cheaper at every price from 0 to 4. Computed for the full 20-slot loop and, as a second figure, for cost to plateau.

### 2. Plateau

The plateau slot is the slot of the last incumbent change. It is observed if at least 5 charged slots follow it (so a last change at slot 14 or earlier); otherwise it is right-censored and recorded as 20. Cost to plateau is jobs, GPU hours, model dollars and total dollars (at 2 USD per A100-hour) up to and including the plateau slot; a censored loop is charged its whole run. Per arm: the mean slot, and the Kaplan-Meier median (smallest event time where survival is at or below 0.5; "not reached" if it never gets there).

### 3. Reach 2^17

The first slot whose incumbent has `TOTAL_BATCH_SIZE <= 2**17`, censored at 20 if there is none. Cost to reach it as in item 2.

### 4. Stall

A loop stalls if its final incumbent has `TOTAL_BATCH_SIZE > 2**17`. It is reported next to the loop's final sealed improvement, in two versions: paired (the mean over the loop's confirmation runs of the same-wave anchor's sealed value minus the run's sealed value) and global-anchor (the mean sealed value of all 30 anchors minus the loop's mean confirmation sealed value). Higher is better. Separation means every non-stalled loop beats every stalled loop in that version.

### 5. Halting rules

- (a) quiet k: stop after k consecutive charged slots with no incumbent change, counted from slot 0, k in {3, 5, 8, 10}.
- (b) after_first k: the same, but counting starts after the first change.
- (c) cap N: stop after N slots, N in {5, 8, 10, 12}.

If a rule never fires, the loop runs to slot 19. For each rule and loop: the halting slot; the fraction of the loop's total dollars (at 2 USD per A100-hour) spent by the end of that slot; and the loss = raw official mean of the incumbent at the halt minus the raw official mean of the final incumbent (positive is worse). A loop is within the margin if loss <= 0.004. Per rule: the share of loops within the margin and the mean dollar fraction.

Selection (burn-in only): among the 12 rules, those keeping at least 80% of loops within 0.004; of those, the one with the lowest mean dollar fraction (ties go to the earlier rule in the list above). If none qualifies, the report says so.

### 6. Noise replay

- **Noise sd:** sd (n-1) of the raw official values of the 52 sentinel runs (16 launch plus 36 live): 0.003777. Mean 1.113987.
- **Converged regime:** in the arm A loops of the half, the candidate first runs made after the incumbent first had `TOTAL_BATCH_SIZE <= 2**17`.
- **Observed:** the number of those candidates; the would-be keeps (first run strictly below the incumbent's raw mean; R1 logged these as decision accept or audit, and the replay checks that the two agree for every candidate); the audits R1 then ran; the audit rejects and accepts. R1's tripwire sends every converged would-be keep to a sealed, fresh-seed audit, and a keep in the last slot gets none.
- **Null:** a pure-noise world where every candidate has the same true level as the incumbent. The level is the mean of the half's converged first runs, and the noise is Gaussian with the sentinel sd. Each simulated loop starts from the observed incumbent after the 2^17 change (its raw mean and number of runs) and plays its remaining slots: a slot is a candidate (a keep if the run is below the incumbent mean) or, after a keep, the audit (a fresh run; accepted if z over both runs against the incumbent is at least 1.5 with sigma 0.004743, otherwise rejected; an accept resets the incumbent to those two runs). 20000 simulated halves, seed 1. Each simulation first draws its sd from the sampling law of an sd estimated from 52 runs and its level from the standard error of the mean, so the interval is predictive.
- **Interval:** the 5th percentile rounded down and the 95th percentile rounded up, as integers. "Inside" includes the end points.
- **Also reported, not decisive:** a bootstrap of the sentinel residuals as the noise, and a Gaussian at the sd of the converged first runs themselves (burn-in value 0.00225, about 0.60 times the sentinel sd, because the sentinels run the unmodified 2^19 baseline).

Frontier by simulation (sentinels and arm A loops only). One slot is one candidate decision; a simulated loop has 20 of them against an incumbent measured by n runs, where n is the median number of runs behind the converged arm A incumbents (2 in burn-in). Rules: keep_if_lower (first run below the incumbent mean); zband (R1's rule: reject below z = 0.75, accept at 2.0 or above, replicate between, accept the replicated mean at 1.5; sigma 0.004743); mean2 (mean of 2 runs below the incumbent mean); median3 (median of 3 runs below the incumbent mean). A keep moves the incumbent to the candidate's own statistic and run count. Injected true effects: 0, 0.001, 0.002, 0.004, 0.008 bpb, meaning each candidate is truly that much better than the current incumbent. Noise: Gaussian at the sentinel sd, bootstrap of the sentinel residuals, and (check) Gaussian at the converged sd. Reported: false keeps per 20 slots at effect 0; the keep probability of one decision at each positive effect; runs per slot; and the mean keep rate over a stream of 20 truly better candidates.

### 7. GPU offsets

One-way ANOVA of the 52 raw sentinel values across the four GPUs: F and p.

## What was chosen on burn-in

1. **Halting rule: after_first_k5.** On the 12 burn-in loops it keeps 10 of 12 within 0.004 (83%) at a mean dollar fraction of 0.482. Rules that keep at least 80%: quiet_k8 (92%, 0.656), quiet_k10 (100%, 0.784), after_first_k5 (83%, 0.482), after_first_k8 (92%, 0.656), after_first_k10 (100%, 0.784), cap_n10 (83%, 0.498), cap_n12 (92%, 0.598). The winner leads cap_n10 by 0.016, which is thin. Both fail the same two loops (D_01 and D_03). The full table is in `loop/runs/R1/p1/burnin/report.md`, section 5.
2. **Stall definition:** final `TOTAL_BATCH_SIZE > 2**17`. It flags 4 of 12 burn-in loops (C_03, D_01, D_03, D_05) and separates them from the rest in both improvement versions (see "Endpoints handed to P2").
3. **Frontier incumbent size:** 2 runs (the burn-in median).
4. **Level of the null:** the pooled mean of the converged first runs (1.0514). Per-loop means run from 1.0507 (A_01, which found a real warmdown gain) to 1.0519.
5. **Simulation settings:** 20000 simulations, 10000 bootstrap resamples, seed 1.

Fixed by the task, not by data: the split; the plateau censoring rule (5 quiet slots, censor at 20); the 0.004 margin and the 80% bar; the price grid; sigma 0.004743 (R1's `anchor.json`); the four frontier rules and five effect sizes.

## What will be computed on the test half

One run, once, after freezing:

```
P1_ALLOW_TEST=1 .venv/bin/python loop/p1/reaccount.py --half test --rule after_first:5
```

It writes `loop/runs/R1/p1/test/results.json` and `report.md`: every table above for the 10 test loops, plus the three decision lines below. Nothing is re-selected, re-tuned or re-defined from the test output. If the program crashes on a data problem, fix the crash without looking at results, rerun, and write the fix under Deviations.

## Decision lines (test half)

**Claim 1: arm A and arm D differ in slots to plateau.** Take each test-half A loop and D loop's plateau slot (censored loops at 20). Compute mean(A) minus mean(D) and a 90% percentile bootstrap interval (resample loops within each arm, 10000 times). PASS if the interval excludes zero, in either direction. Sensitivity: the same with censored loops dropped. Both are reported; the verdict follows the censored-at-20 version, and the PASS is labelled robust only if the sensitivity interval also excludes zero. The comparison also differs in acceptance rule and scheduler, so a PASS says "A and D differ", not "the model tier causes it". The clean contrast, C against D, is reported descriptively; the test half has only one C loop.

**Halting: the selected rule works.** On the test half, apply after_first_k5. PASS if at least 80% of the test loops are within 0.004 of their final incumbent AND the mean dollar fraction (at 2 USD per A100-hour) is at most 0.60. Otherwise KILL. With 10 test loops, 80% means 8 or more.

**N2 Test 1: noise explains R1.** Run the noise replay on the test-half arm A loops, with the level estimated from their own converged first runs. PASS (noise explains R1) if the observed number of converged-regime audit rejects lies inside the 90% predictive interval of the Gaussian, sentinel-sd null. Otherwise PROMOTE: run the live second-seed stage (about 85 to 100 A100-hours) before Project 2. The bootstrap and converged-sd rows are shown beside it. If they disagree with the decisive row, the disagreement is reported, and the verdict does not change.

Everything else (cost reconciliation, Kaplan-Meier medians, reach 2^17, stall, the other eleven halting rules, the frontier, the ANOVA) is reported as description only. No other claim is made from the test half.

## Burn-in results (dry run; not evidence for the decision lines)

Cost per loop at 2 USD per A100-hour, mean by arm: A 5.47 (GPU 4.63), B 5.60, C 6.02, D 4.49 (GPU 4.24). Claude dollars per loop: A 0.84, B 1.03, C 1.40, D 0.22, plus 0.026 open-model dollars in D. GPU time is 77% (C) to 95% (D) of the total (A 85%, B 82%), so D is cheaper than A at every GPU price from 0 to 4 (by 0.60 at price 0, by 1.37 at 4), and D is cheaper than C at every price.

Cost to plateau (total at 2 USD per A100-hour): A 1.01 (mean slot 2.75), B 0.81, C 1.75, D 2.63 (mean slot 10.75, one of four loops censored; Kaplan-Meier median 8). A is cheaper to plateau than D at every positive GPU price. D is cheaper than C to plateau only below 0.374 USD per A100-hour.

Reach 2^17: A and B at slot 2 in every loop; C_01 at slot 1; D_07 at slot 8; C_03, D_01, D_03 and D_05 never reach it (censored).

Stall: 4 of 12 burn-in loops (C_03, D_01, D_03, D_05; all end at 2^18 with depth 6). Paired improvement 0.0466 to 0.0563 (mean 0.0520) against 0.0597 to 0.0657 (mean 0.0630) for the other 8. Global-anchor: 0.0500 to 0.0585 against 0.0633 to 0.0640. Separated in both versions.

Halting: see the selection above.

Noise replay, arm A burn-in: 50 converged candidates (A_01 15, A_03 10, A_05 13, A_07 12), 19 would-be keeps, 18 audits, 17 audit rejects, 1 audit accept (A_01's warmdown). The logged keep decision matches "first run below the incumbent mean" for 50 of 50. Pure-noise prediction (Gaussian, sentinel sd): keeps 22.3 (90% interval 17 to 27), audit rejects 20.7 (15 to 26), audit accepts 0.35 (0 to 1). The observed 17 rejects are inside. Bootstrap: 16 to 24, inside. Gaussian at the converged sd: 17 to 26, inside, but the one observed audit accept falls outside its 0 to 0 interval.

Frontier (Gaussian, sentinel sd, incumbent of 2 runs), false keeps per 20 slots with no true effect: keep_if_lower 2.76, zband 0.25, mean2 2.63, median3 2.61, at 1.00, 1.11, 2.00 and 3.00 runs per slot. Extra runs do not cut false keeps for a rule that keeps whenever a statistic is below a ratcheting bar; they only add power. The z-band cuts false keeps by about 90% at 1.11 runs per slot and pays for it in power: one-decision keep probability at a true gain of 0.004 is 0.20 (zband) against 0.81 (keep_if_lower), and at 0.008 it is 0.57 against 0.96.

GPU ANOVA: F(3, 48) = 0.815, p = 0.49. No GPU effect on the 52 sentinels.

Dry run of the decision lines on burn-in: Claim 1 A minus D = -8.00 slots, 90% interval -13.00 to -3.75 (without the censored loop: -4.92, -7.25 to -2.50). Halting after_first_k5: 83% within 0.004 at 0.482 of the dollars. N2 Test 1: 17 observed audit rejects, interval 15 to 26. All three would pass in sample. The halting rule was selected on these loops, so its pass is built in.

## Endpoints handed to P2

**Stall definition (replace line 46 of P2_PREREG.md).** Use: the loop's final incumbent has `TOTAL_BATCH_SIZE > 2**17`, parsed from `incumbent.py` with the last-module-level-assignment rule. The draft rule ("lacks total batch size at most 2^18 or lacks depth 6") is wrong for R1. On burn-in it flags 8 of 12 loops (every loop that ended at depth 8), whose mean paired improvement is 0.0630, and it flags none of the 4 loops that stalled (0.0520). The recommended rule gets the direction right and separates the groups in both versions: stalled maximum 0.0563 against non-stalled minimum 0.0597 (paired), and 0.0585 against 0.0633 (global-anchor). The global-anchor version is the cleaner check, because the paired version adds the noise of a single anchor run per wave. Burn-in stall counts by arm: A 0 of 4, B 0 of 2, C 1 of 2, D 3 of 4.

**The second "large change" is not depth 6.** In burn-in, all 4 loops with depth 6 in the final incumbent are the stalled ones, and all 8 loops that ended at 2^17 ended at depth 8. So replace "slot at which both large changes are first in the incumbent" by the reach-2^17 slot (the first slot whose incumbent has `TOTAL_BATCH_SIZE <= 2**17`, censored at 20), and keep the plateau slot as the second slot endpoint. In burn-in arm D, 3 of 4 loops are censored on reach-2^17 and 1 of 4 on plateau, so P2 needs a censoring-aware test (log-rank, as drafted) and should state the margin as a difference in mean slot with censored loops at 20.

**Slot margin: use 5 slots, not 3.** Arm D's slot endpoints spread widely in burn-in: sd 6.5 slots (plateau) and 6.0 (reach-2^17), n = 4 loops, censored at 20. For P2's main effects (24 loops against 24) the standard error of a difference is about 1.9 and 1.7 slots, and the 90% half-width is 3.1 and 2.8 slots. A margin of +3 on the upper end of a 90% interval is no larger than the noise: a cell with no true delay would fail it about half the time. A margin of 5 leaves about 2 slots for a real effect. This sd comes from four loops, one censored, so treat 5 as a floor and recompute it from the P2 same-wave loops if they spread more.

**Cost outcome.** GPU time is 77% to 95% of loop dollars (arm means) at 2 USD per A100-hour, so P2's dollars per surviving change should be reported GPU-inclusive over the price grid 0.5, 1, 2, 4, with model-only dollars as the second line.

## Limits

- Twelve burn-in and ten test loops. The A against D comparison has four loops per arm in each half; a bootstrap interval from four loops is coarse.
- A against D also differs in acceptance rule and scheduler, not only model tier. The clean contrast C against D has two burn-in loops and one test loop.
- The halting rule was chosen from 12 loops with a 0.016 lead over the next rule. The test half checks the choice but does not tell whether it is the best rule.
- The sentinel noise (sd 0.0038) is larger than the noise of converged runs (about 0.0023). The decisive null is therefore generous to chance; the third noise row shows the effect.
- The null takes one shared level for all converged candidates, and holds the audit z to R1's raw-value approximation (R1 computed z on offset-adjusted values; the offsets are probably noise).
- R1 ran on four A100s at hannah n7 with 3 to 6 loops sharing them; wall times vary with queue wait. GPU hours here are measured job wall seconds, which exclude queue wait.

## Verifier corrections before freezing

An independent script (no import of `loop/p1`) recomputed the burn-in numbers from the raw ledgers, slot files, sentinels and confirmation runs: per-loop and per-arm GPU hours, Claude and open-model dollars, jobs, changes, plateau and reach slots and their costs, Kaplan-Meier medians, stall flags and both improvement versions, all 12 halting rules per loop, the 52-sentinel sd, the ANOVA, the converged arm A counts, the Gaussian null interval and two frontier rows. All agree with `results.json`. Two corrections, 2026-11-05:

1. The GPU share of loop dollars at 2 USD per A100-hour was written as 85% to 95%. By arm it is A 85%, B 82%, C 77%, D 95%. Corrected in "Burn-in results" and in "Cost outcome".
2. The code re-derived the frontier incumbent size from whichever half it ran, although item 3 of "What was chosen on burn-in" fixes it at 2. The test and all halves now use the burn-in value (`FRONTIER_N_INC = 2`). The burn-in output is unchanged apart from the code hash; the fingerprint above is updated.

## Changes at freezing

The guard now also refuses a test or all run unless `--rule after_first:5` is given exactly (`FROZEN_RULE`), and refuses if the loop directories do not give the split listed above (`FIXED_BURNIN`, `FIXED_TEST`). Two unit tests cover this. The burn-in was rerun with the hardened code, and `results.json` is identical apart from the code hash. The draft's open choices are accepted as written: the sentinel-sd Gaussian null is decisive, the censored-at-20 interval decides Claim 1, and the prior exposure stated above is acknowledged.

## Deviations

- 2026-11-06 04:45Z: after the single frozen test-half run, the same frozen code was also run with `--half all` (output `loop/runs/R1/p1/all/`) to give descriptive all-22-loop figures for the project. It makes no decision and changes nothing in the test-half verdicts.
