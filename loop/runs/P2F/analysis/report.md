# P2 analysis: which loop role earns the expensive model

## Verdict: KILL

| PASS condition | met |
|---|---|
| harm control detected (H slower than SS, one-sided log-rank p < 0.05) | no |
| planner main effect on stall under +0.25 | no |
| coder main effect on stall under +0.25 | yes |
| planner main effect on median slots to 2^17 under +5.0 slots | no |
| coder main effect on median slots to 2^17 under +5.0 slots | no |
| QT at least 5.0x cheaper than SS per surviving change | no |

- PASS (all six conditions): **no**
- KILL (the harm control is not detected): **yes**
- PIVOT (the cheap planner raises stall by more than 0.25 or delays 2^17 by more than 5.0 slots): **yes**
- CODER HURTS (the cheap coder raises stall by more than 0.25 or delays 2^17 by more than 5.0 slots): **no**
- Per role: planner **cheap tier hurts**, coder **unclear**.
- Precedence when more than one holds: KILL, then PIVOT, then CODER HURTS, then PASS. NO VERDICT means none applies.

The harm control was not detected, so the factorial result stands only as a cost breakdown.

Read the KILL with the power of the test in mind: with 12 H against 12 SS loops the one-sided log-rank test misses a real harm effect fairly often (pre-run simulation on the R1 times: power roughly 0.2 to 0.7 for 6 H loops, see the known limits in loop/design/P2_RUNBOOK.md). A non-detection here is weak evidence that the substrate is blind to a wasteful coder; compare the observed H and SS numbers in the harm control section.

## Loops

| cell | loops | unfinished | stall rate | mean slots to 2^17 | median slots to 2^17 | mean slots to last change | accepts | backed accepts | Claude USD | open USD | model USD per backed change | GPU hours | harm no-ops |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| SS | 12 | 0 | 0.58 | 16.6 | not reached | 10.0 | 2.2 | 1.2 | 1.428 | 0.0060 | 1.147 | 2.27 | 0.0 |
| ST | 12 | 0 | 0.50 | 15.3 | 19.0 | 9.1 | 2.1 | 1.3 | 0.224 | 0.0290 | 0.190 | 2.08 | 0.0 |
| QS | 12 | 0 | 0.75 | 17.8 | not reached | 14.5 | 1.2 | 0.8 | 1.649 | 0.0086 | 2.211 | 2.22 | 0.0 |
| QT | 12 | 0 | 0.92 | 19.4 | not reached | 13.9 | 0.7 | 0.3 | 0.115 | 0.0398 | 0.464 | 1.56 | 0.0 |
| H | 12 | 0 | 0.75 | 17.2 | not reached | 11.1 | 1.2 | 0.6 | 1.442 | 0.0064 | 2.484 | 2.29 | 10.6 |

Times are slots used (the slot index plus one for an event); a loop that never reaches the event is counted at its finished slots (20 for a finished loop) and marked censored in the log-rank test and the Kaplan-Meier median. 0 of 60 loops finished fewer than 20 slots (stopped early or failed); the stall rate counts them with the final incumbent they had, so a cell with many unfinished loops overstates its stall.

## Main effects (non-harm loops, hi minus lo)

| effect | loops | stall rate, difference (Fisher p) | slots to 2^17: median difference (decision rule), mean difference | mean slots to last change, difference (log-rank p) |
|---|---|---|---|---|
| Q vs S (planner) | 24 vs 24 | 0.83 vs 0.54: 0.29 (p 0.060) | median not reached vs not reached: n/a; mean 18.6 vs 16.0: 2.6 (log-rank p 0.037) | 14.2 vs 9.5: 4.7 (p 0.001) |
| T vs S (coder) | 24 vs 24 | 0.71 vs 0.67: 0.04 (p 1.000) | median not reached vs not reached: n/a; mean 17.4 vs 17.2: 0.2 (log-rank p 0.741) | 11.5 vs 12.2: -0.8 (p 0.672) |

The PASS and PIVOT slot margins use the median difference (P2_PREREG); a median is 'not reached' when more than half of the group never got to 2^17, which counts as +inf for PIVOT and as not shown for PASS.

Interaction (planner x coder) is descriptive: see the cell table above.

## Cost

| effect | dollars | ratio hi / lo (geometric) | permutation p |
|---|---|---|---|
| planner Q vs S | model dollars (Claude + open) | 0.831 | 0.5512 |
| planner Q vs S | model + GPU at 0.5 USD/h | 0.876 | 0.3433 |
| planner Q vs S | model + GPU at 1 USD/h | 0.870 | 0.1894 |
| planner Q vs S | model + GPU at 2 USD/h | 0.862 | 0.0751 |
| planner Q vs S | model + GPU at 4 USD/h | 0.856 | 0.0277 |
| coder T vs S | model dollars (Claude + open) | 0.128 | 0.0001 |
| coder T vs S | model + GPU at 0.5 USD/h | 0.409 | 0.0001 |
| coder T vs S | model + GPU at 1 USD/h | 0.522 | 0.0001 |
| coder T vs S | model + GPU at 2 USD/h | 0.621 | 0.0001 |
| coder T vs S | model + GPU at 4 USD/h | 0.693 | 0.0001 |

Model dollars per backed accepted change, pooled: SS 1.1472, QT 0.4644, ratio SS / QT 2.47 (PASS wants at least 5.0). Model dollars are Claude at list price plus open models at the price file.
With GPU time added the same ratio is 0.5: 0.73, 1: 0.58, 2: 0.49, 4: 0.44 (USD per A100-hour: ratio); GPU time is the same for every cell, so it shrinks the ratio toward 1.

## Sealed held-out gain (secondary, two-sided permutation, 90 percent interval)

| effect | version | loops | difference | interval | p |
|---|---|---|---|---|---|
| planner Q vs S | global anchor | 24 vs 24 | -0.03618 | -0.04399 to -0.02836 | 0.000 |
| planner Q vs S | paired same wave | 24 vs 24 | -0.03689 | -0.04497 to -0.02881 | 0.000 |
| coder T vs S | global anchor | 24 vs 24 | -0.00584 | -0.01747 to 0.00599 | 0.402 |
| coder T vs S | paired same wave | 24 vs 24 | -0.00634 | -0.01835 to 0.00589 | 0.383 |

Global anchor: mean sealed of all usable anchor runs (1.11515) minus the loop's mean confirmation sealed. Paired: each confirmation run against the anchor of its own wave. Expected to be inconclusive.

## Harm control (H against SS)

- Loops: 12 H, 12 SS. Replaced edits: 127 in total, 10.6 per H loop.
- Slots to 2^17, median H not reached against SS not reached, mean H 17.2 against SS 16.6. One-sided log-rank p (H slower): **0.2161**.
- Stall rate H 0.75 against SS 0.58; Fisher exact two-sided p 0.667, one-sided (H more) p 0.333.

## Served fractions and fallbacks

| cell | served fraction (state) | served fraction (ledger) | planner served fraction (ledger) | fallback events | loops under 0.8 | loops with planner calls not served as designated |
|---|---|---|---|---|---|---|
| SS | 1.000 | 1.000 | 1.000 | 0 | 0 | 0 |
| ST | 1.000 | 0.974 | 1.000 | 0 | 0 | 0 |
| QS | 1.000 | 1.000 | 1.000 | 0 | 0 | 0 |
| QT | 1.000 | 0.991 | 1.000 | 0 | 0 | 0 |
| H | 1.000 | 1.000 | 1.000 | 0 | 0 | 0 |

The ledger column covers the coder, repairer, refresh, analyst and voter roles; the planner column covers the planner_backlog, planner_widen and planner_refresh calls, which carry the planner factor.

## Per GPU (GPU is a blocking factor)

| GPU | loop jobs | load-flagged jobs | mean wall s | jobs by cell | sentinels | sentinel mean | sentinel sd |
|---|---|---|---|---|---|---|---|
| 0 | 285 | 0 | 400 | SS 48, ST 53, QS 60, QT 61, H 63 | 12 | 1.11551 | 0.00484 |
| 1 | 276 | 0 | 399 | SS 65, ST 36, QS 74, QT 41, H 60 | 14 | 1.11504 | 0.00518 |
| 2 | 278 | 0 | 403 | SS 73, ST 55, QS 44, QT 45, H 61 | 10 | 1.11568 | 0.00451 |
| 3 | 281 | 0 | 404 | SS 53, ST 76, QS 54, QT 43, H 55 | 18 | 1.11501 | 0.00412 |

One-way ANOVA across GPUs on 54 sentinel runs: p 0.977.

## Load-flagged jobs (sensitivity)

| cell | load-flagged jobs | loops with one | loops |
|---|---|---|---|
| SS | 0 | 0 | 12 |
| ST | 0 | 0 | 12 |
| QS | 0 | 0 | 12 |
| QT | 0 | 0 | 12 |
| H | 0 | 0 | 12 |

No job was load-flagged (no foreign process on the GPU before or after the run), so the verdict stands as is.

