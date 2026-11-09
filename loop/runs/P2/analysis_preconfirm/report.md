# P2 analysis: which loop role earns the expensive model

## Verdict: KILL

| PASS condition | met |
|---|---|
| harm control detected (H slower than SS, one-sided log-rank p < 0.05) | no |
| planner main effect on stall under +0.25 | yes |
| coder main effect on stall under +0.25 | yes |
| planner main effect on median slots to 2^17 under +5.0 slots | no |
| coder main effect on median slots to 2^17 under +5.0 slots | no |
| QT at least 5.0x cheaper than SS per surviving change | no |

- PASS (all six conditions): **no**
- KILL (the harm control is not detected): **yes**
- PIVOT (the cheap planner raises stall by more than 0.25 or delays 2^17 by more than 5.0 slots): **no**
- CODER HURTS (the cheap coder raises stall by more than 0.25 or delays 2^17 by more than 5.0 slots): **no**
- Per role: planner **unclear**, coder **unclear**.
- Precedence when more than one holds: KILL, then PIVOT, then CODER HURTS, then PASS. NO VERDICT means none applies.

The harm control was not detected, so the factorial result stands only as a cost breakdown.

Read the KILL with the power of the test in mind: with 12 H against 12 SS loops the one-sided log-rank test misses a real harm effect fairly often (pre-run simulation on the R1 times: power roughly 0.2 to 0.7 for 6 H loops, see the known limits in loop/design/P2_RUNBOOK.md). A non-detection here is weak evidence that the substrate is blind to a wasteful coder; compare the observed H and SS numbers in the harm control section.

## Loops

| cell | loops | unfinished | stall rate | mean slots to 2^17 | median slots to 2^17 | mean slots to last change | accepts | backed accepts | Claude USD | open USD | model USD per backed change | GPU hours | harm no-ops |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| SS | 12 | 0 | 0.83 | 18.8 | not reached | 13.8 | 1.2 | 0.6 | 0.886 | 0.0040 | 1.525 | 1.39 | 0.0 |
| ST | 12 | 0 | 0.58 | 15.7 | not reached | 12.6 | 1.2 | 0.8 | 0.130 | 0.0197 | 0.199 | 1.34 | 0.0 |
| QS | 12 | 0 | 0.83 | 18.6 | not reached | 16.4 | 0.8 | 0.4 | 1.099 | 0.0072 | 2.654 | 1.37 | 0.0 |
| QT | 12 | 0 | 1.00 | 20.0 | not reached | 16.4 | 0.4 | 0.1 | 0.080 | 0.0345 | 1.372 | 1.38 | 0.0 |
| H | 12 | 0 | 0.83 | 18.3 | not reached | 15.8 | 0.7 | 0.3 | 0.822 | 0.0039 | 2.476 | 1.28 | 5.2 |

Times are slots used (the slot index plus one for an event); a loop that never reaches the event is counted at its finished slots (20 for a finished loop) and marked censored in the log-rank test and the Kaplan-Meier median. 0 of 60 loops finished fewer than 20 slots (stopped early or failed); the stall rate counts them with the final incumbent they had, so a cell with many unfinished loops overstates its stall.

## Main effects (non-harm loops, hi minus lo)

| effect | loops | stall rate, difference (Fisher p) | slots to 2^17: median difference (decision rule), mean difference | mean slots to last change, difference (log-rank p) |
|---|---|---|---|---|
| Q vs S (planner) | 24 vs 24 | 0.92 vs 0.71: 0.21 (p 0.137) | median not reached vs not reached: n/a; mean 19.3 vs 17.2: 2.1 (log-rank p 0.068) | 16.4 vs 13.2: 3.3 (p 0.154) |
| T vs S (coder) | 24 vs 24 | 0.79 vs 0.83: -0.04 (p 1.000) | median not reached vs not reached: n/a; mean 17.8 vs 18.7: -0.8 (log-rank p 0.689) | 14.5 vs 15.1: -0.6 (p 0.909) |

The PASS and PIVOT slot margins use the median difference (P2_PREREG); a median is 'not reached' when more than half of the group never got to 2^17, which counts as +inf for PIVOT and as not shown for PASS.

Interaction (planner x coder) is descriptive: see the cell table above.

## Cost

| effect | dollars | ratio hi / lo (geometric) | permutation p |
|---|---|---|---|
| planner Q vs S | model dollars (Claude + open) | 226.303 | 0.0091 |
| planner Q vs S | model + GPU at 0.5 USD/h | 337.478 | 0.0074 |
| planner Q vs S | model + GPU at 1 USD/h | 372.754 | 0.0087 |
| planner Q vs S | model + GPU at 2 USD/h | 413.168 | 0.0113 |
| planner Q vs S | model + GPU at 4 USD/h | 459.717 | 0.0131 |
| coder T vs S | model dollars (Claude + open) | 0.357 | 0.6530 |
| coder T vs S | model + GPU at 0.5 USD/h | 1.258 | 0.8894 |
| coder T vs S | model + GPU at 1 USD/h | 1.663 | 0.8108 |
| coder T vs S | model + GPU at 2 USD/h | 2.104 | 0.7188 |
| coder T vs S | model + GPU at 4 USD/h | 2.548 | 0.6989 |

Model dollars per backed accepted change, pooled: SS 1.5252, QT 1.3721, ratio SS / QT 1.11 (PASS wants at least 5.0). Model dollars are Claude at list price plus open models at the price file.
With GPU time added the same ratio is 0.5: 0.28, 1: 0.22, 2: 0.18, 4: 0.16 (USD per A100-hour: ratio); GPU time is the same for every cell, so it shrinks the ratio toward 1.

## Sealed held-out gain (secondary, two-sided permutation, 90 percent interval)

| effect | version | loops | difference | interval | p |
|---|---|---|---|---|---|
| planner Q vs S | global anchor | 0 vs 0 | n/a | n/a to n/a | n/a |
| planner Q vs S | paired same wave | 0 vs 0 | n/a | n/a to n/a | n/a |
| coder T vs S | global anchor | 0 vs 0 | n/a | n/a to n/a | n/a |
| coder T vs S | paired same wave | 0 vs 0 | n/a | n/a to n/a | n/a |

Global anchor: mean sealed of all usable anchor runs (n/a) minus the loop's mean confirmation sealed. Paired: each confirmation run against the anchor of its own wave. Expected to be inconclusive.

## Harm control (H against SS)

- Loops: 12 H, 12 SS. Replaced edits: 63 in total, 5.2 per H loop.
- Slots to 2^17, median H not reached against SS not reached, mean H 18.3 against SS 18.8. One-sided log-rank p (H slower): **0.4992**.
- Stall rate H 0.83 against SS 0.83; Fisher exact two-sided p 1.000, one-sided (H more) p 0.705.

## Served fractions and fallbacks

| cell | served fraction (state) | served fraction (ledger) | planner served fraction (ledger) | fallback events | loops under 0.8 | loops with planner calls not served as designated |
|---|---|---|---|---|---|---|
| SS | 1.000 | 1.000 | 1.000 | 0 | 0 | 0 |
| ST | 1.000 | 0.982 | 1.000 | 0 | 0 | 0 |
| QS | 1.000 | 1.000 | 1.000 | 0 | 0 | 0 |
| QT | 1.000 | 0.992 | 1.000 | 0 | 0 | 0 |
| H | 1.000 | 1.000 | 1.000 | 0 | 0 | 0 |

The ledger column covers the coder, repairer, refresh, analyst and voter roles; the planner column covers the planner_backlog, planner_widen and planner_refresh calls, which carry the planner factor.

## Per GPU (GPU is a blocking factor)

| GPU | loop jobs | load-flagged jobs | mean wall s | jobs by cell | sentinels | sentinel mean | sentinel sd |
|---|---|---|---|---|---|---|---|
| 0 | 183 | 0 | 400 | SS 24, ST 31, QS 29, QT 59, H 40 | 10 | 1.11561 | 0.00523 |
| 1 | 184 | 0 | 399 | SS 45, ST 18, QS 53, QT 34, H 34 | 9 | 1.11517 | 0.00478 |
| 2 | 184 | 0 | 403 | SS 36, ST 47, QS 32, QT 38, H 31 | 6 | 1.11600 | 0.00544 |
| 3 | 172 | 0 | 412 | SS 41, ST 44, QS 27, QT 32, H 28 | 17 | 1.11487 | 0.00420 |

One-way ANOVA across GPUs on 42 sentinel runs: p 0.956.

## Load-flagged jobs (sensitivity)

| cell | load-flagged jobs | loops with one | loops |
|---|---|---|---|
| SS | 0 | 0 | 12 |
| ST | 0 | 0 | 12 |
| QS | 0 | 0 | 12 |
| QT | 0 | 0 | 12 |
| H | 0 | 0 | 12 |

No job was load-flagged (no foreign process on the GPU before or after the run), so the verdict stands as is.

