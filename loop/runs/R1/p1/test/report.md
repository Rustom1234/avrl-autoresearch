# Project 1 re-accounting of R1: test half

Zero GPU, zero API. Every number comes from the files in loop/runs/R1. Loops in this half: A_02, A_04, A_06, A_08, B_02, C_02, D_02, D_04, D_06, D_08.
Dollars: Claude at cost_usd_plan, open models at the per-token prices in loop/models/open_prices.json, GPU at the price shown. GPU hours are measured wall seconds from slots/*/result.json. Slots are 0-based; a censored loop is recorded at slot 20.

Parameters: {"half": "test", "sims": 20000, "boot": 10000, "seed": 1, "sigma_used": 0.004743, "base_official": 1.1148854674525384, "base_tbs": 524288, "code_sha256": "418bfc38c98c96d5343d41ac081935010af2b88126cc740a3437ca2e34ab61f8"}.

## 1. Cost reconciliation

Mean per loop, by arm.

| arm | loops | GPU jobs | GPU hours | Claude USD | open USD | accepted changes | total @0.5 | total @1.0 | total @2.0 | total @4.0 |
|---|---|---|---|---|---|---|---|---|---|---|
| A | 4 | 20.0 | 2.32 | 0.720 | 0.000 | 2.00 | 1.88 | 3.04 | 5.36 | 10.01 |
| B | 1 | 20.0 | 2.29 | 1.129 | 0.000 | 2.00 | 2.27 | 3.42 | 5.71 | 10.28 |
| C | 1 | 20.0 | 2.29 | 1.522 | 0.000 | 2.00 | 2.67 | 3.82 | 6.11 | 10.70 |
| D | 4 | 19.0 | 2.12 | 0.217 | 0.027 | 2.25 | 1.31 | 2.37 | 4.49 | 8.74 |

Dollars per accepted change (arm total dollars divided by arm accepted changes).

| arm | @0.5 | @1.0 | @2.0 | @4.0 |
|---|---|---|---|---|
| A | 0.94 | 1.52 | 2.68 | 5.00 |
| B | 1.14 | 1.71 | 2.85 | 5.14 |
| C | 1.33 | 1.91 | 3.06 | 5.35 |
| D | 0.58 | 1.05 | 2.00 | 3.88 |

GPU price at which arm D's mean total per loop equals the other arm's (full 20-slot loop): no crossing at a positive price, D is cheaper than A at every price from 0 to 4 (D minus A: -0.48 USD at a GPU price of 0, -1.27 at 4); no crossing at a positive price, D is cheaper than C at every price from 0 to 4 (D minus C: -1.28 USD at a GPU price of 0, -1.96 at 4).

Per loop.

| loop | jobs | GPU h | Claude USD | open USD | accepts | total @0.5 | total @1.0 | total @2.0 | total @4.0 |
|---|---|---|---|---|---|---|---|---|---|
| A_02 | 20 | 2.33 | 0.671 | 0.000 | 2 | 1.84 | 3.00 | 5.33 | 9.99 |
| A_04 | 20 | 2.33 | 0.699 | 0.000 | 2 | 1.87 | 3.03 | 5.36 | 10.03 |
| A_06 | 20 | 2.32 | 0.702 | 0.000 | 2 | 1.86 | 3.02 | 5.34 | 9.97 |
| A_08 | 20 | 2.31 | 0.806 | 0.000 | 2 | 1.96 | 3.11 | 5.42 | 10.04 |
| B_02 | 20 | 2.29 | 1.129 | 0.000 | 2 | 2.27 | 3.42 | 5.71 | 10.28 |
| C_02 | 20 | 2.29 | 1.522 | 0.000 | 2 | 2.67 | 3.82 | 6.11 | 10.70 |
| D_02 | 20 | 2.23 | 0.230 | 0.022 | 1 | 1.37 | 2.49 | 4.72 | 9.19 |
| D_04 | 19 | 2.15 | 0.211 | 0.031 | 3 | 1.32 | 2.39 | 4.55 | 8.85 |
| D_06 | 19 | 2.20 | 0.200 | 0.024 | 2 | 1.33 | 2.43 | 4.63 | 9.04 |
| D_08 | 18 | 1.90 | 0.227 | 0.031 | 3 | 1.21 | 2.16 | 4.07 | 7.88 |

## 2. Plateau (slot of the last incumbent change)

Observed if at least 5 charged slots follow it with no change, otherwise censored at 20. Costs run up to and including the plateau slot (a censored loop is charged its whole run), GPU at 2 USD per A100-hour.

| loop | last change | plateau slot | censored | jobs | GPU h | model USD | total USD |
|---|---|---|---|---|---|---|---|
| A_02 | 2 | 2 | no | 3 | 0.35 | 0.107 | 0.81 |
| A_04 | 2 | 2 | no | 3 | 0.35 | 0.060 | 0.76 |
| A_06 | 2 | 2 | no | 3 | 0.35 | 0.059 | 0.75 |
| A_08 | 2 | 2 | no | 3 | 0.35 | 0.059 | 0.76 |
| B_02 | 2 | 2 | no | 3 | 0.35 | 0.110 | 0.81 |
| C_02 | 3 | 3 | no | 4 | 0.49 | 0.261 | 1.24 |
| D_02 | 9 | 9 | no | 10 | 1.18 | 0.141 | 2.51 |
| D_04 | 16 | 20 | yes | 19 | 2.15 | 0.241 | 4.55 |
| D_06 | 19 | 20 | yes | 19 | 2.20 | 0.224 | 4.63 |
| D_08 | 9 | 9 | no | 9 | 1.02 | 0.158 | 2.19 |

| arm | loops | censored | mean slot | Kaplan-Meier median slot | jobs | GPU h | model USD | total USD |
|---|---|---|---|---|---|---|---|---|
| A | 4 | 0 | 2.00 | 2 | 3.0 | 0.35 | 0.071 | 0.77 |
| B | 1 | 0 | 2.00 | 2 | 3.0 | 0.35 | 0.110 | 0.81 |
| C | 1 | 0 | 3.00 | 3 | 4.0 | 0.49 | 0.261 | 1.24 |
| D | 4 | 2 | 14.50 | 9 | 14.2 | 1.64 | 0.191 | 3.47 |

GPU price at which arm D's mean cost to plateau equals the other arm's: no crossing at a positive price, D is dearer than A at every price from 0 to 4 (D minus A: +0.12 USD at a GPU price of 0, +5.28 at 4); D and C cost the same at 0.061 USD per A100-hour (D minus C: -0.07 USD at a GPU price of 0, +4.52 at 4).

## 3. Reach 2^17 (first slot whose incumbent has TOTAL_BATCH_SIZE <= 131072)

| loop | slot | censored | jobs | GPU h | model USD | total USD |
|---|---|---|---|---|---|---|
| A_02 | 2 | no | 3 | 0.35 | 0.107 | 0.81 |
| A_04 | 2 | no | 3 | 0.35 | 0.060 | 0.76 |
| A_06 | 2 | no | 3 | 0.35 | 0.059 | 0.75 |
| A_08 | 2 | no | 3 | 0.35 | 0.059 | 0.76 |
| B_02 | 2 | no | 3 | 0.35 | 0.110 | 0.81 |
| C_02 | 3 | no | 4 | 0.49 | 0.261 | 1.24 |
| D_02 | 9 | no | 10 | 1.18 | 0.141 | 2.51 |
| D_04 | 16 | no | 16 | 1.81 | 0.218 | 3.84 |
| D_06 | 19 | no | 19 | 2.20 | 0.224 | 4.63 |
| D_08 | 6 | no | 6 | 0.68 | 0.096 | 1.46 |

| arm | loops | censored | mean slot | Kaplan-Meier median slot | jobs | total USD |
|---|---|---|---|---|---|---|
| A | 4 | 0 | 2.00 | 2 | 3.0 | 0.77 |
| B | 1 | 0 | 2.00 | 2 | 3.0 | 0.81 |
| C | 1 | 0 | 3.00 | 3 | 4.0 | 1.24 |
| D | 4 | 0 | 12.50 | 9 | 12.8 | 3.11 |

## 4. Stall (final incumbent TOTAL_BATCH_SIZE > 131072)

Improvement = sealed val_bpb gain over the baseline: paired uses the same-wave anchor, global uses the mean of all 30 anchors. Higher is better.

| loop | final batch size | final depth | stalled | draft P2 flag | paired improvement | global-anchor improvement | sealed mean |
|---|---|---|---|---|---|---|---|
| A_02 | 131072 | 8 | no | yes | 0.0623 | 0.0636 | 1.0523 |
| A_04 | 131072 | 8 | no | yes | 0.0661 | 0.0639 | 1.0520 |
| A_06 | 131072 | 8 | no | yes | 0.0640 | 0.0636 | 1.0522 |
| A_08 | 131072 | 8 | no | yes | 0.0642 | 0.0637 | 1.0522 |
| B_02 | 131072 | 8 | no | yes | 0.0651 | 0.0635 | 1.0523 |
| C_02 | 131072 | 6 | no | no | 0.0633 | 0.0636 | 1.0522 |
| D_02 | 131072 | 8 | no | yes | 0.0651 | 0.0635 | 1.0523 |
| D_04 | 131072 | 6 | no | no | 0.0641 | 0.0632 | 1.0527 |
| D_06 | 131072 | 8 | no | yes | 0.0635 | 0.0638 | 1.0521 |
| D_08 | 131072 | 6 | no | no | 0.0632 | 0.0630 | 1.0528 |

| version | stalled mean | not stalled mean | stalled max | not stalled min | groups separated |
|---|---|---|---|---|---|
| paired | n/a | 0.0641 | n/a | 0.0623 | no |
| global | n/a | 0.0635 | n/a | 0.0630 | no |

The draft P2 stall rule (final incumbent has batch size above 2^18 or depth above 6) flags 7 loops; their mean paired improvement is 0.0643 against 0.0635 for the unflagged loops; it misses 0 of the 0 stalled loops.

## 5. Halting rules

quiet k: stop after k consecutive slots with no incumbent change. after_first k: the same, counting starts after the first change. cap N: stop after N slots. Loss = raw official mean of the incumbent at the halt minus the final incumbent's (positive is worse). Within = loss <= 0.004. Dollar fraction at 2 USD per A100-hour.

| rule | share within | mean dollar fraction | mean loss |
|---|---|---|---|
| quiet_k3 | 0.60 | 0.245 | 0.0222 |
| quiet_k5 | 0.70 | 0.393 | 0.0189 |
| quiet_k8 | 0.70 | 0.615 | 0.0121 |
| quiet_k10 | 0.90 | 0.772 | 0.0046 |
| after_first_k3 | 0.70 | 0.365 | 0.0092 |
| after_first_k5 | 0.80 | 0.513 | 0.0058 |
| after_first_k8 | 0.80 | 0.662 | 0.0058 |
| after_first_k10 | 0.90 | 0.772 | 0.0046 |
| cap_n5 | 0.60 | 0.242 | 0.0222 |
| cap_n8 | 0.60 | 0.396 | 0.0130 |
| cap_n10 | 0.80 | 0.497 | 0.0058 |
| cap_n12 | 0.80 | 0.601 | 0.0058 |

Selected on this half (lowest mean dollar fraction among rules with share within >= 80%): **cap_n10**.

Halting slot and loss, per loop.

| loop | quiet_k3 | quiet_k5 | quiet_k8 | quiet_k10 | after_first_k3 | after_first_k5 | after_first_k8 | after_first_k10 | cap_n5 | cap_n8 | cap_n10 | cap_n12 |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| A_02 | 5 / 0.0000 | 7 / 0.0000 | 10 / 0.0000 | 12 / 0.0000 | 5 / 0.0000 | 7 / 0.0000 | 10 / 0.0000 | 12 / 0.0000 | 4 / 0.0000 | 7 / 0.0000 | 9 / 0.0000 | 11 / 0.0000 |
| A_04 | 5 / 0.0000 | 7 / 0.0000 | 10 / 0.0000 | 12 / 0.0000 | 5 / 0.0000 | 7 / 0.0000 | 10 / 0.0000 | 12 / 0.0000 | 4 / 0.0000 | 7 / 0.0000 | 9 / 0.0000 | 11 / 0.0000 |
| A_06 | 5 / 0.0000 | 7 / 0.0000 | 10 / 0.0000 | 12 / 0.0000 | 5 / 0.0000 | 7 / 0.0000 | 10 / 0.0000 | 12 / 0.0000 | 4 / 0.0000 | 7 / 0.0000 | 9 / 0.0000 | 11 / 0.0000 |
| A_08 | 5 / 0.0000 | 7 / 0.0000 | 10 / 0.0000 | 12 / 0.0000 | 5 / 0.0000 | 7 / 0.0000 | 10 / 0.0000 | 12 / 0.0000 | 4 / 0.0000 | 7 / 0.0000 | 9 / 0.0000 | 11 / 0.0000 |
| B_02 | 5 / 0.0000 | 7 / 0.0000 | 10 / 0.0000 | 12 / 0.0000 | 5 / 0.0000 | 7 / 0.0000 | 10 / 0.0000 | 12 / 0.0000 | 4 / 0.0000 | 7 / 0.0000 | 9 / 0.0000 | 11 / 0.0000 |
| C_02 | 6 / 0.0000 | 8 / 0.0000 | 11 / 0.0000 | 13 / 0.0000 | 6 / 0.0000 | 8 / 0.0000 | 11 / 0.0000 | 13 / 0.0000 | 4 / 0.0000 | 7 / 0.0000 | 9 / 0.0000 | 11 / 0.0000 |
| D_02 | 2 / 0.0623 | 4 / 0.0623 | 7 / 0.0623 | 19 / 0.0000 | 12 / 0.0000 | 14 / 0.0000 | 17 / 0.0000 | 19 / 0.0000 | 4 / 0.0623 | 7 / 0.0623 | 9 / 0.0000 | 11 / 0.0000 |
| D_04 | 2 / 0.0634 | 4 / 0.0634 | 15 / 0.0127 | 19 / 0.0000 | 10 / 0.0127 | 12 / 0.0127 | 15 / 0.0127 | 19 / 0.0000 | 4 / 0.0634 | 7 / 0.0127 | 9 / 0.0127 | 11 / 0.0127 |
| D_06 | 2 / 0.0630 | 4 / 0.0630 | 13 / 0.0456 | 15 / 0.0456 | 8 / 0.0456 | 10 / 0.0456 | 13 / 0.0456 | 15 / 0.0456 | 4 / 0.0630 | 7 / 0.0456 | 9 / 0.0456 | 11 / 0.0456 |
| D_08 | 4 / 0.0336 | 14 / 0.0000 | 17 / 0.0000 | 19 / 0.0000 | 4 / 0.0336 | 14 / 0.0000 | 17 / 0.0000 | 19 / 0.0000 | 4 / 0.0336 | 7 / 0.0096 | 9 / 0.0000 | 11 / 0.0000 |

Dollar fraction spent by the halt, per loop.

| loop | quiet_k3 | quiet_k5 | quiet_k8 | quiet_k10 | after_first_k3 | after_first_k5 | after_first_k8 | after_first_k10 | cap_n5 | cap_n8 | cap_n10 | cap_n12 |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| A_02 | 0.30 | 0.40 | 0.55 | 0.65 | 0.30 | 0.40 | 0.55 | 0.65 | 0.26 | 0.40 | 0.50 | 0.60 |
| A_04 | 0.30 | 0.40 | 0.54 | 0.65 | 0.30 | 0.40 | 0.54 | 0.65 | 0.25 | 0.40 | 0.50 | 0.59 |
| A_06 | 0.30 | 0.40 | 0.55 | 0.65 | 0.30 | 0.40 | 0.55 | 0.65 | 0.25 | 0.40 | 0.51 | 0.60 |
| A_08 | 0.29 | 0.39 | 0.54 | 0.65 | 0.29 | 0.39 | 0.54 | 0.65 | 0.24 | 0.39 | 0.49 | 0.59 |
| B_02 | 0.29 | 0.39 | 0.54 | 0.64 | 0.29 | 0.39 | 0.54 | 0.64 | 0.24 | 0.39 | 0.49 | 0.59 |
| C_02 | 0.34 | 0.44 | 0.59 | 0.70 | 0.34 | 0.44 | 0.59 | 0.70 | 0.25 | 0.40 | 0.48 | 0.59 |
| D_02 | 0.16 | 0.26 | 0.42 | 1.00 | 0.64 | 0.73 | 0.89 | 1.00 | 0.26 | 0.42 | 0.53 | 0.64 |
| D_04 | 0.11 | 0.21 | 0.79 | 1.00 | 0.52 | 0.63 | 0.79 | 1.00 | 0.21 | 0.37 | 0.48 | 0.57 |
| D_06 | 0.11 | 0.21 | 0.68 | 0.78 | 0.42 | 0.52 | 0.68 | 0.78 | 0.21 | 0.37 | 0.47 | 0.58 |
| D_08 | 0.24 | 0.82 | 0.94 | 1.00 | 0.24 | 0.82 | 0.94 | 1.00 | 0.24 | 0.42 | 0.54 | 0.65 |

## 6. Noise replay and acceptance-rule frontier (sentinels and arm A loops only)

Noise sd from 52 sentinel runs (raw official): 0.00378. Converged regime = candidate first runs after the incumbent reached 2^17, in arm A loops ['A_02', 'A_04', 'A_06', 'A_08']. Observed: 45 candidates, 24 would-be keeps, 23 audits, 23 audit rejects, 0 audit accepts. The logged keep decision matches 'first run strictly below the incumbent raw mean' for 45 of 45 candidates. Sd of the converged first runs is 0.73 times the sentinel sd.

Pure-noise prediction (true effect zero for every candidate, level = mean converged first run 1.0524, R1's rule, sigma 0.004743 in the audit z). Intervals are 90 percent predictive (include uncertainty in sd and level).

| noise | count | observed | predicted mean | 90% interval | inside | P(sim >= observed) |
|---|---|---|---|---|---|---|
| gaussian | candidates | 45 | 47.57 | 43 to 53 | yes | 0.846 |
| gaussian | keeps | 24 | 21.63 | 16 to 27 | yes | 0.283 |
| gaussian | audits | 23 | 20.43 | 15 to 25 | yes | 0.251 |
| gaussian | audit_rejects | 23 | 20.19 | 15 to 25 | yes | 0.239 |
| gaussian | audit_accepts | 0 | 0.24 | 0 to 1 | yes | 1.000 |
| bootstrap | candidates | 45 | 47.95 | 44 to 52 | yes | 0.917 |
| bootstrap | keeps | 24 | 21.28 | 17 to 26 | yes | 0.206 |
| bootstrap | audits | 23 | 20.05 | 16 to 24 | yes | 0.171 |
| bootstrap | audit_rejects | 23 | 20.02 | 16 to 24 | yes | 0.168 |
| bootstrap | audit_accepts | 0 | 0.03 | 0 to 0 | yes | 1.000 |
| gaussian_conv | candidates | 45 | 47.36 | 43 to 52 | yes | 0.836 |
| gaussian_conv | keeps | 24 | 21.89 | 17 to 27 | yes | 0.310 |
| gaussian_conv | audits | 23 | 20.64 | 16 to 25 | yes | 0.271 |
| gaussian_conv | audit_rejects | 23 | 20.61 | 16 to 25 | yes | 0.269 |
| gaussian_conv | audit_accepts | 0 | 0.02 | 0 to 0 | yes | 1.000 |

Frontier by simulation. One slot is one candidate decision; each loop has 20 of them against an incumbent measured by 2 runs. Rules: keep_if_lower (first run below the incumbent mean), zband (R1's rule: reject z below 0.75, accept at 2.0 or above, replicate in between and accept at 1.5), mean2 (mean of 2 runs below the incumbent), median3 (median of 3 runs below the incumbent). A keep moves the incumbent to the candidate's own measurement. Noise: gaussian and bootstrap use the sentinel sd and residuals; gaussian_conv uses the sd of the converged first runs (sensitivity).

Noise: gaussian. False keeps are per 20 slots with no true effect; the other columns are the chance one decision keeps a candidate that is truly better by that many bpb, with runs per decision in brackets.

| rule | runs/slot (no effect) | false keeps / 20 | true-keep @0.001 (runs) | true-keep @0.002 (runs) | true-keep @0.004 (runs) | true-keep @0.008 (runs) |
|---|---|---|---|---|---|---|
| keep_if_lower | 1.00 | 2.76 | 0.584 (1.00) | 0.668 (1.00) | 0.807 (1.00) | 0.959 (1.00) |
| zband | 1.11 | 0.25 | 0.050 (1.22) | 0.078 (1.29) | 0.199 (1.42) | 0.572 (1.57) |
| mean2 | 2.00 | 2.63 | 0.609 (2.00) | 0.694 (2.00) | 0.854 (2.00) | 0.983 (2.00) |
| median3 | 3.00 | 2.61 | 0.608 (3.00) | 0.708 (3.00) | 0.860 (3.00) | 0.985 (3.00) |

Same, as the mean keep rate over a stream of 20 consecutive truly better candidates (later decisions face a bar already pulled down by earlier keeps).

| rule | stream keep rate @0.001 (runs/slot) | stream keep rate @0.002 (runs/slot) | stream keep rate @0.004 (runs/slot) | stream keep rate @0.008 (runs/slot) |
|---|---|---|---|---|
| keep_if_lower | 0.197 (1.00) | 0.286 (1.00) | 0.550 (1.00) | 0.901 (1.00) |
| zband | 0.017 (1.14) | 0.024 (1.16) | 0.043 (1.20) | 0.141 (1.36) |
| mean2 | 0.220 (2.00) | 0.378 (2.00) | 0.737 (2.00) | 0.979 (2.00) |
| median3 | 0.225 (3.00) | 0.396 (3.00) | 0.761 (3.00) | 0.984 (3.00) |

Noise: bootstrap. False keeps are per 20 slots with no true effect; the other columns are the chance one decision keeps a candidate that is truly better by that many bpb, with runs per decision in brackets.

| rule | runs/slot (no effect) | false keeps / 20 | true-keep @0.001 (runs) | true-keep @0.002 (runs) | true-keep @0.004 (runs) | true-keep @0.008 (runs) |
|---|---|---|---|---|---|---|
| keep_if_lower | 1.00 | 2.61 | 0.566 (1.00) | 0.646 (1.00) | 0.786 (1.00) | 0.968 (1.00) |
| zband | 1.13 | 0.23 | 0.051 (1.26) | 0.084 (1.32) | 0.202 (1.40) | 0.569 (1.52) |
| mean2 | 2.00 | 2.63 | 0.607 (2.00) | 0.695 (2.00) | 0.845 (2.00) | 0.985 (2.00) |
| median3 | 3.00 | 2.53 | 0.575 (3.00) | 0.661 (3.00) | 0.817 (3.00) | 0.981 (3.00) |

Same, as the mean keep rate over a stream of 20 consecutive truly better candidates (later decisions face a bar already pulled down by earlier keeps).

| rule | stream keep rate @0.001 (runs/slot) | stream keep rate @0.002 (runs/slot) | stream keep rate @0.004 (runs/slot) | stream keep rate @0.008 (runs/slot) |
|---|---|---|---|---|
| keep_if_lower | 0.246 (1.00) | 0.368 (1.00) | 0.536 (1.00) | 0.893 (1.00) |
| zband | 0.016 (1.16) | 0.023 (1.18) | 0.041 (1.23) | 0.133 (1.41) |
| mean2 | 0.223 (2.00) | 0.366 (2.00) | 0.733 (2.00) | 0.983 (2.00) |
| median3 | 0.246 (3.00) | 0.356 (3.00) | 0.616 (3.00) | 0.969 (3.00) |

Noise: gaussian_conv. False keeps are per 20 slots with no true effect; the other columns are the chance one decision keeps a candidate that is truly better by that many bpb, with runs per decision in brackets.

| rule | runs/slot (no effect) | false keeps / 20 | true-keep @0.001 (runs) | true-keep @0.002 (runs) | true-keep @0.004 (runs) | true-keep @0.008 (runs) |
|---|---|---|---|---|---|---|
| keep_if_lower | 1.00 | 2.76 | 0.616 (1.00) | 0.723 (1.00) | 0.886 (1.00) | 0.991 (1.00) |
| zband | 1.09 | 0.06 | 0.014 (1.16) | 0.034 (1.24) | 0.126 (1.45) | 0.618 (1.71) |
| mean2 | 2.00 | 2.65 | 0.644 (2.00) | 0.764 (2.00) | 0.923 (2.00) | 0.998 (2.00) |
| median3 | 3.00 | 2.61 | 0.647 (3.00) | 0.774 (3.00) | 0.929 (3.00) | 0.999 (3.00) |

Same, as the mean keep rate over a stream of 20 consecutive truly better candidates (later decisions face a bar already pulled down by earlier keeps).

| rule | stream keep rate @0.001 (runs/slot) | stream keep rate @0.002 (runs/slot) | stream keep rate @0.004 (runs/slot) | stream keep rate @0.008 (runs/slot) |
|---|---|---|---|---|
| keep_if_lower | 0.225 (1.00) | 0.375 (1.00) | 0.723 (1.00) | 0.975 (1.00) |
| zband | 0.007 (1.13) | 0.013 (1.17) | 0.032 (1.25) | 0.189 (1.51) |
| mean2 | 0.267 (2.00) | 0.521 (2.00) | 0.885 (2.00) | 0.998 (2.00) |
| median3 | 0.278 (3.00) | 0.543 (3.00) | 0.903 (3.00) | 0.999 (3.00) |

## 7. GPU offsets (one-way ANOVA on the sentinels)

F(3, 48) = 0.815, p = 0.492.

| GPU | n | mean | sd |
|---|---|---|---|
| 0 | 9 | 1.11440 | 0.00367 |
| 1 | 10 | 1.11239 | 0.00327 |
| 2 | 21 | 1.11411 | 0.00381 |
| 3 | 12 | 1.11480 | 0.00424 |

## Spread of slot endpoints in arm D (for sizing the P2 slot margin)

Sd of the slot (censored loops at 20), the standard error of a difference between two groups of 24 loops, and the half-width of its 90 percent interval.

| endpoint | loops | sd | SE of 24 vs 24 difference | 90% half-width |
|---|---|---|---|---|
| plateau_slot_D | 4 | 6.35 | 1.83 | 3.02 |
| reach_slot_D | 4 | 6.03 | 1.74 | 2.86 |

## Decision lines (verdict)

Claim 1: A minus D mean slots to plateau (censored at 20) = -12.50, 90% bootstrap interval -18.00 to -7.00, excludes zero: True. Censored loops: A 0, D 2.
Sensitivity excluding censored loops: diff -7.00, interval -7.00 to -7.00, excludes zero: True.

Halting: rule after_first_k5, share within 0.80, mean dollar fraction 0.513, pass (>= 80% within at <= 60% of dollars): True.

N2 Test 1: observed converged-regime audit rejects 23, 90% predictive interval 15 to 25, inside: True.
