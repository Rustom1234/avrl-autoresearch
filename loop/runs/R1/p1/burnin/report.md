# Project 1 re-accounting of R1: burnin half

Zero GPU, zero API. Every number comes from the files in loop/runs/R1. Loops in this half: A_01, A_03, A_05, A_07, B_01, B_03, C_01, C_03, D_01, D_03, D_05, D_07.
Dollars: Claude at cost_usd_plan, open models at the per-token prices in loop/models/open_prices.json, GPU at the price shown. GPU hours are measured wall seconds from slots/*/result.json. Slots are 0-based; a censored loop is recorded at slot 20.

Parameters: {"half": "burnin", "sims": 20000, "boot": 10000, "seed": 1, "sigma_used": 0.004743, "base_official": 1.1148854674525384, "base_tbs": 524288, "code_sha256": "418bfc38c98c96d5343d41ac081935010af2b88126cc740a3437ca2e34ab61f8"}.

## 1. Cost reconciliation

Mean per loop, by arm.

| arm | loops | GPU jobs | GPU hours | Claude USD | open USD | accepted changes | total @0.5 | total @1.0 | total @2.0 | total @4.0 |
|---|---|---|---|---|---|---|---|---|---|---|
| A | 4 | 20.0 | 2.32 | 0.839 | 0.000 | 2.25 | 2.00 | 3.15 | 5.47 | 10.10 |
| B | 2 | 20.0 | 2.29 | 1.030 | 0.000 | 2.00 | 2.17 | 3.32 | 5.60 | 10.18 |
| C | 2 | 20.0 | 2.31 | 1.402 | 0.000 | 1.50 | 2.56 | 3.71 | 6.02 | 10.63 |
| D | 4 | 18.8 | 2.12 | 0.215 | 0.026 | 2.25 | 1.30 | 2.36 | 4.49 | 8.73 |

Dollars per accepted change (arm total dollars divided by arm accepted changes).

| arm | @0.5 | @1.0 | @2.0 | @4.0 |
|---|---|---|---|---|
| A | 0.89 | 1.40 | 2.43 | 4.49 |
| B | 1.09 | 1.66 | 2.80 | 5.09 |
| C | 1.70 | 2.47 | 4.01 | 7.09 |
| D | 0.58 | 1.05 | 1.99 | 3.88 |

GPU price at which arm D's mean total per loop equals the other arm's (full 20-slot loop): no crossing at a positive price, D is cheaper than A at every price from 0 to 4 (D minus A: -0.60 USD at a GPU price of 0, -1.37 at 4); no crossing at a positive price, D is cheaper than C at every price from 0 to 4 (D minus C: -1.16 USD at a GPU price of 0, -1.90 at 4).

Per loop.

| loop | jobs | GPU h | Claude USD | open USD | accepts | total @0.5 | total @1.0 | total @2.0 | total @4.0 |
|---|---|---|---|---|---|---|---|---|---|
| A_01 | 20 | 2.31 | 0.986 | 0.000 | 3 | 2.14 | 3.29 | 5.60 | 10.22 |
| A_03 | 20 | 2.34 | 0.700 | 0.000 | 2 | 1.87 | 3.04 | 5.37 | 10.04 |
| A_05 | 20 | 2.31 | 0.869 | 0.000 | 2 | 2.02 | 3.18 | 5.49 | 10.10 |
| A_07 | 20 | 2.31 | 0.801 | 0.000 | 2 | 1.96 | 3.11 | 5.42 | 10.05 |
| B_01 | 20 | 2.29 | 1.062 | 0.000 | 2 | 2.21 | 3.35 | 5.64 | 10.21 |
| B_03 | 20 | 2.29 | 0.998 | 0.000 | 2 | 2.14 | 3.28 | 5.57 | 10.14 |
| C_01 | 20 | 2.35 | 1.531 | 0.000 | 1 | 2.71 | 3.89 | 6.24 | 10.95 |
| C_03 | 20 | 2.26 | 1.273 | 0.000 | 2 | 2.40 | 3.53 | 5.79 | 10.32 |
| D_01 | 18 | 2.00 | 0.219 | 0.032 | 2 | 1.25 | 2.25 | 4.26 | 8.26 |
| D_03 | 20 | 2.31 | 0.219 | 0.019 | 3 | 1.39 | 2.55 | 4.86 | 9.48 |
| D_05 | 18 | 1.99 | 0.201 | 0.030 | 2 | 1.23 | 2.22 | 4.21 | 8.19 |
| D_07 | 19 | 2.18 | 0.223 | 0.023 | 2 | 1.34 | 2.43 | 4.62 | 8.99 |

## 2. Plateau (slot of the last incumbent change)

Observed if at least 5 charged slots follow it with no change, otherwise censored at 20. Costs run up to and including the plateau slot (a censored loop is charged its whole run), GPU at 2 USD per A100-hour.

| loop | last change | plateau slot | censored | jobs | GPU h | model USD | total USD |
|---|---|---|---|---|---|---|---|
| A_01 | 5 | 5 | no | 6 | 0.70 | 0.221 | 1.63 |
| A_03 | 2 | 2 | no | 3 | 0.35 | 0.108 | 0.81 |
| A_05 | 2 | 2 | no | 3 | 0.35 | 0.108 | 0.80 |
| A_07 | 2 | 2 | no | 3 | 0.35 | 0.108 | 0.81 |
| B_01 | 2 | 2 | no | 3 | 0.35 | 0.110 | 0.81 |
| B_03 | 2 | 2 | no | 3 | 0.35 | 0.110 | 0.81 |
| C_01 | 1 | 1 | no | 2 | 0.24 | 0.138 | 0.61 |
| C_03 | 9 | 9 | no | 10 | 1.15 | 0.578 | 2.88 |
| D_01 | 10 | 10 | no | 9 | 1.02 | 0.177 | 2.22 |
| D_03 | 19 | 20 | yes | 20 | 2.31 | 0.238 | 4.86 |
| D_05 | 5 | 5 | no | 5 | 0.56 | 0.067 | 1.20 |
| D_07 | 8 | 8 | no | 9 | 1.05 | 0.138 | 2.23 |

| arm | loops | censored | mean slot | Kaplan-Meier median slot | jobs | GPU h | model USD | total USD |
|---|---|---|---|---|---|---|---|---|
| A | 4 | 0 | 2.75 | 2 | 3.8 | 0.44 | 0.136 | 1.01 |
| B | 2 | 0 | 2.00 | 2 | 3.0 | 0.35 | 0.110 | 0.81 |
| C | 2 | 0 | 5.00 | 1 | 6.0 | 0.69 | 0.358 | 1.75 |
| D | 4 | 1 | 10.75 | 8 | 10.8 | 1.24 | 0.155 | 2.63 |

GPU price at which arm D's mean cost to plateau equals the other arm's: no crossing at a positive price, D is dearer than A at every price from 0 to 4 (D minus A: +0.02 USD at a GPU price of 0, +3.21 at 4); D and C cost the same at 0.374 USD per A100-hour (D minus C: -0.20 USD at a GPU price of 0, +1.96 at 4).

## 3. Reach 2^17 (first slot whose incumbent has TOTAL_BATCH_SIZE <= 131072)

| loop | slot | censored | jobs | GPU h | model USD | total USD |
|---|---|---|---|---|---|---|
| A_01 | 2 | no | 3 | 0.35 | 0.106 | 0.80 |
| A_03 | 2 | no | 3 | 0.35 | 0.108 | 0.81 |
| A_05 | 2 | no | 3 | 0.35 | 0.108 | 0.80 |
| A_07 | 2 | no | 3 | 0.35 | 0.108 | 0.81 |
| B_01 | 2 | no | 3 | 0.35 | 0.110 | 0.81 |
| B_03 | 2 | no | 3 | 0.35 | 0.110 | 0.81 |
| C_01 | 1 | no | 2 | 0.24 | 0.138 | 0.61 |
| C_03 | 20 | yes | 20 | 2.26 | 1.273 | 5.79 |
| D_01 | 20 | yes | 18 | 2.00 | 0.251 | 4.26 |
| D_03 | 20 | yes | 20 | 2.31 | 0.238 | 4.86 |
| D_05 | 20 | yes | 18 | 1.99 | 0.231 | 4.21 |
| D_07 | 8 | no | 9 | 1.05 | 0.138 | 2.23 |

| arm | loops | censored | mean slot | Kaplan-Meier median slot | jobs | total USD |
|---|---|---|---|---|---|---|
| A | 4 | 0 | 2.00 | 2 | 3.0 | 0.81 |
| B | 2 | 0 | 2.00 | 2 | 3.0 | 0.81 |
| C | 2 | 1 | 10.50 | 1 | 11.0 | 3.20 |
| D | 4 | 3 | 17.00 | not reached | 16.2 | 3.89 |

## 4. Stall (final incumbent TOTAL_BATCH_SIZE > 131072)

Improvement = sealed val_bpb gain over the baseline: paired uses the same-wave anchor, global uses the mean of all 30 anchors. Higher is better.

| loop | final batch size | final depth | stalled | draft P2 flag | paired improvement | global-anchor improvement | sealed mean |
|---|---|---|---|---|---|---|---|
| A_01 | 131072 | 8 | no | yes | 0.0597 | 0.0639 | 1.0519 |
| A_03 | 131072 | 8 | no | yes | 0.0630 | 0.0638 | 1.0521 |
| A_05 | 131072 | 8 | no | yes | 0.0634 | 0.0636 | 1.0522 |
| A_07 | 131072 | 8 | no | yes | 0.0637 | 0.0635 | 1.0523 |
| B_01 | 131072 | 8 | no | yes | 0.0607 | 0.0640 | 1.0519 |
| B_03 | 131072 | 8 | no | yes | 0.0629 | 0.0637 | 1.0521 |
| C_01 | 131072 | 8 | no | yes | 0.0657 | 0.0633 | 1.0526 |
| C_03 | 262144 | 6 | yes | no | 0.0532 | 0.0500 | 1.0658 |
| D_01 | 262144 | 6 | yes | no | 0.0519 | 0.0501 | 1.0657 |
| D_03 | 262144 | 6 | yes | no | 0.0563 | 0.0585 | 1.0574 |
| D_05 | 262144 | 6 | yes | no | 0.0466 | 0.0502 | 1.0657 |
| D_07 | 131072 | 8 | no | yes | 0.0651 | 0.0638 | 1.0520 |

| version | stalled mean | not stalled mean | stalled max | not stalled min | groups separated |
|---|---|---|---|---|---|
| paired | 0.0520 | 0.0630 | 0.0563 | 0.0597 | yes |
| global | 0.0522 | 0.0637 | 0.0585 | 0.0633 | yes |

The draft P2 stall rule (final incumbent has batch size above 2^18 or depth above 6) flags 8 loops; their mean paired improvement is 0.0630 against 0.0520 for the unflagged loops; it misses 4 of the 4 stalled loops.

## 5. Halting rules

quiet k: stop after k consecutive slots with no incumbent change. after_first k: the same, counting starts after the first change. cap N: stop after N slots. Loss = raw official mean of the incumbent at the halt minus the final incumbent's (positive is worse). Within = loss <= 0.004. Dollar fraction at 2 USD per A100-hour.

| rule | share within | mean dollar fraction | mean loss |
|---|---|---|---|
| quiet_k3 | 0.58 | 0.252 | 0.0181 |
| quiet_k5 | 0.75 | 0.415 | 0.0114 |
| quiet_k8 | 0.92 | 0.656 | 0.0015 |
| quiet_k10 | 1.00 | 0.784 | 0.0000 |
| after_first_k3 | 0.67 | 0.340 | 0.0091 |
| after_first_k5 | 0.83 | 0.482 | 0.0050 |
| after_first_k8 | 0.92 | 0.656 | 0.0015 |
| after_first_k10 | 1.00 | 0.784 | 0.0000 |
| cap_n5 | 0.50 | 0.243 | 0.0158 |
| cap_n8 | 0.67 | 0.397 | 0.0115 |
| cap_n10 | 0.83 | 0.498 | 0.0050 |
| cap_n12 | 0.92 | 0.598 | 0.0036 |

Selected on this half (lowest mean dollar fraction among rules with share within >= 80%): **after_first_k5**.

Halting slot and loss, per loop.

| loop | quiet_k3 | quiet_k5 | quiet_k8 | quiet_k10 | after_first_k3 | after_first_k5 | after_first_k8 | after_first_k10 | cap_n5 | cap_n8 | cap_n10 | cap_n12 |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| A_01 | 8 / 0.0000 | 10 / 0.0000 | 13 / 0.0000 | 15 / 0.0000 | 8 / 0.0000 | 10 / 0.0000 | 13 / 0.0000 | 15 / 0.0000 | 4 / 0.0044 | 7 / 0.0000 | 9 / 0.0000 | 11 / 0.0000 |
| A_03 | 5 / 0.0000 | 7 / 0.0000 | 10 / 0.0000 | 12 / 0.0000 | 5 / 0.0000 | 7 / 0.0000 | 10 / 0.0000 | 12 / 0.0000 | 4 / 0.0000 | 7 / 0.0000 | 9 / 0.0000 | 11 / 0.0000 |
| A_05 | 5 / 0.0000 | 7 / 0.0000 | 10 / 0.0000 | 12 / 0.0000 | 5 / 0.0000 | 7 / 0.0000 | 10 / 0.0000 | 12 / 0.0000 | 4 / 0.0000 | 7 / 0.0000 | 9 / 0.0000 | 11 / 0.0000 |
| A_07 | 5 / 0.0000 | 7 / 0.0000 | 10 / 0.0000 | 12 / 0.0000 | 5 / 0.0000 | 7 / 0.0000 | 10 / 0.0000 | 12 / 0.0000 | 4 / 0.0000 | 7 / 0.0000 | 9 / 0.0000 | 11 / 0.0000 |
| B_01 | 5 / 0.0000 | 7 / 0.0000 | 10 / 0.0000 | 12 / 0.0000 | 5 / 0.0000 | 7 / 0.0000 | 10 / 0.0000 | 12 / 0.0000 | 4 / 0.0000 | 7 / 0.0000 | 9 / 0.0000 | 11 / 0.0000 |
| B_03 | 5 / 0.0000 | 7 / 0.0000 | 10 / 0.0000 | 12 / 0.0000 | 5 / 0.0000 | 7 / 0.0000 | 10 / 0.0000 | 12 / 0.0000 | 4 / 0.0000 | 7 / 0.0000 | 9 / 0.0000 | 11 / 0.0000 |
| C_01 | 4 / 0.0000 | 6 / 0.0000 | 9 / 0.0000 | 11 / 0.0000 | 4 / 0.0000 | 6 / 0.0000 | 9 / 0.0000 | 11 / 0.0000 | 4 / 0.0000 | 7 / 0.0000 | 9 / 0.0000 | 11 / 0.0000 |
| C_03 | 2 / 0.0498 | 14 / 0.0000 | 17 / 0.0000 | 19 / 0.0000 | 7 / 0.0181 | 14 / 0.0000 | 17 / 0.0000 | 19 / 0.0000 | 4 / 0.0181 | 7 / 0.0181 | 9 / 0.0000 | 11 / 0.0000 |
| D_01 | 4 / 0.0178 | 6 / 0.0178 | 9 / 0.0178 | 19 / 0.0000 | 4 / 0.0178 | 6 / 0.0178 | 9 / 0.0178 | 19 / 0.0000 | 4 / 0.0178 | 7 / 0.0178 | 9 / 0.0178 | 11 / 0.0000 |
| D_03 | 2 / 0.0555 | 4 / 0.0555 | 19 / 0.0000 | 19 / 0.0000 | 9 / 0.0426 | 11 / 0.0426 | 19 / 0.0000 | 19 / 0.0000 | 4 / 0.0555 | 7 / 0.0426 | 9 / 0.0426 | 11 / 0.0426 |
| D_05 | 3 / 0.0305 | 10 / 0.0000 | 13 / 0.0000 | 15 / 0.0000 | 3 / 0.0305 | 10 / 0.0000 | 13 / 0.0000 | 15 / 0.0000 | 4 / 0.0305 | 7 / 0.0000 | 9 / 0.0000 | 11 / 0.0000 |
| D_07 | 2 / 0.0631 | 4 / 0.0631 | 16 / 0.0000 | 18 / 0.0000 | 11 / 0.0000 | 13 / 0.0000 | 16 / 0.0000 | 18 / 0.0000 | 4 / 0.0631 | 7 / 0.0598 | 9 / 0.0000 | 11 / 0.0000 |

Dollar fraction spent by the halt, per loop.

| loop | quiet_k3 | quiet_k5 | quiet_k8 | quiet_k10 | after_first_k3 | after_first_k5 | after_first_k8 | after_first_k10 | cap_n5 | cap_n8 | cap_n10 | cap_n12 |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| A_01 | 0.44 | 0.54 | 0.70 | 0.80 | 0.44 | 0.54 | 0.70 | 0.80 | 0.25 | 0.39 | 0.49 | 0.59 |
| A_03 | 0.31 | 0.41 | 0.56 | 0.66 | 0.31 | 0.41 | 0.56 | 0.66 | 0.26 | 0.41 | 0.51 | 0.61 |
| A_05 | 0.30 | 0.40 | 0.54 | 0.64 | 0.30 | 0.40 | 0.54 | 0.64 | 0.25 | 0.40 | 0.49 | 0.59 |
| A_07 | 0.31 | 0.40 | 0.55 | 0.65 | 0.31 | 0.40 | 0.55 | 0.65 | 0.25 | 0.40 | 0.50 | 0.60 |
| B_01 | 0.30 | 0.40 | 0.55 | 0.65 | 0.30 | 0.40 | 0.55 | 0.65 | 0.25 | 0.40 | 0.50 | 0.60 |
| B_03 | 0.30 | 0.40 | 0.55 | 0.65 | 0.30 | 0.40 | 0.55 | 0.65 | 0.25 | 0.40 | 0.50 | 0.60 |
| C_01 | 0.24 | 0.33 | 0.50 | 0.58 | 0.24 | 0.33 | 0.50 | 0.58 | 0.24 | 0.39 | 0.50 | 0.58 |
| C_03 | 0.16 | 0.73 | 0.90 | 1.00 | 0.40 | 0.73 | 0.90 | 1.00 | 0.26 | 0.40 | 0.50 | 0.59 |
| D_01 | 0.18 | 0.29 | 0.46 | 1.00 | 0.18 | 0.29 | 0.46 | 1.00 | 0.18 | 0.35 | 0.46 | 0.57 |
| D_03 | 0.15 | 0.25 | 1.00 | 1.00 | 0.50 | 0.59 | 1.00 | 1.00 | 0.25 | 0.40 | 0.50 | 0.59 |
| D_05 | 0.17 | 0.55 | 0.68 | 0.78 | 0.17 | 0.55 | 0.68 | 0.78 | 0.22 | 0.40 | 0.50 | 0.62 |
| D_07 | 0.16 | 0.26 | 0.89 | 1.00 | 0.63 | 0.73 | 0.89 | 1.00 | 0.26 | 0.43 | 0.53 | 0.63 |

## 6. Noise replay and acceptance-rule frontier (sentinels and arm A loops only)

Noise sd from 52 sentinel runs (raw official): 0.00378. Converged regime = candidate first runs after the incumbent reached 2^17, in arm A loops ['A_01', 'A_03', 'A_05', 'A_07']. Observed: 50 candidates, 19 would-be keeps, 18 audits, 17 audit rejects, 1 audit accepts. The logged keep decision matches 'first run strictly below the incumbent raw mean' for 50 of 50 candidates. Sd of the converged first runs is 0.60 times the sentinel sd.

Pure-noise prediction (true effect zero for every candidate, level = mean converged first run 1.0514, R1's rule, sigma 0.004743 in the audit z). Intervals are 90 percent predictive (include uncertainty in sd and level).

| noise | count | observed | predicted mean | 90% interval | inside | P(sim >= observed) |
|---|---|---|---|---|---|---|
| gaussian | candidates | 50 | 46.95 | 42 to 52 | yes | 0.188 |
| gaussian | keeps | 19 | 22.27 | 17 to 27 | yes | 0.882 |
| gaussian | audits | 18 | 21.05 | 16 to 26 | yes | 0.879 |
| gaussian | audit_rejects | 17 | 20.70 | 15 to 26 | yes | 0.898 |
| gaussian | audit_accepts | 1 | 0.35 | 0 to 1 | yes | 0.296 |
| bootstrap | candidates | 50 | 47.51 | 43 to 52 | yes | 0.213 |
| bootstrap | keeps | 19 | 21.73 | 17 to 26 | yes | 0.888 |
| bootstrap | audits | 18 | 20.49 | 16 to 25 | yes | 0.879 |
| bootstrap | audit_rejects | 17 | 20.37 | 16 to 24 | yes | 0.928 |
| bootstrap | audit_accepts | 1 | 0.13 | 0 to 1 | yes | 0.123 |
| gaussian_conv | candidates | 50 | 46.16 | 42 to 51 | yes | 0.111 |
| gaussian_conv | keeps | 19 | 23.17 | 18 to 28 | yes | 0.947 |
| gaussian_conv | audits | 18 | 21.84 | 17 to 26 | yes | 0.944 |
| gaussian_conv | audit_rejects | 17 | 21.83 | 17 to 26 | yes | 0.973 |
| gaussian_conv | audit_accepts | 1 | 0.01 | 0 to 0 | NO | 0.007 |

Frontier by simulation. One slot is one candidate decision; each loop has 20 of them against an incumbent measured by 2 runs. Rules: keep_if_lower (first run below the incumbent mean), zband (R1's rule: reject z below 0.75, accept at 2.0 or above, replicate in between and accept at 1.5), mean2 (mean of 2 runs below the incumbent), median3 (median of 3 runs below the incumbent). A keep moves the incumbent to the candidate's own measurement. Noise: gaussian and bootstrap use the sentinel sd and residuals; gaussian_conv uses the sd of the converged first runs (sensitivity).

Noise: gaussian. False keeps are per 20 slots with no true effect; the other columns are the chance one decision keeps a candidate that is truly better by that many bpb, with runs per decision in brackets.

| rule | runs/slot (no effect) | false keeps / 20 | true-keep @0.001 (runs) | true-keep @0.002 (runs) | true-keep @0.004 (runs) | true-keep @0.008 (runs) |
|---|---|---|---|---|---|---|
| keep_if_lower | 1.00 | 2.76 | 0.584 (1.00) | 0.669 (1.00) | 0.806 (1.00) | 0.959 (1.00) |
| zband | 1.11 | 0.25 | 0.050 (1.22) | 0.078 (1.29) | 0.199 (1.42) | 0.572 (1.57) |
| mean2 | 2.00 | 2.63 | 0.609 (2.00) | 0.694 (2.00) | 0.854 (2.00) | 0.983 (2.00) |
| median3 | 3.00 | 2.61 | 0.608 (3.00) | 0.708 (3.00) | 0.860 (3.00) | 0.985 (3.00) |

Same, as the mean keep rate over a stream of 20 consecutive truly better candidates (later decisions face a bar already pulled down by earlier keeps).

| rule | stream keep rate @0.001 (runs/slot) | stream keep rate @0.002 (runs/slot) | stream keep rate @0.004 (runs/slot) | stream keep rate @0.008 (runs/slot) |
|---|---|---|---|---|
| keep_if_lower | 0.197 (1.00) | 0.286 (1.00) | 0.550 (1.00) | 0.901 (1.00) |
| zband | 0.017 (1.14) | 0.024 (1.16) | 0.043 (1.20) | 0.141 (1.36) |
| mean2 | 0.220 (2.00) | 0.378 (2.00) | 0.737 (2.00) | 0.979 (2.00) |
| median3 | 0.225 (3.00) | 0.396 (3.00) | 0.762 (3.00) | 0.984 (3.00) |

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
| keep_if_lower | 1.00 | 2.76 | 0.641 (1.00) | 0.766 (1.00) | 0.931 (1.00) | 0.998 (1.00) |
| zband | 1.06 | 0.01 | 0.004 (1.11) | 0.012 (1.20) | 0.082 (1.44) | 0.651 (1.81) |
| mean2 | 2.00 | 2.65 | 0.674 (2.00) | 0.810 (2.00) | 0.959 (2.00) | 1.000 (2.00) |
| median3 | 3.00 | 2.61 | 0.678 (3.00) | 0.821 (3.00) | 0.965 (3.00) | 1.000 (3.00) |

Same, as the mean keep rate over a stream of 20 consecutive truly better candidates (later decisions face a bar already pulled down by earlier keeps).

| rule | stream keep rate @0.001 (runs/slot) | stream keep rate @0.002 (runs/slot) | stream keep rate @0.004 (runs/slot) | stream keep rate @0.008 (runs/slot) |
|---|---|---|---|---|
| keep_if_lower | 0.254 (1.00) | 0.461 (1.00) | 0.829 (1.00) | 0.993 (1.00) |
| zband | 0.002 (1.10) | 0.006 (1.16) | 0.024 (1.28) | 0.242 (1.64) |
| mean2 | 0.316 (2.00) | 0.636 (2.00) | 0.948 (2.00) | 1.000 (2.00) |
| median3 | 0.332 (3.00) | 0.662 (3.00) | 0.958 (3.00) | 1.000 (3.00) |

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
| plateau_slot_D | 4 | 6.50 | 1.88 | 3.09 |
| reach_slot_D | 4 | 6.00 | 1.73 | 2.85 |

## Decision lines (dry run on burn-in, not a verdict)

Claim 1: A minus D mean slots to plateau (censored at 20) = -8.00, 90% bootstrap interval -13.00 to -3.75, excludes zero: True. Censored loops: A 0, D 1.
Sensitivity excluding censored loops: diff -4.92, interval -7.25 to -2.50, excludes zero: True.

Halting: rule after_first_k5, share within 0.83, mean dollar fraction 0.482, pass (>= 80% within at <= 60% of dollars): True.

N2 Test 1: observed converged-regime audit rejects 17, 90% predictive interval 15 to 26, inside: True.
