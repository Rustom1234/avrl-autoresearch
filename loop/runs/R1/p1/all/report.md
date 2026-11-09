# Project 1 re-accounting of R1: all half

Zero GPU, zero API. Every number comes from the files in loop/runs/R1. Loops in this half: A_01, A_02, A_03, A_04, A_05, A_06, A_07, A_08, B_01, B_02, B_03, C_01, C_02, C_03, D_01, D_02, D_03, D_04, D_05, D_06, D_07, D_08.
Dollars: Claude at cost_usd_plan, open models at the per-token prices in loop/models/open_prices.json, GPU at the price shown. GPU hours are measured wall seconds from slots/*/result.json. Slots are 0-based; a censored loop is recorded at slot 20.

Parameters: {"half": "all", "sims": 20000, "boot": 10000, "seed": 1, "sigma_used": 0.004743, "base_official": 1.1148854674525384, "base_tbs": 524288, "code_sha256": "418bfc38c98c96d5343d41ac081935010af2b88126cc740a3437ca2e34ab61f8"}.

## 1. Cost reconciliation

Mean per loop, by arm.

| arm | loops | GPU jobs | GPU hours | Claude USD | open USD | accepted changes | total @0.5 | total @1.0 | total @2.0 | total @4.0 |
|---|---|---|---|---|---|---|---|---|---|---|
| A | 8 | 20.0 | 2.32 | 0.779 | 0.000 | 2.12 | 1.94 | 3.10 | 5.42 | 10.05 |
| B | 3 | 20.0 | 2.29 | 1.063 | 0.000 | 2.00 | 2.21 | 3.35 | 5.64 | 10.21 |
| C | 3 | 20.0 | 2.30 | 1.442 | 0.000 | 1.67 | 2.59 | 3.75 | 6.05 | 10.66 |
| D | 8 | 18.9 | 2.12 | 0.216 | 0.027 | 2.25 | 1.30 | 2.37 | 4.49 | 8.73 |

Dollars per accepted change (arm total dollars divided by arm accepted changes).

| arm | @0.5 | @1.0 | @2.0 | @4.0 |
|---|---|---|---|---|
| A | 0.91 | 1.46 | 2.55 | 4.73 |
| B | 1.10 | 1.68 | 2.82 | 5.11 |
| C | 1.56 | 2.25 | 3.63 | 6.39 |
| D | 0.58 | 1.05 | 2.00 | 3.88 |

GPU price at which arm D's mean total per loop equals the other arm's (full 20-slot loop): no crossing at a positive price, D is cheaper than A at every price from 0 to 4 (D minus A: -0.54 USD at a GPU price of 0, -1.32 at 4); no crossing at a positive price, D is cheaper than C at every price from 0 to 4 (D minus C: -1.20 USD at a GPU price of 0, -1.92 at 4).

Per loop.

| loop | jobs | GPU h | Claude USD | open USD | accepts | total @0.5 | total @1.0 | total @2.0 | total @4.0 |
|---|---|---|---|---|---|---|---|---|---|
| A_01 | 20 | 2.31 | 0.986 | 0.000 | 3 | 2.14 | 3.29 | 5.60 | 10.22 |
| A_02 | 20 | 2.33 | 0.671 | 0.000 | 2 | 1.84 | 3.00 | 5.33 | 9.99 |
| A_03 | 20 | 2.34 | 0.700 | 0.000 | 2 | 1.87 | 3.04 | 5.37 | 10.04 |
| A_04 | 20 | 2.33 | 0.699 | 0.000 | 2 | 1.87 | 3.03 | 5.36 | 10.03 |
| A_05 | 20 | 2.31 | 0.869 | 0.000 | 2 | 2.02 | 3.18 | 5.49 | 10.10 |
| A_06 | 20 | 2.32 | 0.702 | 0.000 | 2 | 1.86 | 3.02 | 5.34 | 9.97 |
| A_07 | 20 | 2.31 | 0.801 | 0.000 | 2 | 1.96 | 3.11 | 5.42 | 10.05 |
| A_08 | 20 | 2.31 | 0.806 | 0.000 | 2 | 1.96 | 3.11 | 5.42 | 10.04 |
| B_01 | 20 | 2.29 | 1.062 | 0.000 | 2 | 2.21 | 3.35 | 5.64 | 10.21 |
| B_02 | 20 | 2.29 | 1.129 | 0.000 | 2 | 2.27 | 3.42 | 5.71 | 10.28 |
| B_03 | 20 | 2.29 | 0.998 | 0.000 | 2 | 2.14 | 3.28 | 5.57 | 10.14 |
| C_01 | 20 | 2.35 | 1.531 | 0.000 | 1 | 2.71 | 3.89 | 6.24 | 10.95 |
| C_02 | 20 | 2.29 | 1.522 | 0.000 | 2 | 2.67 | 3.82 | 6.11 | 10.70 |
| C_03 | 20 | 2.26 | 1.273 | 0.000 | 2 | 2.40 | 3.53 | 5.79 | 10.32 |
| D_01 | 18 | 2.00 | 0.219 | 0.032 | 2 | 1.25 | 2.25 | 4.26 | 8.26 |
| D_02 | 20 | 2.23 | 0.230 | 0.022 | 1 | 1.37 | 2.49 | 4.72 | 9.19 |
| D_03 | 20 | 2.31 | 0.219 | 0.019 | 3 | 1.39 | 2.55 | 4.86 | 9.48 |
| D_04 | 19 | 2.15 | 0.211 | 0.031 | 3 | 1.32 | 2.39 | 4.55 | 8.85 |
| D_05 | 18 | 1.99 | 0.201 | 0.030 | 2 | 1.23 | 2.22 | 4.21 | 8.19 |
| D_06 | 19 | 2.20 | 0.200 | 0.024 | 2 | 1.33 | 2.43 | 4.63 | 9.04 |
| D_07 | 19 | 2.18 | 0.223 | 0.023 | 2 | 1.34 | 2.43 | 4.62 | 8.99 |
| D_08 | 18 | 1.90 | 0.227 | 0.031 | 3 | 1.21 | 2.16 | 4.07 | 7.88 |

## 2. Plateau (slot of the last incumbent change)

Observed if at least 5 charged slots follow it with no change, otherwise censored at 20. Costs run up to and including the plateau slot (a censored loop is charged its whole run), GPU at 2 USD per A100-hour.

| loop | last change | plateau slot | censored | jobs | GPU h | model USD | total USD |
|---|---|---|---|---|---|---|---|
| A_01 | 5 | 5 | no | 6 | 0.70 | 0.221 | 1.63 |
| A_02 | 2 | 2 | no | 3 | 0.35 | 0.107 | 0.81 |
| A_03 | 2 | 2 | no | 3 | 0.35 | 0.108 | 0.81 |
| A_04 | 2 | 2 | no | 3 | 0.35 | 0.060 | 0.76 |
| A_05 | 2 | 2 | no | 3 | 0.35 | 0.108 | 0.80 |
| A_06 | 2 | 2 | no | 3 | 0.35 | 0.059 | 0.75 |
| A_07 | 2 | 2 | no | 3 | 0.35 | 0.108 | 0.81 |
| A_08 | 2 | 2 | no | 3 | 0.35 | 0.059 | 0.76 |
| B_01 | 2 | 2 | no | 3 | 0.35 | 0.110 | 0.81 |
| B_02 | 2 | 2 | no | 3 | 0.35 | 0.110 | 0.81 |
| B_03 | 2 | 2 | no | 3 | 0.35 | 0.110 | 0.81 |
| C_01 | 1 | 1 | no | 2 | 0.24 | 0.138 | 0.61 |
| C_02 | 3 | 3 | no | 4 | 0.49 | 0.261 | 1.24 |
| C_03 | 9 | 9 | no | 10 | 1.15 | 0.578 | 2.88 |
| D_01 | 10 | 10 | no | 9 | 1.02 | 0.177 | 2.22 |
| D_02 | 9 | 9 | no | 10 | 1.18 | 0.141 | 2.51 |
| D_03 | 19 | 20 | yes | 20 | 2.31 | 0.238 | 4.86 |
| D_04 | 16 | 20 | yes | 19 | 2.15 | 0.241 | 4.55 |
| D_05 | 5 | 5 | no | 5 | 0.56 | 0.067 | 1.20 |
| D_06 | 19 | 20 | yes | 19 | 2.20 | 0.224 | 4.63 |
| D_07 | 8 | 8 | no | 9 | 1.05 | 0.138 | 2.23 |
| D_08 | 9 | 9 | no | 9 | 1.02 | 0.158 | 2.19 |

| arm | loops | censored | mean slot | Kaplan-Meier median slot | jobs | GPU h | model USD | total USD |
|---|---|---|---|---|---|---|---|---|
| A | 8 | 0 | 2.38 | 2 | 3.4 | 0.39 | 0.104 | 0.89 |
| B | 3 | 0 | 2.00 | 2 | 3.0 | 0.35 | 0.110 | 0.81 |
| C | 3 | 0 | 4.33 | 3 | 5.3 | 0.63 | 0.326 | 1.58 |
| D | 8 | 3 | 12.62 | 9 | 12.5 | 1.44 | 0.173 | 3.05 |

GPU price at which arm D's mean cost to plateau equals the other arm's: no crossing at a positive price, D is dearer than A at every price from 0 to 4 (D minus A: +0.07 USD at a GPU price of 0, +4.25 at 4); D and C cost the same at 0.188 USD per A100-hour (D minus C: -0.15 USD at a GPU price of 0, +3.09 at 4).

## 3. Reach 2^17 (first slot whose incumbent has TOTAL_BATCH_SIZE <= 131072)

| loop | slot | censored | jobs | GPU h | model USD | total USD |
|---|---|---|---|---|---|---|
| A_01 | 2 | no | 3 | 0.35 | 0.106 | 0.80 |
| A_02 | 2 | no | 3 | 0.35 | 0.107 | 0.81 |
| A_03 | 2 | no | 3 | 0.35 | 0.108 | 0.81 |
| A_04 | 2 | no | 3 | 0.35 | 0.060 | 0.76 |
| A_05 | 2 | no | 3 | 0.35 | 0.108 | 0.80 |
| A_06 | 2 | no | 3 | 0.35 | 0.059 | 0.75 |
| A_07 | 2 | no | 3 | 0.35 | 0.108 | 0.81 |
| A_08 | 2 | no | 3 | 0.35 | 0.059 | 0.76 |
| B_01 | 2 | no | 3 | 0.35 | 0.110 | 0.81 |
| B_02 | 2 | no | 3 | 0.35 | 0.110 | 0.81 |
| B_03 | 2 | no | 3 | 0.35 | 0.110 | 0.81 |
| C_01 | 1 | no | 2 | 0.24 | 0.138 | 0.61 |
| C_02 | 3 | no | 4 | 0.49 | 0.261 | 1.24 |
| C_03 | 20 | yes | 20 | 2.26 | 1.273 | 5.79 |
| D_01 | 20 | yes | 18 | 2.00 | 0.251 | 4.26 |
| D_02 | 9 | no | 10 | 1.18 | 0.141 | 2.51 |
| D_03 | 20 | yes | 20 | 2.31 | 0.238 | 4.86 |
| D_04 | 16 | no | 16 | 1.81 | 0.218 | 3.84 |
| D_05 | 20 | yes | 18 | 1.99 | 0.231 | 4.21 |
| D_06 | 19 | no | 19 | 2.20 | 0.224 | 4.63 |
| D_07 | 8 | no | 9 | 1.05 | 0.138 | 2.23 |
| D_08 | 6 | no | 6 | 0.68 | 0.096 | 1.46 |

| arm | loops | censored | mean slot | Kaplan-Meier median slot | jobs | total USD |
|---|---|---|---|---|---|---|
| A | 8 | 0 | 2.00 | 2 | 3.0 | 0.79 |
| B | 3 | 0 | 2.00 | 2 | 3.0 | 0.81 |
| C | 3 | 1 | 8.00 | 3 | 8.7 | 2.55 |
| D | 8 | 3 | 14.75 | 16 | 14.5 | 3.50 |

## 4. Stall (final incumbent TOTAL_BATCH_SIZE > 131072)

Improvement = sealed val_bpb gain over the baseline: paired uses the same-wave anchor, global uses the mean of all 30 anchors. Higher is better.

| loop | final batch size | final depth | stalled | draft P2 flag | paired improvement | global-anchor improvement | sealed mean |
|---|---|---|---|---|---|---|---|
| A_01 | 131072 | 8 | no | yes | 0.0597 | 0.0639 | 1.0519 |
| A_02 | 131072 | 8 | no | yes | 0.0623 | 0.0636 | 1.0523 |
| A_03 | 131072 | 8 | no | yes | 0.0630 | 0.0638 | 1.0521 |
| A_04 | 131072 | 8 | no | yes | 0.0661 | 0.0639 | 1.0520 |
| A_05 | 131072 | 8 | no | yes | 0.0634 | 0.0636 | 1.0522 |
| A_06 | 131072 | 8 | no | yes | 0.0640 | 0.0636 | 1.0522 |
| A_07 | 131072 | 8 | no | yes | 0.0637 | 0.0635 | 1.0523 |
| A_08 | 131072 | 8 | no | yes | 0.0642 | 0.0637 | 1.0522 |
| B_01 | 131072 | 8 | no | yes | 0.0607 | 0.0640 | 1.0519 |
| B_02 | 131072 | 8 | no | yes | 0.0651 | 0.0635 | 1.0523 |
| B_03 | 131072 | 8 | no | yes | 0.0629 | 0.0637 | 1.0521 |
| C_01 | 131072 | 8 | no | yes | 0.0657 | 0.0633 | 1.0526 |
| C_02 | 131072 | 6 | no | no | 0.0633 | 0.0636 | 1.0522 |
| C_03 | 262144 | 6 | yes | no | 0.0532 | 0.0500 | 1.0658 |
| D_01 | 262144 | 6 | yes | no | 0.0519 | 0.0501 | 1.0657 |
| D_02 | 131072 | 8 | no | yes | 0.0651 | 0.0635 | 1.0523 |
| D_03 | 262144 | 6 | yes | no | 0.0563 | 0.0585 | 1.0574 |
| D_04 | 131072 | 6 | no | no | 0.0641 | 0.0632 | 1.0527 |
| D_05 | 262144 | 6 | yes | no | 0.0466 | 0.0502 | 1.0657 |
| D_06 | 131072 | 8 | no | yes | 0.0635 | 0.0638 | 1.0521 |
| D_07 | 131072 | 8 | no | yes | 0.0651 | 0.0638 | 1.0520 |
| D_08 | 131072 | 6 | no | no | 0.0632 | 0.0630 | 1.0528 |

| version | stalled mean | not stalled mean | stalled max | not stalled min | groups separated |
|---|---|---|---|---|---|
| paired | 0.0520 | 0.0636 | 0.0563 | 0.0597 | yes |
| global | 0.0522 | 0.0636 | 0.0585 | 0.0630 | yes |

The draft P2 stall rule (final incumbent has batch size above 2^18 or depth above 6) flags 15 loops; their mean paired improvement is 0.0636 against 0.0569 for the unflagged loops; it misses 4 of the 4 stalled loops.

## 5. Halting rules

quiet k: stop after k consecutive slots with no incumbent change. after_first k: the same, counting starts after the first change. cap N: stop after N slots. Loss = raw official mean of the incumbent at the halt minus the final incumbent's (positive is worse). Within = loss <= 0.004. Dollar fraction at 2 USD per A100-hour.

| rule | share within | mean dollar fraction | mean loss |
|---|---|---|---|
| quiet_k3 | 0.59 | 0.249 | 0.0200 |
| quiet_k5 | 0.73 | 0.405 | 0.0148 |
| quiet_k8 | 0.82 | 0.638 | 0.0063 |
| quiet_k10 | 0.95 | 0.779 | 0.0021 |
| after_first_k3 | 0.68 | 0.352 | 0.0091 |
| after_first_k5 | 0.82 | 0.496 | 0.0054 |
| after_first_k8 | 0.86 | 0.659 | 0.0035 |
| after_first_k10 | 0.95 | 0.779 | 0.0021 |
| cap_n5 | 0.55 | 0.242 | 0.0187 |
| cap_n8 | 0.64 | 0.397 | 0.0122 |
| cap_n10 | 0.82 | 0.498 | 0.0054 |
| cap_n12 | 0.86 | 0.599 | 0.0046 |

Selected on this half (lowest mean dollar fraction among rules with share within >= 80%): **after_first_k5**.

Halting slot and loss, per loop.

| loop | quiet_k3 | quiet_k5 | quiet_k8 | quiet_k10 | after_first_k3 | after_first_k5 | after_first_k8 | after_first_k10 | cap_n5 | cap_n8 | cap_n10 | cap_n12 |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| A_01 | 8 / 0.0000 | 10 / 0.0000 | 13 / 0.0000 | 15 / 0.0000 | 8 / 0.0000 | 10 / 0.0000 | 13 / 0.0000 | 15 / 0.0000 | 4 / 0.0044 | 7 / 0.0000 | 9 / 0.0000 | 11 / 0.0000 |
| A_02 | 5 / 0.0000 | 7 / 0.0000 | 10 / 0.0000 | 12 / 0.0000 | 5 / 0.0000 | 7 / 0.0000 | 10 / 0.0000 | 12 / 0.0000 | 4 / 0.0000 | 7 / 0.0000 | 9 / 0.0000 | 11 / 0.0000 |
| A_03 | 5 / 0.0000 | 7 / 0.0000 | 10 / 0.0000 | 12 / 0.0000 | 5 / 0.0000 | 7 / 0.0000 | 10 / 0.0000 | 12 / 0.0000 | 4 / 0.0000 | 7 / 0.0000 | 9 / 0.0000 | 11 / 0.0000 |
| A_04 | 5 / 0.0000 | 7 / 0.0000 | 10 / 0.0000 | 12 / 0.0000 | 5 / 0.0000 | 7 / 0.0000 | 10 / 0.0000 | 12 / 0.0000 | 4 / 0.0000 | 7 / 0.0000 | 9 / 0.0000 | 11 / 0.0000 |
| A_05 | 5 / 0.0000 | 7 / 0.0000 | 10 / 0.0000 | 12 / 0.0000 | 5 / 0.0000 | 7 / 0.0000 | 10 / 0.0000 | 12 / 0.0000 | 4 / 0.0000 | 7 / 0.0000 | 9 / 0.0000 | 11 / 0.0000 |
| A_06 | 5 / 0.0000 | 7 / 0.0000 | 10 / 0.0000 | 12 / 0.0000 | 5 / 0.0000 | 7 / 0.0000 | 10 / 0.0000 | 12 / 0.0000 | 4 / 0.0000 | 7 / 0.0000 | 9 / 0.0000 | 11 / 0.0000 |
| A_07 | 5 / 0.0000 | 7 / 0.0000 | 10 / 0.0000 | 12 / 0.0000 | 5 / 0.0000 | 7 / 0.0000 | 10 / 0.0000 | 12 / 0.0000 | 4 / 0.0000 | 7 / 0.0000 | 9 / 0.0000 | 11 / 0.0000 |
| A_08 | 5 / 0.0000 | 7 / 0.0000 | 10 / 0.0000 | 12 / 0.0000 | 5 / 0.0000 | 7 / 0.0000 | 10 / 0.0000 | 12 / 0.0000 | 4 / 0.0000 | 7 / 0.0000 | 9 / 0.0000 | 11 / 0.0000 |
| B_01 | 5 / 0.0000 | 7 / 0.0000 | 10 / 0.0000 | 12 / 0.0000 | 5 / 0.0000 | 7 / 0.0000 | 10 / 0.0000 | 12 / 0.0000 | 4 / 0.0000 | 7 / 0.0000 | 9 / 0.0000 | 11 / 0.0000 |
| B_02 | 5 / 0.0000 | 7 / 0.0000 | 10 / 0.0000 | 12 / 0.0000 | 5 / 0.0000 | 7 / 0.0000 | 10 / 0.0000 | 12 / 0.0000 | 4 / 0.0000 | 7 / 0.0000 | 9 / 0.0000 | 11 / 0.0000 |
| B_03 | 5 / 0.0000 | 7 / 0.0000 | 10 / 0.0000 | 12 / 0.0000 | 5 / 0.0000 | 7 / 0.0000 | 10 / 0.0000 | 12 / 0.0000 | 4 / 0.0000 | 7 / 0.0000 | 9 / 0.0000 | 11 / 0.0000 |
| C_01 | 4 / 0.0000 | 6 / 0.0000 | 9 / 0.0000 | 11 / 0.0000 | 4 / 0.0000 | 6 / 0.0000 | 9 / 0.0000 | 11 / 0.0000 | 4 / 0.0000 | 7 / 0.0000 | 9 / 0.0000 | 11 / 0.0000 |
| C_02 | 6 / 0.0000 | 8 / 0.0000 | 11 / 0.0000 | 13 / 0.0000 | 6 / 0.0000 | 8 / 0.0000 | 11 / 0.0000 | 13 / 0.0000 | 4 / 0.0000 | 7 / 0.0000 | 9 / 0.0000 | 11 / 0.0000 |
| C_03 | 2 / 0.0498 | 14 / 0.0000 | 17 / 0.0000 | 19 / 0.0000 | 7 / 0.0181 | 14 / 0.0000 | 17 / 0.0000 | 19 / 0.0000 | 4 / 0.0181 | 7 / 0.0181 | 9 / 0.0000 | 11 / 0.0000 |
| D_01 | 4 / 0.0178 | 6 / 0.0178 | 9 / 0.0178 | 19 / 0.0000 | 4 / 0.0178 | 6 / 0.0178 | 9 / 0.0178 | 19 / 0.0000 | 4 / 0.0178 | 7 / 0.0178 | 9 / 0.0178 | 11 / 0.0000 |
| D_02 | 2 / 0.0623 | 4 / 0.0623 | 7 / 0.0623 | 19 / 0.0000 | 12 / 0.0000 | 14 / 0.0000 | 17 / 0.0000 | 19 / 0.0000 | 4 / 0.0623 | 7 / 0.0623 | 9 / 0.0000 | 11 / 0.0000 |
| D_03 | 2 / 0.0555 | 4 / 0.0555 | 19 / 0.0000 | 19 / 0.0000 | 9 / 0.0426 | 11 / 0.0426 | 19 / 0.0000 | 19 / 0.0000 | 4 / 0.0555 | 7 / 0.0426 | 9 / 0.0426 | 11 / 0.0426 |
| D_04 | 2 / 0.0634 | 4 / 0.0634 | 15 / 0.0127 | 19 / 0.0000 | 10 / 0.0127 | 12 / 0.0127 | 15 / 0.0127 | 19 / 0.0000 | 4 / 0.0634 | 7 / 0.0127 | 9 / 0.0127 | 11 / 0.0127 |
| D_05 | 3 / 0.0305 | 10 / 0.0000 | 13 / 0.0000 | 15 / 0.0000 | 3 / 0.0305 | 10 / 0.0000 | 13 / 0.0000 | 15 / 0.0000 | 4 / 0.0305 | 7 / 0.0000 | 9 / 0.0000 | 11 / 0.0000 |
| D_06 | 2 / 0.0630 | 4 / 0.0630 | 13 / 0.0456 | 15 / 0.0456 | 8 / 0.0456 | 10 / 0.0456 | 13 / 0.0456 | 15 / 0.0456 | 4 / 0.0630 | 7 / 0.0456 | 9 / 0.0456 | 11 / 0.0456 |
| D_07 | 2 / 0.0631 | 4 / 0.0631 | 16 / 0.0000 | 18 / 0.0000 | 11 / 0.0000 | 13 / 0.0000 | 16 / 0.0000 | 18 / 0.0000 | 4 / 0.0631 | 7 / 0.0598 | 9 / 0.0000 | 11 / 0.0000 |
| D_08 | 4 / 0.0336 | 14 / 0.0000 | 17 / 0.0000 | 19 / 0.0000 | 4 / 0.0336 | 14 / 0.0000 | 17 / 0.0000 | 19 / 0.0000 | 4 / 0.0336 | 7 / 0.0096 | 9 / 0.0000 | 11 / 0.0000 |

Dollar fraction spent by the halt, per loop.

| loop | quiet_k3 | quiet_k5 | quiet_k8 | quiet_k10 | after_first_k3 | after_first_k5 | after_first_k8 | after_first_k10 | cap_n5 | cap_n8 | cap_n10 | cap_n12 |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| A_01 | 0.44 | 0.54 | 0.70 | 0.80 | 0.44 | 0.54 | 0.70 | 0.80 | 0.25 | 0.39 | 0.49 | 0.59 |
| A_02 | 0.30 | 0.40 | 0.55 | 0.65 | 0.30 | 0.40 | 0.55 | 0.65 | 0.26 | 0.40 | 0.50 | 0.60 |
| A_03 | 0.31 | 0.41 | 0.56 | 0.66 | 0.31 | 0.41 | 0.56 | 0.66 | 0.26 | 0.41 | 0.51 | 0.61 |
| A_04 | 0.30 | 0.40 | 0.54 | 0.65 | 0.30 | 0.40 | 0.54 | 0.65 | 0.25 | 0.40 | 0.50 | 0.59 |
| A_05 | 0.30 | 0.40 | 0.54 | 0.64 | 0.30 | 0.40 | 0.54 | 0.64 | 0.25 | 0.40 | 0.49 | 0.59 |
| A_06 | 0.30 | 0.40 | 0.55 | 0.65 | 0.30 | 0.40 | 0.55 | 0.65 | 0.25 | 0.40 | 0.51 | 0.60 |
| A_07 | 0.31 | 0.40 | 0.55 | 0.65 | 0.31 | 0.40 | 0.55 | 0.65 | 0.25 | 0.40 | 0.50 | 0.60 |
| A_08 | 0.29 | 0.39 | 0.54 | 0.65 | 0.29 | 0.39 | 0.54 | 0.65 | 0.24 | 0.39 | 0.49 | 0.59 |
| B_01 | 0.30 | 0.40 | 0.55 | 0.65 | 0.30 | 0.40 | 0.55 | 0.65 | 0.25 | 0.40 | 0.50 | 0.60 |
| B_02 | 0.29 | 0.39 | 0.54 | 0.64 | 0.29 | 0.39 | 0.54 | 0.64 | 0.24 | 0.39 | 0.49 | 0.59 |
| B_03 | 0.30 | 0.40 | 0.55 | 0.65 | 0.30 | 0.40 | 0.55 | 0.65 | 0.25 | 0.40 | 0.50 | 0.60 |
| C_01 | 0.24 | 0.33 | 0.50 | 0.58 | 0.24 | 0.33 | 0.50 | 0.58 | 0.24 | 0.39 | 0.50 | 0.58 |
| C_02 | 0.34 | 0.44 | 0.59 | 0.70 | 0.34 | 0.44 | 0.59 | 0.70 | 0.25 | 0.40 | 0.48 | 0.59 |
| C_03 | 0.16 | 0.73 | 0.90 | 1.00 | 0.40 | 0.73 | 0.90 | 1.00 | 0.26 | 0.40 | 0.50 | 0.59 |
| D_01 | 0.18 | 0.29 | 0.46 | 1.00 | 0.18 | 0.29 | 0.46 | 1.00 | 0.18 | 0.35 | 0.46 | 0.57 |
| D_02 | 0.16 | 0.26 | 0.42 | 1.00 | 0.64 | 0.73 | 0.89 | 1.00 | 0.26 | 0.42 | 0.53 | 0.64 |
| D_03 | 0.15 | 0.25 | 1.00 | 1.00 | 0.50 | 0.59 | 1.00 | 1.00 | 0.25 | 0.40 | 0.50 | 0.59 |
| D_04 | 0.11 | 0.21 | 0.79 | 1.00 | 0.52 | 0.63 | 0.79 | 1.00 | 0.21 | 0.37 | 0.48 | 0.57 |
| D_05 | 0.17 | 0.55 | 0.68 | 0.78 | 0.17 | 0.55 | 0.68 | 0.78 | 0.22 | 0.40 | 0.50 | 0.62 |
| D_06 | 0.11 | 0.21 | 0.68 | 0.78 | 0.42 | 0.52 | 0.68 | 0.78 | 0.21 | 0.37 | 0.47 | 0.58 |
| D_07 | 0.16 | 0.26 | 0.89 | 1.00 | 0.63 | 0.73 | 0.89 | 1.00 | 0.26 | 0.43 | 0.53 | 0.63 |
| D_08 | 0.24 | 0.82 | 0.94 | 1.00 | 0.24 | 0.82 | 0.94 | 1.00 | 0.24 | 0.42 | 0.54 | 0.65 |

## 6. Noise replay and acceptance-rule frontier (sentinels and arm A loops only)

Noise sd from 52 sentinel runs (raw official): 0.00378. Converged regime = candidate first runs after the incumbent reached 2^17, in arm A loops ['A_01', 'A_02', 'A_03', 'A_04', 'A_05', 'A_06', 'A_07', 'A_08']. Observed: 95 candidates, 43 would-be keeps, 41 audits, 40 audit rejects, 1 audit accepts. The logged keep decision matches 'first run strictly below the incumbent raw mean' for 95 of 95 candidates. Sd of the converged first runs is 0.67 times the sentinel sd.

Pure-noise prediction (true effect zero for every candidate, level = mean converged first run 1.0519, R1's rule, sigma 0.004743 in the audit z). Intervals are 90 percent predictive (include uncertainty in sd and level).

| noise | count | observed | predicted mean | 90% interval | inside | P(sim >= observed) |
|---|---|---|---|---|---|---|
| gaussian | candidates | 95 | 94.25 | 88 to 101 | yes | 0.466 |
| gaussian | keeps | 43 | 44.19 | 37 to 51 | yes | 0.657 |
| gaussian | audits | 41 | 41.75 | 35 to 48 | yes | 0.625 |
| gaussian | audit_rejects | 40 | 41.16 | 33 to 48 | yes | 0.657 |
| gaussian | audit_accepts | 1 | 0.58 | 0 to 2 | yes | 0.419 |
| bootstrap | candidates | 95 | 95.28 | 89 to 101 | yes | 0.582 |
| bootstrap | keeps | 43 | 43.21 | 37 to 49 | yes | 0.583 |
| bootstrap | audits | 41 | 40.72 | 35 to 47 | yes | 0.531 |
| bootstrap | audit_rejects | 40 | 40.62 | 35 to 46 | yes | 0.631 |
| bootstrap | audit_accepts | 1 | 0.10 | 0 to 1 | yes | 0.092 |
| gaussian_conv | candidates | 95 | 93.28 | 87 to 100 | yes | 0.369 |
| gaussian_conv | keeps | 43 | 45.33 | 38 to 52 | yes | 0.760 |
| gaussian_conv | audits | 41 | 42.72 | 36 to 49 | yes | 0.720 |
| gaussian_conv | audit_rejects | 40 | 42.70 | 36 to 49 | yes | 0.795 |
| gaussian_conv | audit_accepts | 1 | 0.02 | 0 to 0 | NO | 0.021 |

Frontier by simulation. One slot is one candidate decision; each loop has 20 of them against an incumbent measured by 2 runs. Rules: keep_if_lower (first run below the incumbent mean), zband (R1's rule: reject z below 0.75, accept at 2.0 or above, replicate in between and accept at 1.5), mean2 (mean of 2 runs below the incumbent), median3 (median of 3 runs below the incumbent). A keep moves the incumbent to the candidate's own measurement. Noise: gaussian and bootstrap use the sentinel sd and residuals; gaussian_conv uses the sd of the converged first runs (sensitivity).

Noise: gaussian. False keeps are per 20 slots with no true effect; the other columns are the chance one decision keeps a candidate that is truly better by that many bpb, with runs per decision in brackets.

| rule | runs/slot (no effect) | false keeps / 20 | true-keep @0.001 (runs) | true-keep @0.002 (runs) | true-keep @0.004 (runs) | true-keep @0.008 (runs) |
|---|---|---|---|---|---|---|
| keep_if_lower | 1.00 | 2.75 | 0.587 (1.00) | 0.666 (1.00) | 0.807 (1.00) | 0.958 (1.00) |
| zband | 1.11 | 0.24 | 0.053 (1.23) | 0.086 (1.29) | 0.195 (1.42) | 0.570 (1.57) |
| mean2 | 2.00 | 2.62 | 0.602 (2.00) | 0.700 (2.00) | 0.857 (2.00) | 0.983 (2.00) |
| median3 | 3.00 | 2.62 | 0.601 (3.00) | 0.710 (3.00) | 0.862 (3.00) | 0.985 (3.00) |

Same, as the mean keep rate over a stream of 20 consecutive truly better candidates (later decisions face a bar already pulled down by earlier keeps).

| rule | stream keep rate @0.001 (runs/slot) | stream keep rate @0.002 (runs/slot) | stream keep rate @0.004 (runs/slot) | stream keep rate @0.008 (runs/slot) |
|---|---|---|---|---|
| keep_if_lower | 0.196 (1.00) | 0.287 (1.00) | 0.550 (1.00) | 0.903 (1.00) |
| zband | 0.018 (1.13) | 0.024 (1.16) | 0.043 (1.20) | 0.141 (1.36) |
| mean2 | 0.221 (2.00) | 0.378 (2.00) | 0.736 (2.00) | 0.978 (2.00) |
| median3 | 0.225 (3.00) | 0.397 (3.00) | 0.762 (3.00) | 0.984 (3.00) |

Noise: bootstrap. False keeps are per 20 slots with no true effect; the other columns are the chance one decision keeps a candidate that is truly better by that many bpb, with runs per decision in brackets.

| rule | runs/slot (no effect) | false keeps / 20 | true-keep @0.001 (runs) | true-keep @0.002 (runs) | true-keep @0.004 (runs) | true-keep @0.008 (runs) |
|---|---|---|---|---|---|---|
| keep_if_lower | 1.00 | 2.61 | 0.566 (1.00) | 0.646 (1.00) | 0.790 (1.00) | 0.967 (1.00) |
| zband | 1.13 | 0.23 | 0.051 (1.26) | 0.085 (1.32) | 0.198 (1.41) | 0.570 (1.51) |
| mean2 | 2.00 | 2.65 | 0.614 (2.00) | 0.700 (2.00) | 0.846 (2.00) | 0.986 (2.00) |
| median3 | 3.00 | 2.53 | 0.573 (3.00) | 0.662 (3.00) | 0.809 (3.00) | 0.982 (3.00) |

Same, as the mean keep rate over a stream of 20 consecutive truly better candidates (later decisions face a bar already pulled down by earlier keeps).

| rule | stream keep rate @0.001 (runs/slot) | stream keep rate @0.002 (runs/slot) | stream keep rate @0.004 (runs/slot) | stream keep rate @0.008 (runs/slot) |
|---|---|---|---|---|
| keep_if_lower | 0.247 (1.00) | 0.367 (1.00) | 0.537 (1.00) | 0.893 (1.00) |
| zband | 0.016 (1.16) | 0.023 (1.18) | 0.041 (1.23) | 0.133 (1.41) |
| mean2 | 0.224 (2.00) | 0.365 (2.00) | 0.734 (2.00) | 0.983 (2.00) |
| median3 | 0.246 (3.00) | 0.358 (3.00) | 0.614 (3.00) | 0.970 (3.00) |

Noise: gaussian_conv. False keeps are per 20 slots with no true effect; the other columns are the chance one decision keeps a candidate that is truly better by that many bpb, with runs per decision in brackets.

| rule | runs/slot (no effect) | false keeps / 20 | true-keep @0.001 (runs) | true-keep @0.002 (runs) | true-keep @0.004 (runs) | true-keep @0.008 (runs) |
|---|---|---|---|---|---|---|
| keep_if_lower | 1.00 | 2.76 | 0.628 (1.00) | 0.741 (1.00) | 0.901 (1.00) | 0.995 (1.00) |
| zband | 1.07 | 0.03 | 0.008 (1.14) | 0.023 (1.22) | 0.106 (1.45) | 0.630 (1.76) |
| mean2 | 2.00 | 2.67 | 0.650 (2.00) | 0.787 (2.00) | 0.945 (2.00) | 0.999 (2.00) |
| median3 | 3.00 | 2.64 | 0.663 (3.00) | 0.792 (3.00) | 0.951 (3.00) | 1.000 (3.00) |

Same, as the mean keep rate over a stream of 20 consecutive truly better candidates (later decisions face a bar already pulled down by earlier keeps).

| rule | stream keep rate @0.001 (runs/slot) | stream keep rate @0.002 (runs/slot) | stream keep rate @0.004 (runs/slot) | stream keep rate @0.008 (runs/slot) |
|---|---|---|---|---|
| keep_if_lower | 0.235 (1.00) | 0.409 (1.00) | 0.769 (1.00) | 0.985 (1.00) |
| zband | 0.004 (1.12) | 0.009 (1.17) | 0.028 (1.26) | 0.211 (1.57) |
| mean2 | 0.287 (2.00) | 0.565 (2.00) | 0.915 (2.00) | 0.999 (2.00) |
| median3 | 0.297 (3.00) | 0.592 (3.00) | 0.930 (3.00) | 1.000 (3.00) |

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
| plateau_slot_D | 8 | 6.28 | 1.81 | 2.98 |
| reach_slot_D | 8 | 6.07 | 1.75 | 2.88 |

## Decision lines (verdict)

Claim 1: A minus D mean slots to plateau (censored at 20) = -10.25, 90% bootstrap interval -13.75 to -6.88, excludes zero: True. Censored loops: A 0, D 3.
Sensitivity excluding censored loops: diff -5.82, interval -7.03 to -4.40, excludes zero: True.

Halting: rule after_first_k5, share within 0.82, mean dollar fraction 0.496, pass (>= 80% within at <= 60% of dollars): True.

N2 Test 1: observed converged-regime audit rejects 40, 90% predictive interval 33 to 48, inside: True.
