# P2B analysis

**Verdict: PARTIAL (C3 not held)**

12 complete blocks (0 dropped for a missing loop). Outcome: sealed held-out gain. Cost: model dollars per loop.

## Gates

- **G1, damaged loop detected: gain(H) below gain(SS):** p = 0.0347. Held.
- **G2, anchors differ: gain(SS) above gain(QT):** p = 0.0005, share of draws with no anchor gap 0.000. Held.

## Claims (in order)

- **C1, fixed table beats the line: interval of ST's advantage above 0:** advantage 0.87 (0.67 to 1.15). Held.
- **C2, planner effect: gain(ST) above gain(QT):** p = 0.0005. Held.
- **C3, cheap coder non-inferior: interval of gain(ST) - gain(SS) above -0.010:** difference -0.0019 (-0.0109 to 0.0081). Not held.
- **C4, first-list rule beats the line: interval of F's advantage above 0:** advantage 0.98 (0.69 to 1.32). Not tested.

## Endpoint by policy (point and 90 percent block-bootstrap interval)

| Policy | Gap recovered | Extra cost | Advantage | Advantage, GPU dollars added |
|---|---|---|---|---|
| ST | 0.95 (0.75 to 1.24) | 0.09 (0.07 to 0.10) | 0.87 (0.67 to 1.15) | 0.67 (0.49 to 0.91) |
| F | 1.02 (0.71 to 1.36) | 0.04 (0.01 to 0.06) | 0.98 (0.69 to 1.32) | 0.74 (0.42 to 1.10) |
| M | 0.21 (-0.11 to 0.43) | 0.02 (-0.00 to 0.05) | 0.19 (-0.12 to 0.41) | 0.11 (-0.14 to 0.32) |
| R | 0.70 (0.38 to 1.04) | 0.64 (0.45 to 0.86) | 0.06 (-0.29 to 0.41) | 0.06 (-0.20 to 0.35) |
| H | 0.47 (0.03 to 0.89) | 1.11 (1.01 to 1.22) | -0.64 (-1.15 to -0.16) | -0.58 (-1.08 to -0.11) |

Reading: 0 is the straight line between all cheap and all strong, 1 is the strong gain at the cheap price.

## Cell means

| Cell | Sealed gain | Model USD per loop | Model + GPU USD |
|---|---|---|---|
| F | 0.0532 | 0.20 | 4.33 |
| H | 0.0314 | 1.49 | 6.04 |
| M | 0.0211 | 0.18 | 3.95 |
| QT | 0.0127 | 0.15 | 3.74 |
| R | 0.0406 | 0.93 | 5.14 |
| SS | 0.0526 | 1.35 | 5.93 |
| ST | 0.0507 | 0.26 | 4.36 |

## Cell N: random search with no language model (descriptive)

Run in its own session after the blocks, scored against that session's own confirmation anchors; 12 blocks.

- Sealed gain 0.0485. Gap recovered 0.90 (0.71 to 1.14), extra cost -0.13 (-0.16 to -0.10), advantage 1.02 (0.84 to 1.27).
- Advantage with GPU dollars added: 0.55 (0.36 to 0.81).
- gain(SS) above gain(N): one-sided p = 0.2297. gain(N) above gain(QT): one-sided p = 0.0002.
- If N recovers most of the gap, the task is too simple to rank loop designs.

## Random tables: gain against each coin (descriptive)

| Key | Loops with the strong model | Mean gain, strong minus cheap | Least-squares coefficient |
|---|---|---|---|
| planner_backlog | 6 | 0.0131 | -0.0073 |
| planner_widen | 4 | 0.0005 | 0.0181 |
| planner_refresh | 4 | 0.0020 | 0.0119 |
| coder_by_class.constant | 7 | 0.0078 | -0.0028 |
| coder_by_class.schedule | 4 | -0.0027 | -0.0011 |
| coder_by_class.structural | 8 | 0.0122 | 0.0062 |
| repairer | 6 | 0.0231 | 0.0345 |

## P2 and P2B side by side (advantage)

| Policy | P2 | P2B | Pooled |
|---|---|---|---|
| ST | 0.96 (0.80 to 1.16) | 0.87 (0.67 to 1.15) | 0.91 (0.78 to 1.08) |
| H | -0.53 (-0.75 to -0.31) | -0.64 (-1.15 to -0.16) | -0.58 (-0.86 to -0.33) |

## Other measures by cell (means, descriptive)

```
     stall   t_2e17  usd_per_backed_change  served_fraction_ledger  planner_served_fraction  n_fallbacks  harm_noops   n_jobs  gpu_hours
F   0.3333  11.3333                 0.1540                  0.9748                      1.0          0.0      0.0000  18.5833     2.0688
H   0.7500  17.2500                 1.3724                  1.0000                      1.0          0.0      9.5833  19.9167     2.2759
M   1.0000  20.0000                 0.1460                  0.9894                      1.0          0.0      0.0000  17.1667     1.8845
QT  0.9167  18.6667                 0.1349                  0.9859                      1.0          0.0      0.0000  17.3333     1.7934
R   0.8182  18.6667                 0.5774                  0.9876                      1.0          0.0      0.0000  19.0000     2.1092
SS  0.5000  16.9167                 0.8962                  1.0000                      1.0          0.0      0.0000  20.0000     2.2865
ST  0.8333  17.8333                 0.1893                  0.9690                      1.0          0.0      0.0000  18.5000     2.0513
```
