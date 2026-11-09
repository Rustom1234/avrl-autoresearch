# S1 result

- Untrained policy on the same evaluation: judge score 7.455, positive rate 0.875
- DRIFT, s1_proxy_s0: PRESENT (ratio 1.739, interval [1.3686893203883495, 2.2754782608695656], first 0.110, last 0.191)
- OVEROPTIMIZATION, s1_proxy_s0: PRESENT (peak 8.400 at step 600, last two 7.600, drop 0.800, rm first 3.550 last 4.450)
- DRIFT, s1_proxy_lr3_s0: ABSENT (ratio 1.063, interval [0.8346445019241688, 1.360090909090909], first 0.139, last 0.147)
- OVEROPTIMIZATION, s1_proxy_lr3_s0: ABSENT (peak 8.200 at step 200, last two 8.200, drop 0.000, rm first 3.800 last 3.800)
- CLAIM 1 AT SCALE: PASS (mean gap full minus threshold 0.100 over 1 seed(s): {'s0': 0.09999999999999964})
- CLAIM 2 AT SCALE: not run (no greedy arm in S1)

| arm | status | steps | calls | corrupted / unverified | rate | rate first 200 | rate last 200 | rm first | rm last | evals (step, score) | final |
|---|---|---|---|---|---|---|---|---|---|---|---|
| s1_proxy_s0 | complete | 2000 | 0 | 1201 / 8000 | 0.150 | 0.110 | 0.191 | 3.550 | 4.450 | [(200, 8.0), (400, 8.3), (600, 8.4), (800, 8.2), (1000, 8.0), (1200, 7.9), (1400, 7.8), (1600, 7.7), (1800, 7.6), (2000, 7.6)] | 7.600 |
| s1_thrcons_s0 | incomplete (500 steps) | 500 | 500 | 78 / 1500 | 0.052 | 0.057 | 0.047 | 3.600 | 3.600 | [(200, 8.1), (400, 8.15)] | 8.150 |
| s1_full_s0 | complete | 2000 | 8000 | 0 / 0 | n/a | n/a | n/a | 3.700 | 3.700 | [(200, 8.25), (400, 8.25), (600, 8.25), (800, 8.25), (1000, 8.25), (1200, 8.25), (1400, 8.25), (1600, 8.25), (1800, 8.25), (2000, 8.25)] | 8.250 |
| s1_proxy_lr3_s0 | complete | 2000 | 0 | 1194 / 8000 | 0.149 | 0.139 | 0.147 | 3.800 | 3.800 | [(200, 8.2), (400, 8.2), (600, 8.2), (800, 8.2), (1000, 8.2), (1200, 8.2), (1400, 8.2), (1600, 8.2), (1800, 8.2), (2000, 8.2)] | 8.200 |
| s1_thrcons_s1 | missing | | | | | | | | | | |
| s1_full_s1 | missing | | | | | | | | | | |
