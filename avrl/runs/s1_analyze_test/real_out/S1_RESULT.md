# S1 result

- DRIFT, s1_proxy_s0: VOID (only 150 steps completed (needs 600))
- OVEROPTIMIZATION, s1_proxy_s0: VOID (rm_score_mean unavailable in this log)
- DRIFT, s1_proxy_lr3_s0: arm missing
- CLAIM 1 AT SCALE: FAIL (mean gap full minus threshold 0.190 over 1 seed(s): {'s0': 0.1900000000000004})
- CLAIM 2 AT SCALE: not run (no greedy arm in S1)

| arm | status | steps | calls | corrupted / unverified | rate | rate first 200 | rate last 200 | rm first | rm last | evals (step, score) | final |
|---|---|---|---|---|---|---|---|---|---|---|---|
| s1_proxy_s0 | incomplete (150 steps) | 150 | 300 | 68 / 900 | 0.076 | 0.076 | n/a | n/a | n/a | [(50, 5.67), (100, 5.74), (150, 5.77)] | 5.770 |
| s1_thrcons_s0 | incomplete (150 steps) | 150 | 300 | 53 / 900 | 0.059 | 0.059 | n/a | n/a | n/a | [(50, 5.7), (100, 5.715), (150, 5.64)] | 5.640 |
| s1_full_s0 | incomplete (150 steps) | 150 | 1200 | 0 / 0 | n/a | n/a | n/a | n/a | n/a | [(50, 5.815), (100, 5.855), (150, 5.83)] | 5.830 |
| s1_proxy_lr3_s0 | missing | | | | | | | | | | |
| s1_thrcons_s1 | missing | | | | | | | | | | |
| s1_full_s1 | missing | | | | | | | | | | |
