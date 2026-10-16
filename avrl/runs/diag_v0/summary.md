# avrl_router diagnostic v0: validation across saved points

Budgets reported: 0.1,0.25,0.5. Target corruption for recommended budget: 0.02. n_boot=1000, seed=0.

| point | confidence_col | n | proxy_error_rate | auc | corrupted_thresh@25% | corrupted_random@25% | rel_reduction@25% | recommended_budget(target=0.02) | regime |
|---|---|---|---|---|---|---|---|---|---|
| E03 | unavailable | - | - | - | - | - | - | - | no feats.parquet/groups.parquet; only result.json (a precomputed experiment summary) exists, no row-level confidence+error table |
| E04 | unavailable | - | - | - | - | - | - | - | has judge_scores.parquet but no agreement/rm_gap12/consistency column; judge_pick_share and judge_error measure something else (judge-vs-gold, not cheap-vs-expensive disagreement) and would need to be invented into this shape, which the spec forbids |
| E05 | unavailable | - | - | - | - | - | - | - | has selfverify.parquet with sv_agreement/self_verify_error, a self-verify audit, not the agreement/proxy_error groups schema; skipped rather than reinterpreted |
| E06 | agreement | 3000 | 0.384 | 0.8741 | 0.1807 | 0.288 | 0.3727 | 0.7 | strong |
| E08 | agreement | 600 | 0.4883 | 0.8342 | 0.2633 | 0.3663 | 0.281 | 0.85 | usable |
| E09 | rm_gap12 | 1500 | 0.2113 | 0.5264 | 0.1593 | 0.1585 | -0.0053 | 0.9 | uninformative |
| E09 | consistency | 1500 | 0.2113 | 0.6739 | 0.1173 | 0.1585 | 0.2597 | 0.85 | usable |
| E09b | rm_gap12 | 1500 | 0.1807 | 0.5092 | 0.1307 | 0.1355 | 0.0357 | 0.9 | uninformative |
| E09b | consistency | 1500 | 0.1807 | 0.6931 | 0.094 | 0.1355 | 0.3063 | 0.8 | usable |
| E09c | rm_gap12 | 1500 | 0.062 | 0.5135 | 0.0447 | 0.0465 | 0.0394 | 0.65 | uninformative |
| E09c | consistency | 1500 | 0.062 | 0.6822 | 0.0307 | 0.0465 | 0.3405 | 0.4 | usable |
| E09d | rm_gap12 | 4000 | 0.0725 | 0.5692 | 0.0508 | 0.0544 | 0.0667 | 0.65 | uninformative |
| E09d | consistency | 4000 | 0.0725 | 0.6895 | 0.0357 | 0.0544 | 0.3425 | 0.5 | usable |
| p0 | agreement | 3000 | 0.1453 | 0.9143 | 0.026 | 0.109 | 0.7615 | 0.3 | strong |
