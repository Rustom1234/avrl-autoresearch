AR3 decision: **KILL**. G0 pass, G1 pass, R1 FAIL, R2 FAIL, K1 False, K2 True. Realized e_hat 0.279.

| contrast | entries (items) | paired diff with 90% CI (5th to 95th pct) | permutation p | Holm p | q and MDE | cost ratio vs all-Sonnet | verdict |
|---|---|---|---|---|---|---|---|
| G1 headroom Q_S - Q_H3 (>= 0.06) | 48 (224) | +0.364 [+0.256, +0.471] | 0.0001 | - | q=2.78, MDE=0.182 | 0.33 | supported |
| R1 cascade - random at e_hat (> 0) | 48 (224) | -0.009 [-0.059, +0.043] | 0.5910 | - | q=0.01, MDE=0.096 | 0.61 | unresolved |
| R2 Q_S - cascade (< 0.05) | 48 (224) | +0.272 [+0.181, +0.368] | 0.9998 | - | q=2.04, MDE=0.155 | 0.61 | not supported |
| K1 gap closed at frozen policy (supported if lower end >= 0.25, killed if upper end < 0.25) | 48 (224) | -0.053 [-0.352, +0.268] | - | - | - | 0.61 | unresolved |
| S1 AUROC conf_agree - conf_verb (> 0) | 48 (224) | -0.280 [-0.440, -0.113] | - | 1.0000 | - | - | not supported |
| S2 P7 - P3 at e=0.25 (< 0.02) | 48 (224) | +0.054 [-0.006, +0.083] | - | 1.0000 | q=0.09, MDE=0.117 | 0.58 | unresolved |
| S3 cascade - all-Haiku k=5 at equal dollars (> 0) | 48 (224) | +0.052 [+0.003, +0.124] | 0.1249 | 0.1428 | q=0.27, MDE=0.101 | 0.54 | unresolved |
