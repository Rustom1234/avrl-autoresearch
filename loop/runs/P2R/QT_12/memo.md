Incumbent mean_adjusted is 1.1146 over 16 runs. Nothing has clearly beaten it yet.

Helped most: higher embedding learning rates gave 1.1113 (-0.0033). Raising SCALAR_LR from 0.5 to 0.7 gave 1.1129 (-0.0016). Lowering initial resid_lambdas to 0.85 gave -0.0010, but its two runs read 1.1102 and 1.1171, so it is noisy. Lowering weight decay gave -0.0001 and -0.0007, probably noise.

Did not help: WARMDOWN_RATIO 0.7 (+0.0022), scalar LR warmup (+0.0106 and +0.0116), increasing weight decay over time (+0.0001), and the per-step LR scaling change (+0.0000).

Failures: three patch failures and three aborts. Copy SEARCH text exactly from the current file. Check shape constraints before proposing a change.

Next: combine the higher embedding LR with SCALAR_LR 0.7 and weight decay 0.1. Replicate the combination, since single runs look noisy. Push the embedding LR further. Avoid more warmdown and scalar-warmup changes.
