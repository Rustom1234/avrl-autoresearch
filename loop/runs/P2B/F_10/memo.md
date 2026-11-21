Helped: structural changes that fit the fixed 300 s budget. Depth 8 to 6 gave 1.0834/1.0875 (delta -0.0291, accepted). TOTAL_BATCH_SIZE ~262K gave 1.0658/1.0664 (delta -0.0193, accepted). Both carry a tripwire flag, so verify them before trusting the gains. Incumbent is 1.0661 over 2 runs.

Promising but rejected: batch 2^17 (~131K) gave 1.0618 (-0.0044), the best single result so far. It was rejected, probably because it was a single run, so it deserves a replicate or a retry.

Did not help: DEPTH 10 (1.1652, +0.0798), HEAD_DIM 64 (+0.0040), MATRIX_LR 0.08 (+0.0006), 0.05 (+0.0000), weight decay 0.5 (+0.0022), LR spike (+0.0016). Schedule and regularization tweaks were near noise: warmdown 0.7 (-0.0001), weight decay 0.1 (-0.0003), warmup (-0.0016), clipping (-0.0005).

Next: replicate or push the batch size lower, try depth 5 and other widths at depth 6, then retune LR for the new batch. Stop tuning schedule and regularization.
