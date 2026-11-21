Status: nothing accepted; incumbent mean_adjusted 1.1145 over 16 runs.

Helped or near neutral: WEIGHT_DECAY 0.5 gave 1.1112 (-0.0033) and bfloat16 autocast gave 1.1142 (-0.0003). Both were rejected, so replicate them. The weight decay result suggests stronger regularization is the best direction.

Did not help: schedule changes (warmup 0.05: 1.1251, warmup 0.02: 1.1245, cosine: 1.1346, cosine with restarts: 1.1357 and 1.1558). Weaker weight decay (0.001: 1.1172, 0.01: 1.1188) was worse. UNEMBEDDING_LR 0.1 gave 1.1509 and 0.001 gave 1.1516, so 0.004 is near optimal. HEAD_DIM 256 gave 1.1170. DEPTH 4 gave 1.1558. Label smoothing gave 1.1185 after a crash and an invalid run.

Next: replicate weight decay 0.5, and try a larger value together with bfloat16. Then try the matrix/Muon LR. Avoid schedule changes and the optimizer swaps that aborted. Keep edits small.
