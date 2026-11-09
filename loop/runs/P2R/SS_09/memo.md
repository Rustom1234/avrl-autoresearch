Helped: more optimizer steps and smaller models. Halving TOTAL_BATCH_SIZE to 2**18 (slot 3, -0.0159), UNEMBEDDING_LR 0.008 (slot 7, -0.0111, replicated) and DEPTH 6 (slot 11, runs 1.0610/1.0684, -0.0228, replicated) were accepted. Incumbent is now mean_adjusted 1.0647 over 2 runs.

Did not help (rejected after the DEPTH 6 change): value embeddings on the last 3 layers only (-0.0020), MATRIX_LR 0.03 (-0.0035), lm_head init halved (-0.0041), SCALAR_LR 0.25 (-0.0031). Earlier rejects: SwiGLU, weight decay 0.1, WARMDOWN_RATIO 0.7, momentum ramp, WARMUP_RATIO 0.03. HEAD_DIM 64 was neutral. Several deltas are small and may be noise. Slot 11 carried a tripwire flag, so check it.

Next: TOTAL_BATCH_SIZE 2**17, since fewer tokens per step with more steps has paid off. Retest MATRIX_LR 0.05, which was close (-0.0045, single run). Try an even higher UNEMBEDDING_LR. Look for throughput gains. Replicate any promising change.
