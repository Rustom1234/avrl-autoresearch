Incumbent mean_adjusted is 1.1146 over 16 runs. No change has been accepted yet, and slot 0 (DEPTH 12) crashed.

Did not help: batch changes were worst (device batch 32: +0.0931; total batch 2**20: +0.0794), so leave them alone. Schedule changes lost (cosine +0.0223, warmup +0.0088). Lower Muon weight decay (+0.0052), lower EMBEDDING_LR (+0.0068), the smoothness penalty (+0.0070) and Adam beta changes (+0.0031 to +0.0094) were all worse.

Near-neutral or slightly better: SCALAR_LR 1.0 (-0.0037), WEIGHT_DECAY 0.3 (-0.0029), SwiGLU (-0.0016), WARMDOWN_RATIO 0.3 (-0.0001). MATRIX_LR 0.08 was +0.0010. These gains are small and each is one run, so they may be noise.

Next: combine SCALAR_LR 1.0, WEIGHT_DECAY 0.3 and SwiGLU in one run. Replicate that run before accepting it. Try EMBEDDING_LR above 0.6. Retry a modest depth increase only if it fits in memory.
