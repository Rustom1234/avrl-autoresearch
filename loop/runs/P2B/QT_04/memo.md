Incumbent mean_adjusted is 1.1145 over 16 runs. Every slot so far is a reject or a crash, so nothing has been accepted.

Helped slightly, but not enough to accept (deltas may be near noise):
- ASPECT_RATIO 64 to 32: 1.1090/1.1103 over two runs, delta -0.0049.
- EMBEDDING_LR 0.6 to 0.9: delta -0.0027.
- MATRIX_LR 0.04 to 0.06: 1.1118, delta -0.0027.
- Batch-size fix: delta -0.0005.

Hurt:
- UNEMBEDDING_LR 0.004 to 0.001: +0.0406.
- Warmup: +0.0142 and +0.0114.
- Reduced regularization: +0.0051.
- Adam betas (0.9, 0.999): +0.0081.
- Halved per-device batch: +0.0939.

Crashed: depth 12 and batch size 96.

Next: combine the small wins (ASPECT_RATIO 32, EMBEDDING_LR 0.9, MATRIX_LR 0.06) in one run and replicate it. Avoid warmup and a lower UNEMBEDDING_LR. Keep batch sizes dividing 2**19 and avoid memory-heavy changes.
