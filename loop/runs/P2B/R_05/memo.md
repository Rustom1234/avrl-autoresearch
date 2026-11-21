Helped: changes giving more optimizer steps in the fixed 300 s. Halving TOTAL_BATCH_SIZE to 2**18 gave -0.0226 (official 1.0919). DEPTH 8 to 6 gave -0.0277 (1.0650/1.0633 over 2 runs) but carries a tripwire flag, so verify it before building on it. Incumbent is mean_adjusted 1.0642.

Did not help: cosine decay (+0.0654 and +0.0537), WARMUP_RATIO 0.03 (+0.0006), grad clip (+0.0012), beta2 0.98 (+0.0009), WARMDOWN_RATIO 0.7 (+0.0017) and 0.3 (+0.0025), HEAD_DIM 64 (+0.0055), SwiGLU (+0.0040), deep supervision (+0.0067), 4 NS iterations (+0.0022), per-layer weight decay (+0.0017). Schedule, optimizer and regularization tweaks look exhausted.

Borderline: TOTAL_BATCH_SIZE 2**17 gave 1.0615 (-0.0026) and was rejected. Replicate it, since it may be real.

Next: investigate the tripwire on DEPTH 6. Try DEPTH 5 or a narrower width, then tune learning rate for the smaller model.
