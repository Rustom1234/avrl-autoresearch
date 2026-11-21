Helped: structural changes fitting the 300 s budget. DEPTH 8->6 (-0.0307), TOTAL_BATCH_SIZE 2**18 (-0.0174), and DEPTH 5 with ASPECT_RATIO 96 (-0.0112, 3 runs) were accepted; each carries a tripwire flag, so verify before building on them. Incumbent mean_adjusted is 1.0552 over 3 runs.

Did not help: optimizer tweaks (MATRIX_LR 0.05, SCALAR_LR 0.25, NS iterations 4, momentum warmup 150), all within about 0.001, so likely noise. WEIGHT_DECAY 0.1 was worse (+0.0070). Schedule changes failed: WARMUP_RATIO 0.02 (-0.0003), cosine decay (+0.0584). Also worse: WINDOW_PATTERN "SL" (+0.0040), DEPTH 10 (+0.0918), SwiGLU (+0.0018), short-sequence curriculum (+0.0112), and gradient clipping (+0.0041 after a crash fix).

Next: stop optimizer, schedule and regularization tuning. Try size and throughput changes: DEPTH 4, TOTAL_BATCH_SIZE 2**17, a different DEVICE_BATCH_SIZE, or another ASPECT_RATIO. Replicate any win near 0.001.
