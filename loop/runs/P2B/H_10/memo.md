Helped: two changes. Halving TOTAL_BATCH_SIZE to 2**18 (slot 1, official 1.0890, delta -0.0255) and DEPTH=6 with ASPECT_RATIO=85 (slot 6, runs 1.0663/1.0600, delta -0.0258, tripwire flag). More steps and a smaller model suit the 300 s budget. Incumbent is mean_adjusted 1.0632 over 2 runs.

Did not help: schedule tweaks (WARMDOWN_RATIO 0.7 and 0.3, FINAL_LR_FRAC 0.05), optimizer tweaks (MATRIX_LR 0.05, Adam betas, grad clipping), regularization changes, DEPTH=10, WINDOW_PATTERN SSL and HEAD_DIM 64. Near-neutral: WARMUP_RATIO 0.03 (-0.0014) and MATRIX_LR 0.03 (-0.0008). Both are within noise, since most results are single runs.

Next: replicate the incumbent once for a noise estimate. Try TOTAL_BATCH_SIZE 2**17 on the depth-6 model, then DEPTH=4 or a narrower width. Combine WARMUP_RATIO 0.03 with MATRIX_LR 0.03. Find out what the slot 6 tripwire flag means before relying on it.
