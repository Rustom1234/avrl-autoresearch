Helped: more optimizer steps in the fixed 300 s. Batch 2**18 (slot 4) gave 1.0947 (-0.0198). DEPTH 6 (slot 9) is the incumbent at mean_adjusted 1.0672 over 2 runs (1.0655/1.0689). The tripwire flag on slot 9 is still unexplained, and the two runs differ by about 0.0034, so treat deltas under ~0.004 as noise.

Did not help: DEPTH 10 (+0.0474), warmup 0.02 (+0.0098), SwiGLU (+0.0101), beta2 0.99 (+0.0028), 4 Newton-Schulz iterations (+0.0030), WARMDOWN_RATIO 0.3 (+0.0009).

Recent small gains, all rejected as noise-level: batch 2**17 (-0.0050, 2 runs), WARMDOWN_RATIO 0.7 (-0.0010), SSSSL (-0.0015), train_loss.item() every 10 steps (-0.0022), Muon momentum peak 0.97 (-0.0005). Embedding init std 0.5 (-0.0051) and grad clipping (-0.0040) were measured before depth 6.

Next: check what the tripwire flag means. Then try DEPTH 4 or 5, and retune LR and final LR fraction. Consider combining the small gains (2**17 batch, init std 0.5, item() every 10 steps), since they may add up. Avoid warmup.
