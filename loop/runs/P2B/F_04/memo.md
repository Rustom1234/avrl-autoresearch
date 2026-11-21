Helped: changes that fit more optimizer steps into the fixed 300 s. DEPTH 8->6 at ASPECT_RATIO 64 gave 1.0828/1.0835 (-0.0314), and TOTAL_BATCH_SIZE 2^17 gave 1.0618/1.0619 (-0.0213, incumbent). Both accepted runs carry a tripwire flag, so check it before building on them.

Did not help: HEAD_DIM 64 (+0.0177), DEPTH 6 with ASPECT_RATIO 96 (+0.0118), DEPTH 10 (+0.0049), WARMDOWN_RATIO 0.3 (+0.0029), matrix LR 0.05 (+0.0020), WINDOW_PATTERN SSSS (+0.0012). Single-run gains are small and likely noise: WARMDOWN_RATIO 0.7 (-0.0020), value-embedding gate 1 (-0.0022), embedding LR 0.4 (-0.0014), ADAM_BETAS (0.9, 0.95) (-0.0011), gradient clipping (-0.0007), FINAL_LR_FRAC 0.05 (-0.0006). Slot 11 was over budget. Slots 6, 15 and 17 hit patch_fail, so their ideas are untested.

Next: try batch 2^16. Combine the small gains (WARMDOWN_RATIO 0.7, gate 1, embedding LR 0.4) in one run and replicate it. Retry UNEMBEDDING_LR 0.008 with an exact SEARCH block. Avoid more architecture changes.
