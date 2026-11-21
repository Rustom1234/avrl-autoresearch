Incumbent mean_adjusted is 1.0757 over 2 runs. Accepted so far: slot 5 (DEVICE_BATCH_SIZE 32, -0.0064), slot 10 (TOTAL_BATCH_SIZE 2**18, -0.0168) and slot 13 (batch ramp from 2**17 tokens, -0.0156, flagged tripwire, so check it before building on it). Batch and step-count changes are the clear winners. They give about 2x more optimizer steps in 300 s.

Rejected: warmup tweaks (slot 4 +0.0093, slot 0 +0.0006), WINDOW_PATTERN "SL" (+0.0018), DEPTH=6 with ASPECT_RATIO=96 (+0.0068), and MATRIX_LR 0.06 (-0.0050, rejected despite the gain). Slot 12 (cosine warmdown) failed to apply, so rewrite it. Slot 2 (FINAL_LR_FRAC 0.05, -0.0038) and slot 3 (q scale 1.2, -0.0011) looked mildly positive but are single runs.

Next: try even smaller batches such as 2**17 outright, retest MATRIX_LR 0.06 on the new incumbent, and add FINAL_LR_FRAC 0.05. Then try weight decay and model size. Check what the tripwire flag on slot 13 means before building on it.
