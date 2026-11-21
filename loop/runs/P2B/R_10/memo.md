Helped: only batch size. Slot 3 (TOTAL_BATCH_SIZE 2**17, grad accum 1) gave 1.0515/1.0498; incumbent mean_adjusted is 1.0506 over 2 runs. It carries a tripwire flag, so check that before building on it.

Did not help: schedule tweaks (warmdown 0.7 gave 1.1175, warmdown 0.3 gave 1.0556, warmup 0.02 gave 1.0536, warmup 0.05 gave 1.0548, FINAL_LR_FRAC 0.05 gave 1.0510). Optimizer tweaks (grad clip 1.0 gave 1.0521, momentum ramp 100 gave 1.0514, SCALAR_LR 0.3 gave 1.0562) were also worse. Architecture changes failed (depth 6 gave 1.1234, SwiGLU 1.0565, HEAD_DIM 64 1.0585, window "L" 1.0538). Batch ramp gave 1.0788. Slot 2 (sync every 10 steps) was invalid: over budget, so 1.0700 is not usable evidence.

Next: try TOTAL_BATCH_SIZE 2**16, the most promising direction, then retune LR for the new step count. Avoid more schedule, architecture and window changes. Keep replicates to a minimum.
