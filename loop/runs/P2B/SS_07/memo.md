Helped: throughput and batch-size changes. Removing per-step host syncs (slot 8, 1.0923, -0.0222), TOTAL_BATCH_SIZE 2**18 (slot 9, -0.0269) and 2**17 (slot 11, 1.0506/1.0536) were accepted. The incumbent is now mean_adjusted 1.0521 over 2 runs. Slots 9 and 11 both carry tripwire flags, so check what triggered them before building on them.

Did not help: head geometry (HEAD_DIM 64, +0.0111), momentum, weight decay, warmdown tweaks (0.7: +0.0023, 0.3: +0.0020, cosine: +0.0036), ADAM_BETAS (-0.0003), WINDOW_PATTERN SSL (-0.0003), and DEPTH 6 (-0.0013). Gains under about 0.003 are within single-run noise, so treat them as unproven.

Next: investigate the tripwires first. Then look for more throughput, such as compile settings, remaining syncs, and DEVICE_BATCH_SIZE between 64 and 128. Then re-tune LR for the smaller batch. Avoid more small schedule or optimizer tweaks, and check the results table before proposing a change, because several slots were aborted as repeats.
