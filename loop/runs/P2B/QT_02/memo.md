Helped: only throughput. Batch size 32 (slot 1) scored 1.1048/1.1024 over 2 runs, delta -0.0109, giving incumbent mean_adjusted 1.1036. Batch size 16 (slot 12) did not help (+0.0036), so the batch gain looks saturated near 32.

Did not help: Adam beta1 0.95 (+0.0116) and 0.9 (+0.0071); warmup 0.025 (+0.0172); warmdown 0.8 (+0.0096); cosine schedule (+0.0323); sawtooth schedule (+0.0681); embedding LR 0.1 (+0.0132) and 0.01 (+0.0640); weight decay 0.01 (+0.0086); WINDOW_PATTERN SSLLL (+0.0028).

Neutral, within the ~0.002 noise: SCALAR_LR 0.1 (+0.0012), MATRIX_LR 0.08 (+0.0002), and slot 8 (+0.0021).

Slot 7 crashed on gradient checkpointing, and slot 15 was aborted.

Next: try a smaller or shallower model for more steps, beta1 below 0.8, warmdown below 0.5, and higher weight decay. Replicate any gain under 0.002. Stop tuning the schedule and embedding LR, which have only lost.
