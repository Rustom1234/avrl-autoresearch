Incumbent is slot 8 (official 1.0942), reached by halving TOTAL_BATCH_SIZE (-0.0203). More optimizer steps in the fixed 300 s is the only thing that has helped clearly. Larger batch (+0.0757) and DEPTH 10 (+0.0188) hurt badly. Schedule-shape changes (cosine warmdown +0.0101, WARMDOWN_RATIO 0.7 +0.0033), LLLL windows (+0.0012), MATRIX_LR 0.06 (+0.0004), the attention output gate (+0.0052) and the EMA (+0.5696) did not help.

Small gains with 1 run each, possibly noise: WARMUP_RATIO 0.02 (-0.0011), half-length sequences (-0.0008), max-autotune (-0.0020). These were measured against different baselines, so some are not directly comparable.

Next: halve the batch again. Try a smaller model for more steps. Combine warmup 0.02, max-autotune and half-length sequences on the new batch, and replicate to check noise. Avoid schedule-shape and window changes. Check each edit applies and is not a no-op.
