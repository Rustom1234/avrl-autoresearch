Incumbent mean_adjusted 1.1145 over 16 runs; none of the 20 slots' evaluated single-run candidates was accepted. Aborted slots 3, 5, 7 and 17 were no-ops or repeats, so check the code and the table first.

Helped slightly (single runs, likely noise): grad clip 1.0 (-0.0020), FINAL_LR_FRAC 0.05 (-0.0019), QK scale 1.2 (-0.0019), WARMDOWN_RATIO 0.3 (-0.0018), ADAM_BETAS (0.9, 0.99) (-0.0018), MATRIX_LR 0.05 (-0.0018), TOTAL_BATCH_SIZE 2**17 with scaled MATRIX_LR (-0.0018).

Hurt: depth 6 width 640 (+0.0073), 10 layers (+0.0058), WARMUP 0.03 with WARMDOWN 0.4 (+0.0096), WARMDOWN 0.7 (+0.0033), WINDOW_PATTERN "SL" (+0.0016), HEAD_DIM 64 (+0.0006), weight decay 0.1 (+0.0007). EMA was badly worse (+0.5406).

Combining the small gains (slot 18: -0.0012) did not add up. Many changes sit near -0.0018, which suggests noise. Next: replicate the best single candidate (grad clip 1.0) to measure noise before accepting anything. Avoid architecture changes.
