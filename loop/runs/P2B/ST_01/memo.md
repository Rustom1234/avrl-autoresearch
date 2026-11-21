Helped: architecture and batch size. DEPTH 8 to 6 gave 1.0824/1.0834 (-0.0316); TOTAL_BATCH_SIZE 2^18 gave 1.0648/1.0671 (-0.0169). Both carried tripwire flags, so verify them. Incumbent is mean_adjusted 1.0659 over 2 runs.

Mixed: TOTAL_BATCH_SIZE 2^17 gave 1.0616 (-0.0043) but was rejected; embedding LR 0.3 gave 1.0630 (-0.0029), also rejected. Both look like real small gains, so consider re-testing them, perhaps combined, with a replicate.

Did not help: schedule tweaks (warmdown 0.7 +0.0032, 0.3 +0.0005, warmup 0.03 -0.0013, final LR frac 0.05 -0.0011), SwiGLU (+0.0002), momentum ramp (+0.0016), MATRIX_LR 0.03 (+0.0002), grad clipping (-0.0005), weight decay 0.1 (+0.0001), HEAD_DIM 64 (+0.0042).

Next: try DEPTH 5 or 4, then retune LRs for the new shape. Avoid more schedule, MLP-variant, clipping and LR-nudge tweaks. Check ideas against the table first, since aborts wasted slots.
