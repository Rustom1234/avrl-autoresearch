Incumbent is slot 6 (1.0957): halving TOTAL_BATCH_SIZE to 2**18 gave -0.0188, the only accepted change. More optimizer steps per 300 s is the productive direction.

Rejected: warmup 0.03 (+0.0106), DEPTH=10 (+0.0177), DEPTH=6 (+0.0077), HEAD_DIM 64 (+0.0028), cosine decay (+0.0023), WARMDOWN_RATIO 0.7 (+0.0016), MATRIX_LR 0.05 (+0.0006), ADAM_BETAS 0.99 (+0.0006), momentum ramp (+0.0032). Shape and schedule look exhausted.

Near misses, all single runs and not yet accepted: SCALAR_LR 0.25 (1.0900, -0.0058), grad clipping at 1.0 (1.0915, -0.0043). Both beat the incumbent by more than the 0.002 replicate threshold. Weight decay 0.1 (-0.0006) and FINAL_LR_FRAC 0.1 with WARMDOWN_RATIO 0.6 (-0.0003) are within noise.

Slot 8 (1.0577) was invalid with a mismatch. Don't count it or resubmit it unchanged.

Next: try TOTAL_BATCH_SIZE 2**17. Combine SCALAR_LR 0.25 with grad clipping and test them together, replicating if the gain is small. Try lowering SCALAR_LR further.
