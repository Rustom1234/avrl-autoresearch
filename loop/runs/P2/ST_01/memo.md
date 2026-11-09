Helped: shrinking the model and taking more optimizer steps. Depth 8 to 6 (slot 0, 1.1026, delta -0.0119) was accepted. Halving the batch to 2**18 (slot 1, 1.0667/1.0643, delta -0.0372) gave the biggest gain. The incumbent is mean_adjusted 1.0655. Slot 1 carries a tripwire flag, so audit it before building on it.

Did not help: a bigger model (depth 10, dim 640: 1.1457, +0.0802), a 3x MLP at depth 8 (1.0745, +0.0090), HEAD_DIM 64 (1.0742, +0.0087), WEIGHT_DECAY 0.1 (+0.0013), and WARMDOWN_RATIO 0.7 or 0.3 (+0.0052, +0.0049). Schedule and regularization look near their optimum.

Near miss: MATRIX_LR 0.05 scored 1.0619 (-0.0036) but was rejected. Replicate it, since the margin is small. Value-embedding LR 0.3 was a wash (-0.0003).

Next: audit the slot 1 tripwire, then try batch 2**17, depth 4-5, and a learning-rate retune. Slot 9 (weight-decay schedule) failed to patch, so retry it.
