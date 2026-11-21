Helped: the batch change in slot 0 (TOTAL_BATCH_SIZE 2^17 with MATRIX_LR 0.032), official 1.0517/1.0515 over 2 runs. It is the incumbent at mean_adjusted 1.0516 and carries a tripwire flag, so resolve that before building on it.

Did not help: changing model size (DEPTH 6 gave 1.0617, DEPTH 10 gave 1.0676), HEAD_DIM 64 (1.0579), SwiGLU (1.0590), cosine decay (1.1141), MATRIX_LR 0.06 (1.0558), WARMDOWN_RATIO 0.3 (1.0554), Adam beta1 0.9 (1.0524), and batch 2^18 (1.0970).

Borderline single runs: SCALAR_LR 0.25 (1.0489, -0.0027), weight decay 0.1 (1.0501, -0.0015), WARMDOWN_RATIO 0.7 (1.0508, -0.0008), SSSS windows (1.0513) and SSL windows (1.0513), both -0.0003. These may be noise.

Next: replicate SCALAR_LR 0.25, then combine it with weight decay 0.1 and warmdown 0.7 to test whether the gains stack. Also try other batch-size and MATRIX_LR pairs. Slot 12 (Muon momentum ramp) failed to apply, so it could be retried with a corrected patch.
