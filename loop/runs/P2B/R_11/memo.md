Incumbent: slot 5, mean_adjusted 1.0624 over 2 runs (official 1.0614 and 1.0635). It halved TOTAL_BATCH_SIZE on top of ASPECT_RATIO 48 (slot 3, -0.0140). The tripwire flag on slot 5 is still unexplained; check it.

Helped: a smaller model and a smaller total batch, since both give more optimizer steps in 300 s. Slot 17 (no grad accumulation) scored 1.0546/1.0574, which beat the incumbent's official numbers, but it was rejected at -0.0064. Re-examine why before trying it again.

Did not help: wider model (+0.0339), weight decay 0.01 (+0.0076), parameter-free norm on output projections (+0.0269), WARMDOWN_RATIO 80% (+0.0019), warmup 0.2 (+0.0047), MATRIX_LR 0.3 (+0.0403), DEVICE_BATCH_SIZE 32 (+0.0061, +0.0675). Neutral: slot 8 (-0.0004), SSSS (-0.0008), EMA (+0.0000), slot 15 (+0.0021).

Crashes: depth 12 and the 1/sqrt residual scaling. Slot 12 (dropout) failed to patch.

Next: lower the total batch further, ASPECT_RATIO 32 or 40, a lower MATRIX_LR, a small weight decay increase. Replicate the incumbent once.
