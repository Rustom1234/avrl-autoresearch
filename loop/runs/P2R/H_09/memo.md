Incumbent is now slot 13: official 1.1035 (-0.0111), shrinking the short attention window from 1024 to 256 tokens. It is the only clear gain, and it is a single run, so replicate it before building on it.

Small gains within noise, not accepted: EMA (-0.0011), batch 2**18 (-0.0009), SSSSSSSL window pattern (-0.0010), DEPTH 6 with ASPECT_RATIO 128 (-0.0019), warmup 0.03 (-0.0006).

Did not help: depth 10 (+0.0514), HEAD_DIM 64 (+0.0120), z-loss (+0.0022), SCALAR_LR 0.25 (+0.0025), MATRIX_LR 0.05 with zero decay in warmdown (+0.0031), momentum 0.97 (+0.0003).

Next: replicate slot 13. Try a shorter window (128) and SSSSSSSL combined with the 256 window. Then stack the small gains (EMA, 2**18 batch, warmup 0.03) on the new base. Avoid model-size, head-dim and LR tweaks.
