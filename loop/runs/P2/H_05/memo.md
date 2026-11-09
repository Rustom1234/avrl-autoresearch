Incumbent is now slot 16: TOTAL_BATCH_SIZE 2**18 gave 1.0957 (-0.0189), the only accepted change and by far the largest gain. The earlier notes (incumbent 1.1146 over 16 runs) are stale. Earlier single-knob tweaks gave only about -0.002 each, which is likely noise.

Did not help on top of 2**18: SwiGLU (1.0975, +0.0019), WARMDOWN_RATIO 0.7 (1.0970, +0.0013). Combining EMBEDDING_LR 0.4, MATRIX_LR 0.05 and WARMDOWN_RATIO 0.3 hurt (1.1041, +0.0084). Those small winners don't stack, and they were measured at the old batch size. Earlier hurts: warmup (+0.0099), HEAD_DIM 64 (+0.0175), SSSS windows (+0.0013).

Next: explore batch size further (2**18 is better than both 2**17 and 2**19 as tested, so try nearby values or retune LR at 2**18). Test single LR changes one at a time at 2**18, with a replicate of any promising result. Avoid stacking untested tweaks.
