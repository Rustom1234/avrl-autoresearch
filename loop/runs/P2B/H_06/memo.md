Incumbent mean_adjusted 1.1145 over 16 runs; 15 slots logged, one aborted duplicate. Every single-run result so far was rejected. Gains of 0.0007 to 0.0030 are probably within noise.

Helped weakly: SwiGLU MLP (slot 3, 1.1115, -0.0030) was best. Lower WEIGHT_DECAY 0.1 (slot 13, -0.0020) and batch 2**18 (slot 4, -0.0015) came next. DEPTH 12 (-0.0014), ASPECT_RATIO 80 (-0.0010), DEPTH 6 (-0.0008), q scale 1.2 (-0.0007) and WARMDOWN_RATIO 0.3 (-0.0004) were marginal. Cosine decay was neutral.

Hurt: EMA 0.99 (+0.0275), WARMUP_RATIO 0.03 (+0.0115), faster momentum ramp (+0.0067), ADAM_BETAS (0.8, 0.99) (+0.0039), SCALAR_LR 0.25 (+0.0016), MATRIX_LR 0.05 (+0.0011).

Next: replicate SwiGLU to check it is real. Then stack SwiGLU, weight decay 0.1 and batch 2**18 in one run. Try lower MATRIX_LR and embedding LR. Avoid warmup, EMA, momentum and Adam-beta tweaks.
