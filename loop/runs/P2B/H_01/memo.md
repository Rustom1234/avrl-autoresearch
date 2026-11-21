Incumbent mean_adjusted is 1.1145 over 16 runs. Every slot so far was rejected, and the single-run deltas are small, so some may be noise.

Helped (negative delta): label smoothing 0.05 decaying to 0 (-0.0028), value-embedding LR 0.3 (-0.0028), depth 6 with aspect ratio 85 (-0.0027), EMBEDDING_LR 0.4 (-0.0024), WARMUP_RATIO 0.03 (-0.0022), TOTAL_BATCH_SIZE 2**18 (-0.0012), Muon weight decay 0.1 (-0.0007). WARMDOWN_RATIO 0.3 was neutral.

Hurt: q scale 1.2 (+0.0095), cosine warmdown (+0.0027), SwiGLU (+0.0034), HEAD_DIM 64 (+0.0011), depth 10 (+0.0192), the combined warmup/warmdown/final-LR change (+0.0121). The EMA run was broken (+0.5320), so its idea is untested.

Next: replicate the top signals (label smoothing, value-embedding LR, depth 6, EMBEDDING_LR, warmup). Then stack the ones that hold, one combination per slot. Drop attention-sharpening, head-count, decay-shape and wider-MLP tweaks.
