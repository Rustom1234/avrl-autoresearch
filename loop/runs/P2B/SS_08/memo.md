Incumbent is mean_adjusted 1.1075 over 2 runs. It comes from the one accepted change, slot 11 (data family): reshaping batches into shorter sequences early in training. It scored 1.1091 and 1.1058 (-0.0070), so build on it.

Small possible gains, all within noise: UNEMBEDDING_LR 0.008 (-0.0041 over 2 runs), SwiGLU (-0.0009), lambda init (-0.0004), weight decay 0.1 (-0.0002), QK scale 1.2 (-0.0003).

Hurt: HEAD_DIM 64 (+0.0137), warmup (+0.0102), the 1.5x MLP LR (+0.0063), value embeddings on every layer (+0.0047), longer warmdown (+0.0039), EMBEDDING_LR 0.4 (+0.0027). The EMA swap gave 1.2599 (+0.1524), probably a bug. max-autotune was overbudget, so avoid it.

Next: explore more data/sequence-length changes, such as a longer short-sequence phase or a different length. Then stack them with UNEMBEDDING_LR 0.008 and SwiGLU. Replicate any gain under 0.005.
