Helped: changes to model size and batch size, which raise step count in the fixed 300 s. DEPTH 8->6 (1.0825/1.0823, -0.0321) and TOTAL_BATCH_SIZE 2**19->2**17 (1.0618/1.0630, -0.0200) were both replicated over 2 runs. Both carried a tripwire flag, so check it. Incumbent is mean_adjusted 1.0624.

Did not help: optimizer, schedule and regularization tweaks, all within about 0.002 of the incumbent (SwiGLU, MATRIX_LR 0.05 and 0.03, weight decay 0.1 and 0.3, WARMDOWN_RATIO 0.3, EMBEDDING_LR 0.9, momentum 0.97, max-autotune, warmup 0.03, grad clipping). Cosine decay was worse (+0.0020).

Closest near-miss: value embeddings on every layer (1.0580, -0.0043). It was rejected as a single run, so replicate it before dropping it.

Next: replicate the all-layer value embeddings. Try DEPTH 5 or 4, a narrower width, and a batch of 2**16. Replicate any win.
