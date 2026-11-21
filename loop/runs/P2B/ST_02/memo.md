Helped: cutting compute per step to fit the 300 s budget. DEPTH 8→6 (slot 2, -0.0288) and TOTAL_BATCH_SIZE 2^19→2^18 (slot 4, -0.0192) were accepted and replicated. Incumbent mean_adjusted is 1.0665 over 2 runs. Both accepted slots carry a tripwire flag whose meaning is unknown, so check it before building on them.

Did not help: cosine warmdown (+0.0063), MATRIX_LR 0.06 (+0.0017), GELU-tanh (+0.0128), ASPECT_RATIO 96 (+0.0343), smaller lm_head init (+0.0216), EMA (+0.1842). Slot 15 failed to apply (patch_fail) and could be retried.

Small single-run gains were rejected: batch 2^17 (-0.0046, official 1.0618, the best single score), WARMUP_RATIO 0.03 (-0.0026), gate 0.5 (-0.0021), reduced regularisation (-0.0013). All are under the 0.01 replicate threshold, so they may be noise.

Next: combine batch 2^17, warmup 0.03 and gate 0.5, then replicate. Try DEPTH 5, then retune MATRIX_LR. Avoid activation swaps and more schedule tweaks.
