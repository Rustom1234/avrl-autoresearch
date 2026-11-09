Helped: changes that give more optimizer steps in the fixed 300 s. DEPTH 8→6 (-0.0304, 1.0822/1.0861) and halving the batch (-0.0182, 1.0650/1.0669) were accepted, but both carry a tripwire flag; find out what it means before trusting them. Incumbent mean_adjusted is 1.0660 over 2 runs.

Small gains, rejected: TOTAL_BATCH_SIZE 2**17 (-0.0044, 1.0616), unembedding LR doubled (-0.0054, replicated 1.0607/1.0604), embedding LR 0.9 (-0.0011), WARMUP_RATIO 0.02 (-0.0004, noise-level), DEPTH 4 with ASPECT_RATIO 128 (-0.0002).

Hurt: DEPTH 10 (+0.0790), removing value embeddings (+0.0149), tied embeddings (+0.0083), HEAD_DIM 64 (+0.0042), SwiGLU (+0.0028).

Next: 1) Resubmit MATRIX_LR 0.05 with an exact SEARCH match (slot 6 failed to patch). 2) Combine 2**17 batch with doubled unembedding LR, and replicate. 3) Avoid more architecture tweaks (2 accepted, 8 rejected).
