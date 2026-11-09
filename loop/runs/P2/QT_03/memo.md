Helped: only the batch-size reduction. Slot 4 (batch 16 with gradient accumulation) scored 1.1063 and 1.1010, delta -0.0109, so the incumbent mean_adjusted is 1.1037 over 2 runs. The two runs differ by about 0.005, so gains of that size are near noise.

Did not help: every schedule change (slots 0-3, 8), with deltas from +0.0077 to +0.0588. Cosine variants were worst. Lowering SCALAR_LR to 0.02 (+0.0032), EMBEDDING_LR to 0.3 (+0.0080) and raising weight decay to 0.5 (+0.0023) were also rejected. Cutting the LM head or matrix learning rate was harmful (+0.1258 to +0.4673). Slot 16 crashed and its fix in slot 17 scored +0.0031.

Closest miss: ASPECT_RATIO 48 scored 1.1016, delta -0.0021, which is within noise. Replicate it.

Next: try batch 8 and the batch-size/LR interaction. Try other model-size variants. Avoid more schedule and LR-lowering work.
