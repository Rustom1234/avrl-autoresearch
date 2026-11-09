Helped: halving TOTAL_BATCH_SIZE to 2**18 with LR scaled by sqrt(batch ratio) gave 1.0916 (-0.0230), the only accept. More optimizer steps in the fixed 300 s seems to matter. Doubling the batch to 2**20 was much worse (1.1820, +0.0674), which fits. Dropping DEVICE_BATCH_SIZE to 32 gave 1.0938 (+0.0022), so it did not help.

Did not help: schedule changes (cosine 1.1233, 0.1 warmup 1.1272, 5 cosine stages 1.1395, plateau decay 1.1000), per-layer Muon LR (1.1167), epsilon floor (1.1149), tied lm_head (1.1291), MoE (1.1089), extra layer (1.1148). Gradient clipping (1.1122) and 5-stage step decay (1.1111) looked slightly better, but each is a single run, so the gaps may be noise.

Next: try 2**17 batch with sqrt LR scaling. Retune LR scale and weight decay at the new batch size. Replicate 1.0916. Retry clipping at 1.0 on the new incumbent. Skip more LR-shape and architecture tinkering. Skip gradient checkpointing, which was aborted four times.
