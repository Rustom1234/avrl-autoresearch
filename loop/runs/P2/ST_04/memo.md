Helped: changes giving more optimizer steps in 300 s. TOTAL_BATCH_SIZE 2**18 (-0.0144) and DEPTH 6 with ASPECT_RATIO 85 (-0.0360, official 1.0681/1.0602, tripwire flag) are the only accepts. Incumbent mean_adjusted is 1.0641 over 2 runs.

Did not help: larger models (depth 10, +0.0191), HEAD_DIM 64 (+0.0122), WARMUP_RATIO 0.02 (+0.0108), cosine warmdown (+0.0574), SSSS windows (+0.0036), MATRIX_LR 0.05, fused tweaks, and the momentum ramp change (+0.0061). Weight decay failed three times (+0.0024, +0.0032), so skip it.

Near misses, each with 1 run: SCALAR_LR 0.25 (-0.0042), embd/unembd LR rebalance (-0.0032), MATRIX_LR 0.03 (-0.0009). These are small and may be noise, since the slot 7 replicates differ by about 0.008.

Next: first replicate or audit the slot 7 incumbent. Then try combining the LR near misses, and try batch 2**17 or a shallower model.
