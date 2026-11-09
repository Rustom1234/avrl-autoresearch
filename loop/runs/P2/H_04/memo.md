Incumbent is now 1.0960 (slot 10), from halving TOTAL_BATCH_SIZE to 2**18. That was the only accepted change and by far the biggest gain (-0.0185). More optimizer steps in the 300 s budget matter most.

The batch change moved the baseline, so the earlier single-run deltas are not comparable to it. Against the new incumbent:
- UNEMBEDDING_LR 0.008 scored 1.0874/1.0899 (-0.0074). It was rejected, but it looks promising and is worth re-testing.
- EMA scored 1.0900/1.0927 (-0.0047), also rejected.
- Stacking init and beta2 scored 1.0941 (-0.0019).
- WARMDOWN_RATIO 0.7 was worse (+0.0027).

Try next:
1. Go smaller on batch (2**17).
2. Combine UNEMBEDDING_LR 0.008 with EMA.
3. Re-test the init and beta2 changes on the new baseline.

Leave MATRIX_LR, weight decay and the MLP alone.
