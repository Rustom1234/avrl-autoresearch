Incumbent mean_adjusted is now 1.0889 (single run), after two accepts. Slot 16 halved TOTAL_BATCH_SIZE to 2**18 and scored 1.0889 (-0.0178), the biggest gain. Slot 14 trained the first 30% of the budget on 512-token sequences. It scored 1.1092 and 1.1040 (-0.0080).

Helped: batch size and data/sequence changes, meaning more optimizer steps in the fixed 300 s.

Did not help: higher MATRIX_LR (0.05 and 0.06 both 1.1160), depth 10 (1.1150), WARMDOWN_RATIO 0.7 (1.1181), EMA (1.4066). The embedding/unembedding LR raise gave 1.1090 and 1.1228, so it was noisy and rejected.

Next: replicate slot 16 since it's a single run. Try an even smaller batch such as 2**17. Retest HEAD_DIM 64 (1.1127), grad clip (1.1122) and WARMDOWN_RATIO 0.3 (1.1135) on the new base. Retune the short-sequence phase length.
