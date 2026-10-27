Helped: two big, replicated wins, both carrying a tripwire flag that should be checked. DEPTH 8 to 6 gave official 1.0825/1.0828 (-0.0320). TOTAL_BATCH_SIZE 2^18 gave 1.0657/1.0641 (-0.0188). The incumbent is mean_adjusted 1.0641 over 2 runs.

Did not help (all rejected, within the noise floor of about 0.003): batch 2^17 (1.0615/1.0629, -0.0011), HEAD_DIM 64 (+0.0119), DEPTH 5 with ASPECT_RATIO 76 (+0.0031), SCALAR_LR 0.25 (+0.0044), WARMUP_RATIO 0.02 (+0.0058), WARMDOWN_RATIO 0.7 (+0.0062) and 0.75 (-0.0011), FINAL_LR_FRAC 0.1 (+0.0035), and the earlier weight decay, ADAM_BETAS and EMBEDDING_LR tweaks. Batch size looks saturated. Schedule tweaks are exhausted (a=1, b=6).

Wasted: three aborts and two patch failures. Copy SEARCH text exactly from the file.

Next: MATRIX_LR 0.03, which has never run because its patch failed. Also try Newton-Schulz with 4 iterations and a different width at DEPTH 6. Replicate any gain.
