Helped: only batch-size reductions, i.e. more optimizer steps in the fixed 300 s. 2**18 (slot 4) gave 1.0963 (-0.0183). 2**17 (slot 12) gave 1.0518/1.0513 (-0.0447) and is the incumbent, mean_adjusted 1.0516. It carries a tripwire flag, so check it.

Did not help: shrinking depth to 6 (+0.0110), HEAD_DIM 64 (+0.0056), MATRIX_LR 0.05 (+0.0009), momentum ramp end 0.97 (+0.0012), grad clipping at 1.0 (+0.0011). Small or noise-level: WARMDOWN_RATIO 0.3, lm_head LR 0.008, ADAM_BETAS, FINAL_LR_FRAC 0.05, SCALAR_LR 0.25 (1.0494, -0.0022).

Caveat: slots 5-11 were judged against the 2**18 reference, so their deltas are stale. Recheck them on the 2**17 incumbent.

Next: investigate the tripwire flag. Retest SCALAR_LR 0.25, WARMUP_RATIO 0.03, EMBEDDING_LR 0.4 and the value-embedding init, and combine any that hold. Also try a smaller batch such as 2**16, if the code allows it.
