Incumbent is mean_adjusted 1.0987 (slot 11, official 1.0987), set by cutting TOTAL_BATCH_SIZE from 2^19 to 2^18 (-0.0158). More optimizer steps in the fixed 300 s is the one clear win. Batch/step-count changes helped most.

Small gains that were rejected: momentum ramp (-0.0020), embedding/lm_head weight decay (-0.0019), gradient clipping (-0.0093 delta, two runs at 1.0906/1.0883), sequence curriculum (-0.0082, two runs at 1.0918/1.0893). Several of these are probably real but were judged against the earlier incumbent. Re-test them on top of 2^18.

Hurt: branch scales (+0.0042), weight averaging (+0.0037), t+2 head (+0.0049), warmup 0.02 (+0.0123). Drop these.

Next: try gradient clipping and the momentum ramp on the new baseline, then combine them. Try an even smaller batch, such as 2^17. Use short single-line SEARCH text to avoid patch failures. Avoid the logsumexp regularizer, which was flagged for target_leak.
