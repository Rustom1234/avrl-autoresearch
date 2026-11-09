Helped: smaller, faster-stepping configs. DEPTH 6 with ASPECT_RATIO 85 (slot 0, 1.1019, delta -0.0126) and TOTAL_BATCH_SIZE 2**18 (slot 2, runs 1.0689/1.0581, delta -0.0384) gave gains because more optimizer steps fit in 300 s. Slot 2 was flagged tripwire and its two runs differ, so the incumbent mean 1.0635 is noisy; replicate it.

Did not help: WARMUP_RATIO 0.02 (+0.0080), HEAD_DIM 64 (+0.0110), DEPTH 10 (+0.0575), model dim 640 (+0.0388), DEPTH 4 (+0.0027), EMBEDDING_LR 0.4 (+0.0050), WARMDOWN_RATIO 0.3 (+0.0066) and 0.7 (+0.0069), Muon momentum 0.97 (+0.0067), weight decay 0.05 (+0.0057). MATRIX_LR 0.05 was neutral (+0.0002). Bigger models hurt, and most schedule and optimizer tweaks sit near noise.

Close call: UNEMBEDDING_LR 0.008 scored 1.0612 (-0.0023). Retest with a replicate.

Avoid: compile-time risk (slot 11 was overbudget). Check that SEARCH text matches exactly (slot 9 failed to patch).

Next: batch 2**17, then the UNEMBEDDING_LR replicate.
