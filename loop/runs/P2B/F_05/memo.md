Helped: shrinking the model and cutting the batch. Slot 5 (DEPTH 8 to 6) was accepted at -0.0286. Slot 18 (total batch 2^19 to 2^18) was accepted at -0.0206, and the incumbent is now 1.0653 over 2 runs. Both accepts carried a "tripwire" flag, so check what triggered it before building on them. Run-to-run spread is about 0.005.

Did not help: window pattern SL (+0.0018), value embeddings on every layer (+0.0015), WARMUP_RATIO 0.03 (+0.0124), the Muon momentum warmup change (+0.4827, so leave momentum alone), and HEAD_DIM 64 (+0.0031). Small optimizer and schedule tweaks were rejected. Their deltas of -0.0011 to -0.0039 are within noise, and the largest is MATRIX_LR 0.05 at -0.0039.

Next: confirm the tripwire. Then try a smaller batch (2^17), DEPTH 5 or 4, or combine the small tweaks. Use exact-match SEARCH blocks, and don't propose changes already in the file.
