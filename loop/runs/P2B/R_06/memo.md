Incumbent is mean_adjusted 1.0662 over 2 runs. Slot 12 (TOTAL_BATCH_SIZE 2**19 to 2**18) gave 1.0652/1.0673 (-0.0177) and was accepted with a tripwire flag, so check it before building on it. It followed slot 5 (DEPTH 8 to 6, -0.0306, also tripwire).

What helped: smaller model and more optimizer steps, meaning architecture and batch size.

What did not: all schedule changes (cosine +0.0293/+0.0289/+0.0473, warmup +0.0113, FINAL_LR_FRAC, soft warmup, WARMDOWN_RATIO 0.3 at +0.0008, Muon decay following the LR at +0.0006). Also flat: WINDOW_PATTERN "L" (+0.0001) and dim 512 (-0.0002). Adam beta1 0.9 was worse (+0.0032).

Next: keep going on batch size (2**17), DEPTH 5 or 4, and retry UNEMBEDDING_LR 0.008. It gave 1.0777/1.0807 against the older incumbent, which was a small gain. Stop schedule work. Keep edits small.
