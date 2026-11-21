Helped: batch size. Slot 2 (DEVICE_BATCH_SIZE 64 to 32, more gradient accumulation) gave 1.1051/1.1036 over 2 runs, delta -0.0101, accepted. Incumbent is mean_adjusted 1.1044.

Promising but unresolved: slot 14 (ASPECT_RATIO 64 to 48) gave 1.0978/1.1001 over 2 runs, delta -0.0054, yet was rejected. Re-examine it, since it looks better than the incumbent. Smaller model dimension may suit the fixed 300 s budget.

Did not help: regularization (weight decay 0.4 gave 1.1118; label smoothing +0.0003; embedding dropout 1.1112; no embedding decay +0.0009). Schedules were worse (cosine 1.1627; periodic cosine 1.1103). Full-context window pattern gave 1.1084. Lower embedding/unembedding LR gave 1.2175, clearly worse.

Process: many crashes and patch failures (slots 4, 6, 9, 8, 12, 13, 18). Keep edits small, and make SEARCH text match exactly.

Next: combine the aspect ratio change with the batch change. Try DEVICE_BATCH_SIZE 16, model size and depth, and LR or warmdown ratio. Replicate any small gain; noise is about 0.001.
