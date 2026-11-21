Helped: only slot 3, cutting TOTAL_BATCH_SIZE from 2**19 to 2**17 (official 1.0515/1.0514, delta -0.0630 over 2 runs). It is the incumbent but carries a tripwire flag, so replicate or audit it before building on it. More optimizer steps in 300 s is the clear lever.

Marginal: slot 18 (MATRIX_LR 0.028, EMBEDDING_LR 0.42) scored 1.0494, -0.0021, rejected. Slot 10 (sequence length) scored -0.0045 over 2 runs, also rejected. Both point to retuning for the smaller batch.

Did not help: schedule and optimizer tweaks (warmdown 0.7 +0.0030, cosine decay +0.0045, beta2 0.99 +0.0012, warmup 0.03 +0.0024, weight decay 0.3 +0.0009, MATRIX_LR 0.05 +0.0027, z-loss +0.0012), DEPTH 6 (+0.0105), 3 Newton-Schulz iterations (+0.0095). Slots 12 and 13 (EMA) crashed. Slots 6 and 9 aborted because the change was already in the code.

Next: re-check the tripwire. Then try batch 2**16 and lower learning rates. Check the current file before proposing changes. Avoid small schedule tweaks.
