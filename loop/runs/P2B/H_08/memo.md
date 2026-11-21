Status: 20 slots are listed (0-19). Only slot 9 was accepted (1.0969, -0.0176). It combined WD 0.1, total batch 2**18 and other changes. It was never replicated, so the gain is unconfirmed.

Helped, but each was rejected against the incumbent on its own: MATRIX_LR 0.03 (1.0915, -0.0055, the best score), WD 0.05 (1.0933, -0.0036), DEPTH=6 with dim 512 (1.0936, -0.0033), SCALAR_LR 0.25 (1.0951, -0.0018).

Did not help: cosine decay (+0.0072), HEAD_DIM 64 (+0.0072), grad clipping (+0.0006), EMA (+0.0018), MATRIX_LR 0.03 with UNEMBEDDING_LR 0.008 (+0.0006). Slots 12 and 13 aborted as no-ops, so check the current file before proposing.

Next: stack the small wins in one run: MATRIX_LR 0.03, WD 0.05, DEPTH=6 and SCALAR_LR 0.25. Then replicate the best result. Avoid further single-change tweaks. Leave UNEMBEDDING_LR alone.
