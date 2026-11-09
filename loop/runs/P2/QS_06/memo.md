Incumbent: mean_adjusted 1.1054 over 2 runs. Only accepted change is slot 5, DEVICE_BATCH_SIZE 16 with the same total batch (1.1075/1.1032, -0.0092). Micro-batch and batch-mechanics changes are the one lever that has worked.

Did not help: schedule changes (warmup 0.05 gave +0.0094, WARMDOWN_RATIO 0.7 gave +0.0112, warmup plus cosine gave +0.0219). Weight decay moves (0.05, 0.08, MLP-only) all lost. MATRIX_LR 0.06 gave +0.0033 and 0.08 gave +0.0036. UNEMBEDDING_LR 0.001 gave +0.0390. TOTAL_BATCH_SIZE 2**20 gave +0.0819. Narrower model gave +0.1675. ASPECT_RATIO 96 gave +0.0344. SwiGLU gave +0.0032.

Next:
- Test halved init on the new incumbent. It gave -0.0037 on the old one.
- Try DEVICE_BATCH_SIZE 8.
- Try MATRIX_LR slightly below 0.04.
- Try a higher UNEMBEDDING_LR.
- Avoid schedule, weight decay, large-batch, width and shrink changes.
- Three proposals were no-ops, and ASPECT_RATIO 96 was proposed again after it had failed, so read train.py and the table first.
