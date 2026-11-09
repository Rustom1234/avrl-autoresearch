Incumbent mean_adjusted is 1.0746 over 2 runs. Two changes were accepted. Training on 1024-token sequences scored 1.0960 (-0.0186). Reducing TOTAL_BATCH_SIZE to 2^18 scored 1.0744/1.0747 (-0.0214), but it carries a tripwire flag, so check that before building on it.

Batch and data changes helped most. Both batch rows and one data row were accepted or near the top.

Did not help against the current incumbent: SwiGLU MLP (+0.0110), DEVICE_BATCH_SIZE 32 (+0.0206), and gradient clipping 1.0 with WARMDOWN_RATIO 0.3 (+0.0082). Earlier rejects were attention window, MATRIX_LR 0.05, softcap 30, momentum 0.97, DEPTH 6 and HEAD_DIM 64. Cosine warmdown (+0.1214) and EMA (+0.8169) were far worse.

Next: investigate the tripwire first. Then try sequence length or batch size changes near the accepted values, such as a smaller TOTAL_BATCH_SIZE. Also try small optimizer changes, like a slightly lower MATRIX_LR or a different weight decay. Avoid repeating SwiGLU, schedule-shape rewrites, EMA and attention-shape changes.
