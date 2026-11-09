Incumbent mean_adjusted is 1.1146 over 16 runs. Nothing tried has clearly beaten it. Three single runs landed slightly below it: gradient-norm clipping at 1.0 (1.1122), previous-token key mixing (1.1127) and the gradient-norm EMA scale (1.1131). These gaps are small and each is one run, so they may be noise.

What hurt: cutting optimizer steps (TOTAL_BATCH_SIZE 2**20, +0.0704), checkpointing with a larger device batch (+0.0419), schedule reshaping (cosine +0.0205, three-phase +0.0143), and extra regularization (layer dropout +0.0151, label smoothing +0.0040). The gated MLP also hurt (+0.0088).

Next: replicate clipping first, then combine it with key mixing. After that, try a smaller TOTAL_BATCH_SIZE for more steps, then tune the learning rate and warmdown ratio, then depth/width trade-offs. Read train.py before proposing ideas. Many slots were wasted on no-edit ideas it already covers.
