Helped: cutting compute per step so the fixed 300 s buys more steps. DEPTH 6 (slot 4) scored 1.0829 and 1.0825, delta -0.0318. Halving TOTAL_BATCH_SIZE to 2**18 (slot 10) scored 1.0655 and 1.0652, delta -0.0173. It is the incumbent at mean 1.0654. Both accepts carry a tripwire flag. Find out what it means before building on them.

Did not help: schedule tweaks (cosine warmdown +0.0030, WARMDOWN_RATIO 0.7 +0.0031, 0.3 +0.0006), HEAD_DIM 64 (+0.0122), SwiGLU (+0.0038), gradient clipping (+0.0002), Adam betas (+0.0014). Small gains, all single runs and possibly noise: value embeddings on every layer (-0.0050), momentum schedule (-0.0006), FINAL_LR_FRAC 0.05 (-0.0007), weight decay 0.1 (-0.0001).

Next: check the tripwire first. Then try DEPTH 5 or 4 and a batch of 2**17. Retune the learning rate for the smaller model and batch. Replicate the value-embedding change on the new incumbent, and retry the loss-sync change. Avoid more schedule tweaks.
