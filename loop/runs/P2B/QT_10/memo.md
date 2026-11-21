Incumbent mean_adjusted is 1.1145 over 16 runs. No tested change has beaten it with confidence.

Did not help: large optimizer changes (Adam betas 0.9/0.999 with higher BASE_LR, +0.0858), the first cosine schedule (+0.0407), weight decay 0.5 (+0.0066), SwiGLU (+0.0115), warmup 0.1 (+0.0120), and the second cosine attempt (+0.0031).

Tiny gains, all within likely noise and all rejected: clipping at max_norm=0.5 (-0.0007), ADAM_ALPHA=0.1 (-0.0005), outlier-batch clipping (-0.0002), and Lookahead removed (-0.0001). Slot 8 crashed. Two slots were aborted.

Takeaway: the incumbent's optimizer and schedule are well tuned. Clipping variants are the only mildly positive family.

Next: replicate the clipping at 0.5 to check whether -0.0007 is real. Then try model size, depth/width and batch size tradeoffs for the 300 s budget, and small warmdown-fraction tweaks. Avoid big rewrites.
