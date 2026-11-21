Incumbent is now mean_adjusted 1.1079 over 2 runs. Slot 7 was accepted: raising UNEMBEDDING_LR from 0.004 to 0.008 scored 1.1089/1.1069 (-0.0066). Lowering it to 0.002 hurt (+0.0208), so output-layer LR looks like the best lever. Next: try UNEMBEDDING_LR 0.012 or 0.016.

Did not help: WARMUP_RATIO 0.2 (+0.0139), cosine schedule (+0.2496), EMBEDDING_LR 0.1 (+0.0112), weight decay 0.3 (+0.0031), and DEPTH 10 (+0.0814, likely too few steps in 300 s). FINAL_LR_FRAC values of 0.01 and 0.05 gave -0.0023 each, and 0.2 gave -0.0012. They were rejected as within noise, but they are consistent and may be worth a replicate.

Untried: smaller warmup, batch size, smaller or narrower models, and architecture tweaks. Avoid large schedule rewrites. Slot 8 is not in the table.
