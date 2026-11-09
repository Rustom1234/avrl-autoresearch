Incumbent: slot 8, mean_adjusted 1.0690 over 2 runs (1.0687/1.0693). Replicate noise is small (0.0006), but slot 8 carries a tripwire flag, so check it before building on it.

Helped: smaller batch only. 2**18 gave 1.0989 (-0.0157) and 2**17 gave -0.0299. 2**20 hurt (1.1803). The regime is step-limited.

Did not help: schedule changes (longer warmdown 1.1022, warmup 1.1058, polynomial 1.1043, step 1.1034), weight decay 0.3 (1.1110), lower UNEMBEDDING_LR (1.1112), ASPECT_RATIO 80 (1.0783), sparse/dense attention (1.0701), lower EMBEDDING_LR (1.0710), variable batch (1.0689) and a separate value-embedding Adam group (1.0693). Those last three are within noise of the incumbent.

Next: first resolve the tripwire. Then try a batch below 2**17 with scaled LRs. Try a higher UNEMBEDDING_LR. Replicate the shorter warmdown (1.0662, -0.0028) against the new incumbent. Skip further schedule and architecture tweaks.
