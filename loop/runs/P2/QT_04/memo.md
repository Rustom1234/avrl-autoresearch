Helped: shrinking the model. ASPECT_RATIO 64 to 48 (dim 512 to 384) gave 1.1019, -0.0127, the only accepted change. Smaller models get more steps in 300 s. Bigger hurt: depth 10 gave 1.1648 (+0.0502).

Did not help: schedule changes (+0.0108, +0.0045, +0.0039, +0.0048, +0.0021), Adam betas (+0.0434, +0.0282), lower weight decay (+0.0008, +0.0010, within noise), UNEMBEDDING_LR increase (+0.0206). DEVICE_BATCH_SIZE 16 was neutral (+0.0002).

Possible small gains, each from a single run, so they may be noise: WINDOW_PATTERN SSSS 1.0987 (-0.0032) and the matrix LR doubling 1.0981 (-0.0038, best official number). Replicate both, then try combining them. Slot 12 (2M batch) failed to apply, so retry it with a correct SEARCH block.

Next: ASPECT_RATIO 40 or 32, DEPTH 6, and a replicate of the incumbent to measure noise. Avoid betas changes.
