Helped: shrinking compute per step so more steps fit in 300 s. Slot 6 (DEPTH 8→6, ASPECT_RATIO 64→48) was accepted at 1.0827/1.0840. Slot 12 (TOTAL_BATCH_SIZE halved to 2**18) was accepted at 1.0654/1.0649, so the incumbent is now mean_adjusted 1.0652. Both carry a tripwire flag. Check what it means before building further.

Did not help: schedule changes (exponential +0.0866, cosine +0.0040, warmup +0.0131), gradient checkpointing (+0.0349), U-Net skips (+0.0176), the 2**20 batch (+0.0748), HEAD_DIM 64 (+0.0212), all-dense attention "L" (+0.0014), Muon weight decay 1.0 (+0.0038) and per-tensor EMA clipping (+0.4500). Clip 5.0 (-0.0003), per-layer LR decay (-0.0022) and momentum rescale (-0.0011) were rejected; gaps under 0.005 are noise.

Next: try an even smaller batch (2**17), a slightly narrower or shallower model, and LR sweeps for Adam and Muon. Avoid anything that adds compute per step.
