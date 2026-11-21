Helped: only the batch-size cut. Slot 4 reduced TOTAL_BATCH_SIZE from 2^19 to 2^17, giving more optimizer steps in 300 s (official 1.0524/1.0516 over 2 runs; incumbent mean_adjusted 1.0520). It carries a tripwire flag, so check that before building on it.

Did not help: nearly everything else. Depth 10 (+0.0195), HEAD_DIM 64 (+0.0064), SwiGLU (+0.0077), warmup/schedule tweaks (+0.0093, +0.0034, +0.0134, +0.0153), and faster momentum ramp (+0.0026) all hurt. Several changes were neutral, within about 0.0004 of the incumbent: weight decay 0.1, scaled decay, shallower/wider model (DEPTH 6), the value-embedding gate, SSSS windows and warmdown 0.7. These are noise-level and not worth more slots.

Next: investigate the tripwire first. Then try 2^16 batch size with a retuned learning rate. Avoid more depth, head-count, warmup, schedule and activation changes. Slots 16 and the first slot 0 patch failed, so write SEARCH blocks carefully.
