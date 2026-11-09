Incumbent mean_adjusted is 1.1146 over 16 runs. No tried change has beaten it yet.

Helped slightly, but not enough to accept: EMBEDDING_LR 0.6 to 1.0 gave 1.1126 (-0.0020). FINAL_LR_FRAC 0.0 to 0.1 gave 1.1128 (-0.0017). Both were rejected, and each is a single run, so the gains may be noise. Replicating them or combining them is worth a slot.

Hurt: WARMUP_RATIO 1.0 gave 1.2070 (+0.0925), so avoid long warmup. WEIGHT_DECAY 0.05 gave 1.1174 (+0.0029). HEAD_DIM 256 gave 1.1193 (+0.0047).

Slots 4 and 5 failed to apply (patch_fail). Slot 4 was the EMBEDDING_LR retry. Slot 5 was Adam beta1 0.9. Slot 0 (gradient accumulation) was aborted. Next time, copy the SEARCH lines exactly from the current file.

Next: combine EMBEDDING_LR 1.0 with FINAL_LR_FRAC 0.1. Then try a higher WEIGHT_DECAY, and untried model-size or depth changes. Retry beta1.
