Status: the incumbent is now mean_adjusted 1.1068 over 2 runs. It comes from slot 12, ASPECT_RATIO 64 to 32 (official 1.1088/1.1049, delta -0.0078), the only accepted change. Architecture and model size are therefore the most promising direction. DEPTH 4 was far worse (+0.1675), so don't go shallower.

Hurt: TOTAL_BATCH_SIZE 2**20 (+0.0674), MATRIX_LR 0.2 (+0.0310) and 0.08 (+0.0038), warmup 0.2 (+0.0120) and 0.1 (+0.0099), Muon weight decay 0.5 (+0.0052) and 0.01 (+0.0048), WARMDOWN_RATIO 0.8 (+0.0048) and 0.2 (+0.0036). Weight decay 0.01 combined with SCALAR_LR 2.0 also hurt (+0.0166).

Small, unreplicated gains, probably noise: SCALAR_LR 1.0 (-0.0025), FINAL_LR_FRAC 0.2 (-0.0022). Both were measured against the old baseline, so re-test them on the new incumbent.

Next: try ASPECT_RATIO below 32, or DEPTH 10 at a small width. Try a batch size below 2**19. Then combine SCALAR_LR 1.0 with FINAL_LR_FRAC 0.2. Avoid warmup and large jumps.
