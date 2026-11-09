No change beat the incumbent (mean_adjusted 1.1146 over 16 runs) clearly. All results were single runs, so deltas under about 0.002 are probably noise.

Helped slightly, maybe noise: smaller batches (2**18: -0.0018, 2**17: -0.0011), HEAD_DIM 64 (-0.0023), WARMDOWN_RATIO 0.3 (-0.0021), SSSSSSSL window pattern (-0.0013), lm_head init 0.0005 (-0.0013), z-loss (-0.0012), gradient clipping (-0.0008).

Hurt: schedule shape changes (cosine warmdown +0.0043, combined warmup/warmdown/final-LR +0.0131), the momentum ramp (+0.0048), DEPTH 10 (+0.0474), Muon weight decay 0.1 (+0.0015), and the higher or lower embedding LR changes (+0.0009 each). The EMA (1.2974) was almost certainly broken.

Next: replicate the best small wins (HEAD_DIM 64, WARMDOWN_RATIO 0.3, 2**18 batch), then stack them with z-loss, lm_head init and the window pattern. Avoid no-op proposals and the EMA.
