Helped: only a smaller model. DEPTH 6 (slot 8) scored 1.0825 and 1.0821, delta -0.0322, so the incumbent is 1.0823 over 2 runs. It carries a tripwire flag, so check that before building on it.

Did not help: larger or wider models, because they get fewer steps in 300 s (DEPTH 10 +0.0498, HEAD_DIM 64 +0.0166, ASPECT_RATIO 48 with DEPTH 10 +0.0516, ASPECT_RATIO 48 with DEPTH 8 +0.0176). Small tweaks on DEPTH 6 are within noise: gradient clipping +0.0003, WINDOW_PATTERN SSSSSSL +0.0003, matrix LR plus beta2 -0.0001, SwiGLU +0.0034, warmdown 0.7 +0.0025, schedule change +0.0053.

Next:
1. Investigate the tripwire.
2. Try DEPTH 5 or 4.
3. Retry the two patch_fail edits (beta1 0.9, batch 2**18) with exact SEARCH text. A smaller batch may suit a smaller model.
4. Retest head LR 0.008, FINAL_LR_FRAC 0.05 and momentum 0.88 on DEPTH 6. They were measured on the old baseline.
