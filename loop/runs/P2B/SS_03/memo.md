Helped: more optimizer steps under the fixed 300 s. Halving TOTAL_BATCH_SIZE to 2**18 (slot 4, 1.0952, -0.0193) and shrinking the model to DEPTH 6 with ASPECT_RATIO 80 (slot 9, runs 1.0670/1.0598, -0.0319, tripwire flagged) both helped. Incumbent is mean_adjusted 1.0634 over 2 runs.

Did not help: wider model (ASPECT_RATIO 80 at depth 8, +0.0249), HEAD_DIM 64 (+0.0119), SwiGLU, WINDOW_PATTERN SSL (+0.0047), WARMUP_RATIO 0.03 (+0.0029), WARMDOWN_RATIO 0.3 (+0.0039) or 0.7 (+0.0052), Muon momentum ramp (+0.0060), Muon weight decay 0.1 (+0.0059).

Marginal and within noise: grad clip 1.0 (1.0602, -0.0031), lm_head init halved (1.0623, -0.0011). Seed noise is large (1.1077 vs 1.1167), so small deltas are unreliable.

Next: check the slot 9 tripwire with a replicate. Try TOTAL_BATCH_SIZE 2**17 on the small model. Combine grad clip and lm_head init and replicate them. Retest lm_head LR and SCALAR_LR on the new incumbent.
