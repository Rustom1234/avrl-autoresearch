Helped: only the batch change. Cutting TOTAL_BATCH_SIZE from 2**19 to 2**17 gave -0.0634 (runs 1.0516/1.0529); incumbent mean is 1.0514. The slot 0 tripwire flag still needs checking.

Did not help: architecture (a=1, b=7), optimizer (a=1, b=5) and schedule tweaks. Worse: Adam beta1 0.9 (+0.0063), SwiGLU (+0.0032), DEPTH 6 (+0.0125), HEAD_DIM 64 (+0.0034), sync every 10 steps (+0.0044), and the batch ramp (+0.0219, 1.0692). Small gains of 0.003 or less may be noise: Muon momentum ramp 1.0506 (-0.0035), WINDOW_PATTERN "SL" (-0.0031), UNEMBEDDING_LR 0.008 (-0.0009 over 2 runs), WARMDOWN_RATIO 0.7 (-0.0006).

Next: try a fixed 2**16 batch, retuning LR and warmdown for it. Try combining the momentum ramp, UNEMBEDDING_LR 0.008 and WINDOW_PATTERN "SL", and replicate any combined gain. Avoid more MLP, activation or depth swaps.
