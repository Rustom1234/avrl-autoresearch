Helped: throughput changes under the fixed 300 s budget. Slot 6 (DEPTH 6, ASPECT_RATIO 85, dim 512) gave 1.1020, delta -0.0155. Slot 11 (halving total batch size) gave 1.0677/1.0657, delta -0.0294, but it has a tripwire flag, so audit it before building on it. Incumbent: mean_adjusted 1.0699.

Did not help: SwiGLU-lite MLP (+0.0068), sliding-window SSL (+0.0026), beta1 0.7 (+0.0013), SCALAR_LR 0.25 (+0.0014), FINAL_LR_FRAC 0.05 (+0.0039), warmup 0.03 (+0.0136), DEPTH 10 (+0.0329), HEAD_DIM 64 (+0.0029), betas (0.9, 0.99) (+0.0064), warmdown 0.3 (+0.0049), cosine decay (+0.0002). Warmdown 0.7 (-0.0015) and gradient clipping (-0.0014) were marginal, likely noise.

Next: run an audit of slot 11 first. Then try DEPTH 5 or 4 and a smaller batch again. Re-test warmdown 0.7 and clipping on the new incumbent. Use replicates for any gain under about 0.003. Skip further MLP, attention-pattern and schedule-shape changes.
