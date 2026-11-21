Helped: shrinking the model and raising optimizer step count. DEPTH 8 to 6 (slot 4, 1.0834/1.0831, -0.0313) and total batch 2**18 (slot 11, 1.0653/1.0680, -0.0166) were the only accepts, both replicated. Both carried a tripwire flag, so check it before building further. Incumbent mean is 1.0666.

Did not help: DEPTH 5 with ASPECT_RATIO 76 (+0.0048), HEAD_DIM 64 (+0.0068), MATRIX_LR 0.03 (+0.0016), warmup 0.03 (+0.0109), scalar LR 0.25 (+0.0011). Small tweaks on the new base were within noise: EMBEDDING_LR 0.4 (-0.0002), warmdown 0.65 (-0.0008), Adam 0.9/0.99 with clipping (-0.0013). Several slots aborted as redundant.

Next: push the batch axis, for example total batch 2**17, and combine the small gains (clipping, weight decay 0.1, shorter momentum ramp). Investigate the tripwire flag. Avoid warmup and further depth cuts.
