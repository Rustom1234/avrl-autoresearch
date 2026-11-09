Incumbent is now mean_adjusted 1.1063 over 2 runs (the earlier note's 1.1146 is out of date). The one accepted change was slot 9: shrinking the sliding-window scratchpad for short-window layers to a quarter, which gave 1.1072/1.1054, delta -0.0083. Slot 13 went further, to an eighth, and gave 1.1041 (delta -0.0022) but was rejected. Replicate it or try a middle value, since that gain may be real.

Small init tweaks (slots 4, 5) and adaptive LR (slot 0) gave small deltas of -0.0018, -0.0045 and -0.0004, all rejected. Replicate noise is about 0.0015. Warmup plus cosine hurt (+0.0097). Removing dead gradient-clipping code (slot 18) gave +0.0012 and was rejected. Slot 12 (+0.0014) was rejected too.

Crashes and patch failures cost many slots. Make edits small, exact and tested. Next: attention-window variants, then other architecture changes. Avoid schedule changes.
