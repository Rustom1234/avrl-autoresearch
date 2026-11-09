Helped: only the architecture change in slot 3, DEPTH 8→6 (official 1.0832/1.0853, delta -0.0303, accepted). It carries an unresolved tripwire flag, so check it before building on it. Incumbent is mean_adjusted 1.0843 over 2 runs.

Did not help: optimizer and schedule tweaks (Muon warmup 150: -0.0002; WARMDOWN 0.3: +0.0004; WARMDOWN 0.7: -0.0016; value-embedding LR 0.3: +0.0006; WARMUP_RATIO 0.02 with FINAL_LR_FRAC 0.05: +0.0006). Batch changes (2**18: -0.0009; 2**17: -0.0011), SwiGLU (-0.0013), DEPTH 4 wide (-0.0029), DEPTH 5 (-0.0019), context 1024 (-0.0028) and wte init std 0.5 (-0.0012) were rejected. HEAD_DIM 64 was worse (+0.0050).

Odd case: UNEMBEDDING_LR 0.008 scored 1.0767/1.0797 (delta -0.0061) but was rejected. It may have been compared against the old incumbent, so recheck it.

Next: 1) resolve the tripwire and the slot 5 comparison; 2) retest UNEMBEDDING_LR 0.008 on DEPTH 6; 3) skip more small optimizer tweaks.
