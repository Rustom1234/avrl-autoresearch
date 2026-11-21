Nothing beat the incumbent (-2.6280 over 10 runs). Only 8 slots gave a score. Most deltas are within the noise sd of about 0.0023, so they mean no gain, not clear harm.

Did not help: radius-init changes (slots 0, 1, 22: +0.0027, +0.0026, +0.0014), repair tweaks (slots 5, 6, 28: +0.0019 each), MAX_ITER 1000 (slot 23: +0.0019), and a jittered grid start (slot 18: +0.0075). Slot 27 was +0.0001, which is flat.

Only slightly positive: seeding from the best packing (slot 16: -0.0010) and a local expansion post-step (slot 24: -0.0017). Both are within noise. Neither was replicated.

Main problem: patching. Many slots ended as patch_fail, abort or crash (slots 11, 14, 15, 29, 30, 35).

Next:
1. Make one small edit per slot, copying exact lines for the SEARCH block.
2. Combine best-packing perturb-and-reoptimize with local expansion, and replicate it.
3. Try fewer, longer SLSQP runs.
4. Drop init and repair tweaks.
5. Keep the packing valid within 1e-10.
