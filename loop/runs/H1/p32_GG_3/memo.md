Nothing has beaten the incumbent (-2.9253). Noise sd is about 0.0034, so every measured delta is within noise.

What did not help:
- Repair tweaks (slots 10, 19, 20, 23, 24, 27, 33, 35) all scored -2.9246. That is +0.0007 against the incumbent, so these edits change nothing measurable.
- Random initial radii scored -2.9239 (slot 16) and -2.9225 (slot 21).
- Force-directed init scored -2.9243 (slot 6).
- Simulated annealing in place of SLSQP scored -1.3670 (slot 1), far worse.
- A decay factor in the repair loop scored -2.9199 (slot 32), the worst of the valid SLSQP-based edits.

Only slot 0 (jittered grid init, -2.9278, delta -0.0025) beat the incumbent. That is inside the noise, so it needs a replicate.

Failures: 14 slots had no valid result. Most were patch_fail from SEARCH blocks that did not match once. Slots 11, 18, 22, 26 and 34 crashed. Slot 12 returned -0.0000.

Next:
1. Replicate slot 0.
2. Use short, verbatim, unique SEARCH anchors, with one small edit per slot.
3. Keep SLSQP.
4. Try basin-hopping from the best solution.
5. Run cheap SLSQP first, then polish only the top candidates.
6. Check validity before the final return.
