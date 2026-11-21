Helped: only slot 7 was accepted (30% of time on random restarts, then perturbing the best packing), at -2.6310/-2.6316. Single runs are noisy (sd about 0.0023), so many single-run rejections may be unreliable.

Did not help: small tweaks to init radii (0, 2), analytic gradients (3), screening or looser tolerances (1, 10), LP refit (5), TIME_FRACTION 0.9 (9), MAX_ITER 500 (12), final polish (13), elite pools (19, 72), and small perturbation variants (22, 25, 26, 28). SLSQP bounds (56) were worse.

Promising: structured or diverse starts and bigger moves. Slot 61 (-2.6360/-2.6334), slot 45 (-2.6354/-2.6334), slot 50 (-2.6349/-2.6331), slot 69 (-2.6343/-2.6349), and slots 34, 41, 47, 54 (about -2.634). Most were rejected only because the decision rule is strict.

Next: adopt a base like slot 69 or 61 (staggered-row or maximin starts). Add the slot 45 window re-solve and the slot 31 shake. Tune the restart share (15-30%). Replicate every result. Avoid more small perturbation variants and duplicate ideas.
