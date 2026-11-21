Helped: only slot 9 (30% random restarts, then perturbing the best packing with an acceptance rule): -2.6360, delta -0.0080, about 3.5 noise sd. It is a single run and has not been replicated, so it may be luck. Near misses, within noise of slot 9: slot 17 adaptive sd (-2.6354) and slot 28 walker (-2.6350). Weakly positive: slot 4 size-class start (-0.0024).

Did not help: SLSQP speed-ups and constraint forms (slots 0, 1, 2, 11), TIME_FRACTION 0.9, LP refit, staged jitter, remove-and-reinsert, short-iteration screening, alternative starts (sum r^p, Beta, lattice, pre-relaxation), swap and crossover moves, elite pools (+0.0092), always shaking (+0.0120), longer refinement maxiter, top-3 basin-hopping, and a 15% explore phase (-2.6318).

Next: 1) replicate slot 9 to confirm it. 2) Tune its perturbation size and restart fraction. 3) Combine it with the slot 4 start. 4) Skip micro-optimisations and further start-generation ideas.
