Incumbent is slot 13 at -2.6360, already below the best known 2.635. Slot 35 (Langevin noise) was accepted but gained about 0, so treat it as neutral.

Helped: multi-start SLSQP (-1.6318), better starts (hex row lattice -0.0201, uniform random centers -0.0127, ring init -0.0058), soft-disk pre-relaxation (-0.0036), annealed penalty (-0.0011), corner/edge seeding (-0.0011).

No gain: pools, screening, swap search, iterated local search, annealing, symmetric starts, polish passes, reparametrisations, schedule tweaks. Harmful: distance-restricted SLSQP (+0.0272), remove-and-reinsert (+0.0210), log-barrier (+0.0165). Seed variance is about 0.004 (slot 64 gave -2.6319), so gains below that are noise.

Next: the search looks saturated, with about 0 gain over the last ~50 slots. Many proposals were aborted as duplicates, so check the table first. Avoid polish, local-search and pool-breeding ideas. Try a genuinely different formulation, or make each start cheaper to get many more starts.
