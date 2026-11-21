Helped: constructive insertion moves. Slot 0 (remove and reinsert the smallest circle) gave -0.0031. Slot 47 (grow about half the restarts from 4 circles to n by insertion) is the incumbent at -2.6347 (runs -2.6348/-2.6346, delta -0.0036).

Did not help: most single-run tweaks moved the loss by about ±0.0002. These include radii inits, wall placement, time fraction 0.9, early stopping and LP repair. Hurt: basin hopping (+0.0030), position screening (+0.0023), sparse constraints (+0.0021), ring starts (+0.0037), mirror starts (+0.0036), greedy-hole starts (+0.0030), quadrant starts (+0.0060).

Noise sd is 0.0023, so single runs mislead. Slots 57 and 65 each scored -2.6360 once but were rejected.

Next: stack near-misses on slot 47 and replicate each at least twice. Candidates: slot 18 annealing acceptance, slot 4 Latin hypercube starts, slot 30 hex/uniform alternation, slot 32 analytic gradients, slot 57 rotated lattice starts and slot 65 staged SLSQP. Also try a larger share of insertion-growth restarts.
