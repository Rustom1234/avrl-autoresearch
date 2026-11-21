Helped: only slot 14, which fixed the hex/grid-seeded starts (-2.9309 and -2.9323, delta -0.0063, accepted). Incumbent is -2.9316 over 2 runs. Structured, diversified starts look like the useful direction.

Possible gains, not yet confirmed: slot 29 (power-mean objective with annealed p) scored -2.9358 and -2.9377 over 2 runs, but it was rejected at delta -0.0051. Slot 22 (perturbing the best packing) scored -2.9344, delta -0.0028, from one run. Slot 36 (Lloyd relaxation) scored -2.9338, delta -0.0022.

Did not help: two-phase refinement (-2.9223), L-BFGS-B penalty continuation (-2.9107, worst), remove-and-reinsert (-2.9227) and simulated annealing (-2.9238). Polish, adaptive MAX_ITER, Beta centres and density-based radii were within noise (sd about 0.0034).

Process: 17 slots failed with engine:patch_fail, 3 were aborted and 4 crashed. Use small SEARCH blocks copied exactly from train.py, each matching once.

Next: re-check slot 29, since it has 2 runs and the best scores, and replicate slot 22. Then try combining the power-mean objective with perturbation and Lloyd starts. Replicate before accepting.
