Helped: only the schedule change in slot 2 (staged radius caps 0.1, 0.2, 0.35, ...). It scored -2.9364, which is -0.0112 against the previous incumbent. That is well above the noise sd of about 0.0034, and it is close to the published ~2.9370. It is still a single, unreplicated run.

Did not help: repair tweaks (slots 0, 1, 6), init changes (Sobol slot 3, Beta slot 10, greedy slot 9), the penalty objective (slot 4, +0.0105), capping only some radii (slot 8, +0.0114), the sum(r^1.5) objective (slot 34, +0.0001), and the randomized iteration cap and ftol (slot 23, +0.0010). Slots 5, 14 and 17 were within noise.

Slots 11 (-2.9396) and 12 (-2.9379) beat the incumbent, but each is a single run and the gaps (-0.0031, -0.0014) are under the noise sd.

Next: replicate the incumbent once. Then replicate slots 11 and 12 (basin hopping, annealing). Then tune the stage count, cap values and iterations per stage. Avoid more repair, init or penalty ideas. Many proposals were aborted as duplicates, so check the table first.
