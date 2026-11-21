All 40 slots (0-39) are used. Incumbent is still slot 8, the radius-cap SLSQP continuation: -2.6343, one run, never replicated (noise sd 0.0023). It is the only clear win (delta -0.0063). It improved per-restart SLSQP convergence.

Slots 39 (-2.6354), 28 (-2.6360) and 16 (-2.6354) came out lower than the incumbent. Their deltas are -0.0012, -0.0017 and -0.0012, which is within noise. Slots 10, 9 and 15 were neutral. Slots 13, 23, 31, 32 and 36 matched -2.6343 or came within 0.0003 of it.

Hurt: extra stages, other solvers (slots 24, 25), pair pruning (slot 18), start-shape tweaks (19, 20, 29, 30, 34, 35, 37), penalties (26, 27) and schedule priors (38).

Next: replicate slot 8, then replicate slots 28, 16 and 39 and stack the ones that hold. Tune the cap schedule. Avoid new penalties, solvers and start-shape changes.
