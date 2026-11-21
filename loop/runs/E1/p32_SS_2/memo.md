Helped: only the slot 0 init (jittered staggered hex grid, 5-7 rows, random subset). Loss went from about -2.9253 to -2.9375, delta -0.0123. It is a single unreplicated run, so replicate it first.

Possibly real, within noise: analytic SLSQP gradient and Jacobian. Slots 6, 44, 47, 50, 65, 67 and 73 all gave -2.9396 (delta -0.0020). Pre-relaxation (slots 19, 33) and the shared-radius warm start (slot 42) gave the same value. Slot 58 combined gradient/Jacobian with pre-relaxation and gave -2.9378, so they don't stack. Slot 31 (rotated hex lattice) gave -2.9380, also within noise.

Did not help (+0.0006 to +0.0087): schedule changes, pools, basin hopping, kicks, swaps, LP repair, screening, pair pruning, and other init variants (irregular rows, mirror symmetry, big seed circles, square grids). Many were repeats and were aborted.

Next: replicate slot 0, then replicate slot 6 or 50 alone. Avoid those failed families. Treat gains under 0.003 as noise (run noise sd is about 0.0034).
