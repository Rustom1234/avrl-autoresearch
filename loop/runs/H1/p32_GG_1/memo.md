Helped: only slot 4, the jittered-grid init in place of uniform random. It scored -2.9301 and -2.9332 (mean -2.9317, delta -0.0064). That is modest next to the noise sd of 0.0034, but it was replicated.

Did not help: other inits (power-law -2.9257, simulated annealing -2.9223, greedy best-fit -2.9095). Repair and iteration tweaks (slots 10, 17, 18, 20, 25, 29, 31, 33) all gave -2.9301, which is within noise and no better than slot 4. The post-SLSQP shake in slot 34 was worse (-2.8719), and slot 7 gave -2.9167.

Failures: 9 patch_fail slots (2, 3, 8, 9, 11, 12, 14, 15, 16) and several crashes (6, 19, 21, 22, 28, 32). Slots 23, 24, 26, 27 and 30 were also patch_fail. Large rewrites and new variables caused most of these.

Next: make small SEARCH/REPLACE edits that match the file exactly. Vary the jitter size, grid shape and restart count on the jittered grid. Try a minimal analytic constraint Jacobian. Replicate any gain before accepting it.
