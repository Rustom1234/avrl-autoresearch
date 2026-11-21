Helped: only slot 1, running SLSQP first on a randomly stretched (elliptical-metric) problem. It scored -2.6323 and -2.6303, mean -2.6313 over 2 runs, against a baseline of about -2.6280. The gain of about 0.0033 is barely above the noise sd of 0.0023, so it is not firmly confirmed. Slot 22 (three successive stretches) scored -2.6322, no better.

Did not help: penalty methods (slot 0 -2.5921, slot 3 -2.6252) and force-directed relaxation (slot 15 -2.6238). Edge-biased starts and greedy placement both scored -2.6287. Many small tweaks scored -2.6308 to -2.6338 (safety margin, early exit, warm-start, annealing, analytic Jacobian, jitter, smallest-circle removal, shake, screening). All are within noise of the incumbent.

Next: replicate the incumbent to confirm it. Tune the stretch distribution and the share of time spent on stretched starts. Keep SLSQP, avoid penalty and relaxation preprocessing, avoid duplicating aborted ideas, and prefer simple changes.
