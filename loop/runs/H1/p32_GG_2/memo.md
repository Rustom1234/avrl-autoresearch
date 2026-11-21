Nothing beat the incumbent (-2.9253); noise sd is about 0.0034, so deltas under ~0.003 are noise.

Did not help: repair tweaks, log-uniform radii, MAX_ITER 1000 and time-weighted MAX_ITER all landed at +0.0007 to +0.0009, which is no gain. Hex-grid init (+0.0174) was worse. The smooth penalty (-1.5356), the squared-distance constraint (-2.1905) and the sum-of-squares objective (-2.8100) were much worse.

Only slot 6 (Sobol-like centre init) hinted at a gain: -2.9311 and -2.9260, delta -0.0033, about one sd. It is the only lead. Replicate it first.

Many slots were lost to crashes and patch failures (family counts: architecture a=1 b=7, optimizer a=1 b=8). Make small edits and copy SEARCH text exactly.

Next: keep the hard constraints, and don't rewrite the repair step. Add Sobol init, then perturb the best packing and re-optimize it. Also try longer SLSQP runs from a few good starts.
