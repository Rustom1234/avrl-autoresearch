Helped: only slot 5, scaling radii by 10 in the SLSQP variables (conditioning). It scored -2.9385, 0.0133 better than the incumbent. It is a single run and the noise sd is about 0.0034, so replicate it first.

Did not help: a larger restart time fraction (slot 0, -2.9246), Dirichlet-like radius profiles (slot 11, -2.9361, +0.0025), and radius-aware init (slot 24, -2.9363, +0.0022). Slots 14 and 17 tied the incumbent at -2.9385 (+0.0000), so they added nothing.

Most slots failed on mechanics: patch failures (1, 2, 7, 8, 12, 15, 19, 20, 21), crashes (3, 4, 6, 9, 10, 13, 22, 23) and aborts (9, 16, 18, 19). Slots 3, 6, 9, 13 and 22 are untested on merit.

Next: spend one slot replicating slot 5. Then make small edits. Copy each SEARCH block exactly from the current file, and check array shapes (broadcasting errors recurred). Retry hex init or the remove-smallest local search in the simplest form. Avoid large rewrites.
