Helped: only the optimizer swap in slot 10 (SLSQP to L-BFGS-B with penalty continuation). It scored -2.6315 and -2.6327, delta -0.0041. The incumbent is -2.6321 over 2 runs, and about 0.004 remains to the published 2.6360.

Did not help: repair-step variants, which were mostly aborted. Also MAX_ITER 500, early stopping, annealing, the quadratic penalty, Sobol starts, penalty weight 1e6, Pareto radii, radius perturbation, OVERLAP_W=3, final polish, basin hopping, adaptive penalty schedules. Most sit within noise (sd about 0.0023) of the incumbent. Several early slots show an identical -2.6261, so check that those edits really change the output.

Promising: structured starts under L-BFGS-B. Jittered hex scored -2.6340, and a second hex variant scored -2.6359 and -2.6321 (delta -0.0019). Greedy centres scored -2.6344 and -2.6354 (delta -0.0028, 2 runs). Jittered grid scored -2.6338.

Next: adopt the greedy-centre start, which is best and replicated. Then combine it with hex or grid starts, and tune the penalty schedule. Replicate before accepting anything.
