Helped: changes to how each solve searches. Slot 8 (equal-radius SLSQP before the free solve) gave -2.6311/-2.6311. Slot 35 (local disc re-randomisation kick walk) is the incumbent at -2.6343/-2.6354 (mean -2.6349, 2 runs). Walks from a good packing beat fresh restarts (slot 11 -2.6354/-2.6334, slot 27 -2.6341/-2.6324).

Did not help: start-diversity tweaks, chain machinery (replica exchange -2.6348, two-island -2.6349, both only tie), and every kick variant (log-uniform +0.0035, record-to-record +0.0022, soft-mode +0.0089, homothety +0.0110, racing -0.0006 in one run, within noise). Polish steps (trust-constr +0.0006) and TIME_FRACTION 0.93 gave about +0.0000. Hurt: radius-weighted restarts +0.0165, log-barrier +0.0045, L-BFGS-B penalty +0.0034, MAX_ITER 1000 +0.0019.

Next: stop varying the kick and polish. Try a cheaper SLSQP per step (lower MAX_ITER or fewer constraints) so each walk gets more steps. Replicate any gain, since noise sd is 0.0023. Keep the code simple.
