Helped: only search-strategy changes. Slot 4 (basin hopping from jittered copies of the best packing after 40% of the time) gave -0.0050 over 2 runs. Slot 7 (cheap common-radius SLSQP init for fresh starts) gave -0.0063 over 2 runs. The incumbent is -2.9366 over 2 runs, about 0.0004 below the published 2.9370, so little headroom is left.

Did not help: constraint reformulation (+0.0140), analytic Jacobians (+0.0007), final polish (+0.0049), mirror starts (+0.0014), L-BFGS-B penalty init (+0.0056), per-circle random radii (+0.0081), log-barrier init (+0.0112). Most later single-run tweaks sit within noise (sd about 0.0034). Many slots scored exactly -2.9374, so the same optimum seems to be found each time.

Next: replicate the incumbent once. Replicate slots 14, 25 and 10 only if budget allows. Don't repeat tried ideas. Further gains are probably below the noise, so consider stopping.
