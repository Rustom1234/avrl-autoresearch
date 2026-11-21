Helped: search-strategy changes only. Slot 4 (basin-hopping from jittered copies of the best packing after 40% of the time) gave -0.0050 over 2 runs. Slot 7 (cheap common-radius SLSQP init for fresh starts) gave -0.0063 over 2 runs. The incumbent is -2.9366, about 0.0004 below the published 2.9370, so little headroom remains.

Did not help: constraint reformulation (slot 0, +0.0140), analytic Jacobians (+0.0007), final polish (+0.0049), mirror starts (+0.0014), L-BFGS-B penalty init (+0.0056), per-circle random radii (+0.0081). Nearly all later single-run tweaks sit within noise (sd about 0.0034). The best were slots 14 and 25 (-0.0030 each) and slot 10 (-0.0020). Several were aborted as duplicates.

Next: replicate the incumbent first. Then replicate slots 14, 25 and 10 and try combining them. Avoid repeating tried ideas. Keep the code simple. Further gains are likely below the noise level, so consider stopping.
