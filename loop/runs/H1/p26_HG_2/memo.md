Helped: only slot 5, a dimension-mismatch bug fix in _solve_once (-2.6299/-2.6328, mean -2.6313 over 2 runs, the incumbent). No new search idea has beaten it yet.

Did not help: L-BFGS-B penalty (slot 3, -2.6092) and Halton init (slot 17, -2.6226) were clearly worse. Remove-and-reinsert (slot 9, -2.6324), elite pool (slot 10, -2.6303), more restart time (slot 15, -2.6299), batch perturbation (slot 16, -2.6294), successive halving (slot 22, -2.6293), hex init (slot 23, -2.6300) and equal-radius warm start (slot 31, -2.6315) were within noise (sd about 0.0023), so none is proven.

Process: many slots were lost to patch failures (10 of the 21 slots listed in the previous notes) and to crashes. Crashes clustered around analytic Jacobians, LP repair and custom initialisations.

Next: copy SEARCH text exactly and keep edits small. Check shapes and syntax before submitting. Replicate the incumbent once to pin down its true mean. Then try basin-hopping from the best packing, with a gentler repair step.
