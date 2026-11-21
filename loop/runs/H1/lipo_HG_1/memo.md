Helped: changing the model family and kernel. The SVR on a Tanimoto kernel (slot 7) scored 0.6489, down 0.0742 from baseline. Averaging Tanimoto kernels over several fingerprints helped again. Slot 20 scored 0.6042. Slot 24 scored 0.5925/0.5961 (two runs). The incumbent, slot 31, averages Morgan and atom-pair kernels and scored 0.5683, down 0.0259.

Did not help: kNN blend (+0.0021), count fingerprints (+0.0000), sample weights (+0.0002), an SVR grid ensemble (+0.0060) and dropping rare/common bits (0.5680, within noise). RF/LightGBM blends and recalibration were clearly worse.

Waste: many patch failures and crashes, mostly from descriptor imports and nested search.

Next: add more diverse fingerprint kernels, such as other radii, topological torsion or MACCS, to the averaged kernel. Try kernel ridge or a GP on the same kernel. Make each SEARCH block short and exact.
