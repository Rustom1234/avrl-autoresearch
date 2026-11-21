Helped: feature and RF-regularization changes. Count-based Morgan gave 0.1150 (-0.0211 vs the 0.1361 baseline). Adding MACCS gave 0.1087/0.1030. The balanced-subsample RF with 1000 trees and sqrt max_features gave 0.0950/0.0914, the best so far (incumbent mean_adjusted 0.0932).

Did not help: model swaps, ensembles and feature pruning. LightGBM alone gave 0.0997, stacking 0.0947, bit filtering 0.0976, top-k bit selection 0.0959, multi-radius counts 0.0941, max_features 0.1 0.0956. Atom Pair/Torsion (0.0916), multi-view (0.0918) and RF+LightGBM rank average (0.0897) were all within noise (sd about 0.0049).

Many slots crashed or failed to patch: standardization, descriptors, Tanimoto kernel, SMARTS groups, min_samples_leaf. Those ideas are untested, not disproved.

Next: (1) retry the descriptor block (MW, TPSA, LogP, H-bond counts) with small, exact edits and checked API calls. (2) Replicate the RF+LightGBM blend. (3) Retry min_samples_leaf. (4) Combine counts, MACCS and descriptors with the balanced RF.
