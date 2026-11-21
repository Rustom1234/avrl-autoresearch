Helped: the rank-average ensemble of RF, LightGBM and CatBoost (0.1361 to 0.0942). Molecule standardization (0.0899, 0.0904) also helped. So did the non-tree members, a Tanimoto kNN and a quantile-scaled MLP. Slot 39 combined these and is the incumbent (0.0859/0.0813, mean_adjusted 0.0836).

Did not help, or was within noise (sd about 0.0049): extra fingerprints, MACCS/EState, stacking, ExtraTrees/SVC/HGB members, tweaks to RF/CatBoost/LightGBM, kernel ridge, pseudo-labeling, scaffold weighting, class balancing, and multi-scale Morgan. Clear harm: physchem-only (0.1323), descriptors-only RF (0.1094), single CatBoost (0.1021), logistic blend (0.1079), XGBRanker (0.1034).

Best recent: noise-augmented MLPs, 0.0832 (slot 51), 0.0825 (slot 65) and 0.0822 (slot 72). Each is only 0.0004 to 0.0013 better than the incumbent, so all are inside the noise. Their consistency may still be real.

Next: replicate slot 72 or 65 once, and adopt it if it holds. Then try kNN k and the weights between tree and non-tree members. Avoid new fingerprints, stacking and tree members. Gains look saturated.
