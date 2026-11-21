Helped: the rank-average ensemble of RF, LightGBM and Tanimoto SVR (slot 0, 0.5505, -0.0598); RF on RDKit descriptors plus Morgan counts (slot 3, 0.5272, -0.0232); rank-transforming train_y for all models (slot 6, 0.5027, -0.0246, incumbent). Gains came from descriptors, the target transform and ensemble diversity.

Did not help beyond noise (sd about 0.0070): RF and feature tweaks (bit filtering, atom-pair/torsion, ExtraTrees, bigger RF, EState, kernel PCA, descriptors only at 0.5010), bagged huber LGBM, stacking weights (0.5015), multi-radius kernels (0.5023). Fourth members XGBoost (0.4960), kernel ridge (0.5038) and descriptor LGBM (+0.0017) failed. CatBoost (0.4895/0.5051) and MLP (0.4899, then a crash) are noisy and unresolved.

Next: stop RF and feature tweaks. Rerun the MLP once, since 0.4899 was the best score seen. Retry CatBoost once. Try descriptor-based kernels and simple weight tuning. Accept only gains above about 0.01 or confirmed by replicates.
