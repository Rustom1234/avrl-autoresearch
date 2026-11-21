Helped: structural changes to the residual setup. XGBoost L1 replacing the random forest gave 0.3828, down from 0.4417. A physchem ridge base with XGBoost on its residuals gave 0.3755/0.3715. Two-stage XGBoost (squared error, then L1) gave 0.3658/0.3652, the incumbent.

Did not help, or was within noise (sd about 0.0037): other model families (LightGBM 0.3830, CatBoost 0.3860, GP 0.3859, MLP 0.4962, kernel methods), ensembles and bagging (0.3793, 0.3795), extra fingerprints (Avalon+ErG 0.3710, atom-pair/torsion 0.3689), 3D shape 0.3760, calibration 0.3745, ridge-base variants (polynomial 0.3940, sparse L1 0.3882, spline 0.3763), heavier regularization 0.3725, monotone constraints 0.3697, winsorizing 0.3727, NGBoost-style 0.3733.

Small gains near noise: collision-free Morgan radius 0-3 (0.3625), feature_weights (0.3642), applicability-domain features (0.3646), multi-radius Morgan (0.3646).

Next: combine those four on the two-stage fit and replicate before accepting. Vary stage round counts and learning rate. Avoid specs that abort at the engine.
