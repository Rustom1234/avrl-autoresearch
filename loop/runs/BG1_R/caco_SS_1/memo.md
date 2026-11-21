Incumbent is slot 17: L1 LightGBM on RDKit 2D descriptors blended with kNN, mean_adjusted 0.3694 over 2 runs (0.3725/0.3664).

Helped: data/feature changes (descriptors + Morgan counts 0.3781; kNN blend 0.3976) and the switch to L1 LightGBM on descriptors (-0.0086).

Did not help: family swaps (SVR 0.4222, CatBoost 0.4102, RF 0.3953), feature additions (MACCS+atom-pair 0.3836, Crippen/EState 0.3728, EState fingerprint 0.3716, MACCS 0.3703), pruning (0.3797), winsorizing (0.3792), bagging/subsampled LightGBM (0.3709, 0.3764, 0.3923), stronger regularization (0.3761), min-max kNN (0.3793), kernel ridge (0.3740), DART (4.0173). XGBoost runs failed (94.3105).

Small blends (SVR 0.3693, atom-pair 0.3686, Huber 0.3691) are within the noise sd of about 0.0037.

Next: gains are near the noise floor. Try only small LightGBM tweaks (leaves, learning rate) and small kNN weight/k steps. Replicate any candidate before accepting. Avoid more family swaps, feature additions and pruning.
