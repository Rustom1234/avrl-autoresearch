Helped: curated 10-feature descriptors (0.1243 vs 0.1361 start). Ensembling three models on different feature sets gave the biggest gain (0.1002). XGBoost in that ensemble gave 0.0934 and 0.0879 (incumbent mean 0.0906).

Did not help: single-model swaps (CatBoost 0.1264, Tanimoto SVC 0.1255, logistic regression 0.1264, LightGBM 0.0979). Also no help: one RF on combined features (0.1011), strong regularization (0.1058), scaffold bagging (0.0995), scaffold weighting (0.0977), self-training (0.0920), count Morgan (0.0912), feature filtering (0.0940), auxiliary regressors (0.0942), naive Bayes (0.0947), GP (0.0926).

Fourth-member additions sit near noise (sd about 0.0049). Best singles: ExtraTrees 0.0866, random-subspace XGBoost 0.0873, Gobbi 0.0880, ErG 0.0884. None replicated.

Next: replicate ExtraTrees and random-subspace first. Then try them together. Accept only if the replicate is below 0.088. Stop small tweaks and new featurizers.
