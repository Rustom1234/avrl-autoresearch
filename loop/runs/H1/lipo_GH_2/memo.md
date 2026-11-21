Helped: richer features and model family. RDKit descriptors replaced Morgan bits (0.7230 to 0.5977), RandomForest to XGBoost (0.5403), and adding Morgan bits to the descriptors (0.5249, the incumbent). Nothing since has beaten it by more than noise (sd 0.0042).

Not helped: LightGBM (-0.0012), min_child_weight 5 (-0.0003), MACCS keys (+0.0002), target standardization (-0.0002), Morgan counts (-0.0004), atom-pair fingerprints (-0.0005), max_depth 4 (-0.0002), 3000 trees at lower learning rate (-0.0002). Worse: early stopping (+0.0099), CatBoost (+0.0079 and +0.0048), max_depth 8 (+0.0143), Morgan radius 3 (+0.0090), colsample 0.3 (+0.0069).

Unresolved: the 3-seed average scored 0.5198/0.5223 and Morgan radius 1 scored 0.5152/0.5258. Both replicates disagree, so they are noise-level. 4096 bits scored 0.5220.

Next: replicate the incumbent to pin down its true loss. Then try the seed ensemble combined with radius 1 or 4096 bits, or XGBoost plus CatBoost averaging. Many slots were wasted on aborts: check that the model and every constant exist in train.py before proposing an edit.
