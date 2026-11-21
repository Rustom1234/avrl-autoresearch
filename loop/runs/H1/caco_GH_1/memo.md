Helped: adding RDKit physchem descriptors to Morgan bits (0.4417 to 0.3911, far above the 0.0037 noise). Taking the median of per-tree RF predictions also helped (0.3861, then 0.3794 on replication). The incumbent mean_adjusted is 0.3828.

Did not help: model swaps (XGBoost, CatBoost, LightGBM, ExtraTrees, stacking, GBR), RF tuning (min leaf 2: 0.3929; max_features 0.1: 0.3889; 1000 trees: 0.3865; max depth cap: 0.3898), Morgan radius 3 (0.3972), descriptor interactions (0.3965), extra count features (0.3928), VarianceThreshold (0.3875) and an RF+median blend (0.3850).

Borderline: MACCS keys (0.3792/0.3779 and 0.3792/0.3771) and Morgan radius 1 (0.3822). These are small gains that sit near the noise. Their combination is worth a try, such as radius 1 plus MACCS.

Next: features. Try the full RDKit descriptor set, count fingerprints, fewer Morgan bits, and a target transform. Avoid more model-family swaps.
