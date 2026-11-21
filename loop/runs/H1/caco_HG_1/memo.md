Incumbent is 0.3679 (mean_adjusted, 2 runs). It uses the rank-preserving normal-scores target transform (slot 35: 0.3698 and 0.3661, -0.0084 vs the previous incumbent). It sits on atom-pair count fingerprints with CatBoost (slot 21: 0.3763).

Helped: count-based features (slot 13: 0.3984), atom-pair counts (-0.0220), the target transform, and the Tanimoto-kernel SVR over the RF (0.4366 and 0.4257).

Did not help: CatBoost regularization (+0.0058), Langevin boosting (+0.0032), early stopping (+0.0015), LightGBM l1 (+0.0048), the CatBoost+RF blend (-0.0010), the median of per-tree predictions (+0.0028) and invariant Morgan (+0.0000). Doubling CatBoost iterations (-0.0022) is within noise (sd 0.0037).

Most failures were crashes from API errors (Descriptors import, fingerprint calls) or patch failures. Keep each SEARCH short and exact.

Next: a Tanimoto SVR or CatBoost blend on atom-pair counts with the transform. Also try atom-pair plus Morgan counts, and descriptors with correct imports and NaN handling. Tune SVR C and epsilon.
