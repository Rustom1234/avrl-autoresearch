Helped: adding chemical information and structure to the features and ensembling. MACCS keys added to Morgan gave 0.5334 (-0.0278). Atom pairs on top gave 0.5172 and 0.5205 (-0.0145, replicated), and this is the incumbent. The RF+XGB+LGBM averaging ensemble (-0.0315) and the log1p target transform (-0.0176) also helped. All of these gains are well above the noise sd of about 0.0070.

Did not help: tuning and regularization. This covers 500 trees, min_samples_split, early stopping, MAX_FEATURES 0.1, ensemble weights, CatBoost, Morgan radius 3, the manual descriptor list, standard scaling of the target (+0.0322), and the rank target transform (slot 36, 0.5242, +0.0054). ExtraTrees (-0.0024), MAX_FEATURES 0.2 (-0.0035) and 3 more trees-count tweaks (-0.0004) were within noise.

Many jobs crashed or failed to patch (RDKit API names, SEARCH text mismatches), so check both before submitting.

Next: other fingerprint types (topological torsion, count-based Morgan via GetHashedMorganFingerprint, path fingerprints). Use the RDKit API version that works for atom pairs. Avoid more hyperparameter tuning.
