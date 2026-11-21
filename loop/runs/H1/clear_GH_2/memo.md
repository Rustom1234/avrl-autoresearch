Incumbent: slot 17, RF+ExtraTrees average on Morgan 2048 + RDKit Properties descriptors, log1p target, max_features 'sqrt'. Loss 0.5146/0.5169 over 2 runs (baseline 0.6103; noise sd about 0.0070).

Helped: log1p target (-0.0369), 'sqrt' (-0.0237), Properties descriptors (-0.0182, replicated), RF+ET averaging (-0.0156, replicated).

Did not help: GBM swaps (LightGBM 0.5475 to 0.5683, CatBoost 0.5839 and 0.5966, XGBoost 0.5891 and 0.5972), a small LightGBM added to the ensemble (+0.0240), radius 3 (+0.0174), min_samples_leaf 5 (+0.0273), atom pairs (+0.0075), median blend (+0.0096), and top-100 feature selection (+0.0193). Target tweaks were within noise or worse.

Near misses, all within noise: radius-1 Morgan added (-0.0064), 10 top descriptors (-0.0056), 1000 trees (-0.0026), MACCS (-0.0023), weighted blend (-0.0007).

Next: combine the near misses (radius-1 bits, MACCS, 1000 trees) in one run and replicate it. Also try separate max_features for RF and ET. Skip more target transforms, GBMs and leaf-shrinking.
