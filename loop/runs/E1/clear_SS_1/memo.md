Helped: rank-blending diverse families (RF + Tanimoto SVR, 0.5407 from the 0.6103 baseline), then XGBoost on descriptors plus Morgan counts (0.5195), a rank-transformed target (0.4994), and Avalon count + ErG features in XGB (0.4718, the incumbent). The incumbent is a single run with noise sd ~0.0070, so it is unconfirmed.

Did not help, or stayed within noise: kNN, ExtraTrees, KernelRidge, GP, LightGBM, Ridge, MLP, MACCS, CatBoost swaps, seed-bagged XGB, feature selection, XGB retuning, stacking, blend weights, target smoothing and sample weights. Most results landed within about ±0.006 of the incumbent.

Near-noise gains: ECFP4/ECFP6/FCFP4 averaged kernel 0.4649, pairwise-difference LightGBM 0.4653, FCFP4 kernel 0.4637/0.4771, ECFP6 kernel 0.4658, CatBoost third member 0.4678.

Next: replicate 0.4718 first. Then combine the multi-radius kernel (0.4649) with the pairwise LightGBM (0.4653) as a third member. Check the abort list first, since many ideas are repeats.
