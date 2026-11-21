Helped: physchem features (Crippen logP plus RDKit 2D descriptors) added to Morgan bits were the biggest gain, 0.6030 (-0.1200). Swapping RF for CatBoost with MAE loss gave the incumbent, 0.5718 (-0.0184). The kNN residual fix reached 0.5902 (-0.0127) but still needs an ablation.

Did not help: rare-bit filtering (0.7231), Tanimoto SVR (0.6491), ExtraTrees (0.6140), count Morgan (0.5930), Atom Pair bits (0.5862), full descList (0.7202), stacking (2.2158), 3D shape descriptors (0.5718, +0.0000), stronger regularization (0.5756). Small gains within noise sd 0.0042: 3-seed bagging (0.5668/0.5689), depth ensemble (0.5682), radius 3 with 4096 bits (0.5671/0.5703), and inner-holdout early stopping (0.5640/0.5643, -0.0076). Early stopping is the best lead.

Waste: 5 crashes and 4 patch failures early on. Copy SEARCH text exactly.

Next: combine the small gains (early stopping, bagging, radius 3), then try MACCS keys and a CatBoost+LightGBM blend. Replicate any combined result. Watch the 10 min limit.
