Helped: richer features and model diversity. RDKit descriptors plus Morgan counts in the RF cut loss from ~0.6103 to 0.5538 (slot 0). Rank-averaging RF with LightGBM gave 0.5292 (slot 4). Adding an RBF SVR on normal-quantile-transformed descriptors gave the incumbent 0.5057 (slot 15, one run only). That is the biggest recent gain, -0.0235.

Did not help: XGBoost on descriptors plus MACCS (0.5519), RF max_features 'sqrt' with 600 trees (0.5656), MACCS in the RF (0.5612), RF on descriptors alone (0.5543). Within noise (sd ~0.0070): CatBoost added (0.5059), stacking (0.5237), stronger LightGBM regularization (0.5258), FCFP counts (0.5215/0.5185), Tanimoto kernel ridge (0.5305).

Next: replicate the 0.5057 incumbent first, since it has one run. Then try quantile-transformed inputs for other members, SVR tuning (C, epsilon), and kernel ridge or FCFP counts as extra members. Treat gains under ~0.01 as noise. Avoid MACCS and RF-only tweaks.
