Only one change helped: slot 33 (0.3882, -0.0535 vs the 0.4417 start). It fixed the broken RDKit descriptor edit by removing the non-existent Descriptors.NumRings attribute. So the descriptors appended in the earlier crashed slots now run, and they carry real signal. The table doesn't say which descriptors slot 33 kept. Check the current train.py before editing.

Did not help: forest tweaks (MAX_FEATURES 0.1 gave 0.4521, MIN_SAMPLES_LEAF 2 gave 0.4542, N_TREES 1000 gave 0.4531), radius 3 (0.4637), salt removal (0.4531), and XGBoost (0.4890, 0.4762). LightGBM gave 0.4046 in slot 34, which is worse than 0.3882 but close. Bit density (0.4400) was neutral.

Next:
1. Extend the descriptor set, ideally the full RDKit 2D descriptors.
2. Blend RF with LightGBM on descriptors.
3. Try a target transform and count-based Morgan features.
4. Use small, exact SEARCH blocks with explicit imports. Many slots crashed or failed to patch.
