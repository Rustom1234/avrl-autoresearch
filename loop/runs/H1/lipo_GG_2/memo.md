Incumbent is now slot 35 at 0.5544. It swapped the RF for LightGBM on the descriptor features, a change of -0.0424. Slot 21 added descriptors (MolWt, LogP, donors, acceptors) and gave -0.1263, the biggest gain. Together these two are the only clear wins.

Neutral or noise: RF tweaks (N_TREES 1000 at 0.7224, MAX_FEATURES 0.5 at 0.7212, n_jobs 8 at 0.7236).

Hurt: max_depth limits (10 at 0.8196, 15 at 0.6389), N_BITS 1024 (0.7498), RADIUS 3 (0.7534), MAX_FEATURES 0.2 (0.6029), a bagging ensemble (0.6173), and extra descriptors TPSA, MolMR and rotatable bonds (0.5985). Slot 28 shows that adding test molecules to training is a bug.

Many slots were lost to patch failures and crashes, so keep edits small and exact.

Next: tune LightGBM (learning rate, leaves, regularization, trees). Then try an LGBM+RF or CatBoost blend. Then try a few descriptors added one at a time, such as Crippen contributions or ring counts.
