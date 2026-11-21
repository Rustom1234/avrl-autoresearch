Incumbent is slot 46 (Ridge prior on 6 physchem descriptors, L1 LightGBM on RDKit descriptors, kNN blend). Its mean_adjusted is 0.3618 over 2 runs (0.3646/0.3591).

Helped: descriptors over fingerprints (0.4101, then 0.3781 with Morgan counts), the kNN blend (0.3976), L1 LightGBM on descriptors (0.3725/0.3664), and the Ridge prior (0.3646/0.3591).

Did not help: family swaps (SVR 0.4222, CatBoost 0.4102, RF 0.3953), extra features (MACCS 0.3703, EState 0.3716, SMARTS 0.3659), pruning (0.3797), bagging (0.3709, 0.3764, 0.3923), kernel methods (0.3740, 0.3832), the MLP blend (0.3626), LR schedules (0.3657, 0.3624) and DART (4.0173). XGBoost scored 94.3105, so that run was broken.

Close calls, within noise sd 0.0037: slot 76 (0.3579/0.3548), slot 57 (0.3585, replicate 0.3651), slot 52 (0.3610) and slot 55 (0.3606). Slot 76 was rejected anyway.

Next: combine slots 52 and 55 on the incumbent and replicate once. Check the table first, because many ideas were aborted as repeats. Gains are now near the noise level.
