No change has beaten the incumbent (mean_adjusted 1.1145 over 16 runs). Of the 13 slots listed, 4 crashed (1, 5, 8, plus slot 9 patch_fail, invalid) and the rest were rejected.

Helped slightly, but rejected: attention-span changes. Slot 10 (curriculum window, 1.1102/1.1073, -0.0058) and slot 12 (depth-aware window, 1.1093/1.1088, -0.0054) were the best, and both were replicated. Slot 7 (-0.0028), slot 2 (-0.0024) and slot 3 (-0.0013) were smaller gains that are likely noise.

Hurt: schedule changes (slot 0 +0.0228, slot 4 +0.0116), the roll fix (slot 6 +0.0071). Slot 14 (embedding LR multiplier) was flat (+0.0002). Complex loss terms and prompting crashed.

Next: combine the curriculum window (slot 10) with the depth-aware window (slot 12) in one simple edit. Then try small hyperparameter changes: weight decay, batch size, model size. Avoid custom losses and schedules.
