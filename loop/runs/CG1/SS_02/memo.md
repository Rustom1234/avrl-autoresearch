Helped: only slot 2, training the first 40% of steps on 24x24 downsampled inputs (official -0.1451 and -0.1278; incumbent mean -0.1364 over 2 runs). It carries a tripwire flag, so audit or replicate it before building on it. A fixed Muon rescaling interval (-0.1429, -0.0064 vs incumbent) is within noise, though it may simplify the code.

Did not help: the 16x16 three-stage ramp (+0.0250), the milder 28x28 phase held for 50% of steps (-0.0053 official), narrower block1 (+0.0921 vs incumbent), wider block3 (+0.0737 vs incumbent), LR warmup (+0.0515), weight decay 3e-06 (+0.0526), default compile mode (+0.0427), TRANSLATE 1 (+0.0360), logit scale 1.5 (+0.0366), no whitening bias training, and TTA with only two views. Small tweaks mostly failed.

Next: audit slot 2 first. Then tune the low-res fraction (30% or 50%) and the resolution (20) around 24. Avoid very low resolution and further tiny tweaks.
