Helped: only slot 3, a pure overhead fix (the Muon helper was recompiled every step because the Python int step count was specialised). It scored -0.0396 and -0.0814 over 2 runs, mean_adjusted -0.0543, and is the incumbent. Slot 14 (24x24 first third) scored -0.1194/-0.0779 official, but only -0.0382 vs the incumbent, so it was rejected. It is worth a replicate.

Did not help: model-shrinking changes (block2 192: +0.1186; widths 48/256/320: +0.0594), fewer Muon NS steps (+0.3952), 2-view TTA (+0.1382), constant norm_freq 8 (+0.0335), shorter whitening-bias schedule (+0.0598). Small tweaks (GELU, no color jitter, BN momentum, 480 whitening images, no-cudagraphs, skipping the epoch-0 flip) were all within noise (sd about 0.026) or worse. Slot 2 (28x28) crashed.

Next: hunt for more overhead (graph breaks, other recompiles, host-device syncs, loader and predict cost). Replicate the 24x24 idea. Avoid epoch tweaks and accuracy-affecting model changes.
