Incumbent is now slot 17, progressive resolution (first 40% of steps downsampled): official -0.1896/-0.1322, mean adjusted -0.1547 over 2 runs. It was flagged "tripwire", so audit or replicate it before building on it. The earlier "no change helped" note is outdated.

Helped: data-side speed changes (progressive resolution).

Neutral within noise (sd about 0.026): colour jitter off (-0.0058 official), extra TTA views (+0.0072 delta), softmax-averaged TTA (-0.0092 delta, rejected).

Hurt: larger batch (+0.0571), narrower middle block (+0.1090), pad 4 (+0.0945), floored LR decay (+0.3154), head on Muon (+4.6760), BN recompute (+0.0358), per-channel jitter (+0.0490), per-sample flip (+0.0814).

Crashes: slots 2, 3, 13, 14, all architecture or BN/head edits. Avoid them.

Next: resolve the tripwire first. Then tune progressive resolution (fraction of steps, start resolution), and try fewer TTA views to save time. Keep slots in reserve for replicates.
