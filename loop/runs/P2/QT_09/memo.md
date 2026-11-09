Incumbent mean_adjusted 1.1146 over 16 runs. Nothing has clearly beaten it. Only three small wins appeared, all single runs: FINAL_LR_FRAC 0.01 (1.1127, -0.0018), removing AdamW bias correction (1.1127, -0.0018), and warmdown ratio 0.3 (1.1143, -0.0003). These gaps are probably within noise, so replicate before trusting them.

Harmful changes: removing warmup/warmdown (+0.0891), warmup 0.2 (+0.0109), symmetric 0.5/0.5 (+0.0248), warmup 0.01 with warmdown 0.8 (+0.0230), beta1 0.9 (+0.0070), unembedding LR 0.04 (+0.0088), weight decay 0.01 (+0.0044). Schedule has been explored heavily, with 7 rejections.

Invalid slots: 3 patch failures and 7 model failures, mostly architecture edits. Check SEARCH matches and model shape constraints carefully.

Next: combine the two small wins (FINAL_LR_FRAC 0.01 plus no bias correction) and replicate. Then try modest LR tuning on matrix or embedding params, and batch size or step-count tradeoffs. Avoid more schedule and weight-decay sweeps.
