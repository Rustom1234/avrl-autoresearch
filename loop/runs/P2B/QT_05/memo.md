Incumbent: mean_adjusted 1.1145 over 16 runs. Nothing has clearly beaten it; most deltas are single runs and may be noise.

Helped slightly: 3x FFN width (-0.0041, two runs: 1.1090/1.1118), WEIGHT_DECAY 0.25 (-0.0033), SCALAR_LR 0.01 (-0.0012). MATRIX_LR 0.08 tied (-0.0001).

Hurt: warmup (0.05: +0.0108, 0.10: +0.0096), sublinear warmup (+0.0099), TOTAL_BATCH_SIZE 2^20 (+0.0661), DEVICE_BATCH_SIZE 32 (+0.0902), low LR for the last 50% (+0.1172), two-stage LR (+0.0055), STAGE1_LR (+0.0038), WEIGHT_DECAY 0.05 (+0.0018), GQA (+0.0010), variable MLP width (+0.0081). DEPTH 12 crashed.

Family record: schedule 1 win/7 losses, architecture 1/5.

Many slots aborted with vague specs, so give concrete edits.

Next: combine 3x FFN, weight decay 0.25, SCALAR_LR 0.01 and replicate. Try TOTAL_BATCH_SIZE 2^18 and WARMUP_RATIO 0.0. Avoid more schedule and warmup changes.
