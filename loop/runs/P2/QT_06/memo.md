Nothing has beaten the incumbent (mean_adjusted 1.1146 over 16 runs). No slot has been accepted.

Schedule changes did not help. WARMUP_RATIO 0.1 gave 1.1237 and 0.3 gave 1.1286. WARMDOWN_RATIO 0.7 gave 1.1157 and 0.3 gave 1.1165. The "marriage" schedule gave 1.1250. Stop tuning the schedule.

Size and batch changes hurt. DEPTH 4 with ASPECT_RATIO 256 gave 1.1713. Halving the batch to 32 gave 1.2050. ASPECT_RATIO 128 crashed. MATRIX_LR 0.08 gave 1.1183, and a matrix-only warmup gave 1.5416.

The only hints of gain are within noise. UNEMBEDDING_LR 0.01 scored 1.1100 and 1.1180, a 0.0080 spread. Weight decay 0.5 gave 1.1111. UNEMBEDDING_LR 0.001 (1.1527) and weight decay 0.05 (1.1188) were worse.

Next: combine UNEMBEDDING_LR 0.01 with weight decay 0.5, and replicate it. Propose only parameters that exist in train.py. Many slots were aborted for ideas with no referent. Avoid target-leaking losses.
