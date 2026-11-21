Helped: two changes. Slot 6 (DEPTH 6, ASPECT_RATIO 85, width 512, 4 heads of 128) scored 1.1031, -0.0114 vs the old incumbent. Slot 10 (total batch 2^17) scored 1.0510/1.0528 over 2 runs and is the incumbent (mean 1.0519); it carried a tripwire flag. Both give more optimizer steps in the fixed 300 s.

Did not help: wider or deeper models (ASPECT_RATIO 96 +0.0171, DEPTH 10 +0.0203), HEAD_DIM 64 (+0.0125), value embedding on every layer (+0.0028), batch 2^18 (+0.0150), warmdown 0.3 (+0.0028), max-autotune compile (-0.0009, rejected), SCALAR_LR 0.25 (-0.0009, rejected). Several nudges (MATRIX_LR 0.05, weight decay 0.1, warmdown 0.7, SwiGLU, warmup 0.02) were judged against the old incumbent, so their deltas are stale.

Next: try an even smaller batch (2^16) and DEPTH 5 at width 512. Retest weight decay 0.1 and MATRIX_LR 0.05 on the new base. Replicate anything within about 0.001. Check the tripwire flag. Two patch failures (Muon momentum warmup, EMA) need exact SEARCH text.
