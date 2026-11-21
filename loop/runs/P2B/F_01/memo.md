Incumbent: mean_adjusted 1.0531 over 2 runs (slot 7, TOTAL_BATCH_SIZE 2^17, official 1.0516/1.0546, delta -0.0614, tripwire flagged). Batch size is the only large win: more optimizer steps in 300 s. 2^18 was worse (1.0951, +0.0420), so the optimum is at or below 2^17.

Did not help: cosine warmdown (+0.0567), HEAD_DIM 64 (+0.0119), DEPTH 6 (+0.0055), DEPTH 10 (+0.0051), warmdown 0.7 (+0.0016), MATRIX_LR 0.05 (+0.0002), warmup 0.03 (+0.0011), embedding LR 0.4 (+0.0013), beta1 0.9 (+0.0007). Structural changes are mostly negative.

Small gains: scalar LR 0.25 (-0.0036), Muon momentum ramp (-0.0018). Both are rejected single runs, so they may be noise. They are worth replicating and combining.

Next: check the tripwire, then try batch 2^16. Re-test scalar LR 0.25, the momentum ramp, wte init 0.5 and SSL on the new base. Three slots aborted, so check that edits apply.
