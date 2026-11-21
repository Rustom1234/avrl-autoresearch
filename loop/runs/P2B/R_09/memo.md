Incumbent is now mean_adjusted 1.0821 over 2 runs, set by slot 12 (DEPTH 6, accepted, -0.0324; flagged tripwire, so treat with some caution). The earlier notes were stale.

Helped: a smaller, faster model (DEPTH 6) was the one big win, probably because it fits more steps in 300 s. Small gains were UNEMBEDDING_LR 0.008 (-0.0036, 2 runs, rejected) and MATRIX_LR 0.05 (-0.0004, 1 run). Earlier, SwiGLU gave -0.0036 and weight decay 0.1 gave -0.0011, but both were measured against the old incumbent.

Did not help: larger or wider models (depth 10 +0.0498, ASPECT_RATIO 80 +0.0256), HEAD_DIM 64 (+0.0135), warmup 0.03 (+0.0118), and most LR or schedule tweaks.

Next: 
1. Go smaller still, e.g. DEPTH 5 or 4.
2. Re-test SwiGLU and weight decay 0.1 on the DEPTH 6 base.
3. Try batch size or sequence length changes.
4. Avoid more LR tweaks.
5. Check the tripwire flag.
