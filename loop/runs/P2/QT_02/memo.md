Incumbent mean_adjusted is 1.1146 over 16 runs. No change has beaten it in 20 slots, so nothing is confirmed as helping.

Helped: nothing. SwiGLU scored 1.1124 in slot 0 (-0.0022), which is probably noise. The slot 13 replicate scored 1.1297 (+0.0151), so drop it.

Hurt:
- Cosine annealing: +0.0579 and +0.0604.
- Stochastic depth: +0.1026.
- Batch 2^20: +0.0778.
- Gradient-norm LR scaling: +0.0451.
- Dropout: +0.0272.
- HEAD_DIM 64: +0.0158.
- Warmup 0.1: +0.0106.
- Warmdown 0.2: +0.0065.
- Weight decay 0.01: +0.0032.

Crashes: slots 4, 9, 11, 12 and 14. Aborts and a patch failure also cost slots.

Next:
1. Try a longer warmdown or higher weight decay, since the shorter warmdown and lower weight decay both lost.
2. Try a smaller total batch size or a model-size change.
3. Check each patch against the file before using a slot.
4. Avoid schedule rewrites, adaptive LR, dropout and other regularization.
