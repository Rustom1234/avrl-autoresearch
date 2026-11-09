Incumbent mean_adjusted is 1.1146 over 16 runs. No tried change beat it.

Architecture changes did not help (1 accepted, 5 rejected by family count). Depth and size changes hurt most: depth 10 with batch 32 scored 1.1603 (+0.0457), and depth 6 with a wider aspect ratio scored 1.1241 (+0.0095). SwiGLU scored 1.1195 (+0.0049). WINDOW_PATTERN "SL" scored 1.1148 (+0.0002), which is a tie.

Optimizer changes are the most promising. Doubling UNEMBEDDING_LR to 0.008 scored 1.1112 (-0.0034). It was rejected, probably because it was a single run within noise, but it's the only change that moved in the right direction.

Next steps:
- Replicate the UNEMBEDDING_LR change, or try a smaller step such as 0.006.
- Tune other learning rates, weight decay and the LR schedule.
- Avoid large architecture changes, since fixed 300 s of training favors the current size.
- Try small tweaks to batch size or warmdown.
