Helped: only the batch-size cut. TOTAL_BATCH_SIZE 2**18 gave 1.0981 (-0.0164), the incumbent. More optimizer steps in 300 s matter most.

Did not help: WARMDOWN_RATIO 0.3 (+0.0006), WINDOW_PATTERN "L" (+0.0047), depth 10 (+0.0227), SwiGLU (+0.0023). Several runs scored better than the incumbent but were rejected: WARMDOWN_RATIO 0.7 (1.0935), embedding LR 0.4 (1.0921/1.0856), Muon momentum 0.96 (1.0915/1.0878), MATRIX_LR 0.05 (1.0900/1.0917), decaying weight decay (1.0974). Their deltas are small and the replicates disagree, so noise is substantial. The incumbent has only 1 run.

Next: replicate the incumbent to measure noise. Try TOTAL_BATCH_SIZE 2**17. Then combine the small gains (WARMDOWN_RATIO 0.7, embedding LR 0.4, MATRIX_LR 0.05). Avoid bigger models and attention-window changes. Check train.py first, since slots 2 and 4 were aborted because it already had those settings.
