Helped: only slot 5, removing per-step .item() and timing syncs (official 1.0918, -0.0227 vs the old incumbent). Throughput gains that allow more steps in 300 s are the winning direction.

Did not help: depth changes (DEPTH=6 +0.0166 and +0.0050, DEPTH=10 +0.0319), HEAD_DIM 64 (+0.0143), SwiGLU (+0.0041), warmup 0.03 (+0.0123), lower EMBEDDING_LR (+0.0046), lower SCALAR_LR (+0.0020), momentum 0.97 (+0.0061), weight decay 0.1 (+0.0002). Warmdown tweaks were near-neutral (-0.0005, -0.0009), so they are not worth more slots.

Doubling UNEMBEDDING_LR to 0.008 was rejected twice, at -0.0044 and -0.0033 versus the incumbent, with a single run each. That is a consistent small gain. Replicate it, then consider accepting it.

Next: more throughput work (batch size, compile or fused-kernel tweaks, remaining syncs). Keep depth 8 and avoid downward LR nudges.
