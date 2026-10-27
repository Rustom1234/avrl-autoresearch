Helped: size and batch changes. DEPTH 6 gave 1.0832 (delta -0.0275, accepted). TOTAL_BATCH_SIZE 2**18 gave 1.0650/1.0651 (delta -0.0231, accepted). The incumbent is mean_adjusted 1.0643 over 2 runs.

Did not help: schedule, optimizer, regularization and architecture tweaks. Rejected: WARMDOWN_RATIO 0.7, UNEMBEDDING_LR 0.008, SwiGLU, WEIGHT_DECAY 0.1 (+0.0035), HEAD_DIM 64 (+0.0016), cosine decay (+0.0061), q scale 1.2 (+0.0026), DEPTH 10 (1.1221, +0.0620). Value embeddings on every layer were rejected with a tripwire flag, over 3 runs.

Open question: 2**17 batch gave 1.0615/1.0630 (delta -0.0052) but was rejected. Its gain is close to the noise level, so replicate it or combine it with another change before judging it.

Check the tripwire flags on slots 5 and 8.

Next: try DEPTH 5 or 4, since going smaller has paid off and going larger hurt. Then retune LRs and warmdown on the new size. Skip small tweaks.
