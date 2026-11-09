No change has beaten the incumbent (1.1146 mean over 16 runs). Nothing has been accepted yet.

Helped slightly: raising EMBEDDING_LR to 1.0 and UNEMBEDDING_LR to 0.008. Two runs gave 1.1102 and 1.1073 (-0.0058 adjusted), but it was still rejected. Raising SCALAR_LR to 1.0 was nearly neutral (-0.0010).

Did not help:
- Lowering the lm_head LR (+0.0188) or the Muon LR (+0.0151).
- Raising the Muon LR to 0.08 (+0.0044).
- Cosine warmdown (+0.0041), longer warmdown (+0.0069) and warmup (+0.0122).
- Weight decay 0.01 (+0.0050) and Adam betas (+0.0022).
- Larger HEAD_DIM (+0.0187 at 144, +0.0082 at 192).
- TOTAL_BATCH_SIZE 2**20 (+0.0679) and checkpointing (+0.0365).

Depth 16 and the batch 32 retry crashed with out-of-memory errors. Two slots were aborted on invalid edits.

Next:
1. Combine the embedding/unembedding LR raise with SCALAR_LR 1.0.
2. Try intermediate embedding/unembedding values.
3. Try depth 10 cautiously on memory.
4. Validate each edit before submitting.
