# P2B cell N (no-LLM random search): implementation notes

Written 2026-11-10, after P2B_PREREG.md was frozen and while the P2B blocks were running. The preregistration fixes the
rule of cell N (eight settings, two directions, same z rule, 20 slots, 12 loops, descriptive only). This file lists
the choices the rule leaves open. They were fixed before any cell N loop ran. Code: loop/core/random_search.py, tests
in loop/core/tests/test_random_search.py.

1. **Moves that cannot change the script are redrawn.** WARMDOWN_RATIO already at its clip (0.1 or 0.9), DEPTH that
   would fall below 2, a total batch smaller than one sequence (2,048 tokens). They never reach a GPU.
2. **DEVICE_BATCH_SIZE follows TOTAL_BATCH_SIZE down.** train.py asserts that the total batch is a multiple of
   DEVICE_BATCH_SIZE x 2,048. The base total is 2^19 and the per-device batch is 64 (2^17 tokens), so the total can be
   halved twice as is. Below that the per-device batch is set to total / 2,048. It is never raised above 64.
3. **All moves used up.** If every possible move has been rejected on the current incumbent, the rejected list of
   that incumbent is cleared and the draw starts over.
4. **Crashes.** A candidate that crashes (for example out of memory after DEPTH plus 2) is rejected and costs its
   slot. There is no crash-fix slot, because that slot is a repairer call.
5. **No critic.** Borderline accepts, audits and periodic reviews go to the critic model in the other cells. Cell N
   makes no model call at all, so these reviews do not happen. The z rule, replication, the tripwire audit run and
   the integrity flags are the engine's own code and apply unchanged.
6. **Session and scoring.** The launcher of the main blocks stops the GPU workers when its confirmation block ends,
   so cell N runs in a second whole-node job on the same node (hannah n7) with the same frozen anchor.json. Its
   final incumbents are confirmed with the same procedure (seeds 1001 and 1002, sealed shard) together with fresh
   anchor runs, and its sealed gain is measured against those anchor runs. Loop seeds are 3101 to 3112.
7. **Analysis.** Cell N is kept out of the tables behind the gates, claims and verdict (a missing N loop cannot drop
   a block). Its gap recovered, extra cost and advantage are reported in their own section
   (analysis_p2b.py, option --n-run).
