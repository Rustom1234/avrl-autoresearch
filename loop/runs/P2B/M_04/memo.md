Incumbent mean_adjusted is 1.1145 over 16 runs. No slot has been accepted so far. Slots 5, 8 and 13 were invalid (engine:patch_fail). Rewrite their edits with SEARCH text that matches exactly once.

Helped slightly (single runs, unconfirmed): FINAL_LR_FRAC 0.1 (slot 9) scored 1.1113, delta -0.0032. Batch ramp-up (slot 2) scored 1.1133, delta -0.0013. q/k gain (slot 0) scored 1.1136, delta -0.0009. EMA (slot 12) was -0.0001, which is neutral.

Hurt: tied lm_head (+1.7839), byte-weighted loss (+0.0635), cosine decay (+0.0617), sublayer RMSNorm (+0.0240), SwiGLU (+0.0104), parallel blocks (+0.0100). Neutral or worse: GQA, WARMDOWN_RATIO 0.7, Newton-Schulz coefficients (+0.0027), extra embedding table (+0.0024).

Next: replicate slot 9. If it holds, combine it with slot 2 and slot 0. Keep changes small and avoid architecture or loss swaps.
