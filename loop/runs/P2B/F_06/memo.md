Helped: shrinking the model and raising optimizer step count under the fixed 300 s budget. DEPTH 8 to 6 gave -0.0330 (runs 1.0828/1.0802). TOTAL_BATCH_SIZE 2^17 gave -0.0193 (1.0613/1.0632). Both were accepted but carry a tripwire flag, so replicate the incumbent (mean_adjusted 1.0623) once to check.

Small gains, all below the acceptance bar: weight decay 0.1 (-0.0036), DEPTH 10 with dim 512 (-0.0031), WARMDOWN_RATIO 0.65 (-0.0021), grad clipping (-0.0014), scalar LR 0.25 (-0.0010). Their effects might stack, so try combining weight decay 0.1 with warmdown 0.65.

Did not help: warmup (+0.0109), 3x MLP (+0.0194), factorized value/output projections (+0.0149), momentum ramp, window pattern, TOTAL_BATCH_SIZE 2^18, DEVICE_BATCH_SIZE 32.

Next: DEPTH 5 or 4, TOTAL_BATCH_SIZE 2^16, and LR changes. Re-submit the MATRIX_LR and UNEMBEDDING_LR edits, which failed to patch. Avoid architecture rewrites.
