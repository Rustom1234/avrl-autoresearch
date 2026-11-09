Helped: only the batch-size cut. TOTAL_BATCH_SIZE 2**19 to 2**17 gave more optimizer steps and moved val_bpb to 1.0522 (delta -0.0624). It carries a tripwire flag, so a replicate or audit is worth a slot.

Did not help (rejected vs incumbent): schedule tweaks (WARMDOWN_RATIO 0.3 -0.0007, 0.7 +0.0045; 3% warmup +0.0109; cosine decay +0.0012), Muon LR 0.05 (+0.0021), DEPTH 6 (-0.0000), DEPTH 10 (+0.0820), "SL" window pattern (-0.0014), beta2 0.98 (-0.0011), weight decay 0.1 (-0.0010), final momentum 0.9 (-0.0007), momentum ramp 100 steps (-0.0017). Batch 2**15 (-0.0010) and the 2**15 to 2**17 ramp (-0.0013) were also small. Those gains are close to noise, since none was clearly better.

Next: audit/replicate the incumbent first. Then combine the small gains (weight decay 0.1, beta2 0.98, "SL", shorter momentum ramp) in one run, possibly with a smaller batch. Avoid more schedule-only tweaks and no-change slots (slots 2, 8, 12 and 14 aborted).
