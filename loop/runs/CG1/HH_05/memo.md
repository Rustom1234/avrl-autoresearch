Helped: only slot 7, random 28x28 crops for the first half of optimizer steps (cost -0.0601, delta -0.0768 vs incumbent, 1 run). It cuts per-step compute, so it is a speed gain. Single run, noise sd about 0.026, so replicate before stacking.

Did not help: structural changes (conv2 residual skip +0.2176; head to Muon +5.3473; removing the final pool crashed twice), schedule changes (flat-then-decay +0.4503; warmup +0.1187), and optimizer tweaks (NorMuon +0.0450). Colour jitter off was +0.0333. Neutral within noise: richer TTA +0.0021, momentum ramp +0.0061, label smoothing anneal +0.0189. Batch 1024 with scaled LRs was +0.0579. Translation 3 px was -0.0099 vs incumbent, a small gain within noise.

Next: replicate slot 7. Then try progressive resolution variants: smaller crops for longer, or a gradual size ramp. Look for other changes that cut per-step cost. Avoid architecture, schedule and optimizer edits.
