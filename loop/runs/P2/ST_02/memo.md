Helped: shrinking the model and taking more optimizer steps. Slot 3 (6 layers, ASPECT_RATIO 85) scored 1.1026 (-0.0120). Slot 5 (total batch 2^18) scored 1.0662 and 1.0593 (-0.0398), the biggest gain. The replicates differ by about 0.007 and slot 5 has a tripwire flag, so audit it before trusting it. The incumbent mean is 1.0627.

Did not help: schedule tweaks (cosine warmdown +0.0607, warmdown 0.3 -0.0003, warmup 0.02 +0.0000), HEAD_DIM 64 (+0.0114), Muon momentum change (+0.0094), weight decay 0.1 (+0.0036), softcap 10 (+0.0042), scalar LR 0.25 (+0.0033), window pattern SSSSSL (+0.0044), depth 8 with 3x MLP (+0.0161). Embedding init 0.5 (-0.0016) and 3x MLP (-0.0021) are within noise. Three patches failed to apply, and the Muon NS-steps change crashed.

Next: (1) audit slot 5. (2) Try batch 2^17. (3) Retry the failed edits (gradient clipping, LR tuning, beta2) with SEARCH text copied exactly from the file. (4) Skip further small schedule and regularization tweaks.
