Incumbent mean_adjusted is 1.1145 over 16 runs. No change has been accepted yet, but four official scores were lower than the incumbent and were still rejected: the sequence length curriculum (slot 12) at 1.1109 (-0.0036), the logging fix (slot 14) at 1.1126 (-0.0019), the sqrt warmdown (slot 8) at 1.1130 (-0.0015), and one more near that range. These are probably within noise, so replicate the best ones before drawing conclusions. Try combining the sequence length curriculum with the sqrt warmdown.

Did not help: larger or more complex architecture changes. Ten layers was +0.0213, HEAD_DIM 64 was +0.0116, first-layer value mixing was +0.0073, GQA was +0.0015, and cosine decay was +0.0038. The EMA run scored 2.4356, which points to a bug.

Process: of the 20 slots, 4 were patch failures, 3 were aborts, 2 were crashes and 1 was a target_leak invalid run. Copy SEARCH text exactly, keep losses shape-safe, and do not touch evaluation code.
