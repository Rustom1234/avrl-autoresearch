# Curation pilot, Stage A: is there room for a curator that understands documents?

Run Nov 10 2026 on lab A4000s (spruce, willow, cherry). Trainer: `task/run.py` (small GPT, 300 steps, T = 9,830,400 tokens, about 2 minutes per run). Score: bits per byte (bpb, lower is better) on held-out target-domain text, a dev set that loops may see and a sealed set they may not. Table script: `python curation/task/pilot_table.py v1|v2`. Results: `curation/results/<pool>/`.

## Question and gates

A research loop can only be ranked on this task if curation matters and simple rules do not already solve it. Two numbers, both in units of the single-run noise sd of the raw pool (8 seeds):
- **G1, headroom:** best no-LLM baseline minus best oracle. Needs 5 or more. The oracle uses the hidden document labels, so it is the ceiling for a curator that understands what each document is.
- **G2, signal:** raw pool minus best no-LLM baseline. Shows that curation moves the score at all.

No-LLM baselines: exact dedup, random half, a hand heuristic, and TF-IDF similarity to the target sample (keep the top 30, 50, 80 or 100 percent, ordered by similarity).

## Pool v1: fails G1 (too easy)

Pool: 17,726 documents, 2 T tokens: wiki 0.25 T, web 0.20 T, code, Q and A, duplicates, lightly damaged text. Raw-pool noise sd: 0.0147 dev, 0.0141 sealed.

| Curator | Dev bpb | Sealed bpb | Sealed change vs raw |
|---|---|---|---|
| Oracle: clean wiki only | 1.8267 | 1.8518 | -0.158 |
| TF-IDF, top 50 percent | 1.8613 | 1.8878 | -0.122 |
| Oracle: web then wiki | 1.8709 | 1.8920 | -0.118 |
| Oracle: wiki then web | 1.8706 | 1.8927 | -0.117 |
| TF-IDF, top 80 percent | 1.8726 | 1.8955 | -0.114 |
| TF-IDF, top 30 percent | 1.8805 | 1.9118 | -0.098 |
| TF-IDF, all, ordered | 1.8911 | 1.9147 | -0.095 |
| Hand heuristic | 1.9538 | 1.9753 | -0.034 |
| Raw pool (8 seeds) | 1.9887 | 2.0095 | 0 |
| Random half | 1.9904 | 2.0117 | +0.002 |
| Exact dedup | 2.0090 | 2.0273 | +0.018 |

- **G1 = 2.3 noise sds on dev, 2.6 on sealed. Fail.** A word-count similarity rule already gets 77 percent of the oracle's gain, so there is little left for a smart curator to find.
- G2 = 8.6 and 8.7 noise sds.

## Pool v2: passes G1

Pool: 20,746 documents, 2 T tokens, built to defeat word-count methods (`build_pool.py --preset v2 --seed 0`): clean wiki 0.40 T, word-shuffled wiki 0.38 T (same words, broken order), spam 0.36 T, clean web 0.24 T, plus code, Q and A, duplicates and other damage. Same dev and sealed sets as v1. Raw-pool noise sd: 0.0063 dev, 0.0072 sealed.

| Curator | Dev bpb | Sealed bpb | Sealed change vs raw |
|---|---|---|---|
| Oracle: clean wiki only | 1.8298 | 1.8552 | -0.129 |
| Oracle: web then wiki | 1.8669 | 1.8890 | -0.095 |
| Oracle: wiki then web | 1.8657 | 1.8894 | -0.095 |
| TF-IDF, all, ordered | 1.9340 | 1.9586 | -0.026 |
| TF-IDF, top 80 percent | 1.9324 | 1.9588 | -0.025 |
| Hand heuristic | 1.9462 | 1.9690 | -0.015 |
| Random half | 1.9572 | 1.9812 | -0.003 |
| TF-IDF, top 50 percent | 1.9536 | 1.9825 | -0.002 |
| Exact dedup | 1.9607 | 1.9832 | -0.001 |
| Raw pool (8 seeds) | 1.9607 | 1.9842 | 0 |
| TF-IDF, top 30 percent | 1.9786 | 2.0107 | +0.027 |

- **G1 = 16.2 noise sds on dev, 14.3 on sealed. Pass.** The best no-LLM rule gets 20 percent of the oracle's gain.
- G2 = 4.5 and 3.5 noise sds: simple rules help a little.

## Decision

Stage B (the two anchor loops) runs on pool v2. Plan and gate: `loop/STAGE_B_PLAN.md`.

## Limits

- Oracles and baselines have 2 seeds each; the raw pool has 8.
- "Oracle" is the best of three label-based recipes, not a proven optimum. A loop could beat it, for example by ordering documents.
- v2's junk was designed by hand to beat bag-of-words. That makes the task a fair test of "does the curator read the documents", and a less fair picture of real web data.
