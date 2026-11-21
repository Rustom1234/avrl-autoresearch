# G1 result: is real web curation with skills deep enough to route? (Nov 12 2026)

Plan, arms, rule and predictions: `curation/design/G1_PREREG.md`, written before any training run on the pool.
Numbers: `curation/runs/G1/report.json` (from `curation/skills/gate1.py report`). Label, as fixed in the plan: **gate**.

## Verdict

**Gate G passes, narrowly. The reads say there is nothing to route inside the skills.**

| Question | Result |
|---|---|
| Gate G: does a full pipeline beat the standard fixed one by more than 3 noise sds? | BASE minus JS = +0.0398 bpb = 3.9 sds. **PASS** |
| Does a judging model add anything over the free embedding skill? | EMB minus JS = +0.0029 bpb = 0.3 sds. **No** |
| Does the strength of the judge matter? | JC12 minus JS = +0.0018 (0.2 sds), JC4 minus JS = +0.0137 (1.4 sds). **No** |

In the plan's outcome table this is row 2: "G passes, JS no better than EMB: no model routing inside the skills; at
most a planner-level study."

## Numbers

Sealed bits per byte on clean Wikipedia, lower is better, 3 seeds per arm. Noise sd of one run: 0.0101 (pooled within
arms, 12 degrees of freedom).

| Arm | Pipeline | Documents | Sealed bpb | By seed | Model cost (nominal) |
|---|---|---|---|---|---|
| RAW | all pages, raw text | 122,383 | 2.7015 | 2.7057 2.7031 2.6959 | 0 |
| BASE | cleaning, heuristics, English check, exact dedup | 34,877 | 2.4537 | 2.4585 2.4437 2.4590 | 0 |
| EMB | BASE, semantic dedup, closest to the target by embedding | 10,718 | 2.4169 | 2.4088 2.4006 2.4413 | 0 |
| JS | BASE, semantic dedup, ranked by the Fable judge | 9,576 | 2.4140 | 2.4114 2.4201 2.4104 | 22.42 USD |
| JC12 | same with the Gemma 4 12B judge | 9,644 | 2.4158 | 2.4144 2.4197 2.4132 | 0.11 USD |
| JC4 | same with the Qwen3.5 4B judge | 9,537 | 2.4277 | 2.4300 2.4277 2.4252 | 0.04 USD |

How to read it. The standard fixed pipeline does most of the work: it removes 0.248 bpb from the raw crawl. A full
pipeline removes 0.04 more (about 16 percent on top). That extra is the same size whether the ranking comes from a
free embedding comparison, a 12B local model or Fable. Fable cost about 200 times more than Gemma for the same score.

Other facts:
- Funnel: 122,383 pages, 34,960 pass the document heuristics, 34,877 after exact dedup (3.03 T tokens), 34,583 after
  semantic dedup (3.00 T). Semantic dedup removed under 1 percent, so it did almost nothing on this pool.
- Judges agree in rank with Fable on the 2,000 scored documents: Gemma 0.81, Qwen 0.68. All 600 judge calls returned
  valid scores.
- The regression that spreads the 2,000 scores over the pool is weak: cross-validated R^2 0.35 (Fable), 0.35 (Gemma),
  0.25 (Qwen).
- Selected sets: JS shares 84 percent of its documents with JC12, 75 percent with JC4, and only 31 percent with EMB.
  The judge route and the embedding route pick mostly different documents and land on the same score.

## Predictions against outcomes

| Prediction | Outcome |
|---|---|
| BASE beats RAW by more than 0.1 bpb (95 percent) | 0.248. Right |
| G passes (75 percent) | Passes at 3.9 sds. Right |
| JS beats EMB by 3 sds (35 percent) | 0.3 sds. Did not happen, as the odds said |
| JS beats JC12 by 3 sds (30 percent) | 0.2 sds. Did not happen |
| JS beats JC4 by 3 sds (45 percent) | 1.4 sds. Did not happen |

## Limits

- The pass is narrow: 3.9 sds against a bar of 3, with 3 seeds per arm. EMB has one seed (2.4413) well above its
  other two.
- The judge reaches the pool through a weak regression on embeddings (R^2 0.35). This can hide a real difference
  between judges. A fairer test of the judge is to score every one of the 34,583 documents directly. That is cheap for
  the local models and was not run.
- One pool, one target (Wikipedia), one small trainer (2 minutes), top third selected. The web pool is far from the
  target: the best arm is at 2.41 bpb, the old pools that contained clean Wikipedia reached 1.87.
- No arm combined the embedding score and the judge score. They pick different documents, so a combination could add.
  Not tested.

## What it means for the thesis

Third time the same shape: on a curation task with a 2 minute trainer, most of the gain is cheap (fixed rules here, one
stored file in S6), and the expensive model buys nothing measurable in the role tested. Per-document routing between
judges has nothing to route on this pool. What is left open is the planner level: which skills to compose and in what
order (for example, combining the two rankings that pick different documents).

## The run

- Pool build: 6 Common Crawl WET files, under a 20 GB cap. Embedding: one A4000 (cherry), under a 16 GB cap.
- Judges: 200 calls each, 10 documents per call. Fable at low effort, 22.42 USD nominal (plan usage, no cash).
  Gemma and Qwen on the existing servers on pine and walnut with 2 threads each, while P2B was live.
- Training: 18 runs on spruce, willow and cherry, 16:05 to 16:20 EST, all finished.
- Shared machines: peak of my total memory from the guard logs: spruce 16.0 GB (P2B and the three judge processes
  included), willow 3.7 GB, cherry 3.7 GB. Department limit 50 GB. The guards stopped nothing and were stopped after
  the last run.
