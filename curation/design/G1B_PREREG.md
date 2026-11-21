# G1B: closing the two holes in gate G1 (addendum, Nov 13 2026)

Written after G1 (`curation/RESULT_G1.md`) and before any G1B model call or training run. Label: **follow-up to a
gate**. It cannot change the registered G1 verdict. It decides whether the curation line is closed or gets one more
build.

## Why

G1 passed narrowly and said a judging model adds nothing over the free embedding ranking. Two things could make that
reading wrong:
1. The judge was handicapped. It scored 2,000 documents and a weak regression on embeddings (R^2 0.35) spread the
   scores over the other 32,583. Maybe a judge that scores every document directly does better.
2. No arm combined skills. The embedding ranking and the judge ranking share only 31 percent of their documents and
   score the same. Maybe together they beat each alone.
Also, 3 seeds per arm is thin for a pass at 3.9 sds.

## Pool and trainer

Unchanged: `data/curation/pool_web1`, `curation/task/run.py`, T = 9,830,400 tokens, sealed bits per byte on clean
Wikipedia. All arms select from the same 34,583 documents left after the fixed pipeline and semantic dedup
(`curation/runs/G1/prep.npz`), and hand them to the trainer in a seeded random order.

## New arms

| Arm | Ranking used to take the best T tokens | Model calls |
|---|---|---|
| JC12D | the Gemma 4 12B judge scores **every** document directly (1 to 5) | 3,459 cheap |
| JC4D | the Qwen3.5 4B judge scores every document directly | 3,459 cheap |
| COMB | sum of two standardized scores: embedding closeness to the target (arm EMB) and the predicted Fable score (arm JS) | none new |

Direct scoring uses the G1 prompt, the same two target excerpts, 10 documents per call, first 1,500 characters each.
Documents are put into calls in a seeded random order. Scores are whole numbers, so many documents tie: ties are
broken by a ridge regression on embeddings fitted to all of that judge's own direct scores. A document the judge
fails to score gets that regression's prediction in place of a score, and is counted.

Code: `curation/skills/gate1b.py` (new file; `gate1.py` and the G1 arm files are not changed).

## Runs

Seeds 0 to 5 for the new arms (18 runs). Seeds 3, 4, 5 are added to BASE, EMB, JS, JC12 and JC4 (15 runs), so every
arm compared here has 6 seeds. RAW is not rerun. Noise sd = pooled within-arm sd of the sealed bpb over these eight
arms. Yardstick, same as G1: a difference of means counts when it is above 3 noise sds of one run. The standard error
of each difference is reported next to it.

## Questions

- **Q1, was the judge handicapped?** JC12 minus JC12D above 3 sds.
- **Q2, does a judge that scores directly beat the free embedding ranking?** EMB minus JC12D above 3 sds.
- **Q3, does composing skills beat each skill alone?** The better of EMB and JS, minus COMB, above 3 sds.
- **Q4, does the G1 gate hold on 6 seeds?** BASE minus JS above 3 sds. Reported next to the registered 3-seed verdict,
  which stays.

| Outcome | What follows |
|---|---|
| Q1, Q2 and Q3 all no | the curation line is closed as the shallow contrast; nothing is built |
| Q3 yes | a planner-level study of skill composition is worth a plan (offered to Rustom with a plan review) |
| Q1 or Q2 yes | stage 2: the Fable judge scores every document directly (about 3,459 calls, about 390 USD of plan usage at list price, no cash) and JS direct is compared with JC12D. Per-document routing is back on the table only if Fable direct beats Gemma direct by 3 sds |

Read without a rule: rank agreement of the direct Gemma scores with Fable on the 2,000 G1 documents; agreement of each
cheap judge with its own G1 scores on those documents (same document, different neighbours in the call); overlap of
the selected sets; JC4D against JC12D.

## Predictions

Q1 yes: 25 percent. Q2 yes: 25 percent. Q3 yes: 30 percent. Q4 holds: 80 percent.

## Limits

One pool, one target, one small trainer. COMB is one fixed way to combine two skills (equal weights), chosen before
any run; a failure of COMB does not rule out every composition.

## Shared machines

Hard rule since Nov 12: no kernel memory kill on a lab machine, including inside my own cap. Judge clients run under
`prlimit --as` (measured peak 1.8 GB resident for the same code in G1). Training runs have the 16 GB address-space
limit of `curate_runner.py` (measured peak of my total memory per host in G1: 3.7 GB on willow and cherry, 16 GB on
spruce with three judge clients and P2B). One `mem_guard.py` per host during training. Each new step is first run at
small size and its peak memory read before the full run.

## Deviations

1. Setup (no effect on any number): the two open-model servers (Gemma on pine, Qwen on walnut) had been stopped by
   the P2B launcher. They were restarted with the same model files and settings as in G1 before direct scoring. A
   240-document test of each judge ran first (written to `_test` files, not used in any arm).
2. Order of work (no effect on any number): seeds 3, 4 and 5 of BASE, EMB, JS, JC12 and JC4 were trained while the
   judges were still scoring. Their arm files are the unchanged G1 files. No result was read before all runs of the
   plan had finished.
