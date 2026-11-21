# G1: is real web curation with skills deep enough to route? (gate, Nov 12 2026)

Written before any training run on pool web1. Label: **gate**. It decides whether the skill-routing version of the
curation task gets built. It is not a result of the thesis.

## Why

C1 and S6 showed the synthetic curation task is shallow: one stored file solves it. The proposed replacement is real
curation, where an agent builds a pipeline out of skills (dedup, cleaning, embedding filters, a model that judges
quality) and routing decides which model runs the model-powered skills. That is only worth building if, on real data,
(a) a full pipeline beats the standard fixed one and (b) the strength of the judge matters.

## Pool

`data/curation/pool_web1` (`curation/skills/build_pool_web.py`, seed 0): 122,383 pages from 6 random text files of
Common Crawl CC-MAIN-2026-43, every page of at least 200 characters, all languages, no filtering. Two views per page:
raw text and line-cleaned text. Target sample, dev and sealed sets: clean Wikipedia, copied from pool v2b. Trainer,
token budget (T = 9,830,400) and score (bits per byte): `curation/task/run.py`, unchanged.

## Arms

Every arm hands the trainer a set of documents in a seeded random order. Skills are in `curation/skills/ops.py`.

| Arm | Pipeline | Model calls |
|---|---|---|
| RAW | all pages, raw view | none |
| BASE | cleaned view, document heuristics (length, word shape, symbols, English), exact dedup. The standard fixed pipeline | none |
| EMB | BASE, then semantic dedup (cosine above 0.9), then the T tokens closest to the target sample in embedding space | none |
| JS | BASE, semantic dedup, then the T tokens with the highest predicted score of the strong judge (Fable) | 200 strong |
| JC12 | same with the Gemma 4 12B judge | 200 cheap |
| JC4 | same with the Qwen3.5 4B judge | 200 cheap |

Judge arms: 2,000 documents sampled after semantic dedup are scored 1 to 5 by the judge (10 documents per call, first
1,500 characters each, same prompt and the same two target excerpts for every judge). A ridge regression on the
embeddings spreads the scores to all documents. Documents a judge fails to score are left out of its regression and
counted. The three judge arms differ only in who gave the 2,000 scores.

All settings above are fixed now. Nothing is tuned on dev or sealed scores.

## Runs and rule

3 seeds per arm (0, 1, 2), 18 runs. Noise sd = pooled within-arm sd of the sealed bpb.

- **Gate G: mean sealed bpb of BASE minus JS is above 3 noise sds.** Pass: real curation has room beyond the fixed
  pipeline. Fail: shallow again, stop, nothing is built.
- Reads that decide what to build if G passes (same 3 sd yardstick, no extra rule):
  - JS against EMB: does a judging model add anything over the free embedding skill?
  - JS against JC12 and JC4: does the strength of the judge matter?
  - Reported next to them: rank agreement of the judges on the 2,000 documents and the overlap of the selected sets.

| Outcome | What gets built |
|---|---|
| G fails | nothing; the curation line stays as the shallow contrast |
| G passes, JS no better than EMB | no model routing inside the skills; at most a planner-level study |
| G passes, JS beats EMB, cheap judges match JS | route is "always cheap judge"; planner-level study only |
| G passes, JS beats EMB and the cheap judges | the full skill-routing task (per-document escalation) |

## Predictions

BASE beats RAW by more than 0.1 bpb (95 percent). G passes (75 percent). JS beats EMB by 3 sds (35 percent).
JS beats JC12 by 3 sds (30 percent). JS beats JC4 by 3 sds (45 percent).

## Limits

One pool, one target (Wikipedia), one small trainer, 3 seeds. The judge reaches the pool through a small regression
on embeddings, which can hide differences between judges. A pass here licenses a build, not a claim.

## Shared machines

Pool build, embedding and training runs are capped (systemd scope or the 16 GB runner cap); one `mem_guard.py` per
host during training runs. Department limit: 50 GB per machine.

## Deviations

1. Order of work (no effect on any number): RAW, BASE and EMB were trained while the judges were still scoring. Arm
   files for the judge arms were first written from partial scores to test the launcher; they were rebuilt from the
   full 2,000 scores before any judge arm was trained.

Result: `curation/RESULT_G1.md`.
