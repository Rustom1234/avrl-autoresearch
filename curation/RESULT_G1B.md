# G1B result: the judge was not handicapped, and the curation line closes for routing

Nov 13 2026. Plan: `curation/design/G1B_PREREG.md` (written before any of these runs). Code:
`curation/skills/gate1b.py`. Numbers: `curation/runs/G1B/report.json`. Follows `curation/RESULT_G1.md`.

## Why this was run

Gate G1 said: on a real web pool, a skill pipeline beats the standard fixed pipeline, but a model that judges
document quality adds nothing over the free embedding ranking, and a small judge does as well as the strong one.
Three things could have made that reading wrong:

1. In G1 a judge only scored 2,000 documents and a small regression guessed the rest (it explained 25 to 35 percent
   of the variance). Maybe the judge was fine and the regression was the weak link.
2. G1 only tried each skill alone. Maybe the value is in combining them.
3. 3 seeds per arm is thin for a pass at 3.9 noise sds.

## What was run

Same pool (34,583 documents left after cleaning and both dedups), same trainer, same token budget, same sealed score
(bits per byte on held-out Wikipedia, lower is better).

| Arm | How it picks its documents | Model calls |
|---|---|---|
| BASE | standard fixed pipeline, no ranking (from G1) | none |
| EMB | closest to the target in embedding space (from G1) | none |
| JS | predicted score of the strong judge, from 2,000 scored documents (from G1) | 200 strong |
| JC12, JC4 | same with the Gemma 12B and Qwen 4B judges (from G1) | 200 cheap |
| **JC12D** | Gemma scores **every** document itself, no regression | 3,459 cheap |
| **JC4D** | Qwen scores every document itself | 3,459 cheap |
| **COMB** | two skills together: embedding closeness plus predicted strong-judge score, equal weight | 200 strong |

6 seeds for every arm (seeds 3, 4, 5 added to the G1 arms), 33 new training runs. Both judges scored all 34,583
documents with 0 failed calls (1.93 and 0.65 USD at nominal prices). Yardstick as in G1: a difference counts when it
is above 3 noise sds of one run. Noise sd here: 0.0100 bpb (40 degrees of freedom), so 3 sds = 0.030.

## Result

| Arm | Sealed bpb (mean of 6) | Against BASE |
|---|---|---|
| BASE | 2.4509 | |
| JC4D | 2.4277 | -0.023 |
| JC4 | 2.4239 | -0.027 |
| JC12 | 2.4179 | -0.033 |
| EMB | 2.4144 | -0.037 |
| JS | 2.4115 | -0.039 |
| JC12D | 2.4104 | -0.041 |
| COMB | **2.3906** | **-0.060** |

| Question | Difference | In noise sds | Answer by the rule |
|---|---|---|---|
| Q1: was the judge handicapped? (JC12 minus JC12D) | +0.0075 | 0.8 | **no** |
| Q2: does a direct judge beat free embeddings? (EMB minus JC12D) | +0.0040 | 0.4 | **no** |
| Q3: does combining beat each skill alone? (JS minus COMB) | +0.0209 | 2.1 | **no** (below 3) |
| Q4: does gate G hold on 6 seeds? (BASE minus JS) | +0.0395 | 3.9 | **yes** |

Predictions written beforehand: Q1 yes 25 percent, Q2 yes 25, Q3 yes 30, Q4 holds 80. All four went the likelier way.

**Outcome by the table in the plan: Q1, Q2 and Q3 are all no, so the curation line is closed as the shallow contrast.
Nothing is built. Stage 2 (the strong judge scoring every document, about 3,459 strong calls) is not run.**

## How to read it

- **The regression was not the weak link.** Letting Gemma read all 34,583 documents instead of 2,000 moves the score
  by 0.0075, under one noise sd. Seventeen times more judge calls bought nothing measurable.
- **A judge still does not beat embeddings.** The direct judge and the free embedding ranking pick very different
  documents (29 percent shared) and land on the same score.
- **Judge strength still does not matter at the level this trainer can see.** JC12 against JS: 0.6 sds. Gemma's
  direct scores agree with the strong judge's at rank agreement 0.80, Qwen's at 0.67; Qwen's arm is worse by 1.7 sds,
  still under the yardstick. Asked twice about the same document in different company, Gemma gives the same score
  69 percent of the time and Qwen 53 percent.
- **Combining two skills is the best arm, by a small margin.** COMB beats JS by 0.021 bpb. That is 2.1 noise sds of
  one run, so the registered answer is no. For the record: with 6 seeds per arm the standard error of that difference
  is 0.0058, so the difference is 3.6 standard errors from zero, and every COMB run beats every EMB and JS run. So the
  effect is very likely real, and it is small: half of what the ranking skills add over BASE, and a twelfth of what
  basic cleaning adds over raw pages (0.25 bpb in G1). It also has no routing side: COMB uses the same 200 judge calls
  as JS, and in that kind of arm the cheap judge matches the strong one (JC12 against JS).
- **Gate G holds on 6 seeds** at the same size as on 3 (3.9 sds). Real web curation has room beyond the fixed
  pipeline; the room is found by cheap skills.

## What this means for the thesis

On this curation task the ordering is: cleaning (0.25 bpb) far above ranking (0.04) above combining rankings (0.02),
and none of the three depends on which model does the judging. There is nothing for a router to decide inside the
skills. This is the third curation result with the same shape (C1 shallow, S6 one stored file is enough, G1 and G1B),
so curation stays in the thesis as the contrast case: a task where the strong model's contribution is small and
cheap to replace. The routing claims rest on the deep task (P2 and P2B).

## Limits

One pool, one target (Wikipedia), one small 2 minute trainer, 6 seeds. A bigger trainer could see differences this
one cannot. The strong judge was never run on every document, so "a perfect judge would not help" is not shown, only
"going from 2,000 to all documents did not help the Gemma judge".

## Shared machines

No kill and no memory error. Peak of my total memory during training: 8.2 GB on spruce, 3.8 GB on willow and
cherry. Judge clients ran under a quiet 8 GB address-space limit (peak 3.4 GB). Model servers on walnut and pine
were stopped after judging.
