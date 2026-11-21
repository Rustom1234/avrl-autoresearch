# C1 preregistration: routing policies in a data-curation loop, scored by gap recovered minus extra cost

Status: FROZEN (see the last line). Changes after the freeze go in the Deviations section only.

## Why this study

The thesis asks which model has to be strong in an automated research loop and when. Its main task is training-recipe search (the nanochat loop, studies P2 and P2B). C1 puts the same question to a second task of a different kind, data curation, with more routing policies than P2B has, and with a routed loop's trace stored slot by slot so that a learned router can be trained on it afterwards.

C1 is also written knowing what the pilot showed (`curation/PILOT_B.md`): on this task one call to the strongest model writes a curator at the label oracle's score in two of four loops, and the other two loops get there with three accepted changes. So C1 is expected to be the **shallow-task arm** of the thesis: a task where routing reduces to "pay for a strong call or two at the start". The predictions below say so before any C1 loop runs. If they hold, the contrast with the deep task (P2B) is the finding: how much routing is worth, and which role to pay for, depends on how many useful steps the task has.

## Task and loop (all frozen before C1)

- **Task.** Pool v2 (`data/curation/pool_v2`, 20,746 documents, about 2 T tokens, T = 9,830,400). A loop edits `curate.py`, whose `select(docs, target_sample)` returns the documents to train on, in order. The frozen trainer `curation/task/run.py` (small GPT, 300 steps, T tokens) scores it in bits per byte on target-domain text: a dev set the loop sees and a sealed set it never sees. See `curation/PILOT_A.md`.
- **Loop.** `curation/loop/engine.py`, 12 slots. Each slot: the planner proposes one change; the coder writes the new file (one repair call if it fails to run); one training run on seed 11; then the thesis z rule on dev bpb with sigma = 0.007266 (reject below 0.75, accept at 2.0 or above, in between run candidate and current best on seed 12 and accept at 1.5 on the means). A training run on another machine that reports no result is checked on that machine before it counts as failed (the shared home directory can show a new file up to 30 seconds late; found in the pilot, fixed in `routed.py` before the freeze).
- **Routing.** `curation/loop/routed.py` picks the planner and coder model before every slot from a policy. The policy sees only what the loop knows at that point (slot number, slots left, slots since the last accept, failures in a row, number of accepts).
- **Models.** Strong = Fable. Cheap = Gemma 12B (served locally). Middle = Haiku. Qwen 4B, the cheap model of the pilot, is not used: in the pilot it mostly failed to write a file that runs, so it would measure "cannot code", not routing.
- **Score safety.** A candidate file that touches files, processes or the network is refused before it runs (`lint` in `engine.py`). Every accepted file of an SS or F1 loop is read by hand before results are used; any file that reads the hidden labels voids its loop.

## Cells

Four blocks. Each block has 12 loops: one per cell, two for RR. 48 loops in all.

| Cell | Policy | What it is |
|---|---|---|
| SS | strong plans and codes, every slot | strong anchor |
| CC | cheap plans and codes, every slot | cheap anchor |
| SC | strong plans, cheap codes | the P2 lead (strong planner, cheap coder) |
| CS | cheap plans, strong codes | the reverse split |
| F1 | strong for both roles in slot 1, cheap afterwards | one strong call at the start |
| E3 | cheap; after 3 slots in a row with no accepted change, one all-strong slot, then cheap again | stall-triggered escalation |
| UA | strong for both roles until the first accepted change, cheap afterwards | pay until it works |
| RR | a fair coin per role and slot | random routing; also the router's training data |
| HH | middle model in both roles | middle rung |
| N | no model: one random move per slot on a menu of hand-written curators (`menu.py`) | what the loop does with zero model dollars |
| DH | SS, but in each slot with probability 0.5 the coder's file is replaced by a copy of the current best (the call is still paid) | damaged loop, the validity check |

Loop n of a cell uses seed n (the second RR loop of block b uses seed 50 + b). The seed fixes the documents shown to the planner and the random draws of RR, N and DH. The loop list and its order (shuffled within each block) are in `curation/runs/C1/plan.json`, written once by `launch_c1.py plan`. Loops run on spruce, willow and cherry (one A4000 each, one training run per GPU at a time; which machine runs a loop is decided by the queue, not by its cell); all model calls are made from spruce. The two pilot loops in `curation/runs/C0` are not part of C1.

## Outcome and endpoint

- **Sealed gain of a loop** = mean sealed bpb of the raw pool (1.9842, 8 runs, single-run sd 0.0072) minus the mean sealed bpb of the loop's final file on seeds 1001 and 1002.
- **Dollars of a loop** = model dollars from its call log (Claude models at the CLI's list price, Gemma at 0.15 and 0.60 USD per million tokens in and out).
- **Endpoint, per cell P**, from cell means:
  - gap recovered = (gain P - gain CC) / (gain SS - gain CC)
  - extra cost = (dollars P - dollars CC) / (dollars SS - dollars CC)
  - **advantage = gap recovered - extra cost.** 0 is the straight line between the anchors (what mixing whole SS and CC loops would give); 1 is the strong loop's gain at the cheap loop's price.
- **Intervals.** Bootstrap over loops within cells, anchors resampled too, 10,000 resamples, seed 20261110, 95 percent percentile interval. With 4 loops per cell these intervals are coarse and known to be somewhat too narrow; they are reported as such.
- Analysis script, written and tested on made-up data before launch: `curation/loop/analysis_c1.py`.

## Decision rules

1. **G2, the anchors differ.** gain(SS) - gain(CC) is at least 2 single-run sds (0.0144 bpb) and the one-sided exact permutation test on the 8 loop gains gives p at or below 0.05. If G2 fails the verdict is **NO GAP**: no advantage is computed, and the cells are only described.
2. **V, validity check.** The damaged loop must be worse: gain(SS) above gain(DH), one-sided exact permutation p at or below 0.05.
   - If V passes, the verdict is **SCORED** and the claims below are read as written.
   - If V fails, the verdict is **SHALLOW**. It means the endpoint cannot tell a loop that wastes half its coder calls from one that does not, so 12 slots are far more than the task needs. Then H1 and H2 are still reported with their intervals but are not called confirmed statements about routing inside research loops; cells are compared on two plain facts only (did the cell reach the SS gain, and at what cost); the router experiment that follows is labeled a demonstration of the pipeline; and loop-level routing claims rest on P2B.
3. **H1.** SC has advantage above 0 (lower end of the interval above 0).
4. **H2.** F1 has advantage above 0 (lower end of the interval above 0).
5. Everything else is descriptive, with no pass or fail: CS, E3, UA, RR, HH, N; the budget curve (dev gain of the current best after slots 1, 2, 4, 8, 12 per cell); the share of the label oracle's gain (0.129) each cell reaches; accepted changes per loop.

**Predictions, written before any C1 loop (Nov 10 2026):**
- G2 passes.
- V fails, so the verdict is SHALLOW, with low confidence (about 55 percent). In the pilot the four SS loops reached the label oracle's dev score in slots 1, 1, 5 and 9; a loop that loses half its coder files still has about six good slots, which is enough for the fast loops and may not be for the slow ones.
- F1 advantage about 0.5, and split in two: near 0.9 in loops where Fable's first file reaches the oracle (the one pilot F1 loop: 0.91), near 0.1 where it does not (in the pilot, slot 1 alone gave a dev gain of 0.13 in two SS loops and 0.015 in the other two).
- UA about the same as F1 (Fable's first file was accepted in every pilot loop, so the two policies mostly coincide). E3 about 0.5.
- SC well below CS: Gemma has to implement Fable's plans, and in the pilot its edits to classifier code often failed to run.
- CC gain about 0.01 to 0.02 (one pilot loop: 0.013). N may beat CC.

## Outage and exclusion rules

- A provider or server failure is waited out (up to 30 tries, 2 minutes apart) and does not use up the slot. If a slot is still lost to such a failure, the loop is left out of the analysis and run again from scratch with the same configuration; the lost run is kept and listed.
- A loop stopped by a machine problem is resumed with `--resume`; results of the slot that was cut off are discarded and the slot is run again. Every resume is counted and listed.
- No loop is dropped for its result. No cell, block or slot count changes after the freeze. Any change goes in the Deviations section with its date and reason.

## Budget

About 43 USD per block at list price (SS and DH about 9.7 each as in the pilot, the two RR loops about 9.7 together, CS about 6.3, SC about 3.4, E3 about 2.4, F1 and UA about 0.5 to 0.8 each, HH about 0.5), about 170 USD for the study, charged to the subscription and not billed. About 10 to 12 hours on three A4000s.

## What follows (its own preregistration, written after C1 and before it runs)

A router trained on the C1 loops (the RR loops first) with reward = dev gain as a share of the anchor gap minus slot dollars as a share of the anchor cost gap, tested live on a held-out pool variant (v3) against SS, CC and the best simple rule from C1.

## Deviations

None yet.

Frozen: 2026-11-10 22:49 EST, before any C1 loop was started. First 12 hex digits of each file's SHA-256: `curation/loop/engine.py` db6d551665f5; `curation/loop/routed.py` 20d7ee9b3e47; `curation/loop/menu.py` 4dc7778a1d0e; `curation/loop/launch_c1.py` 97cced4ce832; `curation/loop/analysis_c1.py` 661a659e7f6c; `curation/loop/table.py` 05f722518e85; `curation/task/run.py` eca4e260e2c3; `curation/task/common.py` 118f17466a9d; `curation/task/curate_runner.py` 086d7d09821c; `curation/task/curate.py` 4e4223e181fc; `curation/runs/C1/plan.json` ab559994e3ae.
