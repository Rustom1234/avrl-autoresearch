# Curation pilot, Stage B: do the two anchors differ on this task?

Written before any Stage B loop ran (Nov 10 2026). It is a pilot, not the main study: its only job is to decide whether the data-curation task can tell a strong loop from a cheap loop.

## What runs

- **Task.** Pool chosen by Stage A (see ../PILOT_A.md). The loop edits `curate.py`; the frozen trainer (`../task/run.py`, 300 steps, T tokens) scores it. Engine: `engine.py`.
- **Loop.** 12 slots. Each slot: the planner proposes one change, the coder writes the new file (one repair call if it fails), one training run on seed 11, then the thesis z rule on dev bpb (reject below 0.75, accept at 2.0 or above, in between run both on seed 12 and accept at 1.5 on the means). Sigma is 1.15 times the sd of the 8 raw-pool pilot runs on dev.
- **Outcome.** Sealed gain of a loop: mean sealed bpb of the raw pool (8 pilot runs) minus the mean sealed bpb of the loop's final file on seeds 1001 and 1002. The loop never sees the sealed set.
- **Cells.** 4 loops each; loop n of every cell uses seed n.

| Cell | Planner | Coder | Role |
|---|---|---|---|
| SS | fable | fable | strong anchor |
| QT | qwen35-4b | qwen35-4b | cheap anchor |

- **Why Fable and not Sonnet as the strong model.** Sonnet's rate limits are in use by the P2B replication, and the pilot asks whether a wide model gap shows up on this task at all. The main study (Step 3) will name its own anchors.
- **Cost.** Model dollars per loop from `calls.jsonl` (Claude at the CLI's list price, the open model at its nominal price).
- **Safety of the score.** A candidate file is refused before it runs if it touches files, processes or the network (`lint` in `engine.py`), because the pool's hidden labels sit next to the documents. Every accepted file is also read by hand before the result is used.
- **Provider failures.** A failed call that is the provider's or the server's fault is retried after a wait and does not use up the slot; the count is stored per loop.

## Gate (fixed now)

Step 3 goes ahead on this task only if both hold:
1. gain(SS) minus gain(QT) is at least 2 single-run noise sds of sealed bpb, and
2. the one-sided exact permutation test on the 8 loop gains gives p at or below 0.05 (the smallest possible value is 1/70).

If the gate fails, the task is changed (harder pool, more slots) or dropped, and that is reported. Also reported, without a rule: each cell's gain as a share of the Stage A oracle gain, dollars per loop, and what the accepted files do.

Command: `python curation/loop/table.py --root curation/runs/B1 --pool <pool>`
