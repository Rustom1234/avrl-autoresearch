# S6: does memory of past strong runs replace the strong planner?

Step 6 of the routing follow-up program (the stretch step: "train the cheap planner"). Written Nov 11 2026 before any
S6 loop ran. Label: **demonstration**. The task is shallow (C1 verdict SHALLOW), the test pool comes from the same recipe
as the training pool, and every cell has 4 loops.

## Question

C1 and R1 found that the planner is the role that needs the strong model. After a strong planner has worked on a task
family once, can a cheap planner use what it found, so that a later job of the same family does not pay for the strong
planner at all? Three forms of memory are compared:

| Cell | Planner | Memory of the strong runs | Model calls |
|---|---|---|---|
| K | none | the final curate.py of an old strong loop, run unchanged | 0 |
| P0 | Qwen3.5 4B | none (the floor for this planner) | cheap |
| PM | Qwen3.5 4B | in the prompt: the 3 best first-slot ideas kept on pool v2, as text | cheap |
| PW | Qwen3.5 4B + LoRA | in the weights: trained on every idea of the strong planner kept on pool v2 | cheap |

The coder is Gemma 4 12B (on pine) in P0, PM and PW. All three use the same full-precision Qwen3.5 4B served from
cherry, so they differ only in the memory.

## Data and training

- Source loops: every finished loop on pool v2 (B1, C0, C1): 54 loops, 648 slots. R1 loops (pool v2b) are not used.
- Planner prompts are rebuilt from the ledgers by `curation/loop/s6_data.py`. The rebuilt length matches the logged
  length in all 648 slots (constant offset 1149, the system prompt).
- Training set: the 55 slots where the strong planner (Fable) gave an idea that was kept (29 loops). 4 are longer than
  7,500 tokens and are dropped; 3 loops (F1_02, RR_04, UA_04) are held out to report a loss, nothing is tuned on them.
  48 training examples.
- `curation/loop/s6_train_planner.py`: LoRA rank 16 on the attention and MLP layers of the language model, 5 epochs,
  learning rate 2e-4, loss on the reply only. One run; no second attempt with other settings.
- PM memory text: `curation/runs/S6/data/memory.txt`, the 3 kept first-slot ideas with the lowest dev bpb, cut to 1,800
  characters each.

## Test

- Pool v2b (the R1 pool), 12 slots, sigma 0.007266, decision seeds 11 and 12, confirmation seeds 1001 and 1002: the R1
  engine and rules, unchanged (`routed.py` through the wrapper `routed_s6.py`, launcher `launch_s6.py`).
- Plan: `curation/runs/S6/plan.json`, 12 loops: P0, PM, PW x 4, loop seeds 1 to 4, random order inside each block.
- K: the final files of the 4 SS and 4 SC loops of C1 (pool v2) run on v2b, seeds 1001 and 1002 (8 files, 16 runs).
- Anchors are taken from R1 (same pool, engine and seeds, run earlier the same day, **not concurrent**):
  SS gain 0.1213 at 8.39 USD, CC gain 0.0182 at 0.01 USD, SC (Fable planner, Gemma coder) 0.1153 at 2.69 USD.
- Gain = raw sealed bpb minus the final file's sealed bpb, mean of the two confirmation seeds.
- Strong level = sealed gain of at least 85 percent of the R1 SS mean (0.1031), as in R1.

## Rules

- **T1, training helps:** PW mean gain minus P0 mean gain is above 0 with one-sided permutation p at most 0.05
  (4 against 4; the smallest possible p is 0.014).
- **T2, the trained planner replaces the strong one:** at least 3 of 4 PW loops reach the strong level.
- Descriptive, no rule: PW minus PM (weights against prompt) and PM minus P0 with bootstrap intervals; the share of K
  files at the strong level; gap recovered, extra cost and advantage of P0, PM, PW against the R1 anchors. Training cost
  (GPU minutes) is reported next to the advantage, not inside it.

## Reading

- T1 and T2 hold: on a repeated task family the routing table can drop the strong planner once its kept ideas have been
  trained into the cheap one.
- T1 holds, T2 does not: training moves the cheap planner but does not replace the strong one.
- T1 does not hold: 48 examples are not enough, or the coder is the limit.
- If K is at the strong level, say so first: then the cheapest route for a repeated job of the same family is no model
  at all, and PM and PW only matter for jobs the old file does not fit. This study has no such job.

## Predictions (before any S6 result was read)

- K: at least 6 of 8 files at the strong level (80 percent).
- P0: mean gain under 0.03.
- PM: 2 of 4 at the strong level. PW: 2 of 4. T1 holds (55 percent). T2 does not hold (65 percent).
- PW minus PM: no clear difference.

## Limits

Same recipe for the training and the test pool, so this measures reuse inside one task family, not planning skill.
Anchors are not concurrent. 4 loops per cell. One training run. The plain Qwen planner here is the full-precision
model, not the 4-bit one used in B1.

## Shared machines

Every candidate script runs under the 16 GB address-space cap in `curate_runner.py`; one `mem_guard.py` per host
(40 GB limit) runs for the whole study; the training job and the model server run inside systemd scopes with a hard
memory limit (24 GB and 16 GB). The department limit is 50 GB per machine.

## Deviations

1. Nov 11, before any S6 loop ran: the first start of the training job stopped in its first forward pass with a GPU
   out-of-memory error (the output layer was applied to the whole 7,500-token prompt, 3 GB of logits on a 16 GB card).
   No weight update had been made and no adapter was saved. The trainer now applies the output layer to the reply
   positions only. The loss is the same number (it was already counted on the reply only); the data, the 48 examples
   and all settings are unchanged. The run that follows is the one training run.
