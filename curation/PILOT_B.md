# Curation task, Stage B: do the anchors differ? (Nov 10 2026)

Question: on the data-curation task of `PILOT_A.md` (pool v2), does a loop run entirely by the strongest model end clearly better than a loop run entirely by a cheap one? If not, there is nothing for a routing policy to recover and the task is no use for the routing study. Plan and gate were written before the runs: `curation/loop/STAGE_B_PLAN.md`.

## Setup

- Loop: `curation/loop/engine.py`, 12 slots, thesis z rule on dev bpb, sigma 0.007266.
- Cells: SS = Fable plans and codes; QT = Qwen 4B plans and codes. 4 loops each, seeds 1 to 4, run directory `curation/runs/B1`.
- Outcome: sealed gain = raw-pool sealed bpb (1.9842, single-run sd 0.0072) minus the mean sealed bpb of the loop's final file on two confirmation seeds.

## Result

| Loop | Sealed bpb | Gain | Accepted changes (slot: dev bpb) | Model dollars | Wall time |
|---|---|---|---|---|---|
| SS_01 | 1.8611 | 0.1231 | 1: 1.830 | 7.02 | 65 min |
| SS_02 | 1.8607 | 0.1235 | 1: 1.947, 2: 1.917, 5: 1.833 | 9.72 | 99 min |
| SS_03 | 1.8430 | 0.1412 | 1: 1.830, 7: 1.817 | 12.18 | 93 min |
| SS_04 | 1.8562 | 0.1280 | 1: 1.946, 4: 1.917, 9: 1.835, 11: 1.819 | 9.86 | 83 min |
| QT_01 | 1.9801 | 0.0041 | none | 0.01 | 25 min |
| QT_02 | 1.9925 | -0.0083 | 12: 1.948 | 0.01 | 64 min |
| QT_03 | 1.9760 | 0.0082 | 9: 1.954, 12: 1.941 | 0.01 | 68 min |
| QT_04 | 1.9593 | 0.0249 | 8: 1.953 | 0.01 | 55 min |

- SS: mean gain 0.1290 (sd 0.0085), 9.70 USD per loop at list price.
- QT: mean gain 0.0072 (sd 0.0137), 0.01 USD per loop.
- **Gate: gap 0.1218 bpb = 16.9 single-run noise sds; one-sided exact permutation p = 0.014 (the smallest possible value). PASS.**
- No loop lost a slot to a provider or server failure.

Reported without a rule, as planned:

- **Share of the label oracle's gain** (0.129 sealed, from Stage A): SS 100 percent, QT 6 percent. One SS loop (SS_03, 0.141) ends better than the label oracle. The oracle keeps the documents whose hidden label is clean wiki; a learned filter can also rank within and outside that set.
- **Dollars by role in SS loops** (mean per loop): planner 3.38, coder 6.32. Per slot that is about 0.28 for the plan and 0.53 for the code (repair calls included). The estimate before the run was 7.6 per loop; the files grow as changes are accepted (SS_04 ends at 775 lines), and so does the price of each call.
- **What the accepted files do.** Every accepted SS file trains a small text classifier inside `select` (TF-IDF features and logistic regression, with the target sample as the positive examples), handles duplicates, and filters or ranks the pool with the classifier's score. Accepted QT files are short (49 to 73 lines) and use simple hand-written rules.
- **What Qwen does with its slots.** Of 48 QT slots, 28 failed to produce a file that runs and 16 were rejected; 4 were accepted, late and small.
- **Score safety.** All 14 accepted files (10 SS, 4 QT) were scanned for file, process, network, label and module-introspection access, and every flagged line was read. None reads anything but the documents, the target sample, the seed variable and the constant T. SS_01's slot 1 file was read in full.

## How deep is the task?

The loop has 12 slots. How many does the strong model need?

- SS_01 and SS_03 reach the label oracle's dev score (1.830) in slot 1.
- SS_02 gets there in slot 5 and SS_04 in slot 9, each with its third accepted change.
- After reaching it, two loops find one more accepted change (SS_03 to 1.817, SS_04 to 1.819).

So the task is shallow but not one-shot: half the loops are done after one strong call, the other half need a few strong slots. Mean dev gain of the current best after slots 1, 2, 4, 8, 12 is the budget curve C1 will report.

## Can the task be made deeper? (pool v3)

A harder pool was built (`curation/task/build_pool_v3.py`: more kinds of damaged text, more near-domain text) and scored with fixed recipes, two seeds each (`curation/results/v3/`):

| Recipe | Sealed gain over raw v3 |
|---|---|
| best oracle mix (clean wiki and web plus the text sampled from a word trigram model of wiki) | 0.1305 |
| Fable's v2 file from SS_01, unchanged | 0.1281 |
| label oracle, clean wiki and web only | 0.1173 |
| best TF-IDF rule | 0.0801 |
| heuristic | 0.0516 |

Fable's v2 file, never shown v3, is within 0.003 of the best oracle mix. A harder pool does not add depth: with a 300-step trainer, more clean data gives fast diminishing returns, and one good filter finds what matters. Real depth would need the loop to co-design the trainer settings, which is the thesis's first task (P2, P2B), not this one. A deeper pool (v4) and a text-cleaning interface were therefore dropped.

Side result for later: raw v3 is noisy from seed to seed (dev sd 0.0355 over 8 seeds, because each seed draws a different mix of documents), but curated files on v3 are not (pooled seed sd 0.0061 over 13 recipes, against 0.0054 on v2). The v2 sigma can be used for loops on v3.

## Two pilot loops for the main study (`curation/runs/C0`)

Run to test the routed loop (`curation/loop/routed.py`) live, with Gemma 12B as the cheap model. One loop each; not part of any analysis.

| Loop | Policy | Sealed gain | Model dollars | Slots |
|---|---|---|---|---|
| CC_01 | Gemma in both roles | 0.0131 | 0.014 | 1 accept (slot 1), 11 rejects |
| F1_01 | Fable in slot 1, Gemma after | 0.1246 | 0.479 | 1 accept (slot 1), 6 rejects, 5 failed to run |

- Gemma can write curators that run (12 of 12 slots in CC_01) and can edit Fable's long file within its 16k-token window (no context errors), but its edits to Fable's classifier code fail to run in 5 of 11 slots and never improve it.
- With these two loops and the SS mean as anchors, F1 would have gap recovered 0.96, extra cost 0.05, advantage 0.91. One loop per cell: an illustration of the metric, not a result.
- **Bug found and fixed.** CC_01 crashed at its first training run with "run failed (exit 0)" although the result file existed. The shared home directory is NFS: a file written on willow or cherry can stay invisible on spruce for 30 seconds if spruce looked at that directory shortly before (8 of 8 in a direct probe). `routed.py` and `menu.py` now ask the GPU host for the result before a run counts as failed. No B1 slot was affected (no ledger has that error).

## Decision

The gate passes, so Step 3 goes ahead on pool v2. Because the task is shallow, the main study (C1, `curation/design/C1_PREREG.md`) is framed as the **shallow-task arm** of the thesis: its predictions say before the run that one strong call at the start should recover most of the gap, and its validity check (a loop that loses half its coder files) has a preregistered branch for the case where 12 slots are more than the task needs. The deep-task arm is P2B. The cheap model in C1 is Gemma 12B, not Qwen 4B: a model that fails to write running code in 28 of 48 slots measures "cannot code", not routing.
