# BG1 result: two real tasks at last, and Haiku is as good as Sonnet on both

Nov 15 2026. Plan: `loop/design/BG1_PREREG.md` (frozen before the run; one deviation). Code: `loop/bench/`.
Numbers: `loop/runs/BG1_R/finish.log`, `loop/runs/BG1_R/confirmation/summary.json`.

## Why this was run

The GPT task was too easy (a random menu matched the strong loop) and the CIFAR speedrun gave gains near the noise.
BG1 tests two task types that other autoresearch papers use: drug-property prediction on four public benchmark
tasks with human leaderboards (as in AutoScientists), and circle packing with an exact score (as in AlphaEvolve and
ShinkaEvolve). It is a gate on the tasks, not a routing result.

## What was run

The same loop engine as before, 40 slots per loop, CPU jobs on the lab machines. Each loop starts from a simple
script (Morgan fingerprints with a random forest; random restarts of a standard optimiser).

| Cell | Who proposes the edits | Loops |
|---|---|---|
| N | random search over the five settings of the starting script, no model | 14 |
| SS | Sonnet as planner, coder and repairer | 14 |
| HH | Haiku in those roles | 14 |

After the loops every final script was scored on data the loop never used: the benchmark's own test molecules
(5 seeds), or 5 fresh seeds for packing.

progress = (start - final) / (start - reference), where the reference is first place on the leaderboard or the best
published packing. 0 is the start, 1 is level with the reference. Example: caco2 starts at an error of 0.413, the
reference is 0.256, a final script at 0.279 has progress (0.413 - 0.279) / (0.413 - 0.256) = 0.85.

## Result

| Task | Start | Reference | Random search | Sonnet loops | Haiku loops |
|---|---|---|---|---|---|
| bbb (AUROC, higher is better) | 0.851, rank 20 of 27 | 0.924 | 0.856, 0.856 | 0.911, 0.911 (rank 7) | 0.898, 0.909 (rank 13, 8) |
| caco2 (error) | 0.413, rank 18 of 25 | 0.256 | 0.396, 0.406 | 0.279, 0.278 (rank 4) | 0.264, 0.263 (rank 2) |
| clearance (Spearman, higher is better) | 0.283, rank 17 of 19 | 0.536 | 0.441, 0.430 | 0.460, 0.482 (rank 4, 3) | 0.430, 0.438 (rank 10, 9) |
| lipophilicity (error) | 0.713, rank 21 of 22 | 0.456 | 0.673, 0.673 | 0.493, 0.494 (rank 5) | 0.477, 0.465 (rank 4, 2) |
| 26 circles (sum of radii) | 2.62689 | 2.63598 | no change, 3 loops | 2.63261, 2.63177, 2.63158 | 2.63484, 2.63138, 2.63430 |
| 32 circles | 2.92540 | 2.93700 | no change, 3 loops | 2.92402, 2.93406, 2.93727 | 2.93619, 2.93428, 2.93424 |

| Family | Mean progress N | SS | HH | Model cost per loop, SS / HH |
|---|---|---|---|---|
| Drug tasks (8 loops per cell) | 0.23 | 0.82 | 0.80 | 1.60 / 0.45 USD |
| Packing (6 loops per cell) | 0.00 | 0.56 | 0.77 | 1.83 / 0.49 USD |

| Check | Rule | Drug tasks | Packing |
|---|---|---|---|
| A headroom | mean progress(SS) at least 0.25, p < 0.05 | 0.82, **pass** | 0.56 (p = 0.008), **pass** |
| B menu | N at most a third of SS, p < 0.05 | ratio 0.28 (p = 0.001), **pass** | ratio 0.00 (p = 0.010), **pass** |
| D gap | HH at most two thirds of SS | ratio 0.98, **fail** | ratio 1.39, **fail** |

Predictions beforehand: drug A 0.85, B 0.7, D 0.3; packing A 0.5, B 0.8, D 0.4. All six calls went the likelier way.

**Outcome by the plan's table: D fails in both families, so there is no room for routing between Haiku and Sonnet
on these tasks. That is the result.**

## What happened inside the loops

- **These are real tasks.** Every model loop accepted at least one edit and made held-out progress, with one
  exception (a Sonnet loop on 32 circles ended slightly below the start on fresh seeds, 2.92402). Random search
  moved little on the drug tasks, apart from clearance, and nothing on packing.
- **Drug tasks: the loops rebuilt the public recipe.** Accepted edits: RDKit descriptors next to the fingerprints,
  gradient boosting (LightGBM, XGBoost, CatBoost) in place of the forest, Tanimoto-kernel models, blends and stacks,
  rank-transformed targets for the rank metric, fitting the residual of a simple logP baseline. All standard
  practice. From rank 17 to 21 the scripts land at rank 2 to 13, without pretrained models.
- **Clearance is the one drug task where a settings search goes far** (0.60 of the way), and the one where Sonnet
  leads Haiku (0.74 against 0.60).
- **Packing: both models found the same family of ideas,** perturbing the best packing instead of restarting from
  scratch (basin hopping, remove and reinsert the smallest circle, staged radius caps). One Sonnet loop on 32
  circles scored 2.93727 on fresh seeds, level with the reference as I rounded it (2.937). Checked Nov 15: the best
  published value for 32 circles is 2.93957, so this is not a record.
- **Haiku at about a quarter of the model cost did as well or better** in five of six tasks.

## What this means for the thesis

1. Four task types tried. Planner routing between a cheap and a strong model paid only on the GPT task, where the
   strong model knew two moves in advance and a menu could find them too. On tasks that resist the menu (CIFAR, drug
   properties, packing) the cheap frontier model matches the strong one.
2. The gate does work as a task test: B passing in both families says the loops do something a settings search
   cannot. The gap that is left to study is between Haiku and small open models, not between Haiku and Sonnet.
3. The proposals are again recall of known practice, combined well. No loop produced an idea that is new to the
   field.

## Limits

Two loops per cell and task on the drug tasks, three on packing. Small test sets (182 to 840 molecules). Leaderboard
ranks compare our 5-seed mean with published entries and the loops never saw the test set, but the leaderboards
include entries with pretrained models that were not allowed here. No pass rule was set for accepted ideas; the
labels are mine, from the ledgers.

## Run notes

- **Deviation 1 (in the plan).** The command-line login used for Claude calls expired at 12:38 EST Nov 14. 13 loops
  ran with no model and 10 lost part of their slots. Every loop with a failed login call (23) was rerun whole in
  `loop/runs/BG1_R` (18:30 Nov 14 to 04:03 Nov 15); the 19 untouched loops were kept. The first scoring, run before
  the failure was noticed, is void. The rerun has no failed model call.
- 22 slots across three rerun loops were labelled infra and not charged (17 in bbb_SS_2); those loops also wrote
  scripts that hit the 600 s job limit. Every loop has 40 charged slots.
- Shared machines: nobody else logged in. No kill and no memory error.
