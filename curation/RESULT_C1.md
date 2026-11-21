# C1 result: routing policies in a data-curation loop (Nov 11 2026)

Plan, cells, endpoint and decision rules: `curation/design/C1_PREREG.md`, frozen Nov 10 22:49 EST before any loop ran. This file reads the result against that plan. Numbers come from the frozen script (`curation/loop/analysis_c1.py`, output in `curation/runs/C1/analysis.json`).

## Verdict

**SHALLOW.** The anchors differ (G2 passes) but the validity check fails (V): a loop that throws away half of its coder's files could not be told apart from the intact all-strong loop. This is the branch the preregistration predicted, with low confidence.

| Rule | Result |
|---|---|
| G2, anchors differ | SS gain 0.1233, CC gain 0.0171; gap 0.1062 bpb = 14.7 single-run sds; permutation p = 0.014 (smallest possible). **PASS** |
| V, damaged loop is worse | SS minus DH = 0.0536; permutation p = 0.200. **FAIL** |
| H1, SC advantage above 0 | 0.77 (95 percent interval 0.61 to 0.92). Interval above 0, but **not called confirmed** under SHALLOW |
| H2, F1 advantage above 0 | 0.48 (interval -0.03 to 0.99). Interval includes 0. Not confirmed |

What SHALLOW means here, as written before the run: 12 slots are more than this task needs, so the endpoint cannot see a loop wasting half its coder calls. H1 and H2 are reported with intervals but are not statements about routing inside research loops in general. Cells are compared on two plain facts only: did the cell reach the all-strong gain, and at what cost. Loop-level routing claims rest on P2B.

## The run

- 48 loops (4 blocks of 12), all finished with exit 0. None left out, none resumed, no provider or server wait used up a slot.
- Started Nov 10 22:52 EST, finished Nov 11 08:58 EST: about 10 hours on spruce, willow and cherry (one A4000 each), launched with `--per-host 3` (three loops per machine share one GPU through a lock; concurrency is not part of the frozen plan).
- Model spend at list price: about 114 USD (SS about 41, DH about 30, RR about 16, CS and SC about 10 each, E3 about 4, UA 1.7, F1 1.5, HH 0.17, CC 0.05). The preregistered budget was about 170. Charged to the subscription, not billed.

## Result table

Gain = sealed bpb of the raw pool (1.9842) minus sealed bpb of the loop's final file (mean of two confirmation seeds). Dollars = model dollars per loop. Means over 4 loops (8 for RR).

| Cell | Policy | Gain | sd | USD | Accepts | Gap recovered | Extra cost | Advantage (95 percent) | Share of label oracle gain |
|---|---|---|---|---|---|---|---|---|---|
| SS | Fable plans and codes | 0.1233 | 0.0111 | 10.28 | 2.2 | (anchor) | | | 96 percent |
| SC | Fable plans, Gemma codes | 0.1244 | 0.0094 | 2.52 | 2.2 | 1.01 | 0.24 | 0.77 (0.61, 0.92) | 96 percent |
| UA | Fable until first accept, then Gemma | 0.0861 | 0.0570 | 0.42 | 1.2 | 0.65 | 0.04 | 0.61 (0.14, 1.04) | 67 percent |
| E3 | Gemma; one Fable slot after 3 stalled slots | 0.0857 | 0.0502 | 0.99 | 2.0 | 0.65 | 0.10 | 0.55 (0.18, 0.93) | 66 percent |
| F1 | Fable in slot 1, then Gemma | 0.0715 | 0.0616 | 0.37 | 1.8 | 0.51 | 0.03 | 0.48 (-0.03, 0.99) | 55 percent |
| RR | coin flip per role and slot | 0.0827 | 0.0545 | 2.03 | 1.8 | 0.62 | 0.20 | 0.42 (0.07, 0.75) | 64 percent |
| N | no model, random moves on a menu | 0.0463 | 0.0139 | 0.00 | 2.2 | 0.28 | 0.00 | 0.28 (0.13, 0.39) | 36 percent |
| HH | Haiku plans and codes | 0.0402 | 0.0264 | 0.04 | 1.5 | 0.22 | 0.00 | 0.21 (-0.02, 0.43) | 31 percent |
| CS | Gemma plans, Fable codes | 0.0554 | 0.0512 | 2.51 | 2.0 | 0.36 | 0.24 | 0.12 (-0.28, 0.59) | 43 percent |
| DH | SS with half the coder files thrown away | 0.0697 | 0.0754 | 7.39 | 1.5 | 0.50 | 0.72 | -0.22 (-0.93, 0.46) | 54 percent |
| CC | Gemma plans and codes | 0.0171 | 0.0117 | 0.01 | 1.0 | (anchor) | | | 13 percent |

How to read one row, SC. The all-strong loop gains 0.1233 and the all-cheap loop 0.0171, so the gap to recover is 0.1062. SC gains 0.1244, which is 0.1073 above the cheap loop: 0.1073 / 0.1062 = 1.01, the whole gap. SC costs 2.52 USD against 0.01 and 10.28 for the anchors: (2.52 - 0.01) / (10.28 - 0.01) = 0.24, a quarter of the extra cost. Advantage = 1.01 - 0.24 = 0.77. A value of 0 would mean "no better than running some all-strong loops and some all-cheap loops"; 1 would mean "the strong loop's gain at the cheap loop's price".

The intervals are a bootstrap over 4 loops per cell and are known to be too narrow. Eleven cells are compared with no correction for the number of comparisons; only H1 and H2 were named before the run.

## The two plain facts (the SHALLOW reading)

"Reached the all-strong gain" below means a loop gain of 0.107 or more, the lowest of the four SS loops. This cut was chosen after the run and is a description, not a test.

| Cell | Loops at the all-strong level | USD per loop |
|---|---|---|
| SS | 4 of 4 | 10.28 |
| SC | 4 of 4 | 2.52 |
| UA | 2 of 4 | 0.42 |
| F1 | 2 of 4 | 0.37 |
| E3 | 2 of 4 | 0.99 |
| DH | 2 of 4 | 7.39 |
| RR | 4 of 8 | 2.03 |
| CS | 1 of 4 | 2.51 |
| HH, N, CC | 0 of 4 each | 0.04, 0.00, 0.01 |

- **Strong planner with a cheap coder (SC) is the only cheaper policy that reached the all-strong gain in every loop**, at about a quarter of the price. Its four loops (0.1347, 0.1203, 0.1134, 0.1293) overlap the four SS loops (0.1322, 0.1293, 0.1244, 0.1073).
- The reverse split (CS: cheap planner, strong coder) costs the same and got there once in four loops. On this task the planning is what has to be strong.
- One strong slot at the start (F1) or strong until the first accept (UA) is a coin flip: the loop ends at the top when Fable's first file is a good one and stays low when it is not, because Gemma cannot plan its way out afterwards.

## Outcomes are two-valued

A loop either finds a curator that filters the pool by similarity to the target sample (a small classifier, or a filter on word pairs seen in the target sample), which gives a gain of about 0.11 to 0.137, or it does not and ends between 0.00 and 0.07. Cell means mostly count how many loops found it. Per loop (sealed gain, model USD, accepted changes):

| Cell | Loop 1 | Loop 2 | Loop 3 | Loop 4 |
|---|---|---|---|---|
| SS | 0.1322, 6.79, 1 | 0.1293, 14.24, 2 | 0.1244, 8.50, 3 | 0.1073, 11.58, 3 |
| SC | 0.1347, 2.67, 1 | 0.1203, 2.56, 3 | 0.1134, 2.41, 3 | 0.1293, 2.45, 2 |
| CS | 0.0409, 2.70, 2 | 0.1304, 2.65, 3 | 0.0357, 1.04, 1 | 0.0147, 3.65, 2 |
| F1 | 0.1218, 0.65, 1 | 0.0243, 0.25, 3 | 0.1276, 0.23, 1 | 0.0124, 0.34, 2 |
| UA | 0.1342, 0.33, 1 | 0.0720, 0.52, 1 | 0.1270, 0.39, 2 | 0.0114, 0.44, 1 |
| E3 | 0.1175, 1.41, 3 | 0.0616, 0.63, 2 | 0.1363, 1.28, 1 | 0.0273, 0.66, 2 |
| DH | 0.0256, 8.29, 1 | -0.0136, 7.33, 1 | 0.1299, 5.74, 1 | 0.1369, 8.19, 3 |
| HH | 0.0341, 0.05, 2 | 0.0489, 0.04, 1 | 0.0702, 0.04, 2 | 0.0074, 0.04, 1 |
| N | 0.0476, 0, 3 | 0.0582, 0, 2 | 0.0530, 0, 2 | 0.0264, 0, 2 |
| CC | 0.0326, 0.01, 1 | 0.0042, 0.01, 0 | 0.0168, 0.01, 2 | 0.0147, 0.02, 1 |

RR, eight loops (gain, USD): 0.0213, 1.57; 0.1359, 1.49; 0.1117, 2.03; 0.0926, 1.52; 0.1335, 2.80; 0.0199, 1.41; 0.1306, 2.53; 0.0160, 2.85.

## Budget curve

Mean dev gain of the current best after slots 1, 2, 4, 8 and 12.

| Cell | 1 | 2 | 4 | 8 | 12 |
|---|---|---|---|---|---|
| SS | 0.059 | 0.095 | 0.110 | 0.117 | 0.129 |
| SC | 0.000 | 0.000 | 0.059 | 0.122 | 0.124 |
| UA | 0.085 | 0.085 | 0.085 | 0.085 | 0.088 |
| E3 | 0.011 | 0.011 | 0.051 | 0.089 | 0.092 |
| F1 | 0.066 | 0.066 | 0.072 | 0.084 | 0.084 |
| RR | 0.043 | 0.053 | 0.080 | 0.084 | 0.084 |
| DH | 0.047 | 0.049 | 0.082 | 0.082 | 0.082 |
| CS | 0.031 | 0.039 | 0.044 | 0.063 | 0.066 |
| N | 0.029 | 0.029 | 0.032 | 0.052 | 0.056 |
| HH | 0.018 | 0.018 | 0.021 | 0.041 | 0.044 |
| CC | 0.013 | 0.021 | 0.021 | 0.024 | 0.024 |

SC starts late: Gemma needs a few tries to turn Fable's plan into a file that runs and wins. Its first accepted change came in slots 3, 4, 4 and 4, and 9 of its 48 slots produced a file that did not run. By slot 8 it is level with SS. So SC is the best policy at 12 slots and one of the worst at 2: the ranking of policies depends on the slot budget.

Slot outcomes by cell (accepted, rejected, failed to run): SS 9, 39, 0; SC 9, 30, 9; CS 8, 40, 0; F1 7, 32, 9; UA 5, 42, 1; E3 8, 40, 0; RR 14, 76, 6; DH 6, 42, 0; HH 6, 42, 0; N 9, 39, 0; CC 4, 43, 1.

## Predictions against the outcome

| Prediction (written Nov 10, before the run) | Outcome |
|---|---|
| G2 passes | Right |
| V fails, verdict SHALLOW (about 55 percent) | Right. DH is two-valued: two loops at the top, two near zero |
| F1 advantage about 0.5, split in two | Right: 0.48; two loops at 0.12 to 0.13, two at 0.01 to 0.02. In the low two, Fable's first file only reached dev 1.948 and 1.950 |
| UA about the same as F1; E3 about 0.5 | Roughly right: UA 0.61, E3 0.55 |
| **SC well below CS** | **Wrong.** SC 0.77, CS 0.12. Gemma did fail to implement Fable's plans at first (9 failed slots), but 12 slots were enough, and Gemma's own plans were the weaker link |
| CC gain 0.01 to 0.02; N may beat CC | Right: CC 0.0171; N 0.0463 |

## Other observations (descriptive)

- **No model beats two cheap models.** The no-model cell N (random moves on a menu of hand-written curators) gained more than all-Gemma (CC) and slightly more than all-Haiku (HH). The menu was written by hand with knowledge of the task, so N is not a zero-knowledge baseline; it shows that the weaker models add little here beyond a sensible fixed search space.
- **The damaged loop costs less than SS** (7.39 against 10.28 USD) because its files stay shorter when half the edits are thrown away, and calls are priced by length.
- **All-strong cost varies by a factor of two** between loops (6.79 to 14.24 USD) for the same reason: accepted files grow, and every later call carries them.

## Post-run checks

Script: `curation/loop/check_c1.py`, output in `curation/runs/C1/check.out`.

- **Served model.** No call in any loop was answered by a model other than the one the policy asked for. In particular HH really ran on Haiku (the model list gives Haiku a fallback that was never used).
- **Lost slots.** None. One Fable planner call (RR_08, slot 3) got "529 Overloaded" from the provider, was retried after 80 seconds and succeeded; the slot ran normally.
- **Score safety.** All 16 accepted files of the SS and F1 loops were scanned for file, process, network, label and module-internals access, and all 16 were read by hand (the final file of each loop in full, earlier files by their differences). Each uses only the documents, the target sample, the seed variable, the constant T and wall-clock timers. `os` appears only as `os.environ.get("CURATE_SEED", 0)`, `sys` and `json` only to print statistics to stderr. No file reads the hidden labels. No loop is voided.

## What can and cannot be said

Can be said:
- On this task the strong model is needed for planning, not for coding: SC reached the all-strong gain in 4 of 4 loops at a quarter of the cost, CS in 1 of 4 at the same cost. This points the same way as the P2 lead on the thesis's main task (strong planner, cheap coder).
- Paying for the strong model only at the start (F1, UA) buys the full gain in about half the loops for under 0.50 USD.
- Which policy is best depends on the slot budget (SC is behind until about slot 8).

Cannot be said:
- That SC's advantage of 0.77 is a confirmed result about routing in research loops. The validity check failed, so the same endpoint also fails to punish a loop that wastes half its coder calls. A cheap coder that needs three or four tries is, on a 12-slot shallow task, the same kind of waste.
- Anything from differences between the middle cells (UA, E3, F1, RR). With 4 loops per cell and two-valued outcomes, each of those means is a count of two or three loops out of four.
- That the pattern carries over to a deep task. That is what P2B tests.

## Post hoc note on seed noise (Nov 11, after the result above; not part of the preregistered analysis)

A given file on a given seed is reproducible, but some (seed, file) pairs land 0.03 to 0.07 bpb away from the rest, in either direction. Signs of it here: the raw v2 file is worse on seed 12 than on seed 11 by 0.034 on average in the 10 loops that ran both; the two confirmation seeds disagree by 0.050 for CC_04 and 0.070 for DH_02, against a typical single-run sd of 0.007.

Check: the final files of those two loops were scored on two more seeds (1003, 1004).

| Loop | sealed bpb, seeds 1001 / 1002 | seeds 1003 / 1004 | two-seed mean | four-seed mean |
|---|---|---|---|---|
| CC_04 | 1.9443 / 1.9947 | 1.9415 / 1.9664 | 1.9695 | 1.9617 |
| DH_02 | 1.9630 / 2.0327 | 1.9711 / 1.9481 | 1.9979 | 1.9787 |

Seed 1002 was the unlucky one in both. With four seeds the DH cell's mean gain rises by about 0.005 bpb and CC's by about 0.002. The verdict does not change: G2 still passes, and V (SS against DH) fails by a little more. The lesson for later studies: two confirmation seeds are thin, and a four-seed rescoring is a cheap sensitivity check (R1 has one).

## Post hoc note on memory (Nov 11)

Candidate runs in four C1 loops (RR_05 slot 9, SC_03 slot 3, F1_04 slot 6, UA_04 slot 3; six kills in the kernel log) and two in the pilot were killed by the kernel for using all 62 GB of a lab machine; the department's administrator flagged it. Each was scored as a failed candidate, so no result here changes. The candidate runner now has a 16 GB memory cap (see Deviations in `curation/design/R1_PREREG.md`); every final file of C1 runs under it.

## Next

A router trained on these loops, tested live on a new pool (v2b, a second draw from the same recipe as v2; v3 was set aside because its seed-11 starting score is an outlier) against SS, CC and SC, with two learned cells (LA from all loops, LB from the RR loops only). Preregistration: `curation/design/R1_PREREG.md`, frozen Nov 11 09:44 EST before any loop. Because the verdict here is SHALLOW it is labeled a demonstration of the pipeline, not a test of learned routing.
