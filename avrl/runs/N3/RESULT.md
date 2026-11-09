# N3 Stage 0 result: why does "cheap passes agree" route the verifier well and the critic badly?

Run 2026-11-05 on spruce, CPU only, zero model calls. Preregistration: avrl/runs/N3/PREREG.md, sha256 21b4803f99749ad7... (file PREREG.sha256), written before any statistic was computed. Code: avrl/exp/n3_signal_stage0.py, 33 unit tests in avrl/exp/n3_tests. Numbers: avrl/runs/N3/results.json, avrl/runs/N3/tables.md. Bootstrap: 10,000 resamples, 90 percent intervals, clusters are prompts (verifier) or ledger entries (critic).

## Verdict: STOP

The preregistered rule gives STOP, and it gives it mechanically. Read it as "Stage 0 could not test the idea with this data", not as "the idea was refuted".

| Test | Result | Why |
|---|---|---|
| Loader gate | pass | p0 AUROC 0.9143141987 reproduced to 1e-6; AR3 `conf_agree` equals items.parquet exactly; the AR3 weighted test-slice AUROCs 0.5777, 0.4228, 0.8575, 0.8925 are reproduced; my cheap decision equals the stored `h3_correct` on all 299 items |
| Strata kept | 3 of 9 | only p0, E06 and E10p0/instruct (all verifier) have at least 40 items, 10 wrong decisions and 10 correct decisions |
| T1 (collision vs ICC) | INCONCLUSIVE | fewer than the 6 strata the rule needs; models deliberately not fitted (3 points cannot carry a 4-parameter model) |
| T2 (reversal across roles) | FAIL | the critic role has no kept stratum, so there is nothing to compare with the verifier |
| T3 (power gate) | POWERED on the 3 kept strata | AGREE AUROC half-widths 0.010, 0.010, 0.058; this says nothing about the critic, which is the part that matters |
| Decision | STOP | T1 is not PASS or T2 is not PASS |

Why so few strata survive, in plain terms: the cheap critic (Haiku, 3 passes) almost never misses a planted defect, so each plant kind has only 1 to 8 wrong decisions out of 49 to 92 items. Its mistakes are false alarms on clean entries: 34 of the 41 clean controls are decided wrongly (83 percent), but that leaves only 7 correct decisions there. Every critic stratum therefore fails one floor or the other. Without my extra "at least 10 correct decisions" floor the clean stratum would be kept, but that makes 5 strata (p0, E06, E10p0 instruct and base, clean), still below 6, so the verdict would be the same.

Wording note: in the original proposal the kill rule says to stop if ICC explains as much as collision. That did not happen here. We stop because the test could not be run, so the collision idea stays open, not dead.

## What the data show (descriptive, after the fact, not part of the verdict)

`a` is pass accuracy, ICC the within-item error correlation, c_w the chance that two wrong passes on the same item give the same output. AGREE is minus the agreement of the first three passes, VERB minus the mean verbalized confidence (critic only). Intervals in brackets are 90 percent. The last three rows pool the critic strata.

| stratum | items | wrong / right decisions | kept | a | ICC | c_w | AUROC AGREE | AUROC VERB | recall at 25% AGREE | recall at 25% VERB |
|---|---|---|---|---|---|---|---|---|---|---|
| p0 (GSM8K 1.5B) | 3000 | 783 / 2217 | yes | 0.64 | 0.29 [0.28, 0.30] | 0.12 [0.11, 0.13] | 0.90 [0.89, 0.91] | n/a | 0.76 [0.73, 0.79] | n/a |
| E06 (GSM8K 0.5B) | 3000 | 1689 / 1311 | yes | 0.40 | 0.33 [0.32, 0.34] | 0.09 [0.08, 0.09] | 0.89 [0.88, 0.90] | n/a | 0.40 [0.39, 0.41] | n/a |
| E10p0 instruct (code) | 464 | 417 / 47 | yes | 0.09 | 0.44 [0.38, 0.48] | 0.24 [0.23, 0.26] | 0.69 [0.64, 0.75] | n/a | 0.27 [0.26, 0.28] | n/a |
| E10p0 base (code) | 464 | 460 / 4 | no | 0.01 | 0.12 [0.05, 0.17] | 0.39 [0.38, 0.41] | 0.26 [0.15, 0.43] | n/a | 0.25 [0.24, 0.25] | n/a |
| N-big | 63 | 4 / 59 | no | 0.93 | 0.79 [0.51, 0.94] | 1.00 [1.00, 1.00] | 0.40 [0.25, 0.63] | 0.53 [0.30, 0.78] | 0.14 [0.00, 0.41] | 0.25 [0.00, 0.67] |
| N-mid | 92 | 8 / 84 | no | 0.92 | 0.86 [0.75, 0.94] | 0.68 [0.46, 0.88] | 0.56 [0.25, 0.86] | 0.66 [0.43, 0.91] | 0.40 [0.01, 0.77] | 0.50 [0.20, 0.86] |
| N-stale | 54 | 4 / 50 | no | 0.93 | 0.65 [0.34, 0.83] | 0.71 [0.43, 0.95] | 0.81 [0.50, 1.00] | 0.42 [0.20, 0.69] | 0.75 [0.33, 1.00] | 0.25 [0.00, 0.67] |
| V (V-body + V-coh) | 49 | 1 / 48 | no | 0.98 | 0.29 [-0.00, 0.37] | 1.00 [1.00, 1.00] | 0.79 [0.71, 0.85] | 0.94 [0.87, 0.98] | 0.63 [0.34, 0.96] | 1.00 [1.00, 1.00] |
| clean controls | 41 | 34 / 7 | no | 0.15 | 0.52 [0.32, 0.67] | 0.65 [0.56, 0.74] | 0.27 [0.06, 0.49] | 0.88 [0.73, 0.99] | 0.15 [0.10, 0.23] | 0.30 [0.27, 0.34] |
| critic, all plants | 258 | 17 / 241 | no | 0.93 | 0.76 [0.62, 0.85] | 0.78 [0.61, 0.91] | 0.59 [0.42, 0.80] | 0.64 [0.51, 0.79] | 0.42 [0.21, 0.68] | 0.38 [0.19, 0.64] |
| critic, all items, unweighted | 299 | 51 / 248 | no | 0.83 | 0.85 [0.80, 0.90] | 0.68 [0.60, 0.77] | 0.55 [0.46, 0.64] | 0.80 [0.74, 0.87] | 0.33 [0.23, 0.44] | 0.67 [0.55, 0.79] |
| critic, all items, design-weighted (75% clean) | 299 | 51 / 248 | no | 0.35 | 0.78 [0.70, 0.86] | 0.65 [0.56, 0.74] | 0.45 [0.34, 0.56] | 0.89 [0.83, 0.94] | 0.21 [0.14, 0.27] | 0.38 [0.33, 0.42] |

Strata with fewer than 10 wrong decisions (N-big, N-mid, N-stale, V) have only a handful of wrong-pass pairs, so their c_w values (some exactly 1.00) and their AUROCs are not reliable. The weighted ICC in the last row treats weights as duplicate counts and is only rough. The same table with the extra columns (c_r, unanimity) is in tables.md.

What stands out:
- **The critic's errors are false alarms, and they are unanimous.** On clean controls 47 percent of wrong decisions have all three passes agreeing, and 89 percent of the unanimous decisions are wrong (ledger line 1689 reports 0.86 on controls, same picture). On GSM8K only 2 percent of wrong decisions (p0 and E06) are unanimous, and a unanimous item is wrong only 2 percent (p0) and 5 percent (E06) of the time. That is why agreement ranks errors well in one place and below chance (0.27 on clean controls, 0.45 design-weighted, 0.42 against truth in the ledger) in the other.
- **Verbalized confidence carries the signal where agreement fails.** Clean controls: AUROC 0.88 against 0.27. Design-weighted pooled critic: 0.89 against 0.45 (the ledger's test-slice numbers are 0.8925 and 0.4228). On clean controls the recall at 25 percent escalation is capped near 0.30, because 83 percent of the items are wrong decisions and only a quarter can be escalated; VERB reaches that cap (0.30), AGREE gets half of it (0.15).
- **Wrong passes collide far more in the critic than in the verifier.** c_w is 0.09 to 0.12 on GSM8K, 0.24 on code, and 0.65 (clean) to 0.68 (all items) for the critic.
- **ICC does not cleanly separate the two stories.** The critic's ICC is also higher (0.52 on clean, 0.76 to 0.85 pooled) than the verifier's (0.29 to 0.44), so with this data both explanations point the same way between roles. The V stratum has p0's ICC (0.29) but cannot be trusted (one wrong decision).
- **Inside the verifier role, collision tracks agreement AUROC, but so does plain accuracy.** Across p0, E06, E10p0 instruct, E10p0 base, the AGREE AUROC is 0.90, 0.89, 0.69, 0.26 while c_w is 0.12, 0.09, 0.24, 0.39 (Spearman -0.8) and ICC is 0.29, 0.33, 0.44, 0.12 (Spearman +0.2). But pass accuracy (0.64, 0.40, 0.09, 0.01) orders them perfectly (Spearman +1.0). With four points that include a stratum with 4 correct decisions, collision and accuracy cannot be told apart.

## What this means

1. H (wrong-output collision is the deciding statistic) is consistent with every descriptive number above, and nothing contradicts it. It is not tested: the preregistered regression had no critic strata to work with, and in the verifier-only data collision is confounded with accuracy.
2. Plant kind is the wrong way to cut the critic data. Nearly all critic decision errors sit in one cell (false alarms on clean entries), so cuts by kind leave kinds with almost no errors. For a Stage 1, strata need at least 10 wrong and 10 correct decisions each; for Haiku k=3 that would take roughly 150 plants per kind and 60 clean entries, far above the 250 items in the Stage 1 sketch. A weaker cheap critic (the 4B to 12B open models in the proposal's Stage 1) would produce errors in both directions and make the floors reachable, but that is a new data collection, not part of Stage 0.
3. The planner role (AR6) cannot enter at all. In CONSIST3 the only label is "Sonnet's pick appears among the three Haiku picks", and Sonnet runs only on iterations where all three Haiku picks differ, so agreement is constant on every labelled row.

## Deviations, simplifications and caveats

Stated in PREREG.md and followed: matched k=3 for signals and decisions; AR6, E08, the E09 family and the "6 percent and 58 percent" points not used; V-body and V-coh merged; natural controls and probes excluded; unweighted strata; difficulty control with oracle sign; Kendall W not evaluated; T2 restricted to strata that have all of a role's signals.

Things that changed or should be known:
- PREREG.md was edited once, before any statistic was computed, to fix a mis-worded sentence about the AR6 traces, and hashed again. The recorded hash is for the final text. The Deviations section of PREREG.md was left unchanged (no analysis step departed from the rules).
- My first run of the script printed T1 numbers (R2 gap 0.90) computed from only 3 strata with a 4-parameter model. Those numbers are meaningless and were removed: the code now declines to fit T1 below 6 strata and reports INCONCLUSIVE. The verdict did not change.
- The "at least 10 correct decisions" floor is my addition to the proposal's floors (an AUROC needs both classes). It matters only for E10p0/base and the clean stratum; the verdict is the same without it.
- Everything under "What the data show" is post hoc and descriptive. The data were already opened by earlier work (AR3 test slice, p0), so even a PASS would have been explanatory, not confirmatory.
- Strata share prompts (p0 and E06 use the same 3000 prompts; E10p0 base and instruct the same tasks) and the critic strata share ledger entries; the bootstrap treats strata as independent.
- A slow independent recomputation (plain loops and sklearn, not the batch code) matched p0 and clean-stratum a, c_w, ICC and AUROC and the p0 recall at 25 percent (0.7612 vs 0.7611 by random tie-breaking).
- No model calls and no GPU were used.
- After the run, PREREG.md was lightly reworded (the stop rule was shortened and the original proposal is now called the proposal). No rule, threshold, test or verdict condition changed. The hash in PREREG.sha256 (21b4803f99749ad7...) was computed on the text before that rewording, so it no longer matches the file byte for byte.
