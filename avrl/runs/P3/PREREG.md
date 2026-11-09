# P3 preregistration: reasoning effort and the cheap critic's escalation signal

Status: FROZEN when PREREG.sha256 was written (2026-11-05 22:20 local, 2026-11-06T03:20Z), before any P3 Claude call. Changes after this point go in avrl/runs/P3/DEVIATIONS.md with a time and a reason; this file is never edited after freezing.

## Question

AR3 found that Haiku's own stated confidence (judgment_confidence) separates right from wrong critic verdicts far better than agreement between repeated cheap passes. All AR3 passes ran at the CLI's default reasoning effort. Two questions follow, both asked two-sided:

1. **H1.** Does reasoning effort change how well Haiku's stated confidence separates right from wrong verdicts?
2. **H2.** At equal mean dollars, is "escalate the least confident verdicts to Sonnet" (model-first) better than "make Haiku think harder" (effort-first)?

## Substrate

- The 328 AR3 critic items (runs/AR3/items.jsonl, sha256 603c8601709751f09f5df1c86e2feb97d59a5cad062854afb6ba68703dac93f7, matches items.sha256). Prompt and schema are AR3's, imported from avrl/exp/ar3_run_critic.py. Their sha256 must equal runs/AR3/critic_hashes.json (system 4282aeb043041670739bdb72d1af571992122ff1bd0011bd1b664f25997ba29f, schema cc43db81400a6ded66441622c22369855627f1af5befeecf580b1c2bcfb082d7); the runner refuses to start otherwise. The user prompt is the item text unchanged.
- Truth labels, the plus or minus 1 line tolerance and the AR3 weights are AR3's: ar3_features.load_items for split, truth and plant; runs/AR3/items.parquet column w for the weights.
- Model: Claude CLI alias `haiku` (claude-haiku-5-5 in AR3), CLI version 2.1.318, through loop/models/call.py `call("haiku", ...)`. The CLI flag is `--effort`.

## Exclusions and exact counts

Items are selected with the final AR3 split (items.parquet logic), exactly as AR3's final analysis does.

| Step | Items |
|---|---|
| AR3 items | 328 |
| Original natural probes (truth unknown) | minus 6 |
| Controls adjudicated "natural" (a real contradiction was found; not clean, so the single-pass rule cannot score them) moved to probe | minus 23 |
| **Analysed: burn** | **75** (16 entries; 68 plants and 7 clean controls) |
| **Analysed: test** | **224** (48 entries; 190 plants and 34 clean controls) |
| Analysed total | **299** items in 64 entries |

- Plant kinds in test: N-big 47, N-mid 70, N-stale 41, V-body 17, V-coh 15. In burn: 16, 22, 13, 9, 8.
- The plan memo says "322 items, about 970 calls". 322 is 328 minus the 6 original probes; it still contains the 23 natural controls, which cannot be scored. With them excluded and the default level added, the sweep is 299 items x 4 levels = **1,196 calls**, of which 80 are the Day 0 calls.
- Further exclusions at analysis time: an item enters an analysis only if it has a valid Haiku row at every analysed level and a valid stored AR3 Sonnet pass 1 (complete cases). Every AR3 item has Sonnet pass 1. No outlier removal. A call served by any model other than haiku (a fallback) is recorded, counts as not valid, and is redone.

## Design

- **Levels:** default (no `--effort` flag, effort=None), low, medium, high. One Haiku pass per item per level.
- **Calls:** `call("haiku", item_text, system=SYSTEM, schema=SCHEMA, effort=level, attempts=3, timeout_s=600)`. `CLAUDE_EFFORT` and `CLAUDE_CODE_EFFORT_LEVEL` are removed from the environment first (the launching shell had CLAUDE_EFFORT=xhigh). The run starts in an empty temp directory so the CLI loads no project instruction file. Concurrency is the shared 4-call cap.
- **Order:** item-major in a fixed pseudo-random order (sha256 of "p3-order|item_id"), so any partial sweep covers whole items from both splits.
- **Valid row:** call ok, served by haiku, a problems list and an integer judgment_confidence in 0 to 100. Invalid rows stay in the file and the key is retried on the next invocation. The first valid row per key is used.
- **Recorded per row:** usage (including output tokens), dollars (cost_usd_plan), latency, served model, the CLI's modelUsage ids, whether `--effort` was in the argv, CLI version, working directory, timestamp.
- **Dollars** are realized per-call cost from the CLI, including prompt-cache effects. Sonnet dollars are the stored AR3 pass 1 `total_cost_usd`.
- **Day 0 rows are main-run data.** They are the same calls on 20 burn items, so the main sweep skips them.
- Estimated cost of the full sweep: about 1,200 calls at roughly $0.0015 to $0.005, so $2 to $6 nominal.

## Definitions

- **Verdict correct (k=1):** ar3_features.judge on the pass's reported lines. A clean control is correct iff no problem lines are reported. A plant is correct iff a reported line is in its accept_lines.
- **Weights:** AR3 w (pi = 0.25 plant prevalence: each clean control 0.0183, each plant 0.0005 to 0.0022), normalised within the split being analysed. **The weighted statistics are the primary ones** because they are AR3's estimand; the unweighted versions are the pre-declared sensitivity analysis and are always reported next to them. The two weightings can disagree because Haiku false-alarms on most clean controls (see below), so a result is called robust only if both give the same verdict label.
- **AUROC:** score = 100 - judgment_confidence, target = "this level's verdict is wrong", ties count one half (ar3_stats.weighted_auroc).
- **Escalation at fraction q** (q = 0, 0.1, ..., 1): the q fraction of items with the lowest confidence at that level go to the stored Sonnet pass 1. Items tied at the cutoff confidence share the remaining escalations in equal fractions, so frontier values are exact expectations. The rule ranks by confidence only; weights enter only when averaging. Cost = Haiku dollars + Sonnet dollars of the escalated items, accuracy = verdict correctness after the swap.
- **Matched dollars:** model-first is Haiku at level `low` plus escalation; its frontier is interpolated linearly in cost between the 11 grid points. Effort-first is Haiku at level `high` alone (q = 0). The contrast is accuracy(model-first at cost = mean cost of Haiku high) minus accuracy(Haiku high). If the target cost lies outside the frontier the value is undefined (nan) and reported.
- **Bootstrap:** ar3_stats.cluster_bootstrap, 2000 draws, seed 20261106, clusters = entry_key (items from one ledger entry stay together). All statistics use the same draws, so contrasts are paired. Intervals are percentile, two-sided, 90 percent.

## Hypotheses and decision rules (test split, primary)

**H1.** Delta1 = AUROC(high) - AUROC(low), weighted. The effect of interest is an absolute difference of at least 0.05. Labels:
- `high_better` or `high_worse`: the 90 percent interval excludes 0 and |estimate| is at least 0.05. The sign is reported.
- `no_effect_within_margin`: the whole interval lies inside (-0.05, 0.05).
- `inconclusive`: anything else.

**H2.** Delta2 = accuracy of model-first minus accuracy of effort-first at equal mean dollars, weighted. Labels: `model_first_better` if the 90 percent interval is above 0, `effort_first_better` if below 0, `no_difference` if it contains 0.

**Kill note (from the proposal).** If the median output-token ratio high/low is below 1.5 and H1 is `no_effect_within_margin`, the result is reported as "no effort spread, nothing to learn about the trigger". With a real token spread, `no_effect_within_margin` is reported as "the compute knob does not corrupt the trigger".

**Not tested, only reported:** burn split results (separately, no decision), per-level accuracy, AUROC, dollars and tokens, medium and default against the others, the matched-dollar grid for every level pair, frontiers for every level (including escalating after thinking harder). There is no multiplicity correction across H1 and H2; they are two separate questions, each with its own interval. Nothing else is claimed from the data.

## Day 0 gate (the only Claude calls before the main sweep)

- 20 fixed burn items (stratified by kind: control 4, N-big 3, N-mid 4, N-stale 3, V-body 3, V-coh 3; within a kind the first by sha256 of "p3-day0|item_id", listed in DAY0.md) x 4 levels = 80 calls, at most 85 attempts.
- **PASS:** median output tokens at high are at least 1.3 times those at low, and at least 95 percent of the 80 calls return schema-valid output with judgment_confidence. If it fails, the effort arm stops and the main sweep is not run.
- Reported with it: token medians per level, validity, dollars, latency, and whether default behaves like one of the levels.
- The analysis script is run once on the Day 0 data as a smoke test. At n = 20 its numbers are not evidence and are not used for any decision beyond the gate.

## What is already known (disclosure)

AR3 has been analysed, so these facts exist before the P3 data do. All are from the default-effort AR3 pass 1 (Oct 17 run, CLI about 2.1.301) and are not reused in the P3 analysis, which reruns the default level:
- Haiku pass 1 accuracy against truth: test 0.808 (43 wrong of 224), burn 0.893 (8 wrong of 75). On clean controls Haiku is right only 14 to 15 percent of the time, so most wrong verdicts are false alarms on clean entries. Weighted accuracy is 0.32 on test.
- Sonnet pass 1: test 0.830 unweighted and 0.68 weighted. Sonnet is better than Haiku on clean controls (0.57 to 0.62 right) and not better on plants. Model-first can therefore only gain on the clean-control share, and the gain is large only in the weighted view.
- Verbalized confidence of pass 1 against pass 1 correctness (the same definition P3 uses, default effort): weighted AUROC 0.888 on test and 0.912 on burn; unweighted 0.748 on test and 0.862 on burn. The weighted baseline is already high, so the room to improve it is about 0.11, and an effort effect of the size of interest (0.05) would be a large share of that. The weighted and unweighted baselines differ by 0.14 on test, which is why both are reported.
- Power: with about 43 wrong verdicts on test (fewer in the weighted effective sample, ESS about 55), the paired AUROC difference will have a standard error of roughly 0.04 to 0.06. A 90 percent interval narrower than plus or minus 0.05 is unlikely, so `inconclusive` is a likely outcome for H1 unless the effect is large.

## Code path

- Runner: avrl/exp/p3_effort_run.py, sha256 741f92d034bab39c57f2797f9e85a8a047860905f7af2fcf81afc3c308da9879.
- Analysis: avrl/exp/p3_effort_analyze.py, sha256 a4b139f150a46590c89359443372fb6780eb8ab538b7b1a7829c4dfcbcb60542. Run as `.venv/bin/python avrl/exp/p3_effort_analyze.py --splits test,burn --out avrl/runs/P3/analysis.json`.
- Tests: avrl/exp/p3_tests/ (`PYTHONDONTWRITEBYTECODE=1 .venv/bin/python -m pytest -q -p no:cacheprovider avrl/exp/p3_tests`).
- Reused, unchanged: avrl/exp/ar3_features.py (judge, parse_pass, load_items), avrl/exp/ar3_stats.py (weighted_auroc, cluster_bootstrap), loop/models/call.py (effort parameter).
- Sweep command, repeated until nothing is pending: `AR_LOCK_DIR=... AR_BREAKER_FILE=... .venv/bin/python avrl/exp/p3_effort_run.py --items all --max-calls N`.

## Deviations

(none yet; see DEVIATIONS.md if it exists)
