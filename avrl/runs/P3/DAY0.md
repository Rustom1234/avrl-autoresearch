# P3 Day 0 gate

Run 2026-11-06 03:21 to 03:25Z (2026-11-05 evening local), CLI 2.1.318, after PREREG.md was frozen (PREREG.sha256 still verifies, runner and analysis code unchanged since the freeze).

## Verdict: PASS

| Gate condition | Needed | Observed |
|---|---|---|
| Median output tokens, high over low | at least 1.3 | **2.21** (3,368 vs 1,523) |
| Calls with schema-valid output and judgment_confidence | at least 95 percent | **100 percent** (80 of 80) |

The effort flag works on Haiku. The main sweep may go ahead. It was not started.

## What was run

- 20 fixed burn items x 4 levels (default, low, medium, high) = 80 calls. 80 calls, 80 CLI attempts (no retries), no fallback to Sonnet, every row served by claude-haiku-5-5 with `--effort` in the argv exactly for low, medium and high.
- Total cost **$0.1202** nominal (limit was 85 attempts).
- Rows are in avrl/runs/P3/calls_<level>.jsonl and count toward the main sweep (as preregistered). The per-attempt log is calllog.jsonl.
- The 20 items, chosen by the preregistered rule (burn split only; control 4, N-big 3, N-mid 4, N-stale 3, V-body 3, V-coh 3; within a kind the first by sha256 of "p3-day0|item_id"). Listed in run order:

| item_id | entry | kind | characters |
|---|---|---|---|
| cf0bfd89a4fb | 57c428b3 | N-stale | 2294 |
| 2ecf9dc70e9a | bd4ac8f1 | V-body | 4419 |
| bc5c21b24185 | 57c428b3 | N-mid | 2294 |
| 9bcad986bad3 | fc48477d | N-mid | 2846 |
| a9051f9ea3b2 | 73fa7305 | V-coh | 1576 |
| 9ed8e8f06b8e | df49996f | control | 1354 |
| 6cb1af4fde9f | cce6693a | V-body | 11293 |
| 7b72a98dc725 | d6752f43 | V-coh | 4428 |
| 9cd5f450a225 | 3c581177 | N-big | 1909 |
| 3e13b97d509e | cce6693a | N-stale | 11293 |
| 75a5b2da0b11 | 845e2e09 | N-stale | 5443 |
| 5acc6a7fa449 | 57c428b3 | control | 2294 |
| 8fa516ca6ec8 | bd4ac8f1 | N-big | 4419 |
| 62f2a59d0c72 | 1ec2581f | N-mid | 1640 |
| 99592bc6af98 | 845e2e09 | V-coh | 5443 |
| c3386b69db86 | fc48477d | V-body | 2846 |
| 15fc3a7b3d55 | 3c581177 | control | 1909 |
| 70ebc06923e1 | f512a405 | N-big | 6568 |
| 728a4a94bfe5 | 73fa7305 | control | 1576 |
| 13559df84ef1 | 845e2e09 | N-mid | 5443 |

## Numbers per level (20 items each)

| Level | Valid | Output tokens median (mean) | Thinking tokens median | Mean $ per call | Median latency s | Mean confidence | Mean problems reported |
|---|---|---|---|---|---|---|---|
| default | 20/20 | 1,854 (2,113) | 1,528 | 0.00156 | 9.3 | 67.8 | 2.35 |
| low | 20/20 | 1,523 (1,691) | 1,172 | 0.00107 | 7.6 | 72.6 | 2.05 |
| medium | 20/20 | 2,040 (2,481) | 1,673 | 0.00131 | 9.6 | 67.8 | 2.55 |
| high | 20/20 | 3,368 (3,890) | 2,868 | 0.00207 | 15.0 | 67.6 | 2.80 |

- **Effort spread is real and monotone.** Per item, high used more output tokens than low in 20 of 20 items (median per-item ratio 2.06, range 1.41 to 3.97); medium used more than low in 19 of 20 (median 1.37). Most of the extra output is thinking tokens (reported in `usage.output_tokens_details.thinking_tokens`, kept in each row's raw_usage).
- **Low is not "no thinking".** Haiku at low still spends a median of 1,172 thinking tokens on this task. The spread is therefore a factor of about 2 between low and high, not a contrast between thinking and not thinking.
- **Dollars rise more slowly than tokens** because input and cache costs are fixed: high costs 1.9 times low per call, medium 1.2 times. Dollars per 1,000 output tokens fall from 0.00063 (low) to 0.00053 (medium, high).
- **Default behaves like medium.** Median output tokens default/medium = 0.91, default/low = 1.22, default/high = 0.55. Closest level by median ratio: medium, within 25 percent. Per item, default used more tokens than medium in only 7 of 20 items. This is a token-count finding; whether confidence also matches is for the main run. Default and medium are therefore close to duplicates, and the analysis keeps both.
- **Same context as AR3.** Total input tokens per call (input + cache read + cache write) have a median of 2,753, against 2,707 for the same items in AR3's Haiku pass 1 (+45 tokens, identical across the four levels). The runner starts in an empty temp directory, so no project instruction file was loaded. The new default-level output tokens (median 1,854) are close to AR3's pass 1 on the same items (median 1,953).
- **Cache effects add noise to dollars.** The first call per prompt writes the cache (about 2,500 input tokens) and later calls read it, so per-call dollars are not only a function of output tokens. The preregistered analysis uses realized dollars as they are.

## Analysis smoke test

`.venv/bin/python avrl/exp/p3_effort_analyze.py --items day0 --splits burn --out avrl/runs/P3/day0_smoke_analysis.json --day0-gate` ran end to end in 14 seconds (B = 2000). Output: avrl/runs/P3/day0_smoke_analysis.json. At n = 20 (11 entries, 4 to 6 wrong verdicts per level) the numbers are not evidence and are not used for any decision; for the record, the weighted H1 estimate was 0.036 [-0.008, 0.094] (`inconclusive`) and the weighted H2 estimate 0.032 [0.000, 0.076] (`no_difference`). The matched-dollar target was reached by escalating only about 5 percent of items, because Haiku high costs about $0.001 more per call than low and a Sonnet pass costs about $0.018.

## Plan for the main sweep (not run)

- 1,116 calls remain (1,196 minus the 80 above). At the Day 0 per-level means that is about $1.8 nominal; at about 10 s per call with 4 concurrent slots about 50 minutes of wall time if the shared slots are free.
- Run in chunks, for example `AR_LOCK_DIR=... AR_BREAKER_FILE=... .venv/bin/python avrl/exp/p3_effort_run.py --items all --max-calls 300`, repeated until the summary shows `pending` 0, then the analysis command in PREREG.md.
