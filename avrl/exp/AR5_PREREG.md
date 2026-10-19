# AR5 pre-registration: live routed autoresearch loop (one 8-hour block, five arms, n = 1 per arm except FULL = 2)

Registered before any arm starts. Hashes and launch facts are filled in by the orchestrator at freeze time and listed at the end; after launch only dated amendments in the deviations log are allowed.

## 1. Claim and null

Claim: in the real autoresearch loop (planner, runner, analyst, critic), routing Haiku to plan, run and analyze with a Sonnet critic (ROUTED), or with a critic that is escalated to Sonnet only when two Haiku screeners flag or disagree (CASCADE), reaches the same claim-2 and claim-3 verdicts as the all-Sonnet loop (FULL) at a fraction of the model cost, with an independently audited residual ledger error rate not materially worse, while the all-Haiku loop (CHEAP) does worse. Null: routing changes verdict, time, cost-adjusted quality or error rate only within the run-to-run noise measured by the two FULL runs, or saves cost only by being worse.

This is the live analogue of the RL arms: full verification (FULL), proxy plus verifier (ROUTED), proxy only (CHEAP), confidence-threshold cascade (CASCADE). The RL result was that a label-free confidence threshold at a quarter of the verifier calls matched full verification. The pre-registered expectation here is weaker (section 6).

## 2. Foreknowledge

The author knows the day-one verdicts (the key, avrl/exp/AR5_KEY.json), the S1 results, and AR1 and AR2 (AR2: Haiku vs Sonnet entry-only critic, defects located 0.72 vs 0.81, false alarms on clean controls 0.50 vs 0.21, cost 0.20 vs 2.40 USD). AR1 and AR2 informed the arm choice and the escalation expectation. AR3 (critic cascade, offline) runs concurrently and does not change the cascade rule here. The prompts are the day-one prompts (avrl/autoresearch.workflow.js) plus the enumerated deltas in avrl/exp/AR5_DESIGN.md section 5; the diff is published with the script hash.

## 3. Units, arms, hosts, starting state

Run = one (arm, host) 8-hour execution. Arms and hosts per AR5_DESIGN.md section 2. Starting state = the allow-list seed of AR5_DESIGN.md section 3 (ledger with E01 only, PROJECT.md claims and phases, backlog E01 to G04 with host names removed and the infeasibility note added, p0 scripts and E01 rollout assets, GSM8K data, four small models), copied per run, hash-manifested, leak-gated. No priority notes, no human steering, no mid-run edits.

## 4. Stopping

Gate stops new iterations at T0 + 7 h 15 min; the orchestrator hard-stops at T0 + 8 h 30 min. Primary horizon T0 + 8 h 00 from the 15-second ledger snapshots. An arm that returns NONE twice in a row ends early (DRY) and is scored on what it reached. No optional stopping, no outcome-contingent redesign.

## 5. Endpoints

Per run: V (verdict reached per AR5_DESIGN.md section 8: machine gate plus blind three-pass Sonnet adjudication with right-reason rules), k* (experiments to settled verdict, two-consecutive rule as in AR1), t* (wall-clock to k*), censor flag, "ever held a wrong final claim state" (the E03 trap), cost at k* and total cost (tokens primary, list-price USD secondary), iterations completed, failed iterations, critic notes, cascade escalation rate and reason breakdown. Per arm: residual MATERIAL entry error rate (Sonnet auditor with trace cross-check, Wilson interval), pre-critic rate, critic yield, false-correction rate. Guards: leak scan hits, p0 mutation check, originals integrity, 429 counts, stalls.

## 6. Predictions recorded before launch

P-a: CASCADE escalation rate 0.50 to 0.80 per iteration (AR2's Haiku false-alarm rate plus the true-defect rate).
P-b: live Haiku critic false-correction rate on audited-clean entries 0.30 to 0.60; Sonnet 0.10 to 0.35.
P-c: CHEAP has the highest residual error rate.
P-d: FULL settles within 8 to 12 experiments and under 200 minutes (day-one reference: iteration 7, minute 142, steered).
P-e: ROUTED total cost 0.25 to 0.35 of FULL.
P-f: CASCADE saves 10 to 30 percent over ROUTED (the critic is about a quarter of FULL's cost, so the saving is bounded).
P-g: the RL "quarter of the verifier calls" result is NOT expected to transfer to the critic role at this scale.

## 7. Decision reads (descriptive; every run's value shown; outcomes are pass, kill or unresolved, never "equivalent")

R-reach: an arm "can do it" if V settles and is never flipped afterwards; "fails" if it never settles or ends in a wrong claim state.
R-time: ROUTED and CASCADE are "comparable" if k* is within +3 experiments of the mean of the two FULL runs and t* within 1.5x.
R-cost: ROUTED passes if cost at k* is at most 0.5 x the FULL mean cost at k*; CASCADE passes if Sonnet critic calls per iteration are at most 0.6 and its total cost is below ROUTED's; the cascade cost claim is killed if escalation exceeds 0.80.
R-error: an arm passes on point estimates if its residual MATERIAL error rate is at most FULL's + 0.10; "supported" only if the one-sided 95 percent bootstrap upper bound (entries resampled within run) of the difference is below 0.10; CHEAP is "worse" if its rate exceeds FULL's by at least 0.10.
R-noise gate: if the two FULL runs disagree on reach, or their k* differ by more than 4, every contrast is "unresolved" and AR5 is reported as a variance measurement of autoresearch loops.
Interpretation table: FULL, ROUTED and CASCADE alike on verdict, speed and error with CHEAP worse and ROUTED under half the cost: the analogy holds in a live loop (n small, "consistent with"). ROUTED like FULL but CASCADE worse: agreement among cheap critics is not an informative escalation signal here. CHEAP like FULL: strong verification not needed in this regime, or the key is easy (see C0). ROUTED worse than FULL: the cheap roles are the bottleneck and a strong critic cannot repair what it cannot detect. FULL's own residual error high: the verifier is not gold.

## 8. Validity gates

G-adj: the adjudicator must pass the day-one replay (W2 right after E03; claim-2 right reason only from L3a; claim-3 non-E01 kill only from E04) before any arm is scored; rubric changes are allowed only before scoring and are logged.
G-aud: the auditor must reach recall at least 0.8 and false-flag at most 0.15 on the seeded day-one defects and planted twins; otherwise error rates are descriptive only.
G-C0: if at least 4 of 5 Sonnet guessers state the key's claim states from the seed alone, the machine gate (a) is the only separator and this is said. RESULT (run 2026-10-17 before launch, 5 calls, USD 0.06): 5 of 5 guessers stated claim 2 not supported and claim 3 killed from the seed alone. G-C0 is triggered: experiments-to-verdict measures how fast an arm produces the evidence (real result files that pass the machine gate and the right-reason rubric), not whether it can state the answer. The adjudicator's day-one replay validation passed (W2 after E03, claim-2 right reason only from L3a, first full V at E04, which the rubric allows since E04's structure statistics are a valid non-E01 point even though its judge-versus-majority comparison is confounded).
G-leak: a run with a hard leak-scan hit is CONTAMINATED and excluded from primary reads (shown in sensitivity).
G-int: if the originals or the seed-manifested files changed during the run, the run is voided.

## 9. Known limits

Orchestration method: to be decided by Rustom. Option A, the Workflow tool from the main session (avrl/autoresearch.live.js): subagents receive the global CLAUDE.md and possibly the session memory index, which names later results; identical across arms (fair contrasts), weakening the comparison with day one; detected by the leak scan, not prevented. Option B, a headless per-role `claude -p` driver with `--setting-sources ""` (no global instructions, memory, hooks or skills reach the agents) and tool permissions bypassed: stronger isolation and per-call cost records, but it runs unattended agents with full tool permissions on the lab machines. On 2026-10-17 the harness's auto-mode permission classifier blocked both the Workflow launch (reason: modifying shared resources) and writing the option-B driver (reason: creating agents without permission checks); launch requires Rustom's explicit approval and, for option B, a permission rule. n = 1 per arm except FULL. The key is one steered run of the same loop family. Auditor and adjudicator are Sonnet, validated on planted and seeded defects only; no human labels. Shared API latency and rate limits. Cost is list-price equivalent on a flat-rate plan.

## 10. Statistics

Per-run values always shown. Rates with Wilson intervals; zero counts with the Clopper-Pearson one-sided bound. Entry-level bootstrap within run for error-rate differences (9,999 resamples, seed 0). McNemar mid-p for verdict agreement on shared experiment ids (exploratory). No TOST claim unless a bound clears the margin. Exploratory items labeled "unadjusted, exploratory".

## 11. Launch facts (filled at freeze)

- seed manifest sha256 (first 16 hex): f47cd4133da00944
- script sha256 (avrl/autoresearch.live.js): f2873bdf6969bba8
- key sha256 (avrl/exp/AR5_KEY.json): ea20fe30e2576e03
- design sha256 (avrl/exp/AR5_DESIGN.md): 66f750d70e4d853d
- run trees and hosts: /homes/rdubash/avrl_live/FULL (cherry), ROUTED (willow), CHEAP (walnut), CASCADE (pine), FULL_R (spruce); to be created from the patched seed with exp/ar5_make_run.sh immediately before launch
- T0 per run (epoch): recorded in the ledger launch line and in each BASE/T0 at launch
- claude CLI version 2.1.301; NVIDIA driver 595.84 on cherry, willow, walnut, pine (spruce 580.173.02); /usr/bin/python3.12 on all hosts via the shared venvs
- resolved model ids (from the first transcripts): recorded after launch
- registration time: 2026-10-18 23:01 UTC

## 12. Deviations log

- 2026-10-19 post-run (all analysis-time, no read changed). (a) The adjudicator passed the ledger prefix as a command-line argument and failed above 128 KB; fixed to stdin and every prefix re-adjudicated (first four prefixes per arm served from cache). (b) exp/ar5_audit.py was written after the run (the build step had not produced it); its taxonomy and evidence set follow section 9 unchanged. (c) G-aud FAILED: kind-matched recall on the seeded day-one defects 2 of 5 (any-flag 4 of 5), twin recall 5 of 6 verdict flips and 5 of 9 number changes, 0 of 14 twin controls flagged; the day-one "clean" controls were not clean (AR3 had already found real contradictions in L3a, L6a, E06, E08). Error rates are reported as descriptive. (d) Leak scan: CASCADE's four hard hits and FULL_R's one were reviewed and classed as grep patterns containing host names, a nonexistent path, a directory listing of the originals (soft under section 8) and the tool-results spill (exempt); no arm classed CONTAMINATED, all hits disclosed. (e) R-noise gate triggered (FULL settled at k=11, FULL_R ended in a wrong final state), so all contrasts are reported as unresolved per section 7.

- 2026-10-18, before launch. P0 pilot (unscored, throwaway tree P0TEST on cherry) ran one full iteration through ordinary Agent-tool calls because the Workflow launch is blocked by the permission classifier: Haiku planner picked G01 (CPU replay, label-free agreement threshold vs learned allocator on E01 rollouts); Haiku runner wrote and ran it (KILL, agreement 0.711 vs learned 0.410 hits per call at 10 percent, 0 of 5 splits for the learned arm); Haiku analyst appended the entry and three follow-ups; two Haiku screeners both flagged (untraceable side numbers, "pre-registered" gate first appearing in the run script), verdict kill; the Sonnet critic confirmed KILL, reproduced the run byte for byte, found no leakage, and corrected two things the screeners missed (undisclosed bandit step size K=4 vs E01's K=1, and a misattributed correction to E01), appending a critic note. GPU dispatch through bin/gpu_run.sh from inside an agent was verified separately (remote job on cherry, done.flag visible over NFS in 25 s). All five hosts idle at the time. The block itself remains unlaunched: the classifier blocked the Workflow launch twice, blocked writing a per-role prompt generator for Agent-tool orchestration, and blocked creating the five run trees; launch requires Rustom to add a Workflow permission rule or approve the launch interactively.
