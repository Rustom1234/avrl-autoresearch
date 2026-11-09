# N3 Stage 0 preregistration: what decides whether "cheap passes agree" is a good escalation signal

Written 2026-11-05 on spruce. CPU only, zero model calls. Frozen when avrl/runs/N3/PREREG.sha256 is written, before any test statistic (accuracy, ICC, collision, AUROC, recall, T1 to T3) is computed. Anything that changes after that goes to the Deviations section at the end, with a reason.

## Question and hypothesis

Agreement among cheap passes ranks the AVRL RL verifier's errors well on GSM8K (AUROC 0.914, avrl/runs/p0) and the AR3 ledger critic's errors badly (0.58 weighted on the test slice against Sonnet disagreement, 0.42 against truth), where the free verbalized confidence reaches 0.86. Why?

Hypothesis H: the deciding statistic is wrong-output collision, c_w = P(two wrong cheap passes on the same item give the same output), not the within-item error correlation (ICC). When wrong answers scatter (GSM8K numbers) a split vote flags a wrong decision. When wrong answers collide (a critic that all three times says "no problem", or all three times flags the same wrong line) a unanimous vote hides a wrong decision.

What is already known, so this stage is not oversold: the effective-sample-size explanation and cross-model disagreement are published (2605.29800, 2607.10139, 2607.25018). All data here were opened by earlier work (AR3 test slice, p0), so Stage 0 is explanatory on seen data. It can reject the idea or justify a live Stage 1 on unopened data. It cannot confirm a mechanism by itself.

## Data (read only)

| Role | Source | What it gives |
|---|---|---|
| verifier | avrl/runs/p0 (Qwen2.5-1.5B, GSM8K, 3000 prompts x 8 rollouts, proxy error 14.5 percent) | answer string, gold flag `correct`, `q_len` |
| verifier | avrl/runs/E06 (Qwen2.5-0.5B, same task, 38.4 percent) | same fields |
| verifier | avrl/runs/E10p0/instruct and /base (code, 464 tasks x 8 rollouts, proxy error 87.3 and 98.7 percent) | execution signature `sig` per rollout, hidden-test flag `correct`, no prompt length |
| critic | avrl/runs/AR3 burn + test calls (5 Haiku passes per item), items.jsonl, control_labels.json, split.json | flagged line sets, `judgment_confidence`, truth, entry key, text length |

Not usable, and why (each is a simplification forced by the data):
- AR6 planner (CONSIST3 arms): three Haiku picks per iteration but no gold for any pick. The only label proxy is "the Sonnet pick appears among the Haiku picks", and Sonnet runs only on the iterations where all three Haiku picks differ (checked from the traces: all 45 escalated Haiku-tier rows and all 12 escalated Qwen-7B-tier rows have a top vote count of 1, and the 42 and 51 kept rows have none of Sonnet's picks to compare with). Agreement is therefore constant on every labelled row and no AUROC exists. The planner role is excluded, so T2 has two roles, not three.
- E08 (MBPP, 48.8 percent): per-rollout outputs were not stored (only the majority signature), so collision cannot be computed.
- E09, E09b, E09c, E09d (6 to 21 percent): free-text answers scored by a reward model, no output identity, no agreement signal. The "6 percent" and "58 percent" proxy-error points named in the proposal text do not exist as saved tables with per-rollout outputs and gold, so they are not used.
- AR3 Sonnet passes are the expensive tier and are not analysed here. AR3 natural controls (23) and the 6 natural probes are excluded because their truth is not clean and not a plant.

## Units, passes, outputs

- Item: one prompt (verifier) or one AR3 item (critic). Bootstrap cluster: the prompt (verifier) or the ledger entry, `entry_key` (critic).
- Cheap pass: one rollout (verifier) or one Haiku pass (critic). Passes per item: 8 (verifier), 5 (critic). Item-level statistics a, ICC, c_r, c_w use all passes. The signals and the cheap decision use the first three passes (rollouts k=0,1,2; Haiku passes 1,2,3), so that every role has the same design and the AUROCs are comparable across roles. This matched-k choice is not the proposal's; the native 8-rollout verifier AUROC is also reported (descriptive only).
- Pass is wrong: verifier, `correct` equals 0. Critic, the pass alone is judged with ar3_features.judge(truth, plant, set(lines), tol=1): a clean item is wrong if the pass flags any line; a plant item is wrong if no flagged line is in `plant.accept_lines`.
- Same output (collision and agreement use one equality): verifier, identical `answer` string (E10p0: identical `sig` list). Critic, ar3_features.sim(lines_a, lines_b) == 1.0, meaning identical line sets up to the project's one-to-one plus or minus 1 matching, and two empty sets are the same.
- Cheap decision on the first three passes: verifier, the plurality class of the three outputs (ties go to the earliest pass, as Counter.most_common does), correct if the class's members have mean `correct` rounded to 1. Critic, ar3_features.supported(lines of passes 1 to 3, 2) judged with judge(..., tol=1), which is the AR3 `h3_correct`. The cheap decision is wrong when it is not correct.

## Strata

role x kind. Verifier strata: p0, E06, E10p0/instruct, E10p0/base. Critic strata: N-big, N-mid, N-stale, V (V-body and V-coh merged because each has fewer than 40 items; the merge rule is "merge within the N or V family any kind with fewer than 40 items"), clean controls. Critic items are the burn and test splits after adjudication (plants plus the 41 clean controls), pooled, unweighted. Within a stratum the AR3 design weights are constant by kind, so no weighting is applied.

A stratum is kept only if it has at least 40 items, at least 10 wrong cheap decisions, and at least 10 correct cheap decisions (an AUROC needs both classes; this last condition is added by me). E10p0/base is expected to fail it (6 of 464 majority decisions were correct, read from groups.parquet before this file was written). I expect 8 kept strata, not the scout's 9. Fewer than 6 kept strata makes T1 INCONCLUSIVE.

## Statistics per stratum (point estimates and bootstrap)

1. a: mean pass accuracy over all passes.
2. ICC: one-way ANOVA ICC(1) of the 0/1 pass-error indicator, balanced design, ICC = (MSB - MSW) / (MSB + (n-1) MSW), n passes per item.
3. c_r and c_w: pooled over items, c_w = (pairs of passes in the same item that are both wrong and give the same output) / (pairs of passes in the same item that are both wrong); c_r the same for both right.
4. Signals, written as suspicion scores (higher means escalate first):
   - AGREE: minus the mean pairwise sameness of the first three outputs (the ar3 `conf_agree`, and for verifiers the fraction of the 3 pairs with equal answers).
   - VERB (critic only): minus the mean of `judgment_confidence` over Haiku passes 1 to 3 (the ar3 `conf_verb`).
   - DIFF (difficulty-only control): item length (critic `n_chars`, verifier `q_len`; not available for E10p0). The sign is the one that gives the higher AUROC in the stratum on the point sample and is then held fixed in all resamples. This favours the control.
5. AUROC of each signal for the cheap decision being wrong (ties count one half).
6. Recall of wrong cheap decisions at 25 percent escalation: escalate the 25 percent of items with the highest suspicion, ties at the boundary included fractionally; recall is the share of wrong decisions escalated.
7. Bootstrap: 10,000 resamples of clusters within each stratum, seed 20261105 plus the stratum index, percentile intervals, 90 percent unless stated. Strata are resampled independently (p0/E06 share prompts and E10p0 base/instruct share tasks; that dependence is ignored).
8. Reproduction checks (a loader gate, not a test): my agreement at k=8 must reproduce avrl/runs/p0 `auc_agreement_only` 0.9143141987 to 1e-6 and groups.agreement exactly; my critic `conf_agree` must equal items.parquet to 1e-9; the pi-weighted AR3 test-slice AUROCs against Sonnet disagreement must reproduce 0.5777 (conf_agree) and 0.8575 (conf_verb). If any fails, stop, report a loader bug, fix only the loader.

## T1 (mechanism): collision versus ICC

Target: the AGREE AUROC of each kept stratum. Two models fitted by ordinary least squares across strata with an intercept: collision model on (a, c_r, c_w), ICC model on (a, ICC). Leave-one-stratum-out (LOSO) prediction for each stratum. R2 = 1 - SSE_loso / SST, with SST around the mean of the target over all kept strata. MAE is the mean absolute LOSO error.
- PASS: MAE of the collision model at most 0.06 and R2(collision) - R2(ICC) at least 0.20.
- KILL: R2(ICC) at least R2(collision) - 0.10 (the idea reduces to the published effective-sample-size story). Stop.
- Otherwise INCONCLUSIVE.
Also reported, not gating: the 90 percent bootstrap interval of the R2 gap (the 10,000 stratum resamples are recombined, one LOSO fit per resample), each stratum's inputs, and the within-role pattern. With about 8 strata and 4 parameters the model is nearly saturated, and two role clusters (verifier high AUROC, critic low) can produce a high R2 for any predictor that separates the roles; the report must say so if it happens.

## T2 (reversal): is there a currency problem

Roles: verifier and critic. Role-level recall at 25 percent escalation is the mean of the stratum recalls over the role's kept strata that have all of the role's signals. Verifier role: p0 and E06 (signals AGREE, DIFF; E10p0 has no DIFF, so it enters T1 only). Critic role: all kept AR3 strata (AGREE, VERB, DIFF). VERB does not exist in the verifier role (p0 and E06 store no verbalized confidence), so a full reversal "VERB beats AGREE in one role and loses in the other" cannot be tested; this is stated as a limit of the data.
- PASS: in the verifier role, AGREE beats DIFF with the 90 percent paired-bootstrap interval of the recall difference above zero; and in the critic role, the signal with the highest recall beats AGREE with the 90 percent paired interval of the difference above zero; and the best signal differs between the two roles.
- FAIL: otherwise.
- The Kendall W kill rule is not evaluated. Only two signals (AGREE, DIFF) are measured in both roles, so W can only be 0 or 1 and carries no information; the proposal itself notes the rule cannot kill. This is declared before the analysis.
Also reported: role-level AUROCs, per-stratum best signal, and the prevalence-weighted pooled AR3 recall (design weights `w`), for comparison with the headline numbers.

## T3 (power gate)

For each kept stratum take the half-width of the 90 percent bootstrap interval of the AGREE AUROC. If it exceeds 0.15 in more than half of the kept strata, Stage 0 is UNPOWERED: no mechanism is claimed from T1, and any move to Stage 1 is only to enlarge n.

## Decision

- PROMOTE (to a live Stage 1 on unopened data, as specified in the proposal): T1 PASS and T2 PASS. If T3 is UNPOWERED the promotion is "to enlarge n only", with no mechanism claim.
- STOP: T1 KILL, T1 INCONCLUSIVE, or T2 FAIL.
The verdict follows these rules mechanically. Stage 1 would additionally need at least 40 entries and 250 items from a corpus that was not opened, a token-logprob or semantic-entropy signal, a cross-family critic, and its own preregistration; none of that is part of this stage.

## Tests and code

Unit tests (avrl/exp/n3_tests) use synthetic strata with known ICC and known wrong-output collision. The analysis script is avrl/exp/n3_signal_stage0.py. Outputs: avrl/runs/N3/results.json and RESULT.md.

## Deviations

(none yet)
