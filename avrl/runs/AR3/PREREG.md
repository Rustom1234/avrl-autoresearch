# AR3 pre-registration: critic cascade (Haiku passes with agreement-based escalation to Sonnet)

Registered before any non-smoke model call. The design is avrl/exp/AR3_DESIGN.md (frozen; its sha256 is listed at the end). Pool counts and hashes are filled at registration; after that only dated amendments in the deviations log.

## 1. Claim and null

RL claim 1 (the thesis): a label-free threshold on the cheap grader's own confidence matches full expensive verification at a quarter of the expensive calls. AR3 tests the analogue on the critic role of the autoresearch loop. Cheap critic = Haiku with k=3 passes per ledger entry; confidence = agreement among the passes (the analogue of response consistency); expensive critic = Sonnet, single pass; cascade = escalate the lowest-agreement quarter of entries to Sonnet; burn-in = 25 percent of entries fully verified, used to set the threshold and to run the regime diagnostic.

Claim: at escalation rate 0.25 the cascade is (R1) better than random escalation at the same rate and (R2) non-inferior to all-Sonnet within M=0.05 on prevalence-weighted critic quality. Null: agreement is no better than random, or the cascade is measurably worse than Sonnet.

## 2. Foreknowledge

AR2 (yesterday) ran the same critic task with contaminated controls, string-match scoring, one planting seed and one pass per model; its numbers (located 0.72 Haiku, 0.81 Sonnet; false alarms 0.57 and 0.31 on all controls, 0.50 and 0.21 on the clean subset) are preliminary and are re-derived here, not used as priors. A read-only audit of AR2 found: 21 of 52 entries carry real corrections in the analyst text, controls were not stripped of them, 9 of 42 verdict plants flipped a rule keyword rather than the verdict, and some number plants landed in leftover critic text. The design was written knowing these results. Prior beliefs recorded for later calibration: AUROC of conf_agree for proxy_error_sonnet 0.70 to 0.82; conf_verb 0.52 to 0.62; diagnose's recommended budget 0.35 to 0.60; most likely outcome INCONCLUSIVE or partial (signal supported, non-inferiority unresolved at this n).

## 3. Units, substrate, split

Item = one rendered entry variant (control or planted); cluster = ledger entry. Substrate per AR3_DESIGN.md section 3: mechanical pool from the ledger snapshot, critic text stripped, five plant kinds (N-big, N-mid, N-stale, V-body, V-coh), six natural-defect probes as a separate descriptive stratum with truth hand-extracted before any call (avrl/exp/ar3_natural_truth.json), line-numbered rendering, split by entry with salt "ar3-2026-10-17" (first 25 percent burn-in, rest test). Controls are clean unless a blind adjudication (by the orchestrator, Claude Fable 5.1, reading the review pile without model, pass or split labels) finds a real contradiction with the entry's own table or thresholds; such controls become natural defects and leave the control pool. Sensitivity: all controls clean; and the mechanical rule "drop controls flagged by the majority of both tiers".

## 4. Models and procedure

Aliases haiku and sonnet via claude -p with the flags in AR3_DESIGN.md section 4; resolved model ids recorded per call; API-default sampling. 5 Haiku passes and 3 Sonnet passes per item, burn-in items first. Frozen prompt and schema (sha256 at the end). No prompt selection on any data. Fixed item set; no item added or removed after outcomes; no optional stopping. Hard cost stop $80; abort if more than 5 percent of calls fail. Contingency if collection runs long: drop the second seeds of N-stale then N-mid from the not-yet-run queue (logged); never partial entries in the analysis.

## 5. Burn-in and freeze

On the burn-in split only: diagnose (avrl_router/diagnose.py) on the pi-weighted replicated frame for conf_agree, conf_vote, conf_verb, conf_verb1 against proxy_error_sonnet (rel:0.25 target; recommended budget b* reported as a diagnostic, with an entry-level bootstrap interval); the threshold (t*, p*) = weighted quantile of conf_agree at e=0.25; early gates (Sonnet macro located recall on burn plants at least 0.50, Sonnet false alarms on burn controls at most 0.40, else NOT TESTABLE and stop). frozen.json and its sha256 are written and appended to the ledger before any test row is opened.

## 6. Gates and reads on the test split (frozen operating point e=0.25)

G0: pool at least 50 entries; at least 95 percent valid calls; Sonnet single-pass macro located recall on test plants at least 0.60; Sonnet false-alarm rate on adjudicated test controls at most 0.35. G1: Q_S minus Q_H3 at least 0.06 with one-sided 95 percent cluster-bootstrap lower bound above 0 (if all-Haiku k=3 already satisfies non-inferiority, R2 is vacuous and is said so).
R1: Q_C minus Q_rand(e_hat) with one-sided 95 percent cluster-bootstrap lower bound above 0 (companion sign-flip permutation p over entries).
R2: Q_S minus Q_C with one-sided 95 percent upper bound below 0.05.
PASS = G0, G1, R1, R2. KILL = K1 (two-sided 90 percent CI of the gap closed at e=0.25 has upper end below 0.25) or K2 (one-sided 95 percent lower bound of Q_S minus Q_C above 0.05). Otherwise INCONCLUSIVE, reported with MDE and q = N/N* and the word "unresolved", never "matches".
Qualifier: any kind with at least 15 test items and a confident-wrong rate of at least 0.50 is named as an exception.
Secondary family (Holm, m=3): S1 AUROC(conf_agree) above AUROC(conf_verb) for proxy_error_sonnet; S2 the learned scorer does not beat the threshold at e=0.25 (upper bound of Q_P7 minus Q_P3 below 0.02); S3 at equal dollars the cascade beats all-Haiku k=5 majority.
Diagnostic (no read): b*, Q and e at b*, diagnose's budget-versus-error table on test against random.
Exploratory (unadjusted, no claims): k in {1,3,5}; conf_vote; escalate-on-flag; spurious controls (length, flag count); per-kind recall; tolerance, pi and M sensitivity; natural probes; 1,000 random entry-level splits; Trust-or-Escalate fixed-sequence scan; oracle versus extra stratum; A/A noise floor.

## 7. Power

Expected test slice 40 to 52 entries, 300 to 400 items; the control side dominates. Expected standard error of Q_S minus Q_C about 0.03 to 0.04, so non-inferiority at M=0.05 has roughly 35 to 55 percent power at a true difference of zero; the MDE for inferiority is about 0.07 to 0.08. AR3 is an estimation experiment with pre-registered reads, not a high-power trial. INCONCLUSIVE is a resolution limit, not equivalence.

## 8. Boundary conditions

This model generation (aliases at call time); this ledger, written by Sonnet analysts; entry-only critique (the live critic reads raw files); synthetic plants plus six natural probes; one prompt; API-default sampling; agreement among passes of one model shares that model's bias, so unanimous-wrong cases are reported by kind. Wording rules: never "matches Sonnet" without the noise ceiling and interval; never accepted-item accuracy without coverage; never a single threshold without the curve; never a null read as equivalence.

## 9. Registration facts (filled at freeze)

- AR3_DESIGN.md sha256: 885a2941d29c984bee918a570ae920170723928f833921ada4aaa5c849627214
- ledger snapshot sha256 and line count: 2fefbee1ba3995dc23763fdfd86552b687b0ca057f21f92577729035ed9a9c16, 1662 lines
- items.jsonl sha256: 603c8601709751f09f5df1c86e2feb97d59a5cad062854afb6ba68703dac93f7; split.json sha256: a5df789844cc455b2e0396954181e75e9c67ea3414dcb638f5480b93f1a013d9; natural_truth sha256: 049da4ecc8fc218d2529949e244c6fe83ef346f1d782cc827311b21990e9bfae
- prompt sha256: 4282aeb043041670739bdb72d1af571992122ff1bd0011bd1b664f25997ba29f; schema sha256: cc43db81400a6ded66441622c22369855627f1af5befeecf580b1c2bcfb082d7
- pool: 64 entries (37 oracle, 27 extra), 16 burn-in and 48 test entries, 6 probes; items 328 (64 controls, 63 N-big, 93 N-mid, 54 N-stale, 26 V-body, 23 V-coh, 6 probe controls); 5 Haiku and 3 Sonnet passes per item, 2,624 calls planned
- smoke (excluded entries, 8 calls): parse 8/8, Haiku $0.0014 and Sonnet $0.0139 per call, model ids claude-haiku-5-5 and claude-sonnet-5-5
- registration time (UTC): 2026-10-17 18:32 UTC

## 10. Deviations log

- 2026-10-17, before any test row was opened. Burn-in adjudication: 18 review-pile rows (majority-bucket rule) labeled blind; 8 of 16 burn controls carried a real contradiction and moved to the natural stratum (L9a, E10p0, M09b, E09, L3a, E09c, E06, E10b threshold arm). First burn step then FAILED the early gate: Sonnet located recall 0.930, false alarms 0.458 on the 8 remaining controls (bar 0.40). Before deciding, every per-pass flag on those 8 controls (both models) was reviewed. One adjudication error was found and corrected: the M08a row had been labeled false alarm on its causal and noise claims, but its trend claim ("admissions fall over training ... while the threshold's do not") is contradicted by the table (greedy seed 1: 5, 10, 4, 7; threshold seed 0: 9, 8, 17, 6), so M08a is natural. No pile rule or read was changed. The burn step was rerun with the corrected labels; its outcome is recorded below.
- Disclosure, same time: the critic prompt asks for "claims the entry's own numbers do not support" (overclaims included), while the adjudication rubric counts as real only claims contradicted by the entry's own numbers or thresholds; flags on unsupported-but-not-contradicted overclaims (for example "holds in its strongest form" from one seed within noise) therefore count as false alarms for both models. This is the registered definition and is kept. Added as a reported sensitivity (no read changes): false-alarm rate counting only flags of kind number or verdict.
- 2026-10-17 18:47 UTC. Burn step rerun after the M08a correction: G0-burn PASS (Sonnet located recall 0.930, false alarms 0.381 on 7 clean controls). Frozen (runs/AR3/frozen.json, sha256 1a85cd27768a5bf5): e=0.25, conf_agree threshold t*=0.7778 with tie probability p*=0.1063; diagnose recommended budget b*=0.75 (median over 20 shuffles, min 0.65; entry bootstrap 5th to 95th percentile 0.35 to 0.85), label usable. Burn AUROCs for Haiku-vs-Sonnet disagreement: conf_agree 0.656, conf_vote 0.600, conf_verb 0.816, conf_verb1 0.611; against truth: conf_agree 0.505, conf_vote 0.575, conf_verb 0.901, conf_verb1 0.846. Power appendix: per-entry sd of Q_S minus Q_C 0.43, MDE at 48 test entries 0.154, so R2 (margin 0.05) is expected unresolved; recorded before any test row is opened. No test row has been read.
