# AR3 implementation spec: critic cascade (Haiku passes, agreement-based escalation to Sonnet)

Status: frozen design, 2026-10-17. Implementers read this file in full before writing code. One file per implementer. Python for stats: /homes/rdubash/yes/bin/python (scipy 1.15, statsmodels 0.14, pandas, sklearn). Python for diagnose: cd /homes/rdubash/active-audit/avrl && /homes/rdubash/active-audit/.venv/bin/python -m avrl_router.diagnose. No em dashes anywhere. New files only under avrl/exp/ and avrl/runs/AR3/. Never modify existing files except appending to avrl/RESEARCH_LOG.md (the orchestrator does that, not implementers).

## 1. Question and claim

RL claim 1 (the thesis): a label-free threshold on the cheap grader's own confidence matches full expensive verification at a quarter of the expensive calls. AR3 tests the analogue on the critic role of the autoresearch loop. Cheap critic = Haiku, k=3 passes per item; confidence = agreement among the passes (analogue of response consistency); expensive critic = Sonnet, one pass. Cascade: escalate the lowest-agreement quarter of items to Sonnet. Burn-in = 25 percent of entries, fully verified, used to set the agreement threshold and to run the regime diagnostic (avrl_router/diagnose.py).

Claim: at a fixed escalation rate e=0.25 the cascade is (R1) better than random escalation at the same rate and (R2) non-inferior to all-Sonnet single pass within M=0.05 on prevalence-weighted critic quality. Null: agreement is no better than random, or the cascade is measurably worse than Sonnet.

## 2. Frozen constants

pi=0.25 (deployment share of defective entries; sensitivity 0.10 and 0.50). M=0.05 (sensitivity 0.03, 0.08, exploratory). k=3 primary (fixed triple = Haiku passes 1,2,3); k=1 and k=5 sweeps exploratory. Haiku passes per item 5, Sonnet passes 3. Primary operating point e=0.25. Budget grid for curves: e in {0.10, 0.25, 0.40, 0.60}. Split salt "ar3-2026-10-17". Bootstrap 10,000 resamples, seed 0. Line tolerance for localization: +-1 (sensitivity: 0 and +-3, and "any flag").

## 3. Substrate: exp/ar3_build_substrate.py

Input: avrl/RESEARCH_LOG.md (snapshot it: copy to runs/AR3/ledger_snapshot.md, record sha256 and line count in pool_report.json). Also runs/AR1/oracle.json for the list of 52 critic-verified ids (the "oracle" stratum).

3.1 Sections: split at lines starting "## ". entry_key = sha1(header line)[:8]. Keep header line, date (regex 2026-\d\d-\d\d), body lines.

3.2 Strip critic text (paragraph-state parser on blank-line-separated paragraphs): from the first paragraph whose stripped text matches ^critic note (case-insensitive) drop that paragraph and every following paragraph that begins with one of: Corrected verdict, Everything else checked out, One real problem, One claim in the entry, Four things, Three real issues, That said, Verdict (when followed by "confirmed" within the paragraph), or that is a numbered list item (^\d+\.) directly continuing a critic paragraph. Stop dropping at the first paragraph that does not match. After stripping, assert that no remaining paragraph matches critic|corrected verdict|review note (case-insensitive) unless its sha1 is in exp/ar3_asis_whitelist.json (the implementer builds this whitelist by printing every tripping paragraph for the oracle entries E01 E06 E09b E09c L9d P1-judge-s1 L10a E10 M14 M15 and including only those that are narration, not corrections; print them in the report). A non-oracle section that trips the assertion is dropped from the pool.

3.3 Pool rule (mechanical, print counts): date before 2026-10-16; header id does not start with AR; 6 to 80 body lines after stripping; at least 2 table lines (lines starting with |, excluding the |---| separator); at least one plantable number (3.6); not in the exclusion list: M01 (interleaved corrections), P1-judge-lr5-s2, P1-judge-5090b-s0, M06b (no tables), and the six natural probes E04, L5a, E05, L7b, L12b, L13a. Ids repeat in the ledger (E10b x several, M10, S1); entries are keyed by entry_key, and the header id is kept as a display field. Expected pool 55 to 70 entries. Record stratum per entry: "oracle" if in the 52, else "extra".

3.4 Natural probes: the six entries above, stripped, rendered like all others, split="probe", kind="control" (their truth is filled later from exp/ar3_natural_truth.json by the orchestrator; the substrate builder only writes them). Never used in thresholds, weights or reads.

3.5 Rendering: first line is the real header line; every line prefixed "L%03d| " counting from 1 including blank lines. Plants never change the line count. item_id = sha1(entry_key + kind + str(seed_idx))[:12]. Rows written shuffled with Random(20261017).

3.6 Plantable numbers: a numeric token (regex (?<![\w.])(\d{1,3}(?:,\d{3})+|\d+\.\d+|\d+)(?![\w.%]*\w) with care for percent signs: a trailing % is allowed and preserved) that appears in at least one table line (cell) AND in at least one prose line (not a table line, not a line starting with Run:, Files:, Script, Hypothesis:, not inside backticks, not a date, not followed within 2 tokens by seeds|seed|min|minutes|steps|epochs|GB|MB|tokens|s |h |UTC|job, and not a 4-digit year). Prose lines only are planted. Thousands separators preserved in format. Decimals preserved (same number of decimal places).

3.7 Plant kinds (per entry at most one control + up to 7 plants; distinct positions; dedupe by content hash; fewer candidates means fewer plants; rng = Random(int(sha1(entry_key+kind+str(seed_idx)),16) % 2**32)):
- N-big (1 seed): value x f, f in {0.4, 0.6, 1.4, 1.6}; same format; new value must occur nowhere in the entry.
- N-mid (2 seeds): value x f, f in {0.85, 0.90, 1.10, 1.15} (5 to 15 percent); same decimals; for integers below 20 use +-2; new value must occur nowhere in the entry. (Not rounding-sized; the critic prompt tells the model not to flag rounding.)
- N-stale (2 seeds): replace with a different numeric value taken from the SAME table column, another row (parse the table by | cells; column = cell index); the new value may occur only in table cells of that column and in no prose line. Record second_prose_copy = whether the old value still appears in another prose line.
- V-body (1): on the first line whose stripped text starts with "Verdict", find the first token among pass|kill (word-boundary, case-insensitive, not inside parentheses or brackets) and flip pass<->kill; header unchanged. Skip entries without such a token.
- V-coh (1): flip the header verdict token, the Verdict-line token and any pass|kill token in the "What it changes" paragraph together (max 3 changed lines), only when the entry's Verdict paragraph contains explicit threshold language (both the words pass and kill or the word threshold, plus at least one number); otherwise skip. Detectable only by checking numbers against thresholds.
Assertions per plant: old value present in at least one table line; exactly the planted lines differ from the control; line count unchanged; accept set nonempty.

3.8 Accept set per plant: N kinds = planted line +-1 plus every table line that contains the old value; V-body = Verdict line +-1 plus header line; V-coh = each flipped line +-1 plus lines containing the threshold rule (lines in the Verdict paragraph with a number). Stored as explicit line lists.

3.9 Split by entry: order entries by sha256(entry_key + salt); first ceil(0.25 N) entries = "burn", rest = "test". Probes = "probe". Never split by variant. Write split.json {entry_key: split}.

Outputs: runs/AR3/items.jsonl (fields: item_id, entry_key, header_id, date, stratum, split, kind, seed_idx, text, n_lines, plant {lines, old, new, accept_lines} or null, covariates {second_prose_copy, n_table_lines, n_numeric_tokens, has_threshold_language, n_chars}), runs/AR3/split.json, runs/AR3/pool_report.json (counts per kind, per split, excluded entries with reasons, whitelist hits, ledger sha256 and line count), runs/AR3/items.sha256. Print a unit-test report: plants per kind, residual critic-text paragraphs 0, every assertion passed.

## 4. Critic task (frozen, identical for both models)

System prompt (exact text): "You review one entry from a research ledger, using only the entry itself. Each line is prefixed with its line number like L012|. Find every claim that the entry's own table, numbers or stated pass/kill thresholds do not support, including a stated verdict that contradicts the entry's own numbers, thresholds or header. Many entries are fully consistent; for those return an empty problems list. Do not flag style, rounding, terminology, forward-looking estimates, or claims about files, runs or code you cannot see. For each problem give the line number where the inconsistency is stated, a kind, the exact quote, and a one-sentence explanation. Also give judgment_confidence, 0 to 100, your confidence that your overall judgment (problems found, or none) is correct."
User prompt = the rendered entry text, nothing else.
Schema: {"type":"object","properties":{"problems":{"type":"array","maxItems":5,"items":{"type":"object","properties":{"line":{"type":"integer"},"kind":{"type":"string","enum":["number","verdict","overclaim","other"]},"quote":{"type":"string"},"explanation":{"type":"string"}},"required":["line","kind","quote","explanation"]}},"judgment_confidence":{"type":"integer","minimum":0,"maximum":100}},"required":["problems","judgment_confidence"]}
Command: claude -p --model <haiku|sonnet> --no-session-persistence --tools "" --strict-mcp-config --setting-sources "" --output-format json --json-schema <schema> --system-prompt <system> <user prompt>. Record sha256 of system prompt and schema in result files.

## 5. Runner: exp/ar3_run_critic.py

Reads items.jsonl and split.json. Schedules: burn entries first (all their items), then test, then probe; within an item the pass order H1,H2,H3,S1,S2,H4,H5,S3; Haiku and Sonnet interleaved in the queue so alias drift is time-balanced. ThreadPoolExecutor, --workers default 12, back off to 6 on repeated 429 or overload text. Per call: catch subprocess.TimeoutExpired (300 s), retry up to 4 times with backoff; parse stdout JSON; a call is valid iff is_error is false and structured_output is a dict with a problems list. Record one JSONL line per call: item_id, split, model_alias, pass_idx, ok, structured_output, model_ids (keys of modelUsage), duration_ms, total_cost_usd, usage (input, output, cache read, cache write tokens), ts (epoch). Files: runs/AR3/calls_burn.jsonl, calls_test.jsonl, calls_probe.jsonl (append; resumable: skip keys already present and valid). Running cost tally printed every 50 calls; hard stop at $80 total. Abort if more than 5 percent of calls fail after retries. --smoke mode: items from the excluded entries only (M01 and the three no-table entries, rendered the same way by a helper in the substrate module exposed as a function), 1 Haiku and 1 Sonnet pass each, output to calls_smoke.jsonl; prints parse success, cost per call and latency per model. Items with fewer than 3 valid Haiku or 2 valid Sonnet passes are listed in runs/AR3/invalid_items.json and excluded downstream. Progress output must not print per-pass outcomes for test items (print counts only).

## 6. Features: exp/ar3_features.py

Per pass p: L_p = set of reported lines (ints). sim(p,q) = 1 if both empty; 0 if exactly one empty; else Dice overlap with greedy matching at |line difference| <= 1.
Per item (from the passes available):
- conf_agree = mean pairwise sim over Haiku passes 1,2,3 (primary). conf_agree_10 = mean over all 10 triples of the 5 passes (exploratory).
- conf_vote = max(f, 1-f) where f = fraction of the 3 passes that flag (nonempty problems).
- conf_verb = mean judgment_confidence/100 over passes 1,2,3; conf_verb1 = pass 1 only.
- structural: log_chars, n_numeric_tokens, has_threshold_language, mean_n_problems (over 3 passes), flag_frac, q_ok (fraction of quotes found verbatim within +-1 of the stated line).
- Cheap aggregate decision (k passes): supported_lines = lines reported (with +-1 merging) by at least ceil(k/2) passes; flagged iff nonempty. For k=1 use pass 1; k=3 passes 1,2,3; k=5 all five.
- Correctness vs truth: control (truth clean) correct iff unflagged; plant correct iff some supported line is within tolerance of the accept set; record flagged_elsewhere (flagged but no accepted line) separately from miss.
- Sonnet: per-pass correctness (same rule, single pass lines); sonnet_single_expect = mean over valid passes; sonnet_majority_correct from lines supported by >= 2 of 3 passes.
- proxy_error_sonnet = Haiku k3 decision disagrees with Sonnet majority (one flags and the other does not; or both flag with no shared line within +-1). Deployable, label-free.
- proxy_error_truth = Haiku k3 decision wrong vs truth.
Truth: from items.jsonl (plants) and from the control adjudication file runs/AR3/control_labels.json ({entry_key: "clean"|"natural"}); controls labeled natural are moved to the probe stratum (kind stays control, split becomes "probe"), their plants stay. Default when the file is missing: all controls clean.
Review pile writer: runs/AR3/review_pile_<split>.csv with one row per (control entry, line bucket +-1) supported by >= 3 of 5 Haiku passes OR >= 2 of 3 Sonnet passes; columns: row_id (random hash), entry_key, header_id, bucket_lines, quotes (joined), explanations (joined, model names removed), table_lines (all table lines of the entry), label (blank). Blind to model, pass and split (the file name carries the split only because the test pile is built after the freeze).
Output: runs/AR3/items.parquet (one row per item with all fields above plus truth, kind, split, weights) written only for the splits requested (--split burn|test|probe|all); the test step refuses to run unless runs/AR3/frozen.json exists and its sha256 matches runs/AR3/frozen.sha256.

## 7. Weights

Five plant kinds fixed: N-big, N-mid, N-stale, V-body, V-coh. Weights from the FULL pool (not per resample): each control item weight (1-pi)/N_controls; each plant item weight (pi/5)/N_kind for its kind. In every bootstrap resample the per-item weights are fixed and renormalized to sum to 1. Report the effective sample size (sum w)^2 / sum w^2 for the test slice and the minimum items per kind. If a kind has fewer than 8 test items, merge it into its family (N-* or V-*) for the primary Q and say so.

## 8. Policies (replay on test; escalated item takes sonnet_single_expect)

P0a all-Haiku k=1; P0b all-Haiku k=3 majority; P0c all-Haiku k=5 majority; P1 all-Sonnet single pass (comparator); P1b Sonnet majority of 3 (reference); Sonnet pass-vs-pass agreement = noise ceiling.
P2 random escalation at rate e: analytic per item (1-e)*q_H3 + e*q_S.
P3 consistency threshold on conf_agree (primary): escalate conf_agree < t*, and conf_agree == t* with probability p* (analytic), t*,p* frozen from the burn-in weighted quantile at e=0.25. P3b same on conf_vote. P3c same at b* (diagnose's recommended budget, secondary).
P4 verbalized: conf_verb1 (cost 1 Haiku pass + e Sonnet) and conf_verb (k=3).
P5 escalate-on-flag: escalate iff Haiku k3 flags (single realized e).
P6 oracle: escalate items where Haiku k3 is wrong and Sonnet right first, then others.
P7 learned scorer: L2 logistic regression (sklearn, C=1) on [conf_agree, conf_vote, conf_verb, mean_n_problems, log_chars, has_threshold_language, flag_frac, q_ok], target proxy_error_sonnet, trained on burn-in only; escalate the top-e predicted error.
P8 spurious controls: escalate longest entries; escalate most mean_n_problems.
Curves: for each policy, rank items by its score (label-free), escalate the top fraction e for e on a fine grid 0 to 1 (fractional ties handled analytically), compute weighted Q; report Q at the grid {0.10,0.25,0.40,0.60}, e needed for 98 and 99 percent of Q_S, gap closed G(e) = (Q - Q_rand)/(Q_oracle - Q_rand) (undefined if denominator < 0.01), and dollar cost = measured mean Haiku cost x k + e x measured mean Sonnet cost, as a ratio to all-Sonnet single pass.
Replay unit tests (must pass before any number is trusted): e=0 equals P0b, e=1 equals P1, P2 at e=0.5 is the midpoint, P6 is maximal at every e.

## 9. Burn-in and freeze (exp/ar3_analyze.py --step burn)

1. Build the burn-in frame from items.parquet (burn split, fixed triple). Replicate each row round(w_i / w_min) times (cap 20,000 rows) so the unweighted diagnose sees the pi=0.25 mixture; shuffle rows with a caller-side seed; no NaN.
2. Run diagnose for confidence columns conf_agree, conf_vote, conf_verb, conf_verb1 against proxy_error_sonnet at budgets [0.1,0.25,0.5], target "rel:0.25", n_boot=0, for 20 caller-side row shuffles; b* = median recommended budget for conf_agree (diagnose returns 1.0 when the target is never reached; treat None or nan as 1.0). Also run against proxy_error_truth (AUC comparison only). Record AUROC per signal on burn-in.
3. Entry-level bootstrap of the burn-in (1,000 resamples of burn entries): distribution of b*, of t* (the e=0.25 quantile) and of the burn-in Haiku-vs-Sonnet disagreement rate; report 5th to 95th percentiles.
4. (t*, p*) = weighted quantile of conf_agree at e=0.25 on burn-in with analytic tie probability.
5. Early gates on burn-in (G0-burn): Sonnet single-pass macro located recall on burn plants >= 0.50 and Sonnet FA on burn controls <= 0.40 (looser than test G0 because n is small); if either fails, print NOT TESTABLE and stop before test rows are opened.
6. Write runs/AR3/frozen.json {k, signal, e_primary 0.25, t*, p*, b*, b*_interval, burn AUROCs, M, pi, weights by kind, items.sha256, split sha256, prompt and schema sha256, model ids seen} and runs/AR3/frozen.sha256. Power appendix: per-entry sd of (Q_S - Q_C) on burn-in and the MDE at test N.

## 10. Test analysis (exp/ar3_analyze.py --step test)

Refuses to run unless frozen.json matches its sha256. Computes on the test split:
Gates: G0 pool >= 50 entries; >= 95 percent valid calls; Sonnet single-pass macro located recall on test plants >= 0.60; Sonnet FA on adjudicated test controls <= 0.35. G1 headroom: Q_S - Q_H3 >= 0.06 with one-sided 95 percent cluster-bootstrap lower bound > 0 (if all-Haiku k=3 already satisfies NI at M, R2 is vacuous; say so).
Reads at the frozen operating point (e=0.25, frozen t*,p*): R1 Q_C - Q_rand(e_hat) > 0 with lower bound > 0 (companion sign-flip permutation p over entries, 9,999). R2 Q_S - Q_C < 0.05 with one-sided 95 percent upper bound < 0.05. Realized e_hat reported with Wilson interval (it is an outcome).
Decision: PASS = G0, G1, R1, R2 all true. KILL = K1 (two-sided 90 percent CI of gap closed at e=0.25 has upper end < 0.25) or K2 (one-sided 95 percent lower bound of Q_S - Q_C > 0.05). Otherwise INCONCLUSIVE with MDE and q = N/N* reported and the word "unresolved".
Qualifier: for any kind with >= 15 test items whose confident-wrong rate (all three Haiku passes agree and the cheap decision is wrong) is >= 0.50, the claim is worded "except kind X".
Secondary family (Holm, m=3): S1 AUROC(conf_agree) - AUROC(conf_verb) > 0 for proxy_error_sonnet; S2 upper bound of Q_P7 - Q_P3 at e=0.25 < 0.02 (threshold not beaten by the learned scorer); S3 at equal dollars, cascade minus all-Haiku k=5 majority > 0 (e chosen so dollar cost equals P0c).
Diagnostic (reported, no read): b*, Q and e at b* (P3c), diagnose's budget-vs-error table on test versus random.
Exploratory, unadjusted: k sweep (1,3,5); conf_vote; P5; P8; per-kind located recall; tolerance, pi, M sensitivity; natural probes (flag rate, located rate, conf_agree); 1,000 random entry-level 25/75 splits (distribution of b*, e_hat, Q_S - Q_C, share meeting NI); Trust-or-Escalate fixed-sequence scan (scan t from high to low, keep the last t whose Clopper-Pearson one-sided 95 percent upper bound on accepted-set error vs truth <= alpha = burn-in Sonnet error rate); oracle versus extra stratum; A/A noise floor (|Q difference| between Sonnet pass pairs and between Haiku triples, 95th percentile at test N).
Statistics (exp/ar3_stats.py): wilson(k,n), clopper_pearson(k,n), holm(pvals), mcnemar_midp(n10,n01), cluster_bootstrap(entries -> statistic, B, seed, one- or two-sided percentile), signflip_perm(per-entry diffs, 9999), weighted_auroc. Unit tests: Wilson 5/12 = [0.193, 0.680]; Clopper-Pearson 5/12 = [0.152, 0.723]; Holm [0.01,0.04,0.03] -> [0.03,0.06,0.06]; n10=5, n01=1 -> exact p 0.219, mid-p 0.125. Per-item means over passes first, then paired statistics over entries; never pool passes into one standard error.
Outputs: runs/AR3/result.json (every number above plus hashes, model ids, per-id cost, deviations list), runs/AR3/table.md (one row per contrast: contrast | entries (items) | paired diff with CI | permutation p | Holm p (secondary only) | q and MDE | cost ratio | verdict in {supported, not supported, unresolved}), runs/AR3/fig/ (cost-quality curves with oracle, random, endpoints, Pareto frontier and A/A noise band; risk-coverage curve; AUROC bars; reliability diagram; conf_agree and conf_verb histograms; confident-wrong by kind; burn-in predicted vs realized e), runs/AR3/cascade_config.json (k, prompt and schema sha256, t*, p*, b*, measured dollars per call by role).

## 11. Timeline and budget

Build and unit test: about 60 minutes (parallel implementers). Smoke: 10 minutes. Collection: about 60 to 90 minutes at 12 workers (about 4,000 calls, about $45 at AR2 per-call prices; hard stop $80). Burn-in analysis and freeze during collection. Control adjudication: the orchestrator labels the blind review pile (burn before freeze, test after freeze). Test analysis: minutes. Ledger result entry and operating card: 15 minutes.
