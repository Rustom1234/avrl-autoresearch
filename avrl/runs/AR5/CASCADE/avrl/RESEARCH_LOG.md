# AVRL research ledger

Entries are appended by the autoresearch loop and by hand. Verdicts are against the pre-registered thresholds in PROJECT.md and EXPERIMENT_BACKLOG.md.

## E01 2026-09-30 kill (for claim 3 on this task/proxy pair)
Hypothesis: proxy error (group majority answer wrong) has exploitable structure in prompt space and is not already captured by cheap features.
Run: Qwen2.5-1.5B-Instruct, GSM8K train, 3,000 prompts x 8 rollouts (vLLM, temperature 1.0), proxy = agreement with group majority, verifier = gold answer. Files: runs/p0/summary.json, result.json, feats.parquet, sim_*.json, analyze.log.

| quantity | value |
|---|---|
| rollout accuracy / majority accuracy | 64.2% / 85.5% |
| group proxy-error rate (majority wrong) | 14.5% |
| proxy error, unanimous groups (18.6% of groups) | 0.5% |
| proxy error, split groups | 17.7% |
| AUC for proxy error: agreement only / all cheap / embedding only / cheap + embedding | 0.914 / 0.923 / 0.606 / 0.922 |
| neighbor ratio of proxy error by embedding cell (k = 30 / 100 / 300) | 1.00 / 1.03 / 1.04 |
| confident subset (agreement >= 0.75, n = 1,551): error rate / cheap AUC / neighbor ratio | 1.1% / 0.63 / 1.16 |

Allocation replay (errors caught per budget, 100 cells, 8 seeds; columns: uniform, static with free labels [approximately the agreement threshold], static budget-matched, TS flat, greedy online self-trained, TS online, TS + free prior, neighbor, oracle):
10%: 28, 165, 98, 31, 138, 95, 118, 53, 218. 25%: 76, 264, 212, 83, 253, 208, 232, 117, 309. 50%: 157, 299, 287, 164, 296, 288, 294, 206, 309. Embedding features change nothing.

Verdict: pre-registered read in PROJECT.md triggers (embeddings add < 0.05 AUC, neighbor ratio < 1.5): claim 3 (edge from prompt-space structure) is dead on GSM8K with a majority-vote proxy. The fixed agreement threshold is near-optimal here; the self-training allocator matches it at 25% and 50% budgets and loses at 10%. This is the "threshold regime" and becomes the first point of the structure-statistic-versus-gain curve (see the revised proposal): cheap-feature AUC 0.92, gain of learned over threshold about 0.
What it changes: the allocator can only earn its keep where the proxy lacks a confidence signal. Next: E04 (small judge model as proxy, no self-consistency), E05 (self-verification proxy), E08 (code task where majority vote over programs is ill-defined), each reporting the same three statistics plus the replay, to populate the curve. Also add an explicit label-free agreement-threshold policy to the replay so the baseline is not approximated by "static with free labels".

## E03 2026-10-18 kill

**Hypothesis (claim 2, against a label-free fixed rule):** at a 25% verification budget on the E01 GSM8K majority-vote proxy, the learned allocator (ts_online: Thompson over embedding cells with an online logistic refit) admits fewer corrupted rewards than the label-free agreement threshold (agreement ascending, then n_distinct descending, the ThresholdAllocator._order key). Pass: 95% CI of D = admitted(threshold) minus admitted(ts_online) excludes 0 and D is positive. Kill: CI includes 0, or D is negative.

**What was run.** No generation, no GPU. Replay of the E01 rollouts (Qwen2.5-1.5B-Instruct, GSM8K train, 3,000 prompts x 8 rollouts, proxy = agreement with group majority, verifier = gold answer) using avrl/exp/e03_threshold_bootstrap.py, a copy of the sim_alloc.py replay plus one label-free arm (threshold). Setup: cell100 buckets (buckets with fewer than 3 groups dropped, seed-0 quarter fit split), cheap+emb features (22 columns: 6 cheap + 16 PCA), sim pool of 2,180 groups in 75 buckets with 309 proxy errors (14.2% of pool), budgets 218 / 545 / 1,090 groups (10 / 25 / 50%), 8 allocator seeds. Bootstrap: 200 resamples of groups with replacement at 25%, 15 s, no reduction needed. Files: avrl/runs/E03/result.json, avrl/runs/E03/run.log.

Checks: E01 numbers reproduce exactly (max abs diff 0.0 against runs/p0/sim_cell100_cheap+emb.json for uniform, static, greedy_online, ts_online). The threshold arm and the bootstrap percentiles were recomputed independently from feats.parquet and match.

**Numbers.** admitted = 1 - caught/309 (lower is better). Caught counts in parentheses, mean over 8 seeds for stochastic arms.

| arm | labels used | 10% | 25% | 50% |
|---|---|---|---|---|
| uniform | no | 0.910 (27.9) | 0.756 (75.5) | 0.491 (157.3) |
| static (reference only) | yes, fixed classifier fit on free labels | 0.505 (153.0) | 0.149 (263.0) | 0.029 (300.0) |
| threshold (label-free, primary baseline) | no | 0.492 (156.9) | 0.183 (252.5) | 0.038 (297.4) |
| greedy_online | yes, online | 0.579 (130.1) | 0.184 (252.3) | 0.041 (296.3) |
| ts_online (primary learned arm) | yes, online | 0.718 (87.3) | 0.354 (199.5) | 0.085 (282.6) |

D = admitted(threshold) minus admitted(arm). Positive means the arm admits fewer corrupted rewards.

| D row | 10% point | 25% point | 25% 95% bootstrap CI (200 resamples) | share of resamples with D > 0 |
|---|---|---|---|---|
| ts_online (primary) | -0.225 | -0.172 | [-0.230, -0.142] | 0.000 |
| greedy_online (secondary) | -0.087 | -0.001 | [-0.044, +0.017] | 0.200 |

At 50%: D(ts_online) = -0.048, D(greedy_online) = -0.004.

**Verdict: KILL.** D(ts_online) at 25% is -0.172 and its CI lies entirely below 0. The pass rule fails and the kill rule (D negative) fires. Claim 2 is not supported on this task and proxy against a label-free fixed rule. greedy_online ties the threshold (CI spans 0). It is a secondary row and does not set the verdict.

**Where the budget is lost** (existing E01 numbers, no new run, 25% caught): labeled static 263.0, label-free threshold 252.5, greedy_online 252.3 (same online learner without the bucket layer), ts_static 225.4 (bucket Thompson with the labeled static classifier as prior), ts_online 199.5. The learner without buckets ties the threshold, so the gap sits in the bucket-level Thompson layer. This is inferred from these five arms, not from an ablation.

**What it changes about the thesis claims.**
- Claim 2 (learned allocator admits fewer corrupted rewards than any fixed rule at equal budget): not supported on GSM8K with majority vote. The best online learned arm ties the label-free threshold, and the LearnedAllocator-style arm admits about 1.9x as many corrupted rewards at 25%. The labeled static classifier beats the threshold by about 10 caught docs, but it needs free labels to fit and is not an online allocator, so it does not rescue the claim as worded.
- Claim 3 (edge comes from prompt-space structure): there is no edge to attribute. E01 structure is unchanged (neighbor ratio 1.0; AUC cheap 0.923, cheap+emb 0.922). The E01 kill on this pair stands. Not tested on any other pair.
- The E01 read that the fixed threshold is near-optimal is stronger now: it is the best label-free policy tested and sits within about 10 caught docs of a labeled reference.
- Phase 1: the kill rule says to reopen the phase-1 funding question before any GPU work. This entry starts no GPU job. Decision for Rustom.

**Caveats.**
- The bootstrap resamples groups i.i.d. and ignores the cell clustering that ts_online uses. A cell-cluster bootstrap was not run. The point gap (0.17) is about four times the CI half-width, so a flip is unlikely but unchecked.
- One task, one proxy pair, one model, cell100 only. The threshold arm was not run on cell30.
- Threshold ties (same agreement and n_distinct) are broken by a seeded random order averaged over 8 seeds. The input-order tie-break gives 254.0 caught at 25% (admitted 0.178). Verdict unchanged.
- The E01 table above matches runs/p0/sim_cell100_cheap.json (cheap features only), not the cheap+emb JSON used here. Differences are up to 12 caught docs per arm (static at 10%). E01's "embeddings change nothing" still holds.
- The spec said 21 features. E01's cheap+emb list has 22 (6 cheap + 16 PCA). It was kept so the baseline reproduces. allocators.py has the same miscount.

**Follow-ups** (CPU only, none needs the GPU; logged in lit/avrl/EXPERIMENT_BACKLOG.md as L3a to L3c).
1. **L3a, robustness of the kill:** cluster bootstrap for D at 25% (resample cell100 buckets, not groups), plus the same config on cell30 with the threshold arm added. E01 has cell30 cheap+emb numbers for the other arms (static 280.0, greedy 278.4, ts_online 252.5 on a 2,278-doc pool with 353 errors). Expected information: medium-low. It can overturn the kill only if the CI widens about fourfold or cell30 reverses the ordering. Under 5 min.
2. **L3b, warm-started bucket prior:** ts_online with its initial bucket prior set from the label-free threshold (per-bucket mean agreement) instead of Beta(1,1), keeping the online refit. Pass: D > 0 with CI excluding 0 at 25%. Expected information: high. It tests whether any learned allocator on the LearnedAllocator design can beat the label-free threshold, and it is the cheapest remaining route to a claim-2 pass. Under 10 min.
3. **L3c, out-of-sample ceiling for a labeled ranker:** five-fold cross-fit by cell100 bucket with the same 22 features, ranking at 25%, compared with threshold (252.5) and static (263.0). Expected information: high for the funding decision. If the ceiling stays near 263 caught, the best achievable edge on this pair is about 3 to 4 points of admitted rate, which bounds the phase-1 prize before any GPU run. Under 2 min.

**Critic note (E03, 2026-10-18).** Verdict KILL stands, unchanged. Two statements in the entry are corrected below; no number that sets the verdict changes.

Verified against raw files: every figure in the two tables, the D rows, the CIs and the bootstrap shares match runs/E03/run.log and result.json (200 draws, no NaN, percentiles recompute exactly). The threshold arm was recomputed independently from feats.parquet (caught 156.875 / 252.5 / 297.375 at 218 / 545 / 1,090 groups, input-order tie-break 155 / 254 / 297, pool 2,180 groups, 75 buckets, 309 errors). The 22 model columns contain no gold: gold, pass_rate and majority_correct exist in feats.parquet but are not in feature_list, and the threshold reads only agreement and n_distinct. Budgets are identical across arms (int(frac x 2,180), K = 1). exit_code 0, run.log has no warning or traceback, 200 of 200 resamples returned. The verdict follows from the pre-registered rule: D(ts_online) = -0.172 is negative and the CI [-0.230, -0.142] excludes 0 on the negative side, so the kill rule fires.

Corrections.
1. Caveat "Differences are up to 12 caught docs per arm (static at 10%)" is true only for the four arms E03 re-ran (uniform, static, greedy_online, ts_online). Across the whole E01 table, which is cheap-feature output, the gap to the cheap+emb JSON is larger for arms E03 did not use: static budget-matched 212.0 vs 187.6 at 25% (24.4) and 97.6 vs 84.1 at 10% (13.5), TS + free prior 232.1 vs 225.4 at 25% and 118.5 vs 112.6 at 10%, neighbor 117.0 vs 109.2 at 25%. The E03 table is unaffected.
2. The ts_online arm is the sim_alloc.py replay arm, not the allocators.py LearnedAllocator. The class differs (cold start by threshold order, per-group Beta draws from a cell posterior with prior_strength 2.0, unweighted logistic, refit_every 20), so "LearnedAllocator-style arm" and L3b's "LearnedAllocator design" overstate what was tested. The kill is on ts_online as pre-registered. As a CPU sensitivity check on the same pool at 25% (8 seeds), giving ts_online and greedy_online a threshold-ordered cold start instead of a random one gives D = -0.096 and -0.004, so the conclusion does not depend on the cold start.

Caveat now checked (CPU, run inline, nothing written). Bucket-cluster bootstrap (resample the 75 buckets with replacement, duplicates relabeled, 200 resamples, seed 123, 25%): D(ts_online) mean -0.181, 95% CI [-0.225, -0.131], share D > 0 = 0.000; D(greedy_online) CI [-0.035, +0.012], share D > 0 = 0.110. Alternate fit-split seeds 0 to 9 (pool 2,180 to 2,389 groups): D(ts_online) at 25% is negative in all ten, from -0.138 to -0.198; D(greedy_online) ranges from -0.029 to +0.004 (two of ten positive, both under 0.004). Seeds and sample size are adequate for the claimed margin, which is about four CI half-widths. The cluster-bootstrap half of L3a is therefore answered (kill holds); the cell30 half is still open.

Corrected verdict: KILL for claim 2 on GSM8K with the majority-vote proxy against the label-free agreement threshold. D(ts_online) at 25% = -0.172, CI [-0.230, -0.142] (cluster CI [-0.225, -0.131]). greedy_online ties the threshold and does not set the verdict. Reopen the phase-1 funding question before any GPU work.

## L3c 2026-10-18 pass

**Hypothesis:** out of sample, a labeled logistic ranker on the same 22 cheap+emb features as the E03 static reference, cross-fit by cell100 bucket on the E03 sim pool (2,180 groups, 75 buckets, 309 proxy errors), does not beat the label-free agreement threshold at 25% by more than a few points of admitted rate. If this holds, the threshold sits near the best any fixed ranker on these features achieves on this pair, and there is no headroom for the warm start in L3b to exploit.

**What was run.** CPU only, no generation, no GPU. Script avrl/exp/l3c_crossfit_ranker.py, copied from the E03 script (E03 and the p0 files untouched). Source rows: avrl/runs/p0/feats.parquet (Qwen2.5-1.5B-Instruct, GSM8K train, 8 rollouts per prompt, proxy = agreement with group majority, verifier = gold answer). Pool built as in E03: cell100 buckets with at least 3 groups, seed-0 bucket shuffle, first quarter of buckets dropped as the fit split (819 rows, used only by the static reference). The spec said about 750 fit groups; 819 is the actual count. Sim pool: 2,180 groups in 75 buckets, 309 proxy errors. Model: StandardScaler + LogisticRegression(max_iter=1000, class_weight="balanced"), label proxy_error, 22 features (agreement, n_distinct, mean_tokens, std_tokens, any_truncated, log_qlen, pc0 to pc15). Five folds of 15 buckets each from a seeded shuffle, fold seed 0 (one fold assignment only). Per-fold train sizes 1,768 / 1,667 / 1,805 / 1,790 / 1,690 (mean 1,744); test sizes 412 / 513 / 375 / 390 / 490. All 2,180 groups ranked by the pooled out-of-fold score. Budgets 218 / 545 / 1,090 (10 / 25 / 50%). Threshold arm: 8 seeds, as in E03. Files: avrl/runs/L3c/result.json, avrl/runs/L3c/run.log, avrl/runs/L3c/exit_code (0), avrl/runs/L3c/done.flag.

**Checks.** exit_code 0; run.log and result.json agree; no NaN in features or out-of-fold scores; each bucket sits in one fold; per-fold train plus test sizes sum to 2,180. After the run, an inline recomputation from feats.parquet (nothing written) reproduced 159 / 264 / 299 caught at 10 / 25 / 50%. Both E03 references reproduce on this pool: threshold 252.5 (8-seed mean, per seed 251 to 256), static 263.

**Numbers.** admitted = 1 - caught/309, lower is better. Caught counts in parentheses.

| arm | labels used | 10% (218) | 25% (545) | 50% (1,090) |
|---|---|---|---|---|
| uniform (E03) | no | 0.910 (27.9) | 0.756 (75.5) | 0.491 (157.3) |
| threshold (E03, 8-seed mean) | no | 0.492 (156.9) | 0.183 (252.5) | 0.038 (297.4) |
| ts_online (E03, deployable learner) | yes, online | 0.718 (87.3) | 0.354 (199.5) | 0.085 (282.6) |
| static (E03, fit on 819 rows, reference only) | yes, fixed | 0.505 (153.0) | 0.149 (263.0) | 0.029 (300.0) |
| L3c cross-fit ranker (5 folds, fold seed 0) | yes, fixed per fold | 0.485 (159) | 0.146 (264) | 0.032 (299) |

Cross-fit minus threshold, caught: +2.1 at 10%, +11.5 at 25%, +1.6 at 50% (admitted gap 0.007, 0.037, 0.005). Cross-fit minus static, caught: +6 at 10%, +1 at 25%, -1 at 50%.

**Verdict: PASS** under the pre-registered rule. Caught 264 at 25% is at or above the pass line of 263.0 (admitted 0.146 against 0.149). The kill line (252.5) did not fire, and 264 is outside the inconclusive band. The pass is marginal:
- The margin over the static reference is one caught error (0.3 points of admitted rate), on one fold assignment. No fold-to-fold variance was measured.
- The gain over the threshold appears only at 25%. At 10% and 50% the cross-fit is within 2 caught of the threshold. The static reference has the same shape (below the threshold at 10%, above it at 25%). A gain at one budget and not its neighbours is more consistent with noise at that cut than with a broad ceiling.
- The cross-fit trains on about 1,744 rows, roughly 2.1 times the 819 rows the static reference uses. Matching static partly reflects more training data.
- The ranker needs full proxy_error labels for every group. A deployed allocator never has them, so this is a ceiling for a labeled fixed ranker, not an allocator.

The hypothesis predicted no more than a few points of admitted rate. The gap is 3.7 points (11.5 caught), at the edge of "a few". Its conclusion, that the threshold sits near the best achievable, is not cleanly supported at 25%. The hypothesis is neither cleanly confirmed nor refuted.

**What it changes about the thesis claims.**
- Claim 2 (learned allocator admits fewer corrupted rewards than any fixed rule at equal budget): stays not supported on this pair. L3c is a labeled ceiling, not a deployable allocator. The only deployable learner tested (ts_online, E03) lost to the threshold. L3c bounds what a labeled fixed ranker on these 22 features can add: about 11 caught (3.7 admitted points) at 25% and not at the other budgets. That bounds the prize for L3b, the only deployable route left on this pair. The pre-registered pass says L3b is worth running, but it should wait for L4a below, since the pass rests on one fold assignment. Phase-1 funding remains a decision for Rustom.
- Claim 3 (edge from prompt-space structure): not tested by L3c. The ranker includes pc0 to pc15, but no cheap-only ablation was run, so the labeled gain cannot be attributed to prompt-space structure. E01 found no embedding structure on this pair (neighbor ratio 1.0; AUC 0.923 cheap vs 0.922 cheap+emb). Claim 3 stays not supported here.

**Follow-ups** (all CPU, all logged in lit/avrl/EXPERIMENT_BACKLOG.md as L4a to L4c; thresholds are declared here before the run).
1. **L4a, fold-seed stability (under 2 min).** Rerun the cross-fit over 20 fold seeds (0 to 19), fit split fixed at seed 0. Pass: mean caught at 25% >= 263.0. Kill: mean <= 252.5. Expected information: high. It decides whether the PASS is a fold artifact before any L3b run, and it is the check the run report asked for.
2. **L4b, cheap-only attribution for claim 3 (under 2 min).** Same folds, 6 cheap features only. Pass for claim 3 (structure matters): 22-feature minus 6-feature cross-fit >= 5 caught at 25%, mean over the 20 fold seeds. Kill: gap <= 2 caught, so the labeled gain comes from cheap signals and not prompt-space structure. Expected information: high for claim 3, the only attribution test on this pair that needs no GPU.
3. **L4c, paired bucket-cluster bootstrap (under 3 min).** 200 resamples of the 75 buckets (seed 123, as in the E03 cluster bootstrap) of cross-fit minus threshold caught at 10, 25 and 50%. Pass: the 95% CI at 25% excludes 0 on the positive side. Kill: the CI includes 0. Expected information: medium-high. It tests whether the 25% gap survives cluster resampling, which is the concern raised by the gain appearing only at 25%.

L3b stays queued in lit/avrl/EXPERIMENT_BACKLOG.md and should run after L4a, not before.

**Critic note (L3c, 2026-10-18).** Verdict corrected from PASS to INCONCLUSIVE. All numbers in the entry are right; the correction is about whether one fold assignment can carry a 1-caught margin.

Verified against raw files: every figure in the table and the text matches runs/L3c/result.json and run.log (cross-fit caught 159 / 264 / 299, admitted 0.4854 / 0.1456 / 0.0324, budgets 218 / 545 / 1,090, train and test sizes sum to 2,180 in every fold, mean train size 1,744, 75 buckets in 5 folds of 15, 309 errors, 819 fit rows). The E03 comparison rows (uniform, static, threshold, ts_online) match runs/E03/run.log, and the deltas (+2.1 / +11.5 / +1.6 vs threshold; +6 / +1 / -1 vs static) recompute. exit_code 0, no warning or traceback in run.log. I recomputed the cross-fit from feats.parquet with an independent copy of the pool build and got 159 / 264 / 299, threshold per seed 251 to 256 (mean 252.5), static 263. Leakage: gold, pass_rate, majority_correct, majority and proxy_error exist in feats.parquet but none is in the 22 model columns, no model column equals any of them, no NaN in the features, and the threshold reads only agreement and n_distinct. Budgets are identical across arms (int(frac x 2,180), one group per visit). The literal rule on this run is satisfied: 264 >= 263.0.

Problem found. The PASS rests on fold seed 0 alone, and it is the top of the fold-seed distribution. Recomputed inline (CPU, nothing written), same pool, fit split fixed at seed 0, only the bucket-to-fold assignment varied. At 25%, 20 fold seeds (0 to 19): mean 260.70, sd 1.75, range 257 to 264, 4 of 20 seeds at or above 263 (seeds 0, 2, 4, 11), and seed 0 ties seed 4 for the maximum. Seeds 20 to 119: mean 260.38, sd 2.10, range 255 to 266, 15 of 100 at or above 263, none at or below 252.5. The seed-averaged value, about 260.5, lies strictly between 252.5 and 263.0, which the pre-registered rule calls inconclusive. The 10% and 50% numbers for seed 0 (159, 299) are typical (20-seed means 159.1 and 299.6). The 1-caught margin over the pass line is about half a fold-seed sd, so one fold assignment cannot support "matches the static reference". This is also the L4a pre-registered test (mean over 20 fold seeds, pass >= 263.0, kill <= 252.5); its outcome on these numbers is inconclusive, not pass.

Two statements in the entry are corrected.
1. "A gain at one budget and not its neighbours is more consistent with noise at that cut than with a broad ceiling" is wrong about the threshold comparison. The 25% gain over the label-free threshold is not noise: the lowest of 120 fold seeds is 255 (above 252.5), and a paired bucket-cluster bootstrap (75 buckets resampled, 200 draws, seed 123, fold seed 0, threshold with its seed-0 tie-break) gives a caught-share gap of +0.034, 95% CI [+0.006, +0.067] (about +1.7 to +20.6 caught of 309), share <= 0 = 0.015. What is noise is the 1-caught margin over the static line. The gains at 10% and 50% are small because the threshold is already near saturation at 50% (297.4 of 309), not because the 25% gain is spurious.
2. "Matching static partly reflects more training data" is reversed in practice. With 2.1 times the training rows, the fold-seed mean (260.7) sits below static (263), and 263 is itself one draw of a noisy reference. Across fit-split seeds 0 to 9 (each its own sim pool, 2,180 to 2,389 groups, pools overlap so these are not independent), static minus threshold at 25% is +10.5 at seed 0 but averages +1.6 (static 271.4, threshold 269.8), and static beats threshold by 5 or more in 3 of 10 pools. Seed 0 is the pool where the threshold looks worst. The cross-fit ranker (mean over 5 fold seeds per pool) beats the threshold in all 10 pools, by +4.3 to +11.4, mean +8.5 (278.3 vs 269.8). The static line is the wrong yardstick for this pair; the threshold is the right one.

Informational, not a replacement for L4b. A cheap-only 6-feature cross-fit on the same 20 fold seeds gives mean 261.55 (sd 1.05) at 25%, against 260.70 for 22 features, so the 16 PCA columns add nothing in this check (gap -0.85 caught, L4b's kill condition is gap <= 2). This agrees with E01 (AUC 0.923 vs 0.922) and means the labeled gain over the threshold comes from cheap signals, not prompt-space structure. Claim 3 stays not supported on this pair; L4b should still be logged as its own run.

Corrected verdict: INCONCLUSIVE under the pre-registered band (fold-seed mean about 260.5, strictly between 252.5 and 263.0). What L3c does establish: a labeled fixed ranker on these features beats the label-free threshold at 25% by about 8 caught (about 2.7 admitted points, not 11.5 caught or 3.7 points), robust to fold seed and bucket resampling, and it does not reliably reach the static line. What it does not establish: the PASS go signal for L3b. L3b stays queued without that signal; the headroom it could exploit on this pair is at most about 3 admitted points at 25% and about 1 at 10% and 50%. Phase-1 funding remains a decision for Rustom and is not made easier by L3c.

## L5a 2026-10-18 pre-registration (written before the run, no L5a outcome seen)

**Hypothesis (claim 2, the shipped class):** on the E03 pool (GSM8K majority-vote proxy, cell100, 22 cheap+emb columns, 2,180 sim groups, 75 buckets, 309 proxy errors), the shipped avrl/allocators.py LearnedAllocator (threshold cold start for the first 20 verified groups, then an online logistic regression refit every 20 verified groups, per-group Thompson draws from cell Beta posteriors seeded at prior_strength 2.0 times the predicted error rate) admits fewer corrupted rewards than the label-free agreement threshold at a 25% budget (545 groups). E03 killed claim 2 only for the ts_online replay arm. The shipped class has not been replayed against the threshold. This entry tests the class that phase 1 would train with.

**What will be run.** Script avrl/exp/l5a_learned_class.py, copied from avrl/exp/e03_threshold_bootstrap.py. E03, allocators.py and the p0 files are not edited. allocators.py is imported read-only. Launcher avrl/exp/L5a.sh, started through bin/gpu_run.sh on the GPU host (pine, CPU work only, run dir avrl/runs/L5a). The spec said no gpu_run.sh was needed. The ground rules route every long-running job through the helper, and this job is expected to take about 20 minutes, so it goes through the helper. Arms: threshold (seeded tie-break, E03 reference 252.5 caught at 25%), threshold_input_order (E03 reference 254.0), learned_class. Budgets 218 / 545 / 1,090 groups (10 / 25 / 50%), seeds 0 to 7. For learned_class, each visit (K = 1) builds records for every still-unverified pool group, calls select(records, 1, rng) on LearnedAllocator() with class defaults, marks the chosen group verified, and calls update() with that group only.

**Bootstraps at 25%.** Group: 200 resamples of the sim groups, BOOT_SEED 0, the same draws as E03, so D is paired with E03. Bucket-cluster: 200 resamples of the 75 buckets with replacement, seed 123, with a repeated bucket relabeled as a new cluster. Each resample recomputes the threshold and learned_class against its own error total.

**Metric.** admitted = 1 - caught / 309, caught = mean over 8 seeds. D(learned_class) = admitted(threshold) - admitted(learned_class). Positive D means the learned class admits fewer corrupted rewards.

**Pass rule (at 25%).** Point D(learned_class) > 0, AND the lower bound of the 95% percentile CI is > 0 under BOTH the group bootstrap AND the bucket-cluster bootstrap.

**Kill rule (at 25%).** Point D(learned_class) <= 0, OR the 95% CI of either bootstrap includes 0. Clarification fixed before the run: a CI lying entirely below 0 while the point D is positive is also recorded as KILL, because it is neither a pass nor covered by the pass rule.

**Descriptive only.** 10% and 50% point D and caught counts, the threshold_input_order row, and the ConvergenceWarning count. The ConvergenceWarning count is reported and does not change the verdict.

**Caveats (stated before the run).**
1. The candidate set is every still-unverified pool group at each visit (the E03 replay convention), not the per-step training batches the phase-1 loop uses. The run does not test the batch loop itself.
2. Features are unscaled, as shipped. The solver is LogisticRegression defaults with max_iter=1000.
3. The class docstring says 21 features. The replay feeds 22 (6 cheap + 16 PCA), as E01, E03 and L3c did.
4. One task (GSM8K), one proxy (majority vote), cell100 only.
5. The threshold arms are recomputed in the same run and must match E03's stored point values to 1e-9. That checks the pool build.
6. The class runs at its defaults (refit_every 20, prior_strength 2.0). If it passes or kills, nothing is retuned inside this run. Feature scaling, cell30 and the E11 hyperparameter grid are separate, later, pre-registered entries.
7. The bucket-cluster bootstrap resamples cluster identity, so it tests cluster-level variance. It is not a fixed-layout test.

**Expected information.** Highest per unit of compute on the claim-2 chain. It is the only untested step: E03's kill covers ts_online, not LearnedAllocator. A kill here would close claim 2 for the class that phase 1 would train with. A pass would support running E11 next.

## L5a 2026-10-18 kill (shipped LearnedAllocator vs label-free threshold)

**Verdict: KILL.** At 25% the shipped LearnedAllocator admits more corrupted rewards than the label-free agreement threshold. Point D(learned_class) = -0.454. Group bootstrap 95% CI [-0.472, -0.361] (mean -0.414). Cluster bootstrap 95% CI [-0.495, -0.385] (mean -0.436). Share of resamples with D > 0 is 0.000 in both bootstraps. Failing conditions: the point estimate is <= 0 (kill rule), and both CIs lie entirely below 0 (the pre-registered clarification sends that case to KILL).

**What was run.** avrl/exp/l5a_learned_class.py through avrl/exp/L5a.sh, started with bin/gpu_run.sh on pine (CPU only, about 9 minutes wall, exit_code 0, no cap hit). Files: avrl/runs/L5a/result.json, avrl/runs/L5a/run.log, avrl/runs/L5a/exit_code, avrl/runs/L5a/done.flag. Pool and arms as pre-registered: 2,180 sim groups, 75 buckets, 309 proxy errors, 22 cheap+emb columns, 8 allocator seeds, K = 1.

**Checks.** The threshold and threshold_input_order arms reproduce E03's stored point values exactly (max abs diff 0.0, pool matches E03). Zero sklearn ConvergenceWarnings in the point runs and in both bootstraps. 200 of 200 resamples returned for each bootstrap, no NaN. A smoke run on a 300-group slice (2 bootstrap resamples, crash check only) was made before the full run. Its numbers were not used and its output directory was deleted.

| budget (groups) | threshold caught (admitted) | threshold_input_order caught (admitted) | learned_class caught (admitted) |
|---|---|---|---|
| 10% (218) | 156.9 (0.492) | 155.0 (0.498) | 73.9 (0.761) |
| 25% (545) | 252.5 (0.183) | 254.0 (0.178) | 112.1 (0.637) |
| 50% (1,090) | 297.4 (0.038) | 297.0 (0.039) | 183.5 (0.406) |

D = admitted(threshold) - admitted(learned_class): 10% -0.269, 25% -0.454, 50% -0.369. The threshold_input_order row gives D of -0.006, +0.005 and -0.001 at the three budgets. Per-seed learned_class caught counts at 25% range from 107 to 121.

**Reading.** The learned class beats uniform (E03 uniform at 25% was 75.5 caught) but loses about 140 caught to the threshold at 25%. This is a larger gap than E03's ts_online (D -0.172). The cause is not tested here. One candidate, not checked: at K = 1 over the whole unverified pool (caveat 1), each group draws its own Beta sample from a cell posterior whose prior mass is 2.0, so the top of the ranking may be set by sampling noise across about 2,000 candidates rather than by agreement. The phase-1 loop uses batch candidate sets, so it could behave differently. That would need its own pre-registered entry.

**What it changes.** Claim 2 is not supported for the shipped LearnedAllocator on this pair (GSM8K, majority-vote proxy, cell100, one task). L5a extends the E03 kill from ts_online to the class phase 1 would train with. The label-free threshold is still the best deployable allocator tested here. E11 (the hyperparameter grid for this class) should not run as planned. The rules say a killed class is not retuned inside this run, and a grid over refit_every and prior_strength would have to close a 0.45 admitted-rate gap, which the pre-registration does not allow in this run. Any variant (per-batch candidate set, feature scaling, cell30) needs its own pre-registered entry. Phase-1 funding stays Rustom's decision.

## L5a 2026-10-18 kill

**Verification (orchestrator, 2026-10-18, nothing rerun).** This entry confirms the run-agent entries above (the pre-registration and the kill) against the raw files. Checks: result.json and run.log agree. Admitted rates and D recompute from the per-seed caught lists (1 - caught/309). Both bootstrap 95% percentile CIs recompute from the 200 stored draws with np.percentile (group [-0.472, -0.361], cluster [-0.495, -0.385]). No NaN, and no bootstrap draw above 0 (group max -0.348, cluster max -0.357). exit_code 0, done.flag present, no warning or traceback lines in run.log. Code read in avrl/exp/l5a_learned_class.py: the records carry the global group index, and LearnedAllocator.select returns the idx field (allocators.py lines 135 and 147), so the chosen index is the global one. update() receives only the group just verified. The 22 features are cheap+emb columns with no gold field. The pre-registered design was followed as written. One deviation in route only: the run went through bin/gpu_run.sh on pine (CPU work), as the ground rules require for long jobs, where the spec said "local".

**Verdict: KILL.** Point D(learned_class) at 25% is -0.454, which meets the kill rule (point D <= 0). Both 95% CIs lie entirely below 0, and 0 of 200 resamples in each bootstrap have D > 0.

**Hypothesis (unchanged from the pre-registration).** The shipped LearnedAllocator admits fewer corrupted rewards than the label-free agreement threshold at a 25% budget.

**What was run.** Proxy: majority vote on GSM8K rollouts from Qwen2.5-1.5B-Instruct (p0 feats). Pool: cell100 buckets with at least 3 groups, seed-0 quarter fit split dropped, 2,180 sim groups in 75 buckets, 309 proxy errors. Features: 22 cheap+emb columns, unscaled. Arms: threshold (seeded tie-break), threshold_input_order, learned_class (shipped defaults, K = 1, candidates = all unverified groups per visit). Budgets 218, 545 and 1,090 groups. 8 seeds for the threshold and learned_class point runs. Bootstraps at 25%: 200 group resamples (BOOT_SEED 0, paired with E03) and 200 bucket-cluster resamples (seed 123). Wall time 535 s. sklearn ConvergenceWarnings: 0 in the point runs and in both bootstraps.

**Numbers.** admitted = 1 - caught/309 (lower is better), caught in parentheses. E03 rows are from avrl/runs/E03/result.json.

| arm | labels used | 10% (218) | 25% (545) | 50% (1,090) |
|---|---|---|---|---|
| uniform (E03) | no | 0.910 (27.9) | 0.756 (75.5) | 0.491 (157.3) |
| threshold (8-seed mean) | no | 0.492 (156.9) | 0.183 (252.5) | 0.038 (297.4) |
| threshold_input_order | no | 0.498 (155.0) | 0.178 (254.0) | 0.039 (297.0) |
| greedy_online (E03 own code, related arm, not the shipped class) | yes, online | 0.579 (130.1) | 0.184 (252.3) | 0.041 (296.3) |
| ts_online (E03) | yes, online | 0.718 (87.3) | 0.354 (199.5) | 0.085 (282.6) |
| learned_class (shipped LearnedAllocator, L5a) | yes, online | 0.761 (73.9) | 0.637 (112.1) | 0.406 (183.5) |

D = admitted(threshold) minus admitted(arm). learned_class: -0.269 at 10%, -0.454 at 25%, -0.369 at 50%. threshold_input_order: -0.006, +0.005, -0.001. E03 at 25% for comparison: ts_online -0.172 (bootstrap CI [-0.230, -0.142]); greedy_online -0.001 (CI [-0.044, +0.017]).

At 25%: learned_class caught 112.1 (per seed 107 to 121). Threshold caught 252.5 (per seed 251 to 256). Group bootstrap D mean -0.414, 95% CI [-0.472, -0.361]. Cluster bootstrap D mean -0.436, 95% CI [-0.495, -0.385]. Share of resamples with D > 0: 0.000 in both.

**Verdict against the pre-registered rules.** Pass rule (point D > 0 and both CIs above 0): not met. Kill rule (point D <= 0, or either CI includes 0): met on both counts. Failing conditions: the point D is <= 0, and both the group and the cluster CIs lie entirely below 0 (the pre-registered clarification sends that case to KILL). Verdict: kill.

**What it changes about the thesis claims.**
- Claim 2 (a learned allocator admits fewer corrupted rewards than any fixed rule at equal budget): not supported on this pair for the shipped LearnedAllocator. It loses to the label-free threshold at all three budgets, by about 140 caught at 25%. E03's kill covered only ts_online. It now covers the class that phase 1 would train with. The threshold remains the best deployable rule tested here.
- E11 as planned (a grid over refit_every and prior_strength for this class) is not justified. A grid is not a fix for a 0.45 admitted-rate gap, and the pre-registration bars retuning a killed class inside this run.
- The cause is open. The run agent's candidate cause is that Thompson draws over about 2,000 candidates at K = 1 set the top of the ranking. E03's greedy_online arm has no bucket layer and ties the threshold at 25% (D -0.001), which is consistent with that candidate. It is a hypothesis, not a result. E03's greedy_online is E03's own code (run_online with greedy=True) and was not checked against the shipped GreedyOnlineAllocator. L5b tests the candidate directly.
- L3b (warm bucket prior for the Thompson layer) targets the same layer, so L5b should run before it.
- Claim 3 (the edge comes from prompt-space structure): not tested by this run. It stays not supported on this pair, on the basis of E01 (AUC 0.923 cheap against 0.922 cheap+emb) and the L3c critic's check (20 fold seeds, 25%: 261.55 with 6 cheap features against 260.70 with 22). L4b should still be logged as its own run.
- Phase-1 funding: Rustom's decision. This run is a negative result for the deployable learner. It does not change the labeled ceiling (L3c, inconclusive).

**Follow-ups** (CPU only, each a separate pre-registered entry, also in lit/avrl/EXPERIMENT_BACKLOG.md as L5b to L5d; run in that order).
1. **L5b, Thompson-layer decomposition (under 10 min).** Same protocol and bootstraps as L5a, with the shipped GreedyOnlineAllocator (same cold start, refit cadence and features, ranks by predicted error rate, no Thompson draw) in place of learned_class. Pass: point D > 0 at 25% and both CIs exclude 0 on the positive side. Expected information: high. A tie or a win localizes the L5a loss to the sampling layer and makes L3b the next design. A loss means the regression features are the problem, and L3b is moot.
2. **L5c, batch candidate sets (under 15 min).** Replace the whole-pool candidate set with a random batch of unverified groups per step, sized to the phase-1 per-step group count. Run threshold and learned_class on the same batches at 25%. Expected information: medium-high. It tests caveat 1 of the pre-registration, and the phase-1 loop never sees the whole pool. avrl/PROJECT.md does not state the per-step group count (checked, lines 6 to 18), so it must come from the phase-1 harness spec before pre-registration.
3. **L5d, feature-scaling sensitivity (under 10 min).** Put an online standardizer, fit on verified groups only, in front of the shipped regression. Everything else identical. Pass: D > 0 with both CIs excluding 0. Expected information: medium-low. L3c's scaled, class-balanced regression beat the threshold by about 8 caught at 25% (fold-seed mean), so scaling is the cheapest cause to rule out before the class is retired. It is a sensitivity variant, not a retune, and it runs after L5b and L5c.

**Critic note (L5a, 2026-10-18).** Verdict KILL stands, unchanged. Every number in the pre-registration, the kill entry and the verification entry matches runs/L5a/result.json, run.log and runs/E03/result.json: all caught means, admitted rates and D points; both CIs recomputed from the 200 stored draws (group max -0.348, cluster max -0.357, 0 of 200 above 0); exit_code 0, 535 s wall, 0 ConvergenceWarnings; the E03 comparison rows (uniform, greedy_online, ts_online, their D and CIs); pool of 2,180 groups, 75 buckets, 309 errors. Independent replay (pool rebuilt from feats.parquet, L5a script not imported): shipped LearnedAllocator at 25%, K = 1, whole-pool candidates, seeds 0 to 2 caught 121, 116 and 110, identical to the stored per-seed list, with 0 ConvergenceWarnings. No leakage found: the 22 columns are rollout statistics and question-embedding PCs, and the label reaches the allocator only through update() for the group just verified. Budget is 545 verified groups in both arms (threshold takes the top 545, learned_class makes 545 visits of K = 1 and never repeats a group). The group bootstrap is paired with E03 (same BOOT_SEED and draw code). Eight seeds (per-seed sd 4.2 caught) and 200 resamples are ample for a gap of 0.45 against CI widths of about 0.1.

Two wording corrections, neither changes the verdict. (1) The verification entry says the kill rule is "met on both counts". It is met on one count: point D = -0.454 <= 0. The other clause, "either 95% CI includes 0", is not met, because both CIs lie entirely below 0 ([-0.472, -0.361] and [-0.495, -0.385]). The pre-registered clarification (a CI entirely below 0 while the point D is positive is KILL) does not apply either, since the point D is negative. Both CIs lying below 0 is corroboration, not a triggered clause. The failing condition is: point D(learned_class) at 25% <= 0. The pass rule fails on all three of its clauses (point D > 0, group lower bound > 0, cluster lower bound > 0). (2) "It now covers the class that phase 1 would train with" overstates the scope. The replay is K = 1 over the whole unverified pool, and p1_grpo.py is not in this tree, so "unscaled features as shipped" and the per-step call pattern were checked only against allocators.py, not the loop. Caveat 1 stands and L5c is the test. Read the kill as: the shipped class, replayed on the E03 protocol, loses to the threshold on this pair. It is not yet a test of the batch loop.

Mechanism, read from the code and not tested: in LearnedAllocator.select the regression output p[i] is used only the first time a cell is scored, to seed that cell's Beta pseudo-counts. After that every group in a cell shares one posterior, so within-cell order is random and only cell-level rates plus sampling noise rank groups. Replay diagnostic (critic only, seeds 0 to 2, not a registered result): the 545 picks touch all 75 cells, 17 to 20 percent of picks are unanimous groups (agreement 1), mean pick agreement is 0.60 to 0.62, and the 20 cold-start picks catch 20 of 20. This is consistent with the candidate cause in the ledger and is still a hypothesis.

Heads-up for L5b (critic only, 2 runs, not a result): the shipped GreedyOnlineAllocator through the same K = 1 whole-pool call pattern caught 245 at 25%, against 252.5 for the threshold and 252.25 for E03's own greedy_online. It is deterministic (seeds 0 and 1 gave the same 245, since it never uses the rng) and raised 1 ConvergenceWarning per run. That is D about -0.024, so a tie with the threshold should not be assumed in advance. L5b should pre-register it as a single deterministic run per resample and report ConvergenceWarnings.

Corrected verdict: KILL, unchanged. Failing condition: point D(learned_class) at 25% = -0.454 <= 0; both 95% CIs lie entirely below 0. Scope: shipped LearnedAllocator, whole-pool K = 1 replay, GSM8K, majority-vote proxy, cell100, one task, one proxy.

## L5c 2026-10-18 pre-registration (written before the run, no L5c outcome seen)

**Hypothesis (claim 2, phase-1 batch shape):** on the E03 pool (GSM8K majority-vote proxy, cell100, cheap+emb 22 columns, 2,180 sim groups, 75 buckets, 309 proxy errors), when the shipped LearnedAllocator chooses among a batch of 16 unverified groups per step at a 25% budget, it admits fewer corrupted rewards than the label-free agreement threshold applied to the same batches. L5a killed the class in a whole-pool K = 1 replay. That replay never shows the allocator the batch candidate sets that the phase-1 loop uses (L5a caveat 1). L5c tests that shape. The batch size of 16 is the measured phase-1 step (per the L5c spec).

**What will be run.** Script avrl/exp/l5c_batch_learned_class.py, copied from avrl/exp/l5a_learned_class.py. Launcher avrl/exp/L5c.sh, started through bin/gpu_run.sh on the GPU host (pine, CPU work, run dir avrl/runs/L5c). Launch note: the spec's command shape used relative paths (avrl/exp/...), but gpu_run.sh runs the command from CASCADE/avrl, so that command would not resolve. The run goes through the launcher, which changes to CASCADE first, the same way L5a was launched. The arguments are the spec's. E03, L5a, allocators.py and the p0 files are not edited. allocators.py is imported read-only.

**Batches and budget.** For allocator seed s (0 to 7), perm = np.random.default_rng(s).permutation(2180), cut into consecutive batches of 16 (136 full batches and one of 4, 137 in total). A group not verified in its batch is accepted on the proxy and leaves the pool. Batch i verifies b_i = floor(f*N_i + 1/2) - floor(f*N_(i-1) + 1/2) groups, where N_i is the number of groups consumed through batch i (exact rational arithmetic). Totals are 218 / 545 / 1,090 for every arm and seed, checked in the run. At 25% each full batch verifies 4 of 16, and the last batch of 4 verifies 1. At 10% the last batch verifies 0.

**Arms, paired on identical batches per seed.** threshold_batch: the b_i groups ranked highest by threshold_order (agreement ascending, then n_distinct descending, exact ties in seeded random order with rng default_rng(s)). learned_batch: LearnedAllocator() with class defaults (refit_every 20, prior_strength 2.0). Per batch: records for the batch's groups only, select(records, b_i, rng) with rng default_rng(s), then update() with the b_i verified (X, label, cell) tuples and nothing else.

**Bootstraps at 25% only.** Group: 200 resamples of the sim groups (BOOT_SEED 0, the same draws as E03 and L5a). Bucket-cluster: 200 resamples of the 75 buckets with replacement (seed 123, repeated buckets relabeled as new clusters). Each resample rebuilds the pool, its batches (permutation seed 0) and both arms (allocator seed 0), and recomputes D against its own error total. The spec sets one allocator seed per resample. The point D uses the 8-seed means. The bootstrap intervals therefore include single-seed allocator noise that the point estimate averages out.

**Metric.** admitted = 1 - caught / 309, with caught the mean over 8 seeds. D = admitted(threshold_batch) - admitted(learned_batch) = (caught_learned - caught_threshold) / 309. Positive D means the learned class admits fewer corrupted rewards.

**Pass rule (25%).** Point D > 0, AND the lower bound of the 95% percentile interval is > 0 under BOTH the group bootstrap AND the bucket-cluster bootstrap.

**Kill rule (25%).** Point D <= 0, OR the 95% interval of either bootstrap includes 0, OR either interval lies entirely below 0. The three clauses are fixed before the run.

**Descriptive only.** D and caught at 10% and 50%, threshold_batch caught at 25% against the whole-pool 252.5 from L5a, and the ConvergenceWarning count. Checks before the verdict is read: budgets exact for every arm and seed, no NaN, ConvergenceWarnings reported.

**Caveats (stated before the run).**
1. Batch layout depends on the allocator seed, so the 8 seeds give 8 different batch layouts. Between-seed variation is part of the point estimate's noise.
2. Features are unscaled, as shipped. The class's feature layout is 22 columns (the class docstring says 21, see L5a).
3. One task (GSM8K), one proxy (majority vote), cell100 only, and one step shape (batch 16).
4. The class runs at its defaults. Nothing is retuned inside this run.

**Expected information.** Medium-high. A kill closes the phase-1 shape of the shipped class as well as the whole-pool shape, and L5b (Thompson-layer decomposition) becomes the fallback test. A pass means claim 2 holds for the class in the shape phase 1 would train, and the next step is L5d (feature-scaling sensitivity).

## L5c 2026-10-18 kill (shipped LearnedAllocator vs label-free threshold, batch-16 shape)

**Numbers.** admitted = 1 - caught/309 (lower is better). Caught is the 8-seed mean, with the sd across seeds in parentheses. Batch layouts come from the permutation of each seed. Source: avrl/runs/L5c/result.json.

| arm | labels used | 10% (218) | 25% (545) | 50% (1,090) |
|---|---|---|---|---|
| threshold_batch | no | 0.561 (135.8, sd 4.1) | 0.235 (236.5, sd 4.5) | 0.061 (290.3, sd 2.3) |
| learned_batch (shipped class) | yes, online | 0.849 (46.8, sd 3.6) | 0.706 (90.8, sd 4.3) | 0.447 (171.0, sd 7.3) |

D = admitted(threshold_batch) minus admitted(learned_batch): -0.288 at 10%, **-0.472 at 25%**, -0.386 at 50%.

At 25%: learned_batch per seed 85 to 97 caught. threshold_batch per seed 230 to 243. Group bootstrap D mean -0.446, 95% CI [-0.514, -0.361], 0 of 200 resamples above 0. Cluster bootstrap D mean -0.477, 95% CI [-0.540, -0.410], 0 of 200 above 0.

**Verdict against the pre-registered rules.** Pass rule: not met (point D is negative, and both CIs lie entirely below 0). Kill rule: met. The point D is <= 0, and both the group and the cluster CI lie entirely below 0. Verdict: **kill**. The shipped LearnedAllocator loses to the label-free threshold in batch-16 shape by about 146 caught at 25%.

**Descriptive.** threshold_batch at 25% catches 236.5, against 252.5 for the whole-pool threshold in L5a and E03. Batch shape costs the threshold about 16 caught. learned_batch catches 90.8, against 112.1 in the whole-pool L5a replay. A uniform-random verifier at 25% would catch 545/2180 x 309 = 77.3 in expectation (computed, not run). So at 25% in this shape the shipped class is only about 14 caught above random, while the threshold is about 159 above. ConvergenceWarnings: 0 in the point runs and in both bootstraps.

**Scope.** Claim 2 for the shipped LearnedAllocator, replayed in batch-16 candidate sets, on GSM8K, majority-vote proxy, cell100, cheap+emb (22 columns), at class defaults. The kill covers the phase-1 shape of this class. It does not test the phase-1 loop itself: p1_grpo.py is not in this tree, and the step size of 16 comes from the L5c spec. The mechanism behind the loss stays a hypothesis. The L5a candidate was Thompson draws over cells, and it is not tested here. L5b (greedy_online, whole-pool) is the spec's fallback test for that layer. A pass in L5b would localize the loss but would not reopen the batch-shape kill.

**Spec deviations and caveats.**
1. Launch: the spec's command shape used relative paths (avrl/exp/...). gpu_run.sh runs the command from CASCADE/avrl, so that command would not resolve. The run used avrl/exp/L5c.sh, which changes to CASCADE first, as L5a did. The arguments are the spec's.
2. Bootstraps use one allocator seed (seed 0) per resample, as the spec says. The point D uses the 8-seed means. Per-seed sd at 25% is about 4.3 caught, or about 0.014 in D. That is small against the 0.47 gap, so the single seed does not change the verdict.
3. The sim pool is 2,180 groups in 75 buckets with 309 proxy errors, identical to L5a. Budgets: 218 / 545 / 1,090 for every arm and seed. The short last batch (4 groups) verifies 1 at 25% and 0 at 10%.

**Critic check (2026-10-18).** Verdict KILL stands.
- Run: exit_code 0. Checks block in result.json: budget_exact_all_arms_seeds true, all_values_finite true, 0 ConvergenceWarnings in the point runs and in each bootstrap. Wall 43 s (30 s bootstrap, 8 workers, pine).
- D point recomputed from the stored means: (90.75 - 236.50)/309 = -0.4717. Admitted rates recomputed: 0.235 and 0.706 at 25%, and all six table entries round correctly.
- Bootstrap CIs recomputed from the 200 stored draws each: group [-0.5138, -0.3607], cluster [-0.5403, -0.4099], both matching the stored values. Maximum draws: group -0.352, cluster -0.386. No draw is above 0.
- Independent replay, not importing the L5c module: own pool build (same filters and fit split), integer budget (N + 2) // 4 for 25%, pandas sort for the threshold, LearnedAllocator called directly with records restricted to each batch, and an assert that each selection lies inside its batch. Per-seed caught matches the stored values exactly for both arms, all 8 seeds. Threshold: 230, 237, 240, 234, 234, 243, 232, 242. Learned: 94, 97, 97, 88, 85, 88, 88, 89. Verified count 545 in every run. 0 ConvergenceWarnings.
- Pre-registered language check: the kill clauses in the log (point D <= 0, CI includes 0, CI entirely below 0) match the verdict_reason in result.json.

## L5c 2026-10-18 kill

**Verification (orchestrator, 2026-10-18).** Read avrl/runs/L5c/result.json, run.log, exit_code (0) and done.flag. Recomputed D from the stored means and totals, the admitted rates, and both 95% percentile CIs from the 200 stored draws of each bootstrap (group [-0.5138, -0.3607], cluster [-0.5403, -0.4099], no draw above 0). Ran a new CPU replay, avrl/exp/l5c_critic_replay.py, which does not import the L5c script. It rebuilds the pool, the batches and both arms (LearnedAllocator imported read-only), and the first 5 draws of each bootstrap. Per-seed caught counts match the stored values for both arms at 10, 25 and 50 percent, all 8 seeds. Budgets are exact (218 / 545 / 1,090) for every arm and seed. The first 5 group and 5 cluster bootstrap draws match with max abs diff 0. ConvergenceWarnings in the replay point runs: 0. Output: avrl/runs/L5c/critic_replay.json. Nothing in the L5c run itself was rerun.

**Hypothesis (claim 2, shipped class, phase-1 batch shape).** As pre-registered in the L5c entry above: with batches of 16 unverified groups per step, the shipped LearnedAllocator admits fewer corrupted rewards than the label-free agreement threshold applied to the same batches, at a 25% budget.

**What was run.** Model and task: Qwen2.5-1.5B-Instruct rollouts on GSM8K, majority-vote proxy (avrl/runs/p0/feats.parquet). Pool: cell100 buckets with at least 3 groups, seed-0 quarter fit split dropped, giving 2,180 sim groups in 75 buckets with 309 proxy errors. Features: 22 cheap+emb columns, unscaled, class defaults (refit_every 20, prior_strength 2.0). Batches: a seeded permutation for each allocator seed 0 to 7, cut into consecutive batches of 16 (137 batches per seed, the last one of 4). Budgets: 10%, 25%, 50% (218 / 545 / 1,090 groups). Bootstraps at 25%: 200 group resamples (BOOT_SEED 0) and 200 bucket-cluster resamples (seed 123), allocator seed 0. Run through avrl/exp/L5c.sh on the GPU helper (CPU work, 43 s wall).

**Table.** admitted = 1 - caught/309 (lower is better), caught as the 8-seed mean. Uniform random is expected, computed, not run.

| arm | labels used | 10% (218) | 25% (545) | 50% (1,090) |
|---|---|---|---|---|
| uniform random (expected, computed) | no | 0.900 (30.9) | 0.750 (77.3) | 0.500 (154.5) |
| threshold_batch | no | 0.561 (135.8) | 0.235 (236.5) | 0.061 (290.3) |
| learned_batch (shipped LearnedAllocator) | yes, online | 0.849 (46.8) | 0.706 (90.8) | 0.447 (171.0) |
| whole-pool threshold (L5a, reference) | no | 0.492 (156.9) | 0.183 (252.5) | 0.038 (297.4) |
| whole-pool LearnedAllocator (L5a, reference) | yes, online | 0.761 (73.9) | 0.637 (112.1) | 0.406 (183.5) |

D = admitted(threshold_batch) minus admitted(learned_batch): -0.288 at 10%, **-0.472 at 25%**, -0.386 at 50%.

At 25%, learned_batch per seed ranges 85 to 97 caught and threshold_batch 230 to 243. The group bootstrap gives D mean -0.446, 95% CI [-0.514, -0.361], 0 of 200 resamples above 0. The cluster bootstrap gives D mean -0.477, 95% CI [-0.540, -0.410], 0 of 200 above 0.

**Verdict against the pre-registered rules (25%).** Point D = -0.472 (replay -0.4717). Pass rule (point D > 0 and both CIs above 0): not met. Kill rule (point D <= 0, and both CIs entirely below 0): met. **Verdict: kill.**

Descriptive: learned_batch is about 13.5 caught above uniform random (90.8 against 77.3), and threshold_batch is about 159 above it. Batch shape costs the threshold 16 caught (236.5 against 252.5 whole-pool) and the learned class about 21 (90.8 against 112.1 whole-pool).

**What it changes about the thesis claims.**
- Claim 2 (a learned allocator admits fewer corrupted rewards than any fixed rule at equal budget): not supported. The shipped LearnedAllocator loses to the label-free threshold in whole-pool shape (L5a) and in phase-1 batch shape (this entry), at every budget. The class phase 1 would train is killed on this pair in the shape it would use. The kill covers this class on this pair (GSM8K, majority-vote proxy, cell100, cheap+emb 22 columns, class defaults). It does not cover the phase-1 loop itself (p1_grpo.py is not in this tree) or any variant.
- Claim 3 (the edge comes from prompt-space structure): unchanged, not supported on this pair (E01, L4b). L5c does not test it.
- Mechanism: still a hypothesis. The shipped ranking takes Thompson draws from cell Beta posteriors. Near the 0.14 base rate, an unseen cell's prior is Beta(0.28, 1.72) (prior_strength 2.0 times the predicted rate), which has most of its mass near 0 and a heavy upper tail, so the draws are noisy. This follows from the prior's shape and is not measured. L6a and L6b test it.

Phase-1 funding stays Rustom's decision.

**Follow-ups (in lit/avrl/EXPERIMENT_BACKLOG.md under Proposed by the loop).**
1. L6a: batch-shape decomposition with GreedyOnlineAllocator (no Thompson layer), same batches and bootstraps. Pass: point D > 0 at 25% with both CIs above 0. Information: high. It decides whether the phase-1 failure is the Thompson layer or the regression.
2. L6b: analysis-only selection diagnostic for the shipped class in batch shape (overlap with the top-b by predicted rate, Thompson-versus-prediction rank correlation, share of draws from unverified cells). Information: medium, and nearly free on CPU. It tells us whether L6c is worth pre-registering.
3. L6c: prior_strength 200 variant in batch shape, pre-registered as a separate variant and run only if L6a passes. Information: medium. It is not a retune of the killed class.

**Critic note (L5c, 2026-10-18).** Verdict KILL stands, unchanged. Checked against runs/L5c/result.json, run.log (exit_code 0, 43.3 s wall, no traceback; the three "Warning" string hits in run.log are the ConvergenceWarnings count lines, all 0), the script avrl/exp/l5c_batch_learned_class.py (there is no avrl/exp/L5c.py, the launcher is L5c.sh), and runs/L5a/result.json and runs/E03 for the reference rows. Every number in the entries matches the raw files: caught means and sds (population sd, ddof 0), admitted rates, the three D points (-0.2880, -0.4717, -0.3859), both CIs recomputed from the 200 stored draws each (group [-0.5138, -0.3607], cluster [-0.5403, -0.4099], max draws -0.352 and -0.386, 0 above 0), the uniform-random expectations (30.9, 77.25, 154.5), the whole-pool L5a rows (156.9 / 252.5 / 297.4 threshold, 73.9 / 112.1 / 183.5 learned), the pool (2,180 groups, 75 buckets, 309 errors), 137 batches (136 x 16 + 4), and the budget schedule 218 / 545 / 1,090 with the verified count exact for every arm and seed. The bootstrap draw code is the same as E03 and L5a (BOOT_SEED 0 integer draws, cluster seed 123). Independent replay (my own pool build, integer budget schedule (N + 2) // 4, python sorted for the threshold, LearnedAllocator called directly with batch-only records, asserts that every selection lies inside its batch and that no group repeats): per-seed caught at 25% identical to the stored values for both arms, all 8 seeds, 545 verified in each run. No leakage found: the 22 columns are rollout statistics and question-embedding PCs (gold, majority, pass_rate, majority_correct and proxy_error are not in COLS, asserted), cell100 is a KMeans id over question embeddings, the threshold reads agreement and n_distinct only, and the label reaches the learned arm only through update() for the groups it just verified. Budget accounting is equal across the two arms (same b_i per batch). Seeds and resamples are adequate: the gap is about 146 caught, per-seed paired D has sd 0.019 and the 8-seed mean has SE 0.007, and the draw nearest to 0 is -0.352.

Positive control (critic only, not a registered result, not in result.json): the shipped GreedyOnlineAllocator through the same batch harness at 25% caught 227.0 on average (per seed 225, 228, 229, 221, 228, 236, 216, 233; 1 ConvergenceWarning over the 8 runs), D about -0.031 against threshold_batch 236.5, and random choice inside the same batches caught 72.5 on average. So the batch harness does not handicap a regression-ranked learner, and the L5c loss sits in the cell-Thompson layer that LearnedAllocator adds. That is consistent with the ledger's candidate and with the L6a plan, and it does not test the mechanism. A second untested candidate: update() gives a cell verified before the first fit the default Beta(2, 2) prior plus its label, so a cold-start cell with an error is Beta(3, 2), mean 0.6 against a 0.14 base rate.

Corrections, none changes the verdict. (1) In the "## L5c 2026-10-18 kill" entry, "Kill rule (point D <= 0, and both CIs entirely below 0): met" states the pre-registered rule as a conjunction. The rule is a disjunction: KILL if point D <= 0, OR either interval includes 0, OR either interval lies entirely below 0. Two clauses fire here (point D = -0.4717 <= 0; both intervals lie entirely below 0) and the "includes 0" clause does not. The pass rule fails on all three of its clauses. (2) "Per-seed sd at 25% is about 4.3 caught, or about 0.014 in D" uses the learned arm's sd alone. Paired per-seed D at 25% ranges from -0.440 to -0.502 with sd 0.019, and the 8-seed mean has SE 0.007. Immaterial against a gap of 0.47. (3) "The class phase 1 would train is killed on this pair in the shape it would use" overstates the scope. Read it as: killed in a batch-16 replay on this pair. The 16 comes from the L5c spec, p1_grpo.py is not in this tree, and no other batch size was run. The entry's own Scope paragraph says this. (4) Housekeeping: the log holds two L5c kill sections (the one headed "kill (shipped LearnedAllocator vs label-free threshold, batch-16 shape)" and "## L5c 2026-10-18 kill"). They agree on every number. Neither was edited.

Corrected verdict: KILL, unchanged. Failing conditions: point D(learned_batch) at 25% = -0.4717 <= 0, and both the group and cluster 95% intervals lie entirely below 0. Scope: shipped LearnedAllocator at class defaults, batch-16 replay, GSM8K majority-vote proxy, cell100, cheap+emb 22 columns, one pair, 8 allocator seeds. The whole-pool L5a kill stands.

## E04 2026-10-18 pre-registration (written before the GPU step, no E04 outcome seen)

**Hypothesis.** On the E01 GSM8K rollouts (Qwen2.5-1.5B-Instruct, 3,000 prompts x 8 rollouts), a gold-free judge proxy has error structure in prompt-embedding space that the cheap features miss, and its error set overlaps less than half with the majority-vote error set, so proxy type is a lever. Secondary claim-2 readout: at 25%, on the judge-error label, the online learned arm (ts_online, E03 protocol) admits fewer corrupted groups than the label-free agreement threshold.

**What will be run.**
- GPU step: avrl/exp/e04_judge_score.py, launched by avrl/exp/E04.sh through bin/gpu_run.sh on the GPU host pine (run dir avrl/runs/E04). Scores all 24,000 rollouts with Qwen2.5-0.5B-Instruct (vLLM, gpu_memory_utilization 0.5, max_tokens 1, top-20 logprobs). Prompt: "Question: <question>. Proposed final answer: <answer>. Is the proposed answer correct? Reply Yes or No." The answer is the rollout's extracted answer string. Gold never appears in the prompt. p_yes = P(Yes) / (P(Yes) + P(No)) from the first-token logprobs. If one of the two tokens is outside the top 20, it is set to the lowest returned logprob, and the fallback is counted.
- CPU step: avrl/exp/e04_judge_structure.py in the shell (active-audit .venv). Reads avrl/runs/E04/judge_scores.parquet and avrl/runs/p0/feats.parquet. The embedding columns (pc0 to pc15), cell30, cell100, cell300 and the 6 cheap columns are read from feats.parquet as p0_analyze.py wrote them, not recomputed, so the features are identical to E01.
- Claim-2 replay: avrl/exp/e04_judge_replay.py, a copy of avrl/exp/e03_threshold_bootstrap.py with LABEL = "judge_error" and nothing else changed, run on avrl/runs/E04/feats_judge.parquet (the p0 feats plus a judge_error column).

**Judge proxy definition.** Per group, the judge-selected answer is the distinct answer with the highest mean p_yes over its rollouts. Exact ties go to the lower answer string in sort order. judge_error = judge-selected answer is not gold. proxy_error stays the E01 majority label, unchanged.

**Descriptive gates (recorded before the verdict is read).**
- G1: sd of p_yes over all 24,000 rows. Below 0.02 means the judge is constant. That is recorded as a failed proxy with no verdict on proxy type.
- G2: fraction of the 3,000 groups where the judge-selected answer equals the majority answer.

**Structure readout (backlog E04 rule).** Metrics: judge error rate overall and on the confident subset (agreement >= 0.75); judge error rate among unanimous and split groups; AUC for judge_error with cheap and with cheap+emb features (5-fold balanced logistic regression, as in p0_analyze.py) and their difference; neighbor ratio at cell100; Jaccard(proxy_error groups, judge_error groups).
- PASS: AUC(cheap+emb) minus AUC(cheap) >= 0.05 AND Jaccard < 0.5.
- KILL: Jaccard > 0.8, OR (AUC gain < 0.05 AND cell100 neighbor ratio < 1.5).
- INCONCLUSIVE: anything else (for example Jaccard between 0.5 and 0.8 with no kill).

**Claim-2 readout (E03 rule on the judge label).** 25% budget (545 of the 2,180 sim groups), 8 allocator seeds, 200 group bootstrap resamples with BOOT_SEED 0, cell100 buckets, cheap+emb (22 columns). D = admitted(threshold) minus admitted(ts_online).
- PASS: point D > 0 AND the 95% percentile interval excludes 0.
- KILL: point D <= 0, or the interval includes 0.
The 10% and 50% budgets are descriptive only.

**Joint verdict.** A go for a judge-proxy allocator requires both readouts to PASS. KILL if no readout passes and at least one is KILL. INCONCLUSIVE otherwise, including when the two readouts disagree (one PASS, one KILL).

**Caveats (stated before the run).**
1. One judge (0.5B), one task (GSM8K), one rollout policy (1.5B), cell100 buckets for the replay only.
2. The replay's built-in reproduction check compares against the E01 proxy-label json, so it will print matches_E01 False. That is expected, not a failure. The replay's internal "exp" field says E03; result.json labels the readout E04.
3. 35 of the 24,000 rollouts have no extracted answer (empty string). They are scored like any other answer, and an empty string can be the judge's pick.
4. Bootstrap intervals include single-seed allocator noise (E03 convention).

**Expected information.** Medium. Both readouts passing would open the judge as a proxy lever, the first point on the structure-versus-gain curve. A structure kill on Jaccard > 0.8 would say the judge makes the same errors as majority vote.

## E04 2026-10-18 kill (judge proxy: Qwen2.5-0.5B-Instruct, E01 GSM8K rollouts)

Run as pre-registered above. No outcome was seen before the gates were recorded. Files: avrl/runs/E04/result.json (all metrics), structure.json, replay/result.json, judge_scores.parquet, judge_gpu_summary.json, run.log (GPU), structure.log, replay.log, assemble.log.

**Gates (read before the verdict).** G1: p_yes sd over 24,000 rows = 0.107, so the judge is not constant (mean 0.470, 5th to 95th percentile 0.29 to 0.65). No fallback tokens were needed (both Yes and No in the top 20 for every row). G2: the judge-selected answer equals the majority answer in 45.7% of groups.

**Structure readout: KILL.** Jaccard is 0.22, so the kill on Jaccard > 0.8 does not fire. The AUC gain is 0.0001 (< 0.05) and the cell100 neighbor ratio is 1.00 (< 1.5), which fires the second kill clause.

| quantity | value |
|---|---|
| judge error, all groups | 56.5% (1,694 of 3,000) |
| judge error, confident subset (agreement >= 0.75, n = 1,551) | 34.8% |
| judge error, unanimous (559 groups) / split (2,441 groups) | 0.5% / 69.3% |
| proxy (majority) error, all / unanimous / split | 14.5% / 0.5% / 17.7% |
| AUC judge error: cheap / cheap+emb / gain | 0.820 / 0.820 / +0.0001 |
| AUC proxy error, recomputed: cheap / cheap+emb | 0.923 / 0.922 (matches E01) |
| neighbor ratio, judge error, cell100 | 1.00 |
| overlap: proxy 436, judge 1,694, both 383 | Jaccard 0.219 |

**Claim-2 readout: KILL.** E03 replay copy with LABEL = judge_error, 25% budget (545 of 2,180 sim groups, 1,224 judge errors in the pool, base rate 56.2%), 8 seeds, 200 group bootstrap resamples, BOOT_SEED 0. Admitted corrupted groups at 25%: uniform 0.750, threshold (label-free) 0.613, greedy_online 0.638, ts_online 0.673. D(ts_online) = -0.060, 95% CI [-0.077, -0.048], so the label-free threshold admits fewer. D(greedy_online) = -0.025, CI [-0.040, -0.020]. Descriptive: D(ts_online) at 10% = -0.030 and at 50% = -0.089. The replay's built-in reproduction check prints matches_E01 False. That is expected because its reference is the proxy-label E01 json. The replay json's "exp" field says E03.

**Joint verdict: KILL.** Both readouts kill, so there is no go for a judge-proxy allocator on this task and judge. Proxy type does not rescue the allocator here.

**Post-hoc (descriptive, added after the readouts were read, not part of any verdict).**
- The judge-selected answer is correct in 43.5% of groups. A uniformly random rollout would be correct 64.2% of the time, and majority vote 85.5%. The judge's pick is worse than random.
- Pooled across questions, p_yes separates correct from wrong rollouts only weakly: rollout-level AUC 0.599 (mean p_yes 0.484 for correct, 0.445 for wrong).
- Within split groups the judge's top pick is wrong 69% of the time, and this barely depends on the margin between the top two answers (top-two gap < 0.01: 70% wrong; gap >= 0.05: 69% wrong). So the errors are a ranking failure, not noise near ties. The mechanism (for example, plausible-looking wrong numbers) was not tested.
- Scoring noise: 98.8% of (question, answer) pairs with duplicate rollouts have identical p_yes. The rest vary by up to 0.062, which is consistent with batch-dependent bf16 numerics. Not tested further. The mean over duplicates is used by the pick, so this does not drive the result.

**What it changes.** The 0.5B judge fails as a verifier on this task, and its error set is nearly disjoint from majority vote (Jaccard 0.22) but unstructured in prompt space and not learnable from cheap features. So the proxy-type lever is not confirmed. A stronger judge (for example Qwen2.5-1.5B-Instruct, same protocol) is the natural next test of whether the failure is capacity. It was not run here.

## E04 2026-10-18 kill

**Scope.** Verification entry for the E04 kill above (the entry that begins "## E04 2026-10-18 kill (judge proxy"). The verdict is unchanged. This entry re-derives the headline numbers from the raw files instead of trusting the run report.

**Hypothesis (as pre-registered above).** A gold-free judge proxy (Qwen2.5-0.5B-Instruct P(Yes)) has prompt-space error structure that cheap features miss, and its error set overlaps less than half with the majority-vote error set (structure readout). Secondary claim-2 readout: at a 25% budget, ts_online admits fewer judge-error groups than the label-free agreement threshold.

**What was run.** GPU (one job, pine): Qwen2.5-0.5B-Instruct via vLLM scores all 24,000 E01 rollouts (3,000 GSM8K train prompts x 8 rollouts from Qwen2.5-1.5B-Instruct). The judge pick per group is the distinct answer with the highest mean p_yes. CPU: 5-fold balanced logistic regression on 6 cheap and 22 cheap + emb features, cell100 neighbor ratio, Jaccard. Claim-2 replay: the E03 copy with LABEL = judge_error, cell100 buckets, 2,180 sim groups, 1,224 judge errors in the pool, budget 545 (25%), 8 allocator seeds, 200 group bootstrap resamples, BOOT_SEED 0.

**Verification (this entry).**
- avrl/exp/e04_verify.py recomputes the headline numbers. The judge pick is a plain loop, and judge correctness is read from rollouts.correct instead of a gold string match. Zero mismatches against the run's judge_error column. All 24,000 scores align with rollouts (answer strings identical). Output: avrl/runs/E04/verify/verify.json.
- avrl/exp/e04_judge_replay.py rerun on avrl/runs/E04/feats_judge.parquet reproduces every admitted rate and both bootstrap CIs exactly (max abs diff 0.0). Output: avrl/runs/E04/verify/replay_judge_label/result.json.
- avrl/exp/e04_replay_proxycheck.py is the same replay with LABEL = proxy_error, run on the same feature file. It reproduces E03 exactly (max abs diff 0.0, matches E01 True; admitted threshold 0.183, ts_online 0.354; D(ts_online) -0.172, CI [-0.230, -0.142]). So the replay path is sound and the judge-label result depends only on the label. Output: avrl/runs/E04/verify/replay_proxy_label/result.json.
- The vLLM "EngineCore died unexpectedly" line comes after "Shutdown complete" and the job exited 0 with complete outputs. It is benign teardown noise.

**Numbers (verified).**

| quantity | E04 value | baseline / note |
|---|---|---|
| p_yes sd over 24,000 rows (gate: failed if below 0.02) | 0.107 | not constant, so a verdict applies |
| judge pick equals majority answer (gate) | 45.7% of groups | |
| judge error, all groups | 56.5% (1,694 of 3,000) | majority proxy 14.5% (436) |
| judge error, confident (agreement >= 0.75, n = 1,551) | 34.8% | majority proxy 1.1% (E01) |
| judge error, unanimous (n = 559) / split (n = 2,441) | 0.5% / 69.3% | majority proxy 0.5% / 17.7% |
| judge pick correct, split groups | 30.7% | uniform pick over distinct answers 29.7%; uniform over rollouts 56.1%; majority 82.3% |
| AUC judge error, cheap / cheap + emb / gain | 0.820 / 0.820 / +0.0001 | majority proxy error 0.923 / 0.922 |
| neighbor ratio, judge error, cell100 | 1.00 (0.9997) | majority proxy 1.03 (matches E01) |
| Jaccard(proxy error, judge error) | 0.219 (383 both, 53 proxy-only, 1,311 judge-only) | |
| claim-2 admitted at 25%: uniform / threshold / greedy_online / ts_online | 0.750 / 0.613 / 0.638 / 0.673 | same arms on the E03 proxy label: 0.756 / 0.183 / 0.184 / 0.354 |
| D(ts_online) at 25%, 95% group bootstrap CI | -0.060, [-0.077, -0.048] | E03 proxy label: -0.172, [-0.230, -0.142] |
| D(greedy_online) at 25%, 95% CI | -0.025, [-0.040, -0.020] | E03 proxy label: -0.001, [-0.044, +0.017] |
| rollout-level AUC of p_yes against correctness (post-hoc) | 0.599 | not a pre-registered metric |

**Verdict against the pre-registered rules.**
- Gates: p_yes sd 0.107 is above the 0.02 constant-judge line, so the proxy gets a verdict.
- Structure: KILL. Pass needs AUC gain >= 0.05 and Jaccard < 0.5, and the gain is 0.0001, so there is no pass. Kill clause 2 fires (gain < 0.05 and cell100 neighbor ratio < 1.5). Kill clause 1 (Jaccard > 0.8) does not fire, since Jaccard is 0.22.
- Claim-2 readout: KILL. Point D(ts_online) = -0.060 is at or below 0, and its CI excludes 0 on the negative side.
- Joint: KILL. Neither readout passes, and both kill.

**Mechanism (post-hoc, descriptive, not a verdict input).** 1,691 of the 1,694 judge errors (99.8%) fall in split groups. The 3 unanimous-group errors are the rest. Inside split groups the judge's pick is correct 30.7% of the time, which is chance among distinct answers (29.7%). The run report's "judge pick worse than random" (43.5% against 64.2%) compared against a rollout-weighted baseline, which overstates the failure. The correct reading is that the judge is at chance inside split groups. Its label is therefore close to a noisy copy of the split indicator, which explains why cheap features reach AUC 0.82 and embeddings add nothing, and why the agreement threshold beats ts_online on this label.

**What it changes about the thesis claims.**
- Claim 2 (a learned allocator admits fewer corrupted rewards than any fixed rule at equal budget): not supported. On the judge-error label, ts_online admits more groups than the label-free threshold at 25% (and at 10% and 50%, descriptively). With E03 (proxy label) and the L5a and L5c kills of the shipped class, no GSM8K label tested so far gives a positive claim-2 readout.
- Claim 3 (the edge comes from structure in prompt space): not supported on this task. Judge error has no prompt-space structure beyond cheap features (gain +0.0001, neighbor ratio 1.00). Changing the proxy from majority vote to a 0.5B judge did not produce findable structure. Its errors are not a usable second error set: 1,311 judge-only errors fall in groups where the majority answer is right, so they are mostly noise.
- Proxy-type lever: not confirmed for a 0.5B judge. The judge is a useless verifier at this size, so this is a capacity question, not evidence against judges in general. A 4B judge is the test (L7b).

**Follow-ups, with expected information value.**
1. L7b (GPU, one job, about 10 min): same E04 protocol with Qwen3-4B as judge (thinking off, gpu_memory_utilization 0.85). Highest value: it separates a 0.5B capacity failure from a failure shared by small judges on this task. Rules unchanged.
2. L7a (CPU, descriptive): whether the 0.5B judge carries any surface signal inside split groups (answer copied from the question, magnitude, length). Medium value: if no surface feature beats the 29.7% chance, a rubric prompt or a larger judge is the only route left for this lever.
3. L7c (CPU): the judge score as an extra feature for the majority-vote label, not as a verifier. Low to medium value: agreement alone gives AUC 0.92, so the expected gain is small.

Files: avrl/runs/E04/result.json, structure.json, replay/result.json, verify/verify.json, verify/replay_judge_label/result.json, verify/replay_proxy_label/result.json, judge_scores.parquet, feats_judge.parquet, run.log (GPU). Scripts: avrl/exp/e04_verify.py, e04_replay_proxycheck.py.

**Critic note (E04, 2026-10-18).** Verdict KILL stands, unchanged (structure KILL, claim-2 readout KILL, joint KILL). Checked against runs/E04/result.json, structure.json, replay/result.json, judge_gpu_summary.json, run.log (GPU), judge_scores.parquet, feats_judge.parquet, verify/verify.json and both verify replays, and the scripts e04_judge_score.py, e04_judge_structure.py, e04_assemble.py, e04_judge_replay.py, e04_verify.py. Every number in both E04 kill entries matches the raw files, except the rounding slip in correction (4). I recomputed the confident-subset rates (judge 34.8%, majority 1.1%), the overlap counts, the duplicate-score noise (3,549 duplicated (question, answer) pairs, 98.8% identical, max range 0.0622), the 35 empty answers, and the split-group pick accuracies directly from the parquet files. Leakage: the GPU script reads only pid, k, answer and groups.question, so gold, majority and the rollout proxy column never reach the judge prompt; the allocator feature set is the 22 cheap+emb columns, none gold-derived; the replay differs from e03_threshold_bootstrap.py in exactly one line (LABEL). Budget: every arm spends 545 verifications (K = 1) on the same 2,180-group pool with 8 seeds (deterministic arms give one run). No truncation or hidden crash: 24,000 of 24,000 rows scored, zero Yes/No top-20 fallbacks, exit code 0, and the "EngineCore died unexpectedly" line follows "Shutdown complete". Sample size: the 25% interval [-0.0769, -0.0478] is far from 0 and D(ts_online) is negative at 10%, 25% and 50%, so seeds and resamples are adequate for a kill.

Corrections, none changes the verdict. (1) First kill entry, "What it changes": "its error set is nearly disjoint from majority vote (Jaccard 0.22)" is wrong. 383 of the 436 majority-vote errors (87.8%) are also judge errors, against about 246 expected if the two were independent. Jaccard is low only because the judge error set is 3.9 times larger (1,694 against 436): it contains almost all majority errors plus 1,311 extra ones. The second entry's reading (judge-only errors are mostly noise) is the right one, and the Jaccard < 0.5 half of the pass rule is satisfied for that artifactual reason. (2) Same sentence: "not learnable from cheap features" is wrong. Cheap features predict judge error with AUC 0.820, and agreement alone gives 0.820 (n_distinct alone 0.822), so all the cheap-feature signal is agreement. Read it as "embeddings add nothing beyond agreement". The second entry's "noisy copy of the split indicator" is also loose: the binary split indicator alone gives AUC 0.712, and the graded agreement gives 0.820, consistent with a pick that is near chance among n_distinct answers. (3) Second entry, "the judge is at chance inside split groups" should read "near chance". Pick accuracy 30.7% against 29.7% for a uniform pick over distinct answers is a paired difference of +1.0 point (SE 0.9), indistinguishable from chance. But the answer-level AUC of mean p_yes inside split groups that hold both a correct and a wrong answer (n = 2,299) is 0.552 (SE 0.007), so the 0.5B judge has a weak real ranking signal that is too small to win the argmax. Not a verdict input. (4) Rounding: "base rate 56.2%" in the claim-2 paragraph of the first kill entry is 1,224 / 2,180 = 56.1%. (5) Scope of the claim-2 readout, not a numerical error: on this label the budget binds. At 25% the budget is 545 groups and the pool holds 1,224 errors, so an oracle admits 0.555, and the label-free threshold at 0.613 already catches 86.9% of the 545 an oracle would (ts_online catches 73.4%). The largest possible D(ts_online) is therefore +0.058 at 25% (+0.014 at 10%), against +0.183 on the E03 proxy label. The PASS bar was reachable only by a near-oracle allocator, so this KILL says ts_online is far from the oracle and the threshold is near the ceiling, not that a learned allocator has no room on judge labels in general. Low power on the pass side. (6) Same ceiling effect on the structure gate: with a 56% base rate the cell100 neighbor ratio cannot exceed 1.77, so the 1.5 clause is hard to reach on this label. It does not change the verdict, because the observed ratio is 0.9997 and the AUC gain is +0.0001, which is no structure at all. (7) The allocator never sees p_yes; the judge enters only through the label, so "a judge-proxy allocator with judge-score features" is untested (L7c). (8) Housekeeping: the log holds two E04 kill headings (the run report and the verification). They agree on every number and neither was edited.

Corrected verdict: KILL, unchanged. Failing conditions: structure, AUC gain +0.0001 < 0.05 and cell100 neighbor ratio 0.9997 < 1.5 (kill clause 2; Jaccard 0.219 does not fire clause 1); claim-2 readout, point D(ts_online) at 25% = -0.0598 <= 0 and the 95% group bootstrap interval [-0.0769, -0.0478] lies entirely below 0. Scope: Qwen2.5-0.5B-Instruct first-token P(Yes) judge, E01 GSM8K rollouts from the 1.5B policy, cell100 replay, 22 cheap+emb allocator features without the judge score, 8 allocator seeds, one pair, low headroom at 25% (max D +0.058). A capacity question for the judge remains open (L7b).

## L7b pre-registration (written before the GPU step)

Date: 2026-10-18. Written before any L7b script ran on the GPU. No L7b outcome has been seen.

**Hypothesis.** The E04 judge failure is a capacity failure. Qwen3-4B, run through the unchanged E04 protocol on the 24,000 E01 GSM8K rollouts (3,000 prompts x 8 rollouts from Qwen2.5-1.5B-Instruct), picks the correct answer in split groups at least 10 points above the 29.7% uniform-over-distinct-answers baseline (so at least 39.7%), and its judge error then clears the E04 structure readout or the E04 claim-2 readout.

Wording note: the hypothesis says "or", while the pass rule below requires both readouts plus capacity for a GO. The per-readout PASS or KILL is logged separately so either reading can be checked. The verdict uses the stricter pass rule.

**Scripts and paths (written before the run).**
- avrl/exp/l7b_judge_score.py: copy of e04_judge_score.py. Changes: gpu_memory_utilization 0.5 to 0.85, and llm.chat gets chat_template_kwargs={"enable_thinking": False}. Declared fallback: if llm.chat raises on chat_template_kwargs, the exact error is written to the log and to the summary json, and each prompt is rendered with tokenizer.apply_chat_template(..., enable_thinking=False) and passed to llm.generate as the same string (logged as a deviation). Prompt wording, p_yes normalization, top-k 20, max_tokens 1, temperature 0, max_model_len 1024 and seed 0 are unchanged. e04_judge_score.py is untouched.
- avrl/exp/L7b.sh: execs the copy with .venv-judge, judge models/Qwen3-4B, out avrl/runs/L7b, mirroring E04.sh.
- GPU step: bin/gpu_run.sh with run dir avrl/runs/L7b, launched once after bin/gpu_status.sh shows under 2 GB used.
- avrl/exp/l7b_judge_structure.py: copy of e04_judge_structure.py (logic unchanged; paths arrive as arguments).
- avrl/exp/l7b_judge_replay.py: copy of e04_judge_replay.py (LABEL judge_error, logic unchanged).
- avrl/exp/l7b_pick_accuracy.py: new script for split-group pick accuracy (metric a).
- avrl/exp/l7b_assemble.py: copy of e04_assemble.py with paths repointed to avrl/runs/L7b, plus the capacity readout and the joint rule below.
- avrl/exp/l7b_verify.py: copy of e04_verify.py with paths repointed to avrl/runs/L7b.

**Pick and structure definitions (unchanged from E04).** Judge pick per group: the distinct answer with the highest mean p_yes over its rollouts, exact ties to the lower answer string. judge_error = pick is not gold. proxy_error stays the E01 majority label.

**Metric (a): split-group pick accuracy.** Split groups are the non-unanimous groups (agreement < 1), 2,441 in E04. The pick is scored against rollouts.correct of the picked answer. Reference: uniform over distinct answers, 29.7% (E04 value 0.2970). The E04 judge scored 30.7%.

**Thresholds.**
- Capacity (a): PASS if at least 39.7%. KILL if at most 34.7%. INCONCLUSIVE strictly between them.
- Structure (b): PASS iff the AUC gain (5-fold balanced logistic, cheap+emb minus cheap, as in p0_analyze.py) is at least 0.05 AND Jaccard(proxy error, judge error) is below 0.5. KILL iff Jaccard > 0.8, or (gain < 0.05 AND cell100 neighbor ratio < 1.5). INCONCLUSIVE otherwise. Gate: p_yes sd below 0.02 is a failed proxy with no verdict, as in E04.
- Claim-2 (c): D = admitted(threshold) minus admitted(ts_online) at 25% (545 of 2,180 sim groups), on judge_error, 8 seeds, 200 group bootstrap resamples, BOOT_SEED 0, cell100 buckets, cheap+emb features. PASS iff point D > 0 AND the 95% percentile interval excludes 0. KILL otherwise.

**Joint verdict (mapping fixed before the run).**
- KILL if any kill clause fires: structure KILL, claim-2 KILL, or capacity KILL.
- GO if structure PASS AND claim-2 PASS AND capacity PASS.
- INCONCLUSIVE otherwise. This includes capacity between 34.7% and 39.7% with both readouts passing, and a structure INCONCLUSIVE with the other two passing.

**Gates, recorded before any verdict.** G1: p_yes sd over the 24,000 rows (constant-judge line 0.02). G2: judge fallback counts (Yes or No outside the top 20). G2 is a validity check. A high count is reported as a caveat in the log and does not change a verdict.

**Expected outputs.** avrl/runs/L7b/: judge_scores.parquet, judge_gpu_summary.json, run.log (GPU), feats_judge.parquet, structure.json, replay/result.json, pick_accuracy.json, verify/verify.json, result.json, done.flag.

**Expected cost.** About 12 GPU minutes (the 0.5B judge took 82 s for 24,000 prompts, and the 4B model has about 8 times the compute), plus model load. Helper cap 45 minutes. About 5 CPU minutes.

**Caveats (stated before the run).**
1. One judge at one size (Qwen3-4B, thinking disabled), one task (GSM8K), one rollout policy (1.5B), and cell100 buckets for the replay only.
2. The replay's built-in reproduction check compares against the E01 proxy-label json, so it will print matches_E01 False. That is expected, as in E04.
3. The replay code is e04_judge_replay.py with no logic change, so the E04 replay-path check (the proxy-label run reproducing E03) carries over. No separate proxy-label check is planned for L7b.
4. The claim-2 ceiling depends on the judge's error count. The maximum possible D at 25% is computed from the L7b pool and reported with the result.
5. A kill is logged as a kill. No other judge is run to rescue it.

**Expected information.** Medium to high. A capacity PASS together with both readout PASSes would reopen the judge lever on this task. A capacity KILL would show that a 4B judge in this protocol fails the same way, which closes the lever on this task.

## L7b 2026-10-18 kill (judge proxy: Qwen3-4B, thinking disabled, E01 GSM8K rollouts)

Run as pre-registered above, with no outcome seen before the gates were recorded. One GPU job on pine: exit code 0, 24,000 prompts judged in 231 s. chat_template_kwargs was accepted by the installed vLLM (0.19.1), so the pre-rendered fallback was not used (chat_template_fallback null). The "EngineCore died unexpectedly" line follows "Shutdown complete" and is benign teardown, as in E04.

Files: avrl/runs/L7b/result.json (all metrics), structure.json, pick_accuracy.json, replay/result.json, verify/verify.json, judge_scores.parquet, feats_judge.parquet, judge_gpu_summary.json, run.log (GPU), structure.log, pick_accuracy.log, replay.log, verify_run.log, assemble.log, done.flag. Scripts: avrl/exp/L7b.sh, l7b_judge_score.py, l7b_judge_structure.py, l7b_pick_accuracy.py, l7b_judge_replay.py, l7b_verify.py, l7b_assemble.py.

**Gates (read before the verdict).** G1: p_yes sd over the 24,000 rows is 0.425 (mean 0.615), so the judge is not constant and a verdict applies. G2 (validity, not a verdict input): 2 rows have no Yes in the top 20, 157 rows have no No in the top 20, and no row lacks both.

**Capacity readout (metric a): PASS.** Split-group pick accuracy is 45.8% (1,117 of 2,441 split groups, binomial SE 1.0 point). The uniform-over-distinct-answers reference is 29.7%, so the paired gain over uniform is +16.1 points (SE 0.9). The pass line is 39.7%. The E04 0.5B judge scored 30.7% on the same groups. Pick accuracy over all 3,000 groups is 55.8%.

**Structure readout: KILL.** Kill clause 2 fires (AUC gain < 0.05 and cell100 neighbor ratio < 1.5). Clause 1 (Jaccard > 0.8) does not fire.

| quantity | L7b, Qwen3-4B | E04, Qwen2.5-0.5B |
|---|---|---|
| judge error, all groups | 44.2% (1,327 of 3,000) | 56.5% (1,694) |
| judge error, confident (agreement >= 0.75, n = 1,551) | 26.1% | 34.8% |
| judge error, unanimous (559) / split (2,441) | 0.5% / 54.2% | 0.5% / 69.3% |
| AUC judge error, cheap / cheap+emb / gain | 0.791 / 0.786 / -0.0056 | 0.820 / 0.820 / +0.0001 |
| neighbor ratio, judge error, cell100 | 1.00 (0.9996) | 1.00 (0.9997) |
| overlap with proxy error (436): both / proxy-only / judge-only | 335 / 101 / 992 (Jaccard 0.235) | 383 / 53 / 1,311 (Jaccard 0.219) |

**Claim-2 readout: KILL.** E03 replay copy with LABEL = judge_error, 25% budget (545 of 2,180 sim groups), 969 judge errors in the pool (base rate 44.5%), 8 seeds, 200 group bootstrap resamples, BOOT_SEED 0, cell100 buckets, cheap+emb features. Admitted corrupted groups at 25%: uniform 0.752, threshold 0.587, greedy_online 0.613, ts_online 0.656. D(ts_online) = -0.070, 95% CI [-0.0914, -0.0551]. D(greedy_online) = -0.026, CI [-0.0485, -0.0222]. Descriptive only: D(ts_online) is -0.043 at 10% and -0.056 at 50%. Ceiling check: an oracle admits 0.438 at 25% on this pool, so the largest possible D is +0.149. The kill is therefore not a budget ceiling effect here, unlike E04, whose ceiling was +0.058.

**Joint verdict: KILL.** Structure KILL and claim-2 KILL, with capacity PASS. The pre-registered rule is KILL when any kill clause fires. The hypothesis fails under both readings of its wording ("or" and "both"), because neither readout clears. Per the pre-registration, no other judge is run to rescue this result.

**Post-hoc (descriptive, added after the readouts were read, not part of any verdict).**
- Judge pick equals the majority answer in 54.8% of groups (E04: 45.7%). Split-group majority accuracy is 82.3%, so the 4B judge is a weak verifier, though well above chance.
- Rollout-level AUC of p_yes against correctness is 0.648 (E04: 0.599). Mean p_yes is 0.684 for correct rollouts and 0.492 for wrong ones.
- Of the 1,327 judge errors, 1,324 fall in split groups, and 992 are judge-only (not in the majority error set).

**What it changes.**
- The E04 null on pick accuracy was partly a capacity failure. The 0.5B judge sat at chance inside split groups (30.7% against 29.7%). The 4B judge picks correct answers at 45.8%, a real ranking signal.
- Capacity did not produce prompt-space structure. Embeddings add nothing beyond the cheap features (gain -0.006), the cheap features carry the signal (AUC 0.79), and the cell100 neighbor ratio is 1.00.
- With the better judge, the label-free threshold still admits fewer corrupted groups than ts_online at 25% (0.587 against 0.656), and the budget is not the binding constraint here. The claim-2 gap is the allocator's.
- The judge-proxy lever is closed for this task at both judge sizes tested under the E04 protocol (Qwen2.5-0.5B and Qwen3-4B, thinking disabled). Other judge settings (thinking mode, rubric prompts, larger models) are untested and are not implied by this kill.

## L7b 2026-10-18 kill

**Review entry (independent check, thesis status, follow-ups).** The full gate readout is in the run entry above (## L7b 2026-10-18 kill, judge proxy: Qwen3-4B). The numbers here were recomputed from the raw parquets before this entry was written.

**Hypothesis.** A 4B judge (Qwen3-4B, thinking disabled, a different model from the 1.5B policy, so no self-judging) clears 39.7% split-group pick accuracy, and its judge error then clears the E04 structure or claim-2 readout. Pre-registered above.

**What was run.** One GPU job on pine through bin/gpu_run.sh: exit 0, 24,000 E01 GSM8K rollouts judged in 231 s, vLLM 0.19.1, bf16, gpu_memory_utilization 0.85, top-20 logprobs, chat_template_kwargs accepted (no fallback). CPU: E04 structure and replay copies with LABEL judge_error on 3,000 groups (2,441 split), a 2,180-group sim pool (75 cell100 buckets), a 25% budget (545 docs), 8 allocator seeds, 200 group bootstrap resamples, BOOT_SEED 0. Outputs in avrl/runs/L7b/.

**Independent check.** avrl/exp/l7b_independent_check.py recomputes the pick rule, judge error and overlap from rollouts.parquet, groups.parquet and judge_scores.parquet (output avrl/runs/L7b/independent_check.json). Every headline number matches result.json. The replay code, run on the L7b feature file with LABEL proxy_error (avrl/exp/l7b_replay_proxycheck.py, output avrl/runs/L7b/replay_proxycheck/result.json), reproduces E01 exactly (max abs diff 0.0) and reproduces E03's proxy-label readout (D(ts_online) -0.172, CI [-0.230, -0.142]). The E04 split-group pick accuracy re-derives as 30.73%.

| quantity | L7b, Qwen3-4B | E04, Qwen2.5-0.5B | baseline or rule |
|---|---|---|---|
| split-group pick accuracy (2,441 groups) | 45.8% (SE 1.0) | 30.7% | uniform over distinct answers 29.7%; pass at least 39.7%; kill at most 34.7% |
| paired gain over uniform, split groups | +16.1 points (SE 0.9) | +1.0 points | not a rule |
| judge error, all groups | 44.2% (1,327) | 56.5% (1,694) | proxy (majority) error 14.5% (436) |
| AUC judge error, cheap / cheap+emb / gain | 0.791 / 0.786 / -0.006 | 0.820 / 0.820 / +0.0001 | structure pass: gain at least 0.05 |
| Jaccard, proxy error vs judge error | 0.235 (335 both, 101 proxy-only, 992 judge-only) | 0.219 (383 / 53 / 1,311) | pass below 0.5; kill above 0.8 |
| cell100 neighbor ratio, judge error | 1.00 (0.9996) | 1.00 (0.9997) | kill clause with gain below 0.05: ratio below 1.5 |
| admitted at 25%: uniform / threshold / greedy_online / ts_online | 0.752 / 0.587 / 0.613 / 0.656 | 0.750 / 0.613 / 0.638 / 0.673 | oracle 0.438 (largest possible D for L7b: +0.149) |
| D(ts_online) at 25%, 95% group bootstrap CI | -0.070 [-0.091, -0.055] | -0.060 [-0.077, -0.048] | pass: D above 0 and CI excludes 0 |
| D(greedy_online) at 25% | -0.026 [-0.048, -0.022] | -0.025 | not a rule |
| D(ts_online) at 10% and 50% (descriptive) | -0.043 and -0.056 | not compared | not a rule |

**Verdict against the pre-registered rules: KILL.**
- Capacity: PASS. Split-group pick accuracy is 45.8% against the 39.7% line, with a paired gain over uniform of +16.1 points (SE 0.9).
- Structure: KILL (kill clause i). AUC gain -0.006 is below 0.05 and the cell100 neighbor ratio is 1.00, below 1.5. Jaccard 0.235 does not fire clause 1 (above 0.8).
- Claim 2: KILL. Point D(ts_online) at 25% is -0.070, at or below 0 (kill), and the 95% CI [-0.091, -0.055] lies entirely below 0.
- Joint: KILL. The hypothesis fails under both readings of its "or" and "both" wording. Per the pre-registration, no other judge is run to rescue it.

**Thesis claims after L7b.**
- Claim 2 (a learned allocator admits fewer corrupted rewards than any fixed rule at equal budget): not supported on this pair under either label. On the judge label, the label-free threshold admits fewer corrupted groups than ts_online at 10%, 25% and 50% (D -0.043, -0.070, -0.056), and fewer than greedy_online at 25% (D -0.026). The better judge did not reverse the sign (E04: -0.060). The 22 allocator features exclude the judge score, so this readout tests the allocator, not the judge. Caveat: the interval is a group bootstrap. A bucket-cluster check is queued as L8c.
- Claim 3 (the edge comes from prompt-space structure): not supported on this pair. There is no edge to attribute, and prompt-space structure in judge error is absent (AUC gain -0.006, neighbor ratio 1.00), matching the E01 proxy-label null. Not tested on any other pair.
- Judge-proxy lever: closed for this task at 0.5B and 4B under the E04 protocol. The 4B judge is a real but weak verifier: its pick agrees with the majority answer in 54.8% of groups, and split-group majority accuracy is 82.3%. Thinking mode, rubric prompts and larger judges are untested and are not implied by this kill.

**Follow-ups (ids in lit/avrl/EXPERIMENT_BACKLOG.md under Proposed by the loop).**
1. **L8a (CPU, under 15 min): L3b, the warm-started bucket prior for ts_online, run now.** Highest information for claim 2. It is the only deployable learner variant on this chain not yet run; it was queued behind L4a. Pass: point D(ts_online) above 0 at 25% with the 95% CI excluding 0 (proxy label, as in L3b). Kill: otherwise. A kill closes the Thompson layer on this pair. A pass would be the first learner to beat the threshold here, and the shipped class would then need the L5a and L5c checks.
2. **L8b (CPU, under 10 min): L7c, judge features for the majority-vote label, rerun with the Qwen3-4B scores.** Medium-low information. L7c was written when the judge was at chance inside split groups; the 4B judge is above chance (45.8% against 29.7%) and its rollout AUC is 0.648 (E04: 0.599). Pass: AUC gain at least 0.02 for proxy_error, to be pre-registered before the run.
3. **L8c (CPU, under 5 min): bucket-cluster bootstrap of the judge-label claim-2 readout.** Low information. The point estimate (-0.070) is far from 0, but the group bootstrap ignores dependence within the 75 buckets. E01's cluster interval was similar in width to its group interval, so a clean check is cheap. A cluster CI that includes 0 would make the claim-2 readout inconclusive; it would not reverse the sign.

**Files.** Review scripts: avrl/exp/l7b_independent_check.py, avrl/exp/l7b_replay_proxycheck.py. Review outputs: avrl/runs/L7b/independent_check.json, avrl/runs/L7b/replay_proxycheck/result.json (and run.log there).

**Critic note (L7b, 2026-10-18).** Verdict KILL stands, unchanged (capacity PASS, structure KILL, claim-2 KILL, joint KILL). Checked against runs/L7b/result.json, structure.json, pick_accuracy.json, replay/result.json, verify/verify.json, independent_check.json, replay_proxycheck/result.json, judge_gpu_summary.json, run.log (GPU), judge_scores.parquet, feats_judge.parquet, and the scripts L7b.sh, l7b_judge_score.py, l7b_judge_structure.py, l7b_pick_accuracy.py, l7b_judge_replay.py, l7b_assemble.py (diffed against the e04 originals; the only logic changes are the ones the pre-registration declares). Every number in both L7b entries matches the raw files. I recomputed from rollouts.parquet, groups.parquet and judge_scores.parquet: split-group pick accuracy 45.76% (1,117 of 2,441) against uniform 29.70%, judge errors 1,327 (3 in unanimous groups), overlap 335 / 101 / 992 (Jaccard 0.2346), pick equals majority 54.8%, confident-subset error 26.1%, rollout AUC 0.648, AUC 0.7912 / 0.7856 (gain -0.0056), neighbor ratio 0.9996 (cell30 1.000, cell300 1.006), E04 split pick accuracy 30.73%. I re-ran l7b_judge_replay.py unchanged into avrl/runs/L7b/critic/replay_rerun/ (log replay_rerun.log): identical admitted rates (uniform 0.752, threshold 0.587, greedy_online 0.613, ts_online 0.656), D -0.0697, CI [-0.0914, -0.0551]. The proxy-label replay on the L7b feature file reproduces E01 exactly (max abs diff 0.0) and E03 (D -0.1715, CI [-0.2299, -0.1422]).

Other checks, all clean. Leakage: the judge prompt holds only the question and the extracted answer; the 22 allocator columns are 6 cheap plus 16 PCA columns of MiniLM embeddings of the question text, with no gold, pass_rate, majority_correct or p_yes; threshold is label-free. Budget: every arm spends 545 verifications at 25% (218 and 1,090 at 10% and 50%) on the same 2,180-group pool. Pre-registration order: l7b_judge_score.py, l7b_pick_accuracy.py, l7b_judge_replay.py, l7b_judge_structure.py, l7b_assemble.py and L7b.sh all carry mtimes of 20:59 to 21:00, before cmd.sh (21:00:42) and every result file (21:04 to 21:05), and none was edited after. Truncation and crash: 24,000 of 24,000 rows scored, no NaN or duplicate (pid, k), exit code 0, thinking mode was effectively off (only 2 and 157 rows miss Yes or No in the top 20, none miss both), and the "EngineCore died unexpectedly" line follows "Shutdown complete". The result.json "matches_E01 false" is the documented expected mismatch. Seeds and samples: the margin D = -0.070 is about four group-CI half-widths from 0 and the sign holds at 10%, 25% and 50%.

Corrections, none changes the verdict. (1) The L8c wording in the review entry and in lit/avrl/EXPERIMENT_BACKLOG.md ("a cluster CI that includes 0 would make the claim-2 readout inconclusive") contradicts the pre-registered claim-2 rule, which is KILL unless point D > 0 AND the CI excludes 0. The point estimate D = -0.0697 fires the kill by itself, so no CI can move it to inconclusive. The check is also already answered: I ran the bucket-cluster bootstrap inline (resample the 75 cell100 buckets with replacement, repeats relabeled, 200 resamples, seed 123, 25% budget, 8 seeds, same code as the replay, nothing written): D(ts_online) mean -0.0745, 95% CI [-0.0909, -0.0609], share D > 0 = 0.000; D(greedy_online) CI [-0.0494, -0.0231], share D > 0 = 0.000. The cluster interval is no wider than the group interval, so L8c adds nothing and can be dropped (the backlog text was not edited). (2) "24,000 prompts judged in 231 s" is the script's total wall time from start, which includes 77 s of weight loading and 33 s of engine init. By the log timestamps (graph capture done 21:03:01, shutdown 21:04:34) generation took about 1.5 minutes. The E04 "82 s" is the same kind of total, so the comparison in the pre-registration stands. (3) "Judge-proxy lever: closed" overstates what the pre-registration licenses. It ties "closes the lever" to a capacity KILL, which did not fire (capacity PASS at 45.8%). What L7b supports is: no GO at 0.5B or 4B under the E04 protocol (judge-error label, allocator features without p_yes). A judge score used as an allocator feature (L8b), thinking mode, rubric prompts and larger judges are untested, so read "closed" as "not reopened" and the L8a title phrase "now that the judge lever is closed" the same way. (4) Labeling: the run entry's "kill clause 2" is the second arm of the structure kill (gain < 0.05 AND neighbor ratio < 1.5); the review entry's "kill clause i" is the structure kill as numbered in the pre-registered rule. They describe the same event. (5) Sensitivity, not an error: 52 groups have an exact tie at the top mean p_yes and go to the lower answer string, as pre-registered. If every one of them were scored wrong, split pick accuracy would fall by at most 2.1 points to 43.6%, still above the 39.7% line.

Corrected verdict: KILL, unchanged. Failing conditions: structure, AUC gain -0.0056 < 0.05 and cell100 neighbor ratio 0.9996 < 1.5 (Jaccard 0.235 does not fire the > 0.8 clause); claim 2, point D(ts_online) at 25% = -0.0697 <= 0 with group CI [-0.0914, -0.0551] and bucket-cluster CI [-0.0909, -0.0609], both below 0. Capacity: PASS (45.8% against the 39.7% line). Scope: Qwen3-4B first-token P(Yes) judge with thinking off, E01 GSM8K rollouts from the 1.5B policy, cell100 replay, 22 cheap+emb features without the judge score, 8 allocator seeds, one pair, ceiling for D +0.149 so the kill is not a budget artifact. Judge-score features, thinking mode, rubric prompts and larger judges remain untested.

## E10 pre-registration (written before the GPU step, no E10 outcome seen)

Date: 2026-10-18. Written before any E10 script ran on the GPU. No E10 outcome has been seen.

**Hypothesis.** Claims 2 and 3 generalize to a second policy family. On the same 3,000 GSM8K train prompts as E01 (sample seed 0), a different-family policy gives the same verdicts: the label-free threshold admits fewer corrupted rewards than ts_online at 25% (claim 2), and prompt-space structure fails the E01 bar (claim 3). This is the first readout on new rollouts, not another analysis of E01 data, so it can revive either claim for this family.

**Model substitution (recorded before the run).** The planned policy, Llama-3.2-3B, is not on disk. The substitute is OLMo-2-0425-1B-Instruct from /homes/rdubash/active-audit/models (shared, read-only, chat template present). vLLM 0.19.1 in .venv-judge registers Olmo2ForCausalLM. The substitute is a different family from the E01 policy (Qwen2.5-1.5B-Instruct), but it is 1B, not 3B. A pass or kill applies to this family at 1B only.

**Scripts and paths (written before the run).**
- avrl/exp/E10.sh: copy of the E04.sh launcher pattern. Execs the unchanged avrl/p0_rollouts.py with .venv-judge, --model /homes/rdubash/active-audit/models/OLMo-2-0425-1B-Instruct, --n_prompts 3000, --G 8, --max_tokens 512, --seed 0, --out avrl/runs/E10.
- GPU step: bin/gpu_run.sh with run dir avrl/runs/E10, launched once after bin/gpu_status.sh shows under 2 GB used. The status read before this entry was 15 MiB used on pine.
- avrl/exp/e10_analyze.py: copy of avrl/p0_analyze.py. Changes: run dir from argv[1] (instead of avrl/runs/p0), and the allocation replay runs the existing avrl/runs/p0/sim_alloc.py (instead of writing a new copy). The MiniLM embedding step keeps device="cuda", the same code path as E01. It runs in the shell after the GPU step has finished.
- avrl/exp/e03_threshold_bootstrap.py: unchanged. --feats avrl/runs/E10/feats.parquet, --cell cell100, --features cheap+emb, --budgets 0.10,0.25,0.50, --seeds 8, --boot 200, --boot_budget 0.25, --workers 8, --out avrl/runs/E10/replay (BOOT_SEED 0).
- avrl/exp/e10_assemble.py: new. Merges summary.json, result.json and replay/result.json into avrl/runs/E10/result.json and applies the rules below mechanically.

**Metrics.** Claim 2: admitted rate at 25% (1 minus caught over total proxy errors, 8-seed mean) for threshold, ts_online and greedy_online. D(ts_online) = admitted(threshold) minus admitted(ts_online), with its 200-resample group bootstrap 95% CI. D at 10% and 50%, and D(greedy_online) at 25%, are descriptive. Claim 3: AUC for proxy_error with cheap and with cheap+emb (5-fold balanced logistic), the gain, the cell100 neighbor ratio, and the confident-subset (agreement >= 0.75) proxy error rate. Descriptive: rollout accuracy, majority accuracy, group proxy error rate.

**Gate, and a known gap in the record.** Gate: confident-subset proxy error rate >= 3% (PROJECT.md Phase 0 read). The E01 entry logs this value as 1.1% (n = 1,551; avrl/runs/p0/result.json, row 17 of this log), so E01 fails the gate as written. The E01 entries did not apply the gate as a stop rule. E10 applies it as written, so the gate can fail again. A gate fail gives no claim-2 verdict. The gate is not changed after seeing E10.

**Pass and kill rules (fixed before the run).**
- Gate: confident-subset proxy error rate >= 3%. If it fails, the entry is logged as a gate fail with no claim-2 verdict.
- Claim 2 (E03 rule): PASS if the gate passes, point D(ts_online) at 25% > 0, and its 95% group bootstrap CI excludes 0. KILL if the gate passes and point D(ts_online) at 25% <= 0. INCONCLUSIVE if the gate passes, D > 0, and the CI includes 0.
- Claim 3 (E01 bar, restated): KILL if AUC gain (cheap+emb minus cheap) < 0.05 AND cell100 neighbor ratio < 1.5. This kill clause is applied regardless of the gate, as written. PASS if the gate passes AND gain >= 0.05 AND ratio >= 1.5. INCONCLUSIVE otherwise, including a gate fail that would otherwise pass.
- A pass reopens that claim for this policy family only. A non-zero exit or crash is recorded as failed with its exact error text. A run that hits the 45-minute cap is recorded as failed.

**Restated claim-3 kill.** The backlog kill text assumed Qwen passed E01 structure. It did not: E01 has AUC cheap 0.923, cheap+emb 0.922 (gain about -0.001, reported as 0.001 in the E01 entry), and cell100 neighbor ratio 1.03. So the claim-3 kill is restated as the E01 bar above, not as a Qwen comparison.

**Expected outputs.** avrl/runs/E10/: run.log, exit_code, done.flag, cmd.sh, wrap.sh, pgid (GPU step); rollouts.parquet, groups.parquet, summary.json (rollouts); feats.parquet, result.json, sim_cell100_cheap.json, sim_cell100_cheap+emb.json, sim_cell30_cheap.json, sim_cell30_cheap+emb.json (analysis); replay/result.json and replay/run.log (E03); result.json with the merged metrics and verdicts (assembly).

**Expected cost.** About 20 GPU minutes (E01 took 1,088 s on the same prompts), under the 45 minute cap. About 10 CPU minutes for the analysis and the bootstrap.

**Caveats (stated before the run).**
1. One substitute policy (OLMo-2-0425-1B-Instruct), one task (GSM8K, 3,000 prompts), one rollout sample per prompt, cell100 buckets for the replay.
2. The E03 reproduction check compares against the E01 proxy-label json, so it will print matches_E01 False. This is the expected mismatch, as in E04 and L7b, because the rollouts differ.
3. Majority-vote proxy only. No judge arm.
4. The claim-2 ceiling depends on the new pool's error count, reported with the result.
5. A kill is logged as a kill. No other policy is run to rescue it.
6. A pass or kill applies to OLMo-2 1B only, not to the planned 3B policy.

## E10 2026-10-18 gate fail (claim 3 kill; claim 2 no verdict). Policy: OLMo-2-0425-1B-Instruct, E01 GSM8K prompts

**Hypothesis and rules.** As pre-registered above. Substitute policy OLMo-2-0425-1B-Instruct for Llama-3.2-3B, same 3,000 prompts, G 8, max_tokens 512, seed 0.

**What was run.** GPU step on pine through bin/gpu_run.sh: exit 0, 24,000 rollouts, generation wall time 821 s (E01: 1,088 s). The line "EngineCore died unexpectedly" follows "Shutdown complete" and is the same benign shutdown message seen in L7b. CPU: avrl/exp/e10_analyze.py (copy of p0_analyze.py, see deviation 1), avrl/exp/e03_threshold_bootstrap.py unchanged (2,180-group sim pool, 75 cell100 buckets, 25% = 545 docs, 8 seeds, 200 group bootstrap resamples, BOOT_SEED 0), then avrl/exp/e10_assemble.py, which applies the pre-registered rules. Outputs in avrl/runs/E10/.

**Numbers (E10 OLMo-2 1B vs E01 Qwen2.5 1.5B).**

| quantity | E10 | E01 | rule or baseline |
|---|---|---|---|
| rollout accuracy / majority accuracy | 76.2% / 86.7% | 64.2% / 85.5% | descriptive |
| group proxy error (all 3,000) | 13.3% | 14.5% | descriptive |
| unanimous groups / proxy error when unanimous | 45.6% / 0.6% | 18.6% / 0.5% | descriptive |
| confident subset (agreement >= 0.75): n / proxy error | 2,158 / 1.6% | 1,551 / 1.1% | gate: at least 3% |
| AUC proxy error, cheap / cheap+emb / gain | 0.932 / 0.931 / -0.002 | 0.923 / 0.922 / -0.001 | claim 3 kill: gain below 0.05 |
| cell100 neighbor ratio (cell30) | 0.998 (1.00) | 1.03 (1.00) | claim 3 kill: below 1.5 |
| admitted at 25%: uniform / threshold / ts_online / greedy_online | 0.746 / 0.134 / 0.337 / 0.167 | not recorded here | descriptive |
| D(ts_online) at 25%, 95% group bootstrap CI | -0.203 [-0.260, -0.175] | -0.172 [-0.230, -0.142] | gated (see verdict) |
| D(ts_online) at 10% / 50% | -0.257 / -0.052 | not compared | descriptive |
| D(greedy_online) at 25%, 95% CI | -0.033 [-0.059, -0.002] | not compared | descriptive |

Total proxy errors in the sim pool: 286 (base rate 13.1%). Reproduction check against the E01 json: matches_E01 False, as pre-registered (different rollouts, so the expected mismatch).

**Verdicts against the pre-registered rules.**
- Gate: FAIL. Confident-subset proxy error is 1.6%, below 3%. Logged as a gate fail.
- Claim 2: no verdict (gate fail). avrl/runs/E10/replay/result.json prints verdict KILL, but that field is the E03 rule applied without the gate. It is not the pre-registered verdict. Descriptively, the label-free threshold admits fewer corrupted rewards than ts_online at 25% (D -0.203), the same sign and similar size as E01.
- Claim 3: KILL. AUC gain -0.002 is below 0.05 and cell100 neighbor ratio 0.998 is below 1.5, so both clauses of the kill fire. Prompt-space structure of proxy error is absent for this policy family, matching E01. The Qwen-specific null is not the only evidence now: the E01 bar fails on a second family.
- Joint: no PASS. Claim 3 is killed for this family. Claim 2 is undecided because the gate failed.

**Deviations and process notes (all logged).**
1. First analysis attempt crashed on import ("No module named 'aa'"). Cause: the copy sits one directory deeper (avrl/exp/), so the ROOT line resolved to avrl/ instead of the CASCADE root. Fixed by adding one dirname to that line, and the docstring notes it. Nothing was written by the failed attempt. Rerun once, exit 0. The rest of the analysis logic is the p0 logic, with the run dir from argv and the existing avrl/runs/p0/sim_alloc.py as the replay script.
2. The MiniLM embedding step keeps device="cuda", as in p0_analyze.py, so the E01 and E10 embeddings come from the same code path. The GPU job had finished before the analysis ran.
3. The gate is the same one E01 already fails (E01 confident-subset error 1.1%, logged above without a stop rule). E10 applies it as written. The project should decide whether this gate governs claim-2 readouts on GSM8K. This entry does not change the gate.

**Thesis claims after E10.**
- Claim 2 (learned allocator admits fewer corrupted rewards than any fixed rule): not decided on this family. The descriptive direction matches E01 (the threshold wins), but the pre-registered gate blocks a verdict.
- Claim 3 (edge from prompt-space structure): KILL on this family. Not tested on any other pair beyond E01 and E10.

**Files.** avrl/exp/E10.sh, avrl/exp/e10_analyze.py, avrl/exp/e10_assemble.py. Outputs: avrl/runs/E10/ (run.log, exit_code, done.flag, summary.json, rollouts.parquet, groups.parquet, feats.parquet, result.json, analyze.log, sim_cell100_*.json, sim_cell30_*.json, replay/result.json, replay/run.log).

## E10 2026-10-18 kill

**Verification and verdict.** I re-derived the E10 headline numbers from the raw files (rollouts.parquet, groups.parquet, feats.parquet, replay/result.json, and avrl/runs/p0/result.json for E01) with read-only CPU checks. Every number in the gate-fail entry above matches the raw files, and the bootstrap CI matches the 200 stored draws. Overall verdict: kill, on claim 3 only. Claim 2 has no verdict because the confident-subset gate failed.

**Hypothesis.** As pre-registered above: claims 2 and 3 on a second policy family, using the same 3,000 GSM8K prompts as E01 (identical question text, confirmed for all 3,000 pids), sample seed 0.

**What was run.** OLMo-2-0425-1B-Instruct (substitute for Llama-3.2-3B, which is not on disk). GSM8K train, 3,000 prompts x G 8 = 24,000 rollouts, max_tokens 512, seed 0, majority-vote proxy, gold verifier. GPU job exit 0, 821 s on pine. Replay: cell100 buckets (75), cheap and cheap+emb features, 2,180 sim groups, 286 proxy errors. Budgets 10%, 25%, 50% (218, 545, 1,090 docs), 8 seeds, 200 group bootstrap resamples, BOOT_SEED 0.

| quantity | E10 (OLMo-2 1B) | E01 (Qwen2.5 1.5B) | rule or baseline |
|---|---|---|---|
| rollout accuracy / majority accuracy | 76.2% / 86.7% | 64.2% / 85.5% | descriptive |
| group proxy error | 13.3% | 14.5% | descriptive |
| unanimous groups / error when unanimous | 45.6% / 0.6% | 18.6% / 0.5% | descriptive |
| confident subset (agreement >= 0.75): n / error | 2,158 / 1.6% (35 errors) | 1,551 / 1.1% | gate: at least 3% |
| AUC proxy error, cheap / cheap+emb / gain | 0.932 / 0.931 / -0.002 | 0.923 / 0.922 / -0.001 | claim 3 kill: gain below 0.05 |
| cell100 neighbor ratio (cell30) | 0.998 (1.00) | 1.03 (1.00) | claim 3 kill: below 1.5 |
| confident-subset AUC, cheap / cheap+emb | 0.768 / 0.705 | 0.627 / 0.534 | descriptive (35 errors) |
| admitted at 25%: uniform / threshold / ts_online / greedy_online | 0.746 / 0.134 / 0.337 / 0.167 | not recorded here | descriptive |
| admitted at 25%, static (free labels, reference only) | 0.157 | not recorded here | reference |
| D(ts_online) at 10% / 25% / 50% | -0.257 / -0.203 / -0.052 | 25%: -0.172 | descriptive except 25% |
| D(ts_online) at 25%, 95% group bootstrap CI | -0.203 [-0.260, -0.175], 0 of 200 resamples above 0 | -0.172 [-0.230, -0.142] | claim 2 rule |
| D(greedy_online) at 25%, 95% CI | -0.033 [-0.059, -0.002] | not compared | descriptive |

**Verdicts against the pre-registered rules.**
- Gate: FAIL. Confident-subset error 1.6% (35 of 2,158) is below 3%.
- Claim 2: no verdict (gate fail). The KILL printed in avrl/runs/E10/replay/result.json is the ungated E03 rule and is not the pre-registered verdict. Descriptively, the label-free threshold admits fewer corrupted rewards than ts_online at all three budgets, and the 25% CI lies entirely below 0.
- Claim 3: KILL. Gain -0.002 is below 0.05 and the cell100 ratio 0.998 is below 1.5. The pre-registered kill clause applies regardless of the gate.
- Overall: kill (claim 3). No PASS.

**Checks against raw files.** Rollout accuracy 0.7616, majority accuracy 0.8673, group error 0.1327, unanimous 0.4557, truncated rollouts 1,620 of 24,000 (6.75%), AUC 0.9325 / 0.9307 (5-fold balanced logistic, as in e10_analyze.py), confident AUC 0.768 / 0.705, cell30/100/300 ratios 1.00 / 0.998 / 1.051, bootstrap CI from the 200 stored draws [-0.2605, -0.1754]. The E01 cell100 ratio is 1.03 in avrl/runs/p0/result.json; the "1.0" in the E10 spec is the cell30 value. New observation, not pre-registered: per-prompt group proxy error on the shared prompts correlates 0.45 between E01 and E10 (phi, n = 3,000).

**What it changes about the thesis claims.**
- Claim 2 (learned allocator admits fewer corrupted rewards than any fixed rule): still undecided on this family. The gate fails on both GSM8K policy families, so this task cannot give a claim-2 PASS under the current gate. Descriptively, the fixed threshold beats ts_online at 10%, 25% and 50%, the same direction as E01.
- Claim 3 (edge from prompt-space structure): killed on this family under the E01 bar, so two families now fail it. The 0.45 cross-family correlation says prompt difficulty is shared across policies. What fails is the MiniLM cell, PC and cheap feature set, not necessarily the existence of prompt-level difficulty. L9a tests that reading.

**Caveats.** One substitute policy at 1B, not the planned 3B. One task, one rollout per prompt, majority-vote proxy only, no judge arm. 6.75% of rollouts hit the 512-token cap (see L9c). The reproduction check prints matches_E01 False, which is expected because the rollouts differ.

**Follow-ups (loop iteration 9, in lit/avrl/EXPERIMENT_BACKLOG.md under "Proposed by the loop").**
1. L9a, cross-family prompt-difficulty check (CPU, under 5 min). Information: high. It separates "no prompt-level structure" from "no structure that MiniLM and cheap features can reach", which decides how claim 3 is worded after two kills.
2. L9b, gate-reachability audit on the existing E01 and E10 groups (CPU, under 5 min, descriptive). Information: high for the program. It shows whether any agreement cutoff reaches the 3% gate with enough errors, which decides whether claim 2 needs a different task or a gate decision from Rustom. It changes no E01 or E10 verdict.
3. L9c, truncation control at max_tokens 1024 (one GPU job, about 25 min, under the 45 min cap). Information: medium-low. It removes the 512-token cap as a source of the gate fail and of the claim-3 AUC before the family-level results are cited.

**Critic note (E10, 2026-10-18).** Verdict unchanged in substance: gate FAIL, claim 2 no verdict, claim 3 KILL. Five statements are corrected below. No number that sets a verdict changes.

Checked: runs/E10/result.json, summary.json, replay/result.json, run.log, analyze.log, rollouts.parquet, groups.parquet, feats.parquet, exit_code, cmd.sh, wrap.sh, and the scripts E10.sh, e10_analyze.py, e10_assemble.py and e03_threshold_bootstrap.py (there is no avrl/exp/E10.py; the launcher is E10.sh). Every number in the pre-registration, the gate-fail entry and the "kill" entry matches the raw files. I recomputed from the parquet files: rollout accuracy 0.76158, majority accuracy 0.86733, group error 0.13267, unanimous 0.45567 with error 0.00585, split error 0.23882, confident subset 2,158 with 35 errors (0.016219), 1,620 rollouts at the 512-token cap (6.75%, 743 groups), agreement, majority, proxy_error and correct all rebuilt from the raw answers, zero duplicate (pid, k), zero NaN, zero empty answers, identical questions and gold to E01 for all 3,000 pids, phi 0.449 against E01. I recomputed the 5-fold balanced logistic AUCs with the e10_analyze.py pipeline (0.93246 / 0.93069, gain -0.00177, confident subset 0.7676 / 0.7047) and the neighbor ratios with aa.data.neighbor_hit_ratio (1.000 / 0.998 / 1.051). I re-ran e03_threshold_bootstrap.py unchanged into avrl/runs/E10/critic/replay_rerun/ (log replay_rerun.log): the point table, D points and all 200 bootstrap draws are identical to replay/result.json (D(ts_online) -0.2032, CI [-0.2605, -0.1754], draws -0.276 to -0.157, 0 of 200 above 0; D(greedy_online) -0.0332, CI [-0.0590, -0.0019], 4 of 200 above 0). The E01 reference rows match avrl/runs/p0/result.json and avrl/runs/E03/result.json (D -0.1715, CI [-0.2299, -0.1422], confident subset 1,551 at 1.096%, AUCs 0.923 / 0.922, confident AUCs 0.627 / 0.534).

Other checks, all clean. Leakage: the 22 allocator columns are 6 rollout statistics (agreement, n_distinct, mean_tokens, std_tokens, any_truncated, log_qlen) plus 16 PCA columns of MiniLM embeddings of the question text; gold, majority, pass_rate, majority_correct and proxy_error are not among them, the proxy is the group's own majority vote (lexicographic tie-break, no gold), and KMeans and PCA are unsupervised. The threshold arm reads agreement and n_distinct only. Budget: every arm spends 218 / 545 / 1,090 docs on the same 2,180-group pool (K = 1, 8 seeds, deterministic arms give one run), and each bootstrap resample uses its own error total. Crash and truncation: exit code 0, no Traceback, the "EngineCore died unexpectedly" line follows "Shutdown complete", 24,000 of 24,000 rollouts written. Provenance: E10.sh (21:26:17) and e10_assemble.py (21:26:40) predate cmd.sh (21:27:14); e03_threshold_bootstrap.py is unchanged since 19:22; e10_analyze.py (21:42:10) is the post-crash path fix, and its diff against p0_analyze.py contains only the declared changes (run dir from argv, ROOT depth, the existing p0 sim_alloc.py, which carries the 10 / 25 / 50% budgets). Sample size: the claim-3 margin is 0.052 of AUC under the 0.05 bar with a gain of -0.002 and a ratio at its permutation null, and the claim-2 descriptive margin is about five bootstrap half-widths from 0, so 8 seeds and 200 resamples are ample.

Corrections.
1. Logged status. The pre-registered rule says a gate fail "is logged as a gate fail". The second E10 entry is headed "kill" and states "Overall: kill". The correct status is GATE FAIL with claim 2 NO VERDICT and claim 3 KILL (the claim-3 kill clause applies regardless of the gate, as pre-registered). It is not a joint kill: claim 2 is undecided.
2. "Generation wall time 821 s" is the script's total wall time (wall_s), which includes about 80 s of imports, weight loading and engine init. The vLLM progress bar shows generation took 12:21 (741 s). E01 was 1,088 s total and 15:47 (947 s) of generation. The faster-than-E01 comparison stands.
3. The cell30 neighbor ratio "1.00" is not evidence of anything. At cell30 every one of the 3,000 docs has a neighbor hit, so the conditional population is the whole population and the ratio is 1.0 by construction. At cell100, 97.0% of docs have a neighbor hit (75.1% at cell300). The informative reading is a 200-permutation null for cell100: mean 0.998, sd 0.015, against the observed 0.998, so the observed ratio sits at the null. The claim-3 kill rests on cell100 and the AUC gain, not on cell30.
4. "Prompt-space structure of proxy error is absent for this policy family" (gate-fail entry) overstates. Embedding-only AUC is 0.581 (E01 0.606), above chance, and the cross-family phi is 0.45. What the data support, and what the "kill" entry already says, is narrower: adding 16 MiniLM PCs to the cheap features adds no AUC (gain -0.002) and cell100 clustering of proxy error is indistinguishable from a label permutation. Read the kill as "no edge from this feature set over cheap features", not "no prompt-level difficulty".
5. The L9c framing "removes the 512-token cap as a source of the gate fail" is already answerable from the E10 files. 1,917 of the 2,158 confident groups contain no capped rollout, and 27 of them are errors (1.41%), still below 3%. Only 8 of the 35 confident-subset errors have a capped rollout. The Wilson 95% interval for the gate value is [1.17%, 2.25%], wholly below 3%. Descriptively, the gate would still fail at agreement >= 0.625 (2.85%, 68 of 2,384) and would pass only at agreement >= 0.5 (5.2%, 136 of 2,613), which is a gate-definition question for L9b and Rustom. The cap may still matter for the AUCs: 268 of the 398 proxy errors sit in groups with a capped rollout, so L9c stays relevant to claim 3, not to the gate.

Corrected verdict: GATE FAIL (confident-subset proxy error 1.62%, 35 of 2,158, below 3%). Claim 2: NO VERDICT; descriptively D(ts_online) at 25% = -0.2032 with group CI [-0.2605, -0.1754] and negative at 10% and 50% (-0.2566, -0.0516), so the label-free threshold admits fewer corrupted rewards than ts_online, and the ungated KILL in replay/result.json is not the pre-registered verdict. Claim 3: KILL (AUC gain -0.0018 < 0.05 and cell100 neighbor ratio 0.998 < 1.5). No PASS, neither claim reopened. Scope: OLMo-2-0425-1B-Instruct (substitute for Llama-3.2-3B), 3,000 GSM8K train prompts, G 8, 512 tokens, one rollout sample, majority-vote proxy, 22 cheap+emb features, 8 allocator seeds, one policy pair with E01.

## L9a pre-registration 2026-10-18 (written before any gain is computed)

**Hypothesis.** Prompt-level difficulty is shared across the two policy families: per-prompt group proxy error correlates 0.45 between E01 (Qwen2.5 1.5B) and E10 (OLMo-2 1B) on the same 3,000 GSM8K prompts. If that shared difficulty carries proxy-error signal beyond E10's own cheap features, then E01's group proxy error is a prompt-difficulty feature from another policy that adds to the cheap set. This is a diagnostic upper bound, not a label-free feature for a single-policy allocator. It tests the E10 critic's reading that the kill is about the MiniLM cell, PC and cheap feature set, not about prompt-level difficulty as such.

**Inputs.** avrl/runs/p0/feats.parquet (E01) and avrl/runs/E10/feats.parquet (E10), inner-joined on pid, 3,000 rows. Before any model is fit, the script asserts identical question text, gold and cell100 labels across the two files. Target y = E10 proxy_error (0/1). CPU only, no GPU, no generation.

**Arms.** All use StandardScaler plus LogisticRegression(max_iter=1000, class_weight="balanced").
- A (cheap, all from E10): agreement, n_distinct, mean_tokens, std_tokens, any_truncated, log_qlen.
- B: A plus E01 proxy_error (0/1). Primary arm.
- C (reference): A plus E10 pc0 to pc15.
- D (reference): E01 proxy_error alone.

**Statistic.** StratifiedKFold(n_splits=5, shuffle=False) on y, out-of-fold predict_proba, pooled OOF ROC AUC over all 3,000 rows. This is not the fold mean used in e10_analyze.py.

**Primary.** gain = AUC(B) minus AUC(A). 95% CI: cluster bootstrap over the 75 cell100 buckets, 200 resamples, numpy default_rng(123). Each resample draws 75 buckets with replacement. Each drawn bucket keeps its rows with OOF scores fixed (no refit), and a bucket drawn twice contributes its rows twice. CI = np.percentile(draws, [2.5, 97.5]).

**Descriptive (not gated).** Reverse direction: y = E01 proxy_error; A_rev = the six cheap features from E01; B_rev = A_rev plus E10 proxy_error; gain_rev with the same bootstrap (same seed, same bucket draws). Context point values: AUC(C) minus AUC(A), and AUC(D).

**Verdict rules (fixed before the run).**
- PASS: gain >= 0.05 AND the 95% CI lower bound > 0. Then prompt-level structure exists beyond the E10 cheap features, and claim 3 is reopened as a representation question (MiniLM plus PCs fail to reach it), not as a rescue of the current feature set.
- KILL: gain < 0.02 (point estimate). Then claim 3 is recorded as killed in its strong form (no prompt-level proxy-error structure beyond cheap features on this task).
- INCONCLUSIVE: gain in [0.02, 0.05), or gain >= 0.05 with a CI lower bound <= 0.

**Caveats (stated before the run).**
1. The E01 arm is a diagnostic upper bound: it needs another policy's rollouts, so even a PASS would not license a label-free allocator for one policy.
2. The bootstrap resamples cells with fixed OOF scores (no refit), so the CI reflects bucket sampling, not refitting variance.
3. One task (GSM8K, 3,000 train prompts), one rollout per prompt per policy, two policies (Qwen2.5 1.5B, OLMo-2 1B), majority-vote proxy.
4. Arm A uses E10's own rollout statistics. Arm B adds E01's proxy error, which comes from E01's answers and is not in arm A.
5. The reverse direction and the context arms are descriptive only.
6. Numbers are logged as computed. The rules are not changed after seeing them.

## L9a 2026-10-18 KILL (claim 3, strong form, at the pre-registered point rule)

**Hypothesis and rules.** As pre-registered above. Target E10 proxy_error; E01 group proxy error added to E10's six cheap features; pooled OOF ROC AUC over 3,000 prompts; cluster bootstrap over cell100 buckets, 200 resamples, default_rng(123).

**What was run.** One CPU run of avrl/exp/l9a_crossfamily_difficulty.py with /homes/rdubash/active-audit/.venv/bin/python, command as specified. Exit 0, no GPU, no generation. Inputs asserted identical in question, gold and cell100 across E01 and E10 for all 3,000 pids before any fit. Outputs: avrl/runs/L9a/result.json (metric values, bootstrap draws, verdict), avrl/runs/L9a/run.log, exit_code, done.flag.

**Numbers.**

| quantity | value | rule or role |
|---|---|---|
| rows / target positives (E10) / positives (E01) | 3,000 / 398 / 436 | sample sizes |
| cell100 buckets used by the bootstrap | 100 | see deviation 1 |
| AUC A, cheap (E10) | 0.9314 | baseline |
| AUC B, cheap + E01 proxy_error | 0.9471 | primary arm |
| gain = B - A | 0.0157, 95% CI [0.0070, 0.0267], boot mean 0.0151 | KILL if point below 0.02 |
| context: AUC C, cheap + pc0..15 (C - A) | 0.9296 (-0.0019) | reference, descriptive |
| context: AUC D, E01 proxy_error alone | 0.7186 | reference, descriptive |
| reverse: A_rev, E01 cheap | 0.9227 | descriptive |
| reverse: B_rev, E01 cheap + E10 proxy_error | 0.9333 | descriptive |
| reverse gain, 95% CI | 0.0106, [0.0039, 0.0174] | descriptive |

The pooled OOF AUC for cheap (0.9314) agrees with the E10 fold-mean AUC (0.9325) from e10_analyze.py, and the E01 cheap AUC (0.9227) agrees with the E01 value (0.923), so the statistic is on the same footing as the earlier entries.

**Verdict against the pre-registered rules.** KILL. The point gain 0.0157 is below 0.02. The KILL clause is point-estimate only, so it applies even though the 95% CI lower bound (0.0070) is above 0. The INCONCLUSIVE band [0.02, 0.05) was not reached. Claim 3 is recorded as killed in its strong form (no prompt-level proxy-error structure beyond the cheap features on this task), as the pre-registration states.

**Deviations (all logged, none changes the verdict).**
1. Bucket count. The pre-registration says 75 cell100 buckets. The 3,000-row statistic has 100 cell100 buckets (E01 and E10 labels identical). The 75 is the replay sim-pool bucket count: avrl/runs/p0/sim_alloc.py holds out a random quarter of the buckets (25) to fit the static classifier and keeps 75 buckets and 2,180 rows as the sim pool. The bootstrap was run over all 100 buckets, which matches the 3,000-row statistic. The bucket count was seen in run.log after the single run. No re-run was made with a different bucket definition. The CI would differ with a 75-bucket resample, but the verdict does not depend on the CI.
2. Pooled OOF AUC, not the fold mean, as pre-registered. No other change to the rules.

**Critic note (L9a, 2026-10-18).** Verdict: KILL at the pre-registered point rule. Two qualifications.
- "Killed in its strong form" is the rule's wording, but the number is not zero. E01's group proxy error adds 0.016 AUC over E10's cheap features with a bootstrap CI of [0.007, 0.027] (100 buckets, not the 75 in the spec). The reverse direction is similar (0.011, CI [0.004, 0.017]). So a small residual of shared prompt difficulty exists, and it is about 0.02 AUC, not "none". The entry should say "residual of about 0.016 AUC, below the 0.02 threshold", not "no structure".
- The residual is not an allocator feature. E01 proxy_error requires another policy's rollouts, and the 16 MiniLM PCs add nothing (C - A = -0.002). So the claim-3 question for a single-policy allocator stays closed at the cheap-feature level. Reading the E10 critic's claim: the kill is about the MiniLM and PC feature set, and the cross-policy oracle shows the shared difficulty is also small (0.016), so the MiniLM set is not missing a large prompt-level effect. This does not reopen claim 3.
- Process note. The 75 versus 100 bucket mismatch was a gap in the pre-registration text (the spec's 3,000-row statistic and its 75-bucket count describe different sets). Future pre-registrations should derive bucket counts from the data they resample, and state the sim-pool count separately.

## L9a 2026-10-18 KILL

**Workflow check entry.** The run step already logged an L9a KILL entry above. This entry re-derives the numbers from the parquet files and the stored bootstrap draws, records the bucket-count deviation, and gives the thesis reading. It does not change the verdict.

**Hypothesis.** As pre-registered above. E01 (Qwen2.5 1.5B) group proxy error, used as a prompt-difficulty feature from a second policy, adds proxy-error signal for E10 (OLMo-2 1B) beyond E10's six cheap features. This is a diagnostic upper bound, not a label-free feature for a single-policy allocator.

**What was run.** One CPU run of avrl/exp/l9a_crossfamily_difficulty.py with /homes/rdubash/active-audit/.venv/bin/python, command as specified, exit 0. No GPU, no generation, no allocation replay. Task: GSM8K train, 3,000 prompts (same prompts for E01 and E10). Model: StandardScaler plus balanced LogisticRegression, 5-fold unshuffled StratifiedKFold, pooled out-of-fold ROC AUC. Bootstrap: 200 resamples, seed 123, cluster bootstrap over cell100 buckets (100 buckets, see deviation).

**Checks.** Recomputed from avrl/runs/p0/feats.parquet and avrl/runs/E10/feats.parquet: 3,000 rows, question, gold and cell100 identical for every pid. AUC A 0.9314 and AUC B 0.9471 reproduce exactly. The 200 stored primary draws give percentiles [0.0070, 0.0267], matching the stored CI, with 0 of 200 at or below 0. The reverse draws have 1 of 200 at or below 0 (minimum -0.0005), so the reverse lower bound sits at about 0.

**Numbers.**

| quantity | value | role |
|---|---|---|
| rows | 3,000 | question, gold, cell100 identical across E01 and E10 |
| positives: E10 target / E01 | 398 / 436 | sample sizes |
| cell100 buckets in the bootstrap | 100 (pre-registration text says 75) | see deviation |
| AUC A, cheap (E10 features) | 0.9314 | baseline |
| AUC B, cheap plus E01 proxy_error | 0.9471 | primary arm |
| gain B minus A, 95% cluster-bootstrap CI | 0.0157, [0.0070, 0.0267] | KILL if point below 0.02 |
| AUC C, cheap plus E10 pc0..pc15 (C minus A) | 0.9296 (-0.0019) | reference |
| AUC D, E01 proxy_error alone | 0.7186 | reference |
| reverse: A_rev (E01 cheap) / B_rev (plus E10 proxy_error) | 0.9227 / 0.9333 | descriptive |
| reverse gain, 95% CI | 0.0106, [0.0039, 0.0174] | descriptive |

**Verdict against the pre-registered rules.** KILL. The point gain 0.0157 is below the 0.02 line. The kill clause is point-estimate only, so the CI lower bound of 0.0070 does not override it. The INCONCLUSIVE band [0.02, 0.05) was not reached. The pre-registered wording "killed in its strong form" is applied here as a point-rule kill. It should not be read as zero residual, because the residual is about 0.016 AUC with a CI above 0.

**Deviation (logged, does not change the verdict).** The pre-registration says to resample 75 cell100 buckets. The 3,000-row statistic has 100 cell100 buckets, identical in E01 and E10. The 75 is the sim-pool bucket count from avrl/runs/p0/sim_alloc.py (25 buckets held out), which the pre-registration text confused with the full statistic. The run resampled all 100 buckets, and the count was visible in run.log only after the single run. No re-run was made. The CI is therefore over 100 buckets, not the pre-registered 75. The verdict is point-based and does not depend on the CI.

**What it changes about the thesis claims.**
- Claim 3 (edge from prompt-space structure): not supported for a single-policy allocator. Prompt difficulty is shared across the two families (per-prompt error correlation 0.45), but the shared part adds only about 0.016 AUC over cheap features, below the pre-registered 0.02 line. The 16 MiniLM PCs add nothing (C minus A = -0.002). The E10 critic's reading (the kill is about the MiniLM, PC and cheap set, not about prompt difficulty as such) is partly supported: the cross-family ceiling is small, so the feature set is not missing a large prompt-level effect. The E01 feature also needs another policy's rollouts, so it cannot be used by a single-policy allocator either way. Claim 3 stays closed.
- Claim 2 (learned allocator admits fewer corrupted rewards than any fixed rule at equal budget): no change from this run, because no allocator was replayed. Its current reading stays not supported (E03 kill on E01; E10 gate fail with no verdict, but the descriptive direction is the same, D(ts_online) at 25% = -0.203).

**Follow-ups** (in lit/avrl/EXPERIMENT_BACKLOG.md under Proposed by the loop, ids L10a to L10c, expected information value in each entry).
1. L10a, oracle-ceiling replay for claim 2 (CPU, about 10 min). Information: high. Adds E01 proxy_error as an allocator feature to the E10 replay (copy of the e03 path, existing scripts unchanged) at 10, 25 and 50%. If the learned arm still does not beat the threshold, claim 2 is closed even with a cross-policy oracle feature, not only with cheap features.
2. L10b, G 16 rollouts on the E10 protocol (GPU, one job, about 28 min projected under the 45 min cap). Information: medium. Tests whether the G 8 majority proxy limits the confident-subset gate (E10 at 1.6% against 3%) and the 0.016 residual.
3. L10c, bucket-definition sensitivity for L9a (CPU, about 5 min, descriptive). Information: low for the claims. It closes the 75 versus 100 record gap by rerunning the same arms on the 2,180-row sim pool (75 buckets) and reporting that CI beside the 100-bucket one.

**Critic note (L9a, adversarial pass, 2026-10-18).** Formal verdict stands: KILL at the pre-registered pooled-AUC point rule (gain 0.0157 below 0.02). The interpretation attached to that KILL is corrected below: "claim 3 killed in its strong form, no prompt-level proxy-error structure beyond cheap features on this task" and "claim 3 stays closed, the MiniLM set is not missing a large prompt-level effect" are NOT supported by the raw data and are withdrawn.

Checked: runs/L9a/result.json, run.log, exit_code, done.flag, exp/l9a_crossfamily_difficulty.py (the file is l9a_crossfamily_difficulty.py, there is no exp/L9a.py), runs/p0/feats.parquet, runs/E10/feats.parquet, runs/p0/sim_alloc.py, runs/E10/analyze.log. Scripts and logs for everything below are in avrl/runs/L9a/critic/ (c1 to c4).

Clean checks. Every number in the pre-registration-era table matches result.json and an independent recompute (AUC A 0.93145, B 0.94713, gain 0.015683, CI [0.00702, 0.02665], boot mean 0.01507, C 0.92956, D 0.71860, reverse 0.92267 / 0.93329 / 0.01063 / [0.00389, 0.01743], positives 398 / 436, phi 0.449, 200 stored draws with 0 at or below 0 for the primary and 1 for the reverse, minimum -0.00048). Exit code 0, run.log has only the summary, no warning or crash text. Inputs: 3,000 unique pids in both files, no NaN, question, gold and cell100 identical, proxy_error equals 1 minus majority_correct in both. The six arm A columns are rollout statistics and question length, none uses gold. The 75 versus 100 bucket explanation is right, with one correction: of 100 cell100 buckets one has fewer than 3 rows and is dropped by MIN = 3, then 24 (not 25) of the remaining 99 are held out, leaving the 75 buckets and 2,180 rows of the sim pool. Budget accounting is not applicable (no allocator was replayed). Verdict mapping: 0.0157 < 0.02 gives KILL under the point rule, correctly applied.

Corrections.
1. The KILL is a pooled-AUC statement and the pooled AUC hides the region that matters. Using the L9a OOF scores, restricted to the 2,158 E10 groups with agreement >= 0.75 (the confident subset, 35 proxy errors, the gate region of E10): AUC of arm A 0.721, arm B 0.916, gain 0.195. In that subset 24 of the 35 E10 errors are also E01 errors, against an E01 error rate of 4.6% among the E10-correct confident groups (97 of 2,123); Fisher exact p = 2e-23. In the 1,367 unanimous groups, 7 of 8 E10 errors are E01 errors (E01 error rate 2.9%). Of the 24 shared confident errors, 17 have the identical wrong majority answer from both models (OLMo-2 1B and Qwen2.5 1.5B), so these are prompt-attached errors (hard or ambiguous problems or gold issues), not independent noise. The pooled gain is small because 363 of the 398 E10 errors sit in the 842 low-agreement groups, where cheap features already rank well (A 0.807, B 0.809). Scope: post hoc and descriptive, 35 errors in the confident subset, no CI computed, not a pre-registered test.
2. The MiniLM PCs do not reach this structure. In the same confident subset the cheap+PC arm C has AUC 0.715 against 0.721 for cheap alone. So the data say: strong cross-policy prompt-level structure exists exactly where E10 is confident and wrong, and the current embedding features do not find it. That is the representation question the PASS clause describes, observed post hoc in a subset, not a pass of the pre-registered rule. The earlier critic note's sentence "the MiniLM set is not missing a large prompt-level effect. This does not reopen claim 3" and the workflow entry's "claim 3 stays closed" should be read as superseded by this note.
3. The KILL margin is inside the noise. The point gain is 0.0043 below the line and the bucket-bootstrap CI [0.0070, 0.0267] contains 0.02 (33 of 200 draws, 16.5%, are at or above 0.02). The point estimate is stable to the fold split (30 shuffled 5-fold splits: 0.0138 to 0.0169, none at or above 0.02; 3-fold 0.0161; 10-fold 0.0161), but not to how the oracle feature is encoded: adding E01 pass_rate gives 0.0189, pe plus pass_rate 0.0194, E01 cheap features plus pe plus pass_rate 0.0229 (INCONCLUSIVE band), and a gradient-boosted model gives 0.0262 with pe and 0.0359 with pe plus pass_rate. The pre-registered logistic arm with a binary E01 label is the weakest reasonable oracle, so "an oracle-style feature adds almost nothing" overstates. The label-free E01 cheap features alone add 0.005.
4. Minor. The 0.016 AUC is 23% of the remaining gap to a perfect ranking (1 - AUC falls from 0.0686 to 0.0529), so "almost nothing" is a wording of the 0.02 line, not a small effect on that scale. The exact command line is not recorded in the ledger or in a cmd.sh.

Corrected verdict. Pre-registered rule: KILL (pooled OOF AUC gain 0.0157, CI [0.0070, 0.0267]), unchanged and not upgraded. Interpretation: claim 3 is NOT recorded as killed in its strong form. What is supported is narrower: no edge on pooled AUC at the 0.02 line, a shared-difficulty residual of 0.016 (more with richer oracle encodings), and a large post hoc cross-policy signal in the confident subset (AUC 0.721 to 0.916, 24 of 35 confident errors shared, 17 with the same wrong answer) that MiniLM PCs fail to reach (0.715). Claim 3 stays open as a representation question for the confident-wrong region only, pending a pre-registered confident-subset test (suggested L10d, CPU, under 5 min: same arms and a bucket-cluster bootstrap on the agreement >= 0.75 subset, with an effect-size bar fixed before running and the shared-wrong-answer share reported). Claim 2 is unchanged by L9a, but L10a (E01 proxy_error as an allocator feature in the replay) is now better motivated, since the gate-region errors are mostly shared across policies.

## L10a pre-registration 2026-10-18 (written before any L10a output was read or computed)

**Source of rules.** lit/avrl/EXPERIMENT_BACKLOG.md, entry "L10a: Oracle-ceiling replay for claim 2" (lines 925 to 941), and the run spec for this session. Motivation: the L9a KILL entry, which found E01 proxy_error to be the only residual signal (AUC gain 0.0157, CI [0.0070, 0.0267]).

**Hypothesis.** Even with E01's proxy_error added as a feature, the label-free agreement threshold still admits fewer corrupted rewards than the learned ts_online allocator at a 25% budget on the E10 pool (OLMo-2-0425-1B, the same 3,000 GSM8K prompts). In D terms: point D(ts_online with the E01 feature) at 25% is <= 0. Reference without the feature (E10 replay, see the E10 kill entry): D(ts_online) = -0.2032, CI [-0.2605, -0.1754]. The backlog quotes -0.203, CI [-0.260, -0.175].

**Design.** avrl/exp/l10a_oracle_ceiling.py is a copy of avrl/exp/e03_threshold_bootstrap.py. The changes are: (a) an --e01_feats argument that joins proxy_error from avrl/runs/p0/feats.parquet onto the E10 pool on pid as e01_proxy_error, with asserts for 3,000 matched pids and no NaN; (b) e01_proxy_error appended to the learned-arm feature list (cheap+emb, 22 features, becomes 23); (c) the verdict relabel below; (d) the output label changed from E03 to L10a. Everything else is unchanged from e03: pool construction (buckets with at least 3 groups, seed-0 bucket shuffle, first quarter fit, 2,180-group sim pool), the threshold arm (agreement and n_distinct only, so it is unaffected), 8 seeds (0 to 7), and the group bootstrap over sim groups with 200 resamples and BOOT_SEED 0.

**Arms reported.**
- threshold: label-free, unchanged from E03.
- ts_online and greedy_online with e01_proxy_error (augmented). greedy_online changes too, so it is reported as an augmented arm, not as the original E03 greedy_online.
- uniform and static, for context only. static is a reference arm that reads labels on the fit split.

**Primary metric.** D = admitted(threshold) minus admitted(ts_online, augmented) at 25% of the sim pool, where admitted = 1 - caught / total_errors. Point estimate from 8 seeds. 95% CI is the percentile interval of 200 group-bootstrap resamples (sim groups drawn with replacement, BOOT_SEED 0, threshold and ts_online recomputed per resample).

**Descriptive (not gated).** Point D at 10% and 50% for ts_online (augmented) and greedy_online (augmented). No CI at those budgets.

**Verdict rules (fixed before any output).**
- PASS: point D > 0 at 25% AND the 95% CI excludes 0 (lower bound > 0).
- KILL: point D <= 0 at 25%.
- INCONCLUSIVE: point D > 0 at 25% AND the 95% CI includes 0. The e03 script labels this case KILL, so the copy relabels it.

**Diagnostic scope.** E01 proxy_error comes from a second policy's rollouts and is defined against gold, so a PASS would not license a single-policy, label-free allocator. A KILL closes claim 2 beyond cheap features, even with a cross-policy oracle feature.

**Expected outcome (from the backlog, not a rule).** KILL.

**Checks before the verdict (logged in the result entry).**
1. 3,000 pids in both files with identical sets, and e01_proxy_error has no NaN after the join.
2. cell100 identical per pid across the two files (L9a assumption).
3. proxy_error = 1 - majority_correct in both files, if the column is present (L9a assumption).
4. Control: the copy run with --e01_feats omitted must reproduce the E10 replay point D(ts_online) at 25% of -0.2032 (avrl/runs/E10/replay/result.json). This checks that the copy adds nothing beyond the feature.
5. The reproduction check against the E01 JSON (matches_E01) is expected to be False on the E10 pool. That is not a failure.

**Caveats.**
1. One cross-policy oracle feature, one task (GSM8K), one learned allocator family (ts_online), 8 seeds, 200 resamples.
2. Group bootstrap over sim groups, as specified and as in e03. Not a bucket-cluster bootstrap.
3. Rules are fixed here. Numbers are logged as computed. If the control fails to reproduce, the copy is re-examined, not the verdict.
4. Compute: local shell CPU with /homes/rdubash/active-audit/.venv/bin/python, 45 min cap. No GPU job, no ssh, no pip install, no writes outside CASCADE.

## L10a 2026-10-18 KILL (claim 2 ceiling: ts_online with the E01 proxy_error feature, E10 pool, cell100, cheap+emb, 25%)

**Verdict against the pre-registered rules.** KILL. Point D(ts_online, augmented) at 25% is -0.2054, which is <= 0, so the kill clause fires. The 95% group-bootstrap CI is [-0.2630, -0.1720] (200 resamples, BOOT_SEED 0) and no resample is above 0 (0 of 200). The INCONCLUSIVE band was not reached. The E01 feature does not give the learned allocator an edge at 25%.

**Numbers (avrl/runs/L10a/result.json, run.log; control in avrl/runs/L10a/control_no_e01/).** Pool: 2,180 sim groups, 75 cell100 buckets, 286 errors, budget 545 at 25%. 8 seeds.

| arm (25% unless noted) | admitted | D point | 95% CI |
|---|---|---|---|
| threshold (label-free, unchanged) | 0.134 (caught 247.62) | reference | |
| ts_online, E01 feature (primary) | 0.340 (caught 188.88 +- 5.33) | -0.2054 | [-0.2630, -0.1720] |
| ts_online, no E01 feature (control, = E10 reference) | 0.337 (caught 189.50 +- 6.02) | -0.2032 | [-0.2605, -0.1754] |
| greedy_online, E01 feature (augmented) | 0.156 | -0.0219 | [-0.0471, +0.0173] |
| greedy_online, no E01 feature (control) | 0.167 | -0.0332 | [-0.0590, -0.0019] |

Descriptive, not gated (point only, no CI): ts_online augmented D at 10% is -0.2574 (control -0.2566), at 50% is -0.0385 (control -0.0516). greedy_online augmented D at 10% is -0.0590 (control -0.0695), at 50% is +0.0149 (control -0.0009). Read the greedy_online augmented 50% value as a single 4-catch difference out of 286 errors (adm 0.010 vs threshold 0.025). It is not a claim.

**Checks (all passed, nothing rerun, logged with avrl/runs/L10a/assumption_checks.json and avrl/runs/L10a/verify.json).**
1. Input assumptions (L9a): 3,000 unique pids in both E10 and E01 feats, identical pid sets, cell100 identical per pid (100 buckets), no NaN in proxy_error, proxy_error = 1 - majority_correct exactly in both files (E10 398 errors, E01 436). Join: 3,000 matched, no NaN, e01_proxy_error mean 0.1453 = 436 / 3,000, from the raw file.
2. Control reproduces the E10 replay exactly: the copy with the feature omitted matches avrl/runs/E10/replay/result.json with max abs difference 0.0 on every point caught mean, every D point and all 200 bootstrap draws for ts_online and greedy_online. The copy adds nothing except the feature.
3. Primary D(ts_online) and D(greedy_online) at 25% recomputed from the stored caught means: -0.20542 and -0.02185, equal to the stored values. Both CIs recomputed from the 200 stored draws: equal to the stored CIs.
4. The threshold arm recomputed independently from avrl/runs/E10/feats.parquet (own code, same split rule): per-seed caught 250, 245, 248, 247, 248, 247, 247, 249, mean 247.625, equal to the stored value.
5. Feature counts: primary 23 (22 cheap+emb plus e01_proxy_error), control 22.
6. The reproduction check against the E01 JSON reports matches_E01 False for both runs (max abs diff 17.0 primary, 24.0 control). Expected on the E10 pool, not a failure.
7. Exit codes 0 for the checks, the control and the primary. done.flag is present.

**Deviations and process notes.**
1. Route: the run used the local shell with /homes/rdubash/active-audit/.venv/bin/python (spec: machine local, CPU only). It did not go through bin/gpu_run.sh, and no GPU was used. The launcher avrl/exp/L10a.sh runs the checks, the control and the primary in parallel, with 8 workers each. Wall time was 36 s (control) and 40 s (primary) for the replay plus bootstrap.
2. The control run (E01 feature omitted, same code) is an addition to the spec. It checks the copy against the E10 baseline. It is not part of the verdict.
3. The verdict relabel (PASS, KILL, INCONCLUSIVE) is in the copy. It did not fire, since point D <= 0.
4. The bootstrap is the group bootstrap of e03, as specified. It is not the bucket-cluster bootstrap that L9a used.
5. 10% and 50% have point estimates only. The script bootstraps only the 25% budget.

**Observed but not gated (no CI, does not change the verdict).** The static reference arm (logistic classifier fit on labels from the fit quarter, deterministic, one value) is the one arm where the E01 feature changes the comparison against the threshold at 25%. Admitted for static with the feature is 0.108 (caught 255), against threshold 0.134. Without the feature static is 0.157, which is worse than the threshold. The same holds at 10% (0.441 vs 0.447, previously 0.458) and at 50% (0.010 vs 0.025, previously 0.035). This arm reads labels from the fit split and is not label-free, so it is not a deployable allocator and was not pre-registered. It is flagged for the critic.

**Scope.** Diagnostic only, one task (GSM8K), one policy (OLMo-2-0425-1B), one learned allocator family (ts_online), cell100, cheap+emb plus the cross-policy E01 feature, 8 seeds, 200 group-bootstrap resamples. E01 proxy_error needs a second policy's rollouts and is scored against gold, so even a PASS would not license a single-policy allocator. This KILL does not show that no learned allocator can match the threshold. It shows that the online ts_online learner, given cheap features or the cross-policy E01 feature, still admits more corrupted rewards than the label-free agreement threshold at 25% on this pool (by 0.205, CI [0.172, 0.263]). The claim 2 wording should say this, and should keep the labeled static arm separate, since that arm's comparison changes with the feature.

**Result check (2026-10-18).** Every headline number in the L10a entry above was checked against avrl/runs/L10a/result.json, control_no_e01/result.json, verify.json, assumption_checks.json and run.log. The control reproduces the E10 replay for both arms (max abs difference 0.0 on point D values and on all 200 draws). The copy differs from e03 only in the --e01_feats join, the feature append, the INCONCLUSIVE relabel and the output label (diff checked). Only one L10a KILL entry is in this log, so this check adds the parts below and does not duplicate the entry.

**Hypothesis (as pre-registered above).** Adding E01 proxy_error does not lift ts_online above the label-free threshold at 25%: point D <= 0.

**Thesis reading.**
- Claim 2 (a learned allocator admits fewer corrupted rewards than any fixed rule at equal budget): not_supported on this pool. The online ts_online learner admits 0.340 against 0.134 for the threshold at 25%, with or without the cross-policy feature. The labeled static arm is the only arm where the feature changes the direction (0.108 against 0.134), but it has one run, no CI, and it is not label-free (see L11b in the backlog).
- Claim 3 (the allocator's edge comes from structure in prompt space): not_supported in the allocator-edge form. On this pool there is no learned edge to attribute, and the one prompt-difficulty feature that carries the L9a structure does not create one for ts_online. The confident-subset structure stays open as a representation question only (L11a).

**Follow-ups (iteration 11, under "Proposed by the loop" in lit/avrl/EXPERIMENT_BACKLOG.md).**
1. L11a, confident-subset test for claim 3 (CPU, under 5 min). Information: medium for the claim 3 wording. Power is low (35 errors in the subset). This replaces the L9a critic's suggested L10d.
2. L11b, robustness of the labeled static arm with the E01 feature (CPU, under 10 min). Information: high for the claim 2 wording. It is the only arm where the feature changes the 25% comparison, and it has one run and no CI.
3. L11c, bucket-cluster bootstrap of the L10a 25% D (CPU, under 5 min). Information: low. The group-bootstrap CI is far from 0, so the kill does not depend on the bootstrap unit.

Scope: diagnostic only. The E01 feature needs a second policy's rollouts and is scored against gold, so a PASS would not license a single-policy allocator.

**Critic note (L10a, adversarial pass, 2026-10-18).** Formal verdict stands: KILL at the pre-registered point rule (point D(ts_online) at 25% = -0.2054 <= 0, CI [-0.2630, -0.1720], 0 of 200 draws above 0; the INCONCLUSIVE band was not reached). The interpretation attached to the KILL is corrected below: "claim 2 ceiling", "the E01 feature does not give the learned allocator an edge", and the pre-registered line "claim 2 is closed beyond cheap features, even with a cross-policy feature" are NOT supported by the raw data and should not be carried into the thesis wording.

Checked and clean. Every number in the entry matches result.json, control_no_e01/result.json, verify.json, assumption_checks.json and run.log: all 18 point cells per run (caught mean, sd, admitted), D points at 10/25/50%, both CIs and frac>0, pool (2,180 groups, 75 buckets, 286 errors, budgets 218 / 545 / 1,090), E01 counts (436 errors, mean 0.1453), matches_E01 False with max abs diff 17.0 and 24.0, wall times (control 36.4 s, primary 40.0 s), exit codes 0/0/0. The diff of l10a_oracle_ceiling.py against e03_threshold_bootstrap.py contains only the declared changes. Leakage: the 23 allocator columns are 6 rollout statistics, 16 MiniLM PCs of the question text (e10_analyze.py line 20, no gold) and e01_proxy_error; the threshold arm reads agreement and n_distinct only. e01_proxy_error is gold-derived by design (oracle, declared in the pre-registration), and the E10 label is a different policy's majority vote against gold, so this is the intended oracle, not an accident. Budget: all arms spend 218 / 545 / 1,090 docs on the same pool, K = 1, 8 seeds (deterministic arms n_runs 1), each bootstrap resample uses its own error total. Sample size: the paired ts_online effect I recomputed below has SE 1.9 catches, while the KILL margin is 59 catches (0.205 of 286), so seeds and resamples are ample for the sign of D. No crash or truncation: run.log has only the summary, no warning text.

Corrections to the interpretation (all post hoc sensitivity runs, imported run_online and make_pool from the unchanged script on the same sim pool, 545 docs, 8 seeds, no files written).
1. ts_online does not respond to features at all, so L10a is not an oracle ceiling for the feature. Per seed, 23 features minus 22 features gives caught differences -4, -1, +7, -6, -1, -8, +4, +4 (mean -0.6, SE 1.9). With only agreement and n_distinct ts_online catches 196.0, with agreement, n_distinct and e01 it catches 196.0, with the six cheap columns plus e01 it catches 195.0, against 188.9 for the full 23. The ts_online deficit to the threshold (247.6) is structural (Thompson sampling over 75 buckets spends labels on exploration, and the classifier only ranks within a bucket), not feature-limited. So "the feature does not help ts_online" is true, but "the feature carries nothing a learned allocator can use" is not tested by this arm.
2. The feature does carry exploitable signal. In the sim pool P(E10 error | E01 error) = 156/309 = 0.505 against P(E10 error | E01 correct) = 130/1,871 = 0.069. The labeled static arm in the entry already shows it (caught 241 without, 255 with the feature, against threshold 247.6). A hand rule with no learning, the threshold order for the first 445 docs plus the next 100 E01-flagged docs, catches 259.5 (D = +0.042 on the same admitted scale), and "all E01-flagged docs first, then threshold order" catches 253.5. The 100-doc split was one untuned value, so these are existence checks, not estimates.
3. The entry's statement that static is the only arm where the feature changes the comparison understates greedy_online. The sign of D at 25% stays negative, but the control CI [-0.0590, -0.0019] excludes 0 and the augmented CI [-0.0471, +0.0173] includes it (frac of draws above 0 rises from 0.02 to 0.13), and greedy_online with fewer columns catches 249.75 (agreement, n_distinct, e01) and 250.6 (cheap6 plus e01), above the threshold 247.6. The 23-column greedy arm loses to the threshold because of the high-dimensional logistic fit on a few hundred labels, not for lack of signal. These are post hoc and single configurations, not a pass of any rule.
4. Context missing from the entry: E10, the pool L10a reuses, FAILED its gate (confident-subset proxy error 1.62% against the 3% minimum, runs/E10/result.json), and E10 recorded "no claim-2 verdict". The threshold already catches 86.6% of all errors in 25% of the pool (247.6 of 286), leaving headroom of 38.4 catches; the static augmented gain (+7.4) is 19% of it. A claim-2 reading from this pool should say that it is the regime in which a label-free agreement rule is hardest to beat.

Corrected verdict. Pre-registered rule: KILL, unchanged and not upgraded (D(ts_online) = -0.2054, CI [-0.2630, -0.1720]). Supported statement: the online ts_online learner (bucket Thompson sampling, K = 1, refit every 68 labels) admits 0.340 against 0.134 for the label-free agreement threshold at 25% on the E10 pool, and the E01 feature changes this by 0.002 (within noise). Not supported: "claim 2 is closed beyond cheap features, even with a cross-policy oracle feature" and the thesis-reading line that there is no learned edge to attribute. Claim 2 stays open for labeled and low-dimensional learners (static with the feature +7.4 catches, greedy with 3 to 6 columns +2 to +3, a hand rule +12) pending L11b, which should add a low-dimensional greedy arm and the hand rule as reference rows and bootstrap them at 25%. Claim 3 is not tested by ts_online for the same reason (the arm ignores features), so the "not_supported in the allocator-edge form" line should read "not tested by this arm".

## L9b pre-registration 2026-10-18 (written before the audit script ran; no subset count or rate computed)

**Hypothesis.** On the existing GSM8K majority-vote groups (E01: Qwen2.5-1.5B-Instruct, avrl/runs/p0/groups.parquet; E10: OLMo-2-0425-1B-Instruct, avrl/runs/E10/groups.parquet), no agreement cutoff in {0.50, 0.625, 0.75} gives a confident subset that meets both the gate (point proxy-error rate >= 3%) and the error floor (>= 100 proxy errors). If that holds, the PROJECT.md confident-subset gate cannot be met on this task at G = 8, so claim 2 has no pre-registered verdict on GSM8K majority vote whatever the policy. The 3% gate is from PROJECT.md. The error floor is from the L9b spec; PROJECT.md states only the 3% gate.

**Why now.** The E01 entries logged the confident-subset gate value (1.1%) without applying it as a stop rule. E10 applied it and failed (1.6%). The claim-2 readouts on the E01 and E10 pools (E03, L5a, L5c, L10a) came after that failure, so they are descriptive under the pre-registered rules until this question is answered. Every later GSM8K claim-2 step inherits it.

**Reading rules (fixed before the run).**
1. Gate: point rate k/n >= 3% only, as E10 applied it. The Wilson 95% interval is reported and not used to gate.
2. Error floor: k >= 100 proxy errors.
3. Agreement is inclusive: a group is in the subset when agreement >= cutoff. The cutoffs 0.50, 0.625 and 0.75 are exact multiples of 1/8 (G = 8). The script asserts that agreement x 8 is an integer in both files, so the comparison has no float ambiguity.
4. Proxy error: proxy_error == 1, meaning the majority answer differs from gold, as p0_rollouts.py defines it. The script also recomputes int(majority != gold) from the raw columns, and checks proxy_error == 1 - majority_correct. Any mismatch is logged and exits non-zero.
5. Per family (E01, E10) and per cutoff: subset group count n, proxy-error count k, point rate k/n, Wilson 95% interval, and two flags (rate >= 3%, k >= 100). Descriptive.
6. Cross-check: the 0.75 row must reproduce the logged values. E01: n 1,551 and rate 0.01096067 (avrl/runs/p0/result.json, confident_subset). E10: n 2,158, k 35 and rate 0.01621872 (avrl/runs/E10/result.json, gate). A mismatch makes the script exit non-zero.

**Outcomes (fixed before the run).**
- Gate unreachable (the hypothesis holds): no cutoff in {0.50, 0.625, 0.75} meets both flags on either family. Consequence: claim 2 has no pre-registered verdict on GSM8K majority vote at G = 8, and the next claim-2 test needs a different task or proxy. This is the PROJECT.md gate-fail read ("move to a harder task or a noisier proxy").
- Some cutoff meets both flags on at least one family: recorded as a finding only. It opens a new pre-registration for a gate, which needs Rustom's decision. No cutoff is adopted for any verdict.
- No E01 or E10 verdict changes under either outcome.

**Scope and limits (stated before the run).** Descriptive only. One rollout sample per prompt (G = 8, seed 0), majority-vote proxy only, two policy families, 3,000 prompts. The subsets at 0.50 and 0.625 include split-vote groups, so their rates are not confident-subset rates in the sense PROJECT.md intends. That is part of what the audit reports. CPU only, run in the local shell, with no GPU job, no generation and no pip install.

**Expected outputs.** avrl/exp/l9b_gate_audit.py (script), avrl/runs/L9b/result.json, run.log, exit_code, done.flag.

**Expected cost.** Under 1 CPU minute.

## L9b 2026-10-18 result (hypothesis not supported: one cell meets both flags, E10 at cutoff 0.50; no verdict changes)

**What was run.** avrl/exp/l9b_gate_audit.py, CPU only in the local shell, with /homes/rdubash/active-audit/.venv/bin/python, command as pre-registered. Exit 0, no GPU job. Outputs: avrl/runs/L9b/result.json, run.log, exit_code, done.flag.

**Checks (all passed).** Both files: 3,000 unique prompts, agreement is a multiple of 1/8 (exact comparisons), proxy_error is 0/1, and proxy_error equals int(majority != gold) with 0 mismatches (E01 and E10), and equals 1 - majority_correct with 0 mismatches. The 0.75 row reproduces the logged values exactly: E01 n 1,551, rate 1.096%; E10 n 2,158, k 35, rate 1.622%. The Wilson intervals match an independent implementation to 1e-17.

**Numbers.** The three rows per family are nested subsets (0.50 contains 0.625, which contains 0.75), so they are not independent samples.

| family | cutoff | n | k | point rate | Wilson 95% | rate >= 3% | k >= 100 | both |
|---|---|---|---|---|---|---|---|---|
| E01 (Qwen2.5-1.5B) | 0.50 | 2,342 | 91 | 3.89% | [3.18%, 4.75%] | yes | no | no |
| E01 | 0.625 | 1,960 | 38 | 1.94% | [1.42%, 2.65%] | no | no | no |
| E01 | 0.75 | 1,551 | 17 | 1.10% | [0.69%, 1.75%] | no | no | no |
| E10 (OLMo-2 1B) | 0.50 | 2,613 | 136 | 5.20% | [4.42%, 6.12%] | yes | yes | yes |
| E10 | 0.625 | 2,384 | 68 | 2.85% | [2.26%, 3.60%] | no | no | no |
| E10 | 0.75 | 2,158 | 35 | 1.62% | [1.17%, 2.25%] | no | no | no |

Total proxy errors: E01 436 of 3,000, E10 398 of 3,000.

**Verdict against the pre-registered rules.** The hypothesis is not supported. The gate is reachable on E10 at cutoff 0.50, and that is the only cell that meets both flags. Per the pre-registration, this is a finding only. It opens a new pre-registration for a gate, which needs Rustom's decision. No cutoff is adopted for any verdict.

**Reading the one passing cell carefully.** Under the inclusive rule, agreement >= 0.50 with G = 8 admits 4-4 splits, so the E10 0.50 subset is not a confident subset in the PROJECT.md sense. E01 at 0.50 clears the gate (3.89%, Wilson lower bound 3.18%) but misses the error floor by nine errors (91 against 100). E10 at 0.625 misses both flags (2.85%, 68 errors).

**What this does and does not change.** No E01 or E10 verdict changes. E10 stays GATE_FAIL with no claim-2 verdict under the pre-registered 0.75 gate, and E01 stays as logged. The GSM8K claim-2 question is therefore not closed by this audit. It is open, and the next step is a decision for Rustom on whether any gate can be defined on a 0.50 subset, and whether the error floor (not in PROJECT.md) applies. The L9b spec names L6a, L8a and L11b as later GSM8K claim-2 steps that inherit this question, so they should wait on that decision (this reading is mine, not part of the pre-registration).

---

## L9b 2026-10-18 inconclusive

**Hypothesis.** No agreement cutoff in {0.50, 0.625, 0.75} gives a subset that meets both the 3% point-rate gate and the 100-error floor on either GSM8K majority-vote family. Result: not supported. E10 at cutoff 0.50 meets both flags. No verdict changes.

**What was run.** avrl/exp/l9b_gate_audit.py, CPU only in the local shell with /homes/rdubash/active-audit/.venv/bin/python, command as pre-registered. No generation, no GPU job, no seeds (no resampling; Wilson intervals only). Inputs are the existing groups tables for 3,000 GSM8K train prompts per family, with G 8 rollouts per prompt at seed 0. E01 is Qwen2.5-1.5B-Instruct (avrl/runs/p0/groups.parquet). E10 is OLMo-2-0425-1B-Instruct (avrl/runs/E10/groups.parquet). Wall clock about 4 min. Outputs: avrl/runs/L9b/result.json, run.log, exit_code (0), done.flag.

**Checks (re-read from result.json and run.log, and re-derived from the parquet files).**
- Both files: 3,000 groups, agreement is a multiple of 1/8, proxy_error is 0/1.
- proxy_error equals int(majority != gold) and 1 - majority_correct, with 0 mismatches on both files.
- Totals: E01 436 proxy errors, E10 398, matching result.json.
- The 0.75 row reproduces the logged values: E01 n 1,551, 17 errors, 1.10%; E10 n 2,158, 35 errors, 1.62%.
- Wilson intervals recomputed independently; they match result.json at two decimals in percent for every cutoff row.

**Numbers.** The cutoff subsets are nested (0.50 contains 0.625, which contains 0.75), so rows are not independent samples. The "none" rows are all 3,000 groups, given as a reference.

| family | cutoff | n | k | point rate | Wilson 95% | rate >= 3% | k >= 100 | both |
|---|---|---|---|---|---|---|---|---|
| E01 | none (all groups, baseline) | 3,000 | 436 | 14.53% | [13.32%, 15.84%] | yes | yes | yes |
| E01 | 0.50 | 2,342 | 91 | 3.89% | [3.18%, 4.75%] | yes | no | no |
| E01 | 0.625 | 1,960 | 38 | 1.94% | [1.42%, 2.65%] | no | no | no |
| E01 | 0.75 | 1,551 | 17 | 1.10% | [0.69%, 1.75%] | no | no | no |
| E10 | none (all groups, baseline) | 3,000 | 398 | 13.27% | [12.10%, 14.53%] | yes | yes | yes |
| E10 | 0.50 | 2,613 | 136 | 5.20% | [4.42%, 6.12%] | yes | yes | yes |
| E10 | 0.625 | 2,384 | 68 | 2.85% | [2.26%, 3.60%] | no | no | no |
| E10 | 0.75 | 2,158 | 35 | 1.62% | [1.17%, 2.25%] | no | no | no |

Exact-agreement band (agreement = 0.50, meaning exactly 4 of 8 rollouts match the plurality answer), derived as the 0.50 row minus the 0.625 row and checked against the parquet files: E01 382 groups, 53 errors, 13.9%; E10 229 groups, 68 errors, 29.7%.

**Verdict against the pre-registered rules: inconclusive.** The pre-registration sets no pass threshold (descriptive). The kill condition (no cutoff meets both flags on either family) is not met, because E10 at 0.50 meets both flags. Under the pre-registration that cell is a finding only. It opens a new pre-registration for a gate, which needs Rustom's decision. The PROJECT.md gate (0.75, 3%) still fails on both families, so the E01 and E10 verdicts are unchanged.

**Correction to the result block above.** That block describes the 0.50 subset as including 4-4 splits. That is only partly right. Agreement 0.50 means exactly 4 of 8 rollouts match the plurality answer. Only 21 of E10's 229 exact-0.50 groups (13 of E01's 382) are two-answer 4-4 ties (n_distinct = 2). The rest have 3 to 5 distinct answers and a unique plurality. The tie rule in p0_rollouts.py (np.unique then argmax, line 34) picks the lexicographically smallest tied answer. So the tie rule can only affect those 4-4 groups in the 0.50 subset.

**What it changes about the thesis claims.**
- Claim 2 (a learned allocator admits fewer corrupted rewards than any fixed rule at equal budget): untested under the pre-registered rules on GSM8K majority vote at G 8. The 3% gate on the 0.75 confident subset fails on both families (1.10% E01, 1.62% E10). The one cell that passes both flags is a 4-of-8 plurality at 0.50, which is not a confident subset and is not a verdict. The claim-2 kills on the E01 and E10 pools (E03, L5a, L5c, L10a) stay descriptive, as the L9b pre-registration says. All of them point the same way (the learned arms admit more corrupted rewards than the label-free threshold), but none is a verdict.
- Claim 3 (the edge comes from structure in prompt space): no change. The logged claim-3 kills (E01 2026-09-30, E10 2026-10-18, L9a 2026-10-18 strong form) stand under their own pre-registered rules.
- The 100-error floor comes from the L9b spec, not PROJECT.md. Whether it applies is part of Rustom's decision.

**Decisions for Rustom.** (1) Whether any gate may be defined on a subset that includes 4-of-8 plurality groups. My reading is that it should not, since that subset is not confident. (2) Whether the 100-error floor applies. (3) Whether to move to a harder task or a noisier proxy, which PROJECT.md names as the gate-fail route.

**Follow-ups (expected information value).**
1. L10d (CPU, descriptive): split the exact-0.50 band by distinct-answer count and recompute the 4-4 ties under a random tie-break from the rollouts.parquet files. Information: whether the only gate-passing cell depends on the lexicographic tie rule. Medium, and cheap. It decides whether any future gate needs a tie rule.
2. L10e (CPU, planning arithmetic): prompts needed for the 100-error floor. At the observed E10 0.75 rate (1.62%), the floor needs about 6,200 confident groups, about 8,600 prompts at the observed 72% share. E01 at 0.75 needs about 9,100 groups, about 17,600 prompts. At a 3% rate the floor needs about 3,300 groups. Information: whether a larger GSM8K prompt pool (new GPU rollouts) can meet the floor, or whether the next claim-2 test has to change task. High for the program decision, low cost.
3. L10b (existing, GPU, one job): G 16 on the E10 protocol is the only backlog route that changes the agreement grid, so this audit raises its priority. Not a new id. It needs its own pre-registration before it runs.

**Critic note (L9b, 2026-10-18).** Verdict unchanged: descriptive, no pass or kill. The hypothesis (no cutoff meets both flags on either family) is not supported, because E10 at 0.50 meets both flags. That cell is a finding only and opens a new pre-registration that needs Rustom's decision. The 0.75 gate still fails on both families, and no E01 or E10 verdict changes. The "inconclusive" label stands. Five statements are corrected or qualified below. No number that sets the verdict changes.

Checked: runs/L9b/result.json, run.log, exit_code, done.flag, avrl/exp/l9b_gate_audit.py, the two groups.parquet files, runs/p0/result.json, runs/E10/result.json, PROJECT.md line 13, the L9b backlog spec, p0_rollouts.py line 34, and the dated ledger snapshots in snap/. I recomputed every table cell from the parquet files without importing the script, using integer rollout counts (agreement x 8) so there is no float comparison. All match: E01 2,342/91/3.89% [3.18%, 4.75%], 1,960/38/1.94% [1.42%, 2.65%], 1,551/17/1.10% [0.69%, 1.75%]; E10 2,613/136/5.20% [4.42%, 6.12%], 2,384/68/2.85% [2.26%, 3.60%], 2,158/35/1.62% [1.17%, 2.25%]; all-group rows 436/3,000 (14.53%, [13.32%, 15.84%]) and 398/3,000 (13.27%, [12.10%, 14.53%]); exact-0.50 bands E01 382 groups, 53 errors (13.9%) and E10 229 groups, 68 errors (29.7%); n_distinct split of the bands 2/3/4/5 = 13/58/115/196 (E01) and 21/57/64/87 (E10), with 3 and 9 errors in the two-answer groups. The 0.75 rows match the logged E01 (1,551, 0.010961) and E10 (2,158, 35, 0.016219, gate pass False). Tie rule: np.unique then argmax at p0_rollouts.py line 34, so only the n_distinct = 2 groups can be affected. The L10e planning arithmetic (about 6,200 and 8,600 for E10, 9,100 and 17,600 for E01, 3,300 at 3%) recomputes. run.log is clean (checks_passed=True, exit 0, no traceback or warning). Leakage: none. Agreement and n_distinct are label-free, gold enters only to define the proxy error being audited, and there is no allocator or budget arm in this audit. Seeds: none used, one G 8 sample per prompt.

Corrections and qualifications.
1. Provenance of the hypothesis. The pre-registration header says no subset count or rate had been computed. That is true of the L9b script, but the E10 cells were already in this ledger: correction 5 of the E10 critic note (present in the snapshot from 21:57, before the 22:46 pre-registration) logged E10 at 0.50 as 5.2% with 136 of 2,613 errors and at 0.625 as 2.85% with 68 of 2,384. The pre-registered hypothesis was therefore already falsified for E10 at registration. The verdict is unaffected. What L9b adds is the E01 rows, the error-floor flags, the all-group reference rows, the band split and an independent recomputation of the E10 cells.
2. "Wall clock about 4 min" (second block) is unsupported. The script was last modified at 22:46:02 and the four output files are stamped 22:46:07 to 22:46:08, so the run took seconds, in line with the pre-registered "under 1 CPU minute". run.log records no timing. Also, "command as pre-registered" cannot be checked: the pre-registration lists only the expected outputs and no command line, and there is no cmd.sh. result.json does record the two source files and the three cutoffs, which match the spec.
3. "The L9b spec names L6a, L8a and L11b" (first block) is wrong. Neither the pre-registration nor the backlog L9b spec names them. They are separate backlog items (L6a, L8a, L11b) on the same E01 and E10 pools. The inference that they should wait on the gate decision is a reasonable reading, as that sentence already says, but it is the author's and not the spec's.
4. "All of them point the same way (the learned arms admit more corrupted rewards than the label-free threshold)" overgeneralizes. The primary online arms lose to the threshold (E03 ts_online, L5a and L5c learned_class, L10a ts_online). But E03 greedy_online ties (D -0.001, CI spans 0), the labeled static arm beats the threshold in E03 (263.0 against 252.5 caught) and in L10a (255 with the E01 feature against 247.6), and L3c passed. Read the claim-2 lines as "the primary deployable online arms lose", not "every learned arm loses". The claim-2 kills still stay descriptive under the L9b pre-registration.
5. "Exactly one cell meets both flags" is not a stable count. The refutation rests on E10 at 0.50 alone, and that cell is robust: Wilson lower bound 4.42% against the 3% gate, and 136 errors is about three binomial SDs above the floor. It is also tie-rule proof. Only the 21 two-answer groups (9 errors) can change under another tie-break, so E10 k stays within 127 to 148 and the rate stays at or above 4.86%. L10d can therefore not flip this cell, and the only live question it answers is the E01 floor. Two neighbors are within noise of their flags: E01 at 0.50 has 91 errors against the floor of 100 (one binomial SD is about 9.5, and the tie rule allows 88 to 101), and E10 at 0.625 has a point rate of 2.85% with a Wilson interval [2.26%, 3.60%] that straddles 3%. The pre-registered point-rate rule classifies both as misses. A second rollout sample could move either, so the "which cells pass" table should be quoted with those two cells marked borderline. The 0.50 and 0.625 subsets also include split-vote groups (4 of 8 matching), so even a passing 0.50 cell is not a PROJECT.md confident subset (agreement >= 0.75).

Corrected verdict. Descriptive, no pre-registered pass or kill. Hypothesis not supported (one robust cell, E10 at agreement >= 0.50: 136 of 2,613, 5.20%). Finding only, which opens a new gate pre-registration that needs Rustom's decision. PROJECT.md gate at 0.75 fails on both families (E01 1.10%, E10 1.62%). No E01 or E10 verdict changes. Label: inconclusive.

## L11b pre-registration 2026-10-18 (written before any L11b output was read or computed)

**Source of rules.** lit/avrl/EXPERIMENT_BACKLOG.md, entry "L11b" (line 1001), and the L10a critic note and correction 3 (the L10a entry), which asked for a low-dimensional greedy arm and the hand rule as reference rows, bootstrapped at 25%. The L11b run spec for this session.

**Hypothesis.** Claim 2 (a learned allocator admits fewer corrupted rewards than the label-free agreement threshold at equal budget) is still open on the E10 pool (OLMo-2-0425-1B, 3,000 GSM8K prompts, cell100, 2,180 sim docs, 286 proxy errors, 8 seeds) at 25% (545 docs). Two routes are tested.
- (a) PRIMARY, label-free and deployable: greedy_online on the six cheap rollout statistics, no E01 feature. Point D(greedy_online, cheap six) at 25% <= 0 kills it.
- (b) CEILING, labeled: static ranker on the cheap six plus the 16 MiniLM PCs plus e01_proxy_error, refit per seed. Seed-mean point D(static) at 25% <= 0 kills it. The L10a +7.4 caught was one fit split.

**Reference rows (not gated).** static on cheap plus PCs without E01; greedy_online on agreement and n_distinct only; greedy_online on cheap six plus E01; the hand rule; uniform; threshold. All are bootstrapped at 25% and given a point D at 10% and 50%.

**Design.** avrl/exp/l11b_low_dim_ceiling.py is a copy of avrl/exp/l10a_oracle_ceiling.py. Unchanged: pool construction (buckets with at least 3 docs, seed-0 bucket shuffle, first quarter = fit, 2,180-doc sim pool), the threshold arm (agreement ascending, n_distinct descending, seeded random tie-break), budgets 10/25/50%, 8 seeds (0 to 7), the group bootstrap with 200 resamples and BOOT_SEED 0, admitted = 1 - caught / total_errors, D = admitted(threshold) - admitted(arm). The L10a copy is not edited, and neither are the e03 and p0 scripts.

Deviations, logged before the run:
1. Static refit per seed. Seed s fits the static classifier (same pipeline as L10a) on a seeded 80% subsample of the buckets in the fixed seed-0 fit quarter, using numpy default_rng(100 + s) and k = round(0.8 x number of fit-quarter buckets). The sim pool is unchanged, so the 8 seeds vary the fit split only.
2. Fixed per-seed scores in the bootstrap. Inside a resample, each seed's static scores for the resampled docs stay as computed on the full sim pool (no refit). Each seed's order is recomputed on the resample's own docs.
3. The arm set is the L10a set minus ts_online and threshold_input_order (not in this spec). The hand rule is added. It is defined at every budget as the threshold order for the first (budget - 100) docs, then the next 100 e01_proxy_error = 1 docs in threshold order, with the same seeded tie-break as the threshold arm. At 25% this is the 445 + 100 rule in the spec. The 10% and 50% values are descriptive. If fewer than 100 flagged docs remain, the rest is filled in threshold order (logged if it fires).
4. Invocations. The greedy_online arm and the static arm use the --features column set, as in L10a. The spec's command shape gives one feature set per invocation, so five invocations run from avrl/exp/L11b.sh: control (cheap+emb, no E01), primary (cheap, no E01), ceiling (cheap+emb with --e01_feats), ref_agree_nd (agree_nd, a new feature set of agreement and n_distinct only), and ref_cheap_e01 (cheap with --e01_feats). Two of these are not in the spec's command shape. Their reason is that the greedy reference arms need a different column set from the ceiling run. Extra rows (static on cheap, static on agree_nd, greedy on cheap+emb+E01) are descriptive only and are not interpreted.

**Primary metric.** D(greedy_online, cheap six, no E01) at 25%, 8-seed mean, with its 95% group-bootstrap percentile CI.

**Ceiling metric.** D(static, cheap+emb+e01_proxy_error, per-seed refit) at 25%, 8-seed mean, with its 95% CI.

**Descriptive (not gated).** Point D at 10% and 50% for both; the reference rows with their 25% CIs and their point D at 10% and 50%.

**Verdict rules (fixed before any output).**
- PASS: point D > 0 at 25% AND the 95% CI lower bound > 0.
- KILL: point D <= 0 at 25%.
- INCONCLUSIVE: point D > 0 at 25% AND the CI includes 0.
These apply to the PRIMARY row (greedy, cheap six, no E01) and to the CEILING row (seed-mean static with E01). A PASS on either is a finding only. Any allocator claim from it needs its own pre-registration. Reference rows get the same status labels, for reading only.

**Checks before the verdict.**
1. Join asserts from L10a (3,000 pids matched, no NaN in e01_proxy_error) run inside the copy for the E01 invocations.
2. Control: the cheap+emb, no-E01 invocation must reproduce avrl/runs/L10a/control_no_e01/result.json exactly for greedy_online and threshold: point caught means and sds and admitted at 10/25/50%, the point D values, and all 200 bootstrap draws of D(greedy_online). Tolerance 1e-9. The static arm is not part of this check, because it is refit. If the control fails, the copy is re-examined and no verdict is issued (as in L10a).
3. Static fit sanity: per-seed fit sizes and error counts logged. Each seed's fit must contain both classes.
4. Bootstrap: 200 draws per row. Pool counts (2,180 sim docs, 286 errors) must match L10a.

**Expected outcome.** Not stated. Both kill clauses are live. No outcome is expected by rule.

**Caveats for any claim 2 wording.** L9b: the 3% confident-subset gate is unmet on both GSM8K families at the pre-registered 0.75 cutoff (E10 1.62%, E01 1.10%), so this result is a finding, not a verdict. One task, one policy (OLMo-2-0425-1B), one pool, 8 seeds, 200 group-bootstrap resamples. E01 proxy_error is a cross-policy oracle scored against gold, so a PASS would not license a single-policy allocator. The hand rule was post hoc, and its 100-doc split was one untuned value (see the L10a critic note).

**Compute.** CPU only, in the local shell with /homes/rdubash/active-audit/.venv/bin/python, in the foreground, with the five invocations run in parallel inside avrl/exp/L11b.sh (8 workers each). No gpu_run.sh, no ssh, no pip install, no writes outside /homes/rdubash/avrl_live/CASCADE/. Expected wall time under 5 min. Outputs: avrl/runs/L11b/ (result.json, run.log, exit_code, done.flag, plus one subdirectory per invocation).

## L11b 2026-10-18 result (PRIMARY KILL; CEILING INCONCLUSIVE; no verdict upgraded, claim 2 stays open for labeled learners only)

**What was run.** avrl/exp/L11b.sh, CPU only in the local shell with /homes/rdubash/active-audit/.venv/bin/python, foreground, 41 s wall, no GPU, no ssh, no pip install. It runs avrl/exp/l11b_low_dim_ceiling.py in five invocations (control: cheap+emb no E01; primary: cheap no E01; ceiling: cheap+emb with E01; reference: agree_nd; reference: cheap with E01) and then avrl/exp/l11b_assemble.py. All exit codes 0. Outputs: avrl/runs/L11b/result.json, run.log, exit_code, done.flag, assemble.log, and one subdirectory per invocation (each with result.json, run.log, exit_code). Pool: 2,180 sim docs, 75 buckets, 286 errors, budget 545 at 25%, 8 seeds, 200 group-bootstrap resamples (BOOT_SEED 0).

**Checks (all passed, in result.json "checks").**
1. Control reproduces L10a exactly: threshold and greedy_online point caught means, sds and admitted at 10, 25 and 50%, the greedy D values, and all 200 bootstrap draws of D(greedy_online) match avrl/runs/L10a/control_no_e01/result.json with max abs difference 0.0. The greedy 25% CI is [-0.0590, -0.0019] in both.
2. Pool identical in all five invocations and equal to L10a (2,180 docs, 75 buckets, 286 errors). Uniform and threshold identical across invocations. The hand rule is identical in the two E01 invocations.
3. Static refit per seed: 8 fits, each on 19 of the 24 fit-quarter buckets, 572 to 708 fit docs, 76 to 100 fit errors, both classes present in every fit.
4. Headline D values recomputed by hand from the stored per-seed caught counts: magnitudes match the stored values exactly (greedy -0.00787, static +0.00830).
5. Two post hoc reference rows reproduce the L10a critic's numbers exactly: the hand rule catches 259.5 (critic: 259.5) and greedy_online on cheap six plus E01 catches 250.6 (critic: 250.6).

**Pre-registered numbers (25%; D = admitted(threshold) minus admitted(arm), 8-seed mean; CI = 200 group-bootstrap percentile; avrl/runs/L11b/result.json).**

| row | arm and columns | caught (sd) | admitted | D point | 95% CI | status |
|---|---|---|---|---|---|---|
| threshold (baseline) | label-free agreement | 247.63 (1.41) | 0.134 | 0 | | |
| PRIMARY | greedy_online, cheap six, no E01 | 245.38 (1.65) | 0.142 | -0.0079 | [-0.0439, +0.0089] | KILL (point D <= 0) |
| CEILING | static, cheap six + 16 PCs + e01_proxy_error, refit per seed | 250.00 (4.50 across seeds) | 0.126 | +0.0083 | [-0.0167, +0.0347] | INCONCLUSIVE (point D > 0, CI includes 0) |

Descriptive at 10% and 50% (point only): PRIMARY D 10% -0.0481, 50% -0.0031. CEILING D 10% -0.0035, 50% +0.0101.

Reference rows (not gated; status shown only for reading):

| row | caught | admitted | D 25% | 95% CI | label |
|---|---|---|---|---|---|
| static, cheap + 16 PCs, no E01 (refit per seed) | 239.75 | 0.162 | -0.0275 | [-0.0532, -0.0058] | KILL |
| greedy_online, agreement and n_distinct only | 241.13 | 0.157 | -0.0227 | [-0.0378, +0.0097] | KILL |
| greedy_online, cheap six plus E01 | 250.63 | 0.124 | +0.0105 | [-0.0280, +0.0328] | INCONCLUSIVE |
| hand rule (post hoc: threshold 445, then 100 E01-flagged) | 259.50 | 0.093 | +0.0415 | [+0.0052, +0.0660] | would read PASS; post hoc, oracle flags, not gated |
| uniform | 72.75 | 0.746 | -0.6115 | [-0.6552, -0.5716] | KILL |

Extra descriptive rows (computed by the same invocations, not in the pre-registration, no verdict): greedy_online on cheap plus 16 PCs without E01 (the L10a control arm) D -0.0332, CI [-0.0590, -0.0019]. Static on cheap six without E01 D +0.0074, CI [-0.0072, +0.0235]. Static on agreement and n_distinct without E01 D +0.0031, CI [-0.0072, +0.0133]. Greedy on cheap plus 16 PCs plus E01 D -0.0219, CI [-0.0471, +0.0173]. Static on cheap six plus E01 (seven columns, no PCs) D +0.0354, CI [+0.0112, +0.0582], which passes the same rule. That row was not pre-registered and is one of several feature sets computed in this run, so it is a lead for its own pre-registration and not a finding.

**Verdict against the pre-registered rules.**
- PRIMARY (label-free greedy on cheap six, no E01): KILL. Point D at 25% is -0.0079 (caught 245.4 against 247.6), which is <= 0, so the kill clause fires. The CI is [-0.0439, +0.0089]. It is wide relative to the gap, and 12 of 200 draws are above 0. The reading is that no detectable edge over the threshold shows up in this run. It is not a proof of a loss.
- CEILING (labeled static with E01, seed-mean): INCONCLUSIVE. Point D is +0.0083 (caught 250.0 against 247.6), so the kill clause does not fire, and the CI [-0.0167, +0.0347] includes 0, so it is not a PASS. The L10a +0.026 single-fit gap (caught 255) falls to +0.008 when the fit split varies. Per seed, static caught 254, 249, 257, 247, 245, 243, 252, 253 against the threshold's 250, 245, 248, 247, 248, 247, 247, 249 (static ahead on 5 seeds, tied on 1, behind on 2). The L10a +7.4 caught sits near the top of the per-seed range (243 to 257).
- Neither gated row is a PASS. Claim 2 stays open for labeled and low-dimensional learners only as a question. The L10a critic's motivation (greedy with three to six columns) gets no support in label-free form here: greedy on cheap six is still below the threshold (-0.008), and it is better than the 22-column greedy (-0.033).

**Wording for any claim 2 statement (carry the caveat).** On the E10 GSM8K pool at 25% (OLMo-2-0425-1B, 286 errors, 8 seeds), a label-free greedy learner on the six cheap rollout statistics does not beat the agreement threshold (D -0.008, CI [-0.044, +0.009]). A labeled static ranker with the E01 feature gives +0.008 seed-mean (CI [-0.017, +0.035]), which is not distinguishable from zero, and the L10a +7.4 caught was one fit split. Caveat, from L9b: the 3% confident-subset gate is unmet on both GSM8K families at the pre-registered 0.75 cutoff (E10 1.62%, E01 1.10%), so this result is a finding, not a verdict. E01 proxy_error is a cross-policy oracle scored against gold, so even a PASS would not license a single-policy allocator.

**Deviations and notes (logged here, as pre-registered).**
1. Two invocations were added beyond the spec's command shape (ref_agree_nd and ref_cheap_e01), because the greedy reference rows need their own column sets. This is pre-registration deviation 4.
2. ts_online and threshold_input_order were not run (not in the spec). The hand rule was generalized to 10% and 50% descriptively (pre-registration deviation 3).
3. The L10a copy's ts_online and the single-fit static arm were replaced by the per-seed refit (deviation 1) and the fixed-score bootstrap (deviation 2).
4. The hand-rule row reads PASS by the verdict label. It is post hoc and reads oracle E01 flags directly, so it is not an allocator candidate and is not gated.

**Scope.** Diagnostic on one pool (E10, GSM8K, OLMo-2-0425-1B), cell100, 8 seeds, 200 group-bootstrap resamples. One task, one policy, one bucketing. No GPU job was run.

**Follow-ups (not run, not pre-registered).** A labeled static ranker on the cheap six plus E01 (the descriptive +0.035 row) and the hand rule both need their own pre-registration before any claim is made from them. A second fit-split sensitivity and a bucket-cluster bootstrap (L11c) remain available.

## L11b 2026-10-18 kill

**Hypothesis.** On the E10 pool at 25% (545 of 2,180 sim groups, cell100, 8 seeds): (a) a label-free low-dimensional online learner (greedy_online on the six cheap rollout statistics, no E01) admits fewer corrupted rewards than the label-free agreement threshold; (b) the labeled static ranker with the E01 proxy_error feature beats the threshold across 8 fit resamples, not only on the L10a fit split. Pre-registration: the "L11b pre-registration" entry above.

**What was run.** Policy OLMo-2-0425-1B-Instruct, GSM8K, E10 pool: 2,180 sim groups, 75 cell100 buckets, 286 errors. Budget 25% = 545 docs (10% and 50% descriptive). 8 seeds. The static arm is refit per seed on a seeded 80% bucket subsample (19 of 24) of the fixed seed-0 fit quarter (572 to 708 fit docs, 76 to 100 fit errors). The 200-resample group bootstrap (BOOT_SEED 0) holds the per-seed static scores fixed. CPU only in the local shell, through avrl/exp/L11b.sh (five invocations of avrl/exp/l11b_low_dim_ceiling.py, then avrl/exp/l11b_assemble.py), under one minute, no GPU, no ssh, no pip install. Outputs: avrl/runs/L11b/ (result.json, run.log, assemble.log, exit_code 0, done.flag).

**Checks (all pass; result.json "checks" and assemble.log).** The control (cheap plus 16 PCs, no E01) reproduces avrl/runs/L10a/control_no_e01 exactly: max abs difference 0.0 on the point values and on all 200 greedy draws. Pool counts (2,180 / 286 / 75) match L10a in all five invocations. Uniform and threshold are identical across invocations, and the hand rule is identical in both E01 invocations. Every static fit has both classes, and the primary has no E01 column. Independent recheck: the per-seed static caught values in avrl/runs/L11b/ceiling/result.json (254, 249, 257, 247, 245, 243, 252, 253) and threshold values (250, 245, 248, 247, 248, 247, 247, 249) give means 250.0 and 247.625, matching the stored values. D = (caught(arm) minus caught(threshold)) / 286 gives -0.0079 for greedy and +0.0083 for static.

**Results (25%; D = admitted(threshold) minus admitted(arm); CEILING is an 8-seed mean; CI is the 200-resample group-bootstrap percentile; avrl/runs/L11b/result.json).**

| row | arm and columns | caught (sd) | admitted | D | 95% CI | status |
|---|---|---|---|---|---|---|
| threshold (baseline) | agreement, n_distinct | 247.63 (1.41) | 0.134 | 0 | n/a | baseline |
| PRIMARY (deployable, label-free) | greedy_online, cheap six, no E01 | 245.38 (1.65) | 0.142 | -0.0079 | [-0.0439, +0.0089] | KILL |
| CEILING (labeled, cross-policy feature) | static, cheap six + 16 PCs + e01_proxy_error, refit per seed | 250.00 (4.50) | 0.126 | +0.0083 | [-0.0167, +0.0347] | INCONCLUSIVE |
| uniform (reference) | random order | 72.75 (8.94) | 0.746 | -0.6115 | [-0.6552, -0.5716] | KILL |

Reference rows, not gated (status shown only for reading):

| row | arm and columns | caught (sd) | admitted | D 25% | 95% CI | status |
|---|---|---|---|---|---|---|
| L10a arm | greedy_online, cheap six + 16 PCs, no E01 | 238.13 (1.90) | 0.167 | -0.0332 | [-0.0590, -0.0019] | KILL |
| static | cheap six + 16 PCs, no E01 | 239.75 (1.85) | 0.162 | -0.0275 | [-0.0532, -0.0058] | KILL |
| greedy_online | agreement + n_distinct | 241.13 (0.93) | 0.157 | -0.0227 | [-0.0378, +0.0097] | KILL |
| greedy_online | cheap six + 16 PCs + E01 | 241.38 (1.73) | 0.156 | -0.0219 | [-0.0471, +0.0173] | KILL |
| greedy_online | cheap six + E01 | 250.63 (1.22) | 0.124 | +0.0105 | [-0.0280, +0.0328] | INCONCLUSIVE |
| static | cheap six, no E01 | 249.75 (0.97) | 0.127 | +0.0074 | [-0.0072, +0.0235] | INCONCLUSIVE |
| static | agreement + n_distinct, no E01 | 248.50 (0.87) | 0.131 | +0.0031 | [-0.0072, +0.0133] | INCONCLUSIVE |
| static | cheap six + E01 (post hoc, not pre-registered) | 257.75 (1.09) | 0.099 | +0.0354 | [+0.0112, +0.0582] | would PASS; descriptive only |
| hand rule (post hoc: threshold for 445 docs, then 100 E01-flagged) | 259.50 (0.50) | 0.093 | +0.0415 | [+0.0052, +0.0660] | would PASS; post hoc, oracle flags, not gated |

Descriptive at 10% and 50% (point D): PRIMARY -0.0481 and -0.0031; CEILING -0.0035 and +0.0101; hand rule +0.0114 and +0.0201. Ceiling per seed (static minus threshold, caught): +4, +4, +9, 0, -3, -4, +5, +4, so ahead on 5 seeds, tied on 1, behind on 2. The L10a single-fit gap was +7.4 caught (+0.026); the seed mean is +2.4 caught (+0.008).

**Verdict against the pre-registered rules.**
- PRIMARY (label-free greedy, cheap six, no E01): KILL. Point D at 25% is -0.0079, so the kill rule (point D at or below 0) fires. The CI [-0.0439, +0.0089] includes 0, and only 12% of draws are above 0, so it is not a pass.
- CEILING (labeled static with E01, 8-seed mean): INCONCLUSIVE. Point D is +0.0083, so the kill rule does not fire, and the CI [-0.0167, +0.0347] includes 0, so the pass rule does not fire. The L10a +7.4 caught was one fit split.
- No verdict is upgraded, and neither gated row is a pass.

**What it changes about the thesis claims.**
- Claim 2 (a learned allocator admits fewer corrupted rewards than any fixed rule at equal budget): not_supported on this pool. For the label-free learner, six cheap columns (D -0.008) beat 22 columns (D -0.033) and two columns (D -0.023), but still do not beat the threshold, so the greedy route is closed at 25% on E10. For the labeled route, no seed-mean edge over the threshold is visible. The one positive labeled row (static, cheap six + E01, D +0.035, CI [+0.011, +0.058]) is the best of five static feature sets computed here, so its CI is optimistic, and it needs its own pre-registration before it is claimed (L12a). The hand rule (+0.042) uses oracle E01 flags and one untuned split, so it is a reference ceiling, not an allocator. E01 is a cross-policy oracle, so even a PASS would not license a single-policy allocator.
- Claim 3 (the edge comes from structure in prompt space): the MiniLM PCs push the labeled arm down, not up. Static on cheap six without E01 gives +0.0074, and adding the 16 PCs gives -0.0275. With E01, cheap six gives +0.0354 and adding the PCs gives +0.0083. The labeled gain follows the cross-policy E01 signal rather than prompt-space structure, which is consistent with the L9a pre-registered KILL of the strong form. These rows are descriptive, so this is a direction, not a verdict.
- Wording: any claim 2 sentence must carry the L9b caveat. The 3% confident-subset gate is unmet on both GSM8K families at the pre-registered 0.75 cutoff (E10 1.62%, E01 1.10%), so this is a finding, not a verdict.

**Follow-ups (not run, not pre-registered; filed in lit/avrl/EXPERIMENT_BACKLOG.md as L12a to L12c).**
1. L12a, confirmatory test of the cheap six + E01 static arm (CPU, under 5 min). Information: high. It is the only labeled row whose CI excludes 0, but it was picked as the best of five, so its CI is optimistic. One pre-registered arm with a bucket-cluster bootstrap decides whether the gain is real or a selection effect.
2. L12b, hand-rule sensitivity to the E01 flagged count k in {0, 50, 100, 150, 200} (CPU, under 5 min). Information: medium. Shows whether +0.042 is a knife edge at the untuned k = 100, and so what bar a learned rule would have to clear against the oracle flag.
3. L12c, the PRIMARY arm on the E01 GSM8K family (Qwen2.5-1.5B-Instruct, avrl/runs/p0/feats.parquet, which has all six cheap columns) (CPU, under 2 min). Information: medium. Tests whether the label-free null (D -0.008) holds beyond the E10 pool. A tie or kill strengthens the claim 2 wording for label-free arms; a pass would reopen that route.

**Critic note (L11b, adversarial pass, 2026-10-18).** Formal verdicts stand: PRIMARY KILL (point D at 25% = -0.0079 <= 0, CI [-0.0439, +0.0089]) and CEILING INCONCLUSIVE (seed-mean point D = +0.0083 > 0, CI [-0.0167, +0.0347] includes 0), both mechanically correct under the pre-registered rules. Every table number in both L11b entries matches avrl/runs/L11b/*/result.json and run.log. The interpretation attached to them is corrected below, and one count is fixed.

Checked and clean. (a) An independent reimplementation (CPU, nothing written) reproduces the per-seed caught counts exactly: threshold 250, 245, 248, 247, 248, 247, 247, 249 (mean 247.625), greedy_online on the cheap six 244, 249, 245, 245, 246, 245, 246, 243 (mean 245.375, D -0.00787), static with E01 254, 249, 257, 247, 245, 243, 252, 253 (mean 250.0, D +0.00830). (b) No leakage into the PRIMARY: its six columns are rollout statistics and log question length, with no E01 column and no gold-derived column. The fit-quarter and sim pools share 0 prompts. E10 and E01 have identical question and gold text for all 3,000 pids, so the E01 join is aligned. E01 is a declared cross-policy oracle (P(E10 error | E01 error) = 0.50 against 0.07), used only in the CEILING and the reference rows. (c) Budget is 545 audited sim docs for every arm at 25% (hand = 445 + 100, no duplicates). The hand-rule fill-in never fires (124 to 128 flagged docs remain after the first 445 in every seed, 309 flagged in the sim pool). (d) No NaN in any feature, no warning, traceback or truncation line in any run.log, all exit codes 0. (e) Uniform, threshold and hand are identical across invocations, and the bootstrap resamples are paired across invocations (same BOOT_SEED and n).

Corrections.
1. "12 of 200 draws are above 0" (PRIMARY, result entry) is wrong. frac_resamples_D_gt_0 = 0.120 is 24 of 200 draws (counted from D_bootstrap_draws). The "12%" in the kill entry is right.
2. "Six cheap columns beat 22 columns and two columns" is a point ordering only, not supported at 95%. Paired bootstrap on the stored draws: greedy cheap six minus greedy agreement and n_distinct has mean +0.0012, CI [-0.0238, +0.0216]. Greedy cheap six minus greedy cheap + 16 PCs has mean +0.0185, CI [-0.0055, +0.0441] (189 of 200 draws positive). The "better than the 22-column greedy" line in the result entry has the same status.
3. "The greedy route is closed at 25% on E10" and "Claim 2: not_supported on this pool" (kill entry) are withdrawn. The KILL is the point-estimate rule, and the CI does not exclude a true edge of about +0.009 (2.5 of 286 errors). The pool resolves D only to about +-0.027 (about 7.7 errors), so it cannot exclude small edges either way. It is one greedy configuration (K = 1, logistic refit every 68 labels, 8 seeds). Per the L9b gate failure the claim-2 readouts on this pool are descriptive with no claim-2 verdict. Supported wording: no detectable edge for greedy_online on the cheap six at 25%. The header "L11b 2026-10-18 kill" should read "PRIMARY KILL, CEILING INCONCLUSIVE".
4. "The L10a +7.4 was one fit split" and "falls to +0.008 when the fit split varies" confound split variation with less training data. Every seed fits on 80% of the fit buckets (19 of 24), and L10a fit on all 24. Recomputed on the same pool with the same pipeline: full fit catches 255 (L10a value reproduced), and 40 random subsamples catch a mean of 246.1 at 60% of buckets, 251.1 at 80% and 253.2 at 90% (sd 5.8, 4.6, 3.0). Caught rises with fit fraction, so the 80% seed-mean (+0.008) understates the full-fit effect, the 8 overlapping subsamples are not independent fits, and the bootstrap CI ignores fit variance (deviation 2). The INCONCLUSIVE label is correct under the pre-registered 80% refit. What is not established is that the L10a +0.026 was a lucky split. Resolving it needs a design that varies the fit set at fixed size (for example swapping fit and sim bucket folds), not 80% subsampling.
5. Equal-budget and "label-free" wording. The 545 budget is equal in audited sim docs only. The static arm trains on 572 to 708 labeled fit docs (the whole fit quarter is 819) and the hand rule and E01 column read oracle flags on every doc, none of it charged to the budget. greedy_online is "label-free" only in the sense that it has no gold-derived feature and no pre-audit labels. It refits on the labels of docs it has already audited, which are counted inside the 545.
6. The Claim 3 line on the MiniLM PCs is a direction only. The paired fixed-fit bootstrap for static cheap + E01 minus static cheap + PCs + E01 is mean +0.0266, CI [+0.0102, +0.0418], but the 23-column fit is also the noisier one (sd 4.50 across seeds against 1.09), and the CI excludes fit variance.

Corrected verdict. PRIMARY: KILL under the pre-registered point rule, unchanged and not upgraded; reading: no detectable edge over the threshold (D -0.0079, CI [-0.0439, +0.0089], 24 of 200 draws above 0), not a proof of a loss and not a closed route. CEILING: INCONCLUSIVE, unchanged; the L10a single-fit gain is neither confirmed nor shown to be a fluke. Claim 2: descriptive under L9b, no verdict, still open. The post hoc rows (static cheap six + E01 at +0.0354, hand rule at +0.0415) stay leads for their own pre-registration (L12a, L12b). Claim 3 stays as logged by L9a, and nothing in L11b changes it.

## E06 pre-registration (written before the GPU step, no E06 outcome seen)

Date: 2026-10-18. Written before any E06 script ran and before any E06 output file was opened. No E06 outcome has been seen.

**Hypothesis.** On the same 3,000 GSM8K train prompts as E01 and E10 (sample seed 0, G 8, max_tokens 512), a smaller policy in the same family, Qwen2.5-0.5B-Instruct, produces (a) enough majority-vote proxy errors to resolve the claim 2 readout, which the existing pools cannot (E01: 436 group errors over 3,000 groups; E10: 398 over 3,000, and 286 in its 2,180-group sim pool, where D resolves only to about +/-0.027), and (b) a confident-subset (agreement >= 0.75) proxy error rate at or above the 3% gate, which the 1.5B pool (1.10% of 1,551 confident groups) and the OLMo-1B pool (1.62% of 2,158) both fail. It also supplies the model-size axis for claim 3. Assumption, unverified until run: the 0.5B group error rate is well above the 1.5B rate of 14.5%.

**Policy.** Qwen2.5-0.5B-Instruct from /homes/rdubash/active-audit/models (shared, read-only; the directory is present). Same family as the E01 policy (Qwen2.5-1.5B-Instruct) at 0.5B, so the claim 3 axis is model size within one family. A pass or kill applies to this policy only.

**Scripts and paths (written before the run).**
- avrl/exp/E06.sh: copy of avrl/exp/E10.sh. Changes: --model /homes/rdubash/active-audit/models/Qwen2.5-0.5B-Instruct and --out avrl/runs/E06. Nothing else. avrl/p0_rollouts.py is unchanged and runs under .venv-judge with --n_prompts 3000 --G 8 --max_tokens 512 --seed 0.
- GPU step: bin/gpu_run.sh with run dir avrl/runs/E06, started only after bin/gpu_status.sh shows under 2 GB used (checked again immediately before launch). Polled in band until done.flag. Hard cap 45 minutes, enforced by the helper.
- avrl/exp/e06_analyze.py: copy of avrl/exp/e10_analyze.py. The one code change is the MiniLM call, device="cpu" (was "cuda"). It runs in the local shell with /homes/rdubash/active-audit/.venv after the GPU step exits 0. The allocation replay still uses the unchanged avrl/runs/p0/sim_alloc.py.
- E03-style replay: avrl/exp/e03_threshold_bootstrap.py, unchanged, run as a separate CPU call after e06_analyze.py, as E10 did (e10_analyze.py does not write the replay): --feats avrl/runs/E06/feats.parquet --cell cell100 --features cheap+emb --budgets 0.10,0.25,0.50 --seeds 8 --boot 200 --boot_budget 0.25 --workers 8 --out avrl/runs/E06/replay (BOOT_SEED 0).
- avrl/exp/e06_gate.py: new. For any run dir it computes the confident-subset group count n and error count k at cutoff 0.75 (primary) and 0.625 (descriptive), the rate k/n with a Wilson 95% interval, and the truncated-rollout share (mean of the truncated column in rollouts.parquet). It is first run on avrl/runs/p0 and avrl/runs/E10 as a reproduction check, which must return the logged values (E01: 1,551 confident groups at 1.10%; E10: 2,158 confident groups with 35 errors, 1.62%). For E06 it also writes avrl/runs/E06/gate.json.
- avrl/exp/e06_assemble.py: new. Copy of avrl/exp/e10_assemble.py that imports e06_gate.py, applies the rules below, and writes avrl/runs/E06/result.json.

**Metrics.**
- Primary (gate): confident-subset (agreement >= 0.75) proxy error count k, group count n, rate k/n with Wilson 95% CI, at cutoff 0.75.
- Claim 2 readout: admitted rate at 25% (1 minus caught over total proxy errors, 8-seed mean) for threshold, ts_online and greedy_online. D(ts_online) = admitted(threshold) minus admitted(ts_online), with its 200-resample group bootstrap 95% CI. D at 10% and 50%, and D(greedy_online) at 25%, are descriptive.
- Claim 3 readout (all groups, as e10_analyze.py computes it): AUC for proxy_error with cheap and with cheap+emb (5-fold balanced logistic regression), their gain, and the cell100 neighbor ratio.
- Secondary, descriptive: the cutoff 0.625 gate values, rollout accuracy, majority accuracy, group proxy error rate, and the truncated-rollout share.

**Rules (fixed before the run).**
- Gate states at cutoff 0.75:
  - OPEN: rate >= 0.03 AND k >= 100.
  - KILL: rate < 0.03. GSM8K majority vote then fails the gate at both policy sizes, and the next step is a task change, not more rollouts.
  - INCONCLUSIVE: rate >= 0.03 but k < 100 (power-limited, descriptive only).
  - TRUNCATION-FLAGGED: if the truncated-rollout share is above 0.10, no gate verdict is issued and the gate reading is flagged as truncation-confounded (L9c logic), whatever its rate.
- Claim 2, only when the gate is OPEN and the truncated share is at or below 0.10: KILL if point D(ts_online) at 25% <= 0. PASS if D > 0 and its 95% bootstrap CI excludes 0. INCONCLUSIVE if D > 0 and the CI includes 0. In any other gate state the claim 2 readout is descriptive and no claim 2 verdict is issued.
- Claim 3 is not gated by the confident-subset gate, because the E06 spec gives it no gate condition. E10 did gate its claim 3 PASS, so this is a difference from E10, recorded here before the run. PASS (structure present) if AUC gain (cheap+emb minus cheap) >= 0.05 AND cell100 neighbor ratio >= 1.5. KILL if gain < 0.05 AND ratio < 1.5. INCONCLUSIVE otherwise. If the truncated share is above 0.10, the claim 3 verdict is still computed and is reported with the truncation share and the L9c caveat.
- Failures: a non-zero exit or crash is recorded as failed with the exact error text. The ground rules allow one fix and one rerun after a crash. A run that hits the 45 minute cap is recorded as failed.

**Expected outputs.** avrl/runs/E06/: run.log, exit_code, done.flag, cmd.sh, wrap.sh, pgid (GPU step, written by bin/gpu_run.sh; the CPU steps do not write a flag, as in E10); rollouts.parquet, groups.parquet, summary.json (rollouts); feats.parquet, result.json, sim_cell100_*.json, sim_cell30_*.json, analyze.log (analysis); replay/result.json, replay/run.log (E03-style replay); gate.json (e06_gate.py); result.json overwritten by e06_assemble.py with the merged metrics and verdicts.

**Expected cost.** About 12 to 15 GPU minutes of generation (estimate from E01 at 1,088 s and E10 at 821 s), under the 45 minute cap. About 5 to 10 CPU minutes for the analysis and the replay.

**Caveats (stated before the run).**
1. One policy (Qwen2.5-0.5B-Instruct), one task (GSM8K, 3,000 prompts), one rollout set per prompt at G 8, majority-vote proxy only, no judge arm.
2. The matches_E01 reproduction check will print False, as expected, because the rollouts differ.
3. D resolution depends on the pool's error count. The result reports the pool size and error count with each verdict.
4. A kill is logged as a kill. No other policy is run to rescue it. The gate threshold (3%) and the k >= 100 power floor are not changed after seeing E06.

## E06 2026-10-18 inconclusive (gate power-limited, descriptive only; claim 3 kill on this policy; claim 2 no verdict). Policy: Qwen2.5-0.5B-Instruct, E01 GSM8K prompts

**Hypothesis and rules.** As pre-registered above (line 1215). Same 3,000 GSM8K train prompts as E01 (sample seed 0), G 8, max_tokens 512, with Qwen2.5-0.5B-Instruct in place of Qwen2.5-1.5B-Instruct. Gate at cutoff 0.75: OPEN needs a confident-subset (agreement at least 0.75) proxy error rate of at least 3% and k at least 100; KILL if the rate is below 3%; INCONCLUSIVE if the rate is at least 3% and k is below 100.

**What was run.** GPU step through bin/gpu_run.sh: exit 0, 24,000 rollouts, generation 747 s (E10 821 s, E01 1,088 s), no traceback. CPU: avrl/exp/e06_analyze.py (the E10 analysis with MiniLM on CPU; the diff against e10_analyze.py is that one call plus the docstring), avrl/exp/e03_threshold_bootstrap.py as a separate replay (cell100, cheap+emb, budgets 10/25/50% = 218/545/1,090 docs, 8 seeds, 200 group-bootstrap resamples, BOOT_SEED 0), and avrl/exp/e06_gate.py for the gate. Sim pool: 2,180 groups in 75 cell100 buckets with 841 proxy errors. The sim pool is the same prompt set as E10 (cell100 assignments agree on 3,000 of 3,000 prompts), which explains the matching 2,180 count.

**Numbers.** E01 and E10 columns are from the E10 entry above, or recomputed from their groups.parquet and rollouts.parquet as checked below.

| quantity | E06 Qwen2.5-0.5B | E01 Qwen2.5-1.5B | E10 OLMo-2 1B | rule or baseline |
|---|---|---|---|---|
| rollout accuracy / majority accuracy | 40.2% / 61.6% | 64.2% / 85.5% | 76.2% / 86.7% | descriptive |
| group proxy error (all 3,000) | 38.4% (1,152) | 14.5% (436) | 13.3% (398) | descriptive |
| unanimous groups / proxy error when unanimous | 4.4% / 3.0% | 18.6% / 0.5% | 45.6% / 0.6% | descriptive |
| confident subset (agreement at least 0.75): n / k / rate | 726 / 31 / 4.27% | 1,551 / 17 / 1.10% | 2,158 / 35 / 1.62% | gate: rate at least 3% AND k at least 100 |
| Wilson 95% on the confident rate | 3.02% to 6.00% | 0.69% to 1.75% | 1.17% to 2.25% | descriptive |
| cutoff 0.625 (descriptive): n / k / rate | 1,089 / 80 / 7.35% | 1,960 / 38 / 1.94% | 2,384 / 68 / 2.85% | descriptive |
| truncated rollout share | 6.5% (1,559 of 24,000) | 2.2% (527 of 24,000) | 6.75% (1,620 of 24,000) | flag if above 10% |
| AUC proxy error, cheap / cheap+emb / gain | 0.887 / 0.883 / -0.003 | 0.923 / 0.922 / -0.001 | 0.932 / 0.931 / -0.002 | claim 3 kill: gain below 0.05 and ratio below 1.5 |
| cell100 neighbor ratio | 0.999 | 1.03 | 0.998 | claim 3 kill: below 1.5 |
| confident subset AUC cheap / cheap+emb (gain) | 0.588 / 0.648 (+0.060) | not recomputed | not recomputed | descriptive, 31 errors |
| admitted at 25%: uniform / threshold / ts_online / greedy_online | 0.751 / 0.472 / 0.595 / 0.487 | not recorded here | 0.746 / 0.134 / 0.337 / 0.167 | descriptive |
| D(ts_online) at 25%, 95% CI | -0.124 [-0.141, -0.100] | -0.172 [-0.230, -0.142] | -0.203 [-0.260, -0.175] | gated (see verdict) |
| D(ts_online) at 10% / 50% | -0.082 / -0.131 | not compared | -0.257 / -0.052 | descriptive |
| D(greedy_online) at 25%, 95% CI | -0.016 [-0.030, +0.002] | not compared | -0.033 [-0.059, -0.002] | descriptive |

E06 replay by budget (cell100, cheap+emb, 8-seed means; admitted = share of the 841 proxy errors not caught):

| budget (docs) | uniform | threshold | ts_online | greedy_online | D(ts_online) |
|---|---|---|---|---|---|
| 10% (218) | 0.900 | 0.746 | 0.828 | 0.795 | -0.082 |
| 25% (545) | 0.751 | 0.472 | 0.595 | 0.487 | -0.124 |
| 50% (1,090) | 0.495 | 0.152 | 0.283 | 0.162 | -0.131 |

**Verification.** The run report was not trusted; the numbers were recomputed from avrl/runs/E06/groups.parquet, rollouts.parquet and replay/result.json (D_bootstrap_draws). All match: confident n 726, k 31 (4.27%); cutoff 0.625 n 1,089, k 80; truncation 1,559 of 24,000; D(ts_online) at 25% -0.1235 with CI [-0.1413, -0.0997] and 0 of 200 draws above 0; D(greedy_online) CI [-0.0298, +0.0015] with 6 of 200 above 0. Gate-code reproduction on the logged pools: E01 n 1,551, k 17 (1.10%); E10 n 2,158, k 35 (1.62%, Wilson 1.17% to 2.25%). The replay's matches_E01 False (max abs diff 416) compares against the E01 replay json, which is a different rollout pool, so it is an expected mismatch, not a reproduction; the E01 and E10 gate values above are the reproduction. No traceback in run.log, analyze.log or replay/run.log. The only ERROR line ("Engine core proc EngineCore died unexpectedly") follows "Shutdown complete", as in E10 and L7b. exit_code is 0.

**Verdicts against the pre-registered rules.**
- Gate: INCONCLUSIVE (power-limited, descriptive only). The rate 4.27% clears the 3% bar; k = 31 misses the k at least 100 floor. Not a kill and not an open gate. Assumption (b) holds. Assumption (a) holds for total errors (1,152 against 436) but not for confident errors.
- Claim 2: no verdict, because the gate is not open and the readout is descriptive by rule. Descriptively, the learned ts_online admits 0.595 of the proxy errors at 25%, against 0.472 for the label-free threshold and 0.751 for uniform. D(ts_online) = -0.124, 95% CI [-0.141, -0.100], 0 of 200 draws above 0. The same sign appears on E10 and E01.
- Claim 3: KILL under the pre-registered rule, which does not gate on the confident subset. Pooled AUC gain -0.003 (bar 0.05) and cell100 neighbor ratio 0.999 (bar 1.5). Truncation share 6.5% is below 0.10, so there is no truncation flag.
- Overall: inconclusive. The gate is not open, claim 3 is killed on this policy, and claim 2 is undecided.

**Power arithmetic (not in the run report).** k at least 100 at the observed confident rate needs about 2,340 confident groups. Confident groups are 24.2% of prompts (726 of 3,000), so that is about 9,700 prompts. GSM8K does not have that many: 7,473 train plus 1,319 test is 8,792 (avrl/data/). At all 8,792 prompts the expected k is about 91 at the same rate, so cutoff 0.75 cannot reach the k floor on this corpus with this policy unless the rate rises. At cutoff 0.625, the 7,473 train prompts would give an expected k of about 199 at a rate of about 7.3%. Those are extrapolations from E06, not results.

**Truncation check on the confident subset.** Truncation is low in the confident subset (0.67% of its rollouts). However, 4 of 31 confident error groups contain a truncated rollout, against 30 of 695 confident non-error groups (13% against 4%). With 31 errors this cannot be read, so it is carried into L13c.

**Process notes.**
1. The pre-registration block (line 1215) contains no E06 outcome numbers.
2. The older backlog entry "E06: Model size variation, small end" in lit/avrl/EXPERIMENT_BACKLOG.md describes an earlier plan (avrl/runs/p0_small, 8 to 12 min) and assumes E01 passed claim 3. The record says E01 failed the claim 3 bar (E10 entry above). The pre-registration governs this run. That backlog entry is left unedited.
3. The 0.625 cutoff is descriptive here. Using it as a gate would be a post hoc change and needs its own pre-registration (L13a). It cannot rescue E06.

**Thesis claims after E06.**
- Claim 2 (learned allocator admits fewer corrupted rewards than any fixed rule at equal budget): untested on a gate-passing pool. The gate has not opened on any policy so far. The descriptive direction is the same on three policies (E01, E10, E06): the label-free threshold beats ts_online at 25%. No verdict is issued.
- Claim 3 (edge comes from structure in prompt space): KILL on this policy. Pooled structure is absent on all three GSM8K policies tested (1.5B, OLMo-2 1B, 0.5B). The only lead is the confident-subset AUC gain of +0.060 on 31 errors, which is descriptive and untested.
- Model-size axis (Qwen family only): total proxy error rises as the policy shrinks (14.5% at 1.5B, 38.4% at 0.5B), and so does the confident-subset rate (1.10% to 4.27%). Only the 0.5B pool passes the 3% rate bar. OLMo-2 1B (1.62%) is a different family, so it is not on this axis.

**Follow-ups (filed in lit/avrl/EXPERIMENT_BACKLOG.md under Proposed by the loop; ids use loop iteration 13).**
1. L13a, pre-registered cutoff-0.625 gate on all 7,473 GSM8K train prompts, 0.5B policy (GPU, one job, about 31 min at the E06 rate, under the 45-minute cap). Information: high. It decides whether a claim 2 readout is possible on GSM8K at all. Caveat: choosing 0.625 after E06 is post hoc, so it must be pre-registered as a new gate before the run.
2. L13b, L10a oracle-ceiling replay on the E06 pool, with the E01 proxy_error as the allocator feature (CPU, under 5 min). Information: medium. The E06 pool has 841 sim errors against 286 on E10, and its D(ts_online) CI half-width is about 0.02 against about 0.04 on E10. It separates a feature-limited failure from a learner-limited one. It is diagnostic only and cannot give a claim 2 verdict while the gate is closed.
3. L13c, confident-subset claim 3 check on the E06 pool, with the truncation cross-tab above (CPU, under 3 min). Information: low. With 31 errors the CI on the +0.060 gain will probably include 0, so the likely result is INCONCLUSIVE. It only says whether the lead justifies the full-pool GPU run in L13a.

**Files.** avrl/exp/E06.sh, avrl/exp/e06_analyze.py, avrl/exp/e06_gate.py, avrl/exp/e06_assemble.py. Outputs in avrl/runs/E06/: run.log, exit_code (0), done.flag, summary.json, rollouts.parquet, groups.parquet, feats.parquet, result.json, gate.json, analyze.log, sim_cell100_*.json, sim_cell30_*.json, replay/result.json, replay/run.log.

**Critic note (E06, 2026-10-18).** Verdict unchanged: gate INCONCLUSIVE (power-limited, descriptive only), claim 2 NO VERDICT, claim 3 KILL on this policy, overall inconclusive. Checked against runs/E06/summary.json, result.json, gate.json, groups.parquet, rollouts.parquet, feats.parquet, run.log, analyze.log, replay/result.json, replay/run.log, cmd.sh, wrap.sh, exit_code, and the scripts E06.sh, e06_analyze.py, e06_gate.py, e06_assemble.py, e03_threshold_bootstrap.py (diffs against E10.sh and e10_analyze.py are the ones the entry states: model path, output dir, MiniLM on CPU). Every number in the pre-registration, the run table, the replay table and the verification paragraph matches the raw files. I recomputed from the parquet files: group error 1,152 (0.384), rollout accuracy 0.40221, majority accuracy 0.616, unanimous 0.04433 with error 0.03008, confident subset 726 with 31 errors (0.04270, Wilson [3.02%, 6.00%]), cutoff 0.625 1,089 with 80 (7.35%), 1,559 of 24,000 rollouts at the 512-token cap (6.50%, 887 groups), AUCs 0.8866 / 0.8834 (gain -0.0032), cell100 neighbor ratio 0.9989, confident-subset AUCs 0.5878 / 0.6477. Agreement rebuilt from the raw answers matches groups.parquet, zero duplicate (pid, k), zero NaN, same questions and gold as E10 for all 3,000 pids, cell100 assignments identical to E10 and E01 on 3,000 of 3,000 prompts, and the rebuilt sim pool is 2,180 groups in 75 buckets with 841 errors (E10 286, E01 309). I re-ran e03_threshold_bootstrap.py unchanged into avrl/runs/E06/critic/replay_rerun/ (log replay_rerun.log): point table, D points and all 200 bootstrap draws are identical to replay/result.json (D(ts_online) -0.1235, CI [-0.1413, -0.0997], 0 of 200 draws above 0; D(greedy_online) CI [-0.0298, +0.0015], 6 of 200 above 0). The E01 and E10 reference rows match p0/result.json, E03/result.json and E10/replay/result.json. Gate and claim rules were applied correctly: rate 4.27% >= 3% and k 31 < 100 is INCONCLUSIVE; truncation 6.5% <= 10% so no flag; gain -0.003 < 0.05 and ratio 0.999 < 1.5 is a claim 3 KILL; claim 2 is descriptive because the gate is not open. No leakage: the allocator and AUC features are the 22 listed columns (agreement, n_distinct, token statistics, any_truncated, log_qlen, 16 question-embedding PCs), none gold-derived, and the online arms see proxy_error only for audited groups. Every arm gets the same budget in groups (218 / 545 / 1,090), 8 allocator seeds, and the threshold arm reads no labels. No traceback in run.log, analyze.log or replay/run.log; the one ERROR line follows "Shutdown complete". Pool-level checks: the cell100 neighbor ratio of 0.9989 sits at its permutation null (200 label permutations: mean 1.0001, sd 0.0012); the cell30 ratio of 1.0 is 1.0 by construction (every doc has a neighbor hit), as noted in the E10 critic note.

Corrections, none changes the verdict. (1) The run table cell "confident subset AUC cheap / cheap+emb (gain): not recomputed" for E01 and E10 is wrong: both are stored. E01 (p0/result.json) 0.627 / 0.534, gain -0.093; E10 0.768 / 0.705, gain -0.063. Both gains are negative, the opposite sign to E06's +0.060. (2) The "only lead" framing of the E06 confident-subset gain of +0.060 is stronger than the data allow. The +0.060 comes from the script's unshuffled 5-fold split on 31 errors. Over 20 shuffled-fold repeats of the same pipeline the E06 gain is mean +0.018, sd 0.032, range -0.042 to +0.076, so the sign is not stable; the same repeat gives E01 -0.065 and E10 -0.063. The entry also omits that the confident-subset cell100 neighbor ratio is 0.464 (result.json; E01 1.159, E10 0.608), which is below 1 and gives no neighbor-structure support. Read the lead as noise-compatible, not as a lead; L13c remains low information and should not be used to justify L13a. (3) The power arithmetic sentence "cutoff 0.75 cannot reach the k floor on this corpus with this policy unless the rate rises" overstates a point extrapolation. Expected k at all 8,792 prompts is about 91 to 94, and reaching k >= 100 needs a true rate of about 4.7%, which is inside the Wilson interval [3.0%, 6.0%]. A Beta-posterior simulation (rate Beta(32, 696), confident share Beta(727, 2275), binomial draws) gives P(k >= 100) of about 0.36 at 8,792 prompts and about 0.12 on the 7,473 train prompts alone. So cutoff 0.75 is unlikely but not excluded on the full corpus. The cutoff 0.625 extrapolation holds up: expected k about 202 on 7,473 prompts, 5th percentile about 160, so L13a would clear a k >= 100 floor at that cutoff almost surely (the cutoff choice itself stays post hoc, as the entry says). (4) replay/result.json carries the E03 script's own ungated "verdict": "KILL" (D(ts_online) <= 0). It is not the pre-registered E06 claim 2 verdict, which is NO VERDICT because the gate is not open. (5) Scope, not an error: the gate counts use all 3,000 groups, while the claim 2 replay uses the 2,180-group sim pool (75 of 100 cell100 buckets; the other quarter of buckets is the fit split), where the confident subset is 517 groups with 27 errors. A gate-open reading would have needed to say which population it applied to. The truncation cross-tab in the entry (4 of 31 confident error groups against 30 of 695 non-error groups hold a capped rollout) has Fisher exact two-sided p = 0.051, so it is borderline rather than unreadable, and still descriptive. Also noted, not a correction: 121 of 24,000 rollouts have an empty extracted answer (20 groups have an empty majority, none of them in the 31 confident errors), a property of the unchanged p0_rollouts.py extractor.

Corrected verdict: unchanged. Gate INCONCLUSIVE (confident-subset rate 4.27%, 31 of 726, at cutoff 0.75; k 31 < 100; truncation 6.5% <= 10%). Claim 2: NO VERDICT; descriptively D(ts_online) at 25% = -0.1235, CI [-0.1413, -0.0997], 0 of 200 draws above 0, reproduced by an independent rerun. Claim 3: KILL on this policy (AUC gain -0.0032 < 0.05 and cell100 neighbor ratio 0.9989 < 1.5, at the permutation null); the confident-subset +0.060 is not a stable lead (shuffled-fold mean +0.018, E01 and E10 negative, subset neighbor ratio 0.46). Overall: inconclusive. Scope: Qwen2.5-0.5B-Instruct, 3,000 GSM8K train prompts, G 8, 512 tokens, one rollout sample, majority-vote proxy, 22 cheap+emb features, 8 allocator seeds.

## L13a pre-registration (written before the GPU step, no L13a outcome seen)

Date: 2026-10-19. Written before any L13a script ran and before any L13a output file was opened. No L13a outcome has been seen.

**Hypothesis.** On all 7,473 GSM8K train prompts (G 8, max_tokens 512, sample seed 0), the Qwen2.5-0.5B-Instruct policy has a confident-subset (agreement at least 0.625) proxy error rate of at least 3% with k at least 100 errors. If so, the gate opens and a claim 2 readout on GSM8K becomes possible. The expected k of about 199 and rate of about 7.3% are extrapolations from E06's descriptive 0.625 row (1,089 groups, 80 errors), not results.

**New gate, cutoff chosen after E06.** The 0.625 cutoff was chosen after E06 had been read, because E06 reported it as a descriptive row. L13a is therefore a new gate with its own pre-registration (this entry). It does not change the E06 verdict (inconclusive at cutoff 0.75, claim 3 kill on this policy, claim 2 no verdict). The 0.75 row is reported as descriptive only.

**Prompt overlap, stated before the run.** avrl/p0_rollouts.py draws prompts with pandas sample(n, random_state=0). At n = 7,473 this is a full permutation of the train set. The first 3,000 of that permutation are expected to be the E06 prompts, because numpy's legacy choice without replacement is a prefix of one permutation. This is to be checked after the run by comparing question text in pid order against avrl/runs/E06/groups.parquet. The rollouts are freshly generated, so the L13a pool is not an independent replication of E06. Up to 3,000 of its 7,473 groups share prompts with E06.

**Policy.** Qwen2.5-0.5B-Instruct from /homes/rdubash/active-audit/models (shared, read-only). The same policy size as E06. A pass or kill applies to this policy only.

**Scripts and paths (written before the run).**
- avrl/exp/L13a.sh: copy of avrl/exp/E06.sh. Changes: --n_prompts 7473 and --out avrl/runs/L13a, plus the header comment. Nothing else. avrl/p0_rollouts.py is unchanged and runs under .venv-judge with --G 8 --max_tokens 512 --seed 0.
- GPU step: bin/gpu_run.sh with run dir avrl/runs/L13a, started only after bin/gpu_status.sh shows under 2 GB used (checked again immediately before launch). Polled in band until done.flag. Hard cap 45 minutes, enforced by the helper (timeout 2700 s, exit 124 recorded as failed).
- CPU step: avrl/exp/e06_analyze.py and avrl/exp/e06_gate.py, unchanged, run on avrl/runs/L13a with /homes/rdubash/active-audit/.venv. e06_gate.py writes avrl/runs/L13a/gate.json with both cutoffs. e06_analyze.py also writes avrl/runs/L13a/result.json, which is renamed to analyze_result.json before the readout so the readout can own result.json.
- avrl/exp/L13a.py: new and small. It reads gate.json, recomputes n and k at cutoff 0.625 from groups.parquet, checks exit_code, the run.log traceback state and the rollout count, applies the rules below, and writes avrl/runs/L13a/result.json.

**Metrics.**
- Primary (gate): confident-subset (agreement at least 0.625) proxy error count k, group count n, and rate k/n with Wilson 95% interval, read from the 0.625 row of gate.json and confirmed from groups.parquet.
- Truncated-rollout share, flagged in the report if above 0.10.
- Descriptive only: the 0.75 row, rollout accuracy, majority accuracy.

**Rules (fixed before the run).**
- PASS (gate opens): rate at least 0.03 AND k at least 100. The claim 2 readout (D(ts_online) at 25%, bucket-cluster 95% CI) then runs as its own pre-registered item. It is not run by this entry.
- KILL: rate below 0.03. GSM8K majority vote then fails the gate for the 0.5B policy on both cutoffs, and the next step is a task change.
- INCONCLUSIVE: rate at least 0.03 but k below 100 (power-limited, descriptive only).
- This entry cannot kill claim 2 by itself. It only decides whether the claim 2 readout can run.

**Failures.** A non-zero exit code, a traceback in run.log, or a 45-minute timeout (exit 124) is recorded as a failed experiment with the exact error text. One fix and one rerun are allowed after a crash. A second failure returns ok=false.

**Expected cost.** About 31 GPU minutes of generation (747 s per 3,000 prompts, scaled by 7,473 / 3,000, about 1,860 s), plus vLLM startup, under the 45-minute cap. About 10 CPU minutes for the analysis.

## L13a 2026-10-19 pass

**Hypothesis.** As pre-registered in the "L13a pre-registration" entry above: on all 7,473 GSM8K train prompts with Qwen2.5-0.5B-Instruct (G 8, max_tokens 512, sample seed 0), the confident-subset (agreement at least 0.625) proxy error rate is at least 3% with k at least 100 errors. The 0.625 cutoff was chosen after E06, so this is a new gate, not a re-reading of E06.

**What was run.** avrl/exp/L13a.sh (copy of avrl/exp/E06.sh with --n_prompts 7473 and --out avrl/runs/L13a), run through bin/gpu_run.sh after bin/gpu_status.sh showed 15 MiB used on pine. Policy Qwen2.5-0.5B-Instruct (shared, read-only), vLLM, G 8, max_tokens 512, seed 0, 7,473 prompts, 59,784 rollouts. Generation 1,824 s, exit code 0, wall time about 33 min. CPU: avrl/exp/e06_analyze.py and avrl/exp/e06_gate.py on avrl/runs/L13a (the analyze result.json was renamed analyze_result.json before the readout). This is a gate only: there is no allocation budget and no claim 2 readout.

**Numbers.** Recomputed from avrl/runs/L13a/groups.parquet and rollouts.parquet. They match gate.json and the run report. The E01 and E10 columns are taken from the E06 entry above.

| quantity | L13a, Qwen2.5-0.5B, 7,473 prompts | E06, same policy, 3,000 prompts | E01, Qwen2.5-1.5B, 3,000 | E10, OLMo-2 1B, 3,000 | rule or baseline |
|---|---|---|---|---|---|
| confident 0.625: n / k / rate | 2,628 / 161 / 6.13% (Wilson 5.27% to 7.11%) | 1,089 / 80 / 7.35% | 1,960 / 38 / 1.94% | 2,384 / 68 / 2.85% | gate: rate at least 3% AND k at least 100 |
| confident 0.75 (descriptive): n / k / rate | 1,773 / 59 / 3.33% (2.59% to 4.27%) | 726 / 31 / 4.27% | 1,551 / 17 / 1.10% | 2,158 / 35 / 1.62% | descriptive; rate passes, k below 100 |
| confident 0.625 on the 3,000 E06 prompts (regenerated) | 1,087 / 72 / 6.62% | (E06 original: 1,089 / 80) | n/a | n/a | overlap check |
| confident 0.625 on the 4,473 prompts E06 did not use | 1,541 / 89 / 5.78% (4.72% to 7.05%) | n/a | n/a | n/a | fresh-prompt check: k below 100 |
| rollout accuracy / majority accuracy | 39.7% / 59.8% | 40.2% / 61.6% | 64.2% / 85.5% | 76.2% / 86.7% | descriptive |
| group proxy error rate (3,005 of 7,473 groups) | 40.2% | 38.4% | 14.5% | 13.3% | descriptive |
| truncated rollout share | 6.70% (4,004 of 59,784) | 6.5% | 2.2% | 6.75% | flag above 10%: not flagged |
| AUC proxy error, cheap / cheap+emb (gain), all groups | 0.8901 / 0.8897 (-0.0004) | 0.887 / 0.883 (-0.003) | 0.923 / 0.922 (-0.001) | 0.932 / 0.931 (-0.002) | claim 3 kill bar: gain below 0.05 and neighbor ratio below 1.5 |
| cell100 neighbor ratio | 1.00 (cell30 1.00, cell300 0.999) | 0.999 | 1.03 | 0.998 | claim 3 kill bar: below 1.5 |
| AUC on confident subset at 0.75, cheap / cheap+emb (gain) | 0.7105 / 0.7398 (+0.029), 59 errors | 0.588 / 0.648 (+0.060), 31 errors | not recomputed | not recomputed | descriptive |
| D(ts_online) at 25%, 95% CI | not run here (separate pre-registered readout) | -0.124 [-0.141, -0.100] | -0.172 [-0.230, -0.142] | -0.203 [-0.260, -0.175] | descriptive |

**Checks.** Exit code 0. run.log has 0 tracebacks. 7,473 groups with unique pids, 8 rollouts each, 59,784 rollouts (the run report is right; the spec's "24,000 rollouts" is the E06 count, 3,000 x 8, and was wrong for this run). Overlap check (pre-registered): question text and gold labels in pid order for pids 0 to 2999 are identical to avrl/runs/E06/groups.parquet, so the first 3,000 prompts of the seed-0 permutation are the E06 prompts. Rollout accuracy, majority accuracy and the truncated share match the run report. The only ERROR line in run.log ("Engine core proc EngineCore died unexpectedly, shutting down client") comes after "generated 59784 rollouts in 1822s", the summary and "Shutdown complete". All outputs were written before it, and the same vLLM shutdown line appears in the E06 log.

**Verdict against the pre-registered rules.** PASS. At cutoff 0.625 the rate is 6.13%, which is at least 3%, and k = 161, which is at least 100. The gate opens. The verdict is read from the 0.625 row. The 0.75 row is descriptive: its rate passes but k = 59 is below 100, which would be INCONCLUSIVE under the E06 rule, and it does not change the 0.625 verdict. Kill is not triggered (the rate is not below 3%). The claim 2 readout is not run by this entry.

**Caveats the PASS does not remove.**
1. The 0.625 cutoff was chosen after E06, as the pre-registration says. The gate is therefore a post hoc cutoff applied to the full pool.
2. 72 of the 161 errors come from the 3,000 E06 prompts, which the cutoff choice saw. On the 4,473 prompts E06 did not use, n = 1,541 and k = 89, so the fresh-prompt k is below the floor of 100. The PASS depends on the E06 overlap. It is a PASS under the pre-registered pool (all 7,473 prompts) and should be treated as provisional until L14a.
3. The E06 extrapolation (expected k about 199, rate about 7.3%) overshot: observed k = 161 and rate 6.13%, and the fresh-prompt rate is 5.78%. The 0.625 row was picked after both E06 rows were read, so its 7.35% is probably optimistic (a selection effect).

**What it changes about the thesis claims.**
- Claim 2 (a learned allocator admits fewer corrupted rewards than any fixed rule at equal budget): L13a does not test claim 2. It makes the GSM8K claim 2 readout possible. Current reading: NOT SUPPORTED for the label-free online learner, the one the thesis actually deploys. D(ts_online) at 25% is negative with a CI below 0 on E01, E06 and E10, and the L5c and L10a kills agree. The labeled-learner lead from L11b (static on cheap six plus E01, D +0.035, not pre-registered) remains open and is not evidence for the claim as stated.
- Claim 3 (the edge comes from structure in prompt space): the descriptive numbers on all 7,473 groups meet the E06 claim 3 kill bar (pooled AUC gain -0.0004 against a bar of 0.05; cell100 neighbor ratio 1.00 against a bar of 1.5). This is not a new pre-registered verdict, because the L13a metric list does not include claim 3. Current reading: NOT SUPPORTED, consistent with the E01, E06 and E10 kills.
- The gate opens, so GSM8K is now a possible claim 2 pool. The pool is weaker than the cutoff suggests: its PASS rests on the E06 overlap (caveat 2).

**Process note.** The run report says it ran ls and find over /homes/rdubash/avrl_live, outside CASCADE, to locate files. That reads directory names outside the CASCADE root, which the ground rules do not allow. No file contents were read and nothing was written outside CASCADE. Recorded here as a rule deviation.

**Follow-ups** (also appended to lit/avrl/EXPERIMENT_BACKLOG.md under "Proposed by the loop").
1. L14a: fresh-prompt confirmation of the 0.625 gate on the GSM8K test split (1,319 prompts, about 8 min GPU) pooled with the 4,473 non-E06 train groups already in avrl/runs/L13a. Expected pooled k about 117 (89 plus about 28), so a PASS is likely but thin. Information: high. It tests whether the PASS survives without the prompts that chose the cutoff.
2. L14b: claim 2 readout, D(ts_online) at 25% with bucket-cluster CI, on the gate-passing pool, pre-registered after L14a. CPU, under 15 min. Information: high, and it is the only item that can kill claim 2 on GSM8K. Prior from E01, E06 and E10 points to a kill.
3. L14c: confident-subset claim 3 and truncation cross-tab on the 161 confident errors (not 31 as in L13c). CPU, under 5 min. Information: medium. It reads the +0.029 confident-subset gain and the truncation cross-tab with a real CI.

**Critic note (L13a, 2026-10-19).** Verdict unchanged: PASS (gate opens) at cutoff 0.625, rate 6.13% (161 of 2,628, Wilson 5.27% to 7.11%), rate >= 0.03 and k >= 100. Checked against gate.json, result.json, summary.json, analyze_result.json, readout.log, exit_code, run.log, cmd.sh, wrap.sh, groups.parquet, rollouts.parquet, feats.parquet, exp/L13a.sh (diff against E06.sh is only the header, n_prompts and out dir), exp/L13a.py, p0_rollouts.py, e06_gate.py, e06_analyze.py, and the E06, E01 (runs/p0) and E10 files. Recomputed independently from the raw parquet files: 7,473 unique pids, 8 rollouts each, 59,784 rollouts; agreement and majority recomputed from the rollout answers match groups.parquet exactly; proxy_error equals (majority != gold) for all groups; n and k at 0.625 (2,628 / 161) and 0.75 (1,773 / 59), all Wilson intervals, rollout accuracy 39.68%, majority accuracy 59.79%, 3,005 proxy-error groups, truncated 4,004 (6.697%, equal to tokens >= 512), the overlap rows (1,087 / 72 and 1,541 / 89), prefix and gold identity with E06 for pids 0 to 2999, and the E06, E01 and E10 comparison columns all match the entry. Exit code 0, no traceback, the single ERROR line is the post-completion vLLM shutdown line, as the entry says. No leakage: the proxy is the majority of the 8 sampled answers and agreement uses answers only; gold enters only through proxy_error, the measured quantity; the allocator feature lists (agreement, n_distinct, mean_tokens, std_tokens, any_truncated, log_qlen, 16 PCs of the question embedding) contain no gold, pass_rate or majority_correct. No allocation budget was run, so equal-budget accounting does not apply.

Corrections, none changes the verdict. (1) The table row "AUC on confident subset at 0.75" lists E01 and E10 as "not recomputed". Both are stored and the E06 critic note already corrected this same cell: E01 (runs/p0/result.json) 0.627 / 0.534, gain -0.093 on 17 errors; E10 0.768 / 0.705, gain -0.063 on 35 errors. Both gains are negative, the opposite sign to the L13a +0.029 on 59 errors. (2) "wall time about 33 min" is imprecise: cmd.sh was written at 00:02:50 and done.flag at 00:33:17, so the GPU job took about 30.5 min (1,827 s; generation 1,822 s in run.log, 1,824 s in summary.json). 33 min is reached only by adding the CPU step (result.json at 00:35:33). Under the 45-minute cap either way. (3) The claim 3 paragraph cites "cell100 neighbor ratio 1.00 against a bar of 1.5" as evidence. On this pool that leg carries no information. At base rate 0.40, every one of the 7,473 groups has a proxy-error neighbor in its cell at cell30 and cell100 (share with a hit neighbor 1.000, 0.9992 at cell300), so P(hit | neighbor hit) equals the base rate by construction and the ratio is 1.0 whatever the clustering (the E06, E01 and E10 files also show cell30 exactly 1.0). The ratio cannot reach 1.5 here. Only the AUC-gain leg (-0.0004 against 0.05; top 10% of cell100 cells hold 12.4% of errors) is informative, so "NOT SUPPORTED" for claim 3 rests on the AUC leg alone. It remains a descriptive reading, not a verdict, as the entry says, and any earlier kill that leaned on a neighbor ratio at a high base rate has the same vacuous leg. (4) The claim 2 wording "Current reading: NOT SUPPORTED" is a prior-based description from E01, E06, E10, L5c and L10a. The formal claim 2 status is still NO VERDICT, because the pre-registered readout has not run on the gate-passing pool, and the L13a rules say this item cannot kill claim 2 by itself. (5) The pre-registration lists exp/L13a.py among scripts "written before the run", but the file on disk was last written at 00:35:31, after generation (00:33) and after gate.json (00:35:13). Its rule logic is three branches and matches the pre-registered thresholds, and I recomputed the verdict independently, so there is no effect, but I cannot confirm it existed in this form before the run.

Additions, descriptive, not errors. (a) Truncation sensitivity: at 0.625, 16.1% of confident-error groups contain a truncated rollout against 7.1% of confident non-error groups (2.3 times), but only 11 of the 161 errors have a truncated rollout among the rollouts that make up the majority, and none has 3 or more. Dropping every confident group with any truncated rollout leaves n = 2,426, k = 135, rate 5.56%, still above both floors, so truncation does not create the PASS. (b) Empty extracted answers: 296 of 59,784 rollouts (0.50%) and 40 groups with an empty majority; none of those groups is in the confident subset (checked directly: 0 confident groups with an empty majority), and 99 empty rollouts sit as minority rollouts inside confident groups. (c) The confident-subset gain of +0.029 (59 errors) over 20 shuffled-fold repeats of the same pipeline is mean +0.031, sd 0.023, range -0.036 to +0.069. The sign is mostly positive but not stable, and E01 and E10 are negative, so keep it as L14c material, not a lead. (d) Sample size and seeds: the rate margin is wide (Wilson lower bound 5.27% against a 3% floor; k = 161 against 100; the fresh-prompt slice alone has rate 5.78%, lower bound 4.72%, and only misses the k floor because it has 4,473 prompts). The same prompts regenerated gave k = 72 against E06's 80, so run-to-run movement of about 10 errors is far smaller than the margin of 61. This is one generation with one seed; every prompt uses SamplingParams seed 0, so the groups are not fully independent noise draws, which I did not test and which cannot close a margin of this size. (e) Caveat 2 in the entry stands: the PASS is a PASS under the pre-registered pool of all 7,473 prompts and is provisional until L14a.

Corrected verdict: PASS, unchanged. Gate opens at cutoff 0.625 (6.13%, 161 of 2,628; truncated share 6.70% not flagged; exit 0). The claim 2 readout is not run by this entry; claim 2 stays NO VERDICT until it runs. Scope: Qwen2.5-0.5B-Instruct, all 7,473 GSM8K train prompts, G 8, 512 tokens, one rollout sample, majority-vote proxy, post hoc cutoff, 3,000 prompts shared with E06.

## L14a pre-registration (written before the GPU step, no L14a outcome seen)

Date: 2026-10-19. Written before any L14a script ran, before any file under avrl/runs/L14a existed, and before any L14a output was opened. No L14a outcome has been seen. The GSM8K test split has not been used for any generation in this log before this entry.

**Hypothesis.** On a pool of 4,473 GSM8K train groups that E06 did not use (avrl/runs/L13a, pid 3000 to 7472) plus all 1,319 GSM8K test groups (fresh rollouts, Qwen2.5-0.5B-Instruct, G 8, max_tokens 512, sample seed 0), the confident-subset (agreement at least 0.625) proxy error rate is at least 3% with k at least 100. If so, the L13a gate PASS survives without the prompts that chose the 0.625 cutoff, and the claim 2 readout (L14b) may run on this pool.

**Why this pool.** The L13a PASS is provisional. The 0.625 cutoff was chosen after E06, and 72 of its 161 confident errors come from the E06 prompts that chose it. The 4,473 fresh train prompts alone, at cutoff 0.625, give n 1,541 and k 89 (rate 5.78%, Wilson 95% 4.72% to 7.05%; counts recomputed from avrl/runs/L13a/groups.parquet before this entry). That k is below the floor of 100. The test split adds fresh prompts. Expected pooled k is about 115 (89 plus about 26 from the test split). This is an expectation, not a result.

**Pool, fixed before the run.**
- Train part: the 4,473 groups with pid at least 3000 in avrl/runs/L13a/groups.parquet. Re-checked before this entry: pid 0 to 2999 match avrl/runs/E06/groups.parquet question for question in order, and none of the 4,473 question texts appears in E06.
- Test part: all 1,319 rows of avrl/data/gsm8k_test.parquet (columns question and answer). No question text is shared with the train parquet.
- Rows are tagged split = "train" or split = "test". The pooled join keys on (split, pid), so test pid 0 to 1318 cannot collide with train pid 3000 to 7472.
- Pooled n and k are counted over the union of the two parts at cutoff 0.625.

**Cutoff.** 0.625, fixed. The cutoff is not re-chosen by this run. The 0.75 row is descriptive only.

**Policy.** Qwen2.5-0.5B-Instruct from /homes/rdubash/active-audit/models (read-only), the same policy as L13a. Majority-vote proxy against gold, as in avrl/p0_rollouts.py.

**Scripts and paths (written before the run).**
- avrl/exp/l14a_rollouts_test.py: copy of avrl/p0_rollouts.py. p0 is unchanged. Changes: (1) the parquet is avrl/data/gsm8k_test.parquet; (2) rows are tagged split = "test"; (3) ROOT is resolved three directories up instead of two, because the copy sits one level deeper, in avrl/exp/. Without change (3) the default data path would be avrl/avrl/data. The sampling line is unchanged: sample(n = 1,319, random_state = 0) on the 1,319 test rows is a full permutation, so the sample is the full split. The default values of the other arguments are unchanged.
- avrl/exp/L14a.sh: copy of avrl/exp/L13a.sh pointing at avrl/exp/l14a_rollouts_test.py with --n_prompts 1319 --G 8 --max_tokens 512 --seed 0 --out avrl/runs/L14a.
- GPU step: bin/gpu_run.sh with run dir avrl/runs/L14a, started only after bin/gpu_status.sh shows under 2 GB used. One GPU job only. Polled in band until done.flag. Hard cap 45 minutes, enforced by the helper (timeout 2700 s, exit 124 recorded as failed).
- CPU step: avrl/exp/L14a.py, run with /homes/rdubash/active-audit/.venv/bin/python. It reads avrl/runs/L13a (pid 3000 and above) and avrl/runs/L14a, applies the rules below, and writes avrl/runs/L14a/result.json. It does not run the claim 2 readout.

**Metrics.**
- Primary (gate): pooled confident-subset proxy error count k, group count n, rate k/n with Wilson 95% interval, at cutoff 0.625.
- Truncated-rollout share over the pooled rollouts, flagged if above 0.10. A flag is reported and does not change the verdict.
- Descriptive only: the 0.75 row, each part on its own (the L13a train pool and the test split), rollout accuracy, majority accuracy.

**Rules (fixed before the run).**
- PASS: pooled rate at least 0.03 AND pooled k at least 100. The gate opens for L14b, the claim 2 readout on this pool, which is pre-registered as its own item.
- KILL: pooled rate below 0.03. The GSM8K gate then fails for the 0.5B policy on this pool, and the next step is a task change.
- INCONCLUSIVE: pooled rate at least 0.03 but pooled k below 100. Power-limited, descriptive only.
- This entry cannot kill claim 2 by itself. It only decides whether L14b can run on this pool.

**Failures.** A non-zero exit code, a traceback in run.log, or a 45-minute timeout (exit 124) is recorded as a failed experiment with the exact error text. One fix and one rerun are allowed after a crash. A second failure returns ok = false.

**Expected cost.** About 8 GPU minutes (1,319 prompts at the L13a generation rate of about 0.24 s per prompt, plus vLLM startup), well under the 45-minute cap. About 2 CPU minutes for the readout.

## L14a 2026-10-19 result

**Hypothesis.** As pre-registered in the "L14a pre-registration" entry above: pooled over the 4,473 non-E06 L13a train groups (pid 3000 to 7472) and the 1,319 GSM8K test groups, the Qwen2.5-0.5B-Instruct confident-subset (agreement at least 0.625) proxy error rate is at least 3% with k at least 100.

**What was run.** avrl/exp/L14a.sh through bin/gpu_run.sh (run dir avrl/runs/L14a), after bin/gpu_status.sh showed 15 MiB used on pine both before the pre-registration step and immediately before launch. The GPU script is avrl/exp/l14a_rollouts_test.py, a copy of avrl/p0_rollouts.py with the three changes listed in the pre-registration (ROOT three levels up, test parquet, split tag). p0 is unchanged. Policy Qwen2.5-0.5B-Instruct (shared, read-only), vLLM, G 8, max_tokens 512, seed 0, 1,319 prompts, 10,552 rollouts. Generation 345 s; the whole job took about 6 min including vLLM startup and shutdown. Exit code 0. Zero tracebacks in run.log. The one ERROR line is the post-completion vLLM shutdown message ("Engine core proc EngineCore died unexpectedly" after "Shutdown complete"), the same as in L13a. CPU readout: avrl/exp/L14a.py with /homes/rdubash/active-audit/.venv/bin/python, which wrote avrl/runs/L14a/result.json. The claim 2 readout was not run.

**Numbers.** Confident subset at cutoff 0.625. Wilson 95% in brackets.

| quantity | pooled (L13a pid 3000+ and L14a test) | train part (L13a pid 3000+) | test part (L14a) |
|---|---|---|---|
| n (agreement at least 0.625) | 1,869 | 1,541 | 328 |
| k (proxy errors) | 118 | 89 | 29 |
| rate k/n | 6.31% [5.30%, 7.51%] | 5.78% [4.72%, 7.05%] | 8.84% [6.23%, 12.41%] |

Descriptive 0.75 row, pooled: n 1,267, k 44, rate 3.47% [2.60%, 4.63%].
Truncated rollouts, pooled: 3,267 of 46,336 (7.05%), not flagged (threshold 0.10). Truncated share inside the confident subset: 1.09%.
Rollout accuracy pooled 37.07%, majority accuracy pooled 55.96%. Test part alone: rollout accuracy 29.92%, majority accuracy 46.47%, unanimous share 2.96%, empty extracted answers 49 of 10,552 rollouts (0.46%, a property of the unchanged p0 extractor).

**Verdict: PASS** at cutoff 0.625 (pooled rate 6.31% at least 0.03, pooled k 118 at least 100). The gate opens for L14b on this pool. L14b is a separate pre-registered item and still needs its own pre-registration entry before it runs.

**Reading.** The PASS does not depend on the E06 prompts that chose the cutoff: the pool contains none of them (checked in L14a.py). Both parts clear the rate floor on their own. The test part alone has k 29, below the k floor, which is expected at 1,319 prompts and not a kill signal. Pooled k is 18 above the floor. Across regenerations of the same prompts, L13a moved k by about 10 (72 against E06's 80), so the k margin is about twice that movement: real, but not large. The rate margin is wide (lower Wilson bound 5.30% against 3%). The 0.625 cutoff itself is still post hoc (chosen after E06). L14a checks the pool, not the cutoff.

**Expected against realized.** Expected test k about 26 (realized 29). Expected pooled k about 115 (realized 118).

**Claim status.** Claim 2 stays NO VERDICT: the readout has not run. Claim 3 is untouched by this entry.

Critic note (L14a, 2026-10-19). Written by the run agent after the readout. It is a self-check, not an independent review.
1. All 1,319 test groups were recomputed from the raw rollouts and the gold answers (matched by question text, because p0 draws the prompts with sample, a permutation, so pid is not the parquet row index). Majority, agreement, proxy error and gold match for 1,319 of 1,319. The first attempt matched gold by parquet row and showed 618 mismatches; that was an error in the check, not in the run, and the corrected check is the one recorded here.
2. The pooled counts reconcile: n 1,541 plus 328 is 1,869, and k 89 plus 29 is 118. The train part (1,541, 89) equals the pre-registered cell, recomputed from the L13a groups before this run.
3. No question text is shared between the test part and the train part, or between the train part and E06 (checked in L14a.py). The pid keys are unique on (split, pid).
4. All integrity checks in result.json are true: group and rollout counts, test pids complete, test questions equal the parquet, exit code 0, zero tracebacks.
5. The test split is a different population for this policy. Its majority accuracy is 46.5%, against 59.8% on all L13a train prompts, and only 24.9% of its groups are confident, against 34.4% in the train part. The test part's error rate is higher (8.8% against 5.8%), and both parts clear the 3% rate floor on their own, so the pooled rate does not hide a failing part; the test part is harder and less often confident. Read the 6.3% pooled rate as a rate over this mix of prompts, not as a claim about GSM8K in general.
6. The 0.75 row is descriptive and would be INCONCLUSIVE if it were the gate (3.47%, k 44). It is not used.
7. Truncation is not the driver: 7.05% of all rollouts are truncated, but only 1.09% in the confident subset. The truncated flag (above 0.10) is not raised.
8. One generation with one seed (every prompt uses SamplingParams seed 0), the same caveat as L13a. Not tested here.
9. Verdict unchanged by any check: PASS at cutoff 0.625. The result supports only the gate: a claim 2 readout on this pool is now allowed, and it is the next item, not this one.

## L14a 2026-10-19 pass

This entry is the independent check of the run recorded in the "L14a 2026-10-19 result" entry above. The numbers below were recomputed on CPU from the avrl/runs/L13a and avrl/runs/L14a parquet files, not copied from result.json. They match.

**Hypothesis.** As pre-registered in the "L14a pre-registration" entry: on the pool of 4,473 non-E06 L13a train groups (pid 3000 to 7472) plus the 1,319 GSM8K test groups, the Qwen2.5-0.5B-Instruct confident-subset (agreement at least 0.625) proxy error rate is at least 3% with k at least 100.

**What was run.** Policy: Qwen2.5-0.5B-Instruct (read-only, active-audit/models), vLLM, bf16, temperature 1.0, G 8, max_tokens 512, sample seed 0. Fresh generation on the 1,319 GSM8K test prompts (10,552 rollouts, one GPU job through bin/gpu_run.sh, exit 0, generation 345 s, no traceback). The train part is reused from avrl/runs/L13a (35,784 rollouts, not regenerated). Script: avrl/exp/l14a_rollouts_test.py, which differs from avrl/p0_rollouts.py only in ROOT, the test parquet path and the split tag (checked with diff). Readout: avrl/exp/L14a.py, CPU. Budget: one sampling seed, about 8 GPU minutes. The claim 2 readout (L14b) was not run.

**Numbers.** Confident subset (agreement at least 0.625 unless stated). Wilson 95% in brackets.

| quantity | pooled (gate) | train part (L13a pid 3000+) | test part (L14a) | L13a, all 7,473 train (baseline) |
|---|---|---|---|---|
| n | 1,869 | 1,541 | 328 | 2,628 |
| k | 118 | 89 | 29 | 161 |
| rate | 6.31% [5.30%, 7.51%] | 5.78% [4.72%, 7.05%] | 8.84% [6.23%, 12.41%] | 6.13% [5.27%, 7.11%] |
| confident share of groups | 32.3% | 34.4% | 24.9% | 35.2% |
| rollout accuracy | 37.1% | 39.2% | 29.9% | 39.7% |
| majority-vote accuracy | 56.0% | 58.8% | 46.5% | 59.8% |
| truncated rollouts, all (flag above 10%) | 7.05% (3,267 of 46,336) | 6.70% (2,398 of 35,784) | 8.24% (869 of 10,552) | n/a |
| truncated rollouts, confident subset | 1.09% | n/a | n/a | n/a |

Descriptive 0.75 row (not the gate): pooled n 1,267, k 44, rate 3.47% [2.60%, 4.63%]. Train part 1,056 / 32 / 3.03%. Test part 211 / 12 / 5.69%.

Other policies at the same 0.625 cutoff, from the L13a table (different policies and prompts, context only): Qwen2.5-1.5B (E01) 1.94% (38 of 1,960), OLMo-2 1B (E10) 2.85% (68 of 2,384). The 3% gate is policy-dependent.

Checks recomputed by the checking agent (all true): the train part gives n 1,541 and k 89, equal to the pre-registered cell; pooled 1,869 = 1,541 + 328 and 118 = 89 + 29; agreement equals the top vote count over 8 for every group in both parts; proxy_error equals (majority != gold) for every group; gold recomputed from the test parquet matches for all 1,319 prompts; the proxy label of all 10,552 test rollouts matches its group majority; no question text is shared between the train pool, the test split and E06; the truncated flag equals tokens at least 512 for every rollout.

**Verdict: PASS at cutoff 0.625.** The pooled rate 6.31% is at least 0.03 (its Wilson lower bound, 5.30%, is also above it), and pooled k 118 is at least 100. The KILL and INCONCLUSIVE branches do not apply. The truncation flag is not raised. The 0.75 row is descriptive only. It would have been INCONCLUSIVE (rate 3.47% passes, k 44 fails), and its Wilson lower bound (2.60%) is below 3%, so the gate is sensitive to the cutoff choice.

**Limits on the pass.**
1. The k margin is thin. The train part alone gives k 89, below the floor, and the pooled k clears it by 18. Regenerating the same prompts moved k by about 10 in L13a (72 against E06's 80), so the margin is about twice the observed regeneration movement.
2. The test split differs from the train part in accuracy, not in visible prompt surface. Majority accuracy is 46.5% against 58.8%, and rollout accuracy 29.9% against 39.2%. Question length (240 against 236 characters) and solution lines (3.66 against 3.58 on average) are almost identical, so prompt surface does not explain the gap. One candidate is that the 0.5B policy saw the GSM8K train split in pretraining, which would inflate train accuracy. Nothing in these files tests that. Until it is tested, read the pooled 6.31% as a rate over this mix of prompts.
3. The 0.625 cutoff is still post hoc (chosen after E06). L14a checks the pool, not the cutoff.

**Claim status.** Claim 2 (a learned allocator admits fewer corrupted rewards than any fixed rule at equal budget): L14a does not test it, so the reading does not change. It stays NOT SUPPORTED for the label-free online learner (D(ts_online) at 25% is negative with CIs below 0 on E01, E06 and E10). The labeled-learner lead from L11b remains open. The L14b readout is now allowed on this pool and needs its own pre-registration entry before it runs. Claim 3 (structure in prompt space): not tested here. The reading stays NOT SUPPORTED (L13a descriptive AUC gain -0.0004 against the 0.05 bar; cell100 neighbor ratio 1.00 against 1.5).

**Follow-up experiments** (also under "Proposed by the loop" in lit/avrl/EXPERIMENT_BACKLOG.md as L15a to L15c).
1. L15a, policy versus prompts for the train-test accuracy gap (GPU, one job, about 6 min). Run OLMo-2-0425-1B-Instruct on the 1,319 test prompts with the E10 protocol and compare its test majority accuracy with its E10 train accuracy. If OLMo's gap is 3 points or less while Qwen's is 12.3, the gap is policy-specific and the pooled rate is a mixture. If OLMo's gap is 8 points or more, the test split is harder for both policies. Information: high for how the pool is read, low for the claim verdicts.
2. L15b, sampling-seed replicate on the test split (GPU, one job, about 6 min, seed 1, otherwise identical). PASS on k if the seed-1 test k is at least 11, so that the train part (89) plus the replicate reaches 100. Information: medium. A kill is unlikely (rate lower bound 5.30% against 3%), but k is the weakest part of the gate.
3. L15c, claim 3 check on the fresh pool (CPU, descriptive). Rerun the L14c arms on the 1,869 pooled groups (118 errors, none from E06) instead of the 161 L13a errors, which include 72 E06 prompts, with the test groups' features computed as for L13a. Information: medium. The prior is a kill, so this mainly replicates the L13a descriptive kill on prompts that chose nothing.

Critic note (L14a, 2026-10-19, independent adversarial check of the "L14a 2026-10-19 result" and "L14a 2026-10-19 pass" entries). Verdict unchanged: PASS at cutoff 0.625, pooled rate 6.31% (118 of 1,869, Wilson 5.30% to 7.51%), rate >= 0.03 and k >= 100. Checked against runs/L14a/{groups,rollouts}.parquet, result.json, summary.json, run.log, exit_code, done.flag, cmd.sh, wrap.sh, runs/L13a/{groups,rollouts}.parquet, runs/E06, runs/p0, runs/E10 result and groups files, data/gsm8k_{train,test}.parquet, exp/L14a.py, exp/L14a.sh, exp/l14a_rollouts_test.py (diff against p0_rollouts.py, which is read-only and last modified 2026-09-24), exp/e06_gate.py. Recomputed independently from the raw parquet files: 1,319 test groups with 8 rollouts each (10,552) and 4,473 train-pool groups with 8 each (35,784); majority and agreement recomputed from the rollout answers match groups.parquet for every group in both parts; gold recomputed from the parquet answer column and matched by question text equals the stored gold for 1,319 of 1,319 test and 4,473 of 4,473 train-pool groups; proxy_error equals (majority != gold) everywhere; the rollout proxy, correct and truncated (tokens >= 512) columns agree with their definitions. Cells match the entries exactly: pooled 1,869 / 118 / 6.31%, train part 1,541 / 89 / 5.78%, test part 328 / 29 / 8.84%, 0.75 row pooled 1,267 / 44 / 3.47% (train 1,056 / 32, test 211 / 12), L13a all-train 2,628 / 161 / 6.13%, E01 1,960 / 38, E10 2,384 / 68, E06 1,089 / 80, all Wilson bounds, pooled rollout accuracy 37.07%, majority accuracy 55.96%, truncated 3,267 of 46,336 (7.05%; train 2,398, test 869; confident subset 1.09%), 49 empty test answers, zero confident groups with an empty majority, test unanimous share 2.96%. No question appears twice in the pool, in both train and test parquets, or in both the pool and E06. Exit code 0, no traceback, the one ERROR line is the post-completion vLLM shutdown line, no warnings beyond two FutureWarnings. No leakage: the proxy is the majority of the 8 sampled answers (the chat prompt holds only the system line and the question), gold enters only through proxy_error, the measured quantity, and no allocator or features were run. Budget accounting is identical across the two parts (G 8, max_tokens 512, temperature 1.0, seed 0, same policy), and no allocation budget was run. L14a.py (written 00:52:48) and the GPU script (00:52:17) pre-date the GPU outputs (00:58), and its rule logic matches the pre-registered thresholds.

Corrections, none changes the verdict. (1) The pass entry's claim 2 line "It stays NOT SUPPORTED for the label-free online learner" is wrong as a formal status and contradicts the "result" entry above it ("Claim 2 stays NO VERDICT"), the E06 rule ("In any other gate state the claim 2 readout is descriptive and no claim 2 verdict is issued"), the E06 "Thesis claims after E06" line ("untested on a gate-passing pool"), and the L13a critic note item (4), which already corrected this exact wording. The D(ts_online) figures it cites are accurate (E01 -0.172 [-0.230, -0.142], E06 -0.124 [-0.141, -0.100], E10 -0.203 [-0.260, -0.175]) but E06 was gate INCONCLUSIVE and E10 was GATE_FAIL, so they are descriptive. The formal status is NO VERDICT until L14b runs on this pool. (2) The pass entry's claim 3 line leans on "cell100 neighbor ratio 1.00 against 1.5". The L13a critic note item (3) showed that leg is vacuous at a 0.40 base rate (every group has a proxy-error neighbor, so the ratio is 1.0 by construction), and the figures are from L13a (7,473 groups, 3,000 of them E06), not from the L14a pool. Only the AUC leg (-0.0004 against 0.05) carries information, and it is a descriptive reading, not a verdict; L14a does not test claim 3. (3) The margin is described more comfortably than the files support. The E06 to L13a regeneration of the same 3,000 prompts moved confident-error k from 80 to 72, a movement of 8, not "about 10", and it was a real resample (62 groups discordant: 45 confident errors in both, 27 only in L13a, 35 only in E06; the majority answer changed for 31.5% of groups). Scaling that noise (sd of one k about 5.6 at 3,000 prompts) to the 5,792 pooled prompts gives an sd of about 7.7 for pooled k, and a bootstrap over the 5,792 prompts gives sd 10.8 with P(k < 100) = 3.7% and P(rate < 0.03) = 0. So the k margin of 18 is about 1.7 to 2.3 sd: real but thin, as the entry says, and the verdict is a point rule on the realized pool. This is one seed (every prompt uses seed 0), untested here, and L15b is the right replicate. (4) What is fresh is narrower than "fresh-prompt confirmation" suggests. The train part (n 1,541, k 89) was computed and written into the pre-registration before the run, so the pooled PASS required only test k >= 11 (realized 29, expected 26). The test split alone, the only data no one had seen, has k 29 and fails the k floor by itself; its rate (8.84%, lower bound 6.23%) is the informative fresh evidence for the rate floor. The PASS is a valid PASS under the pre-registered pool and says little about the cutoff, which stays post hoc. (5) Small items: "generation 345 s" is the script wall time from before the model load (run.log: model init 00:53:11 to 00:53:22, shutdown 00:58:42), so generation itself is about 320 s; the whole job was 5.8 min (cmd.sh 00:52:55, done.flag 00:58:43). Solution lines per example are 3.66 (test) against 3.57 on the 4,473 train-pool prompts, not 3.58, which is the mean over all 7,473 train prompts (3.576); immaterial to the reading. The test-versus-train accuracy gap (majority 46.5% against 58.8%, rollout 29.9% against 39.2%) is as stated, and the pretraining-exposure explanation remains untested.

Corrected verdict: PASS, unchanged. Gate opens at cutoff 0.625 on the pool of 4,473 non-E06 L13a train groups plus 1,319 test groups (pooled 6.31%, 118 of 1,869; Wilson lower bound 5.30%; truncated share 7.05% not flagged; exit 0). Claim 2: NO VERDICT, not NOT SUPPORTED; L14b needs its own pre-registration before it runs on this pool. Claim 3: not tested by L14a. Scope: Qwen2.5-0.5B-Instruct, GSM8K, G 8, 512 tokens, one rollout sample at seed 0, majority-vote proxy, post hoc cutoff, test part alone below the k floor.

## L14b pre-registration 2026-10-19 (written before any L14b script ran or any L14b output was computed)

**Question.** Claim 2 on the gate-passing GSM8K pool opened by L14a: does the label-free online learner ts_online (cell100 buckets, cheap+emb features) admit fewer corrupted rewards than the label-free agreement threshold at a 25% budget? This is the first formal claim 2 readout on a gate-passing pool. The formal claim 2 status stays NO VERDICT until this item runs, and it changes only under the rules below.

**Pool.** 5,792 groups, Qwen2.5-0.5B-Instruct, G 8, max_tokens 512, sample seed 0. Train part: avrl/runs/L13a/feats.parquet rows with pid >= 3000 (4,473 groups), kept unchanged. Test part: all 1,319 GSM8K test groups from avrl/runs/L14a/groups.parquet and rollouts.parquet. Row order is the train part in pid order, then the test part in pid order. The seed-0 bucket split of e03 depends on row order, so this order is fixed. The replay runs on all 5,792 groups, not only the 0.625 confident subset, as E01, E06 and E10 did.

**Features.**
- Six cheap columns (agreement, n_distinct, mean_tokens, std_tokens, any_truncated, log_qlen). Train rows keep their L13a values. Test rows are recomputed from avrl/runs/L14a/rollouts.parquet with the formulas that produced the L13a columns: answer is the most frequent normalized answer (ties go to the first in sorted order), n_distinct is the number of distinct normalized answers, mean_tokens and std_tokens are the mean and population std of the rollout token counts, any_truncated is 1 if the longest rollout has at least 512 tokens, and log_qlen is log1p of the question length in characters. Gate: the recomputation must reproduce the stored L13a columns for pid 0 to 7472 and the stored L14a columns for the 1,319 test groups (integers exact, floats within 1e-9). Otherwise FAILED.
- cell100 and pc0 to pc15. Refit as avrl/exp/e06_analyze.py does: all-MiniLM-L6-v2 on CPU (batch 256, normalized embeddings), KMeans(100, n_init=3, random_state=0) fit on the 7,473 L13a train questions in pid order, PCA(16, random_state=0) fit_transform on the same embeddings. Test questions get KMeans.predict and PCA.transform. Gate: the refit must reproduce L13a cell100 exactly and pc0 to pc15 within 1e-4 for pid 0 to 7472. If it does not, record FAILED with the mismatch and stop. No readout runs.

**Replay.** avrl/exp/l14b_claim2_readout.py is a copy of avrl/exp/e03_threshold_bootstrap.py (e03 itself is not edited). Two changes: the primary interval is the bucket-cluster bootstrap, and the E01 reproduction block is removed because its reference pool is not this pool. Unchanged: buckets with fewer than 3 groups are dropped, and the seed-0 fit split (first quarter of the shuffled cell100 buckets) is held out and used only by the static reference arm. So the replayed sim pool is the remaining about 75% of the 5,792 groups. Its size, bucket count and total proxy errors are reported; the 5,792 is the input pool. Arms: uniform, static (labeled reference), threshold (label-free, seeded random tie-break), threshold_input_order (sensitivity), greedy_online, ts_online. Features cheap+emb (22 columns). Budgets 10%, 25% and 50% of the sim pool. Allocator seeds 0 to 7.

**Metric.** admitted(arm) = 1 - caught(arm) / total, where caught is the seed mean (deterministic arms have one run) and total is the number of proxy errors in the sim pool. The proxy label is proxy_error (majority answer differs from gold). The threshold arm reads agreement and n_distinct only. ts_online sees proxy_error only for the groups it audits, as in E03. Gold enters only through proxy_error. D(ts_online) at 25% = admitted(threshold) - admitted(ts_online). Positive D means ts_online admits fewer corrupted rewards.

**Intervals.** Primary: 95% percentile interval of D(ts_online) at 25% over 200 bucket-cluster resamples (sim-pool buckets drawn with replacement, seed 123, each repeated bucket relabeled as a new cluster, as in avrl/exp/l5a_learned_class.py). Secondary: 95% percentile interval over 200 group resamples (sim groups with replacement, BOOT_SEED 0, the E03 draws). D and caught counts at 10% and 50% are descriptive.

**Verdict rules** (on D(ts_online) at 25%, cell100, cheap+emb).
- PASS: point D > 0 AND the lower bound of the primary bucket-cluster interval is > 0.
- KILL: point D <= 0. This kills claim 2 on this pool for this label-free learner and feature set only. It does not apply to the labeled static learner, greedy_online, or other policies.
- INCONCLUSIVE: point D > 0 with the primary interval including 0.
- The secondary group interval is reported and cannot change the verdict.

**Code check (before the readout).** The copy, run on avrl/runs/p0/feats.parquet (the E03 pool) with the E03 arguments, must reproduce avrl/runs/E03/result.json: caught means and admitted values for every arm and budget, and the 200 group-bootstrap draws of D(ts_online) and D(greedy_online), to 1e-9. If it does not, fix the copy and rerun the check, not the readout. Output goes to avrl/runs/L14b/code_check_E03/.

**Other checks.** Pool: 5,792 rows, 4,473 train and 1,319 test, unique (split, pid), no NaN in the model columns, cell100 in 0 to 99, zero question overlap between train part and test part. Exit code 0 and no traceback in the logs.

**Prior.** D(ts_online) at 25% is negative with intervals below 0 on E01 (-0.172 [-0.230, -0.142], E03), E06 (-0.124 [-0.141, -0.100]) and E10 (-0.203 [-0.260, -0.175]). A KILL is the likely outcome.

**Outputs.** avrl/runs/L14b/: pool_feats.parquet, refit_gate.json, result.json (metric, both intervals, budget D points, caught counts with sd, pool and sim-pool counts, gate and code-check results), readout.log, done.flag.

**Scope.** Qwen2.5-0.5B-Instruct, GSM8K, G 8, one rollout sample at seed 0, majority-vote proxy, cell100 from a MiniLM refit, 22 features, one allocator design. Claim 3 is not tested here.

**L14b pre-registration amendment 2026-10-19 (before the second refit run; no readout computed).** The first refit run (avrl/runs/L14b/refit_run1.log) passed the pre-registered refit gate exactly: cell100 0 mismatches over pid 0 to 7472 and max abs pc0 to pc15 difference 0.0, and the cheap-column recomputation matched the stored L13a and L14a columns. It failed one integrity check that the script added beyond the pre-registration: that the L14a pid order equals the order of gsm8k_test.parquet. The check was wrong. avrl/exp/l14a_rollouts_test.py builds pid order with test.sample(n=1319, random_state=0), a seed-0 permutation, so that order is by design. The check is replaced with three checks: the L14a pid order equals that exact sample, the test questions are unique and set-equal to the test split, and the gold values recompute with the L14a method. The refit, the cheap-column definitions, the pool, the replay, the metric and the verdict rules are unchanged.

## L14b 2026-10-19 kill (claim 2: label-free ts_online on the L14a pool at 25%)

**Hypothesis.** As pre-registered in the "L14b pre-registration" entry above: on the 5,792-group L14a gate-passing pool, ts_online (cell100, cheap+emb) admits fewer corrupted rewards than the label-free threshold at 25%, so D(ts_online) > 0 with the primary bucket-cluster interval above 0.

**What was run.** (1) avrl/exp/l14b_feature_refit.py on CPU. The refit gate passed: cell100 had 0 mismatches over pid 0 to 7472, the largest abs pc0 to pc15 difference was 0.0, and the cheap-column recomputation was exact on the 7,473 L13a groups and the 1,319 L14a test groups. Pool: avrl/runs/L14b/pool_feats.parquet (4,473 train, 1,319 test). (2) avrl/exp/l14b_code_check.py: the copied readout reproduced avrl/runs/E03/result.json exactly on the E03 pool (every caught mean, admitted value, D point and the 200 group-bootstrap draws; max abs diff 0.0; both KILL). The new bucket-cluster resampler reproduced the sim pool exactly with identity draws. (3) avrl/exp/l14b_claim2_readout.py on the pool: exit 0, no traceback, 60 s. Output avrl/runs/L14b/result.json and readout.log.

**Result.** Sim pool: 4,318 groups in 75 buckets (the seed-0 fit quarter, 25 buckets, held out as in E03), with 1,906 proxy errors (rate 0.4414). At 25% (1,079 docs, 8 seeds):

| arm | caught (seed mean, sd) | admitted |
|---|---|---|
| uniform | 473.9 (16.1) | 0.751 |
| static (labeled reference) | 938.0 (deterministic) | 0.508 |
| threshold (label-free) | 916.6 (7.3) | 0.519 |
| greedy_online | 894.8 (7.9) | 0.531 |
| ts_online | 770.5 (9.4) | 0.596 |

D(ts_online) = -0.0767 at 25%. Primary bucket-cluster 95% interval [-0.0958, -0.0651] (200 draws, seed 123, 0 of 200 above 0). Secondary group interval [-0.1005, -0.0738] (BOOT_SEED 0, 0 of 200 above 0). Descriptive only: D(ts_online) = -0.0495 at 10% and -0.0932 at 50%; D(greedy_online) = -0.0115 at 25% with cluster interval [-0.0225, -0.0034] and group interval [-0.0209, -0.0047].

**Verdict against the pre-registered rules.** Point D(ts_online) at 25% is -0.077, which is at or below 0. KILL, as the pre-registration states.

**Claim status.** Claim 2 on this gate-passing GSM8K pool: KILL for the label-free online learner (ts_online, cheap+emb features, cell100 buckets, 25%). This replaces the NO VERDICT status that L14a left. It does not reach labeled learners or other policies. Descriptive only, not verdicts: the labeled static reference beats the threshold at 25% (938.0 against 916.6 caught; admitted 0.508 against 0.519; not bootstrapped in this readout), and greedy_online also admits more than the threshold. Claim 3: not tested here.

**Checks and caveats.**
1. The first refit run (avrl/runs/L14b/refit_run1.log) failed an integrity check that the script added beyond the pre-registration. The amendment above records the replacement. The pre-registered refit gate passed in the rerun. Run 1 exited at the gate before it computed test features; its train-side refit matched the rerun (0 cell100 mismatches, max abs pc difference 0.0).
2. The replayed sim pool is about 75% of the 5,792 input groups, as in E03. Its size and bucket count are reported.
3. The MiniLM weights were loaded by name through the shared Hugging Face cache, the same loader as E06 and L13a. The run printed an unauthenticated-request warning. The weights loaded and the refit reproduced L13a exactly.
4. Scope: one rollout sample (seed 0), Qwen2.5-0.5B-Instruct, GSM8K, majority-vote proxy, MiniLM refit features, one allocator design, 8 allocator seeds. The verdict uses the point estimate and the primary interval, and the secondary interval agrees in sign.

Outputs: avrl/runs/L14b/{pool_feats.parquet, refit_gate.json, refit.log, refit_run1.log, result.json, readout.log, code_check_E03/, done.flag}.


## L14b 2026-10-19 kill

**Hypothesis.** On the 5,792-group L14a gate-passing GSM8K pool, the label-free online learner ts_online (cell100 buckets, cheap+emb features) admits fewer corrupted rewards than the label-free agreement threshold at a 25% budget: D(ts_online) = admitted(threshold) - admitted(ts_online) > 0, with the primary bucket-cluster interval above 0.

**What was run.** Qwen2.5-0.5B-Instruct, GSM8K, G 8, max_tokens 512, sample seed 0. Input 5,792 groups (4,473 L13a train groups with pid >= 3000, 1,319 test groups). Proxy is majority-vote error (proxy_error); gold is used only for scoring. Replay: avrl/exp/l14b_claim2_readout.py (a copy of e03). Seed-0 fit quarter (25 of 100 cell100 buckets, 1,474 groups) is held out as in E03, so the sim pool is 4,318 groups in 75 buckets with 1,906 proxy errors (base rate 0.4414). Budgets 10/25/50% = 431/1,079/2,159 docs. 8 allocator seeds. Primary interval: 200 bucket-cluster resamples, seed 123. Secondary: 200 group resamples, BOOT_SEED 0. CPU only, no GPU.

**Results.** Admitted = 1 - caught/1,906. Caught is the seed mean (sd) at 25%.

| arm (25%, 1,079 docs) | caught (sd) | admitted | D vs threshold |
|---|---|---|---|
| uniform (label-free, baseline) | 473.9 (16.1) | 0.751 | -0.232 |
| static (labeled reference, fit quarter) | 938.0 (deterministic) | 0.508 | +0.011 |
| threshold (label-free, agreement) | 916.6 (7.3) | 0.519 | 0 |
| greedy_online (label-free) | 894.8 (7.9) | 0.531 | -0.011 |
| ts_online (label-free, cheap+emb) | 770.5 (9.4) | 0.596 | -0.077 |

| D at 25% | point | primary bucket-cluster 95% CI | secondary group 95% CI |
|---|---|---|---|
| ts_online | -0.077 | [-0.096, -0.065] | [-0.101, -0.074] |
| greedy_online | -0.011 | [-0.022, -0.003] | [-0.021, -0.005] |

Descriptive, not verdicts: D(ts_online) = -0.050 at 10% and -0.093 at 50%. The earlier D(ts_online) points at 25% quoted in the pre-registration were -0.172 (E01 pool), -0.124 (E06) and -0.203 (E10), so this pool shows the same sign at a smaller size.

**Verdict.** KILL. Point D(ts_online) at 25% is -0.077, which is at or below 0 under the pre-registered rule. The primary interval lies entirely below 0 (0 of 200 cluster draws above 0). The secondary interval agrees. The verdict covers the label-free ts_online with cheap+emb features on cell100 at 25% only.

**Independent recheck (CPU, nothing written).** The sim pool (4,318 groups, 75 buckets, 1,906 errors), the threshold caught counts (915 in input order; seeded mean 916.625, sd 7.330), and the uniform caught mean (473.875, sd 16.066) were recomputed from pool_feats.parquet and reproduce. The 2.5 and 97.5 percentiles of the 200 stored draws reproduce both intervals. The point D reproduces (-0.07667). No traceback in any log. The E03 code check passed with max abs diff 0.0 (runs/L14b/code_check_E03/compare.json). The MiniLM refit was not recomputed here; it rests on the script's refit gate (0 cell100 mismatches, max abs pc diff 0.0).

**What it changes about the thesis claims.**
- Claim 2 (a learned allocator admits fewer corrupted rewards than any fixed rule at equal budget): not_supported on this pool. The formal KILL covers the label-free online learner only. greedy_online also admits more than the threshold, with a CI below 0. The labeled static arm is the only learned arm below the threshold (+0.011 at 25%), and it has no bootstrap in this readout. The earlier labeled-static readout (L12a) found no detectable edge. Arithmetic bound: perfect labels at 25% admit 1 - 1,079/1,906 = 0.434, so no allocator on this pool can exceed D = +0.085 over the threshold. The labeled static arm captures about 13% of that headroom.
- Claim 3 (the edge comes from structure in prompt space): not_supported. L14b does not test it (no embedding-only arm, no confident-subset AUC). The standing reading from E01, E06, E10, L9a and L13a (pooled AUC gain -0.0004 against a bar of 0.05, neighbor ratio 1.00) is unchanged.

**Follow-ups.**
1. L16a, noisier-proxy replay (CPU, descriptive, under 10 min). Rebuild proxy_error and the cheap columns from a seeded 4-of-8 rollout subsample per group, keep the embedding features, and rerun the readout at 25%. Information: high. It is the only CPU route to the proxy-quality question in PROJECT.md before the GPU G 16 item (L10b). A sign flip would be the first regime where claim 2 can win. Prior: kill is likely.
2. L16b, bucket-oracle ceiling (CPU, descriptive, under 5 min). Rank by the true sim-pool bucket error rate, with the threshold key inside each bucket. Information: medium-high. If even this bucket oracle does not beat the threshold, Thompson over cell100 cannot win on this pool, which closes the bucket design and not just the learner.
3. L16c, the L14b readout at cell30 and cell50 (CPU, under 5 min), to check whether the kill depends on bucket granularity. Information: medium.

Outputs: avrl/runs/L14b/{result.json, readout.log, pool_feats.parquet, refit_gate.json, refit.log, refit_run1.log, code_check_E03/, done.flag}.

**Critic note (L14b, 2026-10-19).** Verdict unchanged: KILL. Point D(ts_online) at 25% is -0.0767, at or below 0, so the pre-registered rule gives KILL; the primary bucket-cluster interval [-0.0958, -0.0651] and the secondary group interval [-0.1005, -0.0738] both lie below 0 (0 of 200 draws above 0 in each). There is no avrl/exp/L14b.py; the scripts are avrl/exp/l14b_feature_refit.py, l14b_code_check.py and l14b_claim2_readout.py. Checked against result.json, readout.log, refit_gate.json, refit.log, refit_run1.log, code_check_E03/{compare.json, run.log}, code_check_stdout.log, pool_feats.parquet, the three scripts (the readout diffed against exp/e03_threshold_bootstrap.py: only the changes the pre-registration lists), the L13a and L14a rollouts and groups parquet files, the E03, E06 and E10 result.json files, and the ledger snapshots in snap/ for timing. Recomputed independently on CPU, nothing written. (a) From the raw rollout answers and token counts, majority, agreement, n_distinct, mean_tokens, std_tokens and any_truncated for all 5,792 pool groups match pool_feats.parquet (max abs diff 0.0, majority identical, 8 rollouts per group); proxy_error equals (majority != gold) for every group (1,845 errors in the 4,473 train rows, 706 in the 1,319 test rows); pool order is train pid 3000 to 7472 ascending then test pid 0 to 1318 ascending; no NaN in the 22 model columns plus the label; no duplicated question in the pool and no train/test question overlap. (b) The seed-0 split gives a fit quarter of 25 buckets and 1,474 groups and a sim pool of 4,318 groups in 75 buckets with 1,906 errors (rate 0.4414; 1,386 train-part and 520 test-part errors). (c) Importing the readout functions, the 25% point values reproduce exactly: uniform 473.875 (sd 16.066), threshold 916.625 (7.330), threshold_input_order 915, greedy_online 894.75 (7.902), ts_online 770.5 (9.381), D(ts_online) -0.076666, D(greedy_online) -0.011477; budgets 431, 1,079 and 2,159; every admitted value and D-versus-threshold figure in the entry's tables. (d) The first two stored cluster draws (-0.068517, -0.089149) and the first two stored group draws (-0.089408, -0.095960) of D(ts_online) recompute exactly from the stored seeds, and the 2.5 and 97.5 percentiles of the 200 stored draws reproduce all four intervals. (e) The prior figures quoted from E03 (-0.172 [-0.230, -0.142]), E06 (-0.124 [-0.141, -0.100]) and E10 (-0.203 [-0.260, -0.175]) match their result.json files. (f) The E03 code check has max abs diff 0.0 over every point value, the 200 group draws and both intervals (compare.json, passed true, both KILL). (g) No leakage: the allocator columns are the six cheap columns and pc0 to pc15, none derived from gold; ts_online and greedy_online read proxy_error only for audited documents (the classifier is fit on X[seen], y[seen]); threshold reads agreement and n_distinct only; the fit-quarter labels enter only the labeled static reference arm; cell100 and the PCs come from question text only. (h) Budget accounting is equal: every arm spends int(0.25 * 4,318) = 1,079 audits, and ts_online and greedy_online pay for their warm-up labels inside that budget. (i) Timing from the snapshots: the pre-registration is in the log by 01:15:54, before refit run 1 (refit_run1.log 01:18:35); the amendment is in by 01:19:24, before refit run 2 finished (01:19:58); the readout script was written at 01:21:37 and result.json at 01:23:37, so no readout existed before the amendment. (j) No traceback, error text or NaN in any L14b log. (k) Seed adequacy: the allocator-seed standard error of D is about 0.002 (sd 7.3 and 9.4 caught over 8 seeds, divided by 1,906), so the margin of -0.077 is about 35 standard errors and the bucket-cluster interval is far from 0; the tie-break variant threshold_input_order gives D -0.0758, the same sign.

Corrections, none changes the verdict. (1) The claim 2 bullet says "The earlier labeled-static readout (L12a) found no detectable edge." L12a was never run. It exists only as a pre-registered backlog item (lit/avrl/EXPERIMENT_BACKLOG.md, filed from the L11b follow-ups); there is no avrl/runs/L12a and no L12a ledger entry. The labeled evidence in the log is L11b on the E10 pool (2,180 sim docs): the CEILING static arm (cheap six + 16 PCs + E01) D +0.0083, CI [-0.0167, +0.0347], INCONCLUSIVE, and a post hoc best-of-five static row (cheap six + E01) D +0.0354, CI [+0.0112, +0.0582], which is a lead and not a result. The L14b static row (+0.011 at 25%, 938.0 against 916.6 caught) is a single deterministic fit with no interval and no E01 feature. So the labeled-learner route is untested, not shown to have no edge, and the KILL here does not touch it. (2) The claim 3 bullet again cites "neighbor ratio 1.00" as part of the standing reading. The L13a critic note (item 3) and the L14a critic note (item 2) already showed that leg is vacuous at a high base rate (every group has an error neighbor in its cell, so the ratio is 1.0 whatever the clustering); at this pool's 0.44 base rate it carries no information either. Only the AUC leg is informative, and L14b does not test claim 3 at all, so the line should read "not tested here; the descriptive AUC reading from L13a is unchanged". (3) Item (2) of "What was run" in the first entry says "The new bucket-cluster resampler reproduced the sim pool exactly with identity draws." No file records that check: it is not in l14b_code_check.py, compare.json or any log. I ran it now (identity draws give identical X, y and agreement arrays and an isomorphic bucket coding), so the statement is true but was unrecorded. The code check validates the point replay and the group draws against E03; the cluster draws are validated only by this recheck and by the recomputed draws in (d). (4) Caveat 1 says run 1 "exited at the gate before it computed test features". Run 1 did recompute the test cheap columns (refit_run1.log shows cheap_recompute_matches_L14a true and zero mismatches); what it had not yet produced is the test cell100 and PCs and the pool. The amendment text is accurate; only that sentence is loose. (5) The ledger now holds two sections for this result, "L14b 2026-10-19 kill (claim 2: label-free ts_online on the L14a pool at 25%)" and "L14b 2026-10-19 kill". They agree on every number; (1) and (2) refer to the second, (3) and (4) to the first.

Additions, descriptive, not errors. (a) The readout runs on all 5,792 groups (sim base rate 0.441), as pre-registered and as E01, E06 and E10 did, while the L14a gate that opened it is about the confident subset (agreement at least 0.625, proxy error 6.31%). The KILL is about auditing across the whole pool, where the cheap threshold is already strong: at 25% it catches 916.6 of 1,079 audits (precision 0.850), ts_online 770.5 (0.714), against a base rate of 0.441. It says nothing about finding the confident errors specifically. (b) Headroom: perfect labels would admit 0.434, so the largest possible D over the threshold on this pool is +0.085, and ts_online sits 0.077 below the threshold on a 0.085 scale. (c) The sign holds at every budget (D(ts_online) -0.050 at 10%, -0.077 at 25%, -0.093 at 50%), and greedy_online is within about 0.0003 of the threshold at 50%. (d) One rollout sample at seed 0 and one realization of the pool: the cluster bootstrap resamples buckets, not rollouts, so it does not cover rollout noise; the margin here is far larger than the E06-to-L13a regeneration movement logged earlier. (e) The MiniLM weights come from the shared Hugging Face cache by name (same loader as e06_analyze.py), which the entry discloses; the test-side cell100 and PCs have no stored reference to gate against, but they are built from g14.question in pid order with the same fitted KMeans and PCA, and the 7,473-row train refit matched exactly.

Corrected verdict: KILL, unchanged. Claim 2 (a learned allocator admits fewer corrupted rewards than any fixed rule at equal budget): KILL for the label-free online learner ts_online with cheap+emb features on cell100 buckets at a 25% budget, on the 5,792-group L14a gate-passing pool (sim pool 4,318 groups, 75 buckets, 1,906 errors; point D -0.0767, bucket-cluster 95% CI [-0.0958, -0.0651], group CI [-0.1005, -0.0738]). Formal status moves from NO VERDICT to KILL for that learner only. Labeled learners and other policies are untested here, and L12a has not been run. Claim 3: not tested by L14b. Scope: Qwen2.5-0.5B-Instruct, GSM8K, G 8, 512 tokens, one rollout sample at seed 0, majority-vote proxy, MiniLM refit features, one allocator design, 8 allocator seeds.

## L16a 2026-10-19 kill

**Hypothesis.** On the 5,792-group L14a gate-passing GSM8K pool, when the majority proxy is rebuilt from 4 of the 8 rollouts per group (a noisier proxy), the label-free ts_online allocator admits fewer corrupted rewards than the label-free agreement threshold at a 25% budget: point D(ts_online) = admitted(threshold) minus admitted(ts_online) > 0, with the primary bucket-cluster interval above 0.

**What was run.** Qwen2.5-0.5B-Instruct, GSM8K, G 8 rollouts per group, reused from L13a (train, pid 3000 to 7472) and L14a (test, pid 0 to 1318). No generation, no GPU, CPU only. K = 4 subsample: for each group, 4 rollouts drawn without replacement by np.random.default_rng(0), groups visited in pool order. Builder: avrl/exp/l16a_noisy_proxy_table.py (new). Control: the same builder at K = 8 reproduces avrl/runs/L14b/pool_feats.parquet on all 5,792 rows for majority, proxy_error, agreement, n_distinct, mean_tokens, std_tokens and any_truncated (0 mismatches). Readout: avrl/exp/l14b_claim2_readout.py, unchanged, on the K = 4 table. Features cheap+emb (22 columns: 6 cheap plus 16 PCs), cell100 buckets (100 kept, 25 held out as the seed-0 fit quarter), budgets 10, 25 and 50% (431, 1,079 and 2,159 docs), 8 allocator seeds. Primary interval: 200 bucket-cluster resamples, seed 123. Secondary: 200 group resamples, BOOT_SEED 0. Descriptive repeats with subsample seeds 1 and 2. Sim pool at K = 4: 4,318 groups in 75 buckets, 2,282 proxy errors, base rate 0.5285 (K = 8: 1,906 errors, base rate 0.4414).

**Results.** Admitted = 1 minus caught over total errors. Caught is the seed mean (sd) over 8 allocator seeds, except the deterministic static arm. D is measured against the threshold arm.

At 25% (1,079 docs):

| arm (25%) | K = 4 caught (sd) | K = 4 admitted | K = 4 D vs threshold | K = 8 admitted (L14b) | K = 8 D vs threshold |
|---|---|---|---|---|---|
| uniform (label-free baseline) | 565.9 (10.6) | 0.752 | -0.183 | 0.751 | -0.232 |
| static (labeled reference, fit quarter, deterministic) | 1006.0 (0.0) | 0.559 | +0.010 | 0.508 | +0.011 |
| threshold (label-free, agreement) | 983.8 (2.3) | 0.569 | 0 | 0.519 | 0 |
| greedy_online (label-free) | 941.4 (6.3) | 0.587 | -0.019 | 0.531 | -0.011 |
| ts_online (label-free, cheap+emb) | 813.0 (9.9) | 0.644 | -0.075 | 0.596 | -0.077 |

Pool and interval summary:

| quantity | K = 8 (L14b, clean) | K = 4 (L16a, noisier) |
|---|---|---|
| sim pool groups / buckets | 4,318 / 75 | 4,318 / 75 |
| sim pool proxy errors (base rate) | 1,906 (0.441) | 2,282 (0.529) |
| threshold admitted at 25% | 0.519 | 0.569 |
| D(ts_online) at 10% (descriptive) | -0.050 | -0.022 |
| D(ts_online) at 25%, point | -0.077 | -0.075 |
| D(ts_online) at 25%, primary bucket-cluster 95% CI | [-0.096, -0.065] | [-0.091, -0.060] |
| D(ts_online) at 25%, secondary group 95% CI | [-0.101, -0.074] | [-0.085, -0.068] |
| D(ts_online) at 50% (descriptive) | -0.093 | -0.122 |
| D(greedy_online) at 25%, point / group 95% CI | -0.011 / [-0.021, -0.005] | -0.019 / [-0.025, -0.013] |

Descriptive subsample-seed repeats of the K = 4 readout (verdict unchanged in each):

| subsample seed | proxy errors | D(ts_online) at 25% | primary cluster CI | group CI | verdict |
|---|---|---|---|---|---|
| 0 (primary) | 2,282 | -0.075 | [-0.091, -0.060] | [-0.085, -0.068] | KILL |
| 1 | 2,298 | -0.075 | [-0.092, -0.057] | [-0.088, -0.067] | KILL |
| 2 | 2,340 | -0.075 | [-0.090, -0.059] | [-0.087, -0.069] | KILL |

Whole-pool K = 4 proxy error is 0.532 (train 0.506, test 0.617; total 3,079). K = 4 changes proxy_error on 840 of 5,792 rows.

**Verdict against the pre-registered rules.** Point D(ts_online) at 25% is -0.075. The pass rule (point D > 0 and primary lower bound > 0) is not met. The kill rule (point D at or below 0) is met. KILL, not inconclusive: the primary interval [-0.091, -0.060] lies entirely below 0 (0 of 200 draws above 0). The secondary interval and both seed repeats agree. The control passed, so this is a verdict, not a stop.

**What it changes about the thesis claims.**
- Claim 2 (a learned allocator admits fewer corrupted rewards than any fixed rule at equal budget): not_supported on this pool. The noisier 4-of-8 proxy did not flip the sign. D(ts_online) at 25% is -0.075 at K = 4 against -0.077 at K = 8, and it stays negative at 10% and 50%. Both the threshold and ts_online admit more corrupted rewards at K = 4 (about 0.05 more each), and the gap between them barely moves, so proxy noise is not what made the label-free learner lose. Scope: label-free ts_online and greedy_online only. greedy_online also loses at K = 4 (D -0.019). The labeled static arm is the only learned arm below the threshold at both K (+0.010 at K = 4, +0.011 at K = 8), but it has no interval and L12a has not been run, so labeled learners stay open.
- Proxy-quality question (PROJECT.md): one noisier regime, not a curve. It rules out proxy noise as the reason for the label-free kill, at K = 4 only. It does not measure a sign change at any other K.
- Claim 3 (the edge comes from structure in prompt space): not tested by L16a (no embedding ablation, no confident-subset AUC). Standing reading unchanged: not_supported on GSM8K with the majority-vote proxy (E01 pre-registered kill; L13a pooled AUC gain -0.0004 against a 0.05 bar).

**Follow-ups** (none run; each needs its own pre-registration before it runs).
1. L17a (CPU, descriptive, under 5 min): bucket-cluster bootstrap (seed 123, 200 resamples) of the labeled static arm's D at 25% on both the K = 8 and K = 4 pools. Information: high for claim 2, because it is the only learned arm below the threshold at both proxy qualities and has no interval. Prior: interval includes 0 (the E10 ceiling static arm in L11b had CI [-0.017, +0.035]). Pass would reopen claim 2 for labeled learners only.
2. L17b (CPU, descriptive, about 3 min per K): the unchanged readout at K = 2 and K = 6 on the same bootstrap seeds. Information: medium. It tells whether D is flat across proxy quality or only flat between K = 8 and K = 4. K = 2 agreement takes only {0.5, 1.0}, so read it as a stress point. Prior: kill at both.
3. L10b (existing backlog item, G 16, GPU, about 28 min): parked, not re-proposed. The K = 4 result suggests the label-free sign does not track proxy noise, so its expected information is low unless L17a comes back positive.

**Checks and caveats.**
- Control passed: the K = 8 rebuild matches avrl/runs/L14b/pool_feats.parquet on all 5,792 rows (0 mismatches on the seven proxy columns). Non-proxy columns and pool row order match exactly.
- Checked from stored files: the D points and both intervals, recomputed as the 2.5 and 97.5 percentiles of the stored 200 draws, match result.json and readout.log. The admitted values match caught means over pool error totals (for example 1 - 983.75/2,282 = 0.569). The seed 1 and 2 repeats match their result.json files.
- Not recomputed in this pass: the per-row K = 4 majority from raw rollouts. The builder is the same function that passed the K = 8 control; only the subsample index differs.
- result.json reads exp "L14b" because the unchanged readout writes that label. The L16a identity is in config.feats (avrl/runs/L16a/pool_feats_noisy4.parquet). This is an entry label only.
- Scope: one rollout sample (seed 0), one proxy regime (K = 4) with one primary subsample seed, Qwen2.5-0.5B-Instruct, GSM8K, MiniLM refit features, one allocator design, 8 allocator seeds. Descriptive seed repeats do not change the verdict.
- No GPU job, no new rollouts, no edits to p0 or L14b scripts, no push.

Outputs: avrl/runs/L16a/{result.json, control.json, build.json, build.log, readout.log, pool_feats_noisy4.parquet, seed1/, seed2/, exit_code, done.flag}.

**Critic note (L16a, 2026-10-19).** Verdict unchanged: KILL. Point D(ts_online) at 25% is -0.0748, at or below 0, so the L14b pre-registered rule gives KILL; the primary bucket-cluster interval [-0.0912, -0.0603] and the secondary group interval [-0.0846, -0.0677] both lie below 0 (0 of 200 draws above 0 in each; the largest stored cluster draw is -0.0524). The control passed, so this is a verdict and not a stop. Checked against result.json, control.json, build.json, build.log, readout.log, pool_feats_noisy4.parquet, seed1/ and seed2/ (result.json, readout.log, build.json, control.json, parquet), exp/L16a.sh, exp/l16a_noisy_proxy_table.py, exp/l14b_claim2_readout.py, the L13a and L14a rollouts, L14b/pool_feats.parquet and L14b/result.json, and the ledger snapshots for timing. Recomputed independently on CPU, nothing written. (a) I rebuilt the K = 4 proxy columns from the raw rollouts with my own code (the same np.random.default_rng(seed) draw order, np.unique argmax majority): majority, proxy_error, agreement, n_distinct, mean_tokens, std_tokens and any_truncated have 0 mismatches against the stored parquet for subsample seeds 0, 1 and 2, and every non-proxy column (split, pid, gold, log_qlen, q_len, cell100, pc0 to pc15) is identical to L14b/pool_feats.parquet. Error totals over the 5,792 rows are 3,079, 3,091 and 3,126, with 840, 832 and 825 rows changed against K = 8, as build.json and the entry say. This closes the entry's "not recomputed in this pass" caveat. (b) Importing the readout functions on each table, every stored caught mean at 25% and 50% reproduces to 1e-9 for all six arms (seeds 0 and 1 rechecked, seed 2 read from file), the sim pool is 4,318 groups in 75 buckets with 2,282 / 2,298 / 2,340 errors, the fit quarter is 25 buckets and 1,474 groups, and the first two stored cluster draws of D(ts_online) recompute exactly from CLUSTER_SEED 123 (-0.052357, -0.072280 for seed 0). The 2.5 and 97.5 percentiles of the stored draws reproduce all intervals. Every number in the entry's two tables and the seed-repeat table matches the raw files (rounding only). (c) No leakage: the 22 allocator columns are the six cheap columns and pc0 to pc15, none derived from gold; gold enters only through proxy_error, which ts_online and greedy_online read for audited documents only; the subsample index is drawn without reference to gold; cell100 and the PCs come from question text; the fit-quarter labels enter only the labeled static reference arm. (d) Budget accounting is equal: every arm spends int(bf * 4,318) audits (431, 1,079, 2,159), and ts_online and greedy_online pay for warm-up labels inside it. (e) The readout script's mtime (01:21:37) predates every L16a output, so "unchanged" holds; no traceback, warning, NaN or kill text in any L16a log; the exit code is 0. (f) Seed adequacy for the verdict: the allocator-seed standard error of D(ts_online) is about 0.0015 (sd 9.9 over 8 seeds, divided by 2,282), so the margin is about 50 standard errors, and the three subsample seeds give D -0.0748, -0.0751, -0.0753. Those three draws reuse the same 8 rollouts per group, so they do not cover rollout noise (the entry's scope line already says one rollout sample).

Corrections and qualifications, none changes the verdict. (1) The entry's "proxy noise is not what made the label-free learner lose" and "It rules out proxy noise as the reason for the label-free kill" are stronger than this regime supports. At K = 4 the sim pool has 1,754 documents with all four answers distinct (agreement 0.25, n_distinct 4) and 1,601 of them are errors (precision 0.913). The 10% and 25% budgets (431 and 1,079 docs) both lie entirely inside that one tie class, so the threshold at 25% is a random draw from it (expected 1,079 x 0.913 = 984.9 caught, observed 983.75). Perfect labels at 25% admit 1 - 1,079/2,282 = 0.527, so the largest possible D over the threshold is +0.042, half the K = 8 headroom of +0.085, and ts_online sits 0.075 below, about 1.8 times the whole headroom (the labeled static arm's +0.010 is about 23% of it). So K = 4 raised the base rate (0.441 to 0.529) and made the threshold relatively stronger, which is the opposite of what a regime that favors a learner needs. The rank quality of agreement did not change (whole-pool AUC of minus agreement for proxy_error is 0.879 at K = 8 and 0.877 at K = 4), and 42.9% of K = 4 groups are exact vote ties resolved by the smallest-answer rule (error rate 0.90 inside ties, 0.25 outside). The K = 4 majority error is therefore tie-heavy and tightly tied to the same agreement signal the threshold reads, unlike a noisy judge whose errors are independent of the cheap features. The supportable statement is: one 4-of-8 subsample regime, with a tie-dominated proxy and a halved headroom, did not flip the sign. It is a weaker test of the proxy-quality question than the entry's "Information: high" framing suggests, and it does not rule out proxy noise of other kinds. (2) "The gap between them barely moves" holds at 25% only. D(ts_online) goes from -0.050 (K = 8) to -0.022 (K = 4) at 10% and from -0.093 to -0.122 at 50%, so D is not flat across budget; the sign is negative at all three. (3) "greedy_online also loses at K = 4 (D -0.019)" is true at 25% (and at 10%, -0.008) but not at 50%: D(greedy_online) at 50% is +0.0039 at subsample seed 0 (+0.0024 and +0.0036 at seeds 1 and 2) against -0.0003 at K = 8. It has no interval, the threshold's caught count at 50% depends on its tie-break (1,780 with random ties, 1,761 in input order), and the verdict is about ts_online at 25%, so this is descriptive, but it is the first positive point for a label-free online arm anywhere on the L14/L16 pools and the entry omits it. (4) The labeled static arm at 50% is +0.0237 at K = 4 (+0.0218 and +0.0198 at seeds 1 and 2) and +0.0191 at K = 8, larger than at 25%; descriptive, no interval, and consistent with L17a being the right next step. (5) The entry says the verdict follows "the pre-registered rules". The rule is the L14b rule, carried over unchanged, which is correct, but no L16a pre-registration entry exists: the ledger snapshots show 1,641 lines at 01:36 (the L14b critic note) and the next ledger change is the finished 1,705-line L16a entry at 01:50. The L16a design (K = 4, subsample seed 0 as primary, seeds 1 and 2 descriptive, budgets, bootstrap seeds) is fixed in exp/L16a.sh (mtime 01:44:16) before any L16a output existed (build.json 01:44, result.json 01:45), and the L14b follow-up item 1 announced the replay, so there is no sign of a post hoc choice, but the design was not logged ahead of the run. (6) For L17b: K = 2 and K = 6 are even K and bring more exact ties; K = 3 and K = 5 would test proxy quality with far fewer tie-break outcomes, and L17b should state its tie rule before it runs.

Corrected verdict: KILL, unchanged. Claim 2 (a learned allocator admits fewer corrupted rewards than any fixed rule at equal budget): KILL for the label-free online learner ts_online with cheap+emb features on cell100 buckets at a 25% budget, under the K = 4 majority proxy on the 5,792-group L14a gate-passing pool (sim pool 4,318 groups, 75 buckets, 2,282 errors; point D -0.0748, bucket-cluster 95% CI [-0.0912, -0.0603], group CI [-0.0846, -0.0677]; subsample seeds 1 and 2 agree). Proxy-quality question: one tie-heavy K = 4 regime with headroom +0.042 did not flip the sign; it is neither a curve nor a ruling-out of proxy noise. Labeled learners (static arm, no interval) and greedy_online at 50% remain descriptive leads, untested. Claim 3: not tested by L16a. Scope: Qwen2.5-0.5B-Instruct, GSM8K, G 8 with 4 kept, 512 tokens, one rollout sample at seed 0, majority-vote proxy, MiniLM refit features, one allocator design, 8 allocator seeds.

## L16b pre-registration 2026-10-19 (written before any L16b script ran or any L16b output was computed)

**Question.** On the L14b gate-passing GSM8K pool (cell100 buckets, cheap+emb features, 25% budget): does a bucket oracle that ranks cell100 buckets by their true corrupted-reward rate admit fewer corrupted rewards than the label-free agreement threshold? The oracle uses labels, so it is an upper bound on what any bucket-level allocator can reach with these buckets. A kill means the cell100 bucket design cannot beat the threshold on this pool, whatever the learner, so bucket-level allocators are closed on this pool, not just ts_online. A pass means the L14b failure sits in learning the buckets, not in the bucket design. Either way, claim 2 stays killed for label-free learners.

**Pool and features.** Unchanged from L14b: avrl/runs/L14b/pool_feats.parquet, 5,792 groups, Qwen2.5-0.5B-Instruct, GSM8K, G 8, K = 8 majority proxy (proxy_error = majority answer differs from gold). Buckets with fewer than 3 groups are dropped and the seed-0 fit quarter of shuffled cell100 buckets is held out. Sim pool: 4,318 groups in 75 buckets, 1,906 proxy errors, budget 1,079 docs at 25%. Features cheap+emb (22 columns). No GPU, no rollouts, no refit.

**New arm: bucket_oracle (labels used, deterministic, one value per budget).**
- Each sim-pool bucket's rate is the mean of proxy_error over its groups.
- Buckets are sorted by rate, descending. Equal rates are ordered by bucket code (first appearance in the sim pool).
- Buckets stay contiguous: every group of a bucket comes before any group of the next bucket. Inside a bucket, groups are ordered by the threshold key (agreement ascending, then n_distinct descending), then by pool row order.
- caught = sum of proxy_error over the first budget groups in that order.

**Metric.** admitted(arm) = 1 - caught / total, with total = proxy errors in the replayed pool. D(bucket_oracle) = admitted(threshold) - admitted(bucket_oracle) at 25%. Positive D means the oracle admits fewer corrupted rewards than the threshold. The threshold arm is as in L14b: seed mean over 8 random tie-break seeds.

**Intervals.**
- Primary: 95% percentile interval of D(bucket_oracle) at 25% over 200 bucket-cluster resamples (sim-pool buckets drawn with replacement, CLUSTER_SEED 123, each repeated bucket relabeled as a new cluster, as in L14b). Bucket rates are recomputed inside every resample, and the threshold arm is recomputed on the same resample.
- Secondary: 95% percentile interval over 200 group resamples (BOOT_SEED 0, the L14b group draws), with bucket rates also recomputed per resample.
- The 10% and 50% points are descriptive. The secondary interval cannot change the verdict.

**Sanity bound.** With perfect labels the caught count is at most min(budget, total) = 1,079, so admitted is at least 1 - 1,079/1,906 = 0.434 and D at 25% is at most 0.519 - 0.434 = +0.085. A point D above +0.085 is a bug: the run stops and the bug is reported.

**Verdict rules** (point D and primary interval of D(bucket_oracle) at 25%).
- PASS: point D > 0 AND the lower bound of the primary bucket-cluster interval > 0. This shows bucket-level headroom on this pool and reopens only a labeled-learner follow-up (L17a, then L12a).
- KILL: point D <= 0. This closes the bucket family on this pool, not only ts_online.
- INCONCLUSIVE: point D > 0 and the primary interval includes 0.

**Control (before the oracle row is read).** The unchanged arms of the copy must reproduce avrl/runs/L14b/result.json: for uniform, static, threshold, threshold_input_order, greedy_online and ts_online at 10%, 25% and 50%, the caught mean, caught sd, admitted and n_runs; the point D of ts_online and greedy_online; and the 200 cluster and 200 group draws of D(ts_online) and D(greedy_online). Tolerance 1e-9, 0 mismatches required. In particular the threshold admitted at 25% must be 0.519 and D(ts_online) at 25% must be -0.077. Any mismatch: stop and report it, with no L16b verdict.

**Independent check.** avrl/exp/L16b_check.py recomputes the 25% threshold caught mean (8 seeds) and the bucket_oracle caught count from avrl/runs/L14b/pool_feats.parquet with pandas, not the copy's numpy code. The point estimates must match avrl/runs/L16b/result.json to 1e-9, and the point D must lie below the sanity bound.

**Edits.** avrl/exp/l16b_bucket_oracle.py is a copy of avrl/exp/l14b_claim2_readout.py with one arm added (bucket_oracle), the verdict switched to that arm, and the docstring changed. The L14b script is not edited: md5 2117be0d85dbcb043018c1b618f70bd7 before the copy, and the same md5 is rechecked after the run.

**Run.** CPU only, run in the shell via avrl/exp/L16b.sh (no gpu_run.sh, no GPU). Outputs in avrl/runs/L16b/: result.json, readout.log, exit_code, done.flag. The control and the independent check write control.json and check.json. Expected time about 5 minutes.

**Scope.** One rollout sample (seed 0), K = 8 majority proxy, Qwen2.5-0.5B-Instruct, GSM8K, MiniLM refit features, cell100 buckets, one oracle, 8 threshold tie-break seeds, 200 resamples per interval.

## L16b 2026-10-19 kill

**Hypothesis.** On the 5,792-group L14a gate-passing GSM8K pool (L14b cell100 buckets, cheap+emb features), a bucket oracle that ranks cell100 buckets by their true proxy_error rate (labels used, upper bound) does not admit fewer corrupted rewards than the label-free agreement threshold at a 25% budget. Kill: point D(bucket_oracle) at 25% at or below 0.

**What was run.** avrl/exp/l16b_bucket_oracle.py is a copy of the L14b readout with one arm added (bucket_oracle). The L14b script is unchanged (md5 2117be0d85dbcb043018c1b618f70bd7 before the copy and after the run). Pool: avrl/runs/L14b/pool_feats.parquet, sim pool 4,318 groups in 75 buckets, 1,906 proxy errors. CPU only, through avrl/exp/L16b.sh (about 1 minute). No GPU, no rollouts, no refit.

bucket_oracle: buckets sorted by true mean proxy_error, descending, equal rates by bucket code (first appearance). Buckets are kept contiguous. Inside a bucket, groups are ordered by agreement ascending, then n_distinct descending, then pool row order. Caught = sum of proxy_error over the first budget groups. Deterministic, one value per budget.

**Control (passed before the oracle row was read).** 911 values compared against avrl/runs/L14b/result.json, 0 mismatches at 1e-9: the six L14b arms at 10, 25 and 50% (budget, caught mean and sd, admitted, n_runs), the point D of ts_online and greedy_online, the pool counts, and the 200 cluster and 200 group draws of both D values. Threshold admitted at 25% is 0.519 and D(ts_online) at 25% is -0.077, as required. Output: avrl/runs/L16b/control.json.

**Results.** Admitted = 1 - caught / 1,906. The threshold caught is the mean over 8 tie-break seeds. The oracle is deterministic.

| arm | admitted 10% | caught 25% | admitted 25% | admitted 50% |
|---|---|---|---|---|
| uniform (label-free baseline) | 0.900 | 473.9 | 0.751 | 0.505 |
| threshold (label-free, agreement) | 0.780 | 916.6 | 0.519 | 0.178 |
| bucket_oracle (labels used, upper bound) | 0.857 | 621.0 | 0.674 | 0.413 |
| ts_online (label-free, from L14b) | 0.829 | 770.5 | 0.596 | 0.271 |
| greedy_online (label-free, from L14b) | 0.805 | 894.8 | 0.531 | 0.178 |

At 25% (1,079 docs): D(bucket_oracle) = admitted(threshold) - admitted(bucket_oracle) = -0.155. Primary bucket-cluster 95% interval [-0.182, -0.134] (0 of 200 resamples above 0, largest draw -0.126). Secondary group 95% interval [-0.160, -0.129] (0 of 200 above 0). Descriptive: D at 10% is -0.077 and at 50% is -0.235. The sanity ceiling with perfect labels is +0.085 at 25%, and the point value is far below it, so there is no sign of a bug in the direction that would inflate the oracle.

**Verdict against the pre-registered rules.** KILL. The point D(bucket_oracle) at 25% is -0.155, at or below 0, and the primary interval lies entirely below 0, so this is a kill, not inconclusive. Under the pre-registered rule, rate-ranked whole-bucket allocation on the cell100 buckets is closed on this pool, whatever the learner. L16b does not reopen L17a or L12a.

**Scope of the kill (my reading, not a pre-registered claim).** The oracle drains whole buckets in rate order. Its 1,079 docs at 25% include about 458 non-errors, against about 162 for the threshold. The oracle takes every doc of a high-rate bucket, including its high-agreement docs, while the threshold picks the error-prone docs across buckets. That mechanism is an inference from these counts and was not measured per bucket. The result therefore says that rate-only bucket ranking carries less information than agreement on this pool. It does not test a doc-level score that combines bucket rate with agreement and n_distinct. Until that score is tested, "bucket-level allocators are closed" should be read as "rate-ranked whole-bucket allocators are closed".

**Follow-ups** (none run; each needs its own pre-registration before it runs).
1. Labeled combined oracle (labels used, upper bound, in-sample on the sim pool): score each group by a logistic fit of proxy_error on agreement, n_distinct and its cell100 bucket rate, then rank by score at 25%. Information: high for the scope question above. Prior: it loses to or ties the threshold, because the threshold already captures most of the agreement signal. A pass would say bucket rate adds information beyond agreement, and would justify a labeled-learner follow-up.
2. L17a stays on the backlog as listed in the L16a entry. L16b neither reopens nor closes it.

**Checks and caveats.**
- Independent check (avrl/exp/L16b_check.py, avrl/runs/L16b/check.json): recomputed from avrl/runs/L14b/pool_feats.parquet with pandas. Sim pool 4,318 groups, 75 buckets, 1,906 errors. Threshold caught mean 916.625 and bucket_oracle caught 621 match result.json exactly (max abs diff 0.0), and so does D(bucket_oracle) at 25%. The sanity ceiling check passes.
- Scope: one rollout sample (seed 0), K = 8 majority proxy, Qwen2.5-0.5B-Instruct, GSM8K, MiniLM refit features (cheap+emb, cell100), one bucket oracle definition, 8 threshold tie-break seeds, 200 resamples per interval.
- The bootstrap budget is fixed at 1,079 docs per resample, as in L14b, while resampled pool sizes vary. This is inherited from L14b and was not changed here.
- The 10% and 50% values are descriptive. The secondary group interval cannot change the verdict.
- No GPU job, no new rollouts, no edits to p0 or L14b scripts, no push.

Outputs: avrl/runs/L16b/{result.json, readout.log, control.json, control.log, check.json, check.log, exit_code, done.flag}. Scripts: avrl/exp/l16b_bucket_oracle.py, avrl/exp/L16b_control.py, avrl/exp/L16b_check.py, avrl/exp/L16b.sh.

**Thesis impact and follow-ups (L16b, 2026-10-19).** This block extends the L16b kill entry above, which already carries the pre-registration, the run, the table and the verdict, so the heading is not repeated. Checked on this date against the raw files: result.json, readout.log, control.json (911 values, 0 mismatches at 1e-9), check.json, the L14b script md5 (2117be0d85dbcb043018c1b618f70bd7, unchanged), and the bucket_oracle code (rates are rebuilt inside each bootstrap resample through make_pool; the order matches the pre-registered rule). A separate in-memory recount from avrl/runs/L14b/pool_feats.parquet, with the seed-0 fit split, reproduces the deterministic threshold at 915 caught (input-order tie-break, logged 915.0) and the oracle at 621 caught (admitted 0.674). The largest bucket-cluster draw of D(bucket_oracle) is -0.126 and the largest group draw is -0.119, so 0 of 200 draws are above 0 in either interval.

The mechanism stated in the scope paragraph is now measured at the document level, not only inferred. Among the oracle's 1,079 docs, mean agreement is 0.388 and 1.4% have agreement 1.0. Among the threshold's 1,079 docs (input-order tie-break), mean agreement is 0.182 and 0% have agreement 1.0. The oracle takes 20 buckets, 19 whole and 1 partly. The sim-pool bucket rates have mean 0.441 and sd 0.102. The oracle still admits fewer errors than uniform (0.674 against 0.751), so bucket rates carry some signal, just less than agreement.

**What it changes about the thesis claims.**
- Claim 2 (a learned allocator admits fewer corrupted rewards than any fixed rule at equal budget): no change for label-free learners, which stay killed on this pool (ts_online at L14b and L16a; greedy_online also loses at 25%, interval below 0). New: rate-ranked whole-bucket allocation on cell100 is closed on this pool even with labels. The labeled doc-level route is still open. The labeled static arm is the one learned arm with a positive point estimate (+0.011 at 25%, admitted 0.508 against 0.519, no interval yet; see L17a).
- Claim 3 (the edge comes from structure in prompt space): not tested directly, since L16b has no embedding-only arm and no confident-subset AUC. The standing reading is unchanged: not_supported (E01 kill; L13a pooled AUC gain -0.0004 against a bar of 0.05). L16b adds that prompt-space bucket membership carries less allocation information than agreement, even when bucket rates are exact. The cell100 neighbor-ratio leg is not used, since it is 1.0 by construction at this base rate.
- Scope of the bucket result: the oracle is an in-sample upper bound (rates come from the sim-pool labels, about 57 docs per bucket), so it is a ceiling for rate-ranked whole-bucket allocation, not a learner.

**Follow-ups** (none run; each needs its own pre-registration before it runs; the two new ones are in lit/avrl/EXPERIMENT_BACKLOG.md as L17c and L17d).
1. L17a (existing item): bucket-cluster interval for the labeled static arm at 25%, on the K = 8 and K = 4 pools. Information: high for claim 2, since labeled learners are the untested part. Prior: the interval includes 0. Run it first, because L17d reuses its bootstrap.
2. L17c (new): labeled doc-level score combining the cell100 rate with agreement and n_distinct (logistic, in-sample upper bound on the sim pool), against the threshold at 25%. This is the scope the bucket kill leaves open. Information: high for that scope. Prior: at most a small margin, since agreement already carries most of the ordering. Pass: point D above 0 with the primary interval excluding 0. Kill: point D at or below 0, which closes bucket structure as an allocation signal on this pool. The current fit and sim split holds out whole buckets, so it cannot estimate a sim bucket's rate; any valid labeled bucket-rate test needs a within-bucket split.
3. L17d (new): labeled static ablation by feature family (cheap six only, cheap six plus the 16 PCs, PCs only), with bucket-cluster intervals at 25% and 50%. Information: high for claim 3 in the labeled setting, since the PCs are the prompt-space features. Prior: the cheap-six model matches the full model, because agreement carries the signal, which would mean the labeled edge does not come from prompt-space structure.

Scope: one rollout sample (seed 0), K = 8 majority proxy, Qwen2.5-0.5B-Instruct, GSM8K, MiniLM refit features (cheap+emb, cell100), one bucket oracle definition, 8 threshold tie-break seeds, 200 resamples per interval. CPU only, no GPU job, no push. The file timestamps cannot show that the pre-registration text was written before the L16b script (RESEARCH_LOG.md is appended to and only its last write time is visible); the backlog item for L16b (lit/avrl/EXPERIMENT_BACKLOG.md, last written 01:50) predates the script (02:04).


**Critic note (L16b, 2026-10-19).** Verdict under the pre-registered rule is unchanged: KILL. Point D(bucket_oracle) at 25% is -0.1551, at or below 0, so the rule gives KILL; the primary bucket-cluster interval [-0.1823, -0.1338] and the secondary group interval [-0.1603, -0.1291] both lie below 0 (0 of 200 draws above 0 in each; largest cluster draw -0.1259, largest group draw -0.1190). The control passed (911 values = 96 point values + 3 pool counts + 804 draws and lengths + 8 interval ends, 0 mismatches), so this is a verdict and not a stop. What I checked and found correct: (a) every number in the results table, the D lines, the intervals, the 458 and about 162 non-error counts, the 0.085 ceiling, the 915 input-order threshold and the 621 oracle count matches result.json, readout.log, control.json and check.json; the logs hold no traceback, warning or NaN, exit_code is 0, and the L14b script md5 is 2117be0d85dbcb043018c1b618f70bd7, unchanged, with an mtime (01:21:37) before the copy (02:04:53). (b) I recomputed the sim pool (4,318 groups, 75 buckets, 1,906 errors, buckets of 22 to 114 groups, median 57), the 8 per-seed threshold counts (919, 904, 913, 920, 913, 926, 911, 927, mean 916.625) and the oracle (621 caught, 20 buckets of which 19 whole and 1 partial, mean agreement 0.388, 1.4% at agreement 1.0, against 0.182 and 0% for the input-order threshold; bucket rates mean 0.441, sd 0.102) with my own code, all as logged. (c) check.py covers only the point estimates, so I also recomputed the oracle's bootstrap with separate pandas code (bucket-cluster resampler at seed 123 and the group resampler at seed 0, bucket rates and the 8-seed threshold rebuilt in every resample): all 200 cluster draws and all 200 group draws of D(bucket_oracle) equal the stored draws (max abs diff 0.0), so both intervals are reproduced. (d) No leakage: the 22 allocator columns are agreement, n_distinct, token statistics, any_truncated, log_qlen and the 16 PCs, built from the answer distribution, token counts and question text (l14b_feature_refit.py), none from gold; gold enters only through proxy_error, which the oracle uses by design (labeled arm), static uses on the fit split only, and the online arms read for audited documents only. (e) Budget accounting is equal: every arm spends int(bf * 4,318) = 431, 1,079 and 2,159 audits, and ts_online and greedy_online pay for warm-up labels inside it. (f) Seed adequacy: the threshold's per-seed sd of 7.3 caught gives a standard error of about 2.6 caught (0.0014 of admitted) against a margin of 0.155, and the oracle is deterministic, so tie-break seeds are not the limit; the one rollout sample is.

Corrections and qualifications, none changes the KILL. (1) The label "bucket_oracle (labels used, upper bound)" in the results table, and the pre-registered premise that the oracle "is an upper bound on what any bucket-level allocator can reach with these buckets", are contradicted by the table itself: the label-free ts_online, a Thompson allocator over the same 75 cell100 buckets, catches more than the "upper bound" at every budget (325.25 against 273 at 10%, 770.5 against 621 at 25%, 1,389 against 1,119 at 50%; admitted 0.596 against 0.674 at 25%). The entry never reports this. The oracle is an upper bound only for one policy, strictly rate-ordered contiguous whole-bucket drain with the threshold key inside, and "whatever the learner" in the verdict paragraph holds only for learners that ship that policy with their rate estimates, which ts_online does not. The sentence "bucket-level allocators are closed on this pool, not just ts_online" in the pre-registration therefore does not follow from the KILL, and neither does its mirror image, "the L14b failure sits in learning the buckets". What does follow, and it is informative: exact bucket rates used in a rate-ordered drain are worse than the learned ts_online (D -0.155 against -0.077), so the L14b loss of ts_online is not a rate-estimation failure that exact rates would fix. The supportable statement is the one in the entry's own "Scope of the kill" paragraph: rate-ranked whole-bucket allocation is closed on this pool. (2) Descriptive and not pre-registered, so it does not touch the verdict: bucket rate does carry a small signal beyond agreement. On the sim pool, a logistic score on agreement, n_distinct and the cell100 bucket rate (in-sample, labels used) catches 939 at 25% (D +0.0117 against the threshold's 916.6); with each group's own label removed from its bucket rate it catches 928 (D +0.0060); a logistic on agreement and n_distinct alone catches 915, and using the leave-one-out rate only as a tie-break catches 915 (D -0.0009). Single point estimates, no interval, one pool. This matches the entry's prior for L17c (at most a small margin) and puts the in-sample bucket-rate headroom at about +0.01, so the doc-level route is open but small. Whole-pool in-sample AUC for proxy_error is 0.878 for minus agreement and 0.609 for the bucket rate, which supports the entry's reading that bucket membership carries less allocation information than agreement; the L16b drain alone does not measure that. (3) Timing statements. The entry's last sentence says file timestamps cannot show that the pre-registration preceded the script. events/snapshot.log does: the ledger was 1,711 lines at 01:59:40 and 1,747 lines at 02:03:56, the pre-registration block is lines 1713 to 1747, and l16b_bucket_oracle.py was written at 02:04:53, so the pre-registration was in the ledger before the script existed. The parenthetical "lit/avrl/EXPERIMENT_BACKLOG.md, last written 01:50" is stale and conflicts with the entry's own mention of L17c and L17d: the backlog was last written at 02:11:33 (snapshot.log: 1,401 lines at 01:50:55, 1,441 at 02:11:40, the later growth being L17c and L17d); the L16b backlog item itself was already in the backlog before the script (backlog 1,363 lines at 01:28:09). (4) The entry says the oracle's non-error count is "about 162 for the threshold"; that is the 8-seed random tie-break mean (1,079 - 916.6), while the document-level agreement figures in the thesis-impact block use the input-order threshold (915 caught, 164 non-errors). Both are right for their arm; they are not the same arm.

Corrected verdict: KILL, unchanged under the pre-registered rule. Claim 2 for label-free learners: killed, as before. Scope of the KILL, corrected: a labeled oracle that ranks the 75 cell100 buckets by exact in-sample proxy_error rate and drains them contiguously (point D -0.155 at 25%, bucket-cluster 95% CI [-0.182, -0.134], group CI [-0.160, -0.129]; D -0.077 at 10% and -0.235 at 50%, descriptive) admits more corrupted rewards than the label-free agreement threshold and also more than the label-free ts_online, so it is not an upper bound for bucket-level allocators; rate-ranked whole-bucket allocation is closed on this pool, bucket-level allocation in general is not tested, and a descriptive in-sample combined score puts bucket-rate headroom beyond agreement at about +0.006 to +0.012. L17a and L12a are not reopened. L17c remains the test of the open scope. Scope: one rollout sample (seed 0), K = 8 majority proxy, Qwen2.5-0.5B-Instruct, GSM8K, MiniLM refit features, cell100, one oracle definition, 8 tie-break seeds, 200 resamples per interval.

## L17d pre-registration 2026-10-19 (written before any L17d script ran or any L17d output was computed)

**Question.** Claim 3 for labeled learners, on the L14b sim pool (cell100 buckets, K = 8 majority proxy, 25% budget). The labeled static arm (logistic regression on the fit quarter, class_weight balanced) admits 0.508 corrupted rewards at 25%, against 0.519 for the label-free agreement threshold. Does the six cheap columns alone match the cheap six plus the 16 prompt-space PCs? If the PCs add nothing, the labeled margin comes from the cheap features and claim 3 is killed for labeled learners on this pool. If the PCs help, the labeled edge sits in prompt-space structure and a follow-up is pre-registered before it runs.

**Pool and split.** Unchanged from L14b: avrl/runs/L14b/pool_feats.parquet, 5,792 groups, Qwen2.5-0.5B-Instruct, GSM8K, G 8, K = 8 majority proxy (proxy_error = majority answer differs from gold). Buckets with fewer than 3 groups are dropped. The seed-0 fit quarter of shuffled cell100 buckets is held out and used only to fit the static models. Sim pool: 4,318 groups in 75 buckets with 1,906 proxy errors (the L14b counts, which the control re-checks). No GPU, no rollouts, no refit, no embedding model load.

**Arms.** Three static logistic models, each fit once on the fit quarter with the L14b settings (make_pipeline(StandardScaler(), LogisticRegression(max_iter=1000, class_weight="balanced"))), labels proxy_error from the fit split only, scores on the sim pool. Each static arm ranks sim docs by np.argsort(-score) (the L14b call) and takes the first budget docs.
- static: cheap six plus pc0 to pc15 (22 columns). This is the L14b reference arm and must reproduce 938.0 caught at 25% (deterministic, admitted 0.508).
- static_cheap6: agreement, n_distinct, mean_tokens, std_tokens, any_truncated, log_qlen.
- static_pc: pc0 to pc15 only (descriptive).
The uniform, threshold, threshold_input_order, greedy_online and ts_online arms are unchanged from L14b and are computed only so the control can compare them with avrl/runs/L14b/result.json.

**Metric.** admitted(arm) = 1 - caught(arm) / total, where total is the number of proxy errors in the pool being scored (1,906 at the point estimate, the resampled pool's own total in a bootstrap resample). The threshold arm is the mean over 8 random tie-break seeds, as in L14b. Budgets: 1,079 docs at 25% and 2,159 docs at 50% (int(bf x 4,318)).
- Delta = admitted(static_cheap6) - admitted(static), at 25% and at 50%. This is the decision metric. Sign: positive means the cheap six alone admit more corrupted rewards, so the PCs reduce admission.
- Descriptive: Delta_pc = admitted(static_pc) - admitted(static) at 25% and 50%.
- D(static) = admitted(threshold) - admitted(static) at 25%, with its bucket-cluster interval. This is the K = 8 half of the L17a item (labeled static arm interval). The K = 4 half stays open.

**Intervals.** Primary: 95% percentile interval over 200 bucket-cluster resamples (sim-pool buckets drawn with replacement, CLUSTER_SEED 123, each repeated bucket relabeled as a new cluster, as in L14b). In each resample the threshold and the three static arms are recomputed on the resampled pool, and the static models are not refit (they are fixed). Secondary: 95% percentile interval over 200 group resamples (sim groups with replacement, BOOT_SEED 0, the L14b draws). The bootstrap budgets are fixed at 1,079 and 2,159 docs per resample, as in L14b, while the resampled pool size varies (inherited, not changed here). The ts_online and greedy_online draws at 25% are recomputed as in L14b, for the control only.

**Verdict rules** (on Delta at 25% and at 50%, primary interval).
- PASS: at 25% or at 50%, point Delta > 0 AND the lower bound of the primary bucket-cluster interval > 0. The PCs then admit fewer corrupted rewards than the cheap six alone. This is the first labeled evidence for claim 3, and a follow-up would be pre-registered before it runs.
- KILL: point Delta <= 0 at both 25% and 50%. The PCs add nothing on this pool, so claim 3 is killed for labeled learners here. No equivalence margin is pre-registered, so KILL means the cheap six are not beaten by cheap plus PCs; the reported intervals give the size of the gap.
- INCONCLUSIVE: any other pattern, including point Delta > 0 whose primary interval includes 0.
- The group intervals, Delta_pc, D(static) and the other budgets cannot change the verdict.

**Control (before any L17d row is read).** The copy's L14b arms (uniform, static, threshold, threshold_input_order, greedy_online, ts_online at 10%, 25% and 50%: caught mean, caught sd, admitted, n_runs; point D of ts_online and greedy_online; pool counts; the 200 cluster and 200 group draws and CIs of D(ts_online) and D(greedy_online)) must reproduce avrl/runs/L14b/result.json to 1e-9 with 0 mismatches. Required: static caught 938.0 at 25%, threshold admitted 0.519 at 25%, D(ts_online) at 25% = -0.077. Any mismatch: stop and report it, with no L17d verdict. avrl/exp/L17d_control.py writes control.json.

**Independent check.** avrl/exp/L17d_check.py recomputes from avrl/runs/L14b/pool_feats.parquet with pandas and its own code: it refits the three static models with sklearn on the fit quarter, builds each resampled pool with pandas concat and a stable sort on the source row, and computes the threshold tie-break with a pandas sort. It compares with result.json to 1e-9: the caught counts at 25% and 50% for the three static arms and the threshold mean; the point Delta at 25% and 50%; the point D(static) at 25%; and all 200 cluster draws of Delta at 25% and 50% and of D(static) at 25%. It writes check.json. A mismatch is reported with its numbers, and no verdict is written until it is explained.

**Tie diagnostic (reported, not used for the verdict).** For each static arm and budget the run reports how many sim docs share the score of the cutoff doc (the last one taken). A tie at the cutoff makes the caught count depend on the argsort order, and that is stated as a caveat.

**Prior.** KILL. Agreement carries most of the ordering (in-sample whole-pool AUC for proxy_error 0.878 for minus agreement against 0.609 for the cell100 bucket rate, from the L16b Critic note), and the L16b follow-up 3 prior was that the cheap six match the full model. Adding 16 PCs to a six-feature logistic is not expected to change the ordering much, so the expected |Delta| is small.

**Edits.** avrl/exp/l17d_static_ablation.py is a copy of avrl/exp/l14b_claim2_readout.py, copied and then edited: the single static arm becomes three fixed feature sets, the bootstrap also recomputes the three static arms at 25% and 50%, and the verdict uses Delta. The L14b script is not edited: md5 2117be0d85dbcb043018c1b618f70bd7 before the copy, rechecked after the run. No p0 script is edited.

**Run.** CPU only, in the shell, via avrl/exp/L17d.sh (no gpu_run.sh, no GPU, no vLLM). Readout command: /homes/rdubash/active-audit/.venv/bin/python avrl/exp/l17d_static_ablation.py --feats avrl/runs/L14b/pool_feats.parquet --cell cell100 --budgets 0.10,0.25,0.50 --boot_budget 0.25 --boot 200 --out avrl/runs/L17d. Expected about 3 minutes.

**Outputs.** avrl/runs/L17d/: result.json, readout.log, control.json, control.log, check.json, check.log, exit_code, done.flag.

**Scope.** One rollout sample (seed 0), K = 8 majority proxy, Qwen2.5-0.5B-Instruct, GSM8K, G 8, MiniLM refit features (cheap six, pc0 to pc15, cell100), three fixed static logistic models fit on the fit quarter, 8 threshold tie-break seeds, 200 resamples per interval. The test is for labeled learners only. Label-free learners and other feature families are not in it.

## L17d 2026-10-19 inconclusive (claim 3 for labeled learners: static ablation by feature family at 25% and 50%)

**Hypothesis.** As pre-registered in the "L17d pre-registration" entry above: on the L14b sim pool, the labeled static arm's margin over the threshold at 25% comes from the six cheap columns, and the cheap six alone match cheap plus the 16 PCs. Kill: point Delta <= 0 at both 25% and 50%, Delta = admitted(static_cheap6) - admitted(static).

**What was run.** avrl/exp/l17d_static_ablation.py is a copy of avrl/exp/l14b_claim2_readout.py (the L14b script is unchanged: md5 2117be0d85dbcb043018c1b618f70bd7 before the copy and after the run). Pool: avrl/runs/L14b/pool_feats.parquet, sim pool 4,318 groups in 75 buckets with 1,906 proxy errors. Budgets 1,079 docs (25%) and 2,159 docs (50%), 8 threshold tie-break seeds, 200 bucket-cluster draws (CLUSTER_SEED 123) and 200 group draws (BOOT_SEED 0) for each budget. CPU only through avrl/exp/L17d.sh: readout exit 0, 184 s. Control (avrl/exp/L17d_control.py): 914 values compared with avrl/runs/L14b/result.json, 0 mismatches at 1e-9. Gates pass: static caught 938.0 at 25%, threshold admitted 0.519 at 25%, D(ts_online) at 25% -0.077. Independent pandas check (avrl/exp/L17d_check.py): the three static models refit with sklearn on the fit quarter, resampled pools built with pandas, 4,023 values compared (the caught counts, the point Delta values, and all 200 cluster and 200 group draws of the five metrics at 25% and 50%), max abs diff 0.0.

**Results.** Admitted = 1 - caught / 1,906. Point estimates; the static arms are deterministic, the threshold is the 8-seed mean.

| arm | caught 25% | admitted 25% | caught 50% | admitted 50% |
|---|---|---|---|---|
| uniform (label-free baseline) | 473.9 | 0.751 | 943.9 | 0.505 |
| static = cheap+PC (labeled reference) | 938.0 | 0.508 | 1603 | 0.159 |
| static_cheap6 (labeled) | 955.0 | 0.499 | 1599 | 0.161 |
| static_pc (labeled, descriptive) | 597.0 | 0.687 | 1106 | 0.420 |
| threshold (label-free) | 916.6 | 0.519 | 1566.6 | 0.178 |

| metric | budget | point | primary bucket-cluster 95% CI | secondary group 95% CI |
|---|---|---|---|---|
| Delta = adm(cheap6) - adm(cheap+PC) (decision metric) | 25% | -0.0089 | [-0.0166, +0.0020] | [-0.0158, +0.0005] |
| Delta (decision metric) | 50% | +0.0021 | [-0.0075, +0.0106] | [-0.0052, +0.0103] |
| Delta_pc = adm(pc) - adm(cheap+PC) (descriptive) | 25% | +0.1789 | [+0.1527, +0.2007] | [+0.1624, +0.1956] |
| Delta_pc (descriptive) | 50% | +0.2608 | [+0.2329, +0.2891] | [+0.2375, +0.2801] |
| D(static cheap+PC) = adm(threshold) - adm(static) (K = 8 half of L17a) | 25% | +0.0112 | [+0.0005, +0.0206] | [+0.0012, +0.0203] |
| D(static cheap+PC) | 50% | +0.0191 | [+0.0044, +0.0281] | [+0.0079, +0.0280] |
| D(static cheap6) (descriptive) | 25% | +0.0201 | [+0.0103, +0.0301] | [+0.0084, +0.0286] |
| D(static cheap6) (descriptive) | 50% | +0.0170 | [+0.0046, +0.0297] | [+0.0057, +0.0225] |
| D(static pc) (descriptive) | 25% | -0.1677 | [-0.1913, -0.1393] | [-0.1866, -0.1495] |
| D(static pc) (descriptive) | 50% | -0.2417 | [-0.2731, -0.2114] | [-0.2641, -0.2223] |

The control arms are unchanged from L14b: D(ts_online) at 25% -0.0767 with cluster interval [-0.0958, -0.0651]; D(greedy_online) at 25% -0.0115 with cluster interval [-0.0225, -0.0034]. Tie diagnostic: zero other sim docs share the cutoff score for any static arm at any budget, so the argsort tie order cannot change the caught counts.

**Verdict against the pre-registered rules.** INCONCLUSIVE. The point Delta at 25% is -0.0089 (at or below 0), but at 50% it is +0.0021 (above 0), and its primary interval [-0.0075, +0.0106] includes 0. PASS does not apply, because no budget has point Delta > 0 with a lower bound above 0. KILL does not apply, because it needs point Delta <= 0 at both budgets, and 50% fails that. The 25% point is in the kill direction, but the rule as written does not give KILL.

**Claim status.** Claim 3 for labeled learners on this pool: INCONCLUSIVE. It is neither killed nor supported. Descriptive reading: at 25% the cheap six alone admit 0.499 against 0.508 for cheap plus PCs, so the PCs add no detectable margin, and the labeled gain over the threshold is carried by the cheap six (D(static cheap6) +0.020, interval [+0.010, +0.030]). At 50% the PCs change admission by +0.002 with an interval of about plus or minus 0.01, so this design cannot separate a match from a difference of that size. The PC-only arm admits 0.687 at 25%, better than uniform (0.751) but far worse than the threshold (0.519), so prompt-space features carry some signal, and the cheap features carry most of it.

**L17a K = 8 half (descriptive, not a verdict).** D(static cheap+PC) at 25% is +0.0112 with cluster interval [+0.0005, +0.0206] and group interval [+0.0012, +0.0203]. The L17a prior was that the interval includes 0. It does not: the lower bound is above 0 by 0.0005 (cluster) and 0.0012 (group). This is a marginal exclusion from one rollout sample and 75 bucket clusters. The K = 4 half of L17a stays open.

**Checks and caveats.**
1. Exit codes 0 for the readout, the control and the check. No traceback, warning or NaN in readout.log, control.log or check.log.
2. The control matches all shared L14b arms and draws (914 values, 0 mismatches at 1e-9), including the static reference at 938.0 caught.
3. The independent check refits the static models in separate code and recomputes all draws with pandas resampling: 4,023 values, max abs diff 0.0.
4. Bootstrap budgets are fixed at 1,079 and 2,159 docs while the resampled pool size varies, inherited from L14b and not changed here.
5. The static models use fit-quarter rows only (25 held-out buckets). Proxy labels enter the static arms only through the fit, and the evaluation uses the same proxy_error labels as every other arm.
6. Scope: one rollout sample (seed 0), K = 8 majority proxy, Qwen2.5-0.5B-Instruct, GSM8K, G 8, MiniLM refit features (inherited), cell100 buckets, 75 sim buckets, logistic static models, one threshold rule, 8 tie-break seeds, 200 resamples per interval. Labeled learners only.

Outputs: avrl/runs/L17d/{result.json, readout.log, control.json, control.log, check.json, check.log, exit_code, done.flag}. Scripts: avrl/exp/l17d_static_ablation.py, avrl/exp/L17d_control.py, avrl/exp/L17d_check.py, avrl/exp/L17d.sh.

**Critic note (L17d, 2026-10-19).** Written by the agent that ran the item, against the raw files. Verdict under the pre-registered rule: INCONCLUSIVE, re-derived from result.json (point Delta -0.00892 at 25% and +0.00211 at 50%; primary cluster lower bounds -0.0166 and -0.0075). The control passed (914 values, 0 mismatches), so this is a verdict and not a stop. What I checked: (a) every number in both tables matches result.json and check.json; (b) the pandas check reproduces the point caught counts, the point Delta values, and the 400 cluster and 400 group draws for each of the five metrics at each budget, max abs diff 0.0; (c) the budgets are equal across arms (431, 1,079 and 2,159 docs, recorded in each arm's budget field); (d) the L14b script md5 is unchanged; (e) no em dashes in any new file or log text.

Qualifications, none changes the verdict. (1) The prior in the pre-registration was KILL. The outcome is INCONCLUSIVE because the 50% point is +0.0021; the verdict follows the written rule, and it should not be read as a kill. (2) "The cheap six match the full model" would need an equivalence margin, and none was pre-registered. At this precision, neither "the cheap six match" nor "the PCs help" is supported. (3) The group intervals ignore bucket clustering, so they cannot narrow the verdict. (4) The 25% exclusion for the labeled cheap plus PC arm is marginal and descriptive; it should not be promoted to a claim. (5) The labeled cheap-six lead (D +0.020 at 25%, D +0.017 at 50%, both intervals excluding 0) rests on one pool, one rollout sample and one fixed threshold rule. It is a lead for the K = 4 half of L17a and a rollout-resampling test, neither of which is pre-registered here. (6) The pre-registration was appended to this ledger before avrl/exp/l17d_static_ablation.py was created, in this session's command order.

Corrected verdict: INCONCLUSIVE. Claim 3 for labeled learners on this pool: not killed and not supported. The labeled static arm's margin over the threshold at 25% is carried by the cheap six, with the PCs adding no detectable margin at 25% and an undetermined effect at 50%. Scope: Qwen2.5-0.5B-Instruct, GSM8K, G 8, K = 8 majority proxy, one rollout sample at seed 0, MiniLM refit features, one threshold rule, 8 tie-break seeds, 200 resamples per interval.

## L17d 2026-10-19 inconclusive

**Hypothesis.** As pre-registered in the "L17d pre-registration" entry above: on the L14b sim pool, the labeled static margin at 25% comes from the six cheap columns, and the cheap six alone match cheap plus the 16 prompt-space PCs. Kill: point Delta at or below 0 at both 25% and 50%, with Delta = admitted(static_cheap6) - admitted(static).

**What was run.** This entry is an orchestrator recount of the run recorded in the "L17d 2026-10-19 inconclusive (claim 3 for labeled learners ...)" entry above, which holds the full tables, controls and Critic note. CPU only, no model run in this step. Pool: avrl/runs/L14b/pool_feats.parquet (Qwen2.5-0.5B-Instruct, GSM8K, G 8, K = 8 majority proxy, cell100). Sim pool 4,318 groups in 75 buckets with 1,906 proxy errors. Budgets 1,079 docs (25%) and 2,159 docs (50%). Three static logistic models fit on the seed-0 fit quarter (25 buckets, labels used only there). Threshold is the mean of 8 tie-break seeds. Intervals: 200 bucket-cluster resamples (CLUSTER_SEED 123, primary) and 200 group resamples (BOOT_SEED 0, secondary). Inputs re-read: avrl/runs/L17d/result.json, readout.log, control.json, check.json, exit_code (0), done.flag.

**Recount against the raw files.**
- admitted = 1 - caught / 1,906 recomputed for every arm from the stored caught means; all match to four decimals.
- Point Deltas recomputed from the caught counts: Delta(cheap6 - cheap+PC) at 25% -0.008919 and at 50% +0.002099; Delta_pc at 25% +0.178909; D(static cheap+PC) at 25% +0.011215 and at 50% +0.019084. All match result.json and readout.log.
- Every primary and secondary interval and frac > 0 for the five metrics at 25% and 50% recomputed from the stored 200-draw arrays (numpy percentile, linear): all match to four decimals.
- Control: the L14b arms in avrl/runs/L14b/result.json (static caught 938.0 and threshold 916.625 at 25%, D(ts_online) -0.0767 with cluster interval [-0.0958, -0.0651], D(greedy_online) -0.0115 with [-0.0225, -0.0034]) match what L17d reports. avrl/exp/l14b_claim2_readout.py md5 is still 2117be0d85dbcb043018c1b618f70bd7.
- Cutoff ties: 0 for every static arm at every budget in result.json, so the argsort order does not move the caught counts.
- Not done here: the static models were not refit. The check.json "independent" pandas check was written by the same run step, so it is an internal check and not a second implementation.

**Results.** Point estimates; intervals are primary bucket-cluster 95% and secondary group 95%.

| metric | budget | point | cluster 95% CI | group 95% CI |
|---|---|---|---|---|
| Delta = adm(cheap6) - adm(cheap+PC) (decision) | 25% | -0.0089 | [-0.0166, +0.0020] | [-0.0158, +0.0005] |
| Delta (decision) | 50% | +0.0021 | [-0.0075, +0.0106] | [-0.0052, +0.0103] |
| D(static cheap+PC) = adm(threshold) - adm(static) | 25% | +0.0112 | [+0.0005, +0.0206] | [+0.0012, +0.0203] |
| D(static cheap+PC) | 50% | +0.0191 | [+0.0044, +0.0281] | [+0.0079, +0.0280] |
| Delta_pc = adm(pc) - adm(cheap+PC) (descriptive) | 25% | +0.1789 | [+0.1527, +0.2007] | [+0.1624, +0.1956] |

Admitted baselines (point, 25% / 50%): uniform 0.751 / 0.505, threshold 0.519 / 0.178, static cheap+PC (labeled reference) 0.508 / 0.159, static cheap6 (labeled) 0.499 / 0.161, static pc (labeled, descriptive) 0.687 / 0.420.

**Verdict against the pre-registered rules.** INCONCLUSIVE. PASS needs point Delta > 0 with a primary interval above 0 at 25% or 50%; neither budget meets it (the 25% point is negative, the 50% interval includes 0). KILL needs point Delta at or below 0 at both budgets; the 50% point is +0.0021, so KILL does not apply. The 25% point is in the kill direction, but the written rule does not turn that into a kill.

**What it changes about the thesis claims.**
- Claim 3 (the labeled edge sits in prompt-space structure): not supported for labeled learners on this pool. At 25% the cheap six admit 0.499 against 0.508 for cheap plus PCs, and the PC-only arm admits 0.687 against 0.519 for the threshold, so the prompt-space features carry little of the margin. At 50% the Delta interval allows a PC margin of up to about 0.011, so a small PC effect is not excluded there. Not a kill under the written rule.
- Claim 2 (a learned allocator admits fewer corrupted rewards than any fixed rule at equal budget): supported for labeled learners on this pool, marginally; not supported for label-free learners, which stay killed on this pool (L14b, L16a, L16b). The labeled static arm beats the threshold at both budgets with the cluster interval above 0: D +0.011 at 25% (lower bound +0.0005) and D +0.019 at 50% (lower bound +0.0044). The 50% exclusion is not stated in the run entry's claim paragraph, and it is the more robust of the two.

**Caveats on the intervals.** The intervals resample the sim pool only. The static models are fixed, and each is fit on one fit quarter (seed 0, 25 buckets), so the choice of fit quarter is in no interval. The rollout sample is also single (seed 0). The 25% lower bound of +0.0005 is therefore not robust evidence on its own. The L18 follow-ups in lit/avrl/EXPERIMENT_BACKLOG.md address both gaps.

**Critic note (verification, 2026-10-19).** Verdict re-derived from result.json: INCONCLUSIVE, the same as the run entry. The numbers in the run entry's tables match the raw files. Not checked: the static refits from the parquet (only the same-step internal check covers them), and whether another fit quarter keeps the 25% sign. The run entry's claim paragraphs are consistent with the numbers except that the 50% labeled-arm exclusion is not mentioned there. No em dashes in this entry.

**Follow-ups** (none run; each needs its own pre-registration before it runs; listed in lit/avrl/EXPERIMENT_BACKLOG.md as L18a, L18b and L18c).
1. L18a (CPU, descriptive): fit-quarter sensitivity of the labeled static arms over 20 bucket-shuffle seeds. Information: high, because the 25% margin's interval omits the fit split. Prior: the 25% margin changes sign on a sizable share of fit quarters.
2. L18b (one GPU job through gpu_run.sh): rollout-sampling replicate of the L17d ablation on the L14b prompts with a fresh sampling seed, then L17d unchanged. Information: high for claim 2 (labeled lead) and claim 3, since it is the only test of the single-rollout caveat. Prior: no PC margin at 25% again, and the labeled D over the threshold is the more fragile reading.
3. L18c (CPU, descriptive): low-dimensional prompt-space check, cheap six plus pc0 to pc3 and plus pc0 to pc7, with the L17d bucket-cluster bootstrap. Information: low to medium. Prior: kill, since the 16-PC arm already shows no 25% margin. Any "match" claim needs an equivalence margin pre-registered first; the current 25% interval [-0.017, +0.002] is too wide to show equivalence at +/-0.005.

**Critic note (L17d, adversarial review, 2026-10-19).** Corrected verdict: INCONCLUSIVE, unchanged. Corrections below concern claim status and wording, not the verdict. Checked against result.json, readout.log, control.json, check.json, the three scripts, pool_feats.parquet, the L14b result.json and md5, and events/snapshot.log. Recomputed on CPU with my own code, nothing written to disk. (a) Verdict: stored verdict is INCONCLUSIVE; point Delta(cheap6 - cheap+PC) is -0.00892 at 25% (1,079 docs; caught 955 against 938) and +0.00210 at 50% (2,159 docs; caught 1,599 against 1,603, a 4-document gap), and the 50% primary lower bound is -0.0075. PASS needs a positive point with the lower bound above 0 and KILL needs a point at or below 0 at both budgets, so the written rule gives INCONCLUSIVE. (b) Numbers: every figure in both result tables (caught, admitted, point Delta values, all primary and secondary intervals, frac > 0) matches result.json and readout.log; the 2.5 and 97.5 percentiles and frac > 0 recomputed from the stored 200-draw arrays match for all 20 interval rows; cutoff_ties is 0 for the three static arms at 10, 25 and 50%; control count 914 and check count 4,023 reproduce from the scripts (4,023 = 23 point values + 5 metrics x 2 budgets x 2 interval types x 200 draws); readout.log has no traceback, warning or NaN; wall_seconds 184.4. (c) My own numpy refit of the three static logistic models on the seed-0 fit quarter reproduces caught 938 / 955 / 597 at 25% and 1,603 / 1,599 / 1,106 at 50%, the 8-seed threshold means 916.625 and 1,566.625, and all 200 stored cluster draws of Delta and D(static cheap+PC) at both budgets with max abs diff exactly 0. This is a second implementation beyond L17d_check.py, which copies the readout's split rule and rng-stream convention, so it tests the pandas against the numpy code and not the design; the first entry calls it independent and the second entry calls it internal, and the accurate wording is that the numbers are reproduced and the design is shared. (d) No leakage: the 22 columns are the six cheap columns and pc0 to pc15, none derived from gold (cheap columns from rollout answers and token counts, PCs and cell100 from question text, PCA and KMeans fit without labels); proxy_error equals (majority != gold) on all 5,792 rows; fit and sim share no bucket and no (split, pid) (25 buckets and 1,474 groups against 75 buckets and 4,318 groups); labels reach only the three static fits, on the fit quarter. (e) Budget accounting is equal for the L17d decision: all arms use int(bf x 4,318) docs and the Delta comparison uses the same fit labels for both arms. L14b script md5 is still 2117be0d85dbcb043018c1b618f70bd7; the pre-registration was in the ledger at 1,850 lines at 02:24:26, before l17d_static_ablation.py was created at 02:26:03. No em dashes in the L17d scripts, logs or ledger entries.

Corrections. (1) The second L17d entry's "Claim 2 ... supported for labeled learners on this pool, marginally" is not licensed by L17d and contradicts the pre-registration and the first entry. The pre-registration says D(static) and the other descriptive metrics cannot change the verdict and sets no claim 2 rule, and the first entry's qualification (4) says the 25% exclusion should not be promoted to a claim. The support is also thinner than "supported" implies. The 25% lower bound is +0.0005 from 200 draws. Two fresh 2,000-draw bucket-cluster replicates (seeds 2024 and 7) give D(static cheap+PC) at 25% lower bounds of +0.00013 and +0.00066 (2.5% and 1.9% of draws at or below 0), so the exclusion survives but sits within bootstrap noise of 0; at 50% the replicates give +0.0047 and +0.0048 against +0.0044, which is firmer. The intervals omit the fit-quarter choice (one quarter of 25 buckets) and the rollout sample (one, seed 0). The static arm has 1,474 labeled fit-quarter groups outside the audit budget, while ts_online and greedy_online pay for their labels inside it, so "at equal budget" holds for audits and not for labels. Only one fixed rule, the agreement then n_distinct threshold, was compared, so "any fixed rule" is untested. Corrected status: claim 2 for labeled learners is an open lead (D +0.011 at 25%, +0.019 at 50% over the threshold), not supported by L17d; label-free learners stay killed on this pool. L18a and L18b are the tests that can promote it. (2) The first entry's "The L17a prior was that the interval includes 0. It does not" should read: the K = 8 half of L17a has a cluster lower bound of +0.0005 (200 draws) and +0.0001 to +0.0007 (2,000 draws), at the edge of 0, descriptive only. (3) The first entry's Critic note (b) says "the 400 cluster and 400 group draws for each of the five metrics at each budget". It is 200 cluster and 200 group draws for each metric at each budget (the same 200 resamples serve both budgets). (4) A fact that sharpens the INCONCLUSIVE reading: at 25% the Delta cluster interval is [-0.0166, +0.0020] with 4% of draws above 0 (8 of 200), and in the 2,000-draw replicates the upper bound is 0.0000 and +0.0005, so the 25% data lean toward the PCs not helping, but the 50% interval [-0.0075, +0.0106] with 63% of draws above 0 does not separate a match from a +/-0.01 effect, and no equivalence margin was pre-registered. The "cheap six match" statement is not supported. (5) Fragility of the label: the 50% point Delta is a 4-document difference (1,603 caught against 1,599 of 1,906), so INCONCLUSIVE would become KILL if the cheap six caught 4 more or the 22-column arm 4 fewer at 50%; the verdict is rule-correct and fragile, and L18a (fit-quarter sensitivity) is the right way to see whether the sign at 50% is stable. (6) The ledger has two L17d result sections (the second is an orchestrator recount); they agree on every number and on the verdict, and corrections (1) to (3) apply to both.
