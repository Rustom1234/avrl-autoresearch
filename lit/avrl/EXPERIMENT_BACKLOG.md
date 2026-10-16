# AVRL Experiment Backlog

Pre-registered, 2026-09-30. Twenty experiments, E01 through E20, ordered by expected information value per GPU hour: cheapest, highest-leverage checks first, most expensive and most infrastructure-blocked last. This ordering means analysis-only reruns of already-collected data (near-zero marginal GPU cost) come before any new generation run, and new generation runs on one RTX 5090 come before any GRPO training run on an A100.

## Claim numbering (from PROJECT.md)

- Claim 1: the learned allocator reaches the same policy quality with a fraction of expensive verifier calls.
- Claim 2: the learned allocator admits fewer corrupted rewards into training than any fixed rule at the same budget.
- Claim 3: the allocator's edge comes from structure in prompt space (proxy error clusters), the SOFisher condition (Li et al., 2026, Nature Communications, https://doi.org/10.1038/s41467-026-73404-6).

Source for the existing harness read this session: /homes/rdubash/active-audit/avrl/PROJECT.md, /homes/rdubash/active-audit/avrl/p0_rollouts.py, /homes/rdubash/active-audit/avrl/p0_analyze.py, /homes/rdubash/active-audit/avrl/allocators.py. Confirmed at read time: avrl/p1_grpo.py (the phase-1 GRPO training harness named in PROJECT.md's Files section) does not exist on disk yet, only p0_rollouts.py, p0_analyze.py, and allocators.py do. Every experiment below that needs an actual policy-training loop is marked NEEDS PHASE-1 HARNESS.

## Runtime method note

p0_rollouts.py's own run.log (avrl/runs/p0/run.log, in progress at read time) shows 3,000 GSM8K prompts times G=8 rollouts (24,000 completions) on Qwen2.5-1.5B-Instruct at max_tokens=512 reaching 47% of prompts in about 7.5 minutes, aggregate throughput near 2,460 input tok/s and 5,470 output tok/s on the RTX 5090. Extrapolating linearly gives roughly 16 minutes of generation for that exact config, leaving headroom for the embedding plus allocation-replay analysis step under a 40-minute cap. All runtime estimates below for new model or prompt-count combinations are extrapolations from that single observed data point, not measurements, and are flagged as estimates. Before committing a full run, a 100-prompt dry run with the same --model and --G is recommended to calibrate tokens/sec for that specific model before trusting the 40-minute estimate.

## Dataset identifiers

GSM8K is already local as avrl/data/gsm8k_train.parquet / gsm8k_test.parquet (origin not stated in the scripts read this session; standard HF path is openai/gsm8k, https://huggingface.co/datasets/openai/gsm8k, UNVERIFIED against the local parquet's actual provenance). MATH: HuggingFaceH4/MATH-500 confirmed this session via WebFetch, https://huggingface.co/datasets/HuggingFaceH4/MATH-500, 500 test-split problems curated from OpenAI's MATH benchmark. MBPP: google-research-datasets/mbpp, https://huggingface.co/datasets/google-research-datasets/mbpp (UNVERIFIED this session, standard well-known identifier, not independently refetched). HumanEval: openai/openai_humaneval, https://huggingface.co/datasets/openai/openai_humaneval (UNVERIFIED this session, same caveat). Code-task experiments (E08, E14, E20) need a sandboxed test-execution step that does not exist in the current harness; p0_rollouts.py's answer extraction is GSM8K-specific (`####` regex) and would need a replacement extractor per task.

---

## E01: Baseline proxy-error structure check (in flight)

**Hypothesis:** cheap-proxy (majority-vote) error on GSM8K has exploitable structure in prompt-embedding space, concentrated even among confident (agreement >= 0.75) groups.

**Change vs existing scripts:** none. This is the run already executing per avrl/runs/p0/run.log: Qwen2.5-1.5B-Instruct, GSM8K train, n_prompts=3000, G=8, max_tokens=512, seed=0.

**Command shape:** `python avrl/p0_rollouts.py --n_prompts 3000 --G 8 --seed 0 --out avrl/runs/p0` then `python avrl/p0_analyze.py` (matches avrl/p0_chain.sh as read this session).

**Metric:** proxy_error_when_unanimous vs proxy_error_when_split (summary.json); auc_cheap vs auc_cheap_emb on the confident subset (result.json); neighbor_ratio at cell100/cell300 (result.json).

**Pass threshold:** confident-subset proxy_error_rate >= 3% AND auc_cheap_emb exceeds auc_cheap by >= 0.05 AND neighbor_ratio >= 1.5x, per PROJECT.md's own pre-registered read.

**Kill threshold:** confident-subset proxy_error_rate < 3% (budgeting moot on this task) OR (embedding AUC gain < 0.05 AND neighbor_ratio < 1.5x), in which case claim 3 is dead on this config and only claims 1-2 remain, to be won or lost by a fixed rule.

**Expected runtime:** ~16 min generation + ~5-10 min analysis, RTX 5090, per the run.log extrapolation above.

**Informs:** claim 3 (foundational precondition for all other experiments).

---

## E02: Budget-sweep extension (analysis-only rerun)

**Hypothesis:** the gap between the learned allocator and the threshold allocator (proxy errors caught per verifier call) is not flat across budgets; it is largest at low budgets, where a fixed threshold's single cut point is most information-starved.

**Change vs existing scripts:** no new rollouts. Reuse E01's feats.parquet. Edit p0_analyze.py's sim_alloc invocation: replace `BUDGET_FRACS = 3, 1, 8, (0.10, 0.25, 0.50)` with a finer grid, e.g. (0.02, 0.05, 0.10, 0.15, 0.20, 0.25, 0.35, 0.50, 0.75, 0.90), for both cell100 and cell30 grouping and both cheap and cheap+emb feature sets.

**Command shape:** `python avrl/p0_analyze.py` with the BUDGET_FRACS substitution above applied to the sim_generic.py template before it runs (same replace-and-write pattern p0_analyze.py already uses on `runs/sim_generic.py`).

**Metric:** hit rate (proxy errors caught per verifier call) for random/threshold/learned/oracle at each budget point; plot hit-rate-vs-budget curves per allocator.

**Pass threshold:** learned allocator's hit-rate advantage over threshold is >= 10 percentage points at at least one budget point in the 5-25% range (the range PROJECT.md's phase-1 plan actually intends to spend, 25%).

**Kill threshold:** learned and threshold hit-rate curves are within 3 percentage points of each other at every budget point tested; a single fixed threshold is budget-invariant-equivalent to the learned allocator on this data.

**Expected runtime:** under 10 min, RTX 5090 (CPU-bound analysis step only, no new GPU generation).

**Informs:** claims 1 and 2 (identifies the budget regime where a learned allocator is worth building at all).

---

## E03: Allocation-policy comparison with bootstrap significance

**Hypothesis:** at the budget PROJECT.md's phase 1 plans to spend (25%), the learned allocator admits fewer corrupted-reward groups than the threshold allocator by a margin that survives resampling noise, not just as a point estimate.

**Change vs existing scripts:** no new rollouts. Reuse E01's feats.parquet. Add a bootstrap loop (resample groups with replacement, 200-1000 resamples) around the existing sim_alloc.py replay at budget=0.25, computing the corrupted-reward-admitted rate (fraction of proxy_error=1 groups NOT selected by the allocator) for threshold and learned on each resample.

**Command shape:** wraps `sim_alloc.py` (invoked the way p0_analyze.py already does, with feats.parquet, label proxy_error, group cell100, budget 0.25) inside a bootstrap driver script; no change to sim_alloc.py's core logic.

**Metric:** corrupted-reward-admitted rate for threshold vs learned, with a bootstrap 95% CI on the difference.

**Pass threshold:** the 95% CI on (threshold_corrupted_rate - learned_corrupted_rate) excludes zero and the point estimate is positive (learned admits fewer corrupted rewards).

**Kill threshold:** the 95% CI on the difference includes zero, i.e. the apparent edge in E01/E02 is not distinguishable from resampling noise at this sample size.

**Expected runtime:** under 10 min, RTX 5090 or CPU-only (no GPU needed for this step, listed for continuity with the 5090 budget).

**Informs:** claim 2 directly, with the statistical rigor claim 2's "fewer corrupted rewards ... than any fixed rule" wording demands.

---

## E04: Proxy type variation, small judge model

**Hypothesis:** a small LLM judge's error pattern (which rollouts it mis-scores) is structured in prompt-embedding space similarly to majority-vote's error pattern, but the two proxies are wrong on different, only partially overlapping subsets of prompts.

**Change vs existing scripts:** reuse E01's already-generated rollouts.parquet (no new policy generation needed). New step: score every rollout's answer correctness using Qwen2.5-0.5B-Instruct as a judge (binary correct/incorrect, given the question and the rollout's extracted answer, gold-free), producing a judge-majority label per group analogous to Gd.majority in p0_rollouts.py. Feed judge_error (judge-majority != gold) into p0_analyze.py's structure pipeline in place of proxy_error.

**Command shape:** new script `avrl/p0_judge_score.py --rollouts avrl/runs/p0/rollouts.parquet --judge_model Qwen2.5-0.5B-Instruct --out avrl/runs/p0_judge/judge_scores.parquet` (does not exist yet, small addition to the existing harness), then p0_analyze.py's structure block rerun with y = judge_error instead of y = G.proxy_error.

**Metric:** judge_error_rate overall and on judge-confident subset; auc_cheap_emb for judge_error; Jaccard overlap between {groups where majority-vote is wrong} and {groups where judge is wrong}.

**Pass threshold:** judge_error has auc_cheap_emb - auc_cheap >= 0.05 (structure also present for this proxy type) AND Jaccard overlap with majority-vote errors < 0.5 (the two proxies fail on different prompts, meaning proxy choice itself is a lever, not just budget).

**Kill threshold:** Jaccard overlap > 0.8 (judge and majority-vote fail on essentially the same prompts; proxy-type variation adds no new signal beyond E01).

**Expected runtime:** judge scoring pass on 3,000 x 8 = 24,000 already-generated completions with a 0.5B model, estimated 5-10 min generation-equivalent inference on the 5090 (estimate, not measured), plus the same ~10 min analysis step as E01. Estimated total under 25 min.

**Informs:** claim 3, and the allocator-design question of which proxy type an allocator should be built around.

---

## E05: Proxy type variation, self-verification

**Hypothesis:** asking the policy model to re-derive and check its own answer (self-verification) produces a different, and on average less reliable, error-structure signal than peer majority-vote, because self-verification errors correlate with the same reasoning failure that produced the original wrong answer rather than being independent.

**Change vs existing scripts:** reuse E01's rollouts.parquet for the original answers. New generation pass: for each of the 3,000 x 8 rollouts, prompt the same policy (Qwen2.5-1.5B-Instruct) to independently re-solve the problem and state whether it agrees with the stored answer (yes/no), without showing its own prior chain of thought. Proxy label = self-verification says "agrees" AND self-verification's own re-derived answer matches the original.

**Command shape:** new script `avrl/p0_selfverify.py --rollouts avrl/runs/p0/rollouts.parquet --model Qwen2.5-1.5B-Instruct --out avrl/runs/p0_selfverify/` (does not exist yet), then the same p0_analyze.py structure block with y = self_verify_error.

**Metric:** self_verify_error_rate vs majority-vote proxy_error_rate on the same groups; auc_cheap_emb for self_verify_error; correlation between self_verify_error and proxy_error at the group level.

**Pass threshold:** self_verify_error_rate is measurably different from proxy_error_rate (>= 3 percentage points) in either direction, confirming self-verification is a distinct proxy worth allocator attention rather than a redundant copy of majority vote.

**Kill threshold:** self_verify_error correlates with proxy_error at Pearson r >= 0.9 (self-verification adds no information beyond majority vote; not worth building a separate allocator arm for it).

**Expected runtime:** one extra generation pass over 24,000 short self-verification prompts, estimated 10-15 min on the 5090 (estimate), plus ~10 min analysis. Estimated total under 30 min.

**Informs:** claim 3 (third proxy type, completing the proxy-type variation required by the task brief).

---

## E06: Model size variation, small end

**Hypothesis:** proxy-error structure (claim 3) is present but the overall proxy-error rate is higher for a weaker policy, since majority-vote reliability degrades as the underlying accuracy distribution flattens.

**Change vs existing scripts:** --model Qwen2.5-0.5B-Instruct in place of Qwen2.5-1.5B-Instruct, everything else identical to E01 (n_prompts=3000, G=8, max_tokens=512, seed=0).

**Command shape:** `python avrl/p0_rollouts.py --model <path-to-Qwen2.5-0.5B-Instruct> --n_prompts 3000 --G 8 --seed 0 --out avrl/runs/p0_small` then p0_analyze.py pointed at that --out directory (p0_analyze.py's hardcoded P path needs a --run_dir argument added, a small change to the existing script).

**Metric:** same structure metrics as E01 (confident-subset proxy_error_rate, auc_cheap_emb, neighbor_ratio), compared side by side with E01's numbers.

**Pass threshold:** confident-subset proxy_error_rate at 0.5B is higher than at 1.5B (E01) by >= 2 percentage points, consistent with a weaker model producing a noisier, more exploitable proxy signal, and structure (auc gain, neighbor ratio) still clears E01's thresholds.

**Kill threshold:** structure metrics fail E01's thresholds at this model size even though E01 passed at 1.5B, meaning claim 3 is size-dependent and the allocator's value is narrower than hoped.

**Expected runtime:** a 0.5B model should generate faster than the 1.5B baseline; estimated 8-12 min generation (estimate, extrapolated down from the 1.5B throughput in run.log) plus ~10 min analysis. Estimated total under 25 min.

**Informs:** claim 3, model-size axis required by the task brief.

---

## E07: Model size variation, large end

**Hypothesis:** as policy accuracy rises toward ceiling on GSM8K, majority-vote proxy-error rate among confident groups falls, shrinking the population an allocator has anything to find, so claim 3's practical value shrinks with model capability on an easy task.

**Change vs existing scripts:** --model Qwen2.5-7B-Instruct (ungated on Hugging Face), n_prompts reduced to 1000 (from 3000) to hold runtime under 40 minutes given the larger model's slower decode, G=8, max_tokens=512, seed=0, --out avrl/runs/p0_large.

**Command shape:** `python avrl/p0_rollouts.py --model <path-to-Qwen2.5-7B-Instruct> --n_prompts 1000 --G 8 --seed 0 --out avrl/runs/p0_large` then p0_analyze.py against that --out directory.

**Metric:** same structure metrics as E01, plus rollout_accuracy and majority_accuracy compared to E01 to confirm the "closer to ceiling" premise actually holds.

**Pass threshold:** majority_accuracy (E07) > majority_accuracy (E01) by >= 10 percentage points AND confident-subset proxy_error_rate (E07) < confident-subset proxy_error_rate (E01), confirming the expected ceiling effect, with structure metrics still measurable (even if the base rate is lower).

**Kill threshold:** confident-subset proxy_error_rate at 1000 prompts is 0 or too small a count (fewer than ~10 error cases) to fit any structure model; in that case this model/task pairing cannot inform claim 3 at all and a harder task (MATH, E09) is the only way to test claim 3 at this model size.

**Expected runtime:** UNVERIFIED estimate; a 7B model is roughly 4-5x the parameter count of the 1.5B baseline and decode throughput typically scales down less than linearly with parameter count on a single GPU, so 1000 prompts (one third of E01's prompt count) at proportionally slower per-token speed is estimated at 20-35 min generation plus ~10 min analysis. A 100-prompt dry run is strongly recommended before committing the full budget, per the runtime method note above.

**Informs:** claim 3, model-size axis (large end), and directly tests the risk noted in SYNTHESIS.md section 4 that a cheap proxy alone may already capture most of the achievable gain as models get stronger.

---

## E08: Task variation, code (MBPP)

**Hypothesis:** on a code task, a cheap self-consistency proxy (agreement of generated code across G samples, by output on the one visible test case rather than full execution) has different, likely weaker, structure than GSM8K's numeric-answer majority vote, because code correctness is not well summarized by superficial agreement the way a single number is.

**Change vs existing scripts:** dataset swapped from GSM8K to MBPP; answer extraction replaced (no `####` marker in code tasks); cheap proxy redefined as agreement across G generated solutions using the MBPP problem's one visible/sample test case as a cheap check, not full execution; expensive verifier redefined as running the model's generated code against MBPP's full hidden test suite (this needs a sandboxed executor that does not exist in the current harness, a real new engineering cost, not just a flag change). Model: Qwen2.5-Coder-1.5B-Instruct (ungated coder-specialized variant) or Qwen2.5-1.5B-Instruct for a same-model-different-task comparison against E01; recommend running both if time allows, else prioritize Qwen2.5-Coder-1.5B-Instruct for a realistic code-task baseline.

**Command shape:** new script `avrl/p0_rollouts_code.py --model <path> --task mbpp --n_prompts <MBPP size, ~500-1000 test-split problems> --G 8 --out avrl/runs/p0_code` (does not exist yet; largest single new-code item among E01-E08).

**Metric:** same structure metrics as E01, computed against true test-suite-execution correctness as the gold label and cheap-sample-agreement as the proxy label.

**Pass threshold:** confident-subset proxy_error_rate >= 3% and auc_cheap_emb gain >= 0.05 over cheap-only, mirroring E01's thresholds, confirming claim 3 generalizes off math tasks.

**Kill threshold:** same numeric failure condition as E01 (confident-subset proxy_error_rate < 3% or AUC gain < 0.05 and neighbor ratio < 1.5x), in which case claim 3 is math-specific evidence only and the code-task story needs a different cheap proxy (e.g. static-analysis heuristics) before being tested again.

**Expected runtime:** UNVERIFIED estimate; MBPP's test split is roughly 500 problems (exact count not independently confirmed this session), smaller than GSM8K's 3,000-prompt draw, but code generations often run longer than GSM8K's 512-token cap and the sandboxed-execution step adds CPU-bound overhead not present in E01. Estimated 15-25 min generation plus 5-15 min execution and analysis. A dry run on 50 problems is recommended first given the new, unexercised code path.

**Informs:** claim 3, task-variation axis (math vs code) required by the task brief.

---

## E09: Task variation, harder math (MATH-500)

**Hypothesis:** MATH-500's higher difficulty and more heterogeneous problem types (algebra, geometry, number theory, precalculus, per the confirmed dataset description) produce a higher confident-subset proxy-error rate than GSM8K, giving an allocator more to find.

**Change vs existing scripts:** dataset swapped from GSM8K to HuggingFaceH4/MATH-500 (https://huggingface.co/datasets/HuggingFaceH4/MATH-500, confirmed this session); answer extraction changed from the `####` regex to MATH's boxed-answer convention (`\boxed{...}`), a small parsing change, not a structural harness change; model held at Qwen2.5-1.5B-Instruct to isolate the task-difficulty effect from the model-size effect already covered in E06/E07.

**Command shape:** `python avrl/p0_rollouts.py --model <path-to-Qwen2.5-1.5B-Instruct> --data math500 --n_prompts 500 --G 8 --out avrl/runs/p0_math500` (MATH-500's test split is only 500 rows total per the confirmed dataset page, so n_prompts is capped at 500, not the 3,000 used for GSM8K; requires a `--data` flag and boxed-answer parser that do not exist in the current p0_rollouts.py).

**Metric:** same structure metrics as E01, plus a direct rollout_accuracy comparison against E01 to confirm MATH-500 is in fact harder for this model.

**Pass threshold:** rollout_accuracy (MATH-500) < rollout_accuracy (E01, GSM8K) by >= 15 percentage points (confirms the difficulty premise) AND confident-subset proxy_error_rate clears E01's 3% bar with a higher point estimate than E01's.

**Kill threshold:** confident-subset proxy_error_rate on MATH-500 is statistically indistinguishable from GSM8K's (E01), meaning task difficulty alone does not move the exploitable-structure signal and claim 3's evidence base stays limited to GSM8K-like tasks.

**Expected runtime:** with only 500 prompts (a sixth of E01's prompt count) generation should be well under E01's ~16 minutes even accounting for MATH's typically longer chains of thought; estimated 10-20 min generation plus ~5 min analysis (fewer groups means faster KMeans/logistic regression fitting than E01's 3,000-group run).

**Informs:** claim 3, rounds out the task-variation requirement together with E08.

---

## E10: Cross-family model check (Llama 3.2)

**Hypothesis:** the structure found in E01 is not a Qwen-family artifact; a same-scale Llama model shows qualitatively similar confident-subset proxy-error structure on GSM8K.

**Change vs existing scripts:** --model Llama-3.2-3B-Instruct (ungated on Hugging Face) in place of Qwen2.5-1.5B-Instruct, n_prompts=2000 (reduced from 3000 as a margin against this being a roughly 2x larger model than the E01 baseline), G=8, max_tokens=512, seed=0.

**Command shape:** `python avrl/p0_rollouts.py --model <path-to-Llama-3.2-3B-Instruct> --n_prompts 2000 --G 8 --seed 0 --out avrl/runs/p0_llama` then p0_analyze.py against that --out directory.

**Metric:** same structure metrics as E01, compared side by side.

**Pass threshold:** confident-subset proxy_error_rate >= 3% and auc_cheap_emb gain >= 0.05 over cheap-only, same bar as E01, confirming cross-family generalization.

**Kill threshold:** structure metrics fail E01's bar specifically on Llama while Qwen passed, narrowing claim 3's evidence to one model family and raising the prior that the finding is an artifact of Qwen's training data or tokenizer rather than a general property of proxy-error structure.

**Expected runtime:** UNVERIFIED estimate; roughly 2x the 1.5B baseline's parameter count at two-thirds the prompt count, estimated 15-25 min generation plus ~10 min analysis. Dry run recommended.

**Informs:** claim 3, external-validity check.

---

## E11: Allocator hyperparameter ablation (analysis-only)

**Hypothesis:** LearnedAllocator's two exposed hyperparameters, refit_every and prior_strength (avrl/allocators.py, lines 115-121 as read this session), materially change hit-rate at fixed budget, so the phase-1 GRPO comparison (E15) should not commit to the current defaults (refit_every=20, prior_strength=2.0) without a sensitivity check.

**Change vs existing scripts:** no new rollouts. Reuse E01's feats.parquet. Grid over refit_every in {5, 20, 50, 100} and prior_strength in {0.5, 2.0, 5.0} inside the allocation-replay step (sim_alloc.py or a thin wrapper around allocators.LearnedAllocator), at a fixed budget of 25%.

**Command shape:** a new small driver script instantiating `allocators.LearnedAllocator(refit_every=r, prior_strength=p)` for each grid cell against E01's feats.parquet, reusing the replay logic already exercised by p0_analyze.py's sim_alloc.py calls.

**Metric:** hit rate (proxy errors caught per verifier call) at budget=0.25 for each of the 12 grid cells.

**Pass threshold:** the best grid cell beats the current default (refit_every=20, prior_strength=2.0) by >= 5 percentage points of hit rate, justifying a hyperparameter change before funding E15's GRPO run.

**Kill threshold:** all 12 grid cells are within 3 percentage points of each other; the allocator is insensitive to these hyperparameters in this regime and the current defaults can be used as-is for E15 without further tuning spend.

**Expected runtime:** under 15 min total, RTX 5090 or CPU-only (12 cheap replay passes over already-computed features, no new GPU generation).

**Informs:** claims 1 and 2 (de-risks the hyperparameter choice before the expensive GRPO run).

---

## E12: Embedding/feature ablation (analysis-only)

**Hypothesis:** the structure signal found in E01 is not an artifact of the specific sentence-transformers/all-MiniLM-L6-v2 embedding choice (p0_analyze.py, line 18 as read this session); a different embedder gives a comparable or better AUC gain and neighbor ratio.

**Change vs existing scripts:** no new rollouts. Reuse E01's groups.parquet questions. Swap the embedding model from all-MiniLM-L6-v2 to a larger sentence-transformers model (e.g. a bge or gte variant) and, separately, to the policy model's own final-layer mean-pooled hidden states extracted from the prompt (a same-model-features condition), then rerun the KMeans/PCA/logistic-regression structure pipeline identically to p0_analyze.py's existing code path.

**Command shape:** p0_analyze.py's embedding line re-pointed at each alternate encoder in turn, output written to separate --out subdirectories for comparison.

**Metric:** auc_cheap_emb and neighbor_ratio for each embedding choice, compared to E01's all-MiniLM-L6-v2 numbers.

**Pass threshold:** at least one alternate embedding matches or exceeds E01's auc_cheap_emb gain and neighbor_ratio, confirming the structure signal is a property of the prompts, not a fragile artifact of one specific encoder.

**Kill threshold:** all alternate embeddings show materially weaker (>= 0.05 AUC lower) structure than all-MiniLM-L6-v2, meaning claim 3's evidence is embedder-specific and the choice of embedding model becomes a load-bearing, undocumented design decision.

**Expected runtime:** under 20 min, RTX 5090 (re-encoding 3,000 questions with 1-2 alternate encoders plus the existing KMeans/PCA/logistic-regression steps, no new rollout generation).

**Informs:** claim 3, robustness check.

---

## E13: Non-stationarity / batch-transfer check (analysis-only)

**Hypothesis:** a structure model (cheap features + embeddings -> proxy_error) fit on an early slice of prompts generalizes to a held-out later slice, a necessary precondition for the learned allocator's online, in-training design (PROJECT.md's stated mechanism) to make sense at all; if structure does not transfer across arbitrary prompt slices, it is unlikely to transfer across the non-stationary rollout distribution produced by a changing policy during GRPO.

**Change vs existing scripts:** no new rollouts. Reuse E01's feats.parquet. Split the 3,000 groups into sequential pseudo-batches (e.g. 10 batches of 300, in the order p0_rollouts.py sampled them), fit LogisticRegression (same features as p0_analyze.py's `auc` helper) on batches 1-5, test AUC on batches 6-10, and compare to the in-sample 5-fold cross-validated AUC E01 already reports.

**Command shape:** a small driver script splitting feats.parquet by row index and refitting the same sklearn pipeline p0_analyze.py already uses, no changes to p0_rollouts.py or the core structure-detection code.

**Metric:** held-out (batches 6-10) AUC vs in-sample 5-fold AUC (E01's auc_cheap_emb).

**Pass threshold:** held-out AUC is within 0.05 of the in-sample AUC, supporting that the structure is stable enough across arbitrary prompt order to be worth learning online.

**Kill threshold:** held-out AUC drops by more than 0.10 relative to in-sample AUC, suggesting the structure signal is unstable across prompt batches even without any policy drift, which would predict the online learning in claim 3's mechanism degrades further once the policy itself is changing (GRPO's actual setting).

**Expected runtime:** under 10 min, CPU-only (no new GPU generation).

**Informs:** claim 3's online-learning premise directly, and de-risks E15 before committing A100 time.

---

## E14: Second code benchmark cross-check (HumanEval)

**Hypothesis:** E08's code-task finding (whatever it is, pass or kill) is not specific to MBPP; HumanEval shows a comparable proxy-error structure result using the same self-consistency-vs-execution proxy design.

**Change vs existing scripts:** same as E08 but --task humaneval in place of mbpp, using openai/openai_humaneval (https://huggingface.co/datasets/openai/openai_humaneval, UNVERIFIED this session). Reuses the E08 code-task script and executor rather than building a second one.

**Command shape:** `python avrl/p0_rollouts_code.py --model <path-to-Qwen2.5-Coder-1.5B-Instruct> --task humaneval --n_prompts 164 --G 8 --out avrl/runs/p0_humaneval` (HumanEval's canonical size is 164 problems; exact count UNVERIFIED this session, not independently refetched).

**Metric:** same structure metrics as E08, compared side by side.

**Pass threshold:** E14's pass/kill verdict (using E08's numeric thresholds) matches E08's verdict, confirming the code-task finding is benchmark-general rather than MBPP-specific.

**Kill threshold:** E14 and E08 disagree on pass/kill, meaning the code-task result from E08 alone should not be generalized and any code-task claim needs to name the specific benchmark.

**Expected runtime:** UNVERIFIED estimate; 164 problems is a much smaller prompt count than E08's MBPP run, so generation should be faster, estimated 10-15 min generation plus execution and analysis overhead similar to E08's. Under 30 min total, estimate.

**Informs:** claim 3, robustness of the task-variation finding.

---

## E15: Core phase-1 four-arm GRPO comparison (NEEDS PHASE-1 HARNESS)

**Hypothesis:** at a 25% expensive-verifier budget, the learned allocator arm reaches higher final held-out accuracy than the threshold arm by more than the spread across seeds, i.e. the learned allocator is worth the extra machinery over a fixed rule during actual training, not just in the phase-0 replay.

**Change vs existing scripts:** this is PROJECT.md's own phase-1 plan, read this session: GRPO with LoRA on Qwen2.5-1.5B-Instruct, 400 steps, four arms (full, random, threshold, learned) at a 25% verification budget, 3 seeds, logging expensive calls, corrupted-reward groups admitted (gold used for logging only, per PROJECT.md), and greedy accuracy on 500 held-out test prompts every 50 steps. Requires avrl/p1_grpo.py, which does not exist yet; allocators.py (read this session) is already built as the pluggable allocator module this harness is meant to call.

**Command shape:** UNVERIFIED, harness not built; PROJECT.md implies something like `python avrl/p1_grpo.py --arm {full,random,threshold,learned} --budget 0.25 --steps 400 --seed {0,1,2}`.

**Metric:** greedy accuracy on the 500-prompt held-out set at step 400; expensive calls spent; corrupted-reward groups admitted, cumulative over training.

**Pass threshold (PROJECT.md's own kill test, restated):** the learned arm beats the threshold arm on final accuracy at equal expensive calls by more than the seed spread across 3 seeds.

**Kill threshold:** the learned arm does not clear that margin; per PROJECT.md, the allocator is not worth further investment and the fixed-threshold recipe becomes the deliverable instead.

**Expected runtime:** UNVERIFIED, no comparable run observed this session; PROJECT.md specifies the hannah cluster (4x A100 40GB) for phase 1, not a single A100, and a LoRA GRPO run of 400 steps across 4 arms x 3 seeds (12 runs total) is a multi-hour-to-multi-day job, not a single-GPU-hour estimate; a real number needs the harness built first and a short (e.g. 20-step) timing run.

**Informs:** claims 1 and 2 jointly, the central claim-testing experiment of the whole project.

---

## E16: In-training budget sweep (NEEDS PHASE-1 HARNESS)

**Hypothesis:** the learned-vs-threshold accuracy gap found in E15 (if any) is not flat across budgets; mirroring E02's phase-0 replay finding, the gap should be largest at low budgets (10%) and shrink as budget rises toward 50%.

**Change vs existing scripts:** same p1_grpo.py harness as E15, restricted to the threshold and learned arms only (dropping full and random to save compute), swept over budget in {0.10, 0.25, 0.50}, 3 seeds each (18 runs total).

**Command shape:** UNVERIFIED, same harness as E15 with --budget varied.

**Metric:** final held-out accuracy gap (learned minus threshold) at each budget point.

**Pass threshold:** the gap is monotonically non-increasing in budget and exceeds the seed spread at the 10% point, consistent with E02's phase-0 prediction.

**Kill threshold:** the gap is flat or non-monotonic across budgets, meaning the phase-0 replay's budget-sensitivity prediction (E02) does not transfer to real training dynamics.

**Expected runtime:** UNVERIFIED, comparable order of magnitude to E15 (18 of the planned 12-run E15 scale), multi-hour-to-multi-day on the hannah cluster per PROJECT.md.

**Informs:** claims 1 and 2, refines E15's headline result into a budget-response curve.

---

## E17: Cross-task transfer of the trained allocator (NEEDS PHASE-1 HARNESS)

**Hypothesis:** an allocator trained online during GSM8K GRPO (E15's learned arm) retains some of its edge over a threshold baseline when applied zero-shot to a different task (MATH-500) without retraining, evidence that it has learned a general notion of "where proxies are exploitable" rather than a GSM8K-specific lookup.

**Change vs existing scripts:** requires both p1_grpo.py (does not exist) and a MATH-500 rollout path in that harness (the E09 parsing change, but inside the training loop rather than the offline phase-0 script); E15's learned arm's fitted LogisticRegression and cell_beta state (allocators.py's LearnedAllocator, read this session) would need a save/load path that does not currently exist in allocators.py.

**Command shape:** UNVERIFIED, harness and cross-task loading path not built.

**Metric:** held-out MATH-500 accuracy at a fixed low budget (e.g. 10%), comparing a GSM8K-pretrained-then-frozen allocator against a threshold baseline and against a from-scratch learned allocator trained directly on MATH-500.

**Pass threshold:** the transferred allocator beats the threshold baseline on MATH-500 by a margin comparable to (within half of) the from-scratch learned allocator's margin, suggesting real transfer rather than task-specific overfitting.

**Kill threshold:** the transferred allocator performs no better than threshold on MATH-500, i.e. all of claim 3's value is task-specific and does not generalize, which PROJECT.md's own phase 2 section already flags as an open question ("transfer of the allocator across task families").

**Expected runtime:** UNVERIFIED, this is explicitly PROJECT.md's phase 2 scope, not phase 1; needs the phase-1 harness plus new state-serialization and cross-task rollout code that does not exist yet.

**Informs:** claim 3, generalization.

---

## E18: Reward-hacking-under-concentration study (NEEDS PHASE-1 HARNESS)

**Hypothesis:** an allocator that learns to concentrate expensive verification on the prompt clusters where the cheap proxy is most exploitable reduces measured reward hacking (policy exploiting the cheap proxy specifically) relative to a random or uniform-threshold allocation at the same budget, because the policy gets caught and corrected more often exactly where it would otherwise be rewarded for gaming the proxy.

**Change vs existing scripts:** requires p1_grpo.py (does not exist) plus a hacking-detection metric that does not exist in any script read this session; PROJECT.md's phase 2 lists this as "reward-hacking study (does concentrating verification where the proxy is exploitable reduce measured hacking)" without specifying the detector. A plausible detector: rate of rollouts where the cheap proxy scores the rollout correct but the gold/expensive verifier (logged, not used for the reward) scores it incorrect, tracked as a training-time series per arm.

**Command shape:** UNVERIFIED, harness and hacking metric not built.

**Metric:** rate of proxy-correct-but-gold-incorrect rollouts (a hacking proxy metric) over training steps, per arm (random, threshold, learned), at matched budget.

**Pass threshold:** the learned arm's hacking-proxy-metric rate at step 400 is lower than both random and threshold arms' rates by a margin exceeding the seed spread.

**Kill threshold:** the learned arm's rate is statistically indistinguishable from threshold's, meaning concentrating verification where the proxy is exploitable does not measurably suppress hacking in this setup.

**Expected runtime:** UNVERIFIED, PROJECT.md phase 2 scope; needs the phase-1 harness plus a new hacking-detection logging path.

**Informs:** claim 2 (a hacking-suppression result is a stronger, more specific version of "fewer corrupted rewards admitted") and directly tests the risk flagged in SYNTHESIS.md section 4 risk 2 (Baker et al., 2025, https://arxiv.org/abs/2503.11926, obfuscated reward hacking once a policy can infer a monitor's pattern).

---

## E19: Adversarial drift / long-horizon obfuscation pressure (NEEDS PHASE-1 HARNESS)

**Hypothesis:** over a longer training horizon than E15's 400 steps, the policy begins to adapt its behavior specifically around the learned allocator's decision boundary (behaving differently when it expects escalation to the expensive verifier versus not), degrading the allocator's structure signal over time in a way that a short run would not reveal, mirroring the "obfuscated reward hacking" pattern reported for CoT monitors (Baker et al., OpenAI, 2025, https://arxiv.org/abs/2503.11926, cited in SYNTHESIS.md risk 2 this session).

**Change vs existing scripts:** requires p1_grpo.py (does not exist), run substantially longer than E15's 400 steps (e.g. 2,000+ steps), and a new logged metric tracking the learned allocator's structure-model AUC over training time (does the fitted LogisticRegression's held-out predictive power for proxy_error decay as training progresses, an in-training analog of E13's non-stationarity check).

**Command shape:** UNVERIFIED, harness not built, long-horizon config not specified anywhere read this session.

**Metric:** learned allocator's rolling structure-model AUC (fit on early-training verified groups, evaluated on late-training verified groups) as a function of training step; hacking-proxy-metric from E18 as a function of training step for the learned arm specifically.

**Pass threshold:** rolling AUC stays within 0.10 of its early-training value through step 2,000, and the E18 hacking-proxy-metric for the learned arm does not trend upward over the same horizon.

**Kill threshold:** rolling AUC degrades by more than 0.10 or the hacking-proxy-metric trends upward specifically for the learned arm (not for random/threshold), direct evidence the allocator itself became a new attack surface, confirming SYNTHESIS.md's risk 2.

**Expected runtime:** UNVERIFIED, substantially longer than E15 (5x the step count at minimum), multi-day on the hannah cluster, estimate only; needs the phase-1 harness plus the new rolling-AUC and hacking-trend logging from E13 and E18.

**Informs:** claim 2 and the single biggest named risk to the whole thesis per SYNTHESIS.md section 4.

---

## E20: Real expensive verifier substitution on code (NEEDS PHASE-1 HARNESS)

**Hypothesis:** when the "expensive" verifier is a genuinely costly operation (full hidden-test-suite execution for code, rather than GSM8K's near-free gold-string match), the learned allocator's edge over threshold, measured in actual wall-clock or compute cost saved at matched final accuracy, is larger than E15's GSM8K result, because the cost asymmetry between cheap and expensive is larger and more realistic.

**Change vs existing scripts:** requires p1_grpo.py (does not exist) adapted to a code task (MBPP or HumanEval, per E08/E14's task setup) with the sandboxed test-execution verifier from E08 wired in as the expensive call inside the training loop, not just the offline phase-0 analysis; this is explicitly PROJECT.md's phase 2 scope ("Real expensive judge ... on open-ended and code tasks").

**Command shape:** UNVERIFIED, harness, code-task training loop, and in-training sandboxed executor all need to be built.

**Metric:** final held-out MBPP/HumanEval accuracy at matched wall-clock verification cost (not just matched call count, since each expensive call is now genuinely costly), learned vs threshold vs random.

**Pass threshold:** the learned allocator reaches threshold's final accuracy using measurably less cumulative verification wall-clock time (>= 20% less), a direct cost-saving claim rather than a call-count proxy for cost.

**Kill threshold:** no wall-clock advantage over threshold once the sandboxed-execution overhead is accounted for.

**Expected runtime:** UNVERIFIED, PROJECT.md phase 2 scope; the largest new-infrastructure lift of the twenty experiments (training-loop code generation, in-training sandboxing, and the phase-1 harness all need to exist first).

**Informs:** claims 1 and 2, with the most realistic cost model of any experiment in this backlog.

---

## Summary: what needs building before it can run

Already runnable today with the existing harness (p0_rollouts.py, p0_analyze.py, allocators.py in analysis-only mode): E01, E02, E03, E06, E07, E10, E11, E12, E13.

Needs a small, scoped addition to the existing harness (new proxy-scoring script, new data loader/parser, or a --run_dir flag on p0_analyze.py, but no training loop): E04, E05, E09.

Needs a new code-task rollout and sandboxed-execution path (does not exist in any script read this session): E08, E14.

NEEDS PHASE-1 HARNESS (avrl/p1_grpo.py, confirmed absent from disk this session; PROJECT.md describes it as already "built and reviewed by the workflow," which this session's file listing does not support): E15, E16, E17 (also needs cross-task allocator state save/load), E18 (also needs a hacking-detection metric), E19 (also needs long-horizon rolling-AUC logging), E20 (also needs an in-training sandboxed executor).

## Added after the gap checks

Four experiments, G01 through G04, added after GAP_already-done.md, GAP_cannot-work.md, and GAP_not-a-thesis.md forced three changes to THESIS_PROPOSAL.md: a periodically recalibrated threshold (not fixed-once) as the primary baseline, a VIP-machinery transfer baseline, and a predictive structure-statistic-versus-gain test promoted to Phase 2's centerpiece. Same format as E01-E20.

---

## G01: Recalibrated-threshold baseline in the allocation replay (analysis-only)

**Hypothesis:** a cost-stratified threshold periodically recalibrated on the same cumulative verified-sample set the learned allocator sees closes most, but not all, of the hit-rate gap the fixed-once threshold shows against the learned allocator in E02/E03, testing the cannot-work gap check's claim that a single fixed-once threshold is too easy a target.

**Change vs existing scripts:** no new rollouts. Reuse E01's feats.parquet. Add a `RecalibratedThreshold` allocator to allocators.py that refits its cut point every `refit_every` groups on the cumulative verified subset, at the same cadence as LearnedAllocator's default (refit_every=20), then rerun the E02/E03 allocation-replay comparison (random, fixed-once threshold, recalibrated threshold, learned, oracle) at budget=0.25 and across E02's budget grid.

**Command shape:** a small driver script instantiating `allocators.RecalibratedThreshold(refit_every=20)` alongside the existing allocators against E01's feats.parquet, reusing sim_alloc.py's replay logic.

**Metric:** hit rate (proxy errors caught per verifier call) for fixed-once threshold vs recalibrated threshold vs learned, at budget=0.25 and across the E02 budget grid.

**Pass threshold:** recalibrated threshold's hit rate exceeds fixed-once threshold's by >= 5 percentage points at at least one budget point, confirming recalibration is a materially stronger baseline, with the learned allocator's remaining edge over it reported alongside rather than assumed away.

**Kill threshold:** recalibrated and fixed-once threshold hit rates are within 2 percentage points of each other at every budget point; recalibration adds nothing in the offline-replay setting.

**Expected runtime:** under 15 min, RTX 5090 or CPU-only (reuses E01's already-computed features, no new GPU generation).

**Informs:** claims 1 and 2, and answers the cannot-work gap check's requirement for a stronger primary baseline before Phase 1 is funded. Does not need the phase-1 harness.

---

## G02: VIP-style transfer baseline (analysis-only)

**Hypothesis:** re-applying VIP's allocation approach (a per-group success-probability predictor feeding a budget-constrained optimizer) with verify-yes/no as the allocated resource, instead of rollout count, recovers most but not all of LearnedAllocator's hit-rate advantage over the recalibrated threshold, narrowing the defensible novelty to the specific combination of mechanism and axis rather than either alone, per the already-done gap check.

**Change vs existing scripts:** no new rollouts. Reuse E01's feats.parquet. Add a `VIPTransferAllocator` to allocators.py: a per-group disagreement-probability predictor (same feature set as LearnedAllocator, same refit cadence) feeding a convex top-k budget allocation instead of LearnedAllocator's Thompson-sampling layer, approximating VIP's Gaussian-process-plus-convex-optimization shape (Nguyen et al., ICLR 2026, https://arxiv.org/abs/2602.01601) with verify-yes/no substituted for rollout count.

**Command shape:** a small driver script instantiating `allocators.VIPTransferAllocator()` against E01's feats.parquet, run through the same G01 replay harness, at budget=0.25 and across the E02 budget grid.

**Metric:** hit rate and bootstrap-CI corrupted-reward-admitted rate (per E03's method) for VIP-transfer vs recalibrated threshold vs learned, at matched budget.

**Pass threshold:** the learned allocator's hit rate exceeds VIP-transfer's by >= 5 percentage points at the 25% budget point, evidence the novelty is not just the axis substitution but also the allocator design (Thompson sampling over cells vs VIP's convex optimization).

**Kill threshold:** VIP-transfer and learned are within 3 percentage points of each other; per the already-done gap check, the defensible contribution narrows to "axis substitution using an existing mechanism."

**Expected runtime:** under 15 min, RTX 5090 or CPU-only (reuses E01's features, no new GPU generation).

**Informs:** claim 1, and directly answers the already-done gap check's challenge that VIP's machinery should be tried as a baseline before crediting the allocator design itself. Does not need the phase-1 harness.

---

## G03: Structure-statistic-versus-gain scan across task/proxy pairs

**Hypothesis:** the learned allocator's hit-rate gain over the recalibrated threshold (G01) is monotone in each pair's structure statistics, the neighbor-hit ratio of proxy error over embedding cells and the AUC of cheap features for proxy error (both computed by p0_analyze.py), across at least four task/proxy pairs, the pre-registered prediction that makes claim 3 falsifiable rather than a single-task detection.

**Change vs existing scripts:** reuses E01 (GSM8K, majority-vote), E04 (GSM8K, judge proxy; needs p0_judge_score.py), E09 (MATH-500, majority-vote; needs the --data flag and boxed-answer parser), and a fourth pair combining MATH-500 with the judge proxy (same scripts as E04+E09 composed, no new code). For each pair: compute neighbor-hit ratio and auc_cheap_emb (p0_analyze.py's existing structure block) and the recalibrated-threshold-vs-learned hit-rate gain (G01's replay) at budget=0.25.

**Command shape:** the existing p0_rollouts.py / p0_analyze.py / p0_judge_score.py commands for each of the four pairs (per E01, E04, E09 as already specified), followed by a small driver script that reads each pair's summary.json/result.json and G01's replay output, and fits a Spearman correlation between each structure statistic and the hit-rate gain across the four points.

**Metric:** Spearman correlation between (neighbor-hit ratio, auc_cheap_emb) and (learned-minus-recalibrated-threshold hit-rate gain), across the four task/proxy pairs.

**Pass threshold:** at least one statistic has |Spearman rho| >= 0.7 across the four pairs, sign matching the pre-registered direction (higher structure, higher gain), supporting contribution 3's monotonicity claim even at this small sample size.

**Kill threshold:** both statistics have |Spearman rho| < 0.3, or the sign is inconsistent with the pre-registered direction in more than one pair; contribution 3's predictive claim is unsupported at this sample size and Phase 2's transfer-test centerpiece should be re-scoped before Phase 1 funding.

**Expected runtime:** each of the four task/proxy generation-and-analysis passes is separately budgeted at under 40 minutes on the RTX 5090 (per E01's ~16 min generation + ~10 min analysis baseline and E04/E09's estimates), plus G01's replay (under 15 min) and a negligible correlation-fitting step; total under 3 hours across four sequential runs, each individually under the 40-minute cap.

**Informs:** claim 3 directly, the central pre-registered test named in THESIS_PROPOSAL.md's Contribution 3 and Phase 2 centerpiece, runnable before Phase 1 is funded since it uses only Phase-0-style offline generation and replay. Does not need the phase-1 harness.

---

## G04: Cross-task, cross-proxy transfer test of the trained allocator (NEEDS PHASE-1 HARNESS)

**Hypothesis:** an allocator trained online during GSM8K GRPO (Phase 1's learned arm) retains a measurable fraction of its hit-rate edge over the recalibrated threshold when applied zero-shot, without retraining, to an unseen task/proxy pair, evidence it learned a transferable notion of where proxies are exploitable rather than a task-specific lookup; this is THESIS_PROPOSAL.md's Phase 2 centerpiece, promoted from a late ablation per the not-a-thesis gap check.

**Change vs existing scripts:** requires p1_grpo.py (does not exist, per E15) plus a save/load path for LearnedAllocator's fitted LogisticRegression and cell_beta state (allocators.py, read this session, has no serialization method currently) and a rollout path in p1_grpo.py for at least one additional task/proxy pair (e.g. MATH-500 under majority vote, reusing E09's parsing change inside the training loop rather than offline).

**Command shape:** UNVERIFIED, harness and cross-task state save/load not built; implied shape is `python avrl/p1_grpo.py --arm learned --load_allocator <path-from-gsm8k-run> --task math500 --budget 0.10 --steps 0` (zero further training steps, pure zero-shot evaluation of the frozen, transferred allocator) compared against a from-scratch learned arm and the recalibrated-threshold arm both trained/fit directly on MATH-500.

**Metric:** held-out MATH-500 accuracy and hit rate at a fixed low budget (10%) for: recalibrated threshold (fit on MATH-500), from-scratch learned (trained on MATH-500), and transferred learned (GSM8K-trained, frozen, zero-shot on MATH-500).

**Pass threshold:** transferred-learned's edge over the recalibrated threshold on MATH-500 is at least half of from-scratch-learned's edge over the same threshold, consistent with G03's monotonicity prediction if MATH-500's structure statistics are comparable to GSM8K's.

**Kill threshold:** transferred-learned performs no better than the recalibrated threshold on MATH-500, meaning none of the fitted allocator's value transfers across task/proxy pairs, falsifying the cross-pair generality contribution 3 claims even if G03's within-pair correlation holds.

**Expected runtime:** UNVERIFIED, Phase 2 scope; needs the phase-1 harness plus new state-serialization and cross-task rollout code that does not exist yet.

**Informs:** claim 3's generality directly, and is the single test THESIS_PROPOSAL.md's not-a-thesis gap check says must be central rather than contingent. Needs the phase-1 harness.

## Sources

All paper citations above are carried forward unchanged from /homes/rdubash/active-audit/lit/avrl/SYNTHESIS.md, read this session, which itself states its method: built via WebFetch against arXiv abstracts, the OpenAlex works API, and direct publisher DOIs, with WebSearch unavailable (budget error) and not retried. This document adds one new fact verified this session via WebFetch: HuggingFaceH4/MATH-500's existence, size, and content, https://huggingface.co/datasets/HuggingFaceH4/MATH-500. No paper, author, or number is invented in this document; every runtime figure not directly read from avrl/runs/p0/run.log is explicitly marked as an estimate or UNVERIFIED. G01-G04 were added after this session's three gap checks (GAP_already-done.md, GAP_cannot-work.md, GAP_not-a-thesis.md); their citations (VIP, https://arxiv.org/abs/2602.01601) are carried forward unchanged from SYNTHESIS.md and the gap-check files.

## Proposed by the loop

Follow-ups from RESEARCH_LOG.md's E03 entry (2026-09-30), the autoresearch loop's bootstrap-significance check of "threshold" (static analog) vs "learned" (ts_online analog) at budget=0.25 on cell100/feats.parquet. E03's result: KILL_REVERSED, bootstrap 95% CI on (threshold_rate - learned_rate) = [-0.225, -0.114], i.e. threshold admits fewer corrupted-reward groups than learned, significantly, the opposite of claim 2's requirement. Caveat: "static" gets a free-labeled quarter of cell100 clusters beyond the audit budget that "ts_online" does not, so the comparison is not budget-matched.

### L3a: Budget-matched re-check, static_matched vs ts_online

**Hypothesis:** E03's reversal (threshold beats learned) is driven by static's free-label advantage, not a real effect; with a budget-matched threshold (sim_generic.py's existing "static_matched" policy, no free labels) the gap shrinks or flips back toward claim 2's prediction.

**Change vs existing scripts:** copy avrl/exp/e03_bootstrap_sig.py, swap the "static" policy call for sim_generic.py's "static_matched" logic (already implemented, spends half the budget on random labels then fits and ranks the rest), keep "ts_online" as the learned analog, same bootstrap procedure.

**Command shape:** same as E03's, new script path, e.g. `avrl/exp/e03b_bootstrap_matched.py --feats avrl/runs/p0/feats.parquet --label proxy_error --group cell100 --budget_frac 0.25 --n_boot 500 --seed 0 --out avrl/runs/L3a/result.json`.

**Pass threshold:** 95% CI on (threshold_matched_rate - learned_rate) excludes zero and is positive (learned wins once threshold is budget-matched).

**Kill threshold:** CI still excludes zero and is negative (threshold still wins even budget-matched), or CI includes zero (no significant difference either way) -- both would further weaken claim 2's case for the E15 hannah run.

**Expected runtime:** ~15 min, CPU-only, no GPU.

**Informs:** claim 2 directly; resolves the single caveat E03 itself flagged, cheaper than building G01's RecalibratedThreshold class.

### L3b: Budget-grid re-check of the reversal

**Hypothesis:** the threshold-beats-learned reversal seen at 25% budget is budget-dependent; E01's analyze.log point estimates hint the gap is wider at 10% (165 static vs 95 ts_online) and narrower at 50% (299 vs 288), so a budget regime might exist where learned does not lose.

**Change vs existing scripts:** rerun E03's exact script (avrl/exp/e03_bootstrap_sig.py) unmodified with `--budget_frac 0.10` and `--budget_frac 0.50`, two more result.json files.

**Command shape:** `.venv/bin/python avrl/exp/e03_bootstrap_sig.py --feats avrl/runs/p0/feats.parquet --label proxy_error --group cell100 --budget_frac 0.10 --n_boot 500 --seed 0 --out avrl/runs/L3b_10pct/result.json` (and again with 0.50).

**Pass threshold:** at least one of the two budgets shows a CI that excludes zero on the positive side (learned beats threshold somewhere on the grid).

**Kill threshold:** both budgets show CI negative or unchanged in sign from E03's 25%-budget result; the reversal holds across the budget range phase-1 would plausibly use.

**Expected runtime:** ~15 min each, CPU-only, no GPU (two runs, can run sequentially).

**Informs:** claim 2, and whether any audit budget makes the learned allocator viable before committing hannah A100-hours to E15.

### L3c: Real allocators.py classes, not sim_generic.py analogs

**Hypothesis:** E03's static/ts_online mapping (an explicit assumption, since sim_generic.py's policy names do not literally match allocators.py's phase-1 arm names) reproduces the same reversal when the actual phase-1 LearnedAllocator and a literal fixed-threshold class from allocators.py are run offline against feats.parquet, confirming the finding is not an artifact of the analogy chosen.

**Change vs existing scripts:** new script that imports allocators.py's LearnedAllocator and a fixed-threshold allocator directly (not sim_generic.py's static/ts_online), replays them offline against feats.parquet's cheap features at budget=0.25, and runs the same bootstrap procedure as E03.

**Command shape:** UNVERIFIED, needs scoping; allocators.py's classes may assume an online GRPO-loop calling convention (verified label arriving one group at a time) rather than a batch offline replay, so an adapter may be needed before this can run as a single CLI command.

**Pass/kill thresholds:** same as E03's (CI positive and excludes zero = pass; CI negative and excludes zero, or includes zero = kill/inconclusive for claim 2).

**Expected runtime:** larger than E03's 15 min once scoped; flag for a plan check before running, since the adapter work is unbounded until allocators.py's calling convention is read.

**Informs:** claim 2, with the actual phase-1 code rather than a stated analog; resolves the mapping assumption flagged in every E03-lineage experiment above.

### L4a: Budget-grid re-check using the validated static_matched baseline

**Hypothesis:** L3a's null result (static_matched vs ts_online statistically tied at 25% budget, CI [-0.0568, 0.0348]) is budget-dependent; E01's point estimates hint the gap between threshold-style and learned policies is wider at low budgets and narrower at high budgets, so a budget regime might still show a significant edge either way now that the confound is removed.

**Change vs existing scripts:** rerun avrl/exp/e03b_bootstrap_matched.py unmodified with `--budget_frac 0.10` and `--budget_frac 0.50`. Supersedes L3b's plan (which still used E03's free-label "static", now a known confound) as the budget-grid check to actually run.

**Command shape:** `.venv/bin/python avrl/exp/e03b_bootstrap_matched.py --feats avrl/runs/p0/feats.parquet --label proxy_error --group cell100 --budget_frac 0.10 --n_boot 500 --seed 0 --out avrl/runs/L4a_10pct/result.json` (and again with 0.50).

**Pass threshold:** at least one of the two budgets shows a CI that excludes zero on the positive side (learned significantly beats static_matched somewhere on the grid).

**Kill threshold:** both budgets show a CI that includes zero or excludes zero on the negative side, consistent with L3a's null holding across the budget range phase-1 would plausibly use.

**Expected runtime:** ~15 min each, CPU-only, no GPU.

**Informs:** claim 2, and whether any audit budget makes the learned allocator viable before committing hannah A100-hours to E15.

### L4b: Higher per-resample seed count, to separate noise from a genuine tie

**Hypothesis:** L3a's wide CI ([-0.0568, 0.0348], about 0.09 wide against a point diff of only -0.0133) is partly Monte Carlo noise from averaging each policy over only 8 seeds per bootstrap resample, not purely group-resampling variance; raising the seed count should shrink the CI and clarify whether the 25%-budget result is a genuine tie or an underpowered test.

**Change vs existing scripts:** copy avrl/exp/e03b_bootstrap_matched.py, change only the SEEDS constant from 8 to 32, nothing else.

**Command shape:** `.venv/bin/python avrl/exp/e03c_bootstrap_matched_32seed.py --feats avrl/runs/p0/feats.parquet --label proxy_error --group cell100 --budget_frac 0.25 --n_boot 500 --seed 0 --out avrl/runs/L4b/result.json`.

**Pass/kill thresholds:** same as L3a's (CI positive and excludes zero = pass; CI negative and excludes zero, or includes zero = kill for claim 2). A CI that shrinks substantially but still includes zero would confirm the tie is genuine rather than a power artifact.

**Expected runtime:** ~15 min, CPU-only, no GPU (4x the per-resample seed work, still well under the runtime cap given L3a ran in 113s).

**Informs:** whether to trust L3a's null as final or treat it as inconclusive due to noise; cheap sanity check before spending budget on L4a or L4c.

### L4c: Audit-group granularity sensitivity (cell30 / cell300)

**Hypothesis:** L3a's null result at cell100 is sensitive to the audit-group granularity; coarser buckets (cell30) give the classifier more data per fit but coarser control over the budget, finer buckets (cell300) give the opposite trade-off, and phase-1's actual grouping choice is not guaranteed to match cell100.

**Change vs existing scripts:** rerun avrl/exp/e03b_bootstrap_matched.py unmodified with `--group cell30` and `--group cell300` (both columns confirmed present in avrl/runs/p0/feats.parquet by reading it directly, no new feature engineering needed).

**Command shape:** `.venv/bin/python avrl/exp/e03b_bootstrap_matched.py --feats avrl/runs/p0/feats.parquet --label proxy_error --group cell30 --budget_frac 0.25 --n_boot 500 --seed 0 --out avrl/runs/L4c_cell30/result.json` (and again with `--group cell300`).

**Pass threshold:** at least one granularity shows a CI that excludes zero on the positive side (learned significantly beats static_matched at that grouping).

**Kill threshold:** both granularities show a CI that includes zero or excludes zero on the negative side, consistent with L3a's null holding regardless of grouping choice.

**Expected runtime:** ~10 min each, CPU-only, no GPU.

**Informs:** claim 2's robustness to an implementation detail (group definition) that phase-1's actual harness may set differently than cell100.

Follow-ups from RESEARCH_LOG.md's E04 entry (2026-09-30), the autoresearch loop's judge-proxy structure check (Qwen2.5-0.5B-Instruct as a gold-free judge vs E01's majority-vote proxy, on E01's existing GSM8K rollouts). E04's result: KILL, Jaccard(majority-wrong, judge-wrong) = 0.993, auc_cheap_emb - auc_cheap = -0.0006. Root cause, confirmed directly: judge_pick equals the majority answer in 99.27% of groups (2,978/3,000), because the tally rule (yes-vote weight = self-consistency count when the judge says "yes", ties broken by that same count) collapses to majority vote whenever the judge endorses the majority answer, which it does almost everywhere. This makes E04's KILL a confound risk: it cannot distinguish "the judge genuinely fails on the same prompts as majority vote" from "the aggregation rule silently reproduces majority vote regardless of what the judge thinks."

### L5a: Judge proxy with a consistency-free tie-break

**Hypothesis:** E04's Jaccard = 0.993 is driven by the self-consistency-weighted tie-break, not by the judge genuinely agreeing with majority vote on which prompts are hard; removing the count-weighting should lower the Jaccard overlap and let the judge's real error pattern (if any) show through.

**Change vs existing scripts:** extend avrl/exp/e04_judge_score.py to also save the per-(pid, answer) judge_yes column for the same 10,168 already-judged unique pairs (no new judge inference needed beyond that one extra column), then rerun group-level aggregation with a tie-break that ignores self-consistency count entirely (random tie-break among judge-endorsed answers, or "first distinct answer encountered in rollout order").

**Command shape:** `.venv-judge/bin/python avrl/exp/e05a_judge_score_notie.py --rollouts avrl/runs/p0/rollouts.parquet --groups avrl/runs/p0/groups.parquet --judge_model Qwen/Qwen2.5-0.5B-Instruct --tie_break random --out avrl/runs/L5a/judge_scores.parquet` then the existing e04_structure.py pointed at the new judge_scores.parquet.

**Pass threshold:** Jaccard(majority-wrong, judge-wrong) drops below 0.8 with the consistency-free tie-break, confirming E04's near-total overlap was a tally artifact and reopening the proxy-type question.

**Kill threshold:** Jaccard stays above 0.8 even with a consistency-free tie-break, confirming the judge and majority vote genuinely fail on the same prompts regardless of aggregation, which would be a stronger and more decisive version of E04's kill.

**Expected runtime:** ~15 min, goldberg GPU for the one-column extension of the judge pass, CPU for aggregation and structure.

**Informs:** claim 3 and whether E04's kill should be trusted as a real finding about judge-type proxies or discarded as a design artifact.

### L5b: Self-verification proxy (E05), a mechanism with no self-consistency count

**Hypothesis:** E05's self-verification proxy (the policy re-asked to check its own answer) gives a cleaner second point on the proxy-type axis than E04, since its design has no self-consistency-weighted tie-break to degenerate into majority vote.

**Change vs existing scripts:** run E05 as already speced in this backlog (see the E05 section above), unmodified by this follow-up.

**Command shape:** per E05's own pre-registered command shape.

**Pass/kill thresholds:** per E05's own pre-registered thresholds.

**Expected runtime:** ~25 min, goldberg GPU, per E05's own estimate.

**Informs:** claim 3, independent of the L5a confound check, since E05's mechanism cannot suffer from the same tally artifact.

### L5c: Per-answer judge accuracy, no group aggregation

**Hypothesis:** the most direct diagnostic of whether Qwen2.5-0.5B-Instruct can tell correct from incorrect GSM8K answers at all is judge_yes accuracy at the unique-(pid, answer) level (10,168 pairs), measured before any group-level pick or tie-break; this is cheaper than L5a and tells us whether the judge signal is worth aggregating in the first place.

**Change vs existing scripts:** reuses the per-(pid, answer) judge_yes column from L5a's script extension (no new judge inference); new small analysis script computes precision/recall/accuracy of judge_yes against (answer == gold) over the 10,168 unique pairs.

**Command shape:** `.venv/bin/python avrl/exp/e05c_judge_calibration.py --rollouts avrl/runs/p0/rollouts.parquet --groups avrl/runs/p0/groups.parquet --judge_scores avrl/runs/L5a/judge_scores_peranswer.parquet --out avrl/runs/L5c/result.json`.

**Pass/kill thresholds:** not a pass/kill for a thesis claim, a diagnostic; report accuracy, precision, recall and compare to the 85.5% base rate of majority-vote correctness (a judge below that base rate is not worth building a proxy around at all).

**Expected runtime:** ~5 min, CPU-only, depends on L5a's column extension landing first.

**Informs:** whether further judge-proxy experiments (L5a, E05 variants) are worth the compute before running them.

### L6a: Allocation replay using self-verification agreement as the confidence signal

**Hypothesis:** E05's self-verification proxy has error rate close to majority vote's (12.2% vs 14.5%, -2.37pp) but only partial error-set overlap (Jaccard 0.47, Pearson r=0.587), a middle zone that rate-and-correlation statistics alone cannot resolve into "worth a separate allocator arm" or not; an allocation replay using sv_agreement as the confidence signal, and a combined proxy flagging either signal, tests this operationally.

**Change vs existing scripts:** reuse avrl/runs/sim_generic.py's replay logic (same as E01's allocation-replay table), feeding it sv_agreement (from avrl/runs/E05/selfverify.parquet) as the confidence column instead of agreement, plus one combined column (error flagged if EITHER majority-vote-agreement-threshold OR self_verify_error fires). No new generation.

**Command shape:** `.venv/bin/python avrl/exp/e06a_selfverify_replay.py --feats avrl/runs/p0/feats.parquet --selfverify avrl/runs/E05/selfverify.parquet --out avrl/runs/L6a/result.json`.

**Pass/kill thresholds:** not a claim-level pass/kill, an operational diagnostic; report errors-caught-per-budget (10/25/50%) for sv_agreement-only, agreement-only (E01's column, for reference), and the combined proxy. A combined proxy that catches meaningfully more errors per budget than agreement alone (e.g. >=10% relative improvement at 25% budget) would support building a separate allocator arm; a combined proxy indistinguishable from agreement-only would argue against it.

**Expected runtime:** ~15 min, CPU-only, no GPU.

**Informs:** whether E05's inconclusive rate/correlation result translates into any real allocation-budget benefit, before spending more compute on proxy-type variants.

### L6b: E08 (code task, MBPP) as the next proxy-type/task-type point

**Hypothesis:** the pattern seen across E01 (majority vote), E04 (judge, tie-break confound), L5a (judge, no tie-break) and E05 (self-verification) -- proxy-vs-majority correlation varies widely (Jaccard/r from near-total overlap to near-random) while embedding-space AUC gap stays flat-to-negative in all four -- is a general feature of proxy-error structure, not specific to GSM8K math.

**Change vs existing scripts:** per the backlog's existing E08 section (code task, MBPP, execution as gold verifier, small judge as proxy); this follow-up only reorders it to run immediately next, per the priority note's own existing ordering, now doubly motivated by E05's result.

**Command shape:** per E08's own section below.

**Pass/kill thresholds:** per E08's own section below.

**Expected runtime:** ~25-30 min, goldberg GPU.

**Informs:** whether the flat-embedding-structure finding generalizes across task domains, independent of which proxy type is used.

### L6c: Bootstrap CI on E05's Pearson r and diff_pp

**Hypothesis:** E05's INCONCLUSIVE verdict (diff_pp=-2.37pp, r=0.587, neither crossing its pre-registered line) reflects a stable middle-ground correlation rather than estimation noise at n=3,000; a bootstrap CI over groups should confirm this before further compute is spent on L6a.

**Change vs existing scripts:** copy avrl/exp/e03b_bootstrap_matched.py's resampling pattern, resample (self_verify_error, proxy_error) pairs over the 3,000 groups with replacement (1,000 resamples), report CI on Pearson r and on diff_pp.

**Command shape:** `.venv/bin/python avrl/exp/e06c_bootstrap_selfverify.py --selfverify avrl/runs/E05/selfverify.parquet --feats avrl/runs/p0/feats.parquet --n_boot 1000 --seed 0 --out avrl/runs/L6c/result.json`.

**Pass/kill thresholds:** not a claim-level pass/kill, a robustness check; a CI on r that stays well clear of 0.9 and a CI on diff_pp that stays well clear of 3pp would confirm INCONCLUSIVE is a genuine finding, not a power artifact. A CI that approaches either line would argue for rerunning E05 with a different seed before trusting the verdict.

**Expected runtime:** ~10 min, CPU-only, no GPU.

**Informs:** confidence in E05's own verdict before committing to L6a's allocation replay.

Follow-ups from RESEARCH_LOG.md's L6a entry (2026-09-30), the autoresearch loop's allocation replay of self-verification agreement as a triage signal vs majority-vote agreement. L6a's result: NO_SUPPORT, agreement_only dominated sv_agreement_only and combined_flag at every budget tested (10/25/50%); combined_vs_agreement_rel_improve at 25% budget = -37.4% (pass bar was >=10%).

### L7a: Continuous blend of agreement and sv_agreement, not a binary OR-gate

**Hypothesis:** L6a's combined_flag lost to agreement_only mainly because its binary OR-gate (agreement < 0.75 OR self_verify_error == 1) discards agreement's continuous ranking signal, not because sv_agreement carries zero incremental information; a continuous blend should recover some of that loss if there is any real signal to recover.

**Change vs existing scripts:** copy avrl/exp/e06a_selfverify_replay.py, add a new ranking candidate that is a weighted sum (e.g. 0.5*agreement + 0.5*sv_agreement, rank-normalized first) or min(agreement, sv_agreement), instead of the binary combined_flag; keep agreement_only, sv_agreement_only, uniform and oracle as-is for comparison.

**Command shape:** `.venv/bin/python avrl/exp/e07a_selfverify_blend.py --feats avrl/runs/p0/feats.parquet --selfverify avrl/runs/E05/selfverify.parquet --out avrl/runs/L7a/result.json`.

**Pass threshold:** the blended signal's errors-caught-per-budget exceeds agreement_only's by >=10% relative at the 25% budget point, supporting a separate allocator arm after all.

**Kill threshold:** the blend is still at or below agreement_only at every budget, confirming self-verification adds no operational triage value under any combination tried so far (binary or continuous).

**Expected runtime:** ~10 min, CPU-only, no GPU, reuses L6a's inputs exactly.

**Informs:** whether L6a's NO_SUPPORT verdict is specific to the binary-flag construction or a genuine property of sv_agreement; closes the "just a bad combination rule" possibility before fully ruling out self-verification as a triage signal.

### L7b: Same operational replay for the judge proxy (E04/L5a)

**Hypothesis:** L6a's errors-caught-per-budget replay has only been run for self-verification; the judge proxy (E04/L5a) has rate/correlation diagnostics (Jaccard, AUC) but no operational allocation-replay check, so it is unknown whether judge_error has any budget-constrained triage value either.

**Change vs existing scripts:** copy avrl/exp/e06a_selfverify_replay.py, swap the sv_agreement/self_verify_error inputs for L5a's judge_error-based columns (avrl/runs/L5a/judge_scores.parquet), same budgets, same seed convention.

**Command shape:** `.venv/bin/python avrl/exp/e07b_judge_replay.py --feats avrl/runs/p0/feats.parquet --judgescores avrl/runs/L5a/judge_scores.parquet --out avrl/runs/L7b/result.json`.

**Pass threshold:** judge-based ranking (or a judge+agreement combined flag) exceeds agreement_only's errors-caught-per-budget by >=10% relative at the 25% budget point.

**Kill threshold:** judge-based ranking is at or below agreement_only at every budget, extending L6a's finding (agreement-only dominance) to the judge proxy and closing the operational question for both proxy types tested on GSM8K.

**Expected runtime:** ~10 min, CPU-only, no GPU, reuses E04/L5a's already-computed data.

**Informs:** claim 3's remaining live territory; if both proxy types fail operationally (not just on rate/correlation), GSM8K proxy-type variation is fully exhausted and task variation (E08/E09) becomes the only avenue left.

### L7c: Fast-track E08 (MBPP code task) given both GSM8K proxy types now exhausted

**Hypothesis:** with self-verification (L6a) and the judge proxy (E04/L5a) both failing to beat agreement-only on rate, correlation, and (per L6a) operational triage grounds, the open question on this thesis is no longer "which proxy type works on GSM8K" but "does agreement-only's dominance and the flat embedding-structure finding generalize to a different task domain."

**Change vs existing scripts:** per the backlog's existing E08 section (code task, MBPP, execution as gold verifier, small judge as proxy); this follow-up only reaffirms it as the top remaining priority, now with L6a's result as additional motivation on top of L6b's original reasoning.

**Command shape:** per E08's own section below.

**Pass/kill thresholds:** per E08's own section below.

**Expected runtime:** ~25-30 min, goldberg GPU, new rollout generation (not analysis-only).

**Informs:** whether the agreement-only-dominates pattern is a GSM8K-specific artifact or a general property of proxy-error structure across task types.

Follow-ups from RESEARCH_LOG.md's E06 entry (2026-09-30), the model-size (small end) point on claim 3. E06's result: KILL, Qwen2.5-0.5B-Instruct's confident-subset proxy_error_rate (4.27%) cleared the +2pp bar over E01's 1.5B baseline (1.1%), but auc_cheap_emb minus auc_cheap (-0.0032) and neighbor_ratio at cell100 (0.46x, confident subset) both failed their thresholds decisively, extending the threshold-regime kill to a second, weaker model and ruling out model capability (at the small end) as the explanation for claim 3's repeated GSM8K failure.

### L8a: E08 50-problem MBPP dry run, sandboxed execution harness validation

**Hypothesis:** with both proxy-type (judge/L5a, self-verification/E05) and model-size at the small end (E06) now exhausted for GSM8K, task-type variation (E08, code generation with execution as gold verifier) is the last live untested axis for claim 3; the backlog's own E08 section recommends a small dry run first since the sandboxed test-execution harness does not exist yet.

**Change vs existing scripts:** build a minimal MBPP test-execution sandbox (new script, not a copy of an existing p0/E0N script since none currently runs untrusted code) and run it on 50 MBPP problems only, checking the harness is correct and safe (timeouts, no filesystem/network escape) before committing to the full E08 run.

**Command shape:** per E08's own section in this backlog, scaled to `--n_problems 50` for the dry run.

**Pass/kill thresholds:** not a claim-level pass/kill, an engineering validation gate; pass means the sandbox correctly scores a known-correct and a known-incorrect solution for all 50 problems with no timeouts or escapes, unblocking the full E08 run; fail means the harness needs rework before any E08 result can be trusted.

**Expected runtime:** ~20-25 min, goldberg GPU for generation plus CPU sandbox execution.

**Informs:** whether E08 (the only remaining untested axis for claim 3) can proceed safely next iteration.

### L8b: Model-size variation at the large end (run the existing E07 spec next)

**Hypothesis:** E06 only tested the small end of the model-size axis (0.5B vs E01's 1.5B) and found the same kill; the backlog's own E07 section (large end, Qwen2.5-7B-Instruct) is the needed second point before "model size is ruled out" can be stated for the axis as a whole, since a much larger model could plausibly produce a cleaner majority-vote signal with real embedding-separable structure that neither 0.5B nor 1.5B showed.

**Change vs existing scripts:** per E07's own section above (model Qwen2.5-7B-Instruct, n_prompts=1000, G=8, seed=0); this follow-up only reaffirms it as the correct next step on the model-size axis, now doubly motivated by E06's small-end result, and repeats E07's own recommendation of a 100-prompt dry run first since its runtime estimate is UNVERIFIED and goldberg's card is confirmed 16GB (RTX A4000), not the 32GB figure in the ground rules.

**Command shape:** per E07's own section above.

**Pass/kill thresholds:** per E07's own section above.

**Expected runtime:** per E07's own section above (20-35 min generation plus ~10 min analysis, UNVERIFIED; dry run recommended first).

**Informs:** claim 3's model-size axis, completing what E06 only tested in one direction.

### L8c: Cross-model error-overlap between E01 (1.5B) and E06 (0.5B)

**Hypothesis:** E01 and E06 fail on GSM8K at very different overall rates (14.5% vs 38.4%), but it is unknown whether the two models are wrong on the same prompts (shared, structural task difficulty that might generalize across models) or different prompts (idiosyncratic per-model noise); this bears on whether any "hard prompt" signal could ever transfer across model sizes even though embedding-AUC structure does not.

**Change vs existing scripts:** no new generation; new small CPU-only script merges avrl/runs/p0/feats.parquet (E01) and avrl/runs/E06/feats.parquet (E06) on pid, computes Jaccard and Pearson r between the two models' proxy_error columns.

**Command shape:** `.venv/bin/python avrl/exp/e08c_cross_model_overlap.py --feats_a avrl/runs/p0/feats.parquet --feats_b avrl/runs/E06/feats.parquet --out avrl/runs/L8c/result.json`.

**Pass/kill thresholds:** not a claim-level pass/kill, a diagnostic; high Jaccard/r would indicate shared, model-independent task difficulty (interesting for a future difficulty-prediction angle); low Jaccard/r would confirm the two models' errors are largely idiosyncratic, consistent with model size mattering for error rate but not for error structure.

**Expected runtime:** ~5 min, CPU-only, no GPU.

**Informs:** whether any cross-model difficulty signal exists at all, independent of the embedding-AUC question E06 already settled.

Follow-ups from RESEARCH_LOG.md's L7b entry (2026-09-30), the autoresearch loop's allocation replay of judge-based ranking (E04/L5a) as a triage signal vs majority-vote agreement. L7b's result: KILL/NO_SUPPORT, agreement_only dominated judge_only and combined_flag at every budget tested (10/25/50%); judge_vs_agreement_rel_improve at 25% budget = -58.4%, combined_vs_agreement_rel_improve = -54.6% (pass bar was >=10%). This extends L6a's self-verification finding to the judge proxy, closing the operational-triage question for both GSM8K alternate-proxy types tried this loop.

### L9a: Continuous blend of agreement and judge_pick_share, not a binary OR-gate

**Hypothesis:** L7b's combined_flag lost to agreement_only for the same structural reason L6a's did: the binary OR-gate (agreement < 0.75 OR judge_error == 1) discards agreement's continuous ranking signal, not necessarily because judge_pick_share carries zero incremental information; a continuous blend should recover some of that loss if there is any real signal left to recover.

**Change vs existing scripts:** copy avrl/exp/e07b_judge_replay.py, add a new ranking candidate that is a weighted sum (e.g. 0.5*agreement + 0.5*judge_pick_share, rank-normalized first) or min-rank(agreement, judge_pick_share), instead of the binary combined_flag; keep agreement_only, judge_only, uniform and oracle as-is for comparison. Mirrors L7a's still-open blend test for self-verification, applied to judge.

**Command shape:** `.venv/bin/python avrl/exp/e09a_judge_blend.py --feats avrl/runs/p0/feats.parquet --judgescores avrl/runs/L5a/judge_scores.parquet --out avrl/runs/L9a/result.json`.

**Pass threshold:** the blended signal's errors-caught-per-budget exceeds agreement_only's by >=10% relative at the 25% budget point, supporting a separate allocator arm after all.

**Kill threshold:** the blend is still at or below agreement_only at every budget, confirming judge adds no operational triage value under any combination tried so far (binary or continuous), matching L7a's expected outcome for self-verification if that also kills.

**Expected runtime:** ~10 min, CPU-only, no GPU, reuses L7b's inputs exactly.

**Informs:** whether L7b's kill is specific to the binary-flag construction or a genuine property of judge_pick_share; closes the "just a bad combination rule" possibility before fully ruling out the judge proxy as a triage signal.

### L9b: Three-way ensemble of agreement, sv_agreement and judge_pick_share

**Hypothesis:** self-verification (L6a) and judge (L7b) each fail the operational bar alone, but each disagrees with agreement on a different subset of groups (Jaccard 0.47 self-verification, 0.21 judge, per L5a/E05) and has not been tested jointly with the other; a three-way ensemble is the one combination not yet tried and could in principle catch errors that neither proxy alone nor any single pairwise combination reaches, even though each pairwise test (L6a, L7b) failed independently.

**Change vs existing scripts:** new small script merging avrl/runs/p0/feats.parquet, avrl/runs/E05/selfverify.parquet and avrl/runs/L5a/judge_scores.parquet on pid (three-way inner join); rank by a combined flag (agreement < 0.75 OR self_verify_error == 1 OR judge_error == 1) and by a three-signal weighted blend, same budgets and seed convention as L6a/L7b.

**Command shape:** `.venv/bin/python avrl/exp/e09b_triple_replay.py --feats avrl/runs/p0/feats.parquet --selfverify avrl/runs/E05/selfverify.parquet --judgescores avrl/runs/L5a/judge_scores.parquet --out avrl/runs/L9b/result.json`.

**Pass threshold:** the three-way flag or blend exceeds agreement_only's errors-caught-per-budget by >=10% relative at the 25% budget point.

**Kill threshold:** the three-way combination is still at or below agreement_only at every budget, confirming that neither alternate proxy adds triage value even when tried together, and fully exhausting proxy-type combination variants for GSM8K.

**Expected runtime:** ~10 min, CPU-only, no GPU, reuses already-computed data from E05/L5a/p0.

**Informs:** closes the last untested combination on the proxy-type axis cheaply (each pairwise test already failed) before moving compute to task-type variation (E08) with no loose ends left on this axis.

### L9c: Execute L8a's MBPP dry run next, now doubly motivated

**Hypothesis:** with proxy-type variation now exhausted on rate, correlation, and operational-triage grounds for both self-verification (L6a) and judge (L7b), task-type variation (E08) is the only remaining live, zero-confound axis for claim 3; two consecutive proxy-type kills raise its priority over further proxy-type tuning (L9a, L9b).

**Change vs existing scripts:** none beyond L8a's existing spec (50-problem MBPP dry run, sandboxed execution harness validation); this entry only reaffirms L8a as the next step, now with L7b's result as additional motivation on top of L8a's original reasoning.

**Command shape:** per L8a's own section above.

**Pass/kill thresholds:** per L8a's own section above (engineering validation gate, not a claim-level pass/kill).

**Expected runtime:** per L8a's own section above (~20-25 min, goldberg GPU + CPU sandbox).

**Informs:** whether E08 can proceed safely next iteration, now the highest-priority remaining item given two consecutive proxy-type kills.

### L10a: Bootstrap CI for greedy_online vs static_matched on E06 and E08

**Hypothesis:** M01's regime diagnostic labeled E06 and E08 as "threshold wins" by inference from an unrelated citation (L6a/L7b's ranking-signal test), but each point's own already-computed sim_cell100_cheap.json replay shows greedy_online's point estimate beating static_matched at 25% and 50% budget (E06 +20.9%/+9.2%, E08 +8.3%/+11.6%); no CI has ever been run on this comparison for either point, so the direction is unverified for significance.

**Change vs existing scripts:** none; run avrl/exp/e09_bootstrap.py (already handles arbitrary feats.parquet, group_col, and feature-set arguments) pointed at avrl/runs/E06/feats.parquet and avrl/runs/E08/feats.parquet instead of E09's, same cell100/cheap/300-resamples/8-seeds config used for L9a-d.

**Command shape:** `.venv/bin/python avrl/exp/e09_bootstrap.py --feats avrl/runs/E06/feats.parquet --group_col cell100 --feature_set cheap --out avrl/runs/E06/bootstrap.json` (and the E08 equivalent).

**Pass threshold:** CI on the 25% relative gain (greedy_online vs static_matched) excludes zero on the positive side for either point, confirming the allocator-favoring point estimate M01's critic note found is a real effect, not resampling noise.

**Kill threshold:** CI includes zero at every budget for both points, meaning the point estimates M01 flagged were noise and the original "threshold wins" label was right for the wrong reason.

**Expected runtime:** ~15-30 min total, CPU-only, no GPU, reuses existing feats.parquet for both points.

**Informs:** directly resolves M01's open ground-truth question for 2 of the 9 ledger points, the single biggest source of ambiguity in M01's pass/inconclusive/kill call.

### L10b: Rerun M01's diagnostic with E06/E08 ground truth corrected

**Hypothesis:** once L10a supplies a real CI for E06 and E08, M01's 9-point pass/kill tally (currently 7/9 match by the script's own inferred labels, or 5/9 once those two labels are treated as contested) should be recomputed against the bootstrap-backed truth instead of either guess.

**Change vs existing scripts:** rerun avrl/exp/M01.py with the two POINTS entries for E06 and E08 updated to cite L10a's bootstrap.json verdict instead of the current L6a/L7b/prose inference; no other change.

**Command shape:** `.venv/bin/python avrl/exp/M01.py --runs_root avrl/runs --out avrl/runs/M01b/result.json` (copied script, new output dir, per the project's never-edit-in-place rule).

**Pass threshold:** n_match >= 8/9 against the corrected ground truth.

**Kill threshold:** n_mismatch >= 3/9 against the corrected ground truth.

**Expected runtime:** <1 min, CPU-only, depends on L10a.

**Informs:** replaces M01's provisional "corrected to kill" read with a number built entirely on bootstrap-backed ground truth for every point that has one, settling whether M02-M05 are worth any further investment.

### L10c: Sweep the regime cutoff against corrected ground truth

**Hypothesis:** the fixed 0.85 AUC cutoff sits close to several contested points (E06 0.887, E08 0.868, L5a 0.815) and may be a tuning problem rather than evidence that no single cutoff works.

**Change vs existing scripts:** add a sweep loop to a copy of avrl/exp/M01.py (or L10b's corrected variant) that tries auc_cutoff in {0.75, 0.775, ..., 0.90} and reports match count per cutoff against the corrected (post-L10a/L10b) ground truth; no new data.

**Command shape:** `.venv/bin/python avrl/exp/M01_sweep.py --runs_root avrl/runs --cutoffs 0.75:0.90:0.025 --out avrl/runs/M01_sweep/result.json`.

**Pass threshold:** some cutoff in the swept range achieves n_match >= 8/9, meaning the diagnostic just needs recalibration.

**Kill threshold:** no cutoff in the swept range clears 8/9, meaning a single AUC threshold cannot separate the two outcome groups and the single-statistic router-switch design needs a different signal entirely, not just a different number.

**Expected runtime:** ~5 min, CPU-only, depends on L10a/L10b.

**Informs:** whether M02-M05's regime switch is salvageable by recalibration alone or needs a redesign before any further GPU time is spent on it.

### L11a: Rerun M01's diagnostic now that L10a supplies real CIs for E06 and E08

**Hypothesis:** L10a (2026-10-01, PASS) confirmed with bootstrap CIs that E06 and E08 both belong in the "allocator wins" bucket at the project's primary 25% budget (and at 50%), settling the ground truth M01's critic note had only inferred. L10b's spec already covers this rerun; this entry marks it as unblocked and ready to execute exactly as specced, no new design needed.

**Change vs existing scripts:** none beyond L10b's own spec: rerun avrl/exp/M01.py with the E06 and E08 ground-truth entries updated to L10a's bootstrap.json verdicts (allocator wins at 25%/50%) instead of the citation-inferred "threshold wins" label.

**Command shape:** `.venv/bin/python avrl/exp/M01.py --runs_root avrl/runs --out avrl/runs/M01b/result.json` (copied script, new output dir, per the never-edit-in-place rule).

**Pass threshold:** n_match >= 8/9 against the corrected ground truth.

**Kill threshold:** n_mismatch >= 3/9 against the corrected ground truth.

**Expected runtime:** <1 min, CPU-only, depends on L10a (done).

**Informs:** replaces M01's provisional "corrected to kill" read with a number backed by bootstrap CIs for 7 of 9 ledger points instead of 5, settling whether M02-M05 are worth any further investment.

### L11b: Finer budget grid on E08 to locate the greedy_online/static_matched crossover

**Hypothesis:** L10a found a significant reversal for E08 between 10% budget (static_matched wins, -29.1% [-40.3%,-17.7%]) and 25% budget (greedy_online wins, +8.3% [+1.8%,+13.1%]), a pattern no other ledger point shows at the standard 10/25/50% resolution. This may track the online learner's warm-up window (R = max(budget//8, 40) in L10a_bootstrap_e06_e08.py), meaning very low budgets exhaust before greedy_online's self-trained ranking has enough verified labels to beat a pre-fit threshold.

**Change vs existing scripts:** copy avrl/exp/L10a_bootstrap_e06_e08.py, change BUDGET_FRACS to a finer grid (0.05, 0.10, 0.15, 0.20, 0.25), run on E08 only.

**Command shape:** `.venv/bin/python avrl/exp/L11b_e08_budget_grid.py --feats avrl/runs/E08/feats.parquet --cheap agreement,n_distinct,mean_tokens,std_tokens,any_truncated,log_qlen,frac_syntax_error,frac_exception --group_col cell100 --n_boot 300 --seed 0 --out avrl/runs/E08/bootstrap_fine.json`.

**Pass threshold:** the crossover budget (where the CI flips from excluding zero negative to excluding zero positive) is identified within one grid step, and its location correlates with the R warm-up formula's predicted onset.

**Kill threshold:** the CI is noisy/inconsistent across the finer grid with no identifiable monotonic crossover, meaning the 10%-vs-25% reversal is itself an artifact of coarse budget sampling rather than a real warm-up effect.

**Expected runtime:** ~20 min, CPU-only, reuses E08/feats.parquet, no GPU.

**Informs:** whether greedy_online has a documented minimum label-budget floor below which a fixed threshold is safer, a design-relevant finding for M02-M05 if the router is pitched as working at any label budget.

### L11c: Check for the same low-budget reversal on other ledger points

**Hypothesis:** if greedy_online's low-budget disadvantage (L10a, L11b) is a general cold-start property of the self-trained online allocator rather than an E08-specific artifact, it should also appear on E06 at a finer grid below 10%, and possibly on the weak-signal points (E09, E09b, E09d) which have only ever been reported at 10/25/50% budgets.

**Change vs existing scripts:** copy avrl/exp/L10a_bootstrap_e06_e08.py, change BUDGET_FRACS to (0.02, 0.05, 0.07, 0.10) for E06, and rerun the existing e09_bootstrap.py-style config at the same fine grid for E09/E09b/E09d using their existing feats.parquet.

**Command shape:** `.venv/bin/python avrl/exp/L11c_finegrid.py --feats avrl/runs/E06/feats.parquet --cheap agreement,n_distinct,mean_tokens,std_tokens,any_truncated,log_qlen --group_col cell100 --n_boot 300 --seed 0 --out avrl/runs/E06/bootstrap_fine.json` (and the E09/E09b/E09d equivalents).

**Pass threshold:** at least one other point shows the same low-budget reversal (CI excludes zero on the negative side at a budget where the 25%/50% CI excludes zero positive), confirming a general cold-start pattern.

**Kill threshold:** no other point shows any low-budget reversal, confirming E08's result is specific to that point (e.g. its small n=600 and higher proxy-error rate) rather than a property of greedy_online itself.

**Expected runtime:** ~15 min, CPU-only, reuses E06/E09/E09b/E09d feats.parquet, no GPU.

**Informs:** whether a minimum label-budget floor is a general requirement for any AVRL-Router design (M02-M05) or a narrow caveat limited to E08's regime.

## Harness status note (2026-09-30, by the main session)
The phase-1 GRPO harness exists and passed a smoke test on goldberg: avrl/p1_grpo.py plus avrl/allocators.py, run with /homes/rdubash/active-audit/.venv-rl/bin/python (vLLM, TRL 1.14, PEFT), model path models/Qwen2.5-1.5B-Instruct, data avrl/data/. Smoke (4 prompts, G=4, 3 steps): 21 s per step on the RTX 5090 with HF generation, peak 14.3 GB; log.jsonl carries per-step expensive_calls, corrupted_reward_groups, proxy_error_rate_among_unverified and periodic eval rows. Full settings (16 prompts, G=8, 512 tokens) are roughly 8x heavier per step, so a 400-step arm is about a day on the 5090 with HF generation; a vLLM-generation upgrade is being tested and the hannah job (four arms, 150 steps, job queued behind another user) is the parallel path. Experiments marked NEEDS PHASE-1 HARNESS are therefore runnable on goldberg at reduced steps (50 to 100) within the loop's 45-minute cap only for smoke-scale configs; prefer the analysis-only and rollout-only experiments until the vLLM path is confirmed.

## Priority note for the planner (2026-09-30, main session)
E01, E03 and L3a have settled the GSM8K plus majority-vote point: once the threshold baseline is budget-matched, learned and threshold tie (L3a CI includes zero), and the structure statistics are cheap-feature AUC 0.92 with neighbor ratio 1.0. Further budget, seed and granularity variants of that replay (L4a, L4b, L4c) have low information value now. The next iterations should populate the structure-statistic-versus-gain curve with NEW task and proxy pairs, in this order: E04 (small judge model as proxy, no self-consistency signal, for example Qwen2.5-0.5B-Instruct or Qwen3-0.6B asked "is this answer correct" with no gold), E05 (self-verification proxy), E08 (code task, MBPP, execution as verifier and a judge as proxy), E06 or E07 (model size variation), then G03 (the four-pair scan). Each must report the same three statistics as E01 (cheap-feature AUC for proxy error, embedding neighbor ratio, confident-subset error rate) plus the budget-matched replay at 10, 25 and 50 percent, so the points are comparable. Rollout generation for these fits the 45-minute cap at 1,500 to 2,000 prompts with vLLM in .venv-judge.

## Compute correction (2026-09-30, main session)
goldberg's GPU is an NVIDIA RTX A4000 with 16 GB, not a 32 GB card. Measured with the phase-1 harness (Qwen2.5-1.5B, LoRA r=32, bf16): full settings (16 prompts x G=8, 512 tokens) run at 153 s per step with HF generation and peak 13.2 GB; vLLM colocate does not fit beside training on this card (out of memory at engine start). Smoke settings run at 20 s per step. Plan GRPO experiments here at 8 prompts x G=8 (roughly 80 s per step) and 60 to 100 steps, or run them on hannah's A100 40GB nodes. vLLM-only rollout experiments (phase-0 style) are unaffected and fit in 40 minutes at 1,500 to 3,000 prompts.

## Priority note, second loop pass (2026-09-30, main session)
GSM8K is settled across four proxies and two model sizes (E01, E04/L5a, E05, E06, L6a, L7b): cheap-feature AUC 0.88 to 0.93, neighbor ratio 1.0, the agreement threshold dominates every alternative at every budget. Do not run any further GSM8K variant. The second pass has exactly two jobs, in order: (1) E08, the code task: MBPP (load openai/mbpp or google-research-datasets/mbpp from Hugging Face into avrl/data/, ungated), Qwen2.5-1.5B-Instruct or Qwen2.5-Coder-1.5B-Instruct, 8 rollouts per problem at temperature 1.0, expensive verifier = running the provided test cases in a subprocess with a 5 second timeout, cheap proxy = pass rate on self-written or model-proposed tests OR agreement of program outputs on the first test input (pick one, state it), and report the same three statistics as E01 plus the budget-matched replay; (2) if time remains under the 45-minute cap, E07 or a MATH-level-5 subset with majority vote, whichever is quicker, as the third task point. Then return NONE.

## E09 (added 2026-10-01): a task with no self-consistency signal
Rationale: every point so far (E01, L5a, E05, E06, E08) has a self-consistency confidence signal (agreement of sampled answers) with AUC 0.82 to 0.93 for proxy error, which puts them in the threshold regime. Claim 3 can only survive on a task where that signal is unavailable or weak.
Design: open-ended instruction following or summarisation where answers are free text. Dataset: a public preference or reference set with gold judgments, for example HelpSteer2 or UltraFeedback (gold = a strong reward model or the dataset's human score thresholded), or CNN/DailyMail or XSum with reference summaries (gold = a strong judge comparing to the reference). Policy: Qwen2.5-1.5B-Instruct, G=8 samples per prompt at temperature 1.0 for 1,500 prompts. Cheap proxy: a small reward model score (for example Skywork-Reward or an open 0.5B to 1.5B reward model) thresholded or ranked within the group; expensive verifier: a 7B judge (Qwen2.5-7B-Instruct, runs in 16 GB in bf16 with vLLM at short outputs) with a rubric, or the dataset's gold where available. Cheap features: proxy score and its within-group spread, response length stats, prompt length, plus an "agreement" analogue defined as the pairwise ROUGE-L or embedding cosine among the 8 samples (to test whether a consistency signal re-emerges for free text). Statistics: the same three as E01 (cheap AUC for proxy error, embedding neighbor ratio by cell, confident-subset error rate) and the budget-matched replay. Pre-registered read: if cheap AUC is under 0.75 and the neighbor ratio at k=100 exceeds 1.5, this is the first point in the allocator regime and phase 1 must be rerun on it; if cheap AUC is again above 0.85, the thesis result is that a consistency signal re-emerges even for free text and the threshold regime is general. Runtime: judge pass dominates, about 1,500 x 8 x 0.1 s on the A4000 at short outputs, under 45 minutes; mark GPU.

## Method and product experiments

Five entries proposed by the method-and-product synthesis (lit/avrl/method/METHOD_AND_PRODUCT.md), each runnable on the existing phase-0/phase-1 harnesses with no new infrastructure beyond the stated allocator classes.

### M01: Regime diagnostic, standalone validation across all nine existing points

**Hypothesis:** the two-statistic regime diagnostic (cheap-feature AUC, confident-subset embedding neighbor ratio) correctly labels which allocation policy wins, using only step-0-available data, on every point already in the ledger (E01, L5a, E05, E06, E08, E09, E09b, E09c, E09d).

**Change vs existing scripts:** add a `diagnose(feats_path)` function to `avrl/allocators.py` that reads an existing `feats.parquet` and returns a regime label (STRONG if AUC >= 0.85 else WEAK) and the recommended policy; no new generation, reuses files already on disk for all nine points.

**Command shape:** `.venv/bin/python avrl/exp/m01_diagnose_all.py --runs avrl/runs/{E01,L5a,E05,E06,E08,E09,E09b,E09c,E09d} --out avrl/runs/M01/result.json`.

**Pass threshold:** the diagnostic's STRONG/WEAK label matches the ledger's own verdict (threshold wins vs allocator beats threshold in replay with CI excluding zero) on at least 8 of 9 points.

**Kill threshold:** the diagnostic mislabels 3 or more of the 9 points, meaning the 0.85 AUC cutoff (or the neighbor-ratio bar) needs recalibration before any router can use it as a switch.

**Expected runtime:** ~10 min, CPU-only, no GPU, reuses already-computed data.

**Informs:** whether AVRL-Router's regime switch (algorithm step 2) is trustworthy before wiring it into any training run.

### M02: Recalibrated threshold vs fixed-once threshold, in training, strong-signal regime

**Hypothesis:** an online-percentile-recalibrated threshold (D2) matches the fixed-once threshold's corrupted-reward admission on GSM8K (already near-optimal per E01/P1-goldberg-s0) while removing the one tuning knob (a threshold value fit once on held-out data) the current harness still needs.

**Change vs existing scripts:** add `RecalibratedThresholdAllocator` to `avrl/allocators.py` (percentile refit every K=25 steps from the verification buffer, not from live routing decisions); add it as a third arm alongside the existing `ThresholdAllocator` and `LearnedAllocator` in `avrl/p1_grpo.py`.

**Command shape:** `.venv/bin/python avrl/p1_grpo.py --allocator recalibrated_threshold --budget_pct 0.25 --steps 100 --seed 0 --out avrl/runs/M02/seed0.jsonl` (and seed 1).

**Pass threshold:** recalibrated-threshold's corrupted-reward admission is within 10 percent relative of fixed-once's at matched budget, across both seeds, with no upward drift over 25-step windows (unlike the learned allocator's measured drift in P1-goldberg-s0).

**Kill threshold:** recalibrated-threshold's admissions drift upward like the learned allocator's did, meaning decoupling recalibration from routing did not fix the self-reinforcement failure, and the fixed-once threshold stays the deliverable for this regime.

**Expected runtime:** ~2-3 hours per seed on goldberg (matches existing P1-goldberg-s0 scale).

**Informs:** whether AVRL-Router's strong-signal branch needs the simpler fixed-once threshold or can safely use the recalibrated version (step 7 of the algorithm).

### M03: Structure-aware cell-bandit allocator, in training, weak-signal regime (the decisive test)

**Hypothesis:** the structure-aware allocator (cell Thompson sampling + classifier, D3, without the regime gate since this run is restricted to the known-weak-signal task) preserves in training the replay-measured gain over the budget-matched threshold (L9d: cell bandit +31 percent [14, 37]) on the open-ended HelpSteer2 task with the 7B judge.

**Change vs existing scripts:** add `StructureAwareAllocator` to `avrl/allocators.py` (Beta posteriors per cell, inverse-propensity-weighted classifier refit every M steps); add it as a fifth arm in `avrl/p1_judge_grpo.py` alongside `full`, `random`, `threshold`, `learned`.

**Command shape:** per the queued hannah four-arm job's config, with `--allocator structure_aware` added as a fifth arm, 150 steps, 2 seeds.

**Pass threshold:** structure-aware allocator beats the budget-matched threshold on corrupted-reward admission by a margin exceeding the seed spread, matching or exceeding the learned allocator's in-training showing (if any), at equal expensive-call budget.

**Kill threshold:** structure-aware allocator ties or loses to the budget-matched threshold in training, meaning the replay edge (L9a, L9d) does not survive policy drift, same failure mode already seen for the simpler learned allocator on GSM8K (P1-goldberg-s0); the weak-signal regime's deliverable becomes the budget-matched threshold too, and the paper's claim 2 becomes a replay-only result.

**Expected runtime:** matches the existing queued hannah job scale (4x A100, 150 steps, 2 seeds, several hours).

**Informs:** the paper's central in-training claim and whether AVRL-Router's weak-signal branch is worth shipping as a learned component at all.

### M04: Canary divergence alarm layered on the structure-aware allocator

**Hypothesis:** the canary-based divergence alarm (D5) detects an injected, ground-truth-labeled reward-hacking exploit on the open-ended task and reallocates budget toward the affected prompt cell faster than the structure-aware allocator alone (which has no hacking-specific trigger, only a general corrupted-reward-rate signal).

**Change vs existing scripts:** add the marker-string injection flag to `avrl/p1_judge_grpo.py` (bonus to the reward-model cheap score only, zero effect on judge score, as specified in D5 for `p1_grpo.py`, ported to the judge harness); add `DivergenceAllocator` to `avrl/allocators.py`, layered as an override on top of `StructureAwareAllocator`'s cell ranking (algorithm step 4's HOT-cell override); run three arms (budget-matched threshold, structure-aware alone, structure-aware plus canary alarm), 2 seeds, 150 steps.

**Command shape:** `.venv/bin/python avrl/p1_judge_grpo.py --allocator structure_aware_plus_canary --inject_exploit true --canary_frac 0.02 --budget_pct 0.25 --steps 150 --seed 0 --out avrl/runs/M04/seed0.jsonl`.

**Pass threshold:** the canary-augmented arm's injected hack_rate plateaus lower, or falls faster after onset, than the structure-aware-alone arm's, at matched total expensive-call budget, across both seeds.

**Kill threshold:** hack_rate trajectories are statistically indistinguishable between the two arms (CI overlap at every checkpoint), meaning the general corrupted-reward signal the allocator already tracks is sufficient and the canary mechanism adds nothing beyond D3 alone.

**Expected runtime:** ~2-3 hours per seed on goldberg (AWQ judge fits the 16 GB card at this scale per existing E09c/E09d runs).

**Informs:** whether contribution 3 of the paper (the canary alarm) is a real addition over the structure-aware allocator's existing signal, and whether AVRL-Router needs the canary layer at all or just the regime-conditional allocator.

### M05: AVRL-Router end-to-end, both regimes, no manual tuning

**Hypothesis:** a single router that runs the M01 diagnostic at step 0, switches to the recalibrated threshold on GSM8K and the structure-aware-plus-canary allocator on HelpSteer2 with no per-task hand-tuning, matches each task's own best single-arm result from M02, M03 and M04 at the same budget.

**Change vs existing scripts:** add a thin `Router` class in a new `avrl/router.py` that calls `diagnose()` (M01) once, then dispatches to `RecalibratedThresholdAllocator` or `StructureAwareAllocator` (+`DivergenceAllocator` if canary flagged) accordingly; wire it as a single `--allocator router` option in both `avrl/p1_grpo.py` and `avrl/p1_judge_grpo.py` so the same flag works on both harnesses.

**Command shape:** `.venv/bin/python avrl/p1_grpo.py --allocator router --budget_pct 0.25 --steps 100 --seed 0 --out avrl/runs/M05/gsm8k_seed0.jsonl` and the judge-harness equivalent on HelpSteer2.

**Pass threshold:** on each task, the router's corrupted-reward admission and held-out accuracy/judge-score are within the seed spread of that task's own winning arm from M02 (GSM8K) and M03/M04 (HelpSteer2), with the diagnostic correctly auto-selecting the policy that was pre-registered to win in M01.

**Kill threshold:** the router underperforms the known-best single arm on either task by more than the seed spread, or the diagnostic mis-selects the policy at step 0, meaning the router's dispatch logic (not the underlying allocators) is the bottleneck and the product ships as two separate configs rather than one auto-switching tool.

**Expected runtime:** reuses M02 and M03/M04's per-seed runtimes; no new generation beyond what those entries already specify.

**Informs:** whether the product's headline promise, one tool that picks the right policy in both of the project's own measured regimes without hand-tuning, is actually true, which is the condition the owner's goal (b) (an open framework people can install) depends on.

## Priority note, third loop pass (2026-10-01 evening, main session)
Run the method and product experiments M01 to M05 (above) in order of CPU-only first: anything that reuses saved rollouts (E01, E08, E09d under avrl/runs/) and needs no new generation runs now; GPU experiments must wait until nvidia-smi shows the card idle and no p1_judge_grpo or e10 process is running (the lr 1e-5 training pair and the E10 OLMo smoke and reference evaluations have priority on goldberg). Do not rerun anything already in the ledger. Each entry must report the same statistics and replay format as L9d so the headline figure (cheap AUC on x, allocator gain on y) can be drawn from the ledger.

### M06: In-training test on the setting where the replay predicts a learned-allocator win (GSM8K, 0.5B policy)
Hypothesis: on GSM8K with Qwen2.5-0.5B-Instruct (E06: cheap AUC 0.89, proxy error 38 percent, replay gain +21 percent for the self-training allocator), the learned allocator admits fewer corrupted rewards than the fixed agreement threshold in training at equal budget, reversing the P1-goldberg result obtained with the 1.5B policy (E01 setting, replay tie).
Run: avrl/p1_grpo.py (exact-answer harness), --model models/Qwen2.5-0.5B-Instruct (download if absent, ungated), arms threshold and learned, --budget 0.25 --steps 100 --prompts_per_step 8 --G 8 --max_completion_length 512 --lr 1e-5 --eval_every 25 --eval_n 300, seeds 0 and 1. Fits an A4000 (0.5B policy); about 40 s per step with HF generation.
Metric: corrupted-reward groups admitted per arm (primary), eval accuracy at step 100 (secondary). Pass: learned admits at least 15 percent fewer corrupted groups than threshold in both seeds. Kill: threshold admits fewer in both seeds. Otherwise inconclusive.
Why it matters: it is the first in-training test on a point where the replay favours the allocator with a strong cheap signal; a pass means the replay predicts training and the regime boundary is about proxy error rate, not just AUC; a kill means the replay overstates in-training gains and the in-training story stays "threshold wins" on exact-answer tasks.

### M07: Greedy self-training allocator in training (no cells), E01 setting
Hypothesis: the replay winner on every point is greedy_online (refit a logistic classifier on verified groups, verify the top-scored groups, no cell bandit); the in-training "learned" arm that lost on GSM8K (P1-goldberg s0, s1) was the bandit-flavoured allocator. In training on GSM8K with Qwen2.5-1.5B-Instruct at lr 1e-5, a GreedyOnlineAllocator admits fewer corrupted rewards than the fixed agreement threshold and than the bandit-flavoured learned allocator at equal budget.
Run: avrl/p1_grpo.py with a new allocator 'greedy' in avrl/allocators.py (cold start: threshold rule until 20 verified groups; then refit every 20 verified groups on all verified groups; score = predicted corrupted-reward probability; verify the top n_verify groups; identical budget accounting). Arms: threshold, learned, greedy; 100 steps, 8 prompts x G=8, 512 tokens, budget 0.25, lr 1e-5, seeds 0 and 1. goldberg 5090 after the open-ended run (about 60 s per step with HF generation, so roughly 1.7 h per arm) or an A4000.
Metric: corrupted groups admitted (primary), eval accuracy at step 100 (secondary). Pass: greedy admits at least 15 percent fewer corrupted groups than threshold in both seeds. Kill: threshold admits fewer in both seeds. The learned (bandit) arm is reported for the layout ablation.

Follow-ups from RESEARCH_LOG.md's M01b entry (2026-10-01), the bootstrap-CI-only re-test of M01's regime diagnostic across the 7 points with their own bootstrap.json. Result: KILL (3/7 match, 3/7 mismatch, 1 tie), worse than M01's hand-corrected 5/9. The bigger finding: E01, previously the "one true tie" (L3a, L4a), is not a tie under greedy_online (CI [+15.0%, +26.3%], fully positive); the earlier tie was ts_online (cell bandit) specifically.

### L12a: In-training test, greedy self-training allocator on E01's setting

**Hypothesis:** the in-training kill already on record for claim 2 (P1-goldberg-s0/s1: the bandit-flavoured LearnedAllocator admitted more corrupted-reward groups than the fixed threshold) is an artifact of testing the wrong allocator algorithm; a greedy self-training arm (refit a classifier each step on bought labels, rank unverified groups by it, no cell/Thompson-sampling layer) will admit fewer corrupted groups than the threshold in training on E01's exact setting, matching the replay direction M01b found for every bootstrap-backed point except E09c.

**Change vs existing scripts:** add a `GreedySelfTrainingAllocator` to `avrl/allocators.py` (classifier refit on verified labels each step, rank-based selection, explicitly no cell grouping or Thompson sampling, to isolate the mechanism from the existing bandit-flavoured `LearnedAllocator`); add it as a third arm in `avrl/p1_grpo.py` alongside `threshold` and `learned`.

**Command shape:** `.venv-rl/bin/python avrl/p1_grpo.py --allocator greedy_self_train --budget_pct 0.25 --steps 100 --seed 0 --lr 1e-5 --out avrl/runs/L12a/seed0.jsonl` on goldberg's 5090.

**Pass threshold:** greedy_self_train admits at least 15 percent fewer corrupted-reward groups than the fixed threshold at equal budget, reversing P1-goldberg-s0's result for the learned class.

**Kill threshold:** greedy_self_train admits as many or more corrupted groups than the threshold, same direction as the bandit-flavoured learned arm; the in-training kill for claim 2 stands regardless of allocator algorithm.

**Expected runtime:** ~2-3 hours on goldberg's 5090, one seed first (matches P1-goldberg-s0 scale).

**Informs:** whether claim 2's in-training kill is about learned allocation in general or specifically about the cell-bandit mechanism M03 is built around; the single highest-value open question M01b raises.

### L12b: ts_online bootstrap CIs on the remaining 5 points, to check the algorithm split

**Hypothesis:** the pattern M01b found on E01 (greedy_online wins, ts_online does not) generalizes: ts_online (cell bandit) will fail to beat static_matched's 25%-budget CI on E06, E08, E09, E09b and E09d as well, meaning the cell-structure mechanism M03 specs around is not where the allocator's edge comes from.

**Change vs existing scripts:** rerun avrl/exp/L10a_bootstrap_e06_e08.py and avrl/exp/e09_bootstrap.py's methodology with the comparison arm swapped from greedy_online to ts_online, same cell100/cheap features/300 resamples x 8 seeds, on E06, E08, E09, E09b, E09d (E01 and E09c already have this comparison from L3a/L4a and M01b respectively).

**Command shape:** `.venv/bin/python avrl/exp/L12b_ts_online_bootstrap.py --feats avrl/runs/E06/feats.parquet --group_col cell100 --n_boot 300 --seed 0 --out avrl/runs/E06/bootstrap_ts.json` (and the E08/E09/E09b/E09d equivalents).

**Pass threshold:** ts_online's CI excludes zero on the positive side on at least one of the 5 points, meaning the cell-bandit mechanism has real value somewhere and M03 is testing a viable design.

**Kill threshold:** ts_online's CI includes zero or is negative on all 5 points, confirming the algorithm split M01b found on E01 is general, not a one-point artifact.

**Expected runtime:** ~20 min total, CPU-only, reuses existing feats.parquet for all 5 points, no GPU.

**Informs:** whether M03 (structure-aware cell Thompson sampling) is worth any GPU time, or whether the project's real deliverable is a greedy self-training allocator with no cell layer at all.

### L12c: Tighter bootstrap on E09c, the one tie

**Hypothesis:** E09c's tie (CI [-5.7%, +33.7%], point estimate +11.7%) is a sample-size artifact, not a real null result; a bootstrap with more resamples, or one that pools E09c's error groups with E09b/E09d (same judge family, same rollout lineage), will resolve it to a positive CI like its siblings.

**Change vs existing scripts:** rerun E09c's existing bootstrap script with `--n_boot 1000` instead of 300, and separately with E09b/E09d's error groups concatenated in as additional resamples, same cell100/cheap features/25% budget setup.

**Command shape:** `.venv/bin/python avrl/exp/e09_bootstrap.py --feats avrl/runs/E09c/feats.parquet --group_col cell100 --n_boot 1000 --seed 0 --out avrl/runs/E09c/bootstrap_n1000.json`.

**Pass threshold:** the wider-resample or pooled CI excludes zero on the positive side, resolving E09c to a match under M01b's rule (7/7 bootstrap-backed points would then split 4 match / 3 mismatch, still clearing the kill bar but removing the one "no call").

**Kill threshold:** the CI still includes zero at 1000 resamples, confirming E09c is a genuine tie and not a sample-size artifact.

**Expected runtime:** ~15 min, CPU-only, reuses E09b/E09c/E09d feats.parquet, no GPU.

**Informs:** whether M01b's tally has a real tie or just an underpowered point; low priority relative to L12a and L12b since it does not change the kill verdict either way.

Follow-ups from RESEARCH_LOG.md's L12b entry (2026-10-01), the tsonline-vs-static_matched tally across the same 7 bootstrapped points M01b used. Result: KILL of the Note's "never" claim (2/7 points win: E09, E09d; 0/7 lose; 5/7 tie), reviving a narrow, non-uniform foothold for claim 3's cell-structure mechanism.

### L13a: Head-to-head, greedy_online vs ts_online on the 4 low-cheap-AUC points

**Hypothesis:** ts_online's wins over static_matched on E09 and E09d (the two low-cheap-AUC points where it won) represent a real incremental gain over greedy_online specifically, not just a weaker version of the same classifier-refit effect; a direct greedy_online-vs-ts_online comparison will show ts_online ahead, behind, or tied on each of the 4 low-AUC points (E09, E09b, E09c, E09d).

**Change vs existing scripts:** copy avrl/exp/L12b_tsonline_tally.py to a new script with the comparison arm swapped to greedy_online vs ts_online (both already present in every point's bootstrap.json comparisons list under the same cell100/cheap/25%-budget spec); no new resampling.

**Command shape:** `.venv/bin/python avrl/exp/L13a_greedy_vs_ts.py --points E09,E09b,E09c,E09d --group_col cell100 --feature_set cheap --out avrl/runs/L13a/result.json` (reads existing bootstrap.json files only).

**Pass threshold (for claim 3's narrow-foothold reading):** ts_online beats greedy_online (CI excludes zero positive) on at least one of E09/E09d, showing the cell-bandit mechanism adds something greedy_online's classifier refit does not.

**Kill threshold:** greedy_online matches or beats ts_online (CI ties or excludes zero negative for ts_online) on all 4 points, meaning ts_online's win over static_matched is a weaker echo of the same classifier-refit effect, not a distinct structure-exploiting mechanism.

**Expected runtime:** ~15 min, CPU-only, reuses existing bootstrap.json files for E09/E09b/E09c/E09d, no GPU.

**Informs:** whether claim 3 (prompt-space cell structure is a real, separate source of allocator gain) has any evidence beyond "ts_online beats a budget-matched threshold somewhere," which is the single highest-value open question L12b raises.

### L13b: Tighter bootstrap on E09b and E09c's ts_online ties

**Hypothesis:** E09b and E09c tied (not won) despite sharing E09/E09d's low-cheap-AUC profile; this is a power problem from fewer error groups and wide CIs, not a genuine null, and 1000 resamples (versus the current 300) will resolve at least one tie to a win or a confirmed null.

**Change vs existing scripts:** rerun avrl/exp/e09_bootstrap.py (already used for E09b/E09c) with `--n_boot 1000` instead of 300, same cell100/cheap features/25% budget setup, ts_online vs static_matched comparison.

**Command shape:** `.venv/bin/python avrl/exp/e09_bootstrap.py --feats avrl/runs/E09b/feats.parquet --group_col cell100 --n_boot 1000 --seed 0 --out avrl/runs/E09b/bootstrap_n1000.json` (and the E09c equivalent).

**Pass threshold:** at least one of E09b/E09c resolves to a significant positive CI at 1000 resamples, meaning claim 3's foothold is wider than L12b's 300-resample tally showed.

**Kill threshold:** both CIs still straddle zero at 1000 resamples, confirming the ties are real and claim 3's foothold is limited to E09 and E09d specifically, not a general low-AUC-point effect.

**Expected runtime:** ~20 min, CPU-only, reuses E09b/E09c feats.parquet, no GPU.

**Informs:** whether claim 3's revived foothold is a stable two-point effect or could extend to more points given more power; low priority relative to L13a since it does not change L12b's kill verdict either way.

### L13c: Budget-stability check on E09/E09d's ts_online wins

**Hypothesis:** E09 and E09d's significant ts_online-vs-static_matched wins at 25% budget are stable across budget levels, not a 25%-only artifact; L10a already found a budget-dependent crossover for a different comparison (E08's greedy_online vs static_matched flips sign between 10% and 25%), so the same risk applies here.

**Change vs existing scripts:** none; L12b's result.json already contains the 10%/25%/50% CIs for every point in its `all_budgets` field (zero marginal compute). This is a pure read-and-tabulate pass.

**Command shape:** `.venv/bin/python -c "import json; d=json.load(open('avrl/runs/L12b/result.json')); [print(r['point'], r['all_budgets']) for r in d['rows'] if r['point'] in ('E09','E09d')]"`.

**Pass threshold:** E09 and E09d's CIs exclude zero on the positive side at all three budgets (10%, 25%, 50%), confirming a stable win.

**Kill threshold:** either point's CI flips to include zero or go negative at 10% or 50%, meaning the win is budget-dependent and should be reported as such rather than as a general mechanism result.

**Expected runtime:** under 1 min, CPU-only, no new compute, reads L12b's existing result.json only.

**Informs:** whether E09/E09d's wins can be cited as a stable mechanism result or need a budget-dependent caveat, before citing them in any claim-3 writeup.

Follow-ups from RESEARCH_LOG.md's L13a entry (2026-10-01), the greedy_online-vs-ts_online head-to-head on the two points where ts_online beat static_matched in L12b. Result: INCONCLUSIVE on both points (E09 CI [-9.8%, +12.2%], E09d CI [-15.0%, +2.7%], both include zero), so the claim-3 revival is neither confirmed nor killed.

### L14a: Higher-resolution bootstrap on E09d, the closer-to-significant point

**Hypothesis:** E09d's CI [-15.0%, +2.7%] (ts_online numerically ahead, point estimate -9.2% for greedy_online) is the one point in L13a close enough to significance that more resamples could resolve it; 1000 resamples instead of 300 will show whether ts_online has a real incremental edge over greedy_online there.

**Change vs existing scripts:** rerun avrl/exp/L13a_greedy_vs_ts.py unmodified except `--n_boot 1000` (currently 300), E09d only.

**Command shape:** `.venv/bin/python avrl/exp/L13a_greedy_vs_ts.py --feats avrl/runs/E09d/feats.parquet --label proxy_error --group cell100 --features rm_max,rm_mean,rm_std,rm_gap12,mean_tokens,std_tokens,trunc_frac,consistency,log_qlen --n_boot 1000 --seeds 8 --out avrl/runs/L14a/E09d_bootstrap.json`.

**Pass threshold:** the 1000-resample CI excludes zero on the negative side for greedy_online (i.e. ts_online significantly ahead), confirming a real incremental ts_online effect on E09d.

**Kill threshold:** the CI still includes zero at 1000 resamples, confirming L13a's tie on E09d is a genuine null, not an underpowered 300-resample artifact.

**Expected runtime:** ~15 min (roughly 3x E09d's 289s L13a runtime), CPU-only, reuses E09d/feats.parquet, no GPU.

**Informs:** whether E09d is the one point left that could still revive claim 3, before concluding the head-to-head comparison is flatly inconclusive everywhere.

### L14b: Complete the 4-point greedy_online-vs-ts_online picture on E09b/E09c

**Hypothesis:** E09b and E09c tied against static_matched in L12b (not won like E09/E09d), so a greedy_online-vs-ts_online head-to-head on those two completes the picture of whether ts_online ever adds value over greedy_online across all 4 low-cheap-AUC points, not just the 2 that won.

**Change vs existing scripts:** rerun avrl/exp/L13a_greedy_vs_ts.py unmodified on E09b and E09c's feats.parquet.

**Command shape:** `.venv/bin/python avrl/exp/L13a_greedy_vs_ts.py --feats avrl/runs/{E09b,E09c}/feats.parquet --label proxy_error --group cell100 --features rm_max,rm_mean,rm_std,rm_gap12,mean_tokens,std_tokens,trunc_frac,consistency,log_qlen --n_boot 300 --seeds 8 --out avrl/runs/L14b/{E09b,E09c}_bootstrap.json` (two parallel CPU processes).

**Pass threshold:** ts_online's CI excludes zero on the positive side on either E09b or E09c, showing an incremental effect exists even on points that tied against static_matched.

**Kill threshold:** both CIs tie or favor greedy_online, meaning ts_online never adds value over greedy_online on any of the 4 low-cheap-AUC points, closing claim 3's revival completely.

**Expected runtime:** ~20 min total (roughly 10-11 min each based on E09/E09d's observed runtimes), CPU-only, reuses E09b/E09c feats.parquet, no GPU.

**Informs:** whether claim 3's narrow foothold is worth any further investigation at all, or whether the full 4-point picture closes it for good.

### M08: Greedy self-training allocator on the open-ended setting
Hypothesis: the replay winner (greedy_online) admits fewer corrupted rewards than the threshold and the bandit-flavoured learned arm in training on the open-ended task, where the bandit-flavoured arm only tied (P1-judge-5090-s0).
Run: avrl/p1_judge_grpo.py --arm greedy (GreedyOnlineAllocator), goldberg 5090, 4-bit 7B judge, 8 x G=8, 384 tokens, lr 1e-5, 150 steps, budget 0.25, seeds 0 and 1, alongside the 5090b threshold and learned reruns. Metric: corrupted groups admitted and eval judge score at 150 steps. Pass: greedy admits at least 15 percent fewer corrupted groups than threshold in both seeds. Kill: threshold admits fewer in both.

### M09: Consistency-ranked label-free threshold on the open-ended task
Hypothesis: the in-training threshold arm on the open-ended task ranked by the reward-model top-two gap, whose AUC for proxy error is only 0.51 to 0.57 (near chance), so M08a's self-trainer win may be a win over a near-random rule. A label-free threshold ranking by group consistency (mean pairwise MiniLM cosine among the 8 responses, AUC 0.69 on E09b/E09d) is the fair fixed rule for free text; if the self-trainer still admits at least 15 percent fewer corrupted rewards than it in both seeds, claim 2 stands on this task; if they tie, the open-ended result reduces to "use consistency as the threshold".
Run: avrl/p1_judge_grpo.py --arm threshold --threshold_feature consistency, same configuration and seeds as P1-judge-lr5 and M08a (spruce A4000). Compare to M08a's greedy 30 / 26.

### M10: Full-verification and proxy-only reference arms, GSM8K 1.5B (M07 config)
Why: claim 1 ("equal quality at 25 percent budget") has never been measured against the full-verification arm in the same setting, and the canary alarm (M04) assumes the proxy-only arm's error drifts upward as the policy learns to exploit the cheap grader. Both reference arms are missing from every local panel.
Run: avrl/p1_grpo.py --arm full and --arm proxy, Qwen2.5-1.5B-Instruct, GSM8K, 100 steps, 8x8x512, lr 1e-5, eval_n 300, seeds 0 (spruce) and 1 (cherry). Launcher avrl/m10_refs.sh.
Pre-registered: claim 1 PASS if the M07 threshold arm's final accuracy (0.690 / 0.667) is within 0.03 of the full arm's on both seeds; KILL if full exceeds threshold by more than 0.03 on both seeds. M04 premise PRESENT if the proxy arm's proxy error rate in the last 25 steps exceeds the first 25 steps by at least 50 percent relative on both seeds; ABSENT otherwise (then the alarm has nothing to detect on this task at this horizon).

### Product v0 done (2026-10-02): avrl/avrl_router/diagnose.py
The standalone regime diagnostic from METHOD_AND_PRODUCT.md section 4 exists and is validated on seven saved points (ledger entry DIAG-v0, table in avrl/runs/diag_v0/summary.md). Next product pieces, in order: step-0 diagnostic report inside the TRL harness (reuse diagnose() on the first K verified groups), then the drift alarm pending M10.

### H1: per-group record logging in all three harnesses (infrastructure, do between runs)
Why: M09b seed 0 (7B judge) shows the consistency threshold tying the RM-gap rule in training (61 vs 64 corrupted at step 144) although the E09d replay on the same judge predicts a 34 percent reduction vs random for consistency and 7 percent for the gap. The harnesses only write per-step totals, so the in-training AUC of the confidence feature (and whether it decays as the policy sharpens) cannot be computed. Add one jsonl row per group (step, idx, confidence features, verified flag, proxy score, expensive label when known) to p1_grpo.py, p1_judge_grpo.py, p1_code_grpo.py. Logging only, no effect on allocation. Apply after the M09b and M10 chains finish so no run changes mid-chain. This is also the input the step-0 diagnostic report (product piece 2) needs.

### E10b: OLMo 2 1B Instruct, visible-test cheap grader vs full-test verifier, curriculum of solvable problems
Why: E10 showed the signature vote is wrong on 99 percent of groups for a weak policy and RL from base has no signal (E10p0). E10b asks the question the way real code RL is run: cheap grader = the prompt's visible first assert (1 to 2 percent disagreement at step 0), expensive verifier = the full assert set, policy = OLMo 2 1B Instruct, training prompts restricted to the 108 problems Instruct solves at least once in 8 samples (avrl/runs/E10p0/instruct_solvable_task_ids.json).
Run: avrl/p1_code_grpo.py --proxy visible --curriculum <ids.json> --model OLMo-2-0425-1B-Instruct, arms full / proxy / random / threshold at 25 percent budget, 300 steps, 8 x 8 x 512, lr 1e-5, LoRA r 32, eval every 50 on 200 MBPP test problems, HumanEval at the end. Corrupted group = at least one completion passing the visible test but failing the full asserts (false negatives are impossible because the visible assert is part of the full set). Threshold confidence = signature agreement among the completions that pass the visible test (1.0 when none pass, since no false positive is then possible). Seed 0 four arms on hannah (sbatch after 78891); seed 1 proxy and full arms on goldberg.
Pre-registered: HACKING PRESENT if the proxy arm's corrupted-group rate over steps 250 to 300 is at least twice its rate over steps 0 to 50; CLAIM 1 PASS if threshold at 25 percent matches the full arm's final MBPP pass@1 within 2 points; THRESHOLD USEFUL if its corrupted-group count is at most 0.7 times random's; BEATS INSTRUCT REFERENCE if the full arm's final MBPP pass@1 exceeds 14.5 percent and HumanEval exceeds 18.9 percent (local reference evals).

### E10c: does the visible-test grader get gamed over a longer horizon? (proxy-only arm, 600 steps)
Why: E10b's proxy arm showed no false-positive drift over 300 steps and M10's majority-vote arm none over 100; the "drift is absent" conclusion is horizon-limited. Run: avrl/p1_code_grpo.py --arm proxy --proxy visible, 600 steps, same lr 1e-5 and curriculum, seed 1, spruce A4000 (launcher avrl/e10c_spruce.sh). Pre-registered: DRIFT PRESENT if the false-positive group rate over steps 550 to 600 is at least 1.5x the rate over steps 0 to 50; also report held-out MBPP and HumanEval at 300 vs 600 (does longer training keep helping or start hurting held-out quality while the visible-test reward keeps rising).
E10b seed 2 (threshold and random arms) runs on cherry (avrl/e10b_seed2_cherry.sh) for three seeds on the key comparison.

### M11: full and random reference arms for the open-ended 3B-judge panel
Why: the open-ended panel (P1-judge-lr5, M08a, M09: threshold-gap, greedy, learned, threshold-consistency at 25 percent) has no full-verification bound and no random-allocation control, and hannah's bf16 7B-judge four-arm job has OOMed once. Run: avrl/p1_judge_grpo.py --arm full and --arm random, M09 configuration (Qwen2.5-1.5B-Instruct, e09b prompts, 4-bit Qwen2.5-3B judge, 100 steps, 8 x 6 x 320, lr 1e-5), seeds 0 and 1; spruce runs full, cherry runs random (launcher avrl/m11_refs.sh). Pre-registered: claim 1 PASS on the open-ended task if the consistency threshold's final judge score (M09: 4.68 / 4.565) is within 0.15 of the full arm's on both seeds; threshold USEFUL if its corrupted count (23 / 25) is at most 0.7 x random's.

### E10d: visible-test cheap grader with a second, stronger policy (Qwen2.5-1.5B-Instruct on MBPP)
Why: E10b's result (budget changes corruption, not held-out quality; threshold removes most wrong rewards) is one policy and one curriculum. Qwen2.5-1.5B-Instruct solves 65 percent of MBPP training problems at pass@8 (E08), so no curriculum is needed and the visible-test false-positive rate will differ. Run: run_e10d_hannah.sbatch, four arms (full, proxy, random, threshold at 25 percent), 300 steps, all 464 training problems, then the untrained reference evaluation. Pre-registered: same four tests as E10b (hacking, claim 1 within 2 points of full, threshold at most 0.7x random's corruption, beats the untrained reference on MBPP test and HumanEval).

### E10e: the product loop end to end (burn-in, diagnose, auto budget)
Run: E10b setting, --arm threshold --burnin_steps 25 --budget auto --target_corruption 0.02, seed 1, goldberg after E10c (launcher avrl/e10e_goldberg.sh). Pre-registered: report the diagnostic's chosen budget and regime; PASS if the auto arm's corrupted-group rate after the burn-in is at or below the target 0.02 within its bootstrap interval, or if the diagnostic recommended 1.0 (full) because no budget met the target; and its final held-out MBPP within 2 points of the full arm. The point is whether a user who knows nothing about the regime gets a sensible budget from 200 labelled groups.

### M12: separate label quality from reward shape on the open-ended task
Why: M11's full arm beat every quarter-budget arm on held-out judge score (4.92 vs 4.68), but in the judge harness the proxy reward is rank-based (top half of the group by reward-model score) while the judge reward is absolute (score at or above 7), so the full arm changes both the labels and the reward shape. Run: add --gold_reward_shape {absolute,rank} to p1_judge_grpo.py; "rank" converts verified groups' judge scores to the same top-half-by-score rule the proxy uses. Arms: full with rank-shaped judge rewards (M12 full_rank), M09 configuration, seeds 0 and 1, cherry and spruce after M11. Pre-registered: if full_rank's final judge score is within 0.1 of full (absolute), the gain comes from the labels and claim 1 fails on free text; if full_rank falls to the quarter-budget arms' level (about 4.6 to 4.7), the gain was reward shape and claim 1 stands with the proxy reward redefined.

### M11 seed 2 and M09 seed 2: third seed for the open-ended full-vs-threshold comparison
Why: the full-absolute arm's two seeds differ by 0.28 (4.92 vs 4.64) and M12 attributes the seed-0 excess to reward shape; a third seed of both the full-absolute arm and the consistency threshold decides whether 4.92 recurs. Run: avrl/m11_refs.sh full 2 (spruce) and avrl/m09_seed.sh 2 (cherry). Pre-registered: claim 1 like for like already passes (M12); this pair updates the absolute-reward comparison: PASS if the three-seed mean gap between full-absolute and the consistency threshold is at most 0.15, otherwise the absolute reward shape is reported as a real advantage of full judging on free text.

### PRODUCT-3: relative corruption target as the default for --budget auto
Why: E10e's burn-in on 200 real groups recommended an 85 percent budget for the absolute 2 percent target although the threshold at 25 percent already cut corruption from 16 to 4 percent. Change: --target_corruption accepts "rel:0.25" (a fraction of the measured base error) and the default becomes rel:0.25; the report prints both the absolute and relative readings. Validate on the saved E10e burn-in rows (expect budget 0.25). Also add the budget-versus-corruption curve to the report as a small table so a user can override by eye.

### M14: threshold at 10 percent budget on GSM8K 1.5B (dose-response for claim 1)
Why: claim 1 holds at 25 percent; the diagnostic recommends 0.30 for 2 percent corruption on this task; the product recipe needs to know how low the budget can go. Run: avrl/p1_grpo.py --arm threshold --budget 0.10, M07 configuration, seeds 0 (spruce) and 1 (cherry), launcher avrl/m14_budget10.sh. n_verify = round(0.10 x 8) = 1 of 8 groups per step. Pre-registered: PASS if final accuracy is within 0.03 of the full arm (0.663) on both seeds; report corrupted groups against the random expectation at 10 percent (0.9 x proxy-arm rate, about 86 of 700).
Third seed of the open-ended full and consistency arms moved to goldberg (avrl/m13_goldberg_seed3.sh, after E10e), since seeds 2 and 3 OOM at steps 35 and 18 on the 16 GB cards.

### E10e-rel: the auto-budget loop with the relative default target
Run: E10e setting and seed with --target_corruption rel:0.25 (launcher avrl/e10e_rel_goldberg.sh, after M13 on goldberg). Pre-registered: the burn-in should choose a budget near 0.15; PASS if post-burn-in corruption is at or below 0.04 (a quarter of the 0.16 base error) and final held-out MBPP within 2 points of the full arm, with total expensive calls under 700.

### H2: fixed evaluation subset across seeds (infrastructure) and a common re-evaluation of saved adapters
Why: see the ledger note of 2026-10-02 20:50 UTC; every harness draws its held-out subset with the run seed. Part 1 (agent, in progress): --eval_seed in all three harnesses, default None keeps old behaviour; use --eval_seed 0 in every new sweep. Part 2 (GPU, after the queues drain): re-evaluate every saved final adapter of the E10b, E10d, M07, M10, M09, M11, M12, M13 runs on one fixed subset (eval_seed 0, 300 GSM8K problems, 200 HelpSteer2 prompts, the full 500 MBPP test set) so the cross-seed and reference comparisons in Figures 2 and 3 can be restated on common ground. Needs a small driver that walks the run directories, loads each final adapter with --eval_only, and writes one table.

### PRODUCT-4: realizable budgets and a conservative pick in --budget auto
Why: E10e-rel's burn-in recommended 0.15 for a 4 percent target, but with 8 groups per step the harness verifies round(0.15 x 8) = 1 group per step (12.5 percent), and the realized post-burn-in corruption was 7.4 percent. Two fixes: (1) restrict the recommendation to budgets realizable at the configured groups per step (multiples of 1/n_groups) and round up, not to nearest; (2) pick the smallest realizable budget whose threshold-corruption upper confidence bound (not the point estimate) meets the target. Validate on the saved E10e burn-in rows (expect 0.25 for the 4 percent target: realizable, upper bound 0.06 at 0.25 vs point 0.04, so possibly 0.375) and rerun E10e-rel once.

### E10e-up: the auto-budget loop with realizable candidates and the upper-bound pick
Run: E10e setting and seed with --budget_pick upper --eval_seed 0 (launcher avrl/e10e_up_goldberg.sh). Pre-registered: burn-in should choose 0.625 (5 of 8 groups); PASS if post-burn-in corruption is at or below 0.04 and final held-out MBPP within 2 points of the full arm; report total calls (expected about 25 x 8 + 275 x 5 = 1575).

### M15: two more paired comparisons for claim 2 in the low-error regime (4-bit 7B judge, seeds 2 and 3)
Why: the greedy self-trainer's corrupted count is below the consistency threshold's in all five 7B-judge comparisons (233 vs 258); two more pairs on goldberg (plus hannah's seeds 3 and 4 with the bf16 judge) decide the sign test. Run: avrl/m15_goldberg.sh, the 5090b configuration (150 steps, 8 x 8 x 384, 4-bit 7B judge, --eval_seed 0, fixed evaluation), greedy then consistency threshold for seeds 2 and 3. Pre-registered: with nine comparisons, PASS for a real small edge if greedy is lower in at least eight and the pooled ratio is below 0.9; otherwise claim 2 stays "tie in the low-error regime".

### S1: scale-up on hannah (7B policy, 14B judge, 1000-step horizon on the open-ended task)
Why: the project's conclusions came from 0.5B to 1.5B policies, 3B to 7B judges and 100 to 300 steps; the two findings most exposed to that are "the budget does not change held-out quality" and "no cheap grader was gamed". The reward-model overoptimization literature predicts that a larger policy trained longer against a small reward model will raise its proxy reward while its true quality falls.
Run: avrl/run_s1_hannah.sbatch. Policy Qwen2.5-7B-Instruct (LoRA r32, lr 1e-5, no KL penalty, gradient checkpointing), cheap grader Skywork-Reward-V2-Qwen3-0.6B (top-half rank reward, unchanged), expensive grader Qwen2.5-14B-Instruct bf16 judge on its own GPU (absolute reward, score at least 7; the same judge scores the held-out 200 prompts every 100 steps with eval_seed 0), data e09d (3800 training prompts, about two epochs at 1000 steps), 8 prompts x 8 completions x 384 tokens. Arms at 25 percent budget: proxy, consistency threshold, greedy self-trainer, full; two arms per job on four GPUs, pair a (proxy, consistency) then pair b (greedy, full) chained. Step count set from the probe so each job ends inside the 3-day limit (target 1000).
Pre-registered (revised 2026-10-05 after three design reviews): DRIFT PRESENT if the proxy arm's corrupted-group rate over its last 200 steps is at least 1.5x the rate over its first 200 steps, with the test void if fewer than 600 steps complete or the first window holds fewer than 10 corrupted groups. OVEROPTIMIZATION PRESENT if the proxy arm's held-out judge score at its last two evaluations is at least 0.15 below its peak while the per-step mean raw reward-model score (new log field rm_score_mean, not the rank-based train reward, which is 0.5 by construction) over the same window is above its value over the first 200 steps. CLAIM 1 AT SCALE PASS if the consistency threshold's final held-out score is within 0.15 of the full arm's on the mean of two seeds (seed 1 of both arms runs as the third job). CLAIM 2 AT SCALE: single seed, reported only. Arms and jobs: job a = proxy and consistency threshold (seed 0); job b = full (seed 0) and a second proxy arm at learning rate 3e-5 (the drift probe under more optimization pressure, as the review recommended; the other arms keep 1e-5 for comparability with the small-scale panels); job c = consistency threshold and full at seed 1. Jobs chained with slurm dependencies and the singleton rule. Generation runs at 2 prompts x 8 completions per step (4 x 8 overflowed the 40 GB policy card at step 338 of the first attempt; the reward model now sits on the judge's card), with 4000 steps per arm to keep 8000 groups; the drift and overoptimization windows above are in those steps (200 steps = 400 groups). The 25 percent budget at 2 groups per step is realized by cumulative rounding (one verified group on alternate steps), which gives the threshold rule only a two-way choice; its corruption reduction is therefore a lower bound on the rule's small-batch behaviour.
Caveats to state: the judge is a larger member of the policy's own family (in-family bias possible); the full arm trains against the same judge object that scores its own held-out set, so its score is not an unbiased ceiling; the cheap reward is rank based while the judge reward is absolute, which M12 showed moves held-out scores on its own, so any claim 1 gap at scale may be that shape effect rather than the labels (a full arm with rank-shaped judge reward at scale is the follow-up, S1-rank). Follow-up S2: the proxy arm with a KL penalty and a longer horizon if neither drift test fires.
