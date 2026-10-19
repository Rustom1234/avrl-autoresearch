# AVRL Experiment Backlog

Pre-registered, 2026-09-30. Twenty experiments, E01 through E20, ordered by expected information value per GPU hour: cheapest, highest-leverage checks first, most expensive and most infrastructure-blocked last. This ordering means analysis-only reruns of already-collected data (near-zero marginal GPU cost) come before any new generation run, and new generation runs on this machine come before any GRPO training run on an A100.

## Claim numbering (from PROJECT.md)

- Claim 1: the learned allocator reaches the same policy quality with a fraction of expensive verifier calls.
- Claim 2: the learned allocator admits fewer corrupted rewards into training than any fixed rule at the same budget.
- Claim 3: the allocator's edge comes from structure in prompt space (proxy error clusters), the SOFisher condition (Li et al., 2026, Nature Communications, https://doi.org/10.1038/s41467-026-73404-6).

Source for the existing harness read this session: /homes/rdubash/avrl_live/FULL_R/avrl/PROJECT.md, /homes/rdubash/avrl_live/FULL_R/avrl/p0_rollouts.py, /homes/rdubash/avrl_live/FULL_R/avrl/p0_analyze.py, /homes/rdubash/avrl_live/FULL_R/avrl/allocators.py. Confirmed at read time: avrl/p1_grpo.py (the phase-1 GRPO training harness named in PROJECT.md's Files section) does not exist on disk yet, only p0_rollouts.py, p0_analyze.py, and allocators.py do. Every experiment below that needs an actual policy-training loop is marked NEEDS PHASE-1 HARNESS.

## Runtime method note

p0_rollouts.py's own run.log (avrl/runs/p0/run.log, in progress at read time) shows 3,000 GSM8K prompts times G=8 rollouts (24,000 completions) on Qwen2.5-1.5B-Instruct at max_tokens=512 reaching 47% of prompts in about 7.5 minutes, aggregate throughput near 2,460 input tok/s and 5,470 output tok/s on this machine. Extrapolating linearly gives roughly 16 minutes of generation for that exact config, leaving headroom for the embedding plus allocation-replay analysis step under a 40-minute cap. All runtime estimates below for new model or prompt-count combinations are extrapolations from that single observed data point, not measurements, and are flagged as estimates. Before committing a full run, a 100-prompt dry run with the same --model and --G is recommended to calibrate tokens/sec for that specific model before trusting the 40-minute estimate.

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

**Expected runtime:** ~16 min generation + ~5-10 min analysis, this machine, per the run.log extrapolation above.

**Informs:** claim 3 (foundational precondition for all other experiments).

---

## E02: Budget-sweep extension (analysis-only rerun)

**Hypothesis:** the gap between the learned allocator and the threshold allocator (proxy errors caught per verifier call) is not flat across budgets; it is largest at low budgets, where a fixed threshold's single cut point is most information-starved.

**Change vs existing scripts:** no new rollouts. Reuse E01's feats.parquet. Edit p0_analyze.py's sim_alloc invocation: replace `BUDGET_FRACS = 3, 1, 8, (0.10, 0.25, 0.50)` with a finer grid, e.g. (0.02, 0.05, 0.10, 0.15, 0.20, 0.25, 0.35, 0.50, 0.75, 0.90), for both cell100 and cell30 grouping and both cheap and cheap+emb feature sets.

**Command shape:** `python avrl/p0_analyze.py` with the BUDGET_FRACS substitution above applied to the sim_generic.py template before it runs (same replace-and-write pattern p0_analyze.py already uses on `runs/sim_generic.py`).

**Metric:** hit rate (proxy errors caught per verifier call) for random/threshold/learned/oracle at each budget point; plot hit-rate-vs-budget curves per allocator.

**Pass threshold:** learned allocator's hit-rate advantage over threshold is >= 10 percentage points at at least one budget point in the 5-25% range (the range PROJECT.md's phase-1 plan actually intends to spend, 25%).

**Kill threshold:** learned and threshold hit-rate curves are within 3 percentage points of each other at every budget point tested; a single fixed threshold is budget-invariant-equivalent to the learned allocator on this data.

**Expected runtime:** under 10 min, this machine (CPU-bound analysis step only, no new GPU generation).

**Informs:** claims 1 and 2 (identifies the budget regime where a learned allocator is worth building at all).

---

## E03: Allocation-policy comparison with bootstrap significance

**Hypothesis:** at the budget PROJECT.md's phase 1 plans to spend (25%), the learned allocator admits fewer corrupted-reward groups than the threshold allocator by a margin that survives resampling noise, not just as a point estimate.

**Change vs existing scripts:** no new rollouts. Reuse E01's feats.parquet. Add a bootstrap loop (resample groups with replacement, 200-1000 resamples) around the existing sim_alloc.py replay at budget=0.25, computing the corrupted-reward-admitted rate (fraction of proxy_error=1 groups NOT selected by the allocator) for threshold and learned on each resample.

**Command shape:** wraps `sim_alloc.py` (invoked the way p0_analyze.py already does, with feats.parquet, label proxy_error, group cell100, budget 0.25) inside a bootstrap driver script; no change to sim_alloc.py's core logic.

**Metric:** corrupted-reward-admitted rate for threshold vs learned, with a bootstrap 95% CI on the difference.

**Pass threshold:** the 95% CI on (threshold_corrupted_rate - learned_corrupted_rate) excludes zero and the point estimate is positive (learned admits fewer corrupted rewards).

**Kill threshold:** the 95% CI on the difference includes zero, i.e. the apparent edge in E01/E02 is not distinguishable from resampling noise at this sample size.

**Expected runtime:** under 10 min, this machine or CPU-only (no GPU needed for this step, listed for continuity with this machine's budget).

**Informs:** claim 2 directly, with the statistical rigor claim 2's "fewer corrupted rewards ... than any fixed rule" wording demands.

---

## E04: Proxy type variation, small judge model

**Hypothesis:** a small LLM judge's error pattern (which rollouts it mis-scores) is structured in prompt-embedding space similarly to majority-vote's error pattern, but the two proxies are wrong on different, only partially overlapping subsets of prompts.

**Change vs existing scripts:** reuse E01's already-generated rollouts.parquet (no new policy generation needed). New step: score every rollout's answer correctness using Qwen2.5-0.5B-Instruct as a judge (binary correct/incorrect, given the question and the rollout's extracted answer, gold-free), producing a judge-majority label per group analogous to Gd.majority in p0_rollouts.py. Feed judge_error (judge-majority != gold) into p0_analyze.py's structure pipeline in place of proxy_error.

**Command shape:** new script `avrl/p0_judge_score.py --rollouts avrl/runs/p0/rollouts.parquet --judge_model Qwen2.5-0.5B-Instruct --out avrl/runs/p0_judge/judge_scores.parquet` (does not exist yet, small addition to the existing harness), then p0_analyze.py's structure block rerun with y = judge_error instead of y = G.proxy_error.

**Metric:** judge_error_rate overall and on judge-confident subset; auc_cheap_emb for judge_error; Jaccard overlap between {groups where majority-vote is wrong} and {groups where judge is wrong}.

**Pass threshold:** judge_error has auc_cheap_emb - auc_cheap >= 0.05 (structure also present for this proxy type) AND Jaccard overlap with majority-vote errors < 0.5 (the two proxies fail on different prompts, meaning proxy choice itself is a lever, not just budget).

**Kill threshold:** Jaccard overlap > 0.8 (judge and majority-vote fail on essentially the same prompts; proxy-type variation adds no new signal beyond E01).

**Expected runtime:** judge scoring pass on 3,000 x 8 = 24,000 already-generated completions with a 0.5B model, estimated 5-10 min generation-equivalent inference on this machine (estimate, not measured), plus the same ~10 min analysis step as E01. Estimated total under 25 min.

**Informs:** claim 3, and the allocator-design question of which proxy type an allocator should be built around.

---

## E05: Proxy type variation, self-verification

**Hypothesis:** asking the policy model to re-derive and check its own answer (self-verification) produces a different, and on average less reliable, error-structure signal than peer majority-vote, because self-verification errors correlate with the same reasoning failure that produced the original wrong answer rather than being independent.

**Change vs existing scripts:** reuse E01's rollouts.parquet for the original answers. New generation pass: for each of the 3,000 x 8 rollouts, prompt the same policy (Qwen2.5-1.5B-Instruct) to independently re-solve the problem and state whether it agrees with the stored answer (yes/no), without showing its own prior chain of thought. Proxy label = self-verification says "agrees" AND self-verification's own re-derived answer matches the original.

**Command shape:** new script `avrl/p0_selfverify.py --rollouts avrl/runs/p0/rollouts.parquet --model Qwen2.5-1.5B-Instruct --out avrl/runs/p0_selfverify/` (does not exist yet), then the same p0_analyze.py structure block with y = self_verify_error.

**Metric:** self_verify_error_rate vs majority-vote proxy_error_rate on the same groups; auc_cheap_emb for self_verify_error; correlation between self_verify_error and proxy_error at the group level.

**Pass threshold:** self_verify_error_rate is measurably different from proxy_error_rate (>= 3 percentage points) in either direction, confirming self-verification is a distinct proxy worth allocator attention rather than a redundant copy of majority vote.

**Kill threshold:** self_verify_error correlates with proxy_error at Pearson r >= 0.9 (self-verification adds no information beyond majority vote; not worth building a separate allocator arm for it).

**Expected runtime:** one extra generation pass over 24,000 short self-verification prompts, estimated 10-15 min on this machine (estimate), plus ~10 min analysis. Estimated total under 30 min.

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

**Expected runtime:** under 15 min total, this machine or CPU-only (12 cheap replay passes over already-computed features, no new GPU generation).

**Informs:** claims 1 and 2 (de-risks the hyperparameter choice before the expensive GRPO run).

---

## E12: Embedding/feature ablation (analysis-only)

**Hypothesis:** the structure signal found in E01 is not an artifact of the specific sentence-transformers/all-MiniLM-L6-v2 embedding choice (p0_analyze.py, line 18 as read this session); a different embedder gives a comparable or better AUC gain and neighbor ratio.

**Change vs existing scripts:** no new rollouts. Reuse E01's groups.parquet questions. Swap the embedding model from all-MiniLM-L6-v2 to a larger sentence-transformers model (e.g. a bge or gte variant) and, separately, to the policy model's own final-layer mean-pooled hidden states extracted from the prompt (a same-model-features condition), then rerun the KMeans/PCA/logistic-regression structure pipeline identically to p0_analyze.py's existing code path.

**Command shape:** p0_analyze.py's embedding line re-pointed at each alternate encoder in turn, output written to separate --out subdirectories for comparison.

**Metric:** auc_cheap_emb and neighbor_ratio for each embedding choice, compared to E01's all-MiniLM-L6-v2 numbers.

**Pass threshold:** at least one alternate embedding matches or exceeds E01's auc_cheap_emb gain and neighbor_ratio, confirming the structure signal is a property of the prompts, not a fragile artifact of one specific encoder.

**Kill threshold:** all alternate embeddings show materially weaker (>= 0.05 AUC lower) structure than all-MiniLM-L6-v2, meaning claim 3's evidence is embedder-specific and the choice of embedding model becomes a load-bearing, undocumented design decision.

**Expected runtime:** under 20 min, this machine (re-encoding 3,000 questions with 1-2 alternate encoders plus the existing KMeans/PCA/logistic-regression steps, no new rollout generation).

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

**Expected runtime:** UNVERIFIED, no comparable run observed this session; PROJECT.md specifies a multi-GPU cluster for phase 1, not a single GPU, and a LoRA GRPO run of 400 steps across 4 arms x 3 seeds (12 runs total) is a multi-hour-to-multi-day job, not a single-GPU-hour estimate; a real number needs the harness built first and a short (e.g. 20-step) timing run.

**Informs:** claims 1 and 2 jointly, the central claim-testing experiment of the whole project.

---

## E16: In-training budget sweep (NEEDS PHASE-1 HARNESS)

**Hypothesis:** the learned-vs-threshold accuracy gap found in E15 (if any) is not flat across budgets; mirroring E02's phase-0 replay finding, the gap should be largest at low budgets (10%) and shrink as budget rises toward 50%.

**Change vs existing scripts:** same p1_grpo.py harness as E15, restricted to the threshold and learned arms only (dropping full and random to save compute), swept over budget in {0.10, 0.25, 0.50}, 3 seeds each (18 runs total).

**Command shape:** UNVERIFIED, same harness as E15 with --budget varied.

**Metric:** final held-out accuracy gap (learned minus threshold) at each budget point.

**Pass threshold:** the gap is monotonically non-increasing in budget and exceeds the seed spread at the 10% point, consistent with E02's phase-0 prediction.

**Kill threshold:** the gap is flat or non-monotonic across budgets, meaning the phase-0 replay's budget-sensitivity prediction (E02) does not transfer to real training dynamics.

**Expected runtime:** UNVERIFIED, comparable order of magnitude to E15 (18 of the planned 12-run E15 scale), multi-hour-to-multi-day on a multi-GPU cluster per PROJECT.md.

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

**Expected runtime:** UNVERIFIED, substantially longer than E15 (5x the step count at minimum), multi-day on a multi-GPU cluster, estimate only; needs the phase-1 harness plus the new rolling-AUC and hacking-trend logging from E13 and E18.

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

**Expected runtime:** under 15 min, this machine or CPU-only (reuses E01's already-computed features, no new GPU generation).

**Informs:** claims 1 and 2, and answers the cannot-work gap check's requirement for a stronger primary baseline before Phase 1 is funded. Does not need the phase-1 harness.

---

## G02: VIP-style transfer baseline (analysis-only)

**Hypothesis:** re-applying VIP's allocation approach (a per-group success-probability predictor feeding a budget-constrained optimizer) with verify-yes/no as the allocated resource, instead of rollout count, recovers most but not all of LearnedAllocator's hit-rate advantage over the recalibrated threshold, narrowing the defensible novelty to the specific combination of mechanism and axis rather than either alone, per the already-done gap check.

**Change vs existing scripts:** no new rollouts. Reuse E01's feats.parquet. Add a `VIPTransferAllocator` to allocators.py: a per-group disagreement-probability predictor (same feature set as LearnedAllocator, same refit cadence) feeding a convex top-k budget allocation instead of LearnedAllocator's Thompson-sampling layer, approximating VIP's Gaussian-process-plus-convex-optimization shape (Nguyen et al., ICLR 2026, https://arxiv.org/abs/2602.01601) with verify-yes/no substituted for rollout count.

**Command shape:** a small driver script instantiating `allocators.VIPTransferAllocator()` against E01's feats.parquet, run through the same G01 replay harness, at budget=0.25 and across the E02 budget grid.

**Metric:** hit rate and bootstrap-CI corrupted-reward-admitted rate (per E03's method) for VIP-transfer vs recalibrated threshold vs learned, at matched budget.

**Pass threshold:** the learned allocator's hit rate exceeds VIP-transfer's by >= 5 percentage points at the 25% budget point, evidence the novelty is not just the axis substitution but also the allocator design (Thompson sampling over cells vs VIP's convex optimization).

**Kill threshold:** VIP-transfer and learned are within 3 percentage points of each other; per the already-done gap check, the defensible contribution narrows to "axis substitution using an existing mechanism."

**Expected runtime:** under 15 min, this machine or CPU-only (reuses E01's features, no new GPU generation).

**Informs:** claim 1, and directly answers the already-done gap check's challenge that VIP's machinery should be tried as a baseline before crediting the allocator design itself. Does not need the phase-1 harness.

---

## G03: Structure-statistic-versus-gain scan across task/proxy pairs

**Hypothesis:** the learned allocator's hit-rate gain over the recalibrated threshold (G01) is monotone in each pair's structure statistics, the neighbor-hit ratio of proxy error over embedding cells and the AUC of cheap features for proxy error (both computed by p0_analyze.py), across at least four task/proxy pairs, the pre-registered prediction that makes claim 3 falsifiable rather than a single-task detection.

**Change vs existing scripts:** reuses E01 (GSM8K, majority-vote), E04 (GSM8K, judge proxy; needs p0_judge_score.py), E09 (MATH-500, majority-vote; needs the --data flag and boxed-answer parser), and a fourth pair combining MATH-500 with the judge proxy (same scripts as E04+E09 composed, no new code). For each pair: compute neighbor-hit ratio and auc_cheap_emb (p0_analyze.py's existing structure block) and the recalibrated-threshold-vs-learned hit-rate gain (G01's replay) at budget=0.25.

**Command shape:** the existing p0_rollouts.py / p0_analyze.py / p0_judge_score.py commands for each of the four pairs (per E01, E04, E09 as already specified), followed by a small driver script that reads each pair's summary.json/result.json and G01's replay output, and fits a Spearman correlation between each structure statistic and the hit-rate gain across the four points.

**Metric:** Spearman correlation between (neighbor-hit ratio, auc_cheap_emb) and (learned-minus-recalibrated-threshold hit-rate gain), across the four task/proxy pairs.

**Pass threshold:** at least one statistic has |Spearman rho| >= 0.7 across the four pairs, sign matching the pre-registered direction (higher structure, higher gain), supporting contribution 3's monotonicity claim even at this small sample size.

**Kill threshold:** both statistics have |Spearman rho| < 0.3, or the sign is inconsistent with the pre-registered direction in more than one pair; contribution 3's predictive claim is unsupported at this sample size and Phase 2's transfer-test centerpiece should be re-scoped before Phase 1 funding.

**Expected runtime:** each of the four task/proxy generation-and-analysis passes is separately budgeted at under 40 minutes on this machine (per E01's ~16 min generation + ~10 min analysis baseline and E04/E09's estimates), plus G01's replay (under 15 min) and a negligible correlation-fitting step; total under 3 hours across four sequential runs, each individually under the 40-minute cap.

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

All paper citations above are carried forward unchanged from /homes/rdubash/avrl_live/FULL_R/lit/avrl/SYNTHESIS.md, read this session, which itself states its method: built via WebFetch against arXiv abstracts, the OpenAlex works API, and direct publisher DOIs, with WebSearch unavailable (budget error) and not retried. This document adds one new fact verified this session via WebFetch: HuggingFaceH4/MATH-500's existence, size, and content, https://huggingface.co/datasets/HuggingFaceH4/MATH-500. No paper, author, or number is invented in this document; every runtime figure not directly read from avrl/runs/p0/run.log is explicitly marked as an estimate or UNVERIFIED. G01-G04 were added after this session's three gap checks (GAP_already-done.md, GAP_cannot-work.md, GAP_not-a-thesis.md); their citations (VIP, https://arxiv.org/abs/2602.01601) are carried forward unchanged from SYNTHESIS.md and the gap-check files.

## Compute correction (2026-09-30, main session)
This machine's GPU is an NVIDIA RTX A4000 with 16 GB, not a 32 GB card. Measured with the phase-1 harness (Qwen2.5-1.5B, LoRA r=32, bf16): full settings (16 prompts x G=8, 512 tokens) run at 153 s per step with HF generation and peak 13.2 GB; vLLM colocate does not fit beside training on this card (out of memory at engine start). Smoke settings run at 20 s per step. vLLM-only rollout experiments (phase-0 style) are unaffected and fit in 40 minutes at 1,500 to 3,000 prompts.

## Feasibility note (this run)
Items marked A100, needing a larger cluster, a second machine, or the phase-1 GRPO harness (E15 to E20, G04, anything needing p1_grpo.py; about 89 s per step on this card, so 100 steps take 2.5 hours) are not runnable under the 45-minute cap and must not be planned.

## Proposed by the loop

Added 2026-10-18 after E04 (kill for claim 3 on GSM8K with group-level proxies; see avrl/RESEARCH_LOG.md). Ids are L<iteration><letter>. Ranked by expected information value. All three run on this machine under the 45-minute cap. Every number cited below was re-derived from avrl/runs/E04/ on review.

Protocol change for every entry from here on: report the out-of-fold cell-encoding delta AUC (cell target encoding added to the cheap-feature logistic model, honest two-stage encoding, 3 CV seeds) in place of the neighbor ratio. aa.data.neighbor_hit_ratio conditions on "at least one same-cell neighbor is a hit", which is true for 100% of groups at cell30 and cell100 when the label base rate is 44 to 50%, so the ratio is 1.000 by construction (E04 caveat 7). Replace the "neighbor ratio >= 1.5" arm of the pass condition with "OOF cell-encoding delta AUC >= 0.02".

### L1a: Rollout-level, agreement-free replay on the E04 judge data (CPU only)

**Hypothesis:** E01 and E04 stayed in the threshold regime because group agreement is a free cheap feature (AUC 0.78 to 0.91) for every proxy computed over an 8-rollout group. If each rollout is treated as one decision and the allocator is denied every group aggregate, the only confidence signal left is the judge score (rollout-level AUC 0.654 for 3B, 0.702 for 7B-AWQ), and prompt-space structure (question type, difficulty, answer format) may then carry information the allocator can use.

**Change vs existing scripts:** new script avrl/exp/l1a_rollout_level.py. Join avrl/runs/p0/rollouts.parquet (pid, k, answer, correct, tokens, truncated) with avrl/runs/E04/judge_<name>.parquet on (pid, answer) and with avrl/runs/p0/feats.parquet on pid (pc0..pc15, cell30, cell100, cell300). Per-rollout features allowed: judge_p, tokens, truncated, answer-format features (has decimal point, negative, log magnitude, empty), log question length. Forbidden: agreement, n_distinct, pass_rate, majority, or any other aggregate over rollouts of the same prompt. Copy avrl/runs/E04/sim_alloc.py to avrl/runs/L1a/ and use cell100 as the bucket column. Cross-validation must be grouped by pid (GroupKFold) so rollouts of one question never straddle train and test.

**Command shape:** UNVERIFIED, script not written: `cd /homes/rdubash/avrl_live/FULL_R/avrl && /homes/rdubash/active-audit/.venv/bin/python exp/l1a_rollout_level.py --rollouts runs/p0/rollouts.parquet --judges runs/E04 --feats runs/p0/feats.parquet --out runs/L1a`, run for the 3B and 7B-AWQ judges. Two labels per judge: corrupted = (judge_p > 0.5) and wrong by gold; judge_wrong = (judge_p > 0.5) != correct.

**Metric:** per judge and label: label count and rate (gate: at least 200 positive rollouts); AUC of judge_p alone, of the allowed cheap features, of cheap + 16 PCs (delta AUC), OOF cell-encoding delta AUC at cell30, cell100, cell300; allocation replay at 10, 25, 50% budgets, 8 seeds, ts_online and greedy_online versus a threshold on judge_p (label-free) and versus the free-label static arm and the charged static_matched arm, ratio learned over best label-free threshold with seed SD.

**Pass threshold:** on the 3B or 7B rung with the gate met: delta AUC >= 0.05 OR OOF cell-encoding delta AUC >= 0.02, AND the learned arm (ts_online or greedy_online) catches >= 1.10x the best label-free threshold at 25% with the gap above 2 SD. That is the first positive point for claim 3 and the revised thesis, with cheap AUC on the order of 0.65 to 0.75 as the low-confidence end of the G03 axis.

**Kill threshold:** delta AUC < 0.01 and OOF cell-encoding delta AUC < 0.01 on both rungs, and learned <= 1.05x threshold at 25%. Then the revised thesis has no positive point on GSM8K in either the grouped or the agreement-free regime; the deliverable is the proxy-error measurement plus the fixed-threshold recipe, and further claim-3 work needs a task that is not on disk (code, open-ended).

**Expected runtime:** about 10 to 15 min on CPU (feature join, grouped CV, 8-seed replay on up to 24,000 rollouts, which is 8x the sim size of E04, so cap seeds at 8 and bucket on cell100 as in E04). Verify the rollout-level replay time on a 2,000-row subsample first.

**Informs:** claim 3 directly, G03 (adds a low-cheap-AUC point), and whether L1c is worth a GPU run. Caveat to state in the ledger: agreement is removed by ablation, not by changing the task, so a null here is weaker than a null on a real code task.

### L1b: Fair fixed-rule baseline with calibration-label accounting (CPU only; G01 made concrete)

**Hypothesis:** the only learned-over-fixed edge in E04 is the layout-free greedy_online arm over the single-feature agreement threshold at 50% budget (ratio 1.03 to 1.13 in 12 of 12 judge cells, above 2 pooled SD every time; 25% budget: 5 of 12 above, 2 of 12 below; 10% budget: 11 of 12 below, none above). It exists because greedy_online can combine judge features (cheap_j AUC 0.87 to 0.91 versus agreement alone 0.78 to 0.81). A recalibrated multi-feature fixed rule with its calibration labels charged against the budget should match it, in which case claim 2 has no support on this machine.

**Change vs existing scripts:** copy avrl/runs/E04/sim_alloc.py to avrl/runs/L1b/sim_alloc.py. Add a policy recal_charged(n_cal): spend n_cal of the budget on randomly chosen verified groups, fit the same logistic regression as static_matched on them (cheap_j features only, no PCs), rank the remaining groups, and verify down the ranking (static_matched uses a fixed half of the budget; this sweeps it). Add bootstrap 95% CIs on (greedy_online minus best charged fixed rule) over seeds and over groups. Keep the free-label static arm as an upper bound only.

**Command shape:** UNVERIFIED, policy not written: for each of E01 (label proxy_error), 3B and 7B-AWQ (label judge_error), `cd /homes/rdubash/avrl_live/FULL_R/avrl && /homes/rdubash/active-audit/.venv/bin/python runs/L1b/sim_alloc.py runs/E04/feats_<name>.parquet <label> cell100 agreement,n_distinct,mean_tokens,std_tokens,any_truncated,log_qlen[,judge_top,judge_margin,judge_mean,judge_std] runs/L1b/sim_<name>.json`, with n_cal in 25, 50, 100, 200 as an extra argument or environment variable.

**Metric:** errors caught at 10, 25, 50% budgets, 8 seeds, for ts_online, greedy_online, thr_agreement, recal_charged at each n_cal (best n_cal per budget reported with the sweep also shown), static (free labels); ratio of greedy_online and ts_online to the best charged fixed rule, with bootstrap CI.

**Pass threshold:** greedy_online (or ts_online) exceeds the best charged fixed rule by >= 5% with the CI excluding zero at both 25% and 50% budgets on at least two of the three label sets. That is the first support for claim 2 and justifies building the phase-1 GRPO harness (not runnable here).

**Kill threshold:** the best charged fixed rule is within 2% of greedy_online or ahead at both 25% and 50% on all three label sets. Then claim 2 is not supported on this machine and the fixed-threshold recipe is the deliverable for claims 1 and 2.

**Expected runtime:** about 15 min on CPU (E04's whole CPU analysis, including 16 replay runs, finished in under 2 minutes; this adds 3 label sets x 2 feature sets x 4 n_cal values).

**Informs:** claim 2 directly; closes G01.

### L1c: Trained reward-model proxy at rollout level (GPU, two jobs, smoke check first)

**Hypothesis:** the original claim is about a trained reward model that the policy can exploit. Skywork-Reward-V2-Qwen3-0.6B (on disk at /homes/rdubash/active-audit/models) scores the full reasoning, not just the answer, and is not a function of group agreement at rollout level. Its false accepts (accepted, wrong by gold) should depend on question type and completion style, which is the structure claim 3 needs, and which the answer-only judge could not show.

**Change vs existing scripts:** (1) copy avrl/p0_rollouts.py to avrl/exp/l1c_regen.py and save completion text (the p0 rollouts parquet has none): Qwen2.5-1.5B-Instruct, same 3,000 prompts, 8 rollouts, same seed and sampling as p0 (p0 wall 1088 s). (2) new avrl/exp/l1c_rm_score.py: score each completion with the reward model, through vLLM pooling if the installed .venv-judge version supports the architecture, else transformers AutoModelForSequenceClassification in the same environment on the GPU. Smoke check on 200 completions first: score is not constant, rollout-level AUC against correct is computed. (3) analysis per L1a on the new scores, no group aggregates, accept threshold at the quantile matching the 64% rollout accuracy.

**Command shape:** UNVERIFIED, scripts not written. Two jobs, each through gpu_run.sh: `bin/gpu_run.sh avrl/runs/L1c_regen '<.venv-judge python> avrl/exp/l1c_regen.py ...'` then `bin/gpu_run.sh avrl/runs/L1c_rm '<.venv-judge python> avrl/exp/l1c_rm_score.py ...'`; analysis on CPU with the L1a script pointed at the new scores.

**Metric:** validity gate: rollout-level reward-model AUC vs correct >= 0.65 and <= 0.98 (not saturated) and at least 200 false-accept rollouts; then the L1a statistics (cheap AUC, delta AUC, OOF cell-encoding delta AUC, replay ratio at 10, 25, 50%), plus the completion-level correlates of false accepts (length, hedging, step count).

**Pass threshold:** same as L1a on the false-accept label.

**Kill threshold:** gate failed (record as uninformative, not a verdict), or the L1a kill condition on the false-accept label.

**Expected runtime:** about 25 to 30 min total (regeneration about 18 min, reward-model scoring of 24,000 completions about 5 to 10 min), split over two jobs so each stays under 45 min. Run only after L1a: a null on L1a lowers the prior that L1c finds structure.

**Informs:** claim 3 on a realistic trained-proxy setting; G03 (adds a point with cheap AUC set by the reward model alone).

### Iteration 2 additions to 'Proposed by the loop' (after L1b)

Added 2026-10-18 after L1b (inconclusive for claim 2; see avrl/RESEARCH_LOG.md, second L1b entry). Status of earlier loop items: L1b is done (greedy_online is 1.037 to 1.072x the best charged fixed rule on the 3B and 7B-AWQ judge labels at 25% and 50% budget, 0.99 to 1.00x on E01, 0.86 to 0.95x at 10%; no label set cleared the pre-registered 1.05 at both budgets). L1a and L1c are unchanged by this result, and L1c is still blocked by the E04 stop rule. Ids are L<iteration><letter>. Ranked by expected information value. The headroom ceiling matters for every entry below: the free-label static arm is only 1.08 to 1.14x the best charged fixed rule on the judge labels and 1.00 to 1.03x on E01, so the E01 label set cannot pass a learned-over-fixed test and pass thresholds above about 1.05 sit near the ceiling.

### L2a: Decompose the greedy_online edge against a stronger charged fixed baseline (CPU only)

**Hypothesis:** greedy_online's 3.7 to 7.2% edge on the judge labels comes from calibration labels that are collected on-policy from top-scored groups, so they train the scorer where it matters and also count as hits. It does not come from repeated online learning. A charged fixed rule that spends its calibration budget on the lowest-agreement groups (one fit, then rank the rest) will match it. If so, the 'learned allocator' is a one-shot calibration design and claim 2 has no support beyond a recalibrated fixed rule.

**Change vs existing scripts:** copy avrl/exp/l1b_sim_alloc.py to avrl/exp/l2a_sim_alloc.py. Add policy seeded_charged_<n> for n in 25, 50, 100, 200, 400 (n at most budget // 2): verify the n groups with the lowest agreement (ties random), count them as spent and as hits, fit the same StandardScaler plus balanced LogisticRegression on the cheap_j features, rank the remaining groups by predicted error, verify down the ranking; fall back to agreement order if the sample has 0 or all positives. Parameterize greedy_online by first-fit size n0 in 25, 50, 100 and refit period R in n0, 4 n0 (keep the L1b setting, R = max(budget // 8, 40), as the reference arm). Keep thr_agreement, recal_charged_<n>, static (upper bound only), uniform and oracle. Reuse the same split seeds 0 to 3, policy seeds 0 to 7, and the L1b selection protocol (choose the best fixed rule and best greedy variant on policy seeds 0 to 3, report on 4 to 7). The regression gate is the same: at split seed 0 the L1b policies must reproduce runs/L1b/sim_<name>_split0.json exactly.

**Command shape:** UNVERIFIED, script not written: same xargs -P 12 launcher as exp/L1b.sh with exp/l2a_sim_alloc.py and runs/L2a/, labels 3B judge_error and 7B-AWQ judge_error only (E01 has no headroom), then a summarize script cloned from exp/l1b_summarize.py with the extended fixed family.

**Metric:** per judge label set and budget (10, 25, 50%) on seeds 4 to 7 over 4 splits: errors caught for greedy_online variants, seeded_charged_<n>, recal_charged_<n>, thr_agreement; ratio of the best greedy variant to the best member of the extended charged fixed family with bootstrap 95% CI over the 16 paired runs, split-cluster CI, and SD of the 4 split ratios; the full n and (n0, R) sweep shown.

**Pass threshold:** ratio at least 1.03 with the split-cluster CI above 1 in all four judge cells (3B and 7B-AWQ at 25% and 50%). The edge survives a stronger charged fixed baseline and the n0 / R grid shows whether repeated refitting is the active ingredient. (1.03 rather than 1.05 because L1b measured edges of 3.7 to 7.2% against a ceiling of 8 to 14%.)

**Kill threshold:** ratio at most 1.02 in all four judge cells. Then the edge is a calibration-design effect, claim 2 has no support beyond a one-shot recalibrated threshold, and the Phase 1 GRPO harness is not worth building. Between 1.02 and 1.03, or a CI that includes 1, is reported as inconclusive with the per-cell table.

**Expected runtime:** about 15 to 20 min on CPU (L1b took 8 min wall for 12 jobs; the greedy grid adds six more greedy_online variants per cell).

**Informs:** claim 2 directly (can kill it with no GPU), what a learned allocator must do beyond fitting once, and the Phase 1 harness design.

### L2b: Batch-scale and independent-population sweep (CPU only)

**Hypothesis:** the L1b CIs understate uncertainty about a fresh population (4 splits that are 75% draws of the same 3,000 groups, policy seeds sharing a population), and sim populations of about 2,200 groups are far larger than one GRPO step. If the greedy_online edge comes from having a few hundred cheap calibration labels in a large pool, it shrinks or reverses at batch scale, where charging the calibration labels eats a larger share of the budget.

**Change vs existing scripts:** copy avrl/exp/l1b_sim_alloc.py to avrl/exp/l2b_sim_alloc.py with a subset-size m and a subset seed argument: draw m groups without replacement from the 3,000 (m in 250, 500, 1000; subset seeds 0 to 15; expected pairwise overlap about 8%, 17%, 33%), then apply the existing 25 / 75 fit / sim split to the subset. Cap n_cal at budget // 2 (n_cal in 25, 50, 100). Policies: thr_agreement, recal_charged_<n>, greedy_online, ts_online, static (upper bound), uniform, oracle. Policy seeds 0 to 3. Labels: 3B and 7B-AWQ judge_error only.

**Command shape:** UNVERIFIED, script not written: xargs -P 12 over (label set, m, subset seed), results in runs/L2b/, summarize with a script cloned from exp/l1b_summarize.py but clustering the bootstrap over subsets.

**Metric:** per label set, m and budget: errors caught per policy, greedy_online and ts_online over the best charged fixed rule (best n_cal chosen on subset seeds 0 to 7, reported on 8 to 15), bootstrap 95% CI over the 8 held-out subsets, and the ratio as a function of m.

**Pass threshold:** ratio at least 1.03 with the CI above 1 at m = 500 for both judge sets at 25% and 50% budget. The edge survives a fresh-population draw at near-batch scale.

**Kill threshold:** ratio at most 1.0 at m of 500 or less on both judge sets at 25% and 50%. Then the edge needs a large calibration pool, a per-step allocator cannot reach it, and any harness would have to learn across steps (which needs E13-style non-stationarity evidence first).

**Expected runtime:** about 20 min on CPU (smaller populations are faster per run, but there are 16 subsets x 3 sizes x 2 label sets).

**Informs:** claim 2 (population variance and the scale the Phase 1 harness would actually operate at), and whether Phase 1 allocates per step or across steps.

### L2c: Fresh-population replication on unused GSM8K prompts (GPU, two jobs; run only if L2a does not kill; needs Rustom's explicit go-ahead under the E04 stop rule)

**Hypothesis:** the greedy_online edge over the best charged fixed rule on weak judge labels replicates on rollouts and prompts that none of the L1b / E04 results have seen.

**Change vs existing scripts:** avrl/runs/p0 sampled 3,000 of the 7,473 GSM8K train prompts with random_state = 0, leaving 4,473 unused. Copy avrl/p0_rollouts.py to avrl/exp/l2c_rollouts.py with a prompt-selection change: sample 3,000 from the 4,473 prompts not in p0 (exclude by question text). Same policy (Qwen2.5-1.5B-Instruct), G = 8, max_tokens 512. Then score with avrl/exp/p0_judge_score.py for the 3B and 7B-AWQ judges, build feature tables as in E04, and rerun the unchanged L1b pipeline (n_cal grid, selection protocol, split seeds 0 to 3, policy seeds 0 to 7). This is a claim-2 replication, not a claim-3 variant, but it is GPU spend on GSM8K, so it needs an explicit override of the E04 stop rule.

**Command shape:** UNVERIFIED, scripts not written. Two jobs, each through bin/gpu_run.sh: rollouts (about 18 min for 24,000 rollouts at the p0 rate), then judge scoring of both rungs (about 8 min); a smoke check on 200 prompts first. Analysis on CPU with exp/l1b_sim_alloc.py and exp/l1b_summarize.py pointed at the new feature tables.

**Metric:** the L1b table on the new population: errors caught at 10, 25, 50%, greedy_online and ts_online over the best charged fixed rule with bootstrap and split-cluster CIs; validity gate on the 3B judge unchanged (rollout AUC at least 0.65).

**Pass threshold:** greedy_online / best charged fixed at least 1.03 with the CI above 1 in at least 3 of the 4 judge cells (3B and 7B-AWQ at 25% and 50%).

**Kill threshold:** ratio at most 1.02 in all four judge cells, or the judge validity gate fails (record as uninformative, not a verdict).

**Expected runtime:** about 30 min GPU in two jobs, plus about 10 min CPU.

**Informs:** claim 2 on an independent population, and the last go / no-go before spending effort on a Phase 1 GRPO harness.


### Iteration 3 additions to 'Proposed by the loop' (after L2a)

Added 2026-10-18 after L2a (PASS for claim 2 in the offline replay; see avrl/RESEARCH_LOG.md, entry L2a). Status of earlier loop items: L2a is done (best learned variant / best member of the extended charged fixed family is 1.050, 1.060, 1.139, 1.061 for 3B 25%, 3B 50%, 7B-AWQ 25%, 7B-AWQ 50%, split-cluster CI lower bounds 1.043, 1.048, 1.089, 1.054; 0 of 4 kill cells; unchanged greedy_online alone is 1.039, 1.037, 1.094, 1.033; 10% budget 0.962 and 0.969; regression gate passed with 0 differences). The 2x2 shows the edge comes from the repeated-refit schedule and not from the calibration design. L1a and L1c are unchanged and L1c is still blocked by the E04 stop rule. Ids are L<iteration><letter>. Ranked by expected information value.

Status of L2b and L2c after L2a. L2b (above) is the next experiment, with two edits: (1) in exp/l2b_sim_alloc.py start from exp/l2a_sim_alloc.py rather than l1b, so the learned arm includes greedy_n25_R25 (the L2a winner) and smaller n0 = R = 10, and the fixed family is the extended one (recal, seeded, split charged, n capped at budget // 2); (2) cluster the bootstrap over subsets and keep the pass at ratio >= 1.03 with CI above 1 at m = 500. L2c (fresh population, GPU) is no longer gated by an L2a kill, so its only remaining gate is Rustom's explicit override of the E04 stop rule; ask for it after L2b, because L2b is free and decides whether the extra GPU spend is worth it.

### L3a: Bracket both grids and map the refit-count dose-response (CPU only)

**Hypothesis:** the L2a edge is a refit-frequency effect (n0 = R = 25 beat R = 100 in all four cells, and n100_R100 beat n100_R400 in all four) and the best learned variant sits at the smallest n0 and R in the grid, so the optimum is not bracketed; the fixed grids stop at n = 400 although budget // 2 is about 545 at 50%, and seeded_charged is still rising at n = 400 on 3B 50% (630, 631, 743, 763, 797 against the best fixed 805). If the optimum is interior and the enlarged fixed family does not close the gap, the L2a ratio is robust to the grid choice; if a larger fixed n or a staged rule closes it, the L2a pass was partly a grid artifact.

**Change vs existing scripts:** copy avrl/exp/l2a_sim_alloc.py to avrl/exp/l3a_sim_alloc.py and avrl/exp/l2a_summarize.py to avrl/exp/l3a_summarize.py. Learned grid: n0 in 10, 15, 25 and R in 1, 5, 10, 25 (keep greedy_online reference and greedy_n25_R25; R = 1 refits after every label, which costs about 1,100 logistic fits per run at 50% budget, a few seconds). Fixed grid: n in 25, 50, 100, 200, 400 and budget // 2. New staged policy staged_charged_<n>_k<k>: n random labels, then refit at k - 1 further equally spaced points through the budget, k in 2, 4, 8; it is reported as the bridge between one-shot and repeated and is not a member of the fixed family for the pass rule. Keep the regression gate (reference policies equal runs/L2a at split 0, then all splits) and the L2a selection protocol (choose on policy seeds 0 to 3, report on 4 to 7 over 4 splits).

**Command shape:** UNVERIFIED, scripts not written: the same xargs -P 8 launcher as exp/L2a.sh with exp/l3a_sim_alloc.py and runs/L3a/, labels 3B and 7B-AWQ judge_error, then exp/l3a_summarize.py. The launcher must not write done.flag or exit_code itself (bin/gpu_run.sh does).

**Metric:** per judge label set and budget (10, 25, 50%) on seeds 4 to 7 over 4 splits: ratio of the selected best learned variant to the selected best member of the enlarged fixed family with the paired-run CI, split-cluster CI and SD of the 4 split ratios; the (n0, R) surface and the n sweep; the staged_charged dose-response (ratio against number of refits k, from one-shot through k = 8 to R = 1); whether the selected learned variant is interior to the grid.

**Pass threshold:** ratio >= 1.03 with the split-cluster CI lower bound above 1 in all four judge cells (3B and 7B-AWQ at 25% and 50%) under the enlarged fixed family. Reading: the edge is robust to the grid, the optimum is bracketed or the ratio only grows, and the harness refit cadence is read off the dose-response.

**Kill threshold:** ratio <= 1.02 in two or more of the four cells. Reading: the L2a pass was a grid artifact in the fixed family and claim 2 is back to inconclusive. Anything else is inconclusive with the per-cell table.

**Expected runtime:** about 15 to 20 min on CPU (L2a took 4 min for 26 policies; this adds about 12 learned variants and 3 staged policies per cell, with R = 1 the slowest).

**Informs:** claim 2 robustness, the refit cadence a Phase 1 allocator needs, and whether the L2b edits should use n0 = R = 10 or 25.

### L3b: Label-source and exploration ablation inside the repeated refit (CPU only)

**Hypothesis:** the repeated refit confounds two things: the number of fits, and the fact that after the first fit every training label comes from the top-ranked region (a selection-biased, on-policy label stream that concentrates the fit near the boundary that matters for the budget). If the on-policy stream is the active ingredient, forcing a fraction of each block onto random unseen groups (charged, counted as hits) should lower the hits per label but could lift the later ranking only if the biased labels were hurting; freezing the scorer after a few refits should cost little if the early on-policy fits already locate the boundary, and a lot if the late labels are what matters.

**Change vs existing scripts:** copy avrl/exp/l2a_sim_alloc.py to avrl/exp/l3b_sim_alloc.py and add arms to run_online: (a) explore_<e>: each block of R = 25 labels spends the fraction e (0.1, 0.25) on random unseen groups, the rest on the top-ranked, refit on all labels; (b) frozen_<k>: refit on the first k refit points only (k = 2, 5, 10) and rank by the last fit afterwards; (c) early_only: refit on all labels but only up to the first 25% of the budget, then freeze. Reference arm greedy_n25_R25 and the L2a selected fixed rule as comparators. Same splits, seeds and selection protocol; the arms are decided in advance and none is chosen on the held-out seeds.

**Command shape:** UNVERIFIED, script not written: same launcher pattern as exp/L2a.sh with runs/L3b/.

**Metric:** per label set and budget, held-out mean errors caught for each arm, ratio to greedy_n25_R25 with split-cluster CI, ratio to the selected fixed rule.

**Pass threshold (informative result):** at least one arm differs from greedy_n25_R25 by 2% or more with the split-cluster CI excluding 1 in at least 3 of the 4 judge cells. Reading: the label stream or the freeze point is a design lever, and the Phase 1 harness spec records it (an exploration term, a refit horizon).

**Kill threshold (null):** all arms are within 1% of greedy_n25_R25 in all four cells. Reading: the edge is simply having a recent multi-feature fit and the harness needs no exploration term in the replay setting; non-stationarity (E13) is then the only remaining reason to add one.

**Expected runtime:** about 15 min on CPU.

**Informs:** the mechanism behind claim 2, and the Phase 1 allocator spec (exploration, refit horizon).

### Iteration 4 additions to 'Proposed by the loop' (after L2b)

Added 2026-10-18 after L2b (INCONCLUSIVE for claim 2; see avrl/RESEARCH_LOG.md, entry L2b). Status of earlier loop items: L2b is done and its gate passed (116 checks, 0 differences). Best learned variant / best member of the extended charged fixed family at m = 500: 0.978 (3B 25%), 1.039 (3B 50%), 0.988 (7B-AWQ 25%), 1.091 (7B-AWQ 50%); 2 of 4 pass cells and 5 of 8 kill cells, so neither rule fired. The main finding, post hoc: the ratio tracks the absolute number of verifications B and not m or the budget fraction (0.70 to 0.78 at B = 25, 0.94 at B = 100, 0.98 to 1.02 at B = 125, 1.03 to 1.09 at B = 250, 1.07 to 1.08 at B = 500, 1.05 to 1.14 at the L2a pool, B about 550 to 1,100). The learner's best variant is the smallest n0 and R in the grid (n10_R5, best in 16 of 18 cells), so the grid is unbracketed at the small end again. L3a (existing, L2a population) is superseded at the small end by L4a; run L4a first and fold L3a's staged-policy and fixed-grid extension into it only if L4a is not enough. L3b is unchanged. L2c is replaced by L4c (same GPU plan, redesigned for B >= 250). L1a and L1c are unchanged (L1c still blocked by the E04 stop rule). Ids are L<iteration><letter>. Ranked by expected information value.

### L4a: Bracket the learner's small end and add a threshold-offset warm start (CPU only)

**Hypothesis:** the L2b loss at B <= 125 is a cold-start cost: a random-start learner spends its first n0 labels at the base rate and its first fits are noisy, while thr_agreement is free. A learner whose score is a fixed decreasing function of agreement plus a ridge-penalized correction on the other features equals thr_agreement at zero labels and only adds information as labels arrive, so it should not lose at small B; and the plain learner at (n0, R) below (10, 5) may cross 1.00 at B = 125, where the n25_R25 to n10_R5 step gained 4.7 hits against a remaining gap of 2.1 to 2.2 hits.

**Change vs existing scripts:** copy avrl/exp/l2b_sim_alloc.py to avrl/exp/l4a_sim_alloc.py and avrl/exp/l2b_summarize.py to avrl/exp/l4a_summarize.py; keep run() and run_online() verbatim. (1) Plain learned grid adds (n0, R) in (3,1), (5,1), (5,2), (5,5), (10,1), (10,2) to the L2b grid. (2) New run_offset(n0, R, lam): block 1 is the n0 highest-agreement groups (n0 = 0 means the first R), then refit after every R labels by ridge-penalized logistic regression on all labels so far (numpy IRLS, ten standardized features), with offset -2 x standardized agreement (the constant 2 fixed in advance) and ridge lam in 1, 10, 100 on all coefficients (the penalty pulls the correction to 0, so the ranking stays the agreement ranking until the labels say otherwise); R in 5, 10; n0 in 0, 5. Arms decided in advance, none chosen on held-out subsets. (3) Same design as L2b: judge_error labels (3B, 7B-AWQ), m in 250, 500, 1000, subset seeds 0 to 15, policy seeds 0 to 7, budgets 10 / 25 / 50% of m, selection of the best fixed member and best learned variant (plain and offset pooled, and each family separately) on subsets 0 to 7, report on 8 to 15, swapped fold and pooled 16 as cross-checks. (4) Regression gate first: at m = 500, subset seed 0, both judges, the reference policies (thr_agreement, uniform, oracle, greedy_n10_R5, greedy_n25_R25, recal / seeded / split charged_n for n in the L2b grid) must equal the per-seed hits in runs/L2b/sim_<judge>_m500_s0.json exactly; a failed gate voids the run.

**Command shape:** UNVERIFIED, scripts not written: the L2b launcher pattern (exp/L2b.sh: gate, then xargs -P 24 over judge x m x subset seed, then the summary; the launcher must not write done.flag or exit_code), results in runs/L4a/, launched through bin/gpu_run.sh after bin/gpu_status.sh shows under 2 GB used.

**Metric:** per judge, m and budget: held-out errors caught for thr_agreement, selected fixed, selected plain learner, selected offset learner, static_out, oracle; selected learned / selected fixed with the 8-subset bootstrap CI, swapped and pooled ratios; offset learner / thr_agreement at each B; ratio against absolute B (the L2b curve with the new arms overlaid); whether the selected (n0, R) is now interior.

**Pass threshold:** the best learned family (plain or offset) reaches ratio at least 1.02 with the CI lower bound above 1 in at least 3 of the 4 cells with B = 125 (3B and 7B-AWQ at m = 250 50% and m = 500 25%), and at least 0.97 in all 12 cells with B <= 125. Reading: the small-B loss is a fixable cold-start cost, the Phase 1 allocator spec gets the warm-start form and a refit cadence, and the break-even B moves down to about 60 to 125.

**Kill threshold:** no learner variant reaches 1.00 in any of the four B = 125 cells, and the best offset arm is below 0.97 in at least 3 of the 12 cells with B <= 125. Reading: the judge features do not pay back their labels below B of about 125 to 150, thr_agreement is the deliverable at small B, and any learned allocator has to persist labels across steps (L4b). Anything else is inconclusive with the per-cell table and the B curve.

**Expected runtime:** about 15 to 20 min on CPU (L2b took about 2 min for 96 jobs at 24 workers; R = 1 adds up to B refits per run, still seconds per job).

**Informs:** claim 2 scope (is the regime boundary a hard floor or a cold-start artifact), the cold-start design and refit cadence for the Phase 1 allocator, and removes the grid-corner objection raised in the L2a Critic note and again in L2b.

### L4b: Cross-step carry-over replay (CPU only)

**Hypothesis:** the L2b dependence on B is a labels-in-hand effect, so an allocator that keeps its labels across batches reaches the B >= 250 ratios even when each batch has a small per-step B. If instead the carry-over arm stays at or below the threshold, the B curve was a subset-size artifact and the Phase 1 harness cannot rescue claim 2 by persisting labels.

**Change vs existing scripts:** new script avrl/exp/l4b_carry.py importing run()/run_online() building blocks from avrl/exp/l2b_sim_alloc.py. For each order seed 0 to 7, permute the 3,000 prompts and cut 12 consecutive disjoint batches of m = 250 (no prompt appears twice). Arms per batch, budget 25% (B = 62) and 50% (B = 125) of the batch: thr_agreement (free); restart (greedy_n10_R5 from scratch every batch, the L2b arm); carry (batch 1 as restart with n0 = 10, later batches keep every earlier label, refit every R = 5 labels, rank only the current batch); carry_window_W (fit on the last W labels, W in 125, 250, 500, as a drift proxy); calibrate_once (recal_charged_50, labels paid in batch 1 only, the fitted rule reused free in batches 2 to 12; the fairest fixed amortization). Policy seed = order seed. Labels: 3B and 7B-AWQ judge_error. Gate before any verdict: on one batch, the restart arm's per-seed hits must equal the L2b greedy_n10_R5 code path run on the same prompts (call the L2b run_online() on that batch's rows), and thr_agreement and uniform hits must equal a direct recomputation; a failed gate voids the run.

**Command shape:** UNVERIFIED, script not written: one python process per (judge, budget) through bin/gpu_run.sh, results in runs/L4b/.

**Metric:** per cell, errors caught per batch index (mean over 8 order seeds) for every arm; arm / thr_agreement against batch index with the bootstrap CI over order seeds; pooled ratio over batches 4 to 12 (labels in hand at least about 250); the first batch at which carry reaches 1.00 and 1.03; carry_window_W / carry.

**Pass threshold:** carry / thr_agreement at least 1.03 pooled over batches 4 to 12 with CI above 1 in all four cells (3B and 7B-AWQ at 25% and 50%), and carry_window_250 within 2% of carry. Reading: persistence converts small per-step B into large effective B, the Phase 1 allocator is a persistent one, and the window curve gives the label staleness it can tolerate (input to E13).

**Kill threshold:** carry / thr_agreement at most 1.00 pooled over batches 4 to 12 in at least 3 of the 4 cells. Reading: label history is not the driver, the L2b B curve was a subset-size artifact, and claim 2 needs a different mechanism than persistence. Anything else is inconclusive.

**Expected runtime:** about 10 min on CPU.

**Informs:** claim 2 regime and the Phase 1 architecture (persistent versus per-step allocator, label retention horizon). Caveat to state in the ledger: batches are exchangeable draws from one fixed policy's rollouts, so there is no drift by construction and the unbounded carry arm is an upper bound; the 8 order seeds permute the same 3,000 prompts, so the order-seed CIs are optimistic; real drift needs E13 and training-time rollouts.

### L4c: Fresh-population replication at B >= 250 (GPU, two jobs, smoke check first; needs Rustom's explicit go-ahead under the E04 stop rule; replaces L2c)

**Hypothesis:** the learner's edge at B >= 250 (L2b: 1.025 to 1.091 at m = 500 50% and m = 1000; L2a: 1.05 to 1.14) replicates on rollouts and prompts that no result so far has seen, and the L2b one-population caveat does not hide it.

**Change vs existing scripts:** as L2c: avrl/runs/p0 sampled 3,000 of the 7,473 GSM8K train prompts with random_state = 0, leaving 4,473 unused. Copy avrl/p0_rollouts.py to avrl/exp/l4c_rollouts.py with the prompt selection changed to 3,000 prompts from the 4,473 not in p0 (exclude by question text); same policy (Qwen2.5-1.5B-Instruct), G = 8, max_tokens 512; score with avrl/exp/p0_judge_score.py for the 3B and 7B-AWQ judges; build feature tables as in E04. Analysis: three disjoint blocks of 1,000 fresh prompts; the fixed family (L2b grid) is selected on block 1 and reported on blocks 2 and 3 (swap for a cross-check); the learned arms are pre-specified, not selected: greedy_n10_R5, greedy_n10_R10, and the best offset arm from L4a if L4a passes. Budgets 25% and 50% of m = 1000 (B = 250, 500). Ask Rustom for the override only after L4a and L4b have fixed the arms, because L4c is the only item that spends GPU on GSM8K.

**Command shape:** UNVERIFIED, scripts not written. Two jobs through bin/gpu_run.sh: rollouts (about 18 min for 24,000 rollouts), then judge scoring of both rungs (about 8 to 15 min), smoke check on 200 prompts first; CPU analysis with the L2b/L4a simulator pointed at the new feature tables.

**Metric:** per judge and budget, per block: errors caught for thr_agreement, selected fixed, each pre-specified learner, static_out, oracle; learner / selected fixed per block with the swapped-fold check; validity gate: 3B rollout AUC for the group-majority proxy at least 0.65 and the judge label error rate within the E04 range.

**Pass threshold:** a pre-specified learner reaches ratio at least 1.03 in both reported blocks in at least 3 of the 4 judge cells. Reading: the B >= 250 edge replicates on a fresh population and the Phase 1 harness spec (persistent allocator, small-n0 frequent-refit learner, or its L4a form) is justified.

**Kill threshold:** ratio at most 1.00 in all four cells on both blocks, or the judge validity gate fails (record as uninformative, not a verdict). Reading: the L2a and L2b edge was a property of the original 3,000 prompts and claim 2 returns to unsupported.

**Expected runtime:** about 30 to 45 min GPU in two jobs (each under the 45 min cap) plus about 10 min CPU.

**Informs:** claim 2 on an independent population, and the last go / no-go before spending effort on a Phase 1 GRPO harness.

### Iteration 5 additions to 'Proposed by the loop' (after L4a)

Added 2026-10-18 after L4a (INCONCLUSIVE for claim 2; see avrl/RESEARCH_LOG.md, entry L4a). Status of earlier loop items: L4a is done and its gate passed (124 required checks, 0 differences; lam = infinity sanity max abs diff 0 over 384 per-seed comparisons). Results: the warm-start (offset) learner removes the small-B loss (selected pooled / selected fixed 0.984 to 1.028 in all 12 cells with B <= 125, against 0.705 to 0.989 in 11 of 12 in L2b) but only by collapsing onto thr_agreement (offset / thr_agreement 0.984 to 1.009 at B <= 100, every CI contains 1); the B = 125 edge is 1.015, 1.000, 1.028, 1.003 with 1 of 4 cells at ratio >= 1.02 and CI lower bound above 1 (pass needs 3 of 4); the plain optimum moved from (10, 5) to the new corner (3, 1) (selected in 15 of 18 cells) and is now a design boundary (n0 >= 2, R >= 1); plain beats offset at B >= 250 in 5 of 6 cells, offset beats plain at B <= 100; static_out (free labels) is 5.5% to 17% above thr_agreement everywhere while the selected learner captures -32% to 7% of that gap at B <= 100, 39% to 63% at B = 250 and 80% to 97% at B = 500. So labels in hand, not features, are the bottleneck. L4b is superseded by L5a (same design plus the L4a arms). L4c is superseded by L5c (arms now fixed by L4a). L3a stays superseded at the small end; L3b, L1a and L1c are unchanged (L1c still blocked by the E04 stop rule). Ids are L<iteration><letter>. Ranked by expected information value.

### L5a: Cross-step carry-over replay with the L4a arms (CPU only; supersedes L4b)

**Hypothesis:** the L2b and L4a dependence on B is a labels-in-hand effect, so a learner that keeps its labels across batches recovers the B >= 250 free-label-gap capture (39% to 63% at B = 250, L4a) even when each batch has a per-step B of 62 or 125, while a warm-start (offset) carry arm never starts below thr_agreement. If instead the carry arms stay at or below the threshold, the B curve was a subset-size artifact and persistence does not rescue claim 2.

**Change vs existing scripts:** new script avrl/exp/l5a_carry.py that copies run_online() and run_offset() (and ridge_logit()) out of avrl/exp/l4a_sim_alloc.py (that file runs at import, so copy the functions, do not import the module). For each order seed 0 to 7, permute the 3,000 prompts and cut 12 consecutive disjoint batches of m = 250 (no prompt appears twice); budgets 25% (B = 62) and 50% (B = 125) of the batch; features standardized on the current batch (label-free). Arms per batch: thr_agreement (free); restart_offset (offset_n0_R5_lam10 from scratch each batch, the L4a cold-start-safe per-step arm); restart_plain (greedy_n3_R1 from scratch, the L4a selected plain arm); carry_plain (greedy_n3_R1 logic with every earlier label kept; batch 1 equals restart_plain); carry_offset_lam10 and carry_offset_lam1 (ridge-offset fit on all carried labels, refit every 5 labels; batch 1 equals restart_offset); carry_window_W for W in 125, 250, 500 on the better carry_offset arm (drift proxy: fit on the last W labels only); calibrate_once (recal_charged_50 labels paid in batch 1 only, the fitted rule reused free in batches 2 to 12; the fairest fixed amortization). Policy seed = order seed. Labels: 3B and 7B-AWQ judge_error. Gate before any verdict: on batch 1, restart_offset and restart_plain per-seed hits must equal the L4a code path run on the same prompts, and thr_agreement and uniform hits must equal a direct recomputation; a failed gate voids the run.

**Command shape:** UNVERIFIED, script not written: one python process per (judge, budget, order seed) through bin/gpu_run.sh with xargs -P 24, results in runs/L5a/, summary script exp/l5a_summarize.py; the launcher must not write done.flag or exit_code.

**Metric:** per cell (judge x budget), errors caught per batch index (mean over 8 order seeds) for every arm; arm / thr_agreement against batch index with the bootstrap CI over order seeds; pooled ratio over batches 4 to 12 (at least about 250 labels in hand); first batch at which each carry arm reaches 1.00 and 1.03; share of the (static_out - thr_agreement) gap captured at batches 4 to 12 (static_out from the 3,000 - 250 outside prompts, as in L4a); carry_window_W / carry.

**Pass threshold:** the better of carry_offset_lam10 and carry_offset_lam1 (chosen on order seeds 0 to 3, reported on 4 to 7) reaches carry / thr_agreement >= 1.03 pooled over batches 4 to 12 with the CI lower bound above 1 in all four cells (3B and 7B-AWQ at B = 62 and 125), and carry_window_250 is within 2% of carry. Reading: persistence converts a small per-step B into a large effective B, the Phase 1 allocator is a persistent warm-start learner, and the window curve gives the label staleness it can tolerate (input to E13).

**Kill threshold:** carry / thr_agreement <= 1.00 pooled over batches 4 to 12 for both carry_plain and the better carry_offset in at least 3 of the 4 cells. Reading: label history is not the driver, the L2b and L4a B curve was a subset-size artifact, and claim 2 needs a different mechanism than persistence. Anything else is inconclusive.

**Expected runtime:** about 10 to 15 min on CPU (R = 1 refits on up to about 1,500 carried labels, 12 batches, 8 order seeds, 2 judges, 2 budgets, about 10 arms).

**Informs:** claim 2 regime and the Phase 1 architecture (persistent versus per-step allocator, label retention horizon). Caveat to state in the ledger: batches are exchangeable draws from one fixed policy's rollouts, so there is no drift by construction and the unbounded carry arm is an upper bound; the 8 order seeds permute the same 3,000 prompts, so the order-seed CIs are optimistic; real drift needs E13 and training-time rollouts.

### L5b: Why plain beats offset at B >= 250, and a single B-agnostic arm (CPU only)

**Hypothesis:** the plain learner's B >= 250 advantage over the offset form (1.039 to 1.091 against 1.020 to 1.080 in 5 of 6 cells, largest gap 7B-AWQ m = 1000 25%: 1.090 against 1.024) comes from the free agreement slope and class weighting, not from the random start; a prior-mean ridge (agreement slope free, ridge toward -2) or a decaying ridge keeps the offset's floor at B <= 100 and matches plain at B >= 250, which would give one allocator form that does not need to know B.

**Change vs existing scripts:** copy avrl/exp/l4a_sim_alloc.py to avrl/exp/l5b_sim_alloc.py, keep run(), run_online() and run_offset() verbatim, add six arms fixed in advance: (1) plain_nocw = greedy_n3_R1 with class_weight = None, otherwise identical; (2) offset_freeagree_la1 and offset_freeagree_la10 = offset_n0_R5_lam10 with an extra free coefficient on z_agreement penalized toward 0 with lam_a in 1, 10 (lam_a = infinity is L4a's offset arm); (3) offset_decay = offset_n0_R5 with lam_t = 1000 / max(t, 10) at t labels in hand (lam 100 at 10 labels, 10 at 100, 4 at 250); (4) handoff_25 and handoff_50 = offset_n0_R5_lam100 for the first k = 25, 50 labels, then greedy_n3_R1-style plain refits on all labels so far. Cells: all 18 L4a cells (2 judges x m in 250, 500, 1000 x 10 / 25 / 50%), subsets 0 to 15, policy seeds 0 to 7, same fixed family and selection protocol as L4a (select on 0 to 7, report on 8 to 15, swapped fold and pooled 16 as cross-checks). Regression gate: at m = 500, subset 0, both judges, thr_agreement, greedy_n3_R1 and all 12 L4a offset arms must equal the per-seed hits in runs/L4a exactly; a failed gate voids the run.

**Command shape:** UNVERIFIED, scripts not written: the L4a launcher pattern (gate, xargs -P 24 over judge x m x subset, summary), results in runs/L5b/, via bin/gpu_run.sh; the launcher must not write done.flag or exit_code.

**Metric:** per cell, held-out errors caught for thr_agreement, selected fixed, L4a selected plain, L4a selected offset, each new arm; each new arm / selected fixed with the 8-subset bootstrap CI; each new arm / better of (L4a plain, L4a offset); the number of cells with B <= 100 at ratio >= 0.97 and with B >= 250 at ratio >= 1.03 per arm.

**Pass threshold:** one new arm, chosen on subsets 0 to 7 as the arm with the highest minimum over the 18 cells of (arm / better of L4a plain and offset), has on subsets 8 to 15 ratio >= 0.97 to the selected fixed rule in all 8 cells with B <= 100 and >= 1.03 in all 6 cells with B >= 250. Reading: a single B-agnostic allocator form exists and the Phase 1 spec takes it instead of a B switch; the ingredient identified (free agreement slope or ridge schedule) is what to port.

**Kill threshold:** no new arm meets both conditions and plain_nocw is within 1% of greedy_n3_R1 in at least 5 of the 6 cells with B >= 250 while offset_freeagree is within 1% of the L4a offset arm in at least 5 of 6. Reading: neither class weighting nor the agreement slope explains the gap, so it comes from learning every feature from labels with no prior, and the B switch (offset below about 125, plain above about 250) is the Phase 1 spec. Anything else is inconclusive.

**Expected runtime:** about 5 to 8 min on CPU (L4a took about 8 min including the gate and sanity jobs).

**Informs:** the Phase 1 allocator form (one arm versus a B switch), whether lam < 1 (the edge of the L4a lam grid) is where the offset family wants to be at large B, and which ingredient explains the plain learner's edge. Lower value than L5a because B is known at allocation time and the B switch is already available.

### L5c: Fresh-population replication at B = 125, 250 and 500 with the L4a arms (GPU, two jobs, smoke check first; needs Rustom's explicit go-ahead under the E04 stop rule; replaces L4c's arm list)

**Hypothesis:** the learner's edge at B >= 250 (L2b and L4a: plain 1.039 to 1.091 against the best fixed member in 6 of 6 cells; L2a: 1.05 to 1.14) replicates on rollouts and prompts that no result so far has seen, and the offset floor (>= 0.97 at B <= 125) holds on fresh data, so the one-population caveat shared by L2a, L2b and L4a does not hide a population-specific effect.

**Change vs existing scripts:** as L4c: avrl/runs/p0 sampled 3,000 of the 7,473 GSM8K train prompts with random_state = 0, leaving 4,473 unused. Copy avrl/p0_rollouts.py to avrl/exp/l5c_rollouts.py with the prompt selection changed to 3,000 prompts from the 4,473 not in p0 (exclude by question text); same policy (Qwen2.5-1.5B-Instruct), G = 8, max_tokens 512; score with avrl/exp/p0_judge_score.py for the 3B and 7B-AWQ judges; build feature tables as in E04. Analysis with the L4a simulator pointed at the new tables: three disjoint blocks of 1,000 fresh prompts; the fixed family (L4a grid) is selected on block 1 and reported on blocks 2 and 3 (swap for a cross-check); learned arms pre-specified, not selected: greedy_n3_R1 (L4a selected plain arm), offset_n0_R5_lam10 and offset_n0_R5_lam1 (L4a offset arms). Budgets: B = 250 and 500 (25% and 50% of a 1,000-prompt block) and B = 125 (50% of four 250-prompt sub-batches per block). L4c's "best offset arm from L4a if L4a passes" condition is not met (L4a is inconclusive) and its pre-specified n10_R5 and n10_R10 arms are no longer the selected plain arms, which is why the arm list changes. Ask Rustom for the override only after L5a has fixed the architecture, because this is the only item that spends GPU on GSM8K.

**Command shape:** UNVERIFIED, scripts not written. Two jobs through bin/gpu_run.sh: rollouts (about 18 min for 24,000 rollouts), then judge scoring of both rungs (about 8 to 15 min), smoke check on 200 prompts first; CPU analysis afterwards.

**Metric:** per judge, B and block: errors caught for thr_agreement, selected fixed, each pre-specified learner, static_out, oracle; learner / selected fixed per block with the swapped-fold check; validity gate: 3B rollout AUC for the group-majority proxy at least 0.65 and the judge label error rate within the E04 range.

**Pass threshold:** greedy_n3_R1 reaches ratio >= 1.03 against the selected fixed rule at B = 250 and B = 500 in both reported blocks in at least 3 of the 4 judge x B cells, and offset_n0_R5_lam10 is >= 0.97 against thr_agreement at B = 125 in all 4 (judge x block) cells. Reading: the B >= 250 edge replicates on a fresh population and the small-B floor holds, so the Phase 1 harness spec (persistent warm-start allocator, small-n0 frequent-refit learner) is justified.

**Kill threshold:** greedy_n3_R1 ratio <= 1.00 in all four judge x B cells on both blocks, or the judge validity gate fails (record as uninformative, not a verdict). Reading: the L2a, L2b and L4a edge was a property of the original 3,000 prompts and claim 2 returns to unsupported.

**Expected runtime:** about 30 to 45 min GPU in two jobs (each under the 45 min cap) plus about 10 min CPU.

**Informs:** claim 2 on an independent population, and the last go / no-go before spending effort on a Phase 1 GRPO harness.

### Iteration 6 additions to 'Proposed by the loop' (after L5a)

Added 2026-10-18 after L5a (INCONCLUSIVE under the pre-registered rules; see avrl/RESEARCH_LOG.md, entry L5a). Results: the regression gate passed (128 per-seed comparisons, 0 differences; 176 carry self-checks, 0 differences); pooled over batches 4 to 12 the persistent learners beat thr_agreement in all four cells (3B and 7B-AWQ at B = 62 and 125): carry_plain 1.088, 1.102, 1.148, 1.143, carry_offset_lam1 (lam chosen on order seeds 0 to 3 in all cells) 1.041, 1.062, 1.037, 1.135 (verdict fold, order seeds 4 to 7: 1.037, 1.061, 1.0295, 1.135), against restart arms at 0.93 to 1.04; PASS missed because 7B-AWQ B=62 is 1.0295 (needs 1.03) and carry_window_250 is within 2% of the carry arm in only 2 of 4 cells (0.984, 0.966, 0.998, 0.972); KILL cells 0 of 4. The B curve is therefore a labels-in-hand effect, not a subset-size artifact, even though carried labels are 65% to 83% errors against a 44% to 47% population rate. calibrate_once with 50 labels is a weak amortized baseline (0.99 to 1.01), so the strongest fixed rule in the persistent setting has not been tried. Status of earlier items: L5b is now lower value (plain beats offset under persistence in 3 of 4 cells; B is known at allocation time); L5c stays blocked on Rustom's explicit override of the E04 stop rule and should add the persistent arms (carry_plain and carry_offset_lam1 over 12 batches of 250 at B = 62 and 125) when it is run, so one GPU spend also answers the one-population caveat for L5a; L1a, L1c and L3b are unchanged. Ids are L<iteration><letter>. Ranked by expected information value.

### L6a: Large-n amortized fixed-rule control against the persistent learner (CPU only)

**Hypothesis:** carry_plain's 7% to 13% total-budget edge over thr_agreement (all 12 batches, L5a post hoc: 1.069, 1.090, 1.118, 1.135) is a labels-in-hand effect that a fixed rule would also get if it were calibrated once on a larger paid label set and then reused free for the rest of the stream; L5a's calibrate_once used only 50 labels (0.991 to 1.014 of thr_agreement) and is too weak to count as the fairest amortized baseline. Falsifier: a seeded amortized rule (calibration labels = lowest-agreement prompts, which double as hits, as in L2a) at n = 250 or 500 reaches within 1% of carry_plain, which would reduce claim 2 in the persistent setting to "calibrate a fixed rule once on a few hundred paid labels".

**Change vs existing scripts:** copy avrl/exp/l5a_carry.py to avrl/exp/l6a_amort.py (copy functions, never import; same 12-batch layouts: order seeds 0 to 7, permutation seed 10000 + order, m = 250, B = 62 and 125, both judges, policy seed = order seed). Re-run thr_agreement, restart_offset, carry_plain, carry_offset_lam1 verbatim as references and gate them against the per-seed hits in runs/L5a (exact equality required, else the run is void). New arms fixed in advance: amort_seed_n for n in 125, 250, 500 (the first n picks across batches 1 to ceil(n / B) are the lowest-agreement prompts of each batch, charged to those batches' budgets and counted as hits; the rest of those batches follows thr_agreement; then one balanced logistic on the ten features, global label-free standardization as in the carry arms, fitted once on the n labels and reused free with the full budget in every later batch) and amort_rand_n for the same n (random calibration picks, otherwise identical; expected to lose at B = 62, kept as the L1b-style baseline). Report amort_seed_n and amort_rand_n at n = 50 as a reproduction check of calibrate_once up to the calibration-pick rule (not an exact-equality gate).

**Command shape:** UNVERIFIED, script not written: the L5a launcher pattern (gate, then xargs -P 24 over judge x budget x order seed, then a summarize script) through bin/gpu_run.sh, results in runs/L6a/; the launcher must not write done.flag or exit_code.

**Metric:** per cell, cumulative hits over all 12 batches (equal total budget, calibration cost included) as a ratio to thr_agreement and to carry_plain, steady state over batches 7 to 12, each with the percentile bootstrap CI over order seeds (10,000 resamples); the best amortized arm chosen per cell on order seeds 0 to 3 and reported on 4 to 7, swapped fold and pooled 8 as cross-checks; first batch at which the cumulative ratio of each arm passes 1.00.

**Pass threshold:** carry_plain / best amortized fixed >= 1.03 over all 12 batches with the CI lower bound above 1 in at least 3 of the 4 cells (3B and 7B-AWQ at B = 62 and 125). Reading: online selection from the learner's own stream adds value beyond having labels, so claim 2 stands against the strongest persistent fixed baseline and the Phase 1 harness is justified on that ground.

**Kill threshold:** best amortized fixed >= 0.99 x carry_plain in at least 3 of 4 cells. Reading: the edge is a label-count effect that any rule fitted on the same labels gets; claim 2 reduces to a one-shot paid calibration and Phase 1 should use the recalibrated fixed rule, not a learned allocator. Anything else is inconclusive.

**Expected runtime:** about 8 to 12 min on CPU (L5a took about 16 min wall with 15 arms per job; this has 8 arms per job, six of them single fits).

**Informs:** the strongest remaining baseline for claim 2, and therefore whether the Phase 1 GRPO harness is justified. Caveats to state in the ledger: no drift by construction, order-seed CIs optimistic (same 3,000 prompts), one population, offline replay.

### L6b: Drift injection and forgetting horizon on the L5a layout (CPU only)

**Hypothesis:** L5a's window curve (W = 500 labels keeps 98.5% to 100.1% of unbounded carry_offset_lam1, W = 125 keeps 93% to 97%) cannot measure staleness, because exchangeable batches have no drift and a window arm there only loses sample size; it gives the minimum label horizon, not the tolerable staleness. With a proxy that changes mid-stream, the persistent learner's edge survives only if it forgets stale labels fast enough, and the best forgetting horizon is the quantity the Phase 1 harness needs (the policy and the verifier both move during training). Falsifier: unbounded carry_plain falls below thr_agreement after the change and no windowed or decayed arm recovers, which would make persistence unsafe without a drift detector.

**Change vs existing scripts:** copy avrl/exp/l5a_carry.py to avrl/exp/l6b_drift.py. Drift built from existing tables only ("judge swap"): for each order seed, batches 1 to 6 are drawn from the 3B feature table and judge_error label (prompts 1 to 1,500 of the permutation) and batches 7 to 12 from the 7B-AWQ table and label (prompts 1,501 to 3,000), so every prompt is used once and the learner sees the proxy change at batch 7 with no signal; each batch is standardized on its own rows (label-free), carried rows keep the z they had when collected. Arms fixed in advance: thr_agreement, restart_offset, carry_plain (unbounded), carry_plain_window_W for W in 125, 250, 500 (last W labels), carry_plain_decay (exponential weights, half-life 250 labels), carry_offset_lam1 unbounded, and reset_at_switch (carry_plain with the label store cleared at batch 7, an upper bound that knows the switch). A no-drift control uses the 3B table for all 12 batches (reproduces L5a for the shared arms, gate: carry_plain and carry_offset_lam1 per-seed hits equal runs/L5a on the 3B cells). Budgets B = 62 and 125, policy seed = order seed.

**Command shape:** UNVERIFIED, script not written: L5a launcher pattern, one job per (budget, order seed, drift or control), results in runs/L6b/, via bin/gpu_run.sh; the launcher must not write done.flag or exit_code.

**Metric:** ratio to thr_agreement in batches 4 to 6 (pre-switch), 7 to 9 (post-switch) and 10 to 12 (recovery), bootstrap CI over order seeds; batches until each arm returns to >= 1.03 after the switch; best W by post-switch ratio.

**Pass threshold:** some windowed or decayed arm beats unbounded carry_plain by >= 3% in batches 7 to 9 in both B cells while staying within 2% of it in batches 4 to 6, and the no-drift gate holds. Reading: the Phase 1 allocator needs a forgetting horizon, the best W is the spec, and E13 gets its first number.

**Kill threshold:** the best of all persistent arms (excluding reset_at_switch) stays below thr_agreement in both batches 7 to 9 and batches 10 to 12 in both B cells. Reading: persistence is unsafe under a proxy change without a drift detector, and Phase 1 must either detect drift or fall back to thr_agreement. Anything else is inconclusive (including the case where unbounded carry is already best, meaning this drift is too mild; then add a harsher one such as flipping the sign of one feature's relation to the label before concluding).

**Expected runtime:** about 8 to 12 min on CPU.

**Informs:** the label retention horizon for Phase 1 and E13 (non-stationarity), and whether the persistent-allocator architecture needs a drift detector. Caveat: a judge swap is one kind of drift (proxy change); policy drift (rollout distribution change under training) is not covered and needs training-time rollouts.

### L6c: Persistence at the thesis-relevant small budget fractions (CPU only)

**Hypothesis:** the persistent learner's edge holds at the budget fractions the thesis cares about (10% to 12.5% of prompts verified per step), and it scales with labels in hand rather than with the per-step fraction, so the carry gain at n labels in hand is the same function of n at B = 25, 62 and 125. Falsifier: carry_plain stays at or below thr_agreement at 10% because the free-label gap itself is small there (static_out / thr_agreement about 1.08 at 3B m = 250 in L4a) and noise dominates.

**Change vs existing scripts:** copy avrl/exp/l5a_carry.py to avrl/exp/l6c_frac.py with only thr_agreement, uniform, restart_offset, carry_plain, carry_offset_lam1 and static_out. Cells (m, B): (250, 25), (250, 31), (500, 50), (500, 62), (500, 125) (m = 500 cuts 6 batches of 500 from the 3,000 prompts), both judges, 8 order seeds, same permutation seed rule; gate: the (250, 62) and (250, 125) cells reproduce runs/L5a per-seed hits for the shared arms exactly.

**Command shape:** UNVERIFIED, script not written: L5a launcher pattern, 80 jobs (5 cells x 2 judges x 8 order seeds), results in runs/L6c/, via bin/gpu_run.sh; must not write done.flag or exit_code.

**Metric:** per cell, ratio to thr_agreement against labels in hand (cumulative labels), pooled over batches where at least 250 labels are in hand, with bootstrap CI over order seeds; carry ratio at matched labels in hand (125, 250, 500) across B at equal fraction (m = 250 B = 31 against m = 500 B = 62, both about 12.5%); static_out / thr_agreement per cell so the carry gain can be read as a share of the free-label gap.

**Pass threshold:** carry_plain / thr_agreement >= 1.03 with the CI lower bound above 1 in at least 3 of the 4 (judge x {10%, 12.5%}) cells, and carry ratios at matched labels in hand differ by <= 0.03 across B within each fraction. Reading: persistence works at the verification fraction the thesis targets and the label count, not the per-step B, is the sizing rule for the Phase 1 harness.

**Kill threshold:** carry_plain <= 1.00 in both 10% cells for both judges. Reading: the persistent edge needs a verification fraction above 10%, which is above the budget the thesis claims to save, and claim 1 would then have to be argued at 25% or more. Anything else is inconclusive.

**Expected runtime:** about 8 to 12 min on CPU.

**Informs:** claim 1's regime (how few verifier calls the allocator can save) and the sizing rule for the Phase 1 per-step verification budget.

### Iteration 7 additions to 'Proposed by the loop' (after L6a)

Added 2026-10-18 after L6a (PASS under the pre-registered rules; see avrl/RESEARCH_LOG.md, entries L6a). Results: the regression gate against runs/L5a passed (0 differences in 2,688 per-batch values) and was re-derived by the loop from the raw JSONs; the best amortized fixed arm (9 pre-registered: seeded, random and n_distinct > 1 calibration picks at n = 125, 250, 500, calibration charged and counted as hits) chosen on order seeds 0 to 7 sits at 0.936, 0.943, 0.885, 0.933 of carry_plain on seeds 8 to 15 (carry_plain / best 1.068, 1.060, 1.129, 1.072, CI lower ends 1.052, 1.030, 1.069, 1.063; pass in 4 of 4 cells, kill in 0 of 4, same in the swapped fold and pooled 16). Decomposition: the same count of labels from the learner's own stream, frozen (freeze_plain_500), keeps 0.988 to 1.001 of carry_plain, fixed rules fitted on thr_agreement, random or n_distinct > 1 picks reach at most 0.945, so what carries the edge is which labels are in hand, not how many; 30 to 60 percent of the best fixed arm's deficit is calibration cost. A post hoc stationary extrapolation says the margin over the best fixed rule shrinks with horizon (carry_plain / best fixed 1.07 and 1.11 at 12 batches falling to 1.01 and 1.02 at 120 batches in the two 7B-AWQ cells, 1.06 to 1.07 falling to 1.03 in the 3B cells), so claim 2 at GRPO horizons rests on drift and on the label stream, which L7a and L7b test. Status of earlier items: L6a done (PASS); L6b is superseded by L7a (same drift layout, freeze arms and both directions added; run L7a instead); L6c and L5c unchanged (L5c is still blocked on Rustom's explicit override of the E04 stop rule, and L7c below folds into it); L1a, L1c, L3b unchanged. Ids are L<iteration><letter>. Ranked by expected information value.

### L7a: Drift injection with the freeze arms (judge swap, both directions; CPU only; supersedes L6b)

**Hypothesis:** L6a showed that a rule frozen after 250 to 560 of the learner's own labels keeps 0.97 to 1.00 of the unbounded persistent learner on exchangeable batches, so continued refitting buys almost nothing when nothing moves. Persistence can therefore only justify itself under drift (the policy and the verifier both move in GRPO). With a proxy that changes mid-stream, a frozen rule should degrade after the switch while a persistent arm with a forgetting horizon recovers; the best window W is the Phase 1 spec. Falsifier: freeze_plain_k6 stays within 1% of the best persistent arm after the switch (the frozen rule transfers across the judge change), or every persistent arm falls below thr_agreement after the switch (persistence is unsafe without a drift detector).

**Change vs existing scripts:** copy avrl/exp/l6a_amort.py to avrl/exp/l7a_drift.py (copy functions, never import). Layout of L6b: for each order seed (16, permutation default_rng(10000 + order seed)), batches 1 to 6 (prompts 1 to 1,500 of the permutation) use table A and batches 7 to 12 (prompts 1,501 to 3,000) use table B, every prompt once; direction 1 = 3B table then 7B-AWQ table, direction 2 = 7B-AWQ then 3B (the 7B-AWQ labels are cleaner, so direction is a confound to report). Each batch is standardized on its own rows (label-free); carried rows keep the z they had when collected. Arms fixed in advance: thr_agreement, restart_offset, carry_plain (unbounded), carry_plain_window_W for W in 125, 250, 500, carry_plain_decay (exponential weights, half-life 250 labels), carry_offset_lam1 (unbounded), freeze_plain_k3 and freeze_plain_k6 (carry_plain through batch k, then the last fit frozen; k <= 6 so the freeze is always before the switch), reset_at_switch (carry_plain with the label store cleared at batch 7; an upper bound that knows the switch, never in the verdict). No-drift controls: table A for all 12 batches in each direction; gate: thr_agreement, carry_plain and carry_offset_lam1 per-seed hits equal runs/L6a on the matching judge, and freeze_plain_k3 at B = 62 equals L6a freeze_plain_125 (k = 3), else the run is void. Budgets B = 62 and 125, policy seed = order seed. Selection of the best persistent arm per cell on order seeds 0 to 7, reported on 8 to 15 (verdict fold), swapped fold and pooled 16, percentile bootstrap over order seeds.

**Command shape:** UNVERIFIED, script not written: the L6a launcher pattern (gate, then xargs -P 24 over direction x B x order seed x {drift, control}, then a summarize script) through bin/gpu_run.sh, results in runs/L7a/; the launcher must not write done.flag or exit_code.

**Metric:** per cell (direction x B, 4 cells), ratio to thr_agreement in batches 4 to 6 (pre-switch), 7 to 9 (post-switch) and 10 to 12 (recovery); headline = best persistent arm / freeze_plain_k6 over batches 7 to 12, with the bootstrap CI over order seeds; batches until each arm returns to >= 1.03 of thr_agreement after the switch; best W by post-switch ratio.

**Pass threshold:** gate passes, and best persistent arm / freeze_plain_k6 >= 1.03 over batches 7 to 12 with the CI lower bound above 1 in at least 3 of the 4 cells, and the best persistent arm is >= thr_agreement in batches 10 to 12 in those cells. Reading: refitting after the proxy moves earns its keep beyond learn-then-freeze; the Phase 1 allocator must keep fitting with a forgetting horizon, the best W is the spec, and the E13 non-stationarity line gets its first number.

**Kill threshold:** (i) freeze_plain_k6 >= 0.99 x the best persistent arm over batches 7 to 12 in at least 3 of 4 cells: this drift does not separate persistence from a one-shot learned rule, Phase 1 can learn-then-freeze, but a harsher drift (for example flipping the sign of one feature's relation to the label) must be tried before concluding persistence is unnecessary; or (ii) the best persistent arm (excluding reset_at_switch) stays below thr_agreement in both batches 7 to 9 and 10 to 12 in at least 3 of 4 cells: persistence is unsafe under a proxy change without a drift detector, and Phase 1 falls back to thr_agreement or adds a detector. Anything else is inconclusive.

**Expected runtime:** about 10 to 15 min on CPU (L6a took 37 s wall for 64 jobs with 19 arms; this has about 13 arms and 128 jobs including controls).

**Informs:** whether the Phase 1 allocator needs persistence at all or only a learn-then-freeze phase, the label retention horizon, and whether a drift detector is required. Caveat: a judge swap is one kind of drift (proxy change); policy drift under training is not covered and needs training-time rollouts.

### L7b: Designed (non-adaptive) calibration sets against the learner's own stream (CPU only)

**Hypothesis:** L6a's decomposition says which labels are in hand, not how many, carries the persistent edge, and its stated mechanism (the greedy stream is not range-restricted the way a lowest-agreement sample is) was consistent with the data but not isolated. A fixed rule fitted once on a covering set chosen without any online adaptation (stratified over the agreement range, or centred on the thr_agreement cutoff) might close the remaining gap to freeze_plain_n. L6a's fixed family (seeded, random, n_distinct > 1 picks) did not include such a design, which the ledger caveat flags as a stronger fixed baseline. Falsifier: the best designed arm reaches >= 0.99 x carry_plain in at least 3 of 4 cells, which would reduce claim 2 to a one-shot designed calibration.

**Change vs existing scripts:** copy avrl/exp/l6a_amort.py to avrl/exp/l7b_design.py (copy functions, never import; same layout: 16 order seeds, 12 batches of 250, B = 62 and 125, both judges). New arms fixed in advance, each at n = 125, 250, 500 with the same calibration structure as amort_rand_n (picks fill batches 1 to k = ceil(n / B), the last batch's remainder ranked by the fitted rule, one StandardScaler + balanced logistic fit, reused free): amort_strat_n (equal numbers of picks from each of 10 quantile bins of the batch's label-free z-agreement) and amort_bnd_n (picks at agreement ranks B/2 to 3B/2 within each batch, i.e. centred on the thr_agreement cutoff). Keep carry_plain, freeze_plain_{125,250,500}, amort_split_n and thr_agreement as references; gate: their per-batch hits equal runs/L6a exactly on order seeds 0 to 15, else the run is void. Selection as L6a (best designed arm per cell on order seeds 0 to 7, reported on 8 to 15; swapped fold and pooled 16). Also report coverage statistics of each fit set (range and spread of z-agreement, label error rate, share with n_distinct > 1) next to the freeze fit sets.

**Command shape:** (as run) from the FULL_R root, `bin/gpu_run.sh avrl/runs/L7b 'bash exp/L7b.sh'` (scripts avrl/exp/l7b_design.py, l7b_summarize.py, l7b_verify.py, L7b.sh; 64 jobs, results in avrl/runs/L7b/; the launcher does not write done.flag or exit_code).

**Metric:** per cell, cumulative hits over all 12 batches as a ratio to carry_plain and to thr_agreement (calibration charged), plus the L6a deficit decomposition (during calibration vs after) and the steady-state ratio to carry_plain after calibration, each with the percentile bootstrap CI over order seeds.

**Pass threshold:** (claim 2 survives designed calibration) carry_plain / best designed >= 1.03 with the CI lower bound above 1 in at least 3 of the 4 cells, gate passing. Reading: the learner's own label stream beats any static covering design at equal budget, so the allocator's online selection is what matters.

**Kill threshold:** best designed arm >= 0.99 x carry_plain in at least 3 of 4 cells. Reading: the edge is available to a one-shot designed calibration, claim 2 reduces to "calibrate once on a covering set", and Phase 1 should use that fixed rule. Anything else is inconclusive; if the 12-batch totals miss but the steady-state ratio after calibration is >= 0.99 in 4 of 4 cells, the result is calibration cost only and feeds L7c (horizon).

**Expected runtime:** about 5 to 10 min on CPU.

**Informs:** the last unexplored fixed-rule family for claim 2 and the coverage mechanism behind the freeze result. Caveats: same as L6a (no drift, optimistic order-seed CIs, one population, offline replay).

**Status:** DONE 2026-10-18, PASS (see avrl/RESEARCH_LOG.md, entry L7b). Gate vs runs/L6a: 0 differences in 11,520 per-batch values (960 vectors, order seeds 0 to 15). Verdict fold: carry_plain / best designed arm 1.113, 1.078, 1.136, 1.067 (CI lower ends 1.101, 1.057, 1.105, 1.054), pass in 4 of 4 cells, kill in 0 of 4 (swapped fold and pooled 16 the same). Best designed arm is 0.880 to 0.937 of carry_plain at a 12-batch horizon; n = 500 designs reach 0.96 to 1.00 of carry_plain after calibration, so the horizon (L7c) is the one open route.

### L7c: Long-horizon replication on additional GSM8K prompts (GPU; BLOCKED on Rustom's explicit override of the E04 stop rule; fold into L5c)

**Hypothesis:** the L6a post hoc stationary extrapolation predicts that carry_plain / best fixed arm falls with horizon (about 1.06, 1.05, 1.07, 1.04 at 29 batches in cell order, from 1.07, 1.06, 1.11, 1.07 at 12 on the same extrapolation basis), because calibration cost is amortized away; 12 batches is the whole 3,000-prompt table, so only new prompts can test it. Falsifier: carry_plain / best fixed < 1.03 at 29 batches in at least 3 of 4 cells.

**Change vs existing scripts:** run the E04 pipeline (E04.sh rollout and judge steps; copy, do not edit) on the unused rows of data/gsm8k_train.parquet (7,473 rows, E04 used 3,000; up to 4,473 more, so 7,473 prompts = 29 batches of 250), for the 3B and 7B-AWQ judges; then rerun the L6a analysis (copy l6a_amort.py and l6a_summarize.py) on the 7,473-prompt tables with 29 batches, the same 9 amortized arms plus freeze_plain_n, 16 order seeds, B = 62 and 125. Gate: the first 12 batches of the 3,000 original prompts reproduce runs/L6a hits exactly.

**Command shape:** UNVERIFIED, scripts not written: generation and judging through bin/gpu_run.sh (one job per judge or per half of the prompts so each stays under the 45-minute cap), analysis on CPU. Do not launch without Rustom's explicit override of the E04 stop rule.

**Metric:** carry_plain / best fixed arm (chosen on order seeds 0 to 7, reported on 8 to 15) over batches 1 to 12, 1 to 20 and 1 to 29 per cell, bootstrap CI over order seeds; cumulative ratio to thr_agreement against batch index.

**Pass threshold:** carry_plain / best fixed >= 1.03 with the CI lower bound above 1 at 29 batches in at least 3 of 4 cells. Reading: the persistent edge survives a longer stationary horizon, and the extrapolation caveat on claim 2 is retired.

**Kill threshold:** best fixed >= 0.99 x carry_plain over 29 batches in at least 3 of 4 cells. Reading: at long horizons a one-shot calibrated fixed rule matches the learner, and claim 2 survives only through drift (L7a). Anything else is inconclusive.

**Expected runtime:** UNVERIFIED; two E04-sized GPU passes on 4,473 prompts (check the E04 wall time in runs/E04/run.log before sizing), analysis about 5 min on CPU.

**Informs:** whether claim 2's margin over a fixed rule is a finite-horizon effect, and the horizon at which the Phase 1 harness should evaluate it. Same population and task, so the one-population caveat is not retired by this experiment.

### Iteration 8 additions to 'Proposed by the loop' (after L7a)

Added 2026-10-18 after L7a (PASS under the pre-registered rules; see avrl/RESEARCH_LOG.md, entry L7a). Results: the regression gate against runs/L6a passed (0 differences in 2,112 per-batch values, re-derived by the loop from the raw JSONs); under a one-step judge swap (batches 7 to 12) the best persistent arm / freeze_plain_k6 is 1.038, 1.078, 1.039, 1.006 in the verdict fold (cells d1 B=62, d1 B=125, d2 B=62, d2 B=125; CI lower ends 1.030, 1.045, 1.017, 1.002; pass in 3 of 4, kill (i) in 1 of 4, kill (ii) in 0 of 4), and the best persistent arm is 1.07 to 1.15 of thr_agreement in both post-switch groups. But unbounded carry_plain is the best persistent arm in 3 of 4 cells (best / carry_plain 1.000 to 1.004), windows and decay do not beat it, reset_at_switch (which knows the switch) is below it, and the drift-specific part of the edge over freeze_plain_k6 is clear in only two cells (d1 B=125 and d2 B=62). The loop's transfer check explains why: a rule fitted on one judge's labels transfers to the other at AUC 0.868 and 0.859 against 0.882 and 0.871 in-sample, so this judge swap changes label noise but nearly preserves the feature-to-label relation, and the per-batch z removes the mean shift. The null case in the L7a spec fired, so a harsher drift is needed before concluding anything about forgetting. Negating the z of judge_top and judge_mean on the target table drops the transferred AUC to 0.489 (3B to 7B-AWQ) and 0.674 (7B-AWQ to 3B). Status of earlier items: L7a done (PASS, narrow reading); L7b unchanged; L7c and L5c unchanged and still blocked on Rustom's explicit override of the E04 stop rule; L6c unchanged. Ids are L<iteration><letter>. Ranked by expected information value.

### L8a: Sign-flip drift on the L7a layout (CPU only)

**Hypothesis:** L7a's judge swap barely moves the feature-to-label relation (transfer AUC loss 0.012 to 0.014), so it cannot tell a forgetting horizon from carrying stale labels. If the relation itself changes after the switch (here: the z of judge_top and judge_mean negated on every post-switch row, labels unchanged), the table-A rule becomes anti-informative on the judge features, thr_agreement (agreement only) is untouched, and unbounded carry_plain should fall below thr_agreement until the stale labels are outvoted, while a window or decay arm recovers sooner. Falsifiers: (i) unbounded carry_plain stays within 1% of the best forgetting arm over batches 7 to 12 (forgetting is not needed even for a relation change), or (ii) every persistent arm stays below thr_agreement in both post-switch groups (persistence is unsafe without a detector).

**Change vs existing scripts:** copy avrl/exp/l7a_drift.py to avrl/exp/l8a_flip.py (copy functions, never import; l7a files untouched). Same layout (16 order seeds, permutation default_rng(10000 + order seed), 12 batches of 250, batches 1 to 6 from table A, 7 to 12 from table B, both directions, B = 62 and 125, policy seed = order seed). After the per-batch z, multiply the judge_top and judge_mean columns of every post-switch row by a flip factor f (f = -1 in the experiment, f = +1 as the gate). Arms: all L7a arms (thr_agreement, restart_offset, carry_plain, windows 125 / 250 / 500, decay with half-life 250, carry_offset_lam1, freeze_plain_k3 / k6, reset_at_switch) plus carry_plain_decay_hl100. Gate: the f = +1 run reproduces runs/L7a drift per-batch hits exactly for every shared arm on order seeds 0 to 15, else the run is void. Selection as L7a (best persistent arm chosen on seeds 0 to 7, reported on 8 to 15, swapped fold and pooled 16, bootstrap over order seeds). Report the direction confound and keep the directions separate.

**Command shape:** UNVERIFIED, script not written: the L7a launcher pattern (xargs -P 24 over direction x B x order seed x {flip, gate}, then a summarize script) through bin/gpu_run.sh, results in runs/L8a/; the launcher must not write done.flag or exit_code.

**Metric:** per cell, ratio to thr_agreement in batches 4 to 6, 7 to 9 and 10 to 12; best forgetting arm (windows, decays, chosen on 0 to 7) / carry_plain over batches 7 to 12 and 10 to 12 with the bootstrap CI over order seeds; best persistent arm / freeze_plain_k6; batches until each arm returns to >= 1.03 of thr_agreement; reset_at_switch next to the headline as an upper bound.

**Pass threshold:** gate passes, and the best forgetting arm beats unbounded carry_plain by >= 3% over batches 7 to 12 with the CI lower bound above 1 in at least 3 of 4 cells, and is >= thr_agreement in batches 10 to 12 in those cells. Reading: a forgetting horizon is required when the proxy relation changes, the best W (or half-life) is the Phase 1 spec, and E13 gets its harsh-drift number.

**Kill threshold:** (i) unbounded carry_plain >= 0.99 x the best forgetting arm in at least 3 of 4 cells: persistence needs no forgetting horizon even under a relation flip, and the Phase 1 spec stays carry or learn-then-freeze; or (ii) the best persistent arm (excluding reset_at_switch) stays below thr_agreement in both batches 7 to 9 and 10 to 12 in at least 3 of 4 cells: persistence is unsafe under a relation change, and Phase 1 needs a detector (L8c) or a fallback to thr_agreement. Anything else is inconclusive. VOID if the gate fails.

**Expected runtime:** about 10 min on CPU (L7a took about 10 min for 192 jobs; this has about 128).

**Informs:** whether Phase 1 needs a forgetting horizon or a drift detector at all, the retention horizon, and the last unexplored robustness axis of claim 2 on this machine. Caveats to state in the ledger: a feature-sign flip is a synthetic drift (the labels are real, the relation is edited), step-shaped, with a fixed policy; order-seed CIs optimistic; one population, offline replay.

### L8b: Equal-label staleness test and gradual drift (CPU only)

**Hypothesis:** L7a's pass rule could not separate staleness from label count, because freeze_plain_k6 stops collecting labels at batch 6 while the persistent arms keep collecting. The loop's drift-excess table shows a freeze penalty that is clearly drift-caused in d1 B=125 (1.057 to 1.095) and d2 B=62 (1.022 to 1.047, reset_at_switch 1.084) and absent in d2 B=125. A fresh window of exactly the freeze set's size, compared with freeze_plain_k6 (same label count, only freshness differs), isolates staleness, and a gradual switch (table-B share rising over several batches) is closer to policy drift in GRPO than a step. Falsifier: the equal-label fresh window is within 1% of freeze_plain_k6 after a gradual or step switch in at least 3 of 4 cells (the freeze penalty is a label-count effect, not staleness).

**Change vs existing scripts:** copy avrl/exp/l7a_drift.py to avrl/exp/l8b_stale.py. Layouts: step switch after batch 4, 6 or 8, and a gradual layout in which batch j in 4 to 9 draws a fraction (j - 3) / 7 of its rows from table B (rows still disjoint, each prompt once). New arms: window_eq (a fresh window of the same number of labels as freeze_plain_k6 holds at the freeze point, i.e. the last n_k6 labels, n_k6 = 372 at B = 62 and 750 at B = 125 on average), freeze_plain_k6, carry_plain, carry_plain_window_250, thr_agreement; controls with no drift (all table A) for each. Gate: step-6 drift reproduces runs/L7a hits for the shared arms on order seeds 0 to 15 exactly. Cells: direction x B = 4.

**Command shape:** UNVERIFIED, script not written: L7a launcher pattern through bin/gpu_run.sh, results in runs/L8b/; the launcher must not write done.flag or exit_code.

**Metric:** window_eq / freeze_plain_k6 over the batches after the switch, divided by the same ratio in the no-drift control (drift-specific staleness effect), with the bootstrap CI over order seeds, per layout and cell; carry_plain / freeze_plain_k6 likewise.

**Pass threshold:** the drift-specific window_eq / freeze_plain_k6 excess is >= 1.03 with the CI lower bound above 1 in at least 2 of the 4 cells in at least 2 of the 4 layouts. Reading: stale labels are a real cost at equal label count, the Phase 1 refresh interval is bounded by the label turnover time, and the W for the L8a result gets a second source.

**Kill threshold:** window_eq / freeze_plain_k6 excess <= 1.01 in at least 3 of 4 cells in every layout. Reading: staleness is not the mechanism, the L7a edge over freeze is label count, and learn-then-freeze plus a longer collection phase is the simpler spec. Anything else is inconclusive.

**Expected runtime:** about 10 to 15 min on CPU.

**Informs:** the freeze point and refresh interval for the Phase 1 allocator, and the attribution of the L7a edge. Same caveats as L7a.

### L8c: Drift-detector arm (CPU only; run only if L8a shows unbounded carry_plain below thr_agreement or a forgetting arm beating it)

**Hypothesis:** if L8a shows persistence is unsafe or forgetting is needed under a relation change, an online detector that clears or down-weights the label store when the learner's realized precision collapses recovers most of reset_at_switch's upper bound without knowing the switch. Falsifier: the detector arm recovers less than half of the gap between carry_plain and reset_at_switch in batches 7 to 12 in at least 3 of 4 cells, or false-resets in the no-drift control cost more than 1%.

**Change vs existing scripts:** copy avrl/exp/l8a_flip.py to avrl/exp/l8c_detect.py. New arm detect_reset: after each batch, compare the hit rate of the batch's picks with the fit's own predicted precision on those picks; if the realized rate is more than 2 binomial SE below it in two consecutive batches, clear the store (reset_at_switch behaviour from there). Detector parameters fixed in advance, no tuning on the verdict seeds. Controls: no-drift runs (false-reset rate), L7a step layout, L8a flip layout. Gate: arms shared with L8a reproduce runs/L8a exactly.

**Command shape:** UNVERIFIED, script not written: L7a launcher pattern through bin/gpu_run.sh, results in runs/L8c/; the launcher must not write done.flag or exit_code.

**Metric:** detect_reset position between carry_plain (0) and reset_at_switch (1) over batches 7 to 12; detection delay in batches; false-reset rate and cost in the no-drift control; ratio to thr_agreement.

**Pass threshold:** detect_reset recovers >= 50% of the carry_plain to reset_at_switch gap in at least 3 of 4 cells under the flip layout, with a false-reset cost <= 1% in the no-drift control. Reading: a cheap detector makes persistence safe, and the Phase 1 allocator is persistent plus detector.

**Kill threshold:** recovers < 25% of the gap in at least 3 of 4 cells, or false-resets cost > 2% in the control. Reading: this detector family does not work at these label counts, and Phase 1 should fall back to thr_agreement or a short window. Anything else is inconclusive.

**Expected runtime:** about 10 min on CPU.

**Informs:** whether the Phase 1 harness needs a drift detector. Caveats: synthetic drift, fixed policy, detector family minimal.

### Iteration 9 additions to 'Proposed by the loop' (after L7b)

Added 2026-10-18 after L7b (PASS under the pre-registered rules; see avrl/RESEARCH_LOG.md, loop-level entry "L7b 2026-10-18 pass"). Results: the regression gate against runs/L6a passed (0 differences in 11,520 per-batch values, re-derived by the loop from the raw JSONs); the best of the six designed calibration arms (amort_strat_n, amort_bnd_n at n = 125, 250, 500, chosen on order seeds 0 to 7) is 0.898, 0.928, 0.880, 0.937 of carry_plain on seeds 8 to 15 (carry_plain / best 1.113, 1.078, 1.136, 1.067, CI lower ends 1.101, 1.057, 1.106, 1.054; pass in 4 of 4 cells, kill in 0 of 4, same in the swapped fold and pooled 16). New loop finding: at equal label count (B = 125, where freeze_plain_n holds exactly n labels), the post-calibration hits of strat_n and bnd_n are 0.987 to 1.018 of freeze_plain_n in all 12 (cell, n) pairs, so the composition of the fit set (the learner's error-rich stream versus a stratified random or boundary-window set) does not drive post-calibration rule quality; the learner's 12-batch edge is earn-while-labelling (calibration batches) plus continued refitting past n labels. The 12-batch kill rule therefore has little power against a design that does not earn while calibrating (n = 125 at B = 62 already loses about 6 percent of the total in calibration), and the horizon is the one remaining route by which a one-shot design reaches 0.99. Status of earlier items: L7b done (PASS); L7c unchanged and still blocked on Rustom's explicit override of the E04 stop rule (L9a below is the CPU stand-in); L8a, L8b, L8c unchanged and still queued (L8a stays the top drift test); L6c and L5c unchanged. Ids are L<iteration><letter>. Ranked by expected information value.

### L9a: Long-horizon stationary replay with iid-with-replacement streams (CPU only)

**Hypothesis:** the L6a and L7b post hoc extrapolations say a one-shot n = 500 design closes on carry_plain as the horizon grows (strat_500 / carry_plain 0.854 to 0.951 at 29 batches, 0.940 to 0.991 at 120), and the L7b equal-label comparison says a frozen designed rule is as good as the learner's frozen rule at equal n, so the only thing separating them at long horizons is continued refitting beyond n labels. 12 batches is the whole 3,000-prompt table, so the horizon has never been measured; L7c needs GPU and the E04 override. Falsifier (kills claim 2 at long horizon under stationarity): the best fixed arm reaches >= 0.99 x carry_plain over a 60-batch stream in at least 3 of 4 cells.

**Change vs existing scripts:** copy avrl/exp/l7b_design.py to avrl/exp/l9a_horizon.py (copy functions, never import). New layout: each batch draws m = 250 prompts uniformly with replacement from the 3,000-prompt table (stream seed = policy seed), H = 60 batches (prefixes 12, 24, 36, 60 reported; 120 only if the 45-minute cap allows after a timing check). Arms: thr_agreement, uniform, carry_plain, freeze_plain_n, amort_strat_n, amort_split_n and amort_seed_n for n in 125, 250, 500, 1000 (n = 1000 added because long horizons favour large n; k = ceil(n / B) can reach 17 at B = 62). Same B = 62 and 125, both judges, 16 stream seeds, 64 jobs. Gate: a mode "disjoint12" in the same script (the L7b permutation layout) reproduces runs/L7b hits for the shared arms exactly on order seeds 0 to 15, else the run is void. Memorisation control, fixed in advance: after the last batch, the final carry_plain fit's AUC on prompts it never labelled versus on prompts it did; if the gap exceeds 0.02 in any cell, rerun that cell with already-paid prompts removed from later candidate pools (fresh draw refills the batch). Selection as L7b: best fixed arm per cell chosen on stream seeds 0 to 7, reported on 8 to 15, swapped fold and pooled 16 as cross-checks.

**Command shape:** UNVERIFIED, script not written: the L7b launcher pattern (xargs -P 24 over judge x B x stream seed, then a summarize script with the gate) through bin/gpu_run.sh, results in runs/L9a/; time one H = 60 job first (an H = 12 job took about 12 s, so expect about 1 to 3 min), the launcher must not write done.flag or exit_code.

**Metric:** carry_plain / best fixed arm (chosen on seeds 0 to 7, reported on 8 to 15) cumulative over the first H batches for H = 12, 24, 36, 60, with percentile bootstrap CI over stream seeds, plus the same ratio over the last 12 batches (steady state) and the post-calibration ratio per n; report the H at which the ratio crosses 1.03 and 0.99 by extrapolation in 1/H.

**Pass threshold:** carry_plain / best fixed >= 1.03 with the CI lower bound above 1 at H = 60 in at least 3 of 4 cells. Reading: the persistent edge survives a long stationary horizon, and the finite-horizon caveat on claim 2 is retired for this population.

**Kill threshold:** best fixed >= 0.99 x carry_plain at H = 60 in at least 3 of 4 cells. Reading: at long horizons a one-shot calibrated rule matches the learner under stationarity, claim 2 survives only through drift (L8a), and Phase 1 should be learn-then-freeze at the n that wins here. Anything else is inconclusive and the slope (ratio against H) is the result. The loop's prior from the extrapolation: inconclusive in most cells, kill possible only in 7B-AWQ B=125.

**Expected runtime:** about 10 min on CPU (64 jobs at H = 60).

**Informs:** the Phase 1 recipe (persistent versus learn-then-freeze, and n), and whether claim 2 is a finite-horizon effect. Caveats to state in the ledger: iid-with-replacement draws from one 3,000-prompt table (duplicate prompts, the same weak-judge labels), no drift, a 10-feature logistic cannot memorise prompts but the control above checks it, order-seed-like CIs over stream seeds are optimistic about prompt sampling.

### L9b: Earn-while-calibrating designed rule (CPU only)

**Hypothesis:** the L7b loop finding says every designed arm loses to the learner through calibration-phase earnings (random-like picks catch 0.44 to 0.58 of B against 0.68 to 0.78) and through freezing at small n, not through the label set it fits on. A non-adaptive design that keeps most of the calibration earnings is a mix: each calibration batch spends a fixed share (1 - e) of B on the lowest-agreement picks and a share e on stratified exploration picks, the rule is fitted once on all n paid labels and reused free. This is the last fixed family that can earn while calibrating without using the learner's own labels to choose picks. Falsifier: the best mix arm reaches >= 0.99 x carry_plain in at least 3 of 4 cells.

**Change vs existing scripts:** copy avrl/exp/l7b_design.py to avrl/exp/l9b_mix.py. New arms amort_mix_e_n with e in 0.25, 0.5 and n in 250, 500 (4 arms; e and n fixed in advance, nothing chosen on the verdict seeds beyond the standard 0-to-7 selection): in batches 1..k-1 the picks are the (1 - e) B lowest-agreement ranks plus e B equal-population stratified picks from the rest; the batch-k split and the free top-B afterwards follow amort_strat_n; one-class fit falls back to thr_agreement. Keep carry_plain, thr_agreement, freeze_plain_n, amort_strat_n and amort_bnd_n as references; gate: their per-batch hits equal runs/L7b exactly on order seeds 0 to 15, else void. Selection and folds as L7b.

**Command shape:** UNVERIFIED, script not written: the L7b launcher pattern through bin/gpu_run.sh, results in runs/L9b/; the launcher must not write done.flag or exit_code.

**Metric:** per cell, 12-batch hits of each mix arm as a ratio to carry_plain and to thr_agreement, and the post-calibration ratio, with bootstrap CIs over order seeds; also the equal-label post-calibration ratio to freeze_plain_n at B = 125.

**Pass threshold:** carry_plain / best mix >= 1.03 with the CI lower bound above 1 in at least 3 of 4 cells, gate passing. Reading: even a design that earns while calibrating does not reproduce the learner at 12 batches, and the "designed set" line of attack on claim 2 is closed. The loop's prior from the L7b numbers is a pass at about 0.93 to 0.95 of carry_plain.

**Kill threshold:** best mix >= 0.99 x carry_plain in at least 3 of 4 cells. Reading: a fixed, label-free design recovers the learner, claim 2 reduces to "calibrate once with a mixed design". Anything else is inconclusive.

**Expected runtime:** about 5 to 10 min on CPU.

**Informs:** whether the calibration-phase earnings are available to a non-adaptive design; low to medium value because the expected outcome is another pass. Same caveats as L7b.

### L9c: Prompt-population bootstrap of the L6a and L7b verdict counts (CPU only)

**Hypothesis:** every CI so far is over order seeds on the same 3,000 prompts, so none includes prompt-sampling variance. The pass margins (carry_plain / best fixed 1.06 to 1.14 against the 1.03 bar) are probably larger than that noise, but it was never measured. Falsifier: with the population itself resampled, carry_plain / best designed or fixed arm drops below 1.03 or the CI lower bound falls to 1 or below in at least 2 of 4 cells.

**Change vs existing scripts:** copy avrl/exp/l7b_design.py to avrl/exp/l9c_boot.py. For resample r in 0 to 15, draw 3,000 prompt indices with replacement from the table (seed 20000 + r), then run the L7b layout (12 disjoint batches of 250 from a permutation of the resampled population, order seed = r mod 4 nested under r) with the L6a and L7b arms. Selection per cell: best fixed arm over the 15 amortized and designed arms chosen on resamples 0 to 7, reported on 8 to 15. Gate: the original (non-resampled) layout reproduces runs/L7b exactly for the shared arms.

**Command shape:** UNVERIFIED, script not written: the L7b launcher pattern through bin/gpu_run.sh, results in runs/L9c/; the launcher must not write done.flag or exit_code.

**Metric:** per cell, carry_plain / best fixed arm over 12 batches with the bootstrap CI over resamples, and the ratio of the across-resample SD to the across-order-seed SD from L7b (variance inflation of the old CIs).

**Pass threshold:** carry_plain / best fixed >= 1.03 with the CI lower bound above 1 in at least 3 of 4 cells. Reading: the L6a and L7b verdict counts survive prompt-sampling variance, and the optimistic-CI caveat can be softened.

**Kill threshold:** fewer than 2 of 4 cells clear 1.03 with CI lower above 1. Reading: the 12-batch edge is within prompt-sampling noise on this population. Anything else is inconclusive.

**Expected runtime:** about 5 to 10 min on CPU.

**Informs:** how much weight the order-seed CIs in L6a, L7a, L7b deserve. Caveats: bootstrap duplicates prompts (a duplicate in the fit and stream sets is a mild leak), still one population and one task, so it does not replace L5c or L7c.

### Iteration 10 additions to 'Proposed by the loop' (after L9a)

Added 2026-10-18 after L9a (formally VOID: the pre-registered memorisation control fired in 3 of 4 cells and its remedy cannot reach H = 60 on a 3,000-prompt table; the PASS / KILL rule applied with the control waived reads PASS in 3 of 4 cells and KILL in 0 of 4; see avrl/RESEARCH_LOG.md, entry "L9a 2026-10-18 void"). Results: regression gate against runs/L7b passed (0 differences in 19,968 per-batch values); carry_plain / best fixed arm (chosen on stream seeds 0 to 7, reported on 8 to 15) at H = 60 is 1.085, 1.063, 1.060, 1.025 (3B B=62, 3B B=125, 7B-AWQ B=62, 7B-AWQ B=125; CI lower ends 1.075, 1.059, 1.048, 1.013), best / carry 0.922, 0.940, 0.943, 0.975, so the kill rule holds in 0 of 4; the ratio is flat from H = 24 to 120 in the first three cells and decays 1.059, 1.047, 1.034, 1.025, 1.016 over H = 12, 24, 36, 60, 120 in 7B B=125, whose last-12 steady state is 0.982 (60) and 0.979 (120), a fixed arm ahead of the learner. Control: the pre-registered AUC gap (labelled minus never labelled) is 0.026, 0.013, 0.313, 0.332 and is confounded by selection; a split-half in-sample minus held-out gap is 0.057, 0.019, 0.041, 0.023 on iid streams against 0.004 to 0.008 on no-repeat streams; 72 to 78 percent of carry_plain's picks re-pay a known prompt and its hits are four times the number of error prompts in the table. The no-repeat probe (49 to 51 batches at B = 62, 25 at B = 125, depleted stream) moves the 7B cells by -0.03 to -0.11 toward parity. Status of earlier items: L9a done (void); L7c unchanged and blocked on the E04 override; L8a still queued; L8b, L8c, L9b, L9c, L6c, L5c unchanged. Ids are L<iteration><letter>. Ranked by expected information value.

### L10a: Fresh-prompt long-horizon stream (GPU, blocked on the E04 override)

**Hypothesis:** the L9a ratio-versus-H slope is partly an artifact of drawing 60 batches from a 3,000-prompt table that contains only 1,313 to 1,419 error prompts: at B = 125 the learner makes 7,500 picks and 72 to 78 percent of them re-pay a known prompt. With at least 10,000 additional GSM8K prompts (fresh rollouts, E04 judge features) a 60-batch stream at B = 125 needs no repeat, the memorisation control is satisfiable, and the horizon question is answerable. Falsifier (kills claim 2 at long horizon): the best fixed arm reaches >= 0.99 x carry_plain at H = 60 in at least 3 of 4 cells on the fresh stream.

**Change vs existing scripts:** the L7c generation pipeline (p0_rollouts.py copy) for a new prompt block, E04-style judge features for it, then avrl/exp/l9a_horizon.py with a new layout mode "fresh" (batches are disjoint draws from the enlarged table, no replacement within the run), the same arms and selection, the original 3,000 prompts kept as the calibration-free reference (gate: mode disjoint12 reproduces runs/L7b).

**Command shape:** UNVERIFIED, script not written: gpu_run.sh for the rollouts, then the L9a launcher pattern; needs Rustom's explicit override of the E04 stop rule.

**Metric:** carry_plain / best fixed at H = 12, 24, 36, 60, steady state over the last 12 batches, post-calibration ratio per n, percentile bootstrap over stream seeds; the split-half memorisation gap per cell.

**Pass threshold:** carry_plain / best fixed >= 1.03 with the CI lower bound above 1 at H = 60 in at least 3 of 4 cells, and the memorisation control (split-half gap <= 0.02) satisfied.

**Kill threshold:** best fixed >= 0.99 x carry_plain at H = 60 in at least 3 of 4 cells. Anything else is inconclusive.

**Expected runtime:** rollouts and judge scoring about 1 to 2 hours on the GPU, replay about 15 min on CPU.

**Informs:** whether claim 2 survives a genuinely long stationary horizon and the Phase 1 recipe (persistent versus learn-then-freeze, n). Caveats: needs the override; one task.

### L10b: Corrected memorisation control and per-arm label cache (CPU only)

**Hypothesis:** the 7B-AWQ parity in the L9a no-repeat probe (carry / best 0.909 to 0.995 at 25 to 49 batches) is real and not an artifact of the depleted common stream, which means the 7B iid edge is partly re-catching known errors. Falsifier of the iid reading: with a per-arm label cache (a known prompt is free, not counted, and the budget refills with the arm's next-ranked prompt) the carry_plain / best fixed ratio at H = 24 stays >= 1.03 with the CI lower end above 1 in at least 3 of 4 cells.

**Change vs existing scripts:** copy avrl/exp/l9a_horizon.py to l10b_cache.py; add a per-arm set of paid prompts to carry_plain, amortized and freeze_plain (picks skip prompts the arm has paid, hits count first-time labels only); H capped at about 25 batches (the supply of unpaid error prompts); the memorisation control replaced by the split-half in-sample minus held-out AUC gap with threshold 0.02 (no-repeat baseline 0.004 to 0.008). Gate: mode disjoint12 reproduces runs/L7b exactly.

**Command shape:** UNVERIFIED, script not written: the L9a launcher pattern through bin/gpu_run.sh, results in runs/L10b/; the launcher must not write done.flag or exit_code.

**Metric:** per cell, carry_plain / best fixed arm (chosen on stream seeds 0 to 7, reported on 8 to 15) at H = 12 and 24 with bootstrap CIs, steady-state ratio over batches 13 to 24, and the split-half gap per cell.

**Pass threshold:** carry_plain / best fixed >= 1.03 with the CI lower bound above 1 at H = 24 in at least 3 of 4 cells with the control satisfied. Reading: the learner's edge survives removing re-paid prompts.

**Kill threshold:** best fixed >= 0.99 x carry_plain at H = 24 in at least 2 of 4 cells. Reading: the iid edge is largely re-catching known errors. Anything else is inconclusive.

**Expected runtime:** about 10 min on CPU.

**Informs:** how much of the L6a, L7b and L9a edge is a table-size artifact, and whether L10a is worth the override. Caveats: a per-arm cache changes the payment model; the horizon is short.

### Iteration 10 loop-level additions to 'Proposed by the loop' (after the L9a review)

Added 2026-10-18 by the loop after re-deriving L9a from the raw JSONs (avrl/RESEARCH_LOG.md, loop-level entry "L9a 2026-10-18 inconclusive"). Loop verdict: inconclusive (formally VOID; control waived reads PASS 3 of 4, KILL 0 of 4). L10a and L10b above stand as written. Loop ranking by expected information value: (1) L10b, because the L9a no-repeat probe predicts its 7B outcome (carry / best 1.005 and 0.995 at H = 24 in the probe) but not the 3B outcome, so the result is not predetermined; (2) L10c below, which keeps the payment model and varies the repeat share, and decides whether L10a is worth the E04 override; (3) L8a, unchanged. L10a waits for L10b and L10c. Ids are L<iteration><letter>.

### L10c: Table-size dose-response of the iid edge (CPU only)

**Hypothesis:** in L9a, 72 to 78 percent of carry_plain's picks re-pay a known prompt, and the no-repeat probe moved the 7B carry / best ratio by -0.03 to -0.11 (toward parity) and the 3B B=62 ratio by +0.07 to +0.09, so the repeat share changes the ratio in a cell-dependent direction. Shrinking the table raises the repeat share without changing the payment model or needing new generation: if the iid edge is partly re-catching known errors, carry_plain / best fixed at H = 60 rises as the table shrinks (the 7B cells first), and extrapolating the dose-response to zero repeats brackets the no-repeat value. Falsifier of the re-pay explanation: the ratio is flat within 0.02 across N = 750, 1500, 3000.

**Change vs existing scripts:** copy avrl/exp/l9a_horizon.py to l10c_tablesize.py; add a subtable argument (N prompts drawn without replacement from the 3,000 with a fixed rng, one fixed subset per N and per subset seed 0 to 3, nested so that the N = 750 set is inside the N = 1500 set) and run the iid stream (m = 250, H = 60, stream rng 30000 + seed) on it; same arms, same selection (best fixed arm chosen on stream seeds 0 to 7, reported on 8 to 15), same B = 62 and 125, both judges, 16 stream seeds, N in 750, 1500, 3000 (N = 3000 must reproduce runs/L9a hz_* exactly: regression gate, else void). Record the repeat share and the split-half memorisation gap per (cell, N).

**Command shape:** UNVERIFIED, script not written: the L9a launcher pattern (xargs -P 24 over judge x B x N x subset seed x stream seed) through bin/gpu_run.sh, results in runs/L10c/, then a summarize script; the launcher must not write done.flag or exit_code. Time one job first (an H = 60 job took 31 s).

**Metric:** per cell and N, carry_plain / best fixed at H = 12, 24, 60 with percentile bootstrap CI over stream seeds, repeat share of carry_plain's picks, split-half gap; the slope of carry / best against log(N) per cell with its bootstrap CI.

**Pass threshold:** carry_plain / best fixed at H = 60 is higher at N = 750 than at N = 3000 by >= 0.03 (CI lower end above 0) in at least 3 of 4 cells, gate passing. Reading: the iid edge is partly re-caught known errors; the no-repeat limit is below the L9a headline and L10b is the decisive test.

**Kill threshold:** the N = 750 versus N = 3000 difference is within +-0.02 in at least 3 of 4 cells. Reading: re-pay is not the driver of the edge and L10a is not worth the E04 override. Anything else is inconclusive.

**Expected runtime:** about 15 min on CPU (3 table sizes x 64 jobs, about 30 s each, 24 parallel).

**Informs:** whether the L6a, L7b and L9a edges are partly a table-size artifact, and whether L10a (GPU, override) is needed. Caveats: subtables also scale the error supply and change the error-prompt count (about 330, 660, 1,310), one population and task, no drift.

### Iteration 11 additions to 'Proposed by the loop' (after L1a)

Added 2026-10-19 after L1a (kill for claim 3 on GSM8K in the agreement-free rollout-level regime; see avrl/RESEARCH_LOG.md, entry "L1a 2026-10-19 kill"). Results re-derived from avrl/runs/L1a/: gate passed (108 policy-budget cells, 0 differences vs runs/E04); on 3B and 7B-AWQ the 16 PCs add +0.0097 and +0.0084 OOF AUC to the 0.7045 and 0.7090 cheap_r model (CI upper ends 0.0149 and 0.0152; permutation null about -0.003), cell encodings add at most +0.0013, ts_online / greedy_online at 25 percent is 1.013 and 0.996 (greedy with PCs / without 1.012 and 0.996), the widened KILL rule holds on both rungs and PASS fails; the literal backlog rule fires neither. Claim-2 side point: greedy_online is 1.010 and 1.011 over the best charged recal rule at 25 percent (1.013 and 1.005 at 50 percent) and 0.96 to 0.97 of the free-label static ceiling, at absolute budgets about 5x those of L1b. Status of earlier items: L1a done (kill, claim 3 on GSM8K); L1c stays blocked on the E04 stop rule and, on the L1a evidence alone, is not worth an override request, gated on L11b below; L10a, L10b, L10c, L8a, L9b, L9c unchanged. Ids are L<iteration><letter>. Ranked by expected information value.

### L11a: Budget-scale matched rollout replay in the agreement-free regime (CPU only)

**Hypothesis:** L1a found greedy_online only 1.010 to 1.013 over the best charged recal rule at 25 and 50 percent on the corrupted label (3B and 7B), against 1.037 to 1.072 in L1b. The two differ in feature regime (agreement-free vs group-level) and in absolute budget (about 2,970 labels at 25 percent in L1a vs about 560 in L1b, so a 400-label calibration charge is 13 percent of the budget vs most of it). If budget scale is the driver, shrinking the L1a sim population to L1b's absolute budget should lift greedy_online / best charged rule back to 1.04 or more. Falsifier: the ratio stays at 1.02 or less at 25 percent on both rungs.

**Change vs existing scripts:** copy avrl/exp/l1a_sim_alloc.py to l11a_sim_alloc.py; add an argument that keeps a random subset of the sim prompts (all 8 rollouts of each, drawn within sim cells, subsample seeds 0 to 3) so that n_sim is about 2,250 rollouts; keep the fit set unchanged so static and static_matched are the same arms as in L1a; n_cal in 25, 50, 100, 200 (400 omitted: it exceeds the budget); use cell30 as the layout column for ts_online (cell100 would leave about 30 rollouts per bucket) and report greedy_online as the main arm. Regression gate: a subsample fraction of 1.0 reproduces runs/L1a/sim_*_cheap_cell100_s*.json with 0 per-seed differences, else void.

**Command shape:** UNVERIFIED, script not written: `cd /homes/rdubash/avrl_live/FULL_R/avrl && mkdir -p runs/L11a && /homes/rdubash/avrl_live/FULL_R/bin/gpu_run.sh runs/L11a 'bash exp/L11a.sh'` (CPU only; xargs -P 24 over rung x label in corrupted, judge_wrong x split seed 0 to 3 x subsample seed 0 to 3 x feature set cheap_r only, using runs/L1a/tab_*.parquet; the launcher must not write done.flag or exit_code). Time one job on a 2,000-row subsample first.

**Metric:** per rung and label: errors caught at 10, 25, 50 percent with absolute label counts; greedy_online / best charged recal rule (n_cal sweep shown), ts_online / greedy_online, greedy_online / static, greedy_online / best label-free threshold, pooled over subsample x split x policy seeds with the pooled-SD gap sqrt(a^2 + b^2) and a percentile bootstrap over subsample seeds.

**Pass threshold:** greedy_online / best charged recal rule >= 1.04 with the pooled gap above 2 SD at 25 percent on the corrupted label on both rungs, gate passing. Reading: budget scale, not feature regime, explains the L1a vs L1b gap; claim 2 holds at thesis-relevant small per-step budgets even without agreement.

**Kill threshold:** the ratio <= 1.02 at 25 percent on both rungs. Reading: the learner's edge over a fair charged multi-feature rule needs group-level features; the claim-2 formulation must be narrowed to that regime. Anything else is inconclusive.

**Expected runtime:** about 6 min on CPU (the 72 L1a jobs took 247 s; these jobs are smaller, about 64 of them).

**Informs:** claim 2 and the revised thesis directly (the edge is not larger where the proxy lacks a confidence signal, unless this reverses it). Caveats: one task, one policy, subsampling changes the population size and bucket occupancy, the fit set is not subsampled.

### L11b: Oracle-difficulty ceiling for prompt-space structure on GSM8K (CPU only)

**Hypothesis:** the 16 PCs add only +0.008 to +0.010 AUC because a 16-dimensional sentence embedding may not encode what drives the policy's false accepts, not because GSM8K prompts carry no structure. Gold-solution step count (the number of <<...>> calculation annotations in data/gsm8k_train.parquet and gsm8k_test.parquet, an oracle covariate the allocator would not have) is a direct difficulty measure. Falsifier of "the L1a kill is a representation kill": the oracle covariates add under 0.01 AUC to cheap_r on both rungs.

**Change vs existing scripts:** new avrl/exp/l11b_oracle.py: join the 3,000 prompts of runs/p0/feats.parquet to the GSM8K parquet files on question text (assert 3,000 matched, 0 duplicates), compute n_steps, count of numbers in the question and gold answer magnitude; import the CV, bootstrap and permutation-null code from exp/l1a_structure.py; add the oracle covariates to cheap_r on the L1a tables (corrupted, 3B and 7B, accept rule q); also OOF ridge R^2 of PCs and cell encodings for n_steps by pid-grouped folds; replay greedy_online with cheap_r + oracle covariates (a layout-free ceiling), K = 1, 8 policy seeds, 4 split seeds, 25 percent. Gate: cheap_r AUC reproduces 0.7045 (3B) and 0.7090 (7B-AWQ) to 3 decimals.

**Command shape:** UNVERIFIED, script not written: the L1a launcher pattern, `bin/gpu_run.sh runs/L11b 'bash exp/L11b.sh'`, CPU only; the launcher must not write done.flag or exit_code.

**Metric:** per rung: OOF AUC of cheap_r, cheap_r + oracle covariates (delta with pid-bootstrap CI and permutation null), cheap_r + PCs, cheap_r + both; OOF R^2 of PCs for n_steps; greedy_online errors caught with and without the oracle covariates at 25 percent.

**Pass threshold:** oracle delta AUC >= 0.03 with the bootstrap CI lower end >= 0.015 on at least one rung and PC OOF R^2 for n_steps < 0.2. Reading: prompt-level structure exists but the 16 PCs do not carry it; the L1a kill is a representation kill, L1c's prior rises and an override request with a text-reading scorer is justified.

**Kill threshold:** oracle delta AUC < 0.01 on both rungs. Reading: even an oracle difficulty covariate adds nothing to the rollout features; there is no prompt-level structure beyond the judge and rollout features on GSM8K, claim 3 is closed on this task for any representation, and the L1c override should not be requested. Anything else is inconclusive.

**Expected runtime:** about 10 min on CPU (structure stage 13 s in L1a plus a short replay).

**Informs:** claim 3, and whether Rustom should be asked to override the E04 stop rule for L1c. Caveats: n_steps may already be mediated by judge_logit and token counts in cheap_r; the oracle is a ceiling, not an available feature.

### L11c: Sibling-aware fixed rule versus the learner on the pid layout (CPU only)

**Hypothesis:** the pid-layout edge in L1a (ts_online / greedy_online 1.09 to 1.16 on the corrupted label, z 3.4 to 6.7) is prompt-level hardness, not identical-answer duplication: only 24.9 percent (3B) and 26.4 percent (7B) of corrupted rollouts share (pid, answer) with a sibling, labels are constant within (pid, answer) (0 inconsistent of about 2,400 groups), and a prompt with any corrupted rollout has 2.3 to 2.5 of them on average against 6.2 to 6.35 accepted rollouts per prompt overall. A hand-written rule with no learning should then capture most of the edge. Falsifier: the rule stays at 0.93 or less of ts_online (pid).

**Change vs existing scripts:** copy avrl/exp/l1a_sim_alloc.py to l11c_sim_alloc.py; add policy sibling_rule: rank unverified accepted rollouts by the cheap_r logistic fitted on the fit set (as static_matched), and after each verified corrupted rollout move that prompt's unverified accepted siblings to the front (a fixed rule, no online learning); add sibling_dedup: collapse identical (pid, answer) rollouts into one verifiable unit weighted by multiplicity. Same budgets, seeds, rungs and labels as L1a, pid and cell100 layouts. Gate: the existing arms reproduce runs/L1a/sim_*_pid_s0.json with 0 per-seed differences.

**Command shape:** UNVERIFIED, script not written: the L1a launcher pattern through `bin/gpu_run.sh runs/L11c 'bash exp/L11c.sh'`, CPU only; the launcher must not write done.flag or exit_code.

**Metric:** errors caught at 10, 25, 50 percent: sibling_rule and sibling_dedup versus ts_online (pid), greedy_online, static and the charged recal rule, with pooled-SD gaps.

**Pass threshold (the edge is a fixed rule):** sibling_rule >= 0.97 x ts_online (pid) at 25 percent on both rungs, gate passing. Reading: the pid edge is a rule available through group aggregates and supports neither claim.

**Kill threshold (the edge needs learning):** sibling_rule <= 0.93 x ts_online (pid) at 25 percent on both rungs. Reading: the learner finds something a simple rule does not; a claim-2 mechanism for a separate test. Anything else is inconclusive.

**Expected runtime:** about 8 min on CPU.

**Informs:** claim 2 (a fixed rule matching the learner where the learner has its clearest edge) and claim 3 (sibling structure is not prompt-space structure). Caveats: the pid layout has about 1,650 to 2,250 buckets, so ts_online is close to a per-prompt bandit; one task and one policy.

### Iteration 12 additions to 'Proposed by the loop' (after L11a)

Added 2026-10-19 after L11a (inconclusive; see avrl/RESEARCH_LOG.md, entry "L11a 2026-10-19 inconclusive"). Results re-derived from avrl/runs/L11a/ with separate code: gate passed (172 policy-budget cells, 0 differences vs runs/L1a); on the corrupted label at 25 percent and n_sim 2,250 (B = 562) greedy_online / best charged recal rule is 1.061 [1.034, 1.092] (3B) and 1.071 [1.034, 1.112] (7B-AWQ), so the ratio half of the pass rule holds on both rungs and the pooled-SD z half fails (0.49, 0.50; that statistic includes sub-population spread shared by both arms). Dose-response against B: 1.09 to 1.19 at B of 225 to 280, 1.06 to 1.07 at 562, 1.02 to 1.04 at 1,125, about 1.00 to 1.02 at B of 3,000. judge_wrong does not replicate (0.92 to 1.04, mostly below 1). ts_online equals greedy_online (0.982 to 1.023, 24 corrupted cells) and the free-label static rule is at or above the learner everywhere. Status of earlier items: L11a done; L11b and L11c unchanged (L11b still decides whether an E04 override for L1c is worth asking); L10a, L10b, L10c, L8a, L9b, L9c unchanged; the GPU items (L1c, L2c, L4c, L5c, L7c, L10a) stay blocked on the E04 stop rule. Ids are L<iteration><letter>. Ranked by expected information value: L12c, L12b, L12a.

### L12c: Is the learner's edge over the charged rule pure calibration-cost accounting? (CPU only)

**Hypothesis:** in L11a the persistent learner beats the best charged recal rule by 6 to 7 percent at B = 562 on the corrupted label, the edge falls to about 0 at B of 3,000, and a free-label static rule sits 5 to 6 percent above the learner at 25 percent budget (greedy / static 0.93 to 0.97). That pattern is what a pure cost-accounting effect predicts: the fixed rule spends n_cal of its B labels on random calibration picks (about 28 percent hit rate) instead of top-ranked picks, and the cost shrinks relative to B as B grows. If so, the same logistic rule fitted on n_cal labels that are NOT charged (drawn from the fit half) is at or above greedy_online at every B. Falsifier of the accounting reading: greedy_online / best uncharged size-matched rule stays at 1.03 or more at 25 percent and n_sim 2,250 on both rungs, i.e. the learner uses labels better than a fixed rule given the same number of labels for free.

**Change vs existing scripts:** copy avrl/exp/l11a_sim_alloc.py to l12c_sim_alloc.py; add policies recal_free_<n> for n in 25, 50, 100, 200, 400 (no budget filter needed, labels are free): draw n rows from the fit half with a seeded rng per policy seed, fit the same class-balanced logistic on the same cheap_r features and regularisation as recal_charged_<n>, rank the whole sim population, spend the full budget B on picks (no labels subtracted, none counted as hits). Nothing else changes. Gate: with the new policies skipped, the existing arms reproduce runs/L11a/sim_*_n2250_sp0_sub0.json and the target-0 split-0 files with 0 per-seed differences, else void. Grid: corrupted label only, rungs 3B and 7B-AWQ, n_sim 1,125 / 2,250 / 4,500 x split 0 to 3 x subsample 0 to 3 plus the full-scale reference (target 0, split 0 to 3), cell30, K = 1, 8 policy seeds, budgets 10 / 25 / 50 percent (about 104 jobs).

**Command shape:** UNVERIFIED, script not written: `cd /homes/rdubash/avrl_live/FULL_R/avrl && mkdir -p runs/L12c && /homes/rdubash/avrl_live/FULL_R/bin/gpu_run.sh runs/L12c 'bash exp/L12c.sh'` (CPU only; the L11a.sh pattern: gate, time one job, xargs -P 24, summarize; the launcher must not write done.flag or exit_code).

**Metric:** per rung and cell: greedy_online / best recal_free_n (n chosen on policy seeds 0 to 3, reported on 4 to 7, hindsight-best n shown), greedy_online / best recal_charged (reproduces L11a), the charge cost (recal_free_n minus recal_charged_n at the same n, in hits and percent), and the share of greedy_online's edge over the charged rule that the charge cost explains, (free - charged) / (greedy - charged); percentile bootstrap CIs over the 16 split x subsample clusters, plus the cluster-level paired t.

**Pass threshold (the edge is calibration-cost accounting):** at 25 percent budget and n_sim 2,250 on the corrupted label, greedy_online / best recal_free_n <= 1.00 with the bootstrap CI upper end <= 1.02, and the charge cost explains at least 80 percent of greedy_online's edge over the charged rule, on both rungs, gate passing. Reading: claim 2 must be restated as 'a persistent learner is the cheapest way to pay for calibration labels', the 'any fixed rule' wording is dropped, and the Phase 1 comparison is reported at equal total verifier calls including calibration.

**Kill threshold (the edge is more than accounting):** greedy_online / best recal_free_n >= 1.03 with the CI lower end above 1 on both rungs. Reading: given the same number of labels for free, the learner still wins, so there is a learning component beyond cost accounting. Anything else is inconclusive.

**Expected runtime:** about 5 min on CPU (the 208 L11a jobs took 171 s).

**Informs:** claim 2 directly (what the edge is) and the Phase 1 metric (verifier calls including calibration). Bracket from the L11a numbers: recal_free_100 at B = 562 must lie between the charged rule (250.8) and the free-label static ceiling (280.1) on 3B, against greedy_online 265.7, so the ratio could land anywhere from about 0.95 to 1.06 and the outcome is open. Caveats: one population, one task, the free rule is not available in deployment (it is a decomposition arm, not a competitor).

### L12b: Boundary-distance feature on the judge_wrong label (CPU only)

**Hypothesis:** on judge_wrong (the judge's accept decision is wrong, over all 24,000 rollouts) L11a found the learner at 0.92 to 1.04 of the charged recal rule (mostly below 1, all four splits below 1 at 25 percent and 2,250 rows on both rungs) and 0.91 to 0.95 of a free single-feature threshold (thr_judge_low on 3B, thr_boundary on 7B-AWQ). The label is non-monotone in judge_logit: errors cluster near the accept boundary, which a score linear in judge_logit cannot express, while thr_boundary can. If the judge_wrong negative is a representation limit of the linear learner, giving both learned and charged arms the distance-to-boundary feature restores the edge. On the corrupted label the population is accepted rollouts only, where the distance is monotone in judge_p, so the same feature should change nothing (control). Falsifier: with the feature greedy_online / best charged recal rule stays at 1.00 or below on judge_wrong at 25 percent on both rungs.

**Change vs existing scripts:** copy avrl/exp/l11a_sim_alloc.py to l12b_sim_alloc.py; add an argument that appends the existing table column `boundary` and log1p(boundary) to the cheap_r feature list for every learned and fitted arm (greedy_online, ts_online, static, static_matched, recal_charged_n), identical for both; no other change. Gate: with the argument off, the script reproduces runs/L11a/sim_*_n2250_sp0_sub0.json with 0 per-seed differences. Grid: judge_wrong (primary) and corrupted (control), rungs 3B and 7B-AWQ, n_sim 2,250 and full (target 0), split 0 to 3 x subsample 0 to 3 at 2,250, cell30, K = 1, 8 policy seeds, budgets 10 / 25 / 50 percent (about 80 jobs).

**Command shape:** UNVERIFIED, script not written: `cd /homes/rdubash/avrl_live/FULL_R/avrl && mkdir -p runs/L12b && /homes/rdubash/avrl_live/FULL_R/bin/gpu_run.sh runs/L12b 'bash exp/L12b.sh'` (CPU only; L11a.sh pattern; the launcher must not write done.flag or exit_code).

**Metric:** per rung, label and budget: greedy_online(+boundary) / best charged recal(+boundary) (n_cal chosen on seeds 0 to 3, reported on 4 to 7), greedy_online(+boundary) / greedy_online(L11a), greedy_online(+boundary) / best label-free threshold, and the OOF AUC of cheap_r with and without the boundary features on the judge_wrong table (pid-grouped folds); bootstrap CIs over the 16 clusters and split-level paired t.

**Pass threshold (the negative is a representation limit):** greedy_online(+boundary) / best charged recal(+boundary) >= 1.03 with the CI lower end above 1 at 25 percent on judge_wrong on both rungs, and the corrupted-label control within +-0.02 of L11a, gate passing. Reading: the label dependence in L11a is a feature-representation limit; claim 2 holds on judge_wrong once the allocator can express the boundary, and the Phase 1 feature set must include boundary distance.

**Kill threshold (the negative is real):** the ratio <= 1.00 on judge_wrong at 25 percent on both rungs. Reading: the charged rule's lead on judge_wrong is not a representation artifact; claim 2 is specific to the corrupted-reward label and the judge-error framing is dropped for Phase 1. Anything else is inconclusive.

**Expected runtime:** about 5 min on CPU.

**Informs:** the scope of claim 2 (which label the edge lives on) and the Phase 1 feature list. Outcome open: thr_boundary already beats greedy_online on 7B-AWQ judge_wrong, which favours the pass, but the charged rule also loses to the free thresholds there, which suggests label noise rather than representation. Caveats: `boundary` is derived from judge_p and the q-rule cut (a disclosed gold-derived scalar, the overall rollout accuracy); one population.

### L12a: Disjoint-block paired re-test of the L11a primary cell (CPU only)

**Hypothesis:** L11a's INCONCLUSIVE verdict comes from the pre-registered pooled-SD z (0.49, 0.50), which cannot exceed 2 at a ratio of 1.06 because it counts sub-population spread shared by both arms; the post hoc paired statistics (bootstrap CI above 1 on both rungs, split-level paired t 3.73 and 2.93 at df 3) are not clean because the 4 subsamples inside a split overlap. If the 6 to 7 percent edge is real, it reappears on non-overlapping sub-populations with a pre-registered paired criterion. Falsifier: the block-level mean ratio falls to 1.02 or below on either rung.

**Change vs existing scripts:** copy avrl/exp/l11a_sim_alloc.py to l12a_sim_alloc.py; replace the random-pid subsample with a block mode: shuffle the sim-half pids with a rng seeded from the split seed, cut them into consecutive blocks of about 2,250 rows (block index = argv[8]; about 5 blocks per split), assert that blocks are pairwise disjoint and each is within 1 percent of 2,250 rows. Gate: target 0 reproduces runs/L1a cell100 split 0 with 0 differences (the L11a gate), plus the disjointness assertion. Grid: corrupted label, 3B and 7B-AWQ, split 0 to 3, all blocks, cell30, cheap_r, K = 1, 8 policy seeds, budgets 10 / 25 / 50 percent (about 40 jobs). n_cal is fixed in advance at 100 (the value L11a selected in every primary cell) with n_cal 200 as the pre-registered sensitivity arm, so no selection step is needed.

**Command shape:** UNVERIFIED, script not written: `cd /homes/rdubash/avrl_live/FULL_R/avrl && mkdir -p runs/L12a && /homes/rdubash/avrl_live/FULL_R/bin/gpu_run.sh runs/L12a 'bash exp/L12a.sh'` (CPU only; L11a.sh pattern; the launcher must not write done.flag or exit_code).

**Metric:** per rung at 25 percent budget: mean greedy_online / recal_charged_100 over blocks (ratio of means), block-level paired t over about 20 blocks, two-stage bootstrap CI (splits, then blocks within a split), per-split ratios, and the same at 10 and 50 percent and for n_cal 200.

**Pass threshold:** mean ratio >= 1.04 with the two-stage bootstrap CI lower end above 1.00 and the block-level paired t above 3 on BOTH rungs, gate passing. Reading: the L11a edge is real on non-overlapping sub-populations; claim 2 is written as 'a 4 to 7 percent edge over a charged multi-feature rule at B of about 560 on the corrupted label'.

**Kill threshold:** mean ratio <= 1.02 on either rung. Reading: the L11a edge was driven by overlapping sub-populations or by a few splits; the 6 to 7 percent is not a result. Anything else is inconclusive.

**Expected runtime:** about 3 min on CPU.

**Informs:** the formal status of claim 2 at the L1b budget scale (it repairs the one defect in L11a's verdict rule). Lower information value than L12c and L12b because the post hoc numbers predict a pass. Caveats: blocks share the split's fit set, so splits are the real independent unit; one population.

### Iteration 13 additions to 'Proposed by the loop' (after L12c)

Added 2026-10-19 after L12c (inconclusive; see avrl/RESEARCH_LOG.md, entry "L12c 2026-10-19 inconclusive"). Results re-derived from avrl/runs/L12c/ with separate code: gate passed (104 files, 4,360 pre-existing policy-budget cells, 0 per-seed differences vs runs/L11a). On the corrupted label at 25 percent budget and n_sim 2,250 (B = 562): greedy_online / best free recal rule (n 400, top of the grid) is 0.965 [0.951, 0.979] (3B) and 0.976 [0.958, 0.994] (7B-AWQ), split-level paired t (df 3) -5.23 (p 0.014) and -4.66 (p 0.019), so the learner is below a fixed rule given 400 free fit-half labels and the KILL line (>= 1.03) is far away. The charge cost at the charged rule's own n = 100 is 3 and 2 hits of a 15 to 16 hit edge, so the pre-run share-explained clause (>= 0.80) fails at the headline definition (-0.05 and 0.11), but it is knife-edge in n (0.80 to 1.25 at n = 200, 1.4 to 1.6 best-vs-best) because the charged curve is flat from n = 100 to 200. Reading: the edge over the charged rule is a label-supply effect, not a per-label tax; the learner is worth about 170 to 210 free labels at B = 562 (30 to 37 percent of B) and reaches parity with the 400-label rule at B of about 1,100 or more. Claim 2 as worded ('any fixed rule') is not supported; the narrow form (ahead of rules that pay for calibration, 6 to 7 percent at B = 562) is suggestive. L12a and L12b stay in the backlog; L12a drops in value because the charged-rule edge is no longer the headline.

### L13a: Does online learning add anything once labels exist? Warm-started learner against the free fixed rule (CPU only)

**Hypothesis:** L12c showed that a fixed rule given 200 free fit-half labels ties the cold-start learner and one given 400 beats it, but the free rule never learns, so L12c cannot say whether the persistent learner has a learning component beyond label supply. If online refitting on the learner's own stream adds information (coverage of the decision region, labels concentrated where the top-B cut is decided), a learner warm-started with the same n free labels beats the fixed rule fitted once on them. If the learner's edge is only that its picks double as labels, warm start gives no gain over the fixed rule at the same n. Falsifier of the 'label supply only' reading: warm_online_200 / recal_free_200 >= 1.03 with the CI lower end above 1 on both rungs at 25 percent budget and n_sim 2,250.

**Change vs existing scripts:** copy avrl/exp/l12c_sim_alloc.py to l13a_sim_alloc.py; add arms warm_online_<n> for n in 100, 200, 400: the greedy online loop (run_online with greedy = True) whose initial score is the recal_free_<n> logistic's predict_proba on the sim population (same fit-half draw, rng seeded [policy seed, 12], so warm_online_n is paired with recal_free_n per policy seed) instead of random scores, and whose refit every R = max(B // 8, 40) labels uses the n fit-half labels pooled with all seen sim labels (unweighted, same StandardScaler + balanced LogisticRegression). Hits are counted on picks only; the n free labels count nothing. Gates: (1) all existing arms reproduce runs/L12c/sim_*_n2250_sp0_sub0.json and the target-0 split-0 files with 0 per-seed differences; (2) identity check: with refits disabled (R larger than B) warm_online_n equals recal_free_n per policy seed on one file, 0 differences; else the run is VOID. Grid: corrupted label, rungs 3B and 7B-AWQ, n_sim 2,250 (primary, split 0 to 3 x subsample 0 to 3) plus 1,125 / 4,500 / full (descriptive), cell30, cheap_r, K = 1, 8 policy seeds, budgets 10 / 25 / 50 percent (about 104 jobs).

**Command shape:** UNVERIFIED, script not written: `cd /homes/rdubash/avrl_live/FULL_R/avrl && mkdir -p runs/L13a && /homes/rdubash/avrl_live/FULL_R/bin/gpu_run.sh runs/L13a 'bash exp/L13a.sh'` (CPU only; L12c.sh pattern: gate, time one job, xargs -P 24, summarize; the launcher must not write done.flag or exit_code).

**Metric:** per rung and cell: warm_online_n / recal_free_n at n = 200 (pre-registered primary, no selection step), n = 100 and 400 secondary; warm_online_200 / recal_free_400 (does 200 free labels plus the stream beat 400 free labels); warm_online_n / greedy_online (cold); warm_online_n / static; percentile bootstrap CI over the 16 clusters and the split-level paired t at df 3.

**Pass threshold (a learning component beyond label supply):** at 25 percent and n_sim 2,250, warm_online_200 / recal_free_200 >= 1.03 with the bootstrap CI lower end above 1 on BOTH rungs, gates passing. Reading: persistent online learning adds value on top of available labels; claim 2 is restated as 'a warm-started persistent learner beats a fixed rule given the same labels', and the Phase 1 allocator should warm-start from earlier verified labels.

**Kill threshold (label supply only):** warm_online_200 / recal_free_200 <= 1.01 with the CI upper end <= 1.02 on BOTH rungs. Reading: once labels exist a fixed rule is equivalent; claim 2 is restated as 'the persistent learner is the cheapest source of calibration labels when none exist', and the Phase 1 case rests on label scarcity. Anything else is inconclusive.

**Expected runtime:** about 6 min on CPU (three extra online arms on top of the 80 s L12c grid).

**Informs:** claim 2's mechanism directly and the Phase 1 design (warm start or cold start). Highest value of the three because L12c left exactly this question open. Caveats: the free labels are same-population fit-half rows (an optimistic prior-data case); one population, one task; the pooled refit weights free and seen labels equally, which is one of several reasonable choices.

### L13b: Per-label value of the learner's own label stream (CPU only)

**Hypothesis:** L12c puts the cold-start learner at about 200 random free labels at B = 562, although it ends with 562 labels. Either its stream labels are less informative per label than random labels (they sit near the top of the ranking and are class-imbalanced), or only the early labels matter. Separately, L12c's equal-n charge cost mixes the charge with a fit-half vs sim-population shift; a sim-population uncharged control isolates it. Falsifier of 'stream labels are no better per label than random labels': precision at k of a rule frozen after n stream labels is at least 3 percent above that of a rule fitted on n random free labels at n = 200.

**Change vs existing scripts:** copy avrl/exp/l12c_sim_alloc.py to l13b_sim_alloc.py; add per policy seed (a) stream_frozen_<n> for n in 50, 100, 200, 400: run greedy_online with its normal refit schedule plus a forced refit at exactly n labels, freeze that model, record the set S_n of the n sim rows it labelled; (b) simfree_<n>: n random sim rows, fit the same logistic (uncharged, a population-shift control for recal_free_n); (c) the existing recal_free_<n> fit. Score all three models on the same evaluation set E = sim minus (S_n union the simfree_<n> rows) and record precision at k (k = B minus n top-ranked rows of E, hits among them) and the AUC on E. No hits are counted for the n calibration labels of any arm, so the pick-hit accounting asymmetry between stream and random labels is removed. Gate: existing arms reproduce runs/L12c sim files (n2250 sp0 sub0, target 0 split 0) with 0 per-seed differences, else VOID. Grid: corrupted label, rungs 3B and 7B-AWQ, n_sim 2,250 x split 0 to 3 x subsample 0 to 3 plus the full reference (target 0, split 0 to 3), 25 percent budget only, cell30, cheap_r, K = 1, 8 policy seeds (about 40 jobs).

**Command shape:** UNVERIFIED, script not written: `cd /homes/rdubash/avrl_live/FULL_R/avrl && mkdir -p runs/L13b && /homes/rdubash/avrl_live/FULL_R/bin/gpu_run.sh runs/L13b 'bash exp/L13b.sh'` (CPU only; L12c.sh pattern; the launcher must not write done.flag or exit_code).

**Metric:** per rung: precision at k and AUC of stream_frozen_n / recal_free_n at n = 50, 100, 200, 400 (n = 200 pre-registered primary); the number of random free labels whose precision equals the stream's n labels (interpolated on the recal_free curve); simfree_n / recal_free_n (population-shift check, should be within 1 percent); bootstrap CI over the 16 clusters.

**Pass threshold (stream labels at least as informative as random labels):** stream_frozen_200 / recal_free_200 precision at k >= 1.03 with the CI lower end above 1 on BOTH rungs, simfree within 1 percent of recal_free, gate passing. Reading: the learner's own stream is a better calibration set than random labels; the 'worth about 200 labels' figure understates its quality because it is diluted by late refits, and the Phase 1 freeze point can be early.

**Kill threshold (stream labels are worth less per label):** stream_frozen_200 / recal_free_200 <= 0.97 with the CI upper end below 1 on BOTH rungs. Reading: the learner wins because its labels are a free by-product of picking, not because they are better labels; the Phase 1 freeze point should be set later (L6a's freeze at about 500 labels is the safer default). If simfree differs from recal_free by more than 1 percent, L12c's equal-n charge cost is confounded by population shift and the share-explained numbers must be recomputed on simfree. Anything else is inconclusive.

**Expected runtime:** about 5 min on CPU.

**Informs:** the L12c mechanism (label supply vs label quality), the Phase 1 freeze-point knob, and the one caveat of L12c that the spec made unavoidable (fit-half vs sim population). Also resolves the tension between L6a ('which labels, not how many') and L12c (200 random free labels tie the learner). Caveats: precision at k on E is a ranking-quality proxy, not the budgeted hit count; one population.

### L13c: Refit-period and regularisation headroom of the cold-start learner (CPU only)

**Hypothesis:** greedy_online refits every R = max(B // 8, 40) labels with the default C = 1; R was set in earlier iterations and never tuned in the agreement-free regime. At B = 562 the learner refits only 7 times, so late-run picks use a stale model. If a faster refit or different regularisation closes the 2.4 to 3.6 percent gap to the 400-label free rule, the L12c comparison at B = 562 is a tuning artifact and a restricted 'any fixed rule' claim can survive at that scale. Falsifier: no (R, C) setting brings greedy_online within 1 percent of its control, and none reaches the best free rule.

**Change vs existing scripts:** copy avrl/exp/l12c_sim_alloc.py to l13c_sim_alloc.py; add arms greedy_R<k> with R = max(B // k, 20) for k in 4, 16, 32 (k = 8 is the control and must equal greedy_online) and greedy_C<c> with the logistic C in 0.1 and 10 at R = max(B // 8, 40) (C = 1 is the control). Nothing else changes; recal_free and recal_charged arms stay for the same-cell comparison. Gate: the control arms reproduce greedy_online per seed on runs/L12c sim files (n2250 sp0 sub0, target 0 split 0), 0 differences, else VOID. Grid: as L12c (corrupted label, 3B and 7B-AWQ, n_sim 2,250 x 16 clusters primary; 1,125 / 4,500 / full descriptive; 8 policy seeds; budgets 10 / 25 / 50 percent; about 104 jobs).

**Command shape:** UNVERIFIED, script not written: `cd /homes/rdubash/avrl_live/FULL_R/avrl && mkdir -p runs/L13c && /homes/rdubash/avrl_live/FULL_R/bin/gpu_run.sh runs/L13c 'bash exp/L13c.sh'` (CPU only; L12c.sh pattern; the launcher must not write done.flag or exit_code).

**Metric:** per rung and cell: best tuned greedy (the (R, C) setting chosen on policy seeds 0 to 3, reported on 4 to 7) / greedy_online control, / best recal_free, / static; the dose-response of the ratio to the control in R and C; bootstrap CI over the 16 clusters and the split-level paired t at df 3.

**Pass threshold (the learner has tuning headroom):** at 25 percent and n_sim 2,250, best tuned greedy / best recal_free >= 1.00 on BOTH rungs, gate passing. Reading: the L12c deficit to the 400-label rule is a tuning artifact; report claim 2 with a tuned learner and keep a restricted 'any fixed rule with at most a few hundred labels' form at B of about 500.

**Kill threshold (no headroom):** best tuned greedy / greedy_online control <= 1.01 on BOTH rungs. Reading: the learner is capped near 200 to 300 free-label equivalents at this scale regardless of R and C, the L12c comparison stands as written, and the Phase 1 default (R = B / 8) is adequate. Anything else is inconclusive.

**Expected runtime:** about 8 min on CPU (the extra online arms refit more often).

**Informs:** whether L12c's ratio of 0.965 and 0.976 is a property of the learner design or of one untuned setting, and the Phase 1 refit schedule. Medium value, cheap, uses the L12c arms unchanged. Caveats: tuning on the same populations as the evaluation (policy seeds differ, populations do not); two knobs only, no change to the feature set or the pick rule.

### Iteration 14 additions to 'Proposed by the loop' (after L11b)

Added 2026-10-19 after L11b (inconclusive; see avrl/RESEARCH_LOG.md, entry "L11b 2026-10-19 inconclusive"). Results re-derived from avrl/runs/L11b/ with separate code (avrl/exp/l11b_critic.py, runs/L11b/critic.log): gate passed (cheap_r OOF AUC 0.7045 and 0.7090; no-oracle replay vs runs/L1a, 390 policy-budget cells, 0 differences). On the corrupted label the oracle gold step count (number of <<...>> annotations, join 3,000 of 3,000) adds +0.0155 [0.0098, 0.0210] (3B) and +0.0171 [0.0113, 0.0235] (7B-AWQ) OOF AUC to cheap_r, far above the permutation null (max +0.0024 and +0.0014), against +0.0097 and +0.0084 for the 16 PCs; question number count and gold magnitude add nothing alone; PCs and the oracle are nearly additive (+0.0242 and +0.0249 combined, descriptive); the 16 PCs explain R^2 0.046 of n_steps (log_qlen alone 0.152, already in cheap_r) and cell encodings 0.012 or less. The pre-registered PASS (>= 0.03) and KILL (< 0.01) both fail, so claim 3 is neither closed nor reopened on GSM8K. Leak-free replay (greedy_online, 25 percent): +5.8 to +8.0 percent errors caught, ts_online / greedy_online 0.989 to 1.011 (layout adds nothing), fixed rules gain as much as the learner. The HGB arm with gold magnitude is label leakage (HGB delta +0.25); the HGB arm without it (+0.0234, +0.0343) sits on a weak baseline (HGB cheap_r 0.686 and 0.682 against logistic 0.7045 and 0.7090), and its 7B-AWQ value meets the numeric PASS thresholds, which is the one interpretive fork L14a settles. Status of earlier items: L11b done (inconclusive); L1c stays blocked on the E04 stop rule and an override request is not justified by L11b (gated now on L14a and L14b); L13a, L13b, L13c, L12a, L12b, L11c, L10a, L10b, L10c unchanged, and L13a remains the highest-value claim-2 item. Ids are L<iteration><letter>. Ranked by expected information value.

### L14a: Fitted nonlinear baseline for the oracle step-count delta (CPU only)

**Hypothesis:** L11b's HistGradientBoosting deltas for n_steps + n_numbers_q (+0.0234 on 3B, +0.0343 on 7B-AWQ, null max +0.0046 and +0.0058) are measured against an HGB cheap_r baseline (0.6864 and 0.6821) that is 0.02 to 0.03 AUC below logistic cheap_r (0.7045 and 0.7090); the HGB had 100 iterations at learning rate 0.1, no regularisation, no tuning. If the delta is a weak-baseline artifact, a nonlinear model fitted to match the logistic baseline shows a delta near the logistic +0.015 to +0.017. If step count interacts with the other covariates (for example it matters mainly near the judge's accept boundary), the delta over a fitted nonlinear baseline stays at 0.03 or more. Falsifier of the weak-baseline reading: on at least one rung the fitted-baseline delta is 0.03 or more with the baseline within 0.005 AUC of logistic cheap_r.

**Change vs existing scripts:** new avrl/exp/l14a_nonlinear.py importing the CV, bootstrap and permutation-null helpers from exp/l1a_structure.py and reading runs/L11b/tab_{3B,7B}_q_corrupted.parquet (read only; outputs under runs/L14a). Same pid-grouped 5 folds x 3 CV seeds, 200-resample pid bootstrap (seed 12345), 20-draw donor-pid null. Two fitted nonlinear families, each fitted on cheap_r alone and on cheap_r + n_steps + n_numbers_q (no gold magnitude), with hyperparameters selected on the cheap_r fit only (inner pid-grouped 3-fold in the training fold) and reused for the oracle fit, so tuning is not credited to the covariate: (i) LOGIT2, balanced logistic on standardized cheap_r plus all pairwise products and squares, C in 0.01, 0.1, 1; (ii) HGB_T, HistGradientBoosting with max_depth 3, learning_rate 0.05, l2_regularization 1.0, min_samples_leaf 50, max_iter in 100, 200, 400, 800. Gate: logistic cheap_r AUC 0.7045 and 0.7090 and the logistic orc_ng delta +0.0150 and +0.0167 reproduce to 3 decimals through the L11b code path, else VOID.

**Command shape:** UNVERIFIED, script not written: `cd /homes/rdubash/avrl_live/FULL_R/avrl && mkdir -p runs/L14a && /homes/rdubash/avrl_live/FULL_R/bin/gpu_run.sh runs/L14a 'bash exp/L14a.sh'` (CPU only; L11b.sh pattern with one process per rung; the launcher must not write done.flag or exit_code).

**Metric:** per rung: OOF AUC of each fitted nonlinear cheap_r and its gap to logistic cheap_r; delta of +orc_ng over its own cheap_r baseline with bootstrap CI and null max; the best absolute AUC across all families for cheap_r + orc_ng and for cheap_r alone, and the delta between those two bests (a model-selection-robust version).

**Pass threshold (the nonlinear reading was real):** on at least one rung, with the fitted baseline within 0.005 of logistic cheap_r, delta >= 0.03 with the bootstrap CI lower end >= 0.015 and above the null max, gate passing. Reading: step count interacts with the other covariates, the L11b logistic delta understated it, the L11b fork resolves to PASS, claim 3 is reopened as an interaction structure, the L1c override question becomes live, and the Phase 1 allocator would need a nonlinear scorer to use it (the replay learner is logistic).

**Kill threshold (the HGB delta was a baseline artifact):** delta <= 0.02 on both rungs with each fitted baseline within 0.005 of logistic cheap_r. Reading: the step-count ceiling is a small linear +0.015 to +0.017 AUC, L11b closes as inconclusive-small, claim 3 stays not supported, no override request. If no fitted baseline gets within 0.005 of logistic, record as uninformative. Anything else is inconclusive.

**Expected runtime:** about 6 min on CPU (the whole L11b structure stage took 41 s; inner CV and the interaction features add roughly an order of magnitude).

**Informs:** the verdict fork of L11b and with it claim 3 and the L1c override decision; cheapest item that can flip a verdict. Caveats: one covariate family, one label, the HGB and interaction grids are small; a nonlinear scorer is not part of the current allocator.

### L14b: How much of the step-count ceiling can a text-only reader recover? (CPU only)

**Hypothesis:** L11b's oracle n_steps adds +0.0155 and +0.0171 AUC, but the 16 PCs explain only R^2 0.046 of n_steps and the PC and oracle gains are nearly additive (combined +0.0242 and +0.0249), so the PCs' +0.0097 and +0.0084 come from something other than step count. A text-only predictor with more capacity than 16 PCs of one embedding (the full 384-d all-MiniLM-L6-v2 vector, TF-IDF unigrams and bigrams, counts of arithmetic cue words such as each, per, twice, half, percent, total, more than, plus log_qlen and n_numbers_q) may recover most of the step-count gain. Falsifier of 'light text readers cannot recover the ceiling': cheap_r + predicted n_steps adds 0.012 AUC or more (about 80 percent of the oracle step gain) on both rungs.

**Change vs existing scripts:** new avrl/exp/l14b_textsteps.py (outputs under runs/L14b; reads runs/L11b/tab_*.parquet, runs/L11b/oracle_pid.parquet, runs/p0/feats.parquet). Embeddings: all-MiniLM-L6-v2, the model p0_analyze.py used (UNVERIFIED that its weights are readable offline in the CPU environment; if not, run TF-IDF + cue words only and report that the embedding arm was skipped); encode the 3,000 questions on CPU (about 1 to 2 min). Predictor: RidgeCV on [full embedding, 100-d TruncatedSVD of TF-IDF, cue-word counts, log_qlen, n_numbers_q] with target n_steps. Nesting: for each outer fold of the downstream pid-grouped AUC CV (the same fold ids as L11b), the predictor is trained on outer-training pids only; training rows get inner 5-fold pid-OOF predictions and test rows the prediction of the model fit on all training pids, so no pid contributes to its own feature. The predicted column is added to cheap_r (and to cheap_r + 16 PCs). Controls: the oracle n_steps arm must reproduce +0.0155 and +0.0171; a donor-pid permutation null on the predicted column (20 draws). Gate: cheap_r AUC 0.7045 and 0.7090 and the oracle step delta reproduce to 3 decimals, else VOID.

**Command shape:** UNVERIFIED, script not written: `cd /homes/rdubash/avrl_live/FULL_R/avrl && mkdir -p runs/L14b && /homes/rdubash/avrl_live/FULL_R/bin/gpu_run.sh runs/L14b 'bash exp/L14b.sh'` (CPU only, sentence-transformers from the active-audit .venv; the launcher must not write done.flag or exit_code).

**Metric:** per rung: OOF R^2 of the predicted n_steps (3,000 prompts); AUC delta of cheap_r + predicted n_steps over cheap_r (bootstrap CI, null max); share of the oracle delta recovered (with CI); delta over cheap_r + 16 PCs; a descriptive layout-free greedy_online replay at 25 percent with the predicted column (exp/l1a_sim_alloc.py unchanged).

**Pass threshold (a text reader recovers the ceiling):** delta over cheap_r >= 0.012 and at least 0.002 above the 16-PC delta, bootstrap CI lower end >= 0.006, on both rungs, gate passing. Reading: the step-count gradient is readable from the question; the GSM8K headroom for claim 3 is about +0.015 AUC and +6 to 8 percent errors caught at 25 percent; a text-reading scorer (L1c) may be worth an override request if that size matters for Phase 1, and the request should state the size.

**Kill threshold (not recoverable by light readers):** delta <= 0.005 on both rungs and R^2 < 0.2. Reading: step count is not recoverable from the question surface by embedding or lexical readers, claim 3 stays at the 16-PC level (+0.008 to +0.010), and the L1c override should not be requested on structure grounds. A heavier reader (LLM hidden states, a reward model) is exactly what L1c would test, so a kill lowers but does not close that case. Anything else is inconclusive.

**Expected runtime:** about 10 min on CPU.

**Informs:** whether the L1c override request is worth making and what size of claim-3 gain it could buy; the Phase 1 feature list (a cheap step-count proxy as a linear feature, never a layout). Caveats: step count is one covariate; the oracle bounds only that covariate; one task and one label.

### L14c: Does the learner's edge over charged rules grow with feature informativeness? (CPU only)

**Hypothesis:** in L11b's descriptive replay (full-scale sim populations, B about 2,900) greedy_online / best charged recal rule rises from 1.003 to 1.011 with cheap_r to 1.023 to 1.041 with the leak-free oracle covariates at 25 percent (1.001 to 1.049 -> 1.020 to 1.088 at 10 percent, no change at 50 percent), while the free-label static rule gains about as much as the learner. L11a attributed the L1b versus L1a gap to budget scale; L11b suggests feature informativeness matters too. If the edge over charged rules depends on the information in the feature set, the ratio should rise along cheap_r < +16 PCs < +orc_ng < +PCs +orc_ng at the L11a budget scale (n_sim 2,250, B = 562), where cheap_r gave 1.061 (3B) and 1.071 (7B-AWQ). Falsifier: the ratio stays within +-0.015 of the cheap_r value across the four sets.

**Change vs existing scripts:** run exp/l12c_sim_alloc.py unchanged (it carries recal_charged_n, recal_free_n and greedy_online; command-line arguments not re-read here, UNVERIFIED) on runs/L11b/tab_{3B,7B}_q_corrupted.parquet with four feature sets: F0 = cheap_r, F1 = cheap_r + pc0..pc15, F2 = cheap_r + n_steps + n_numbers_q (oracle), F3 = F1 + F2 (oracle). Grid as L12c: corrupted label, n_sim 2,250 x split 0 to 3 x subsample 0 to 3, cell30, K = 1, 8 policy seeds, budgets 10 / 25 / 50 percent, both rungs (about 128 jobs). Gate: F0 reproduces runs/L12c n_sim 2,250 files with 0 per-seed differences, else VOID.

**Command shape:** UNVERIFIED, script not written: `cd /homes/rdubash/avrl_live/FULL_R/avrl && mkdir -p runs/L14c && /homes/rdubash/avrl_live/FULL_R/bin/gpu_run.sh runs/L14c 'bash exp/L14c.sh'` (CPU only; L12c.sh pattern: gate, time one job, xargs -P 24, summarize; the launcher must not write done.flag or exit_code).

**Metric:** per rung and feature set: greedy_online / best charged recal (n_cal chosen on policy seeds 0 to 3, reported on 4 to 7, as in L11a and L12c) and greedy_online / best free recal (n = 400); the change in each ratio against F0 with a 16-cluster bootstrap CI and the split-level paired t (df 3), laid against each set's OOF AUC from L11b (0.7045, 0.7141, 0.7195, 0.7287 on 3B).

**Pass threshold (the edge grows with information):** at 25 percent, F2 minus F0 in greedy_online / best charged recal >= +0.025 with the bootstrap CI lower end above 0 on both rungs, gate passing. Reading: claim 2's size is a function of the feature set's information; Phase 1 should invest in verifier-side features before allocator tuning, and the F1 change says how much of the effect is deployable today.

**Kill threshold (label supply, not feature information):** F2 minus F0 <= +0.010 on both rungs. Reading: at B about 560 the edge is a label-supply effect (L12c), and the L11b full-scale rise is a large-B artifact. Anything else is inconclusive.

**Expected runtime:** about 8 min on CPU (L12c ran 104 jobs in 80 s of grid wall).

**Informs:** the mechanism behind claim 2's edge and the Phase 1 priorities; it also tells whether L11a's budget-scale explanation was incomplete. Caveats: F2 and F3 are oracle sets; splits share prompts, so the honest unit is 4 splits; one task and policy; the online learner exploits label-revealing columns that fixed rules do not (L11b), so no gold-derived column may enter any set (the gold magnitude column is excluded by construction).

### Iteration 15 additions to 'Proposed by the loop' (after L13a)

Added 2026-10-19 after L13a (pass; see avrl/RESEARCH_LOG.md, entry "L13a 2026-10-19 pass"). Results re-derived from avrl/runs/L13a/ raw sim JSONs with separate code: all three gates pass (gate 1 254 cells, gate 2 18 cells, post gate 6,440 cells, 0 per-seed differences). On the corrupted label at 25 percent budget and n_sim 2,250 (B = 562, 16 clusters x 8 seeds): warm_online_200 / recal_free_200 is 1.042 [1.032, 1.055] (3B) and 1.041 [1.030, 1.051] (7B-AWQ), 4 of 4 splits above 1.00 on both rungs, split-level paired t (df 3) 4.68 (p 0.018) and 15.74 (p 0.001); 24 of 24 rung x scale x budget cells above 1. The warm learner closes 75 to 85 percent of the gap between the 200-label fixed rule and the free-label static ceiling and reaches, but does not meaningfully pass, that ceiling (warm_200 / static 0.987 and 0.993). The critic's label-count control (not pre-registered) shows the gain equals what 600 to 762 free labels would buy (warm_200 / fixed rule with 762 free labels 0.993 and 1.011), so 'learning component' means 'beyond the initial label supply', not a coverage or concentration effect. Claim 2 now has a pre-registered surviving form (a warm-started persistent learner beats a fixed rule given the same initial labels); 'any fixed rule' remains false against rules with larger free label banks. Claim 3 untouched (cold ts_online / greedy_online 1.009 and 1.003). Status of earlier items: L13a done (pass); L13b is partly answered by the label-count control and its stream_frozen half overlaps L15b, keep it only if L15b is inconclusive; L13c, L14a, L14b, L14c, L12a, L12b, L11c, L10a, L10b, L10c unchanged; every GPU item stays blocked on the E04 stop rule and L13a gives no reason to request an override (the claim-2 question is answerable on CPU). Ids are L<iteration><letter>. Ranked by expected information value.

### L15a: Stale-label warm start (CPU only)

**Hypothesis:** L13a's 4 percent warm-start edge uses free labels that are same-policy and same-population (rows of a random quarter of the cell30 cells, disjoint prompts from the sim rows). In Phase 1 the earlier verified labels come from a previous policy checkpoint and are stale. A fixed rule fitted once on stale labels cannot repair itself; the learner refits on its own stream and can. If so, the learner's margin over the same-label fixed rule widens with staleness, which would be a learning component that is not a label-count effect (more stale free labels do not help the fixed rule). If instead stale labels poison the pooled refit, warm start falls below the cold learner and Phase 1 must default to cold start or weight labels by age. Falsifier of 'warm start is robust to staleness': at 25 percent budget, n_sim 2,250, with each free label flipped with probability 0.25, warm_stale_200 / greedy_online (cold) is below 0.97 with the CI upper end below 1 on either rung.

**Change vs existing scripts:** copy avrl/exp/l13a_sim_alloc.py to l15a_sim_alloc.py; add argv[10] STALE in {none, flip25, flip50, lowlen} applied to the n free labels only, identically for recal_free_n and warm_online_n so the two stay paired per policy seed: flip25 and flip50 flip each of the n free labels independently with probability 0.25 or 0.5 (rng seeded [policy seed, 15], applied to the labels, not the features); lowlen draws the n free rows from the half of the fit rows with the lowest log_qlen (a covariate-shifted prior). Only n = 200 is run. Add the mitigation arm warm_online_200_w25: stale labels carry sample_weight 0.25 in every refit (sample_weight passed to the logistic step of the pipeline), stream labels weight 1. STALE = none must reproduce runs/L13a for the n = 200 warm and free arms. Gates: (1) STALE = none reproduces runs/L13a n2250 files (warm_online_200, recal_free_200, greedy_online, static) with 0 per-seed differences, else VOID; (2) with refits disabled, warm_online_200 equals recal_free_200 per policy seed under every STALE variant, else VOID. Grid: corrupted label, rungs 3B and 7B-AWQ, n_sim 2,250 x split 0 to 3 x subsample 0 to 3, cell30, cheap_r, K = 1, 8 policy seeds, budgets 10 / 25 / 50 percent, 4 STALE variants (about 128 jobs; each job runs two or three arms).

**Command shape:** UNVERIFIED, script not written: `cd /homes/rdubash/avrl_live/FULL_R/avrl && mkdir -p runs/L15a && /homes/rdubash/avrl_live/FULL_R/bin/gpu_run.sh runs/L15a 'bash exp/L15a.sh'` (CPU only; L13a.sh pattern: gate 1, identity gate 2, time the largest job, xargs -P 24, post gate, summarize; the launcher must not write done.flag or exit_code).

**Metric:** per rung and STALE variant: warm_stale_200 / greedy_online (cold), warm_stale_200 / recal_free_200 under the same stale labels, warm_stale_200 / warm_online_200 (clean, from L13a), and warm_online_200_w25 / greedy_online; ratio of means with the 16-cluster percentile bootstrap and the split-level paired t at df 3. flip25 is the pre-registered primary; flip50 (prior destroyed) and lowlen are descriptive.

**Pass threshold (warm start is robust and the learner repairs staleness):** at 25 percent and n_sim 2,250 under flip25, on BOTH rungs, warm_stale_200 / greedy_online >= 1.00 and warm_stale_200 / recal_free_200 (stale) >= 1.06 with the bootstrap CI lower end above 1, gates passing. Reading: a learning component that is not a label-count effect exists (adaptation to stale labels), Phase 1 warm-starts from earlier verified labels by default and keeps refitting, and claim 2's surviving form gains a drift-robustness clause.

**Kill threshold (warm start is fragile):** warm_stale_200 / greedy_online <= 0.97 with the CI upper end below 1 on either rung under flip25. Reading: the 1.04 of L13a is a same-distribution artifact; the Phase 1 default becomes cold start, or warm start with label-age weighting if the w25 arm restores at least 0.99 of cold, and claim 2 is restated as 'the learner is the cheapest label source when none exist'. Anything else is inconclusive (flip25 only mildly attenuates a monotone score, so an inconclusive result shifts the weight to the flip50 and lowlen rows).

**Expected runtime:** about 6 min on CPU.

**Informs:** the Phase 1 default (warm or cold start, and whether stale labels need age weights), and the one place left where a learning component beyond label count could appear. Highest value of the three. Caveats: label flips and a question-length shift are crude proxies for policy drift; real drift changes the feature-to-error relation, which L8a would model; one population and one task; staleness is applied to the n free labels only.

### L15b: Drip-fed random-label control for the warm learner (CPU only)

**Hypothesis:** L13a's critic found warm_online_200 about equal to a fixed rule handed 600 to 762 free labels at once (0.993 [0.983, 1.002] on 3B, 1.011 [0.997, 1.024] on 7B-AWQ), but that rule gets all labels up front. A control that refits on the learner's own schedule using the n initial labels plus as many uncharged RANDOM labels as the learner has picks separates 'a pick is worth a random label' from 'picks are more informative (coverage of the decision region, labels concentrated near the top-B cut)' or 'less informative (top-ranked picks are class-skewed)'. Falsifier of 'picks are worth random labels one for one': warm_online_200 / drip_free_200 is outside [0.99, 1.01] at 25 percent budget and n_sim 2,250.

**Change vs existing scripts:** copy avrl/exp/l13a_sim_alloc.py to l15b_sim_alloc.py; add arm drip_free_<n> for n = 200: same initial score (the recal_free_n logistic), same refit schedule R = max(B // 8, 40), picks are the top-ranked unseen sim rows by the current score and count as hits, but each refit pools the n initial labels with the next m fit-half rows from the same seeded permutation (rf = default_rng([seed, 12]), rows n to n + m of rf.permutation(len(fit))), where m is the number of picks made so far, instead of the picks' labels. The drip labels are fit rows, so they never overlap the sim picks and are never counted as hits. Restrict to n_sim 2,250 (n + B at most 1,325 against a minimum n_fit of 2,663). Gates: (1) all existing arms reproduce runs/L13a n2250 files with 0 per-seed differences; (2) with refits off drip_free_200 equals recal_free_200 per seed; else VOID. Grid: corrupted label, rungs 3B and 7B-AWQ, n_sim 2,250 x split 0 to 3 x subsample 0 to 3, cell30, cheap_r, 8 policy seeds, budgets 10 / 25 / 50 percent (32 jobs).

**Command shape:** UNVERIFIED, script not written: `cd /homes/rdubash/avrl_live/FULL_R/avrl && mkdir -p runs/L15b && /homes/rdubash/avrl_live/FULL_R/bin/gpu_run.sh runs/L15b 'bash exp/L15b.sh'` (CPU only; L13a.sh pattern; the launcher must not write done.flag or exit_code).

**Metric:** per rung and budget: warm_online_200 / drip_free_200 and drip_free_200 / recal_free_200, 16-cluster percentile bootstrap CI, split-level paired t at df 3, plus the share of the warm gain over recal_free_200 that drip_free_200 reproduces.

**Pass threshold (picks are better than random labels):** at 25 percent, warm_online_200 / drip_free_200 >= 1.02 with the CI lower end above 1 on BOTH rungs, gates passing. Reading: a coverage or concentration effect exists beyond label count, the stream is a better calibration set than random labels, and the L13b freeze-point question is live.

**Kill threshold (picks are worth random labels one for one):** warm_online_200 / drip_free_200 within [0.99, 1.01] with the CI inside [0.98, 1.02] on BOTH rungs. Reading: no coverage effect; the learner's value is zero-charge label supply, Phase 1 could swap the stream for a label bank, and claim 2's final form is 'the persistent learner is the cheapest source of calibration labels'. A ratio of 0.98 or less with the CI upper end below 1 means picks are worse per label than random labels (class skew), which argues for an exploration share. Anything else is inconclusive.

**Expected runtime:** about 5 min on CPU.

**Informs:** the mechanism behind claim 2's surviving form; partly supersedes the stream_frozen half of L13b. Lower decision value than L15a because the Phase 1 allocator already uses its picks as labels. Caveats: drip labels are fit-half rows (a different layout-cell population from the sim rows), one population and task.

### L15c: Disjoint-prompt block replication of the L13a primary cell (CPU only)

**Hypothesis:** the 16 clusters at n_sim 2,250 come from 4 splits that reuse the same 3,000 prompts, so the honest independent unit is 4 splits (df 3), and the 3B primary ratio clears the 1.03 line by only 0.012 with one split at 1.025. A design with fully prompt-disjoint blocks gives more independent units. Falsifier of the L13a PASS: on prompt-disjoint blocks the pooled warm_online_200 / recal_free_200 is at or below 1.01 on either rung.

**Change vs existing scripts:** write exp/l15c_blocks.py that shuffles the 3,000 pids with a fixed seed into 8 disjoint blocks of 375 prompts and writes block tables under runs/L15c (read-only inputs: runs/L1a/tab_{3B,7B}_q_corrupted.parquet); run exp/l13a_sim_alloc.py unchanged on each block with split seed 0 and target n_sim 1,125 (block rows about 1,950, fit quarter about 490, sim about 1,460; UNVERIFIED: cell30 counts per block and the MIN group filter may drop cells, so check n_fit and n_sim per block before the grid and fall back to 6 blocks of 500 prompts if any block has fewer than 400 fit rows). Gates: the L13a code path reproduces runs/L13a on an unblocked file (0 differences) and the identity gate holds on each block, else VOID. Grid: 2 rungs x 8 blocks x 3 budgets (16 jobs).

**Command shape:** UNVERIFIED, script not written: `cd /homes/rdubash/avrl_live/FULL_R/avrl && mkdir -p runs/L15c && /homes/rdubash/avrl_live/FULL_R/bin/gpu_run.sh runs/L15c 'bash exp/L15c.sh'` (CPU only; the launcher must not write done.flag or exit_code).

**Metric:** per rung at 25 percent: pooled warm_online_200 / recal_free_200 with a block-level percentile bootstrap, the per-block ratios, and the paired t on block means at df 7.

**Pass threshold (the L13a PASS replicates on disjoint prompts):** pooled ratio >= 1.02 with the CI lower end above 1 on BOTH rungs, ratio above 1.00 in at least 7 of 8 blocks on both rungs, paired t p < 0.05. Reading: the shared-prompt caveat is removed and the claim-2 surviving form stands on independent prompts.

**Kill threshold:** pooled ratio <= 1.01 on EITHER rung. Reading: the L13a PASS was a shared-prompt artifact, the surviving form of claim 2 reverts to inconclusive. Anything else is inconclusive.

**Expected runtime:** about 4 min on CPU.

**Informs:** robustness of the headline only. Lowest value of the three: 24 of 24 cells and 4 of 4 splits already agree in sign, so a flip is unlikely, but it is cheap and it removes the shared-prompt caveat. Caveats: 375-prompt blocks are small, the fit sets are about 490 rows so n = 200 is 41 percent of them, and the cell30 layout is thinly populated per block (the warm and free arms do not use it).

### Iteration 16 additions to 'Proposed by the loop' (after L15a)

Added 2026-10-19 after L15a (pass; see avrl/RESEARCH_LOG.md, second entry "L15a 2026-10-19 pass"). Results re-derived from avrl/runs/L15a/ raw sim JSONs with separate code (exp/l15a_critic.py, runs/L15a/critic.log): all three gates pass (38, 54 and 1,856 cells, 0 per-seed differences). On the corrupted label at 25 percent budget and n_sim 2,250 (B = 562, 16 clusters x 8 seeds) with each of the 200 free labels flipped with probability 0.25: warm_stale_200 / greedy_online (cold) is 1.013 [1.002, 1.023] on 3B (split t interval [0.983, 1.042], 3 of 4 splits above 1) and 1.037 [1.026, 1.049] on 7B-AWQ (4 of 4 splits); warm_stale_200 / recal_free_200 (same stale labels) is 1.138 [1.113, 1.165] and 1.153 [1.121, 1.183]; the learner keeps about a third (3B) and two thirds (7B-AWQ) of the clean warm-start edge over cold. Descriptive: flip50 poisons the warm start (0.942 and 0.972 of cold; label-age weighting 0.25 recovers 0.969 and 0.996), lowlen (covariate shift) is harmless (1.037 and 1.043 of cold). The critic's reading: the pre-registered PASS holds, but 'a learning component that is not a label-count effect' is not established, because no stale fixed rule with more labels was run and the 1.06 clause is easy (the fixed rule is damaged to 0.89 and 0.90 of cold and cannot refit). Cross-judge labels cannot serve as structured staleness (the (pid, k) join of the two tables has 12,309 shared rollouts and the corrupted label is identical on all of them). Status of earlier items: L15a done (pass); L15b (drip-fed random-label control) and L15c (disjoint-prompt block replication) unchanged, L15c is amended by L16c below; L14a, L14b, L14c, L13b, L13c, L12a, L12b, L11c, L10a, L10b, L10c unchanged; every GPU item stays blocked on the E04 stop rule and L15a gives no reason to request an override. Ids are L<iteration><letter>. Ranked by expected information value.

### L16a: Stale label-count control and noise ladder (CPU only)

**Hypothesis:** L15a's PASS is read as 'adaptation to stale labels', but a fixed rule that is handed more stale labels might recover the same ground, because symmetric label flips mostly attenuate a logistic score without reordering rows (heuristic: at flip rate p the per-label signal is cut by (1 - 2p)^2, a quarter at p = 0.25, so about 800 stale labels carry roughly what 200 clean labels do). If so, the learner's 14 to 15 percent margin over the 200-label stale rule is a label-count effect again. Falsifier of 'the repair is not a label-count effect': at flip25, a fixed rule fitted once on 762 (200 + B) stale labels is at least as good as warm_stale_200.

**Change vs existing scripts:** copy avrl/exp/l15a_sim_alloc.py to l16a_sim_alloc.py. Add STALE variants flip10, flip35, flip40, flip45 (same flip mechanism, rng [policy seed, 15] on the labels). Add arms recal_free_400 and recal_free_762 under the stale labels (free_draw already nests: the permutation prefix and the first n flip draws are identical for every n, so the 200-label set is a prefix of the 762-label set; add an assert for it). Arms per job: uniform, static, greedy_online, recal_free_200/400/762 (stale), warm_online_200 (stale). Rungs 3B (primary) and 7B-AWQ, corrupted label, cheap_r, cell30, K = 1, 8 policy seeds, budgets 10 / 25 / 50 percent, n_sim 2,250 x split 0 to 3 x subsample 0 to 3, STALE in {flip10, flip25, flip35, flip40, flip45} (160 jobs). Gates: (1) STALE = flip25 reproduces runs/L15a for uniform, static, greedy_online, recal_free_200 and warm_online_200 with 0 per-seed differences, else VOID; (2) with refits off warm_online_200 equals recal_free_200 per policy seed under every STALE variant, else VOID; (3) the nesting assert holds.

**Command shape:** UNVERIFIED, script not written: `cd /homes/rdubash/avrl_live/FULL_R/avrl && mkdir -p runs/L16a && /homes/rdubash/avrl_live/FULL_R/bin/gpu_run.sh runs/L16a 'bash exp/L16a.sh'` (CPU only; L15a.sh pattern; the launcher must not write done.flag or exit_code; bin/gpu_status.sh first).

**Metric:** per rung at 25 percent budget and n_sim 2,250: recal_free_762 (stale) / warm_online_200 (stale) and recal_free_400 / warm_online_200 at each flip rate, 16-cluster percentile bootstrap, split-level paired t at df 3 with the t interval of the 4 split ratios; and the noise break-even p* where warm_online_200 / greedy_online crosses 1.00 (linear interpolation between the ladder points, with the bootstrap interval of the bracketing points).

**Pass threshold (the repair is not a label-count effect):** gates pass and at flip25, 25 percent, recal_free_762 (stale) / warm_online_200 (stale) <= 0.99 with the CI upper end below 1 on BOTH rungs. Reading: stale labels cannot be bought back with count, the stream is doing something a larger stale bank does not, and L15a's 'drift-robustness clause' stands as a learning component.

**Kill threshold (the repair is a label-count effect):** recal_free_762 (stale) / warm_online_200 (stale) >= 1.00 on EITHER rung. Reading: L15a's PASS stays true as 'warm start is not worse than cold under moderate noise' but is withdrawn as evidence of adaptation; claim 2's surviving form reverts to 'beyond the initial label supply, a label-count effect'. Anything else is inconclusive.

**Expected runtime:** about 6 min on CPU (L15a ran 128 jobs in 34 s).

**Informs:** whether the one place a non-label-count learning component could show actually shows it, and the noise level at which Phase 1 must switch from warm to cold. Caveats: symmetric flips are the benign staleness (see L16b), one task, one policy, shared-prompt splits (honest unit 4 splits).

### L16b: Structured staleness that changes the ranking (CPU only)

**Hypothesis:** symmetric flips preserve most of the ranking, so L15a's flip25 PASS may be specific to rank-preserving noise. Real staleness (an earlier checkpoint) changes the feature-label relation in the region the rule ranks highest. If stale labels reverse the relation there, the pooled refit is misled until the stream outweighs them, and the warm start loses to cold. Falsifier of 'warm start is robust to realistic staleness': under the relation-reversal variant, warm_stale_200 / greedy_online <= 0.97 with the CI upper end below 1 on either rung.

**Change vs existing scripts:** copy l15a_sim_alloc.py to l16b_sim_alloc.py. Add STALE variants applied to the 200 free labels only, identically for recal_free_200 and warm_online_200: jlflip50 (flip each label with probability 0.5 only for free rows whose judge_logit is above the fit-half median; primary), fn80 (positives flipped to negative with probability 0.8; about 23 percent of labels change), fp35 (negatives flipped to positive with probability 0.35; about 25 percent of labels change), all with rng [policy seed, 15] on the labels. Add the drop arm warm_online_200_w0 (stale labels carry weight 1 only until the first refit, then 0) next to warm_online_200_w25. Grid as L15a (2 rungs, 4 splits x 4 subsamples, 8 seeds, 3 budgets, 3 variants plus the flip25 control, 64 jobs). Gates: STALE = flip25 reproduces runs/L15a on uniform, static, greedy_online, recal_free_200, warm_online_200, warm_online_200_w25; the refits-off identity holds under every variant; the corruption is shown to be active.

**Command shape:** UNVERIFIED, script not written: `cd /homes/rdubash/avrl_live/FULL_R/avrl && mkdir -p runs/L16b && /homes/rdubash/avrl_live/FULL_R/bin/gpu_run.sh runs/L16b 'bash exp/L16b.sh'` (CPU only; the launcher must not write done.flag or exit_code).

**Metric:** per rung and variant at 25 percent budget and n_sim 2,250: warm_stale / greedy_online, warm_stale / recal_free_200 (same stale labels), w25 / greedy_online, w0 / greedy_online, ratio of means with the 16-cluster percentile bootstrap and the split-level paired t at df 3.

**Pass threshold (robust to ranking-changing staleness):** gates pass and under jlflip50 at 25 percent, on BOTH rungs warm_stale / greedy_online >= 1.00 and warm_stale / recal_free_200 (stale) >= 1.06 with the CI lower end above 1. Reading: the Phase 1 warm-start default holds beyond rank-preserving noise.

**Kill threshold:** under jlflip50, warm_stale / greedy_online <= 0.97 with the CI upper end below 1 on EITHER rung. Reading: warm start by default is withdrawn for ranking-changing staleness; Phase 1 starts cold, or warm with label-age weighting if w25 / cold or w0 / cold >= 0.99 on both rungs. Anything else is inconclusive and the fn80 / fp35 rows decide the weight.

**Expected runtime:** about 8 min on CPU.

**Informs:** the Phase 1 default under realistic drift, and which mitigation (weighting or dropping) is needed. Caveats: still simulated, not a real earlier checkpoint (a real one needs GPU rollouts and stays blocked on the E04 stop rule); the variant definitions are a design choice made before seeing results.

### L16c: L15c amended with the flip25 arms (CPU only)

**Hypothesis:** the weakest number in L15a's PASS is the 3B warm_stale / greedy_online margin (1.013, split t interval [0.983, 1.042], 3 of 4 splits above 1), and L15c's disjoint-prompt blocks give df 7 instead of df 3. Falsifier: pooled over prompt-disjoint blocks, warm_stale_200 / greedy_online at flip25 is at or below 1.00 on either rung.

**Change vs existing scripts:** run L15c's block design (8 disjoint blocks of 375 prompts, or 6 of 500 if any block has fewer than 400 fit rows) with l15a_sim_alloc.py instead of l13a_sim_alloc.py, STALE in {none, flip25}, target n_sim 1,125 per block, so both the L13a headline (none) and the L15a headline (flip25) are replicated on independent prompts. Gates: l15a_sim_alloc.py with STALE = none reproduces runs/L15a on an unblocked file; refits-off identity on each block.

**Command shape:** UNVERIFIED, script not written: `cd /homes/rdubash/avrl_live/FULL_R/avrl && mkdir -p runs/L16c && /homes/rdubash/avrl_live/FULL_R/bin/gpu_run.sh runs/L16c 'bash exp/L16c.sh'` (CPU only; the launcher must not write done.flag or exit_code).

**Metric:** per rung at 25 percent: pooled warm_stale / greedy_online and warm_stale / recal_free_200 with a block-level percentile bootstrap, per-block ratios and the paired t on block means at df 7.

**Pass threshold:** pooled warm_stale / greedy_online >= 1.00 with the CI lower end above 1 on BOTH rungs, in at least 6 of 8 blocks on 7B-AWQ and 5 of 8 on 3B, paired t p < 0.05. Reading: the thin 3B margin replicates on independent prompts.

**Kill threshold:** pooled warm_stale / greedy_online <= 0.99 on EITHER rung. Reading: the flip25 robustness clause was a shared-prompt artifact and Phase 1 defaults to cold start. Anything else is inconclusive.

**Expected runtime:** about 5 min on CPU.

**Informs:** robustness of the L15a headline on independent prompts; supersedes L15c when run. Caveats: 375-prompt blocks give fit sets of about 490 rows so n = 200 is 41 percent of them, which makes the stale labels a larger share of the fit than in L15a.


### Iteration 17 additions to 'Proposed by the loop' (after L16b)

Added 2026-10-19 after L16b (inconclusive; see avrl/RESEARCH_LOG.md, entry "L16b 2026-10-19 inconclusive"). Results re-derived from avrl/runs/L16b/ raw sim JSONs with separate code (the run's exp/l16b_critic.py, 0 mismatches over 144 ratios, and my own inline recompute): all gates pass (44, 120 and 3,200 cells, 0 per-seed differences). On the corrupted label at 25 percent budget and n_sim 2,250 (B = 562, 16 clusters x 8 seeds) with each free label whose judge_logit is below the fit-half median flipped with probability 0.5 (jlflip50, realised 24.4 percent of labels): warm_stale_200 / greedy_online (cold) is 0.992 [0.980, 1.003] on 3B (2 of 4 splits above 1) and 1.030 [1.020, 1.041] on 7B-AWQ (4 of 4); warm_stale_200 / recal_free_200 (same stale labels) is 1.168 and 1.100. PASS fails only on the 3B 1.00 floor, KILL does not fire. Descriptive: the unweighted warm start loses to cold at 25 percent under full relation reversal (0.940 and 0.966) and under positive deletion on 3B (0.954), so the literal kill rule would fire on those stress cells; dropping the stale labels from every refit (w0) never fell below 0.993 of cold in 36 cells, w25 reaches 0.966. The w0 price when the earlier labels are clean was not measured (no STALE = none in the grid). Status of earlier items: L16a (stale label-count control and noise ladder) and L16c (L15c with flip25 arms) unchanged and still unrun; L17b below extends L16a's count control to the L16b variants and L17c extends L16c with jlflip50 and w0; L15b, L15c, L14a, L14b, L14c, L13b, L13c, L12a, L12b, L11c, L10a, L10b, L10c unchanged; every GPU item stays blocked on the E04 stop rule and L16b gives no reason to request an override. Ids are L<iteration><letter>. Ranked by expected information value.

### L17a: The price of the w0 insurance (CPU only)

**Hypothesis:** L16b found that dropping the stale labels from every refit (w0) keeps the learner at or above 0.993 of cold in all 36 cells, which makes w0 the safe Phase 1 form. But w0 was only measured when the earlier labels are damaged. When they are clean, w0 throws away the label supply that gave L13a its 4 to 6 percent warm edge (warm 276.4 and 262.2 against cold 265.6 and 247.9 errors caught), so w0 may be a bad default if staleness is usually mild. Falsifier of 'w0 is a free insurance': under STALE = none, w0 / warm_online_200 (clean) is clearly below 1.

**Change vs existing scripts:** copy avrl/exp/l16b_sim_alloc.py to l17a_sim_alloc.py. STALE already accepts none. Add arm warm_online_200_wd50 (stale labels carry weight 1 in refits until the stream holds 50 labels, then weight 0; a delayed drop that should keep most of the clean edge and still remove stale labels once the stream can stand alone). Arms per job: uniform, static, greedy_online (cold), recal_free_200, warm_online_200, warm_online_200_w0, warm_online_200_wd50. Grid: corrupted label, rungs 3B and 7B-AWQ from runs/L1a/tab_{3B,7B}_q_corrupted.parquet, cell30, cheap_r, K = 1, 8 policy seeds, budgets 10 / 25 / 50 percent, n_sim 2,250 x split 0..3 x subsample 0..3, STALE in {none, flip25, jlflip50, jlrev100, fn80} (160 jobs). Gates: (1) STALE = none reproduces runs/L13a for uniform, static, greedy_online, recal_free_200 and warm_online_200 and STALE = jlflip50 reproduces runs/L16b for all arms they share, with 0 per-seed differences, else VOID; (2) refits off: warm, w0 and wd50 equal recal_free_200 per policy seed under every STALE variant, else VOID; (3) the drop step is shown active: wd50 differs from warm_online_200 in at least one budget cell for every STALE variant, and equals it per seed in any cell where the stream never reaches 50 labels.

**Command shape:** UNVERIFIED, script not written: `cd /homes/rdubash/avrl_live/FULL_R/avrl && mkdir -p runs/L17a && /homes/rdubash/avrl_live/FULL_R/bin/gpu_run.sh runs/L17a 'bash exp/L17a.sh'` (CPU only; L16b.sh pattern; the launcher must not write done.flag or exit_code; bin/gpu_status.sh first).

**Metric:** per rung and STALE at 25 percent budget and n_sim 2,250: w0 / warm_online_200, wd50 / warm_online_200, w0 / cold and wd50 / cold, ratio of means with the 16-cluster percentile bootstrap and the split-level paired t at df 3 with the t interval of the 4 split ratios; the same at 10 and 50 percent descriptively; and the minimum over the five STALE variants of each schedule against cold.

**Pass threshold (w0 is a free insurance):** gates pass and under STALE = none at 25 percent, on BOTH rungs, w0 / warm_online_200 >= 0.98 with the CI lower end above 0.96. Reading: dropping stale labels costs at most 2 percent when they are clean and w0 becomes the Phase 1 default without a staleness detector.

**Kill threshold (w0 forfeits the clean-label edge):** w0 / warm_online_200 <= 0.97 with the CI upper end below 1 on EITHER rung under STALE = none. Reading: the Phase 1 default is a choice that needs either a staleness detector (compare the stale-fit rule's AUC on the first stream labels against its cross-validated AUC) or wd50; if wd50 / cold >= 0.99 in every variant and wd50 / warm_online_200 >= 0.98 under none, wd50 becomes the default. Anything else is inconclusive.

**Expected runtime:** about 8 min on CPU (L16b ran 192 jobs in 56 s).

**Informs:** the Phase 1 default (warm, w0 or delayed drop) and whether any detector is worth building. Caveats: synthetic staleness only (a real earlier checkpoint needs GPU rollouts and stays blocked on the E04 stop rule), one task, one policy, shared-prompt splits (honest unit 4 splits).

### L17b: Label-count control under ranking-changing staleness (CPU only)

**Hypothesis:** L15a's critic showed that under symmetric flip25 a fixed rule given 762 stale labels reaches 0.993 and 1.002 of cold and is only 2.0 and 3.5 percent behind warm_stale, so the 1.14 to 1.15 margin over the 200-label stale rule was mostly label count. Symmetric flips average out with more labels; the L16b flips do not, because they are systematic in the region the rule ranks highest (jlflip50 flips each label in the below-median half with probability 0.5, which cancels the error signal the rule needs exactly where it ranks highest). If so, a fixed rule with 762 stale labels stays damaged and the 1.168 and 1.100 of L16b are not a label-count effect. Falsifier of 'the repair is not a label-count effect under ranking-changing staleness': recal_free_762 under jlflip50 is at least as good as warm_stale_200.

**Change vs existing scripts:** copy avrl/exp/l16b_sim_alloc.py to l17b_sim_alloc.py (or add the arms to the L16a script if L16a is run first, and use the union of the two STALE sets). Add arms recal_free_400 and recal_free_762 under the stale labels (free_draw already nests: the permutation prefix and the first n flip draws are identical for every n; assert that the 200-label set is a prefix of the 762-label set, and that the median used by the jl variants is still the fit-half median, not the free-draw median). Arms per job: uniform, static, greedy_online, recal_free_200 / 400 / 762, warm_online_200. STALE in {jlflip50, jlrev100, fn80, flip25} (rung x variant x 16 = 128 jobs), corrupted label, both rungs, cheap_r, cell30, K = 1, 8 policy seeds, budgets 10 / 25 / 50 percent, n_sim 2,250. Gates: (1) STALE = flip25 reproduces runs/L15a and STALE = jlflip50 reproduces runs/L16b for the shared arms with 0 per-seed differences, else VOID; (2) the nesting assert holds; (3) the flip statistics per variant equal L16b's (realised fraction 0.24 to 0.25 for jlflip50 and flip25).

**Command shape:** UNVERIFIED, script not written: `cd /homes/rdubash/avrl_live/FULL_R/avrl && mkdir -p runs/L17b && /homes/rdubash/avrl_live/FULL_R/bin/gpu_run.sh runs/L17b 'bash exp/L17b.sh'` (CPU only; the launcher must not write done.flag or exit_code).

**Metric:** per rung and variant at 25 percent budget and n_sim 2,250: recal_free_762 (stale) / warm_online_200 (stale) and recal_free_400 / warm_online_200, 16-cluster percentile bootstrap, split-level paired t at df 3 with the t interval of the 4 split ratios.

**Pass threshold (not a label-count effect under systematic staleness):** gates pass and under jlflip50 at 25 percent recal_free_762 (stale) / warm_online_200 (stale) <= 0.99 with the CI upper end below 1 on BOTH rungs. Reading: more stale labels cannot buy back what the stream buys, so claim 2's surviving form gains its first comparator-fair robustness clause (still only against a fixed rule that holds stale labels, not against one given fresh labels).

**Kill threshold (a label-count effect again):** recal_free_762 (stale) / warm_online_200 (stale) >= 1.00 on EITHER rung. Reading: the L16b 1.10 to 1.17 margin is withdrawn as evidence of adaptation, claim 2's surviving form reverts to 'beyond the initial label supply, a label-count effect' under every staleness tried. Anything else is inconclusive.

**Expected runtime:** about 8 min on CPU.

**Informs:** the one remaining reading in claim 2's surviving form that is not yet comparator-fair. Caveats: synthetic staleness, a fixed rule with fresh labels (200 stale plus B fresh random-draw labels) is a different and fairer comparator that this run does not include and L13a's static ceiling only approximates.

### L17c: L16c amended with the jlflip50 and w0 arms (CPU only)

**Hypothesis:** the 3B primary of L16b is a statistical tie at the honest unit (warm_stale / cold 0.992, split t interval [0.961, 1.023], 2 of 4 splits above 1), and the 4 splits share prompts. L16c's block design (8 disjoint blocks of 375 prompts, or 6 of 500 if a block has fewer than 400 fit rows) gives df 7 on independent prompts. Falsifier of 'warm start is not worse than cold under below-median staleness': pooled over prompt-disjoint blocks warm_stale_200 / greedy_online under jlflip50 is at or below 0.99 on 3B.

**Change vs existing scripts:** run the L16c block design with l16b_sim_alloc.py instead of l15a_sim_alloc.py, STALE in {none, flip25, jlflip50}, target n_sim 1,125 per block, arms uniform, static, greedy_online, recal_free_200, warm_online_200, warm_online_200_w25, warm_online_200_w0, so the L13a headline (none), the L15a headline (flip25) and the L16b primary (jlflip50) are all replicated on independent prompts (merge with L16c if L16c has not run). Gates: l16b_sim_alloc.py with STALE = jlflip50 reproduces runs/L16b on an unblocked file; refits-off identity on each block.

**Command shape:** UNVERIFIED, script not written: `cd /homes/rdubash/avrl_live/FULL_R/avrl && mkdir -p runs/L17c && /homes/rdubash/avrl_live/FULL_R/bin/gpu_run.sh runs/L17c 'bash exp/L17c.sh'` (CPU only; the launcher must not write done.flag or exit_code).

**Metric:** per rung at 25 percent: pooled warm_stale / greedy_online, warm_stale / recal_free_200 and w0 / greedy_online under jlflip50 with a block-level percentile bootstrap, per-block ratios and the paired t on block means at df 7.

**Pass threshold:** pooled warm_stale / greedy_online >= 1.00 with the CI lower end above 1 on 7B-AWQ, and >= 1.00 on 3B with the CI lower end above 0.98, with the sign agreeing in at least 5 of 8 blocks on 3B and 6 of 8 on 7B-AWQ. Reading: the 3B L16b miss was a shared-prompt artifact and the unweighted warm start is safe under below-median staleness.

**Kill threshold:** pooled warm_stale / greedy_online <= 0.99 with the CI upper end below 1 on 3B. Reading: below-median staleness is a real small loss on the 3B rung, the unweighted warm start is withdrawn for ranking-changing staleness and w0 (or wd50 from L17a) is the default. Anything else is inconclusive.

**Expected runtime:** about 6 min on CPU.

**Informs:** whether the one non-passing clause of L16b is a tie or a loss on independent prompts; lower value than L17a and L17b because w0 already covers the loss case if L17a prices it as cheap. Caveats: 375-prompt blocks give fit sets of about 490 rows so n = 200 is 41 percent of them, which makes the stale labels a larger share of the fit than in L16b.


### Iteration 18 additions to 'Proposed by the loop' (after L8a)

Added 2026-10-19 after L8a (PASS under the pre-registered rules; see avrl/RESEARCH_LOG.md, entry "L8a 2026-10-19 pass"). Results re-derived from the raw runs/L8a JSONs with separate code: the regression gate against runs/L7a drift has 0 differences in 9,216 per-batch values (64 pairs); under a two-feature relation flip (z of judge_top and judge_mean negated on every post-switch row, transfer AUC 0.499 in d1 and 0.692 in d2) the best forgetting arm (window_125, decay_hl100, window_125, window_250 in the four cells) beats unbounded carry_plain by 1.122, 1.190, 1.051 and 1.091 over batches 7 to 12 (CI lower ends 1.086, 1.173, 1.026, 1.080) and is 1.040 to 1.133 of thr_agreement in batches 10 to 12; kill (i) 0 of 4 and kill (ii) 0 of 4 in all three folds. Unbounded carry_plain is 0.896 to 0.950 of thr_agreement in batches 7 to 9 and 0.940 to 0.985 in 10 to 12; freeze_plain_k6 is 0.54 to 0.82 of it. The best persistent arm is only 1.007 to 1.018 of thr_agreement in batches 7 to 9 in three cells, and its optimum sits on the grid edge in three of four cells. A 250-label window or a 100-label half-life pays 1.5 to 2 percent against carry_plain when the relation does not change (post hoc), and an oracle drift detector would add at most about 2.5 percent over such a fixed horizon (post hoc ceiling, single cells up to 5.3 percent). Status of earlier items: L8a done; L8b (equal-label staleness and gradual drift) unchanged and its gradual-drift part is folded into L18b below; L8c is superseded by L18c (amended comparator) and conditional; L17a, L17b, L17c, L16a, L16c and everything older unchanged; every GPU item stays blocked on the E04 stop rule and L8a gives no reason to request an override. Ids are L<iteration><letter>. Ranked by expected information value.

### L18a: Harsh relation flip with a shorter-horizon grid (CPU only)

**Hypothesis:** L8a flipped only judge_top and judge_mean, so the other eight features stayed truthful and an unbounded learner was never pushed far below thr_agreement (0.90 to 0.95 in batches 7 to 9); kill (ii) therefore had little chance of firing. The L7a diagnostic shows a harsher flip (also negating judge_std and n_distinct) drops the transferred AUC to 0.176 (3B to 7B-AWQ) and 0.168 (7B-AWQ to 3B), i.e. the table-A rule is strongly anti-informative while thr_agreement (agreement only) is untouched. Falsifier of 'a roughly 250-label forgetting horizon keeps the learner at or above the fixed rule through a relation change': under the four-feature flip the best persistent arm, including the new shorter horizons, stays below thr_agreement in both batches 7 to 9 and 10 to 12.

**Change vs existing scripts:** copy avrl/exp/l8a_flip.py to avrl/exp/l18a_harsh.py (copy, never import; l8a files untouched). FLIP_COLS = (judge_top, judge_mean, judge_std, n_distinct), flip factor f as before (f = -1 experiment, f = +1 gate). Add arms carry_plain_window_62 and carry_plain_decay_hl50 (same sample-weight path as the existing window and decay arms); every other arm and the layout (16 order seeds, default_rng(10000 + order seed), 12 batches of 250, batches 1 to 6 table A and 7 to 12 table B, both directions, B = 62 and 125, policy seed = order seed) verbatim. Gates: (1) f = +1 reproduces runs/L8a gate on all shared arms and order seeds 0 to 15 with 0 differences, else VOID; (2) premise: the rule fit on table A of order seed 0 and scored on table B has transfer AUC <= 0.35 in both directions at f = -1 (else report 'flip not harsher than L8a', not VOID).

**Command shape:** UNVERIFIED, script not written: `cd /homes/rdubash/avrl_live/FULL_R/avrl && mkdir -p runs/L18a && /homes/rdubash/avrl_live/FULL_R/bin/gpu_status.sh && /homes/rdubash/avrl_live/FULL_R/bin/gpu_run.sh runs/L18a 'bash exp/L18a.sh'` (CPU only; L8a.sh pattern, xargs -P 24 over direction x B x order seed x {flip, gate}, 128 jobs, then the summarizer; the launcher must not write done.flag or exit_code).

**Metric:** as L8a (verdict fold = order seeds 8 to 15 with the best arm chosen on 0 to 7, swapped fold and pooled 16, bootstrap over order seeds, 10,000 resamples): best forgetting / carry_plain over batches 7 to 12, best persistent / thr_agreement in 7-9 and 10-12, carry_plain and freeze_plain_k6 / thr_agreement, reset_at_switch as upper bound only; plus window_62 / window_125 and decay_hl50 / decay_hl100 over batches 7 to 12 (does the optimum lie below the L8a grid edge).

**Pass threshold:** gates pass, and the best forgetting arm beats carry_plain by >= 3 percent over batches 7 to 12 with the CI lower bound above 1 and is >= thr_agreement in batches 10 to 12 in at least 3 of 4 cells. Reading: the horizon requirement and the persistence-with-horizon safety claim both survive the harshest synthetic relation change; E13's harsh-drift number is the 4-feature one.

**Kill threshold:** the best persistent arm (excluding reset_at_switch) is below thr_agreement in both batches 7 to 9 and 10 to 12 in at least 3 of 4 cells. Reading: persistence with any horizon in this grid is unsafe under a strong relation change, Phase 1 needs a detector or a fallback to thr_agreement, and claim 2's drift-robustness reading is withdrawn at this severity. Anything else is inconclusive.

**Expected runtime:** about 10 min on CPU (L8a: 128 jobs in about 5 min; two more arms).

**Informs:** the one falsifier L8a could not stress (kill (ii)), and whether the optimal horizon is below 125 labels. Caveats: still synthetic, known-time and step-shaped, fixed rollouts, order-seed CIs optimistic (16 permutations of the same 3,000 prompts), direction confound (7B-AWQ labels are cleaner).

### L18b: Horizon price and minimax horizon over five drift scenarios (CPU only)

**Hypothesis:** L8a left two facts that pull in opposite directions: a short memory gains about 10 percent when the relation flips, and costs 1.5 to 4.4 percent when it does not (post hoc: forgetting arm / carry_plain at f = +1, mean of 4 cells: window_125 0.956, window_250 0.982, decay_hl100 0.984), and the best horizon sat on the grid edge in 3 of 4 cells. Phase 1 needs one horizon. Falsifier of 'a single fixed horizon is a safe Phase 1 spec': the minimum over arms of the maximum regret across scenarios is 5 percent or more in at least 3 of 4 cells.

**Change vs existing scripts:** copy avrl/exp/l8a_flip.py to avrl/exp/l18b_horizon.py. Scenarios per (cell, order seed): none (control_z, all 12 batches from table A), swap (f = +1), zero (f = 0, judge_top and judge_mean z zeroed on post-switch rows), flip (f = -1), ramp (f_j = max(-1, 1 - 2 (j - 6) / 3) for batches j = 7 to 12, i.e. 0.333, -0.333, -1, -1, -1, -1). Arms: carry_plain, windows 62 / 94 / 125 / 188 / 250 / 375 labels, decays with half-life 50 / 100 / 150 / 250 labels, thr_agreement, reset_at_switch (reference). Gates: none reproduces runs/L7a control_z, swap reproduces runs/L8a gate and flip reproduces runs/L8a flip on the shared arms with 0 differences, else VOID. Selection: the minimax-regret arm per cell is chosen on order seeds 0 to 7 (regret = best arm in that cell and scenario / arm over batches 7 to 12), reported on seeds 8 to 15, swapped fold and pooled 16 as cross-checks; also report the single arm with the best mean-over-cells worst-case regret.

**Command shape:** UNVERIFIED, script not written: `cd /homes/rdubash/avrl_live/FULL_R/avrl && mkdir -p runs/L18b && /homes/rdubash/avrl_live/FULL_R/bin/gpu_status.sh && /homes/rdubash/avrl_live/FULL_R/bin/gpu_run.sh runs/L18b 'bash exp/L18b.sh'` (CPU only; xargs -P 24 over direction x B x order seed x scenario, 320 jobs; the launcher must not write done.flag or exit_code; the 45-minute cap is the limit, job count may need a split into two launches if a smoke job exceeds 3 s per arm set).

**Metric:** per cell: regret of each arm in each scenario, maximum regret per arm, the minimax arm and its worst scenario, the arm's ratio to thr_agreement in batches 7-9 and 10-12 under zero, flip and ramp, and the ratio of the minimax arm to carry_plain under none and swap (the price).

**Pass threshold:** gates pass and one single arm has maximum regret <= 1.03 across the five scenarios in at least 3 of 4 cells on the report fold (the same arm in those cells within one grid step). Reading: that horizon is the Phase 1 spec, a detector is unnecessary, and the E13 table gets one horizon with its no-drift price.

**Kill threshold:** the minimum over arms of the maximum regret is >= 1.05 in at least 3 of 4 cells. Reading: no single horizon covers stationary and drifting settings, and Phase 1 needs a detector (L18c) or a horizon ensemble. Anything else is inconclusive.

**Expected runtime:** about 15 to 20 min on CPU (L8a: 128 jobs in about 5 min; 320 jobs with about twice the arms).

**Informs:** the number Phase 1 uses for its forgetting horizon, its price when nothing drifts, and the gradual-drift check that L8b asked for. Caveats: synthetic drifts, regret is measured against the best of the same grid on the same seeds (mildly optimistic for the reference), order-seed CIs optimistic.

### L18c: Drift detector against the fixed horizon (amended L8c; CPU only, run only if L18b finds no single horizon within 3 percent)

**Hypothesis:** L8c as written measured a detector against the gap between carry_plain and reset_at_switch; L8a shows a fixed 250-label window already closes that gap (best persistent / reset_at_switch 0.97 to 1.00 in the verdict fold), and the oracle ceiling for a detector over a fixed 250-label window or hl100 is about 1.025 averaged over a 50/50 mix of flip and swap (post hoc, single cells 1.000 to 1.053). The only honest comparator is the L18b horizon. Falsifier of 'a detector is worth building': detect_window / the L18b minimax horizon is at or below 1.00 averaged over the scenarios none, swap and flip, or the false-alarm cost in the none scenario exceeds 1 percent.

**Change vs existing scripts:** copy avrl/exp/l8a_flip.py to avrl/exp/l18c_detect.py. New arm detect_window: behaves as carry_plain until, after a batch, the realised hit rate of that batch's picks is more than 2 binomial SE below the fit's own predicted precision on those picks in two consecutive batches; it then applies the L18b horizon W* to the store (drop labels older than W*) and keeps W* from there. Parameters fixed before the run, not tuned on the verdict seeds. Comparators: carry_plain, the fixed W*, reset_at_switch (reference). Scenarios none, swap and flip. Gate: shared arms reproduce runs/L18b exactly.

**Command shape:** UNVERIFIED, script not written: L18b launcher pattern through bin/gpu_run.sh runs/L18c (CPU only; the launcher must not write done.flag or exit_code).

**Metric:** detect_window / W* over batches 7 to 12 per scenario and cell with the bootstrap CI over order seeds, detection delay in batches under flip, false-alarm rate and cost under none.

**Pass threshold:** mean over scenarios (none, swap, flip) of detect_window / W* >= 1.01 with the CI lower bound above 1 in at least 3 of 4 cells, and false-alarm cost <= 1 percent in none. Reading: a cheap detector earns about 1 to 2 percent over the fixed horizon and Phase 1 is persistent plus detector.

**Kill threshold:** mean detect_window / W* <= 1.00 in at least 3 of 4 cells, or false-alarm cost > 2 percent in none. Reading: the fixed horizon is the Phase 1 spec and the detector family is retired. Anything else is inconclusive.

**Expected runtime:** about 10 min on CPU.

**Informs:** whether a detector is worth any complexity given a 2.5 percent ceiling. Caveats: the ceiling is post hoc and uses the oracle choice per scenario; detection cannot recover the first post-switch batch (every detector loses batch 7, 0.81 to 0.95 of thr for unbounded carry_plain); synthetic drift, one population, offline replay.
