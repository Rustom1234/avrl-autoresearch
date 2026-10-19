# AVRL Experiment Backlog

Pre-registered, 2026-09-30. Twenty experiments, E01 through E20, ordered by expected information value per GPU hour: cheapest, highest-leverage checks first, most expensive and most infrastructure-blocked last. This ordering means analysis-only reruns of already-collected data (near-zero marginal GPU cost) come before any new generation run, and new generation runs on this machine come before any GRPO training run on an A100.

## Claim numbering (from PROJECT.md)

- Claim 1: the learned allocator reaches the same policy quality with a fraction of expensive verifier calls.
- Claim 2: the learned allocator admits fewer corrupted rewards into training than any fixed rule at the same budget.
- Claim 3: the allocator's edge comes from structure in prompt space (proxy error clusters), the SOFisher condition (Li et al., 2026, Nature Communications, https://doi.org/10.1038/s41467-026-73404-6).

Source for the existing harness read this session: /homes/rdubash/avrl_live/FULL/avrl/PROJECT.md, /homes/rdubash/avrl_live/FULL/avrl/p0_rollouts.py, /homes/rdubash/avrl_live/FULL/avrl/p0_analyze.py, /homes/rdubash/avrl_live/FULL/avrl/allocators.py. Confirmed at read time: avrl/p1_grpo.py (the phase-1 GRPO training harness named in PROJECT.md's Files section) does not exist on disk yet, only p0_rollouts.py, p0_analyze.py, and allocators.py do. Every experiment below that needs an actual policy-training loop is marked NEEDS PHASE-1 HARNESS.

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

All paper citations above are carried forward unchanged from /homes/rdubash/avrl_live/FULL/lit/avrl/SYNTHESIS.md, read this session, which itself states its method: built via WebFetch against arXiv abstracts, the OpenAlex works API, and direct publisher DOIs, with WebSearch unavailable (budget error) and not retried. This document adds one new fact verified this session via WebFetch: HuggingFaceH4/MATH-500's existence, size, and content, https://huggingface.co/datasets/HuggingFaceH4/MATH-500. No paper, author, or number is invented in this document; every runtime figure not directly read from avrl/runs/p0/run.log is explicitly marked as an estimate or UNVERIFIED. G01-G04 were added after this session's three gap checks (GAP_already-done.md, GAP_cannot-work.md, GAP_not-a-thesis.md); their citations (VIP, https://arxiv.org/abs/2602.01601) are carried forward unchanged from SYNTHESIS.md and the gap-check files.

## Compute correction (2026-09-30, main session)
This machine's GPU is an NVIDIA RTX A4000 with 16 GB, not a 32 GB card. Measured with the phase-1 harness (Qwen2.5-1.5B, LoRA r=32, bf16): full settings (16 prompts x G=8, 512 tokens) run at 153 s per step with HF generation and peak 13.2 GB; vLLM colocate does not fit beside training on this card (out of memory at engine start). Smoke settings run at 20 s per step. vLLM-only rollout experiments (phase-0 style) are unaffected and fit in 40 minutes at 1,500 to 3,000 prompts.

## Feasibility note (this run)
Items marked A100, needing a larger cluster, a second machine, or the phase-1 GRPO harness (E15 to E20, G04, anything needing p1_grpo.py; about 89 s per step on this card, so 100 steps take 2.5 hours) are not runnable under the 45-minute cap and must not be planned.

## Proposed by the loop

Ids are L<iteration><letter>. Iteration 1 = the loop run whose first experiment was E21-lowT-confident-errors (RESEARCH_LOG.md, 2026-10-18, inconclusive). The iteration number is inferred (no counter exists under FULL); renumber if the loop keeps one elsewhere. All three items are runnable on this machine (one A4000, 45 minute cap) and none needs p1_grpo.py.

### L1a: Warm-started learned allocator replay (CPU only)

**Hypothesis:** the learned allocator's deficit against the label-free agreement threshold at the 10% budget (learned / threshold 0.70 to 0.84 in E21, 0.79 to 0.83 at T=1.0) is mostly a cold-start artifact: in sim_e21.py run_online the scores are random until the first refit after max(budget // 8, 40) labels, so 40 of 154 audits at 10% (48 of 386 at 25%) are random picks at the 13.7% base rate. If the learner starts from the agreement ordering and then refits, it may match or beat the threshold at equal budget.

**Change vs existing scripts:** copy exp/sim_e21.py to a new file; replace the initial `score` in run_online with an agreement-based score (agreement ascending, n_distinct descending) and keep the refit schedule, greedy and TS variants; no other change. Data already exists: runs/E21-lowT-confident-errors/feats_T0.25.parquet, feats_T0.5.parquet, feats_T1.0.parquet (cells 30 and 100, cheap and cheap+emb).

**Command shape:** CPU in the shell, `/homes/rdubash/active-audit/.venv/bin/python exp/sim_warm.py <feats> proxy_error cell100 <cols> <out.json>` for each of the three feats files; no gpu_run.sh needed.

**Metric:** errors caught per budget (10/25/50%) for threshold, warm-started greedy, warm-started TS, oracle; learned_warm / threshold and the paired gap in seed SDs (8 seeds).

**Pass threshold:** learned_warm / threshold >= 1.05 with the gap larger than 2 paired seed SDs at the 25% budget at T=0.25 or T=0.5 (claim 2 survives offline in its fair form).

**Kill threshold:** learned_warm / threshold <= 1.02 at every budget and temperature (claim 2 is closed offline on GSM8K with majority vote; the fixed threshold is the deliverable and the cold-start confound is removed).

**Expected runtime:** under 10 minutes on CPU (E21's 12 replay jobs took about 2 minutes; this adds one policy variant).

**Informs:** claim 2 directly. Prior (judgment): ratio 1.00 to 1.04, since static-with-free-labels, an upper bound on cheap-feature learning here, is only 1.00x to 1.06x the threshold at 25%. Highest information per cost in the loop.

**Result (run 2026-10-18, runs/L1a-warm-start-replay/, RESEARCH_LOG.md entry "L1a-warm-start-replay"): logged as inconclusive (driver label MARGINAL, neither pass nor kill; numbers re-derived from the raw JSONs by the critic pass).** CPU, 1.5 min wall; script exp/sim_warm.py (+ exp/L1a-warm-start-replay.sh, exp/l1a_table.py). Sanity passed (refit-disabled greedy_warm equals agree_thr per seed; reproduced E21 baselines equal per seed). Headline cell100 learned_warm / threshold at 10/25/50%: T=0.25 cheap 1.084/1.029/1.011, cheap+emb 0.951/1.014/0.984; T=0.5 cheap 1.061/0.994/0.997, cheap+emb 1.049/1.014/0.992; T=1.0 cheap 0.853/1.044/1.020, cheap+emb 0.832/1.049/1.015. Best 25% ratio at T=0.25 or 0.5 is 1.029 (pass bar 1.05), pool-bootstrap 95% CI [0.990, 1.069] (50 resamples x 2 seeds), so no pass; the kill (<= 1.02 everywhere) fails because the warm learner is 1.05 to 1.08 at 10% in three headline cells. Cold start explains the whole 10% deficit at T=0.25 and 0.5 (warm minus cold +14 to +23 errors) but almost none at T=1.0 (+2 to +4; the first refit sees 97.8% errors in its 40 labels). Claim 2 stays open offline: about a tie at 25%, closes only -2% to +8% of the threshold-to-oracle gap. Consequence for the loop: any later learned-vs-threshold replay (L1b judge proxy, MATH-500, code) must use sim_warm.py, not sim_e21.py; L1b is the next pick; an optional 2-minute CPU follow-up (bootstrap the 10% headline cells, blended-score refit) is described in the log entry.

### L1b: Judge proxy on the saved E21 rollout text (E04 variant, GPU)

**Hypothesis:** with a learned judge as the proxy (no free agreement signal), proxy errors cluster in prompt space more than majority-vote errors do (E21: none), and judge errors overlap only partly with majority-vote errors.

**Change vs existing scripts:** new script exp/l1b_judge_score.py (does not exist) scoring each rollout in runs/E21-lowT-confident-errors/T0.25/rollouts.parquet (16,000 rows, `text` column saved by E21) with the local /homes/rdubash/avrl_live/FULL/models/Qwen3-4B as a prefill-only Yes/No logprob judge on question plus completion (Skywork-Reward-V2 weights are not under FULL/models; if they are provided there later, swap them in). Proxy error = judge accepts (P(Yes) >= tau, tau set so the accept rate matches the rollout accuracy) and gold is wrong; aggregate to a group label. Drop agreement from the feature set so the baseline is a judge-score threshold; keep length and truncation stats and the MiniLM PCA features.

**Command shape:** `/homes/rdubash/avrl_live/FULL/bin/gpu_run.sh /homes/rdubash/avrl_live/FULL/avrl/runs/L1b-judge '/homes/rdubash/active-audit/.venv-judge/bin/python exp/l1b_judge_score.py --model /homes/rdubash/avrl_live/FULL/models/Qwen3-4B --rollouts /homes/rdubash/avrl_live/FULL/avrl/runs/E21-lowT-confident-errors/T0.25/rollouts.parquet --out /homes/rdubash/avrl_live/FULL/avrl/runs/L1b-judge'` then CPU analysis with a --run_dir copy of e21_lowT_analyze.py.

**Metric:** judge-proxy error rate overall and judge-confident subset; AUC of judge score alone, cheap, cheap+emb (delta AUC cheap+emb minus cheap), confident neighbor ratio at cell100 against its permutation null; Jaccard overlap of judge-wrong and majority-vote-wrong groups; the 10/25/50% replay with the judge-score threshold as baseline.

**Pass threshold:** E04's: delta AUC (cheap+emb minus cheap) >= 0.05 AND Jaccard < 0.5, plus confident neighbor ratio at cell100 >= 1.5 and above its null p95.

**Kill threshold:** E04's: Jaccard > 0.8, or delta AUC < 0.03 with neighbor ratio < 1.3 (claim 3 dead for both proxy types tried).

**Expected runtime:** UNVERIFIED estimate 20 to 35 minutes: about 7M prefill tokens through a 4B bf16 model on an A4000, throughput not measured here; run T=0.25 only, and cut to a 8,000-row subsample of pids if the first 5 minutes project past 40 minutes.

**Informs:** claim 3 directly (first regime where the proxy has no free confidence signal) and the allocator-design question of which proxy to build around.

### L1c: E21b, powered T=0.25 rerun (GPU, confirmatory)

**Hypothesis:** the E21 verdict is inconclusive only because n_conf_err = 94 missed the pre-registered 100 at T=0.25 (binomial SD 9.4; P(>=100) about 28% at the observed rate); with 3,000 prompts the same rule becomes decisive (expected about 141 confident errors).

**Change vs existing scripts:** copy exp/e21_lowT_rollouts.py with a --pid_start flag so only pids 2000..2999 of runs/p0/groups.parquet are generated at T=0.25 (same model, G=8, max_tokens 512, seed 0), merge with runs/E21-lowT-confident-errors/T0.25 into a 3,000-prompt dataset, rerun exp/e21_lowT_analyze.py with --n_prompts 3000 and the same pre-registered rules. Per-request seeding makes the merged set statistically equivalent, not bit-identical, to one 3,000-prompt run.

**Command shape:** `/homes/rdubash/avrl_live/FULL/bin/gpu_run.sh /homes/rdubash/avrl_live/FULL/avrl/runs/L1c-e21b '/homes/rdubash/active-audit/.venv-judge/bin/python exp/l1c_rollouts.py --model /homes/rdubash/avrl_live/FULL/models/Qwen2.5-1.5B-Instruct --temps 0.25 --pid_start 2000 --n_prompts 3000 --G 8 --max_tokens 512 --seed 0 --ref_groups /homes/rdubash/avrl_live/FULL/avrl/runs/p0/groups.parquet --out /homes/rdubash/avrl_live/FULL/avrl/runs/L1c-e21b'`, then the CPU analysis.

**Metric:** E21's: n_conf_err, confident delta AUC (cheap+emb minus cheap) with fold SD, confident neighbor ratio at cell100 vs null p95, learned / threshold at 10% and 25%, H(25%).

**Pass threshold:** E21's: n_conf_err >= 100 AND delta AUC >= 0.05 with neighbor ratio >= 1.5, OR learned >= 1.10x threshold at 25% with the gap > 2 seed SDs; and H(25%) >= 0.10.

**Kill threshold:** E21's: n_conf_err >= 100 at T=0.25 AND delta AUC < 0.03, neighbor ratio < 1.3, learned <= 1.03x threshold at 10% and 25%.

**Expected runtime:** about 8 to 10 minutes of GPU (1,000 prompts x 8 rollouts plus engine load; E21 generated 2,000 x 8 in 653 s) plus about 2 minutes CPU.

**Informs:** claim 3 and the offline form of claim 2 (formal verdict on GSM8K with majority vote). Expected outcome is a formal kill, so lower priority than L1a and L1b; E12 (embedder ablation) stays unwarranted (embedding gain 0.010 and 0.001 vs the 0.03 trigger).

### Iteration 2 (proposed after L1a-warm-start-replay, 2026-10-18; verdict inconclusive)

Iteration 2 = the loop run that executed L1a. L1a's own result paragraph above is the run agent's; the critic pass re-derived its numbers from the raw JSONs (RESEARCH_LOG.md, entry "L1a-warm-start-replay") and set the verdict to inconclusive (the driver calls it MARGINAL). All three items are CPU only, reuse exp/sim_warm.py and runs/E21-lowT-confident-errors/feats_T*.parquet, and need no gpu_run.sh. L1b (GPU) stays the next GPU pick and must use sim_warm.py as its replay.

### L2a: Pool bootstrap of the 10% and T=1.0 headline cells, plus blended-score refit (CPU)

**Hypothesis:** the warm learner's 10% edge over the agreement threshold (cell100 cheap: 1.084 at T=0.25, 1.061 at T=0.5) survives pool resampling, whereas L1a only bootstrapped the 25% budget.

**Change vs existing scripts:** copy exp/sim_warm.py to exp/sim_warm_boot.py; make the BOOT mode take a budget fraction (env BOOT_FRAC, default 0.25); add a greedy variant `greedy_blend` whose refit score is the mean of the warm rank score and the rank of the refit probability (so the refit cannot discard the agreement ordering, which is what hurts at T=1.0 and 10%).

**Command shape:** `BOOT_B=50 BOOT_SEEDS=2 BOOT_FRAC=0.1 /homes/rdubash/active-audit/.venv/bin/python exp/sim_warm_boot.py runs/E21-lowT-confident-errors/feats_T0.25.parquet proxy_error cell100 <cheap cols> runs/L2a/boot_T0.25_0.1.json`, repeated for T=0.25 and 0.5 at 0.1, and T=1.0 at 0.1 and 0.25.

**Metric:** bootstrap ratio greedy_warm / threshold and greedy_blend / threshold (mean, 95% CI, fraction of resamples above 1), and gap in errors with CI.

**Pass threshold:** the 95% CI of greedy_warm / threshold excludes 1.0 at 10% in cell100 cheap at both T=0.25 and T=0.5 (the L1a kill is formally off and claim 2 has a small low-budget edge for cheap features).

**Kill threshold:** both CIs include 1.0 (the 10% edge is pool noise; with every headline 25% and 50% ratio at T<=0.5 already <= 1.03, L1a converts to a kill for claim 2 at T<=0.5).

**Expected runtime:** about 5 minutes CPU (the L1a bootstrap of 50 x 2 seeds took under 10 seconds per cell).

**Informs:** claim 2. Prior (judgment): 60% that the T=0.25 CI excludes 1.0 (the 25% bootstrap implies a pool-noise SD of about 3 errors against a +7.9 gap), about even odds at T=0.5.

### L2b: Can a fixed rule reproduce the learned edge? (CPU)

**Hypothesis:** the warm learner's gain over the plain agreement threshold at low budgets and T=0.25 comes from length and truncation statistics (E21: +0.039 confident-subset AUC over agreement), so a label-free fixed tie-break reproduces it, which would falsify claim 2 as worded ("any fixed rule").

**Change vs existing scripts:** copy exp/sim_warm.py; add label-free policies `fixed_len` (agreement ascending, n_distinct descending, mean_tokens descending, random last) and `fixed_trunc` (any_truncated first, then fixed_len), plus the E21 `static` logistic rule as an unequal-budget comparator only (free labels from the held-out quarter of cells).

**Command shape:** the same 12-job xargs -P 6 loop as exp/L1a-warm-start-replay.sh with the new script.

**Metric:** fixed_len / greedy_warm and fixed_trunc / greedy_warm at 10% and 25% in cell100 for T=0.25 and T=0.5, cheap features, with paired seed SD.

**Pass threshold (claim 2 survives):** no label-free fixed rule reaches 0.98x of greedy_warm at both 10% and 25% at both T=0.25 and T=0.5.

**Kill threshold:** a label-free fixed tie-break reaches >= 0.99x of greedy_warm at 10% and 25% at both temperatures (the learner's edge is a better fixed tie-break; claim 2 is false as worded on this task).

**Expected runtime:** about 5 minutes CPU.

**Informs:** claim 2 in its exact wording. Prior (judgment): 60% that a length-based tie-break closes at least half of the 25% gain at T=0.25.

### L2c: Calibrated Thompson prior for the layout arm (CPU)

**Hypothesis:** ts_warm's weakness in L1a (below greedy_warm in all 18 headline cell-budget combinations, 0.69x to 0.98x of the threshold) is partly an unfair handicap, because its cell prior is a rank-score mean on a 0.6 to 1.0 scale against a 13.7% base rate. With a calibrated prior, embedding-cell structure may still add nothing.

**Change vs existing scripts:** copy exp/sim_warm.py; in run_online(warm=True, ts) set bprior from the per-cell mean of a calibrated hit-rate estimate (the rank score mapped to the empirical error rate of the threshold's first `budget` picks, or the static classifier probability from the disjoint quarter of cells, rescaled to the 13.7% base rate), with the TS prior strength constant (10) unchanged; everything else as L1a.

**Command shape:** the same 12-job loop as L1a with the new script.

**Metric:** ts_warm_cal / greedy_warm and ts_warm_cal / threshold at 10/25/50%, cell100, cheap and cheap+emb.

**Pass threshold (layout adds value):** ts_warm_cal / greedy_warm >= 1.03 at 25% at T=0.25 or T=0.5 in cell100 cheap.

**Kill threshold:** ts_warm_cal / greedy_warm < 1.0 in all of those cells (embedding-cell structure is unused even with a fair prior; claim 3 stays dead on GSM8K with majority vote).

**Expected runtime:** about 3 minutes CPU.

**Informs:** claim 3, by removing the one known unfair handicap on the layout arm. Moderate value, since E01 and E21 neighbor ratios already sit at their permutation null; run it in the same session as L2a.

### Iteration 3 (proposed after L2a-pool-bootstrap-10pct, 2026-10-18; verdict kill by the pre-registered percentile CI, not robust to a basic-bootstrap CI at T=0.25 and T=1.0)

Iteration 3 = the loop run that executed L2a. L2a found no pool-robust edge for the warm learner over the label-free agreement threshold at 10% on cell100 cheap (T=0.25 ratio 0.999, CI [0.837, 1.134]; T=0.5 ratio 1.064, CI [0.980, 1.210]), and the pool SD of the ratio (0.057 to 0.078 at 10%, 0.016 to 0.028 at 25%) means a true +6% edge is undetectable on GSM8K even with every prompt. So learner-side variants stop being informative: L2b (fixed_len tie-break) and L2c (calibrated TS prior) are demoted, and L1c (powered T=0.25 rerun) is not worth it for claim 2. The live question is which information source can capture the oracle headroom that cheap features and MiniLM prompt embeddings leave uncaptured. L3a and L3b need the GPU (one A4000, 45 minute cap, one job at a time); L3c is CPU only. All replays should use exp/sim_warm_boot.py (B=200 pool bootstrap, percentile 95% CI as the verdict, basic CI reported beside it).

### L3a: Policy hidden-state probe on the saved E21 T=0.25 rollouts, then equal-budget replay (GPU forward pass, CPU analysis)

**Hypothesis:** the confident errors that carry the oracle headroom (the warm learner closes only -2% to +8% of the threshold-to-oracle gap at 25%, so over 90% is uncaptured) are visible in the policy's own hidden state, which rollout statistics (agreement, n_distinct, token length, truncation) and MiniLM prompt embeddings do not expose. If so a learned allocator has an input a fixed agreement rule lacks, which is the only offline route left for claim 2. The feature is free at training time, since GRPO already runs the forward pass that produces it.

**Change vs existing scripts:** new script exp/l3a_hidden_probe.py (does not exist): HF transformers forward pass (not vLLM) of /homes/rdubash/avrl_live/FULL/models/Qwen2.5-1.5B-Instruct over question plus the majority-answer completion of each of the 2,000 groups (text saved in runs/E21-lowT-confident-errors/T0.25/rollouts.parquet), save the last-token hidden state of a late layer; PCA to 16 dimensions fit on the fit-cells quarter only (the same held-out-cells split as sim_warm.py); add columns hs1..hs16 to a copy of feats_T0.25.parquet. Stage 1 (ceiling, free labels, CPU): confident-subset and overall AUC of cheap + hs vs cheap, cross-fit by cell as in E21. Stage 2 (only if stage 1 clears): sim_warm_boot.py with the hs columns appended to the cheap list, refit on audited labels only, equal budget.

**Command shape:** `/homes/rdubash/avrl_live/FULL/bin/gpu_run.sh /homes/rdubash/avrl_live/FULL/avrl/runs/L3a-hidden-probe '/homes/rdubash/active-audit/.venv-judge/bin/python exp/l3a_hidden_probe.py --model /homes/rdubash/avrl_live/FULL/models/Qwen2.5-1.5B-Instruct --rollouts /homes/rdubash/avrl_live/FULL/avrl/runs/E21-lowT-confident-errors/T0.25/rollouts.parquet --feats /homes/rdubash/avrl_live/FULL/avrl/runs/E21-lowT-confident-errors/feats_T0.25.parquet --out /homes/rdubash/avrl_live/FULL/avrl/runs/L3a-hidden-probe'`, then the CPU stages in the shell with /homes/rdubash/active-audit/.venv/bin/python. Time the first 200 sequences and abort to a subsample if the projection passes 35 minutes.

**Metric:** confident-subset delta AUC (cheap + hs minus cheap) with fold SD; at stage 2, greedy_warm(cheap + hs) / agree_thr with percentile and basic 95% CI at 10% and 25%, and the closed fraction of the threshold-to-oracle gap.

**Pass threshold:** confident-subset delta AUC >= 0.05 AND the stage 2 CI lower end above 1.0 at 25% (detectable: the 25% SD of the ratio is 0.016 to 0.028).

**Kill threshold:** confident-subset delta AUC < 0.02 (no information source the policy exposes, rollout statistics, prompt embeddings or hidden state, beats agreement on GSM8K with majority vote; claim 2 offline then needs a different task or proxy).

**Expected runtime:** UNVERIFIED 5 to 10 minutes GPU (about 2,000 sequences of roughly 350 tokens, about 0.7M tokens through a 1.5B model in bf16), plus about 5 minutes CPU.

**Informs:** claim 2 (is there any input the learner can use that a fixed rule cannot), and claim 3 indirectly (internal-state structure instead of prompt-space structure). Prior (judgment): 30% pass.

### L3b: Judge proxy on the saved E21 rollout text, rescoped from L1b (GPU)

**Hypothesis:** as L1b: with a learned judge as the proxy (no free agreement signal) proxy errors cluster in prompt space more than majority-vote errors do (E21: none), and judge errors overlap only partly with majority-vote errors; plus, new here, a judge-score threshold and an agreement threshold combined by a fixed rank blend may already match any learner (the "any fixed rule" wording of claim 2).

**Change vs existing scripts:** as L1b (new exp/l1b_judge_score.py, Qwen3-4B Yes/No logprob judge, tau set so the accept rate matches rollout accuracy), with three changes. (1) Score only the majority-answer completion and one other sampled completion per group (4,000 of the 16,000 rows) and define the group label as judge accepts the majority completion and gold is wrong; this cuts the prefill about 4x against the registered L1b and removes most of its 45 minute risk. (2) Baselines: judge-score threshold, agreement threshold, and a fixed 50/50 rank blend of the two. (3) Replay through sim_warm_boot.py with B=200 and a verdict on the percentile CI lower end against the best fixed rule, not on point-estimate ratios.

**Command shape:** as L1b: `/homes/rdubash/avrl_live/FULL/bin/gpu_run.sh /homes/rdubash/avrl_live/FULL/avrl/runs/L3b-judge '/homes/rdubash/active-audit/.venv-judge/bin/python exp/l1b_judge_score.py --model /homes/rdubash/avrl_live/FULL/models/Qwen3-4B --rollouts /homes/rdubash/avrl_live/FULL/avrl/runs/E21-lowT-confident-errors/T0.25/rollouts.parquet --per_group 2 --out /homes/rdubash/avrl_live/FULL/avrl/runs/L3b-judge'` (the script is new, so `--per_group` is part of it), timing probe on the first 200 rows, then the CPU analysis.

**Metric:** judge-proxy error rate overall and judge-confident subset; AUC of judge score alone, cheap, cheap + emb; confident neighbor ratio at cell100 vs its permutation null; Jaccard of judge-wrong and majority-vote-wrong groups; replay ratios at 10% and 25% of greedy_warm and of each fixed rule to the best fixed rule, with percentile CI.

**Pass threshold:** E04's for claim 3 (delta AUC cheap + emb minus cheap >= 0.05 AND Jaccard < 0.5 AND confident neighbor ratio >= 1.5 above null p95); for claim 2, the greedy_warm CI lower end above 1.0 against the best fixed rule at 25%.

**Kill threshold:** E04's (Jaccard > 0.8, or delta AUC < 0.03 with neighbor ratio < 1.3), which closes claim 3 for both proxy types tried; claim 2 offline closes if the best fixed rule has ratio >= 0.98 to greedy_warm at 25%.

**Expected runtime:** UNVERIFIED 10 to 15 minutes GPU for 4,000 rows through a 4B bf16 model, plus about 5 minutes CPU.

**Informs:** claim 3 (the only remaining untested regime where the proxy has no free confidence signal) and claim 2 against the strongest label-free fixed rule. Prior (judgment): 15% pass for claim 3.

### L3c: Subsampling without replacement to settle the bootstrap method dependence (CPU)

**Hypothesis:** L2a's percentile bootstrap shifts toward 1.0 relative to the full-pool point at T=0.25 and 10% (bootstrap mean 0.999 vs 1.084, 1.1 bootstrap SD) and away from the full-pool point at T=1.0 and 10% (0.912 vs 0.853), because resampling with replacement duplicates rows (63% unique) and changes how the tie-heavy agreement ordering and the refit behave. If so, the basic-bootstrap intervals (T=0.25 10% [1.034, 1.332], T=1.0 10% [0.682, 0.945]) are the better ones and L2a's T=0.25 kill and its high-temperature flag are method artifacts. If instead the full-pool 1.084 was a favourable draw, subsamples of the same pool will center near 1.0.

**Change vs existing scripts:** copy exp/sim_warm_boot.py to exp/sim_warm_sub.py; replace the with-replacement resample by `rng.choice(n, m, replace=False)` for m = n/2 (773 rows) and m = 3n/4 (1,159 rows), budget int(frac * m); keep policies agree_thr, greedy_warm, greedy_blend and both gates (gate 2 on the full pool unchanged).

**Command shape:** the same xargs -P 6 loop as exp/L2a-pool-bootstrap-10pct.sh with the new script and an env SUB_FRAC (0.5, 0.75): `SUB_B=200 SUB_SEEDS=4 SUB_M=0.5 BOOT_FRAC=0.1 /homes/rdubash/active-audit/.venv/bin/python exp/sim_warm_sub.py runs/E21-lowT-confident-errors/feats_T$T.parquet proxy_error cell100 agreement,n_distinct,mean_tokens,std_tokens,any_truncated,log_qlen runs/L3c-subsample/sub_T${T}_m${M}_f${F}.json` for T in 0.25, 0.5, 1.0, M in 0.5, 0.75, F in 0.1, 0.25 (12 jobs, each under a minute).

**Metric:** subsample mean and SD of the paired ratio per cell, and a subsampling CI centered on the full-pool ratio with the SD scaled by sqrt(m / n).

**Pass threshold:** subsample mean ratio >= 1.05 at T=0.25 and 10% (m = n/2) AND the subsampling CI lower end above 1.0 (the L2a T=0.25 kill is a method artifact; relabel L2a mixed and report claim 2 as a small low-budget edge for cheap features).

**Kill threshold:** subsample mean ratio <= 1.02 at T=0.25 and T=0.5 at 10% (L1a's 1.084 and 1.061 were favourable draws; the L2a kill holds under both interval types).

**Expected runtime:** about 3 minutes CPU (200 resamples x 4 seeds took under 2 minutes for 8 jobs in L2a).

**Informs:** the wording of the claim 2 verdict (kill vs power-limited tie) and the high-temperature kill-side flag, not the plan. Prior (judgment): 50% that the half-pool mean at T=0.25 and 10% is >= 1.04. Run it alongside L3a; it needs no GPU.

### Iteration 4 (proposed after L3a-hidden-state-probe, 2026-10-18; verdict inconclusive, power-limited)

Iteration 4 = the loop run that executed L3a. L3a (RESEARCH_LOG.md) found that 16 PCA dims of the layer-20 last-completion-token state lift confident-subset cross-fit AUC for proxy error by +0.028 over the cheap features (fold SD 0.039, bootstrap CI [-0.002, +0.068]; the pre-registered pass bar is +0.05 with CI lower end above 0; kill needs delta below 0.02 and CI upper end below 0.05, so neither is met), against +0.010 for the dimension-matched MiniLM control. At the allocation level the free-label ceiling shows no capture (cheap + hsA20 static is 1.08x the agreement threshold at 10% and 1.04x at 25%, and the cheap-only static is 1.04x and 1.06x). Stage 2 was not run. A post-hoc residual-shuffle null says 16 unsupervised noise columns cost about 0.025 AUC at 94 errors, so the 16-PC probe wastes information; the best single estimate of the true primary delta is +0.03 to +0.035. L4a to L4c target that: a better-conditioned probe on the same states (CPU), label-directed scalars instead of PCA dims (GPU), and, only if one of them shows a point delta of +0.035 or more, a powered sample (GPU). L3b (judge proxy) stays queued from Iteration 3 and is the next GPU pick if L4a and L4b both land below +0.03. All replays still use exp/sim_warm_boot.py (B=200 pool bootstrap, percentile 95% CI as the verdict, basic CI beside it).

### L4a: Supervised pooled-label probe on the saved L3a hidden states (CPU only)

**Hypothesis:** the information is in the policy's hidden state but a top-16 unsupervised PCA block fit on 94 confident errors wastes it (L3a post-hoc null: noise columns cost about 0.025 AUC, hidden-state blocks clear that null with P below 0.005). A regularized probe on the full 1536-d state, trained on the labels of all 2,000 groups (285 errors, 3x the positives) and scored only on the confident subset, lifts cross-fit AUC for proxy error over cheap by at least 0.05.

**Change vs existing scripts:** new script exp/l4a_supervised_probe.py, reading runs/L3a-hidden-state-probe/hs.npz and runs/E21-lowT-confident-errors/feats_T0.25.parquet; no PCA. Primary set = cheap + the standardized full hsA20 state (layer 20, position A, as registered in L3a), logistic regression with L2 and class_weight balanced, C chosen by an inner 3-fold CV from the fixed grid {0.0003, 0.001, 0.003, 0.01} (grid fixed here, no other tuning); control = cheap + MiniLM pc0..15 refit the same way on the pooled labels; exploratory = hsP14, hsA14, hsA28, hsP20 (not counted toward the pass). Outer CV: RepeatedStratifiedKFold 5 x 3 on all 2,000 groups stratified by label (random_state 0), train on pooled labels, report the AUC on the held-out confident-subset rows only (94 errors), delta over cheap trained the same way, 1,000-resample paired bootstrap of the averaged out-of-fold predictions. Include agreement in cheap so a gain cannot be the probe re-learning agreement.

**Command shape:** `cd /homes/rdubash/avrl_live/FULL/avrl && OMP_NUM_THREADS=4 /homes/rdubash/active-audit/.venv/bin/python exp/l4a_supervised_probe.py --run_dir runs/L3a-hidden-state-probe --e21 runs/E21-lowT-confident-errors --out runs/L4a-supervised-probe`, with a reproduction gate (the cheap-only confident AUC must be within 0.01 of E21's 0.8067 when trained on confident rows only, to show the pooled-label cheap model is a fair baseline).

**Metric:** confident-subset delta AUC (primary minus cheap, both trained on pooled labels) with fold SD and paired bootstrap 95% CI; the same for the MiniLM control and the exploratory sets; overall-subset delta as a secondary.

**Pass threshold:** primary delta AUC >= 0.05 with bootstrap CI lower end > 0 (then stage 2 as in L3a, with the supervised-probe score appended and refit on audited labels only).

**Kill threshold:** primary delta < 0.02 AND CI upper end < 0.05 AND no exploratory variant >= 0.05; with L3a this would close the hidden-state route for claim 2 offline on GSM8K with majority vote. Anything else is inconclusive (power-limited).

**Expected runtime:** about 5 minutes CPU (1536-d logistic on 2,000 rows, 15 outer x 3 inner folds x 4 values of C, UNVERIFIED).

**Informs:** claim 2 (is the hidden-state information capturable with a conditioned probe) and whether L4c is worth its 25 minutes of GPU. Prior (judgment): 25% that the point delta reaches 0.05, 15% that the CI lower end also clears. Caveat: pooled labels come mostly from non-confident groups where agreement already explains the error, so the probe may add little on the confident subset; the cheap baseline is trained on the same pooled labels for that reason.

### L4b: Policy-confidence scalars of the majority completion (GPU forward pass)

**Hypothesis:** the policy's own token-level confidence on the majority-answer completion separates the confident-but-wrong groups better than agreement alone, and, because it is 3 to 4 label-directed scalars rather than 16 PCA dims, it avoids the overfitting cost seen in L3a and the format-dependence of the last-text-token position (only 10.8% of majority completions end in "#### <number>"). Delta AUC over cheap on the confident subset reaches 0.05.

**Change vs existing scripts:** new script exp/l4b_policy_confidence.py, a copy of exp/l3a_hidden_probe.py that loads AutoModelForCausalLM (with the LM head), keeps the same inputs (chat-template prompt + the first proxy==1 completion, same gates a to c), and computes with log_softmax only at completion positions (batch 4, so the logits stay under about 2 GB): mean token log-probability, minimum token log-probability, log-probability and entropy at the final answer-number token (located by the same answer parser the rollout analysis used). Writes 4 columns lp_mean, lp_min, lp_ans, ent_ans into a copy of feats_T0.25.parquet. Stage 1 as in L3a (E21 code path, same folds, reproduction gate, bootstrap) with the sets cheap, cheap + 4 scalars (primary), cheap + MiniLM (control), cheap + 4 scalars + hsA20 (exploratory).

**Command shape:** `cd /homes/rdubash/avrl_live/FULL/avrl && /homes/rdubash/avrl_live/FULL/bin/gpu_status.sh && /homes/rdubash/avrl_live/FULL/bin/gpu_run.sh /homes/rdubash/avrl_live/FULL/avrl/runs/L4b-policy-confidence '/homes/rdubash/active-audit/.venv-judge/bin/python exp/l4b_policy_confidence.py --model /homes/rdubash/avrl_live/FULL/models/Qwen2.5-1.5B-Instruct --rollouts /homes/rdubash/avrl_live/FULL/avrl/runs/E21-lowT-confident-errors/T0.25/rollouts.parquet --groups /homes/rdubash/avrl_live/FULL/avrl/runs/E21-lowT-confident-errors/T0.25/groups.parquet --out /homes/rdubash/avrl_live/FULL/avrl/runs/L4b-policy-confidence'`, then the CPU stage 1 with the .venv python.

**Metric:** confident-subset delta AUC with fold SD and paired bootstrap 95% CI; AUC of each scalar alone (overall and confident); free-label ceiling in errors caught at 10% and 25% via sim_e21.py static with the 4 columns.

**Pass threshold:** primary delta >= 0.05 with CI lower end > 0 (then stage 2 via sim_warm_boot.py as in L3a).

**Kill threshold:** primary delta < 0.02 AND CI upper end < 0.05; combined with L3a and L4a this closes every policy-exposed information source on this task.

**Expected runtime:** about 5 minutes GPU (one 1.5B forward pass over 0.64M tokens took 45 s in L3a; the head adds a little; UNVERIFIED) plus 2 minutes CPU.

**Informs:** claim 2 (a cheaper, better-conditioned policy-side signal than hidden states). Prior (judgment): 15% pass. Run order: after L4a, which costs no GPU.

### L4c: Powered extension of the confident-error sample, conditional (GPU, then CPU)

**Hypothesis:** the L3a/L4a/L4b deltas are real but below the power of 94 confident errors (bootstrap SD of the delta 0.0177 at 94 errors; separating a true +0.05 from +0.02 needs about 215 to 360). With about 250 confident errors the SD falls to about 0.011 and the registered bins become decidable; a larger pool also lets the stage 2 replay run on about 4,900 groups instead of 1,546, shrinking the L2a ratio SD by about sqrt(3).

**Run only if** L4a or L4b gives a primary point delta >= 0.035 (otherwise the information is probably too small to be worth 25 minutes of GPU and the hidden-state route is closed by L3a, L4a and L4b together).

**Change vs existing scripts:** new exp/l4c_rollouts_ext.py, a copy of exp/e21_lowT_rollouts.py that reads data/gsm8k_train.parquet directly for pids 2000 to 5399 (the E21 script is tied to runs/p0/groups.parquet, which holds pids 0..2999), single temperature 0.25, G = 8, max_tokens 512, seed 0, the same SYS string and answer parser, saving text. Then the L3a probe (and the L4b scalar script if it passed) on the extension through a copy of exp/l3a_hidden_probe.py with a --pid_offset argument (the original asserts pids 0..n-1). Analysis pools pids 0..5399 with a fixed fit/sim split by question hash; the registered pass and kill bins are re-applied unchanged.

**Command shape:** two gpu_run.sh jobs, one at a time, each under the 45 minute cap: `/homes/rdubash/avrl_live/FULL/bin/gpu_run.sh /homes/rdubash/avrl_live/FULL/avrl/runs/L4c-rollouts '/homes/rdubash/active-audit/.venv-judge/bin/python exp/l4c_rollouts_ext.py --temps 0.25 --pid_start 2000 --pid_end 5400 --out /homes/rdubash/avrl_live/FULL/avrl/runs/L4c-rollouts'`, then the probe job on its output, then the CPU analysis.

**Metric:** as L3a stage 1 on the pooled set (about 254 confident errors), then stage 2 (greedy_warm / agree_thr at 25% and 10%, B=200, percentile and basic CI) only if stage 1 clears.

**Pass threshold:** as L3a (delta AUC >= 0.05 with CI lower end > 0, and the 25% percentile CI lower end of greedy_warm(cheap + probe) / agree_thr above 1.0).

**Kill threshold:** as L3a (delta < 0.02 and CI upper end < 0.05 and no exploratory variant >= 0.05), now decidable at the 0.011 SD.

**Expected runtime:** UNVERIFIED about 20 minutes GPU for 27,200 rollouts (scaled from E21's 1,428 s for 32,000 rollouts), about 2 minutes GPU for the probe, about 10 minutes CPU.

**Informs:** claim 2, with enough power to settle it for hidden-state or confidence signals. Prior (judgment): 30% pass given the trigger; about 8% unconditional.

### Iteration 5 (proposed after L3b-judge-proxy, 2026-10-18; verdict inconclusive for claims 3 and 2)

Iteration 5 = the loop run that executed L3b. L3b (RESEARCH_LOG.md, entry plus critic note) used the saved E21 T=0.25 rollouts with a Qwen3-4B Yes/No judge as the proxy (1,840 of 2,000 groups accepted, 175 judge-accepted gold-wrong groups, Jaccard with majority-vote errors 0.614). Claim 3: MiniLM adds +0.0074 AUC over the strongest cheap baseline JA (bootstrap CI [-0.0096, +0.0233], stable across CV partitions at +0.001 to +0.007), so the AUC route is closed at this power; the confident neighbor ratio at cell100 is 2.093 against a null p95 of 1.41 to 1.45 on the 49 errors of the top half by judge score (cell bootstrap CI [1.47, 3.01]), but only 1.066 over all of ACC and absent at cell30. Neither pre-registered bin is met (pass needs delta >= 0.05 and Jaccard < 0.5, kill needs neighbor ratio < 1.3). Claim 2: against the registered fixed set the best fixed rule was blend50 (112.9 errors at 25%) and greedy_warm / blend50 was 1.079 with percentile CI [0.947, 1.181]; the critic note found an unregistered fixed rule, agree_acc (agreement among judge-accepted groups, rejected last), that catches 125.6 at 25% and puts greedy_warm at 0.969 [0.860, 1.022] of it. The three items below supersede the L5a, L5b, L5c list in the L3b entry. Machine: all three run on the single A4000 box under the 45 minute cap, L5a and L5b are CPU only.

### L5a: Corrected-baseline replay of the judge-proxy pool (CPU only)

**Hypothesis:** the learner's apparent lead in L3b (greedy_warm 1.079x, greedy_blend 1.208x the registered best fixed rule at 25%) comes from a weak fixed-rule set. With agree_acc in the set and the learner warm-started from the best fixed rule, no learner arm beats the best fixed rule with a CI above 1.0.

**Change vs existing scripts:** copy exp/sim_warm_boot_j.py to exp/sim_warm_boot_j2.py. Add two fixed policies to orders(): agree_acc = lexsort((u1, -ndist, agree, ~acc)) (accepted groups by agreement ascending then n_distinct descending, rejected last, random tie-break) and blend50_acc (0.5 x rank of agree_acc + 0.5 x rank of judge_thr). FIXED = (agree_thr, judge_thr, blend50, agree_acc, blend50_acc); best fixed = highest full-pool mean at 25% over 8 seeds. greedy_warm and greedy_blend warm-start from the best fixed ordering (keep the norefit-equals-warm-rule gate per seed and on the first 20 resamples). Keep the resample generator default_rng(12345) so the stored arms are paired. Register the pass and kill bins before running.

**Command shape:** `cd /homes/rdubash/avrl_live/FULL/avrl && FEATS=s_maj,s_oth,s_diff,mean_tokens,std_tokens,any_truncated,log_qlen,agreement,n_distinct; R=runs/L3b-judge-proxy; BOOT_B=200 BOOT_SEEDS=4 BOOT_FRAC=0.25 /homes/rdubash/active-audit/.venv/bin/python exp/sim_warm_boot_j2.py $R/feats_J.parquet judge_error cell100 $FEATS runs/L5a-corrected-baseline/boot_f0.25.json`, then the same with BOOT_FRAC=0.1 and with POOL=held BOOT_FRAC=0.25 (three jobs, OMP_NUM_THREADS=2 each, run in parallel on CPU).

**Metric:** errors caught at 10% and 25% for every policy; greedy_warm / best fixed and greedy_blend / best fixed with percentile CI (verdict) and basic CI (beside it); H = (oracle - best fixed) / oracle.

**Pass threshold:** greedy_blend / best fixed percentile CI lower end above 1.0 at 25% on the 2,000-group pool AND point ratio at least 1.05 on the held-out 1,546-group pool.

**Kill threshold:** best fixed at least 0.98 x greedy_warm AND at least 0.98 x greedy_blend at 25% (full-pool point estimates) and both percentile CIs include 1.0. Anything else is inconclusive.

**Expected runtime:** about 5 to 8 minutes CPU (the three L3b replay files took about 3 minutes in parallel; two more fixed arms add little).

**Informs:** claim 2 in the judge regime and the wording of the structure-versus-gain curve. Prior (judgment): 80% kill, 10% pass. Highest information per cost in the iteration because it settles whether any learner arm is alive offline.

### L5b: Is the judge-error clustering usable? Cell-rate feature and cell-prior rerank (CPU only)

**Hypothesis:** the cell100 clustering of judge-accepted errors among the judge-confident half (ratio 2.09, cell bootstrap CI [1.47, 3.01]) carries predictive information that 16 MiniLM PCs cannot extract linearly, because it is a local, cell-level effect. Feeding the cell's error rate in directly lifts AUC over JA and lifts the equal-budget replay over agree_acc.

**Change vs existing scripts:** new exp/l5b_cell_rate.py reading runs/L3b-judge-proxy/feats_J.parquet. Feature: leave-one-out shrunk cell error rate, (errors among other members of the cell in the training fold + 5 x base rate) / (other training members + 5), for cell100 and cell300, computed inside each CV fold from training labels only (test rows get the rate from training rows of their cell). CV = the L3b code path (ACC only, RepeatedStratifiedKFold 5 x 3, random_state 0, StandardScaler + balanced logistic, paired bootstrap of 1,000 resamples). Sets: JA, JA + cell100 rate, JA + cell300 rate, JA + both. Replay arm: copy exp/sim_warm_boot_j2.py (from L5a) and add cell_rerank = agree_acc order, but groups in the top half by s_maj are re-sorted within the first 2 x budget positions by the audited error count of their cell100 (ties by agree_acc order), updated online. Same resamples as L5a.

**Command shape:** `cd /homes/rdubash/avrl_live/FULL/avrl && OMP_NUM_THREADS=8 /homes/rdubash/active-audit/.venv/bin/python exp/l5b_cell_rate.py --run_dir runs/L3b-judge-proxy --out runs/L5b-cell-rate` then the replay jobs as in L5a with the extra arm.

**Metric:** AUC and delta over JA with fold SD and paired bootstrap 95% CI (ACC, and the top half by s_maj as a secondary); replay errors caught at 10% and 25% for cell_rerank vs agree_acc with percentile CI.

**Pass threshold:** delta AUC (best cell set minus JA) at least 0.03 with bootstrap CI lower end above 0 AND cell_rerank / agree_acc percentile CI lower end above 1.0 at 25%.

**Kill threshold:** delta AUC below 0.02 AND CI upper end below 0.05 for all cell sets (the clustering exists but cannot be used by a supervised or online prior at 175 errors).

**Expected runtime:** about 10 minutes CPU.

**Informs:** claim 3, the one live lead. Prior (judgment): 15% pass, 50% kill. Caveat: the top-half subset has only 49 errors, so the AUC test is run on all of ACC (175 errors, bootstrap SD of the delta about 0.009) and will dilute a structure that exists only in the judge-confident half.

### L5c: Second and third proxy on the same 4,000 rows: a reward model and a different judge (GPU, then CPU)

**Hypothesis:** the top-half cell100 clustering (2.09; 1.76 with the run 1 prompt) reflects blind spots of the Qwen3-4B judge on certain problem types, so it either replicates across proxies (structure in the problems) or does not (structure in the judge). Either answer populates the structure-versus-gain curve with a proxy type RLHF actually uses.

**Change vs existing scripts:** copy exp/l3b_judge_score.py to exp/l5c_proxy_score.py with a --kind argument: (i) rm = Skywork-Reward-V2-Qwen3-0.6B from /homes/rdubash/active-audit/models via AutoModelForSequenceClassification (scalar logit, chat template on problem + solution; loading is UNVERIFIED, abort and log if the head does not load); (ii) judge = Qwen2.5-3B-Instruct Yes/No logit, same prompt and gates as L3b run 2 (unbatched, mass above 0.9, sanity AUC above 0.70 else failed experiment). Same 4,000 rows (majority completion plus label-free other completion). RM has no natural zero, so accept = score above the quantile that makes the accept rate equal the majority-vote accuracy (85.75%). CPU analysis: copy exp/l3b_analyze.py with the feature columns renamed, run for each proxy, with agree_acc added to the replay.

**Command shape:** `cd /homes/rdubash/avrl_live/FULL/avrl && /homes/rdubash/avrl_live/FULL/bin/gpu_status.sh && /homes/rdubash/avrl_live/FULL/bin/gpu_run.sh /homes/rdubash/avrl_live/FULL/avrl/runs/L5c-proxies '/homes/rdubash/active-audit/.venv-judge/bin/python exp/l5c_proxy_score.py --kinds rm,judge3b --rollouts /homes/rdubash/avrl_live/FULL/avrl/runs/E21-lowT-confident-errors/T0.25/rollouts.parquet --groups /homes/rdubash/avrl_live/FULL/avrl/runs/E21-lowT-confident-errors/T0.25/groups.parquet --out /homes/rdubash/avrl_live/FULL/avrl/runs/L5c-proxies'`, then the CPU analysis per proxy.

**Metric:** per proxy: n_err in ACC (at least 100 needed for any kill), sanity AUC, Jaccard with majority-vote errors, delta AUC (JA + emb minus JA) with bootstrap CI, top-half cell100 neighbor ratio against its 1,000-draw permutation null and a cell bootstrap CI, replay vs agree_acc.

**Pass threshold:** top-half cell100 neighbor ratio at least 1.5 and above null p95 for BOTH new proxies AND delta AUC at least 0.03 with CI lower end above 0 for at least one.

**Kill threshold:** top-half cell100 neighbor ratio below 1.3 for both new proxies (the Qwen3-4B clustering is judge-specific, claim 3 closes for every proxy tried). Anything else is inconclusive.

**Expected runtime:** UNVERIFIED about 8 minutes GPU (L3b scored 4,000 rows with a 4B model in 326 s; the 0.6B and 3B models are smaller or similar) plus 2 minutes CPU.

**Informs:** claim 3 across proxy types and the thesis fallback curve. Prior (judgment): 40% that the neighbor ratio replicates for at least one proxy, 10% for the full pass. Queued behind L5a and L5b because they cost no GPU; L4a and L4b from iteration 4 remain queued.

### Iteration 6 (proposed after L5c-proxies, 2026-10-18; verdict inconclusive for claim 3, split: judge3b replicates, rm does not; claim 2 secondary mixed)

Iteration 6 = the loop run that executed L5c. L5c scored the same 4,000 E21 T=0.25 rows with Skywork-Reward-V2-Qwen3-0.6B (accept = top 1,715 by score, 175 accepted-wrong groups) and Qwen2.5-3B-Instruct Yes/No (accept s > 0, 218 accepted-wrong groups); both passed all gates. Top-half cell100 neighbor ratio: rm 1.237 (null p95 1.295, cell-bootstrap CI [0.81, 1.68], kill-side), judge3b 1.671 (p95 1.231, CI [1.27, 2.17], replicates), Qwen3-4B reference 2.093; the registered bins give inconclusive at the top-half cut and the verdict flips to kill at top75 and all-of-ACC and to inconclusive at top25 (both ratios 2.05 and 2.11). The AUC route is dead on all three proxies (delta over JA +0.003 to +0.007, CI upper ends at most 0.025). The accepted-wrong sets overlap 0.69 to 0.73 across proxies (138 of 235 accepted by all three) and all proxies are Qwen-family, so the three proxies are not independent replications. Claim 2: agree_acc is the best fixed rule in all three pools; greedy_warm is 0.87 to 0.95 of it; greedy_blend is 1.046 [0.980, 1.090] (rm), 1.009 [0.988, 1.058] (judge3b) and 1.111 [1.039, 1.204] (Qwen3-4B). L5a and L5b are still queued and are narrowed by the two items below that supersede them. Machine: L6a and L6b are CPU only, L6c needs the single A4000 under the 45 minute cap.

### L6a: Is the clustering usable? Cell-rate feature and cell-prior rerank on all three pools (CPU only; supersedes L5b)

**Hypothesis:** the cell100 clustering of accepted-wrong groups among the proxy-confident half (2.09, 1.67, 1.24 for Qwen3-4B, Qwen2.5-3B, rm) carries predictive information that 16 MiniLM PCs cannot extract linearly because it is a local, cell-level effect. Feeding the cell's error rate in directly, computed from training labels only, lifts AUC over JA and lifts the equal-budget replay over agree_acc on at least two of the three pools.

**Change vs existing scripts:** new exp/l6a_cell_rate.py (L5b's design, taking the pool directory as an argument), reading <pool>/feats_J.parquet for pool in runs/L5c-proxies/rm, runs/L5c-proxies/judge3b and runs/L3b-judge-proxy. Feature: leave-one-out shrunk cell error rate, (errors among the other training-fold members of the cell + 5 x base rate) / (other training-fold members + 5), for cell100 and cell300, computed inside each CV fold from training labels only (test rows take the rate from the training rows of their cell). Sets: JA, JA + cell100, JA + cell300, JA + both, same 5 x 3 folds (RepeatedStratifiedKFold random_state 0) and paired 1,000-resample bootstrap as exp/l5c_analyze.py; AUC on ACC and, secondary, on the top half by s_maj. Add a consensus label: accepted by all three proxies and gold-wrong (138 errors, consensus-accepted groups 1,593; top half by mean rank of s_maj is a post hoc cut, report only). Replay arm cell_rerank in a copy of exp/sim_warm_boot_j2.py (exp/sim_warm_boot_j3.py): agree_acc order, then groups in the proxy-confident half re-ranked by the count of audited errors in their cell100 (audit-time labels only), against agree_acc, same default_rng(12345) resamples, B = 200 x 4 seeds, 25% budget. Register the bins in the docstring and smoke-test on synthetic labels (null and planted cell effect) before reading real labels.

**Command shape:** `cd /homes/rdubash/avrl_live/FULL/avrl && for P in runs/L5c-proxies/rm runs/L5c-proxies/judge3b runs/L3b-judge-proxy; do OMP_NUM_THREADS=4 /homes/rdubash/active-audit/.venv/bin/python exp/l6a_cell_rate.py --pool $P --out runs/L6a-cell-rate/$(basename $P); done`, then the cell_rerank replay jobs in parallel (OMP_NUM_THREADS=2 each) as in L5c.

**Metric:** per pool, delta AUC (best cell set minus JA) with fold SD and paired bootstrap 95% CI on ACC and on the top half; replay errors caught at 25% for cell_rerank vs agree_acc with percentile CI (verdict) and basic CI (beside it).

**Pass threshold:** on at least two of the three pools, delta AUC at least 0.03 with bootstrap CI lower end above 0 AND cell_rerank / agree_acc percentile CI lower end above 1.0 at 25%.

**Kill threshold:** delta AUC below 0.02 AND CI upper end below 0.05 for every cell set on all three pools (the clustering exists but cannot be used by a supervised or online prior at 175 to 218 errors). Anything else is inconclusive.

**Expected runtime:** about 12 minutes CPU (UNVERIFIED; L5c's per-pool analysis took about 1 minute and the three replay files about 3 minutes in parallel).

**Informs:** claim 3, the one remaining descriptive lead, and whether L6c is worth 25 GPU minutes. Prior (judgment): 10% pass, 70% kill, 20% inconclusive. Caveat: the top-half subsets hold only 49 to 77 errors, so the AUC test runs on all of ACC and will dilute a structure that exists only in the confident half; the three pools share prompts and about 70% of their errors, so a pass on two pools is weaker than two independent passes.

### L6b: Unrun remainder of L5a on three pools: 10% budget and held-out pool (CPU only; supersedes L5a)

**Hypothesis:** greedy_blend's lead over agree_acc at 25% on the Qwen3-4B pool (1.111, percentile CI [1.039, 1.204], runs/L5c-proxies/qwen3_4b_ref/boot_f0.25.json) is a property of that pool, not of the arm: with the 10% budget and the held-out-cells pool, and on the rm and Qwen2.5-3B pools, no learner arm beats the best fixed rule with a CI above 1.0.

**Change vs existing scripts:** none (exp/sim_warm_boot_j2.py already exists and passed its norefit gates on three pools in L5c). Only the budget and pool variants are new: BOOT_FRAC=0.1 on the full pool and POOL=held BOOT_FRAC=0.25 (E21 held-out-cells quarter split, 1,546 groups), for each of the three pool files. Keep default_rng(12345) so arms are paired with the stored 25% files.

**Command shape:** `cd /homes/rdubash/avrl_live/FULL/avrl && FEATS=s_maj,s_oth,s_diff,mean_tokens,std_tokens,any_truncated,log_qlen,agreement,n_distinct; for P in L5c-proxies/rm L5c-proxies/judge3b L3b-judge-proxy; do N=$(basename $P); OMP_NUM_THREADS=2 BOOT_B=200 BOOT_SEEDS=4 BOOT_FRAC=0.1 /homes/rdubash/active-audit/.venv/bin/python exp/sim_warm_boot_j2.py runs/$P/feats_J.parquet judge_error cell100 $FEATS runs/L6b-remainder/$N/boot_f0.1.json & OMP_NUM_THREADS=2 POOL=held BOOT_B=200 BOOT_SEEDS=4 BOOT_FRAC=0.25 /homes/rdubash/active-audit/.venv/bin/python exp/sim_warm_boot_j2.py runs/$P/feats_J.parquet judge_error cell100 $FEATS runs/L6b-remainder/$N/boot_held_f0.25.json & done; wait` (six parallel CPU jobs; create the output directories first).

**Metric:** errors caught at 10% and, on the held-out pool, 25% for every policy; greedy_warm / best fixed and greedy_blend / best fixed with percentile CI (verdict) and basic CI (beside it); H = (oracle - best fixed) / oracle; the norefit gates must pass per file.

**Pass threshold:** for at least two of the three proxies, greedy_blend / best fixed percentile CI lower end above 1.0 at 25% on the 2,000-group pool (already met for Qwen3-4B, not met for rm and Qwen2.5-3B) AND point ratio at least 1.05 on the held-out pool.

**Kill threshold:** best fixed at least 0.98 x greedy_warm AND at least 0.98 x greedy_blend (full-pool point estimates) and both percentile CIs include 1.0, at 25% on at least two of the three pools. Anything else is inconclusive.

**Expected runtime:** about 6 minutes CPU (UNVERIFIED; three L5c replay files ran in about 3 minutes in parallel, now six).

**Informs:** claim 2 in the judge and reward-model regime, the last live lead for a learned arm. Prior (judgment): 20% pass, 45% kill, 35% inconclusive. Caveat: the pools share prompts and about 70% of their errors, so agreement across them is not independent confirmation; the best fixed rule is selected in-sample (agree_acc leads by 10 to 20% in every pool, so selection noise is not material).

### L6c: Fresh-prompt out-of-sample test of the cell structure (GPU, then CPU; run only if L6a does not hit its kill)

**Hypothesis:** the cell-level enrichment of proxy-accepted gold-wrong groups found on pids 0..1999 is a property of the problems and transfers to unseen prompts: with the pids 0..1999 KMeans100 cells held fixed (no refit), the top-half-by-s_maj neighbor ratio on pids 2000..3999 is at least 1.5 and above its null p95 for both rm and Qwen2.5-3B. This is the only test free of the forking paths of L5c (four cuts, three cell sizes, three proxies on one pool) and of the shared-pool dependence between proxies.

**Change vs existing scripts:** new exp/l6c_rollouts_fresh.py, a copy of exp/e21_lowT_rollouts.py reading data/gsm8k_train.parquet rows 2000..3999 directly, single temperature 0.25, G = 8, max_tokens 512, seed 0, same SYS string and answer parser, saving text (the same data L4c would use for pids 2000..5399; if L4c has run, reuse its rollouts for this range). Score with exp/l5c_proxy_score.py --kinds rm,judge3b unchanged. Cells: the E21 cells were KMeans(n_clusters=100, n_init=3, random_state=0) on the stored MiniLM embedding runs/E21-lowT-confident-errors/q_emb_minilm.npy (centroids were not saved): refit on that array, GATE that the labels reproduce feats_T0.25.cell100 exactly (exit if not), embed the new questions with the same MiniLM and assign with predict(). Analysis: exp/l6c_analyze.py, a copy of exp/l5c_analyze.py with the cell column taken from the old centroids, same accept rules (rm top-K with K = round(majority-vote accuracy x n), judge3b s > 0), same four cuts, 1,000-draw nulls; plus the old-cell error rate (from pids 0..1999, per proxy) as a score for new gold-wrong groups inside the top half, reported as AUC next to agreement alone. Register the bins in the docstring and smoke-test on synthetic scores first.

**Command shape:** two gpu_run.sh jobs, one at a time, each under the 45 minute cap, after `/homes/rdubash/avrl_live/FULL/bin/gpu_status.sh` shows under 2 GB used: `/homes/rdubash/avrl_live/FULL/bin/gpu_run.sh /homes/rdubash/avrl_live/FULL/avrl/runs/L6c-fresh-rollouts '/homes/rdubash/active-audit/.venv-judge/bin/python exp/l6c_rollouts_fresh.py --pid_start 2000 --pid_end 4000 --temps 0.25 --out /homes/rdubash/avrl_live/FULL/avrl/runs/L6c-fresh-rollouts'`, then `gpu_run.sh .../runs/L6c-fresh-proxies '/homes/rdubash/active-audit/.venv-judge/bin/python exp/l5c_proxy_score.py --kinds rm,judge3b --rm_model /homes/rdubash/active-audit/models/Skywork-Reward-V2-Qwen3-0.6B --judge_model /homes/rdubash/active-audit/models/Qwen2.5-3B-Instruct --rollouts .../L6c-fresh-rollouts/rollouts.parquet --groups .../L6c-fresh-rollouts/groups.parquet --per_group 2 --probe_rows 200 --out .../L6c-fresh-proxies'`, then the CPU analysis per kind.

**Metric:** per proxy: n_err in ACC (at least 100 needed for any kill; about 175 to 220 expected), sanity AUC, top-half cell100 ratio with 1,000-draw null p95 and cell-bootstrap CI, ratios at the other three cuts, AUC of the old-cell rate for gold-wrong inside the top half vs agreement alone.

**Pass threshold:** both proxies pass gates, have n_err in ACC at least 100, and have top-half cell100 ratio at least 1.5 and above own null p95 on the fresh prompts.

**Kill threshold:** both proxies pass gates with n_err in ACC at least 100 and top-half cell100 ratio below 1.3 or at or below own null p95 (the pids 0..1999 clustering did not transfer to fresh prompts, claim 3 closed on GSM8K for this policy). Anything else is inconclusive.

**Expected runtime:** UNVERIFIED about 12 minutes GPU for 16,000 rollouts (E21 took 1,428 s for 32,000), about 1 minute of rm and 4 minutes of judge3b scoring plus model loads, about 5 minutes CPU.

**Informs:** claim 3 as a property of the problems, and what the thesis can say about prompt-space structure beyond the 2,000-prompt pool. Prior (judgment): 30% pass, 40% kill, 30% inconclusive. Caveat: all proxies and the policy are Qwen-family, so even a pass does not separate a property of the problems from a shared Qwen blind spot; a non-Qwen judge (for example OLMo-2-0425-1B-Instruct, which may fail the 0.70 sanity gate) would be the next control.

### Iteration 7 (proposed after L6a-cell-rate-usability, 2026-10-18; verdict kill by the pre-registered AUC bin, replay concordant)

Iteration 7 = the loop run that executed L6a. L6a (CPU) found no usable cell-level signal: JA + cell-rate minus JA is +0.0014 to +0.0035 AUC on ACC with CI upper ends at most +0.016 on all three pools (and +0.0004 to +0.0015 at the rm K = 1,840 sensitivity), against a kill bin of delta below 0.02 and CI upper below 0.05; the cell_rerank replay is 11 to 20% worse than agree_acc (percentile CIs below 1.0 on every pool). A reviewer recomputation reproduced every delta to four decimals and showed the result survives a nested out-of-fold encoding and shrinkage 1 to 20 (deltas at most +0.0039). Claim 3 stays not supported with no live edge on GSM8K. The post hoc finding that shapes this iteration: at the 25% budget agree_acc audits every accepted agreement tier up to 0.75 and then picks a random subset of one boundary tier (rm: 60 slots of 245 groups at agreement 0.875 with 32 errors; Qwen3-4B: 30 of 279 with 34 errors; judge3b: 4 of 260), so every arm's ratio to agree_acc is decided by 30 to 60 audits, and inside that tier the proxy's score of the dissenting completion carries signal (AUC about 0.70 to 0.75 for s_oth higher or s_diff lower; the cell rate scores 0.51 to 0.56). A label-free tie-break on s_oth catches 142.0 / 168.0 / 131.0 (rm / judge3b / Qwen3-4B) against agree_acc 129.9 / 165.6 / 125.5 and greedy_blend 136.4 / 167.2 / 141.1 in the preview, so the registered fixed-rule set may be incomplete. Machine: L7a and L7c are CPU only, L7b needs the single A4000 under the 45 minute cap. Standing queue: L6b overlaps L7a at 10% and should run with it; L6c is superseded by L7b.

### L7a: Complete the fixed-rule set with a label-free dissenting-score tie-break, then rerun the learner comparison (CPU only)

**Hypothesis:** the learner's gain over agree_acc at 25% (greedy_blend 1.046 / 1.009 / 1.111 on rm / judge3b / Qwen3-4B) is the choice of 30 to 60 audits inside one agreement tier, and a label-free fixed tie-break on the proxy's score of the minority completion (s_oth descending, or s_diff ascending) in place of agree_acc's random tie-break closes at least 98% of the gap to greedy_blend on at least two of three pools, which would make claim 2 ("any fixed rule") false as worded in this regime.

**Change vs existing scripts:** copy exp/sim_warm_boot_j3.py to exp/sim_warm_boot_j4.py. Add two fixed policies to orders(): agree_acc_soth = lexsort((-s_oth, -ndist, agree, ~acc)) and agree_acc_sdiff = lexsort((s_diff, -ndist, agree, ~acc)) (the same keys as agree_acc with the random tie-break u4 replaced by the score; rejected groups last in random order as before). Two reporting modes in one file: (i) the new arms are context arms outside the best-fixed selection, so every j3 column still reproduces runs/L6a-cell-rate-usability/<pool>/boot_cellrerank_f0.25.json exactly (keep the gate); (ii) BEST over all seven fixed rules, with greedy_warm and greedy_blend warm-started from it (keep the norefit-equals-best-fixed gate per seed and on the first 20 resamples). Verdict read is cross-fitted to avoid the in-sample feature choice made in the L6a diagnostics: split the 2,000 pids into even and odd halves, choose between s_oth and s_diff on one half by within-boundary-tier AUC (labels of that half only), apply the choice to the other half at 25% of that half's groups, bootstrap B = 200 x 4 seeds within each half with default_rng(12345), and form the per-resample ratio of the summed errors caught over the two held-out halves. Register the bins in the docstring and smoke-test on synthetic labels with a planted dissenting-score effect and on null labels (where the tie-break must not beat random) before reading real labels.

**Command shape:** `cd /homes/rdubash/avrl_live/FULL/avrl && mkdir -p runs/L7a-tiebreak/{rm,judge3b,L3b-judge-proxy} && FEATS=s_maj,s_oth,s_diff,mean_tokens,std_tokens,any_truncated,log_qlen,agreement,n_distinct; for P in L5c-proxies/rm L5c-proxies/judge3b L3b-judge-proxy; do N=$(basename $P); for F in 0.25 0.1; do OMP_NUM_THREADS=2 BOOT_B=200 BOOT_SEEDS=4 BOOT_FRAC=$F /homes/rdubash/active-audit/.venv/bin/python exp/sim_warm_boot_j4.py runs/$P/feats_J.parquet judge_error cell100 $FEATS runs/L7a-tiebreak/$N/boot_tb_f$F.json > runs/L7a-tiebreak/$N/boot_f$F.log 2>&1 & done; done; wait` (six parallel CPU jobs; the cross-fit mode is selected by an env var, for example XFIT=1, and writes its own JSON). Include the rm K = 1,840 sensitivity with ACCEPT_COL=accept_sens.

**Metric:** per pool, errors caught at 25% and 10% for agree_acc, agree_acc_soth, agree_acc_sdiff, greedy_warm, greedy_blend, oracle; the cross-fitted tie-break / greedy_blend ratio and greedy_blend / cross-fitted tie-break ratio with percentile CI (verdict) and basic CI beside it; H = (oracle - best fixed) / oracle under the seven-rule best-fixed set; gates must pass.

**Pass threshold (hypothesis true, claim 2 as worded fails here):** cross-fitted tie-break / greedy_blend point ratio at least 0.98 on at least two of three pools AND the greedy_blend / tie-break percentile CI includes 1.0 on those pools.

**Kill threshold (learner survives the completed fixed set):** greedy_blend / best fixed (seven rules) percentile CI lower end above 1.0 at 25% on at least two of three pools. Anything else is inconclusive.

**Expected runtime:** about 8 minutes CPU (UNVERIFIED; the four L6a replay files took about 1 minute in parallel, six files plus the cross-fit add little).

**Informs:** claim 2 in its exact wording, and whether the thesis should pivot from "learned structure" to "a better label-free audit order". Prior (judgment): 45% pass, 20% kill, 35% inconclusive. Caveats: the pools share prompts and about 70% of their errors, so two of three is weaker than two independent passes; the boundary tier holds only 32 to 34 errors, so the ratio carries about 0.04 of bootstrap SD (about 5 errors); the feature scan (three features, three proxies) was done on the same pool before this was registered, which the cross-fit only partly removes; all proxies and the policy are Qwen-family.

### L7b: Fresh-prompt out-of-sample comparison of frozen arms (L6c re-scoped; GPU, then CPU)

**Hypothesis:** the claim 2 comparison (fixed tie-break versus greedy_blend versus agree_acc at 25%) holds out of sample: with the best fixed rule, the tie-break feature and the learner configuration frozen from the pids 0..1999 results, the ordering of the arms on unseen prompts pids 2000..3999 is the same as on the first pool, and the boundary tier is large enough to decide it. As a free secondary read, the original L6c descriptive check (pids 0..1999 cell100 centroids held fixed, top-half neighbor ratio on the fresh prompts) is reported without a verdict role, because L6a closed the usability route.

**Change vs existing scripts:** as L6c above (exp/l6c_rollouts_fresh.py, a copy of exp/e21_lowT_rollouts.py over data/gsm8k_train.parquet rows 2000..3999, temperature 0.25, G = 8, same SYS string and parser; then exp/l5c_proxy_score.py --kinds rm,judge3b unchanged), plus exp/sim_warm_boot_j4.py run on the new feats_J (the rm accept rule must be label-free here: a fixed top quantile of s_maj equal to the pids 0..1999 accept rate 85.75%, not a K computed from the fresh majority-vote accuracy). GATE before the replay: the boundary tier at 25% must hold at least 60 errors summed over the two proxies' pools or the replay verdict is recorded as underpowered. If L4c has already rolled out pids 2000..5399, reuse its rollouts for this range.

**Command shape:** the two gpu_run.sh jobs of L6c (rollouts, then proxy scoring), one at a time, each under the 45 minute cap, after `/homes/rdubash/avrl_live/FULL/bin/gpu_status.sh` shows under 2 GB used, then the CPU replay and the L6c analysis as in L7a with the rules frozen.

**Metric:** errors caught at 25% for agree_acc, the frozen tie-break, greedy_warm, greedy_blend, oracle per proxy; ratios of each arm to agree_acc and tie-break / greedy_blend with percentile CI (verdict) and basic CI; boundary-tier size and errors; secondary: the old-cell top-half neighbor ratio with 1,000-draw null p95.

**Pass threshold:** the frozen tie-break reaches at least 0.98 of greedy_blend on both proxies (claim 2 as worded fails out of sample) OR greedy_blend / agree_acc percentile CI lower end above 1.0 on both proxies (the learner is real out of sample); the two outcomes are reported as different verdicts.

**Kill threshold:** neither holds on either proxy and the boundary tier has at least 60 errors (the first-pool ordering did not transfer). Underpowered tiers are inconclusive.

**Expected runtime:** about 12 minutes GPU for rollouts, about 5 minutes for scoring (rm and judge3b), about 5 minutes CPU (UNVERIFIED; from the L6c estimate).

**Informs:** the first out-of-sample test of either thesis claim. Prior (judgment): 35% pass, 25% kill, 40% inconclusive. Run it after L7a leaves a learner-versus-fixed question open; if L7a already shows the learner clearly ahead or clearly redundant on the cross-fit, spend the GPU time on a non-Qwen judge control instead. Caveat: still one policy (1.5B), one task, Qwen-family proxies.

### L7c: Boundary-tier test of prompt-space structure (CPU only, lowest priority)

**Hypothesis:** the structure that claim 3 needs lives where the budget decision is made, the boundary agreement tier, not in ACC-wide AUC. Inside that tier, MiniLM PCs and the cell error rate add AUC over the dissenting-score features.

**Change vs existing scripts:** new exp/l7c_tier_structure.py (a copy of the CV and bootstrap path of exp/l6a_cell_rate.py restricted to the boundary tier of each pool at the 25% budget: agreement 0.875 for the three registered pools, 245 to 279 groups, 32 to 34 errors). Sets: T0 = s_oth, s_diff, std_tokens, mean_tokens, log_qlen (the features with signal in the L6a diagnostics, chosen a priori by the dissenting-score argument), T0 + 16 MiniLM PCs, T0 + shrunk in-fold cell100 rate. 5 x 3 RepeatedStratifiedKFold, balanced logistic, paired 1,000-resample bootstrap, same registered-bins-in-docstring and synthetic smoke test as L6a.

**Command shape:** `cd /homes/rdubash/avrl_live/FULL/avrl && for P in runs/L5c-proxies/rm runs/L5c-proxies/judge3b runs/L3b-judge-proxy; do OMP_NUM_THREADS=4 /homes/rdubash/active-audit/.venv/bin/python exp/l7c_tier_structure.py --pool $P --out runs/L7c-tier-structure/$(basename $P); done`

**Metric:** per pool, within-tier delta AUC of each set minus T0 with fold SD and paired bootstrap 95% CI.

**Pass threshold:** delta AUC at least 0.05 with CI lower end above 0 on at least two of three pools.

**Kill threshold:** delta AUC below 0.03 and CI upper end below 0.08 for every set on all three pools. Anything else is inconclusive (expected: with 32 to 34 errors per tier a single AUC has a standard error near 0.05).

**Expected runtime:** about 5 minutes CPU (UNVERIFIED).

**Informs:** the last cheap close-out of claim 3 on GSM8K. Prior (judgment): 5% pass, 55% kill, 40% inconclusive. Low information because of power; it can be folded into L7b, where the boundary tier on the fresh pool doubles the error count.

### Iteration 8 (proposed after L7a-tiebreak, 2026-10-19; verdict inconclusive by the registered bins, the literal spec PASS bin would read PASS on rm and judge3b on a 0.0014 margin)

Iteration 8 = the loop run that executed L7a. L7a (CPU, 20 jobs, about 4.5 minutes wall) completed the fixed-rule set with two label-free tie-breaks inside the boundary agreement tier (s_oth descending, s_diff ascending). Full pool, 25%, errors caught for rm K = 1,715 / judge3b / Qwen3-4B: agree_acc 130.4 / 165.8 / 127.0, agree_acc_soth 142.0 / 168.0 / 131.0, agree_acc_sdiff 137.0 / 166.0 / 135.0, greedy_blend 136.4 / 167.2 / 141.1, oracle 175 / 218 / 175. Registered bins: P_p on 1 of 3 pools (rm, cross-fitted tie-break / greedy_blend 1.074), K_p on 0 of 3 (greedy_blend7 / best of seven fixed 0.965 [0.899, 1.046], 1.018 [0.975, 1.063], 1.074 [0.972, 1.151]). Claim 2 as worded ("any fixed rule") is not supported in this regime and not refuted: the learner's one significant old edge (Qwen3-4B over agree_acc, 1.111) becomes 1.045 [0.961, 1.145] once the tie-breaks are in the fixed set. The recorder's check found that the 10% Qwen3-4B cell compared the learner with a comparator chosen at 25%; against the best fixed rule at 10% (blend50_acc) greedy_blend is 1.160 [0.975, 1.316], not 1.179 [1.008, 1.299]. The within-tier tie-break signal is real (AUC 0.70 to 0.75, permutation p at most 0.001), but the 25% budget audits only 60 / 4 / 30 groups of the 245 / 260 / 279 in the boundary tier, so every ratio is decided by a handful of audits and judge3b cannot discriminate at all. Machine: L8a and L8b are CPU only and run together; L8c needs the single A4000 under the 45 minute cap per job. L7b is superseded by L8c; L7c is unchanged and lowest priority. After L8a and L8b stop re-slicing pids 0..1999: further CPU reads on the same prompts add multiplicity, not independent errors.

### L8a: Budget curve of the learner against the best fixed rule, comparator re-chosen at each budget (CPU only)

**Hypothesis:** the learner's gain over the seven-rule fixed set is budget-specific noise: summed over the budget curve (5 to 40%), greedy_blend7 / best fixed rule (chosen in sample at each budget) has a percentile CI that includes 1 on every pool. The alternative is a positive Qwen3-4B curve (point +4.5% to +7.4% at 25%, +15% to +16% at 10% against the correct comparator).

**Change vs existing scripts:** copy exp/sim_warm_boot_j4.py to exp/sim_warm_boot_j5.py. One change: BEST and BEST7 are chosen from the full-pool 8-seed mean at the run's own BOOT_FRAC, not at 25% (in L7a the 10% files inherited the 25% choice; at 10% on Qwen3-4B the best fixed rule is blend50_acc, 71.1, not agree_acc_sdiff, 70.0). Hard gate: the 25% run reproduces runs/L7a-tiebreak/<pool>/boot_tb_f0.25.json exactly (BEST is unchanged at 25%), and the norefit gates of j4 pass at every budget. Register in the docstring before any label is read: S per pool = mean over the seven budgets of log(greedy_blend7 / best fixed at that budget), formed per resample on the same 200 resamples (default_rng(12345) gives identical resamples at every budget, which makes the curve statistic paired), percentile CI over resamples. Skip the cross-fit mode.

**Command shape:** adapt exp/L7a-tiebreak.sh: `cd /homes/rdubash/avrl_live/FULL/avrl && FEATS=s_maj,s_oth,s_diff,mean_tokens,std_tokens,any_truncated,log_qlen,agreement,n_distinct; for P in L5c-proxies/rm L5c-proxies/judge3b L3b-judge-proxy; do N=$(basename $P); mkdir -p runs/L8a-budget-curve/$N; for F in 0.05 0.1 0.15 0.2 0.25 0.3 0.4; do OMP_NUM_THREADS=2 BOOT_B=200 BOOT_SEEDS=4 BOOT_FRAC=$F /homes/rdubash/active-audit/.venv/bin/python exp/sim_warm_boot_j5.py runs/$P/feats_J.parquet judge_error cell100 $FEATS runs/L8a-budget-curve/$N/boot_f$F.json > runs/L8a-budget-curve/$N/boot_f$F.log 2>&1; done; done` (21 jobs under xargs -P 20 as in L7a, then a new exp/l8a_finalize.py).

**Metric:** per pool and budget: errors caught for agree_acc, both tie-breaks, the best fixed rule at that budget, greedy_warm7, greedy_blend7, oracle, H = (oracle - best fixed) / oracle, and greedy_blend7 / best fixed with percentile CI (descriptive); the curve statistic S with percentile CI per pool (verdict).

**Pass threshold:** S percentile CI lower end above 0 on Qwen3-4B and on at least one other pool.

**Kill threshold:** S plug-in below +0.02 on all three pools. This is a point-estimate kill: at 2,000 prompts the CI of S is about plus or minus 0.07, so it can say "no edge shown", not "edge excluded". Anything else is inconclusive.

**Expected runtime:** about 8 minutes CPU (UNVERIFIED for 21 jobs; L7a's 20 jobs took about 4.5 minutes wall, and the 40% budget has a longer learner loop).

**Informs:** claim 2 in its exact wording, and repairs the 10% comparator error. Prior (judgment): 10% pass, 20% kill, 70% inconclusive; its main value is a clean comparator and a curve, not a verdict. Caveats: the seven budgets are correlated points on the same 2,000 prompts, not seven tests; the best fixed rule is picked in sample at every budget (favours the fixed set); the pools share about 70% of their errors.

### L8b: Which features does the learner's edge need? Ablation and tier decomposition (CPU only; run with L8a)

**Hypothesis:** the learner's gain over agree_acc at 25% on Qwen3-4B and rm (greedy_blend / agree_acc 1.111 and 1.046) comes from the proxy's completion scores (s_maj, s_oth, s_diff), not from prompt-side or format features, so the learner is an approximate version of the label-free dissenting-score order and claim 3 has no live route through the learner.

**Change vs existing scripts:** new exp/l8b_ablation.py (the greedy_blend path of exp/sim_warm_boot_j5.py with a feature-subset argument, warm-started from BEST7 as in L7a). Subsets: ALL9 (control, must reproduce L7a greedy_blend7 exactly: gate), NO_QLEN (drop log_qlen, the only prompt-side feature), SCORES (s_maj, s_oth, s_diff), FORMAT (mean_tokens, std_tokens, any_truncated, log_qlen). Same B = 200 x 4 seeds, same resamples, 25% and 10% (best fixed re-chosen at 10% as in L8a). Plus a tier table on the unresampled pool, 8 seeds: per agreement tier, groups audited and errors caught by greedy_blend (ALL9), agree_acc_soth, agree_acc_sdiff and agree_acc, so the extra 6 errors of the Qwen3-4B learner are located in the order (boundary tier versus lower tiers). Register bins and a synthetic-label smoke (null labels: no subset beats ALL9 or agree_acc; planted score effect: SCORES matches ALL9) in the docstring first.

**Command shape:** `cd /homes/rdubash/avrl_live/FULL/avrl && for P in L5c-proxies/rm L5c-proxies/judge3b L3b-judge-proxy; do N=$(basename $P); for F in 0.25 0.1; do OMP_NUM_THREADS=2 BOOT_B=200 BOOT_SEEDS=4 BOOT_FRAC=$F /homes/rdubash/active-audit/.venv/bin/python exp/l8b_ablation.py runs/$P/feats_J.parquet judge_error runs/L8b-ablation/$N/ablate_f$F.json & done; done; wait` (six parallel jobs).

**Metric:** per pool: caught for each subset, ratio of each subset to ALL9 and to agree_acc and to the best fixed rule with percentile CI (verdict) and basic CI beside it; the tier table.

**Pass threshold:** on every pool where the non-vacuity guard holds (greedy_blend ALL9 / agree_acc point ratio at least 1.04, expected: rm and Qwen3-4B only), greedy_blend(SCORES) / greedy_blend(ALL9) point ratio at least 0.98 with percentile CI including 1.0 AND greedy_blend(FORMAT) / agree_acc percentile CI includes 1.0; the pool count must be at least two, which means both rm and Qwen3-4B.

**Kill threshold:** greedy_blend(NO_QLEN) / greedy_blend(ALL9) percentile CI upper end below 1.0 on at least two of three pools (removing the one prompt-side feature costs errors, so prompt-side structure carries the edge). Anything else is inconclusive.

**Expected runtime:** about 8 minutes CPU (UNVERIFIED; six greedy-path jobs, each fewer arms than the L7a files).

**Informs:** claim 3 for the learner itself (L6a closed it for clusters only) and the case for the label-free-order pivot. Prior (judgment): 40% pass, 10% kill, 50% inconclusive. Caveats: log_qlen is prompt length, not semantic prompt-space structure (MiniLM PCs and cell rates were closed by L6a); the Qwen3-4B edge is 5 to 11 errors, so subset differences are 1 to 4 errors against a ratio SD of about 0.04, which is why inconclusive is the modal outcome; the guard makes the pass require the two pools where a learner edge exists.

### L8c: Fresh-prompt frozen-arm test with doubled pool (GPU, then CPU; supersedes L7b)

**Hypothesis:** out of sample, with everything frozen from pids 0..1999, either the label-free tie-break reaches at least 0.98 of greedy_blend7 where the learner has an edge (claim 2 as worded fails out of sample), or greedy_blend7 stays above the best fixed rule (the learner is real out of sample). L7a could not separate these at 32 to 34 boundary errors per pool.

**Change vs existing scripts:** new exp/l8c_rollouts_fresh.py (a copy of exp/e21_lowT_rollouts.py with a start-row offset; --temps 0.25, G = 8, same SYS string and parser) over data/gsm8k_train.parquet rows 2000..3999 and 4000..5999 as two shards, one gpu_run.sh job each so each stays under the 45 minute cap (reuse L4c rollouts for pids 2000..5399 if they exist). Score each shard with exp/l5c_proxy_score.py --kinds rm,judge3b and with exp/l3b_judge_score.py (the Qwen3-4B judge, because the learner's clearest edge lives there). Then exp/sim_warm_boot_j5.py on the concatenated feats with every choice frozen from L7a: the accept rule must be label-free (a fixed top quantile of s_maj equal to the pids 0..1999 accept rate of 85.75% for rm, the judge's own threshold for the judges, never a K computed from fresh majority-vote accuracy); BEST7 fixed per proxy to the L7a choice (agree_acc_soth for rm and judge3b, agree_acc_sdiff for Qwen3-4B); both tie-breaks reported as two pre-declared arms because the cross-fit does not pick one; no reselection, no cross-fit refit. Register bins and the guard V (agree_acc / greedy_blend7 below 0.98) in the docstring before any fresh label is read.

**Command shape:** the gpu_run.sh jobs one at a time, each after `/homes/rdubash/avrl_live/FULL/bin/gpu_status.sh` shows under 2 GB used: rollouts shard A, rollouts shard B, then proxy scoring per shard; then the CPU replay as in L7a with the rules frozen. GATE before the replay verdict: the boundary tier at 25% holds at least 60 errors per proxy pool (not summed over proxies, they share about 70% of errors), else the verdict is underpowered.

**Metric:** per proxy: errors caught at 25% and 10% for agree_acc, both frozen tie-breaks, greedy_warm7, greedy_blend7, oracle; tie-break / greedy_blend7, greedy_blend7 / agree_acc and greedy_blend7 / best fixed with percentile CI (verdict) and basic CI beside it; boundary-tier size and errors per proxy.

**Pass threshold:** two different passes, reported as different verdicts. Tie-break pass (claim 2 as worded fails out of sample): frozen best tie-break / greedy_blend7 point ratio at least 0.98 with the greedy_blend7 / tie-break percentile CI including 1.0, on at least two of three proxies where the guard V holds. Learner pass (the learner is real out of sample): greedy_blend7 / best fixed percentile CI lower end above 1.0 on Qwen3-4B and one other proxy.

**Kill threshold:** neither pass holds on any proxy and the boundary tier gate holds on every proxy (the first-pool ordering did not transfer). Underpowered tiers are inconclusive.

**Expected runtime:** about 12 minutes GPU per rollout shard, about 6 minutes per scoring job for rm plus judge3b and a similar amount for the Qwen3-4B judge, about 6 minutes CPU (all UNVERIFIED, from the L6c estimate and the 375 s L5c scoring run); total about 50 to 60 minutes across five to six jobs, each under the cap.

**Informs:** the only out-of-sample test of claims 2 and 3. Power stated up front: at 4,000 prompts the ratio SD falls from about 0.04 to about 0.028, enough to resolve a 7 to 8% gap (rm tie-break against learner) at about 2.6 SD but not the 4.5% Qwen3-4B learner edge at about 1.6 SD. Prior (judgment): 30% tie-break pass, 20% learner pass, 15% kill, 35% inconclusive. Run it unless L8a kills the learner outright; if it does, freeze only the tie-break and drop the Qwen3-4B judge scoring. Caveats: one policy (1.5B), one task, Qwen-family proxies only; a non-Qwen judge control is the next step if either pass holds.

**Result of the first half (run 2026-10-19 as L8c-A-fresh-tiebreak, runs/L8c-A-fresh-tiebreak/, RESEARCH_LOG.md entry "L8c-A-fresh-tiebreak"): inconclusive, formally UNDERPOWERED.** Shard A only (pids 2000..3999, rm and Qwen3-4B, judge3b dropped), rules frozen, tie-break arms only as the verdict (learner arms context). Within-tier AUC replicated (rm 0.752 [0.644, 0.843], Qwen3-4B 0.781 [0.669, 0.882]; old 0.748 and 0.740), tie-break / agree_acc at 25% was 1.035 [1.000, 1.078] and 1.026 [1.000, 1.074] (old 1.089 and 1.063), short of the 1.04 PASS ratio, and rm had 23 boundary-tier errors against the 25 required. Shard B (pids 4000..5999) is not yet run; it is L9a below.

### Iteration 9 (proposed after L8c-A-fresh-tiebreak, 2026-10-19; verdict inconclusive, underpowered on rm by two boundary-tier errors; AUC replicates, caught-error gain is about +3% at 25%)

Iteration 9 = the loop run that executed L8c-A (GPU 737 s rollouts + 77 s rm + 361 s Qwen3-4B scoring, CPU replay, 52 minutes wall). Fresh pool A, 25% budget, rm / Qwen3-4B (old pids 0..1999 in brackets): agree_acc 142.0 / 136.5 [130.4 / 127.0], verdict tie-break 147.0 / 140.0 [142.0 / 135.0], greedy_blend7 148.0 / 151.0 [137.0 / 145.0], oracle 186 / 192. Recorder finding that shapes every spec below: the tie-break gain at a budget lives in the single partially audited (agreement, n_distinct) cell, so the 25% ratio is a lumpy single-cell statistic (fresh gain +5.6 and +4.7 errors from 16 and 59 slots, old +12.2 and +9.3 from 60 and 30 slots); across seven budgets the fresh and old gains match (mean +2.4 and +3.9 against +3.0 and +4.5 errors; at 30% +9.1 and +11.5 against +8.1 and +10.2). So the next verdicts use a budget-averaged statistic and a label-free operative cell, not the fixed 0.875 tier. Status of earlier proposals: L8a and L8b stay proposed; run L8a's curve statistic on the pooled A + B data inside L9a (no extra GPU) and L8b only if L9b is inconclusive; neither should be run on pids 0..1999 alone. Machine: L9a and L9c need the single A4000, one job at a time under the 45 minute cap; L9b is CPU only.

### L9a: Shard B (pids 4000..5999) and the budget-averaged tie-break test on 4,000 fresh prompts (GPU, then CPU)

**Hypothesis:** the label-free dissenting-score tie-break catches more proxy errors than plain agree_acc when averaged over the budget curve, on rm and Qwen3-4B, with every rule frozen from pids 0..1999. Null: the shard-A gain (12 of 14 budget cells positive, budget-mean ratio about 1.017 rm and 1.034 Qwen3-4B) is noise or a post hoc budget pick. Secondary: greedy_blend7 / best fixed rule on Qwen3-4B, the one cell where the learner edge replicated in point estimate (1.074 old, 1.079 fresh) with a CI covering parity.

**Change vs existing scripts:** copy exp/l8c_rollouts_fresh.py to exp/l9a_rollouts_fresh.py. It asserts len(ref) >= pid_start with ref = the 3,000-row p0 groups table, so --pid_start 4000 fails as written: pass --ref_groups a table holding the questions of orig pids 0..3999 (p0 pids 0..1999 followed by runs/L8c-A-fresh-tiebreak/rollouts/T0.25/groups.parquet with its pid column overwritten by orig_pid; the script sorts by pid and asserts pid == arange, and needs the question column, which p0 groups.parquet has) and keep the prefix-equality and disjointness asserts on all 4,000 rows; confirm sample(n=6000, random_state=0) is prefix-consistent with sample(n=4000) before launch (it is for sample(n=3000) per the L8c-A check). Score with exp/l5c_proxy_score.py --kinds rm and exp/l3b_judge_score.py unchanged. Copy exp/l8c_analyze.py and exp/l8c_boot_forced.py to l9a_* with three changes, all registered in the docstring before any shard-B label is read: (1) the pool is the concatenation of A and B (4,000 groups, orig_pid kept), rm accept = top 85.75% of s_maj in the pooled pool (label-free), Qwen3-4B accept = s_maj > 0; (2) new curve statistic S_tb per proxy = mean over budgets 10 / 15 / 20 / 25 / 30 / 40% of log(tie-break / agree_acc), formed per resample on the same 200 x 4 resamples (default_rng(12345) gives the same resample indices at every budget), percentile CI over resamples; (3) the within-tier AUC becomes the tier-stratified AUC over all audited agreement tiers (agreement below 1.0) with a 2,000-resample group bootstrap, because the 0.875 tier does not hold the budget boundary for Qwen3-4B. Keep BEST7 forced (rm agree_acc_soth, Qwen3-4B agree_acc_sdiff), MIN_CELL=1, gate (i) forced-BEST7 replay on pids 0..1999, scorer sanity AUC >= 0.70, null permutation control.

**Command shape:** `cd /homes/rdubash/avrl_live/FULL/avrl; F=/homes/rdubash/avrl_live/FULL; R=$F/avrl/runs/L9a-shardB; $F/bin/gpu_status.sh` (under 2 GB used), then one job at a time: `$F/bin/gpu_run.sh $R/rollouts '/homes/rdubash/active-audit/.venv-judge/bin/python exp/l9a_rollouts_fresh.py --model '$F'/models/Qwen2.5-1.5B-Instruct --temps 0.25 --pid_start 4000 --n_prompts 2000 --G 8 --max_tokens 512 --seed 0 --ref_groups <orig pids 0..3999 table> --out '$R'/rollouts'`; then `$F/bin/gpu_run.sh $R/rm '/homes/rdubash/active-audit/.venv-judge/bin/python exp/l5c_proxy_score.py --kinds rm --rm_model /homes/rdubash/active-audit/models/Skywork-Reward-V2-Qwen3-0.6B --judge_model /homes/rdubash/active-audit/models/Qwen2.5-3B-Instruct --rollouts '$R'/rollouts/T0.25/rollouts.parquet --groups '$R'/rollouts/T0.25/groups.parquet --per_group 2 --probe_rows 200 --out '$R'/rm'`; then `$F/bin/gpu_run.sh $R/q3_4b '/homes/rdubash/active-audit/.venv-judge/bin/python exp/l3b_judge_score.py --model '$F'/models/Qwen3-4B --rollouts '$R'/rollouts/T0.25/rollouts.parquet --groups '$R'/rollouts/T0.25/groups.parquet --per_group 2 --out '$R'/q3_4b'`; each polled with `until [ -f DIR/done.flag ]; do ls DIR > /dev/null; sleep 30; done; cat DIR/exit_code`. CPU: `BOOT_B=200 BOOT_SEEDS=4 /homes/rdubash/active-audit/.venv/bin/python exp/l9a_analyze.py --run_dir $R` once for the budget list. Before launching any GPU job, run the design-only power check: the SD of S_tb over the 200 resamples of shard A alone (labels of A are known, nothing about B is read); the expected SD at 4,000 prompts is that SD divided by about 1.4.

**Metric:** per proxy, on pooled A + B (verdict) and on B alone (sign check): S_tb with percentile CI, tie-break / agree_acc at each of the six budgets with percentile CI, tier-stratified AUC with bootstrap CI, boundary errors, greedy_blend7 / best fixed (re-chosen at each budget) at 10% and 25%, and the L8a curve statistic for the learner (log greedy_blend7 / best fixed, same resamples).

**Pass threshold:** on BOTH rm and Qwen3-4B, pooled A + B: S_tb percentile CI lower end above 0 AND tier-stratified AUC point at least 0.62 with CI lower end above 0.50 AND S_tb on shard B alone has a positive plug-in; all gates passing and the measured pooled-pool SD of S_tb at most 0.015 (a measured-power gate, otherwise UNDERPOWERED).

**Kill threshold:** S_tb plug-in at most +0.005 on both proxies on pooled A + B, gates passing and the SD gate met (the shard-A budget-mean ratios 1.017 and 1.034 did not hold on fresh data). Anything else is inconclusive. A pass makes the label-free order a finding, and the deliverable becomes agreement threshold plus dissenting-score tie-break with a gain of about 2% to 4% of errors caught; a kill retires the tie-break pivot and moves the loop to regime variation (L9c, then E07 and E09).

**Expected runtime:** GPU about 20 minutes in three jobs (measured in L8c-A: 737 s rollouts, 77 s rm, 361 s Qwen3-4B), CPU about 12 minutes (UNVERIFIED; L8c-A's CPU stage covered 2,000 prompts and two budgets, this has 4,000 prompts and six budgets, run as parallel jobs).

**Informs:** claims 2 and 3 and the label-free-order pivot; the only way to turn the 12-of-14 positive but small shard-A gain into a verdict, and it doubles the power of the learner comparison (ratio SD about 0.03 at 2,000 prompts becomes about 0.02 to 0.03 at 4,000, so the 7.9% Qwen3-4B learner edge moves from about 2 SD to about 3 SD if it is real). Prior (judgment): 45% pass, 15% kill, 40% inconclusive. Caveats: shard A is not blind for S_tb because the proposer has seen its per-budget gains, which is why the pass needs a positive shard-B-alone sign and why the budget list is fixed here; the two proxies share about 70% of their errors; one policy, one task, Qwen-family proxies only.

**Result (run 2026-10-19, runs/L9a-shardB-tiebreak/, RESEARCH_LOG.md entry "L9a-shardB-tiebreak"): logged as PASS by the registered bins on both proxies (numbers recomputed independently, exp/l9a_recorder_check.py).** S_tb pooled A + B: rm +0.0133 (percentile CI [+0.0091, +0.0268], basic CI [-0.0002, +0.0175]), Qwen3-4B +0.0281 ([+0.0177, +0.0353]); measured pooled SD 0.0048 on both; tier-stratified AUC 0.662 [0.624, 0.699] and 0.730 [0.694, 0.767]; shard B alone (blind) +0.0192 and +0.0158; all gates pass. The gain sits at the 25% and 30% budgets (rm without them -0.0007, Qwen3-4B +0.0168), looks like a tier-landing effect (cuts inside the 7/8 agreement tier), and is +1.3% (rm) and +2.9% (Qwen3-4B) of agree_acc catches, below the 2% to 4% in the registration for rm. Secondaries: greedy_blend7 / best fixed re-chosen per budget (pooled, in sample) curve +0.0254 rm and +0.0524 Qwen3-4B (CI [+0.0182, +0.0740]); L11b statistic C_B (15/20/25%, comparator from shard A, scored on shard B) -0.0001 rm and +0.1344 Qwen3-4B [+0.0382, +0.1646], 1 of 2 cells positive, no L11b verdict. Follow-ups: Iteration 12 below (L12a, L12b); L10c stays queued and is next on the GPU.

### L9b: Is the learner's Qwen3-4B edge a static function of the scores? Frozen logistic model from pids 0..1999 on fresh shard A (CPU only)

**Hypothesis:** the learner's edge over the best fixed rule (greedy_blend7 / best fixed 1.074 old and 1.079 fresh on Qwen3-4B; 0.965 old and 1.007 fresh on rm) is a static function of agreement, n_distinct and the completion scores, so one fixed logistic model fitted once on pids 0..1999 and frozen reaches at least 98% of greedy_blend7 on fresh shard A, and no online learning is needed.

**Change vs existing scripts:** new exp/l9b_static_logit.py plus one extra arm static_logit in a copy of exp/l8c_boot_forced.py, appended after every existing random draw so all existing columns stay bit-identical (gate: every existing column reproduces runs/L8c-A-fresh-tiebreak/boot/<pool>/boot_tb_f0.25.json exactly). Model: sklearn LogisticRegression(C=1.0, max_iter=1000) on standardized [agreement, n_distinct, s_maj, s_oth] (s_diff = s_maj - s_oth exactly, so it is left out), fitted on the accepted groups of pids 0..1999 (runs/L5c-proxies/rm/feats_J.parquet and runs/L3b-judge-proxy/feats_J.parquet), label judge_error, one fit, no cross-validation and no tuning; coefficients saved to the output before any fresh label is read. Order: accepted groups by predicted risk descending, random tie-break, rejected groups last in random order (as agree_acc). Evaluate on runs/L8c-A-fresh-tiebreak/pool/{rm,q3_4b}/feats_J.parquet.

**Command shape:** CPU in the shell, `OMP_NUM_THREADS=2 BOOT_B=200 BOOT_SEEDS=4 BOOT_FRAC=0.25 /homes/rdubash/active-audit/.venv/bin/python exp/l9b_boot_static.py <fresh feats_J> judge_error cell100 <FEATS> <out.json>` for each proxy at 0.25, 0.1 and 0.4, no gpu_run.sh.

**Metric:** errors caught at 10 / 25 / 40% for agree_acc, the frozen tie-break, static_logit, greedy_blend7, oracle; static_logit / greedy_blend7 and static_logit / tie-break with percentile CI (paired, same resamples). Registered reference points on shard A at 25% (known to the proposer): greedy_blend7 148.0 (rm) and 151.0 (Qwen3-4B), so 98% is 145.0 and 148.0 caught errors.

**Pass threshold:** static_logit / greedy_blend7 point ratio at least 0.98 at 25% on both rm and Qwen3-4B (the learned allocator is unnecessary for this edge; claim 2 weakens further).

**Kill threshold:** static_logit / greedy_blend7 point ratio at most 0.94 on Qwen3-4B at both 10% and 25% (the learner's edge needs online adaptation or features the static model lacks). Anything else is inconclusive.

**Expected runtime:** about 5 minutes CPU (UNVERIFIED; six single-model jobs on 2,000 groups).

**Informs:** claim 2 directly, as an out-of-sample ablation that costs no GPU: it separates "better label-free order" (static) from "learned online structure". Prior (judgment): 35% pass, 25% kill, 40% inconclusive. Caveats: fit on pids 0..1999 and tested on shard A only, so one 2,000-prompt read with a ratio SD of about 0.03 to 0.04; the proposer has seen shard A's learner numbers (not any static-model result); a pass on rm alone is not enough because rm's learner edge is zero; greedy_warm7 sits at 0.84 to 0.87 of the tie-break at 25%, so only the blend arm is the comparator.

### L9c: Frozen tie-break on a weak policy, Qwen2.5-0.5B-Instruct, fresh pids 2000..3999 (GPU, then CPU)

**Hypothesis:** the dissenting-score tie-break transfers across error regimes: on a weak policy (many more proxy errors than the 1.5B policy) the label-free order inside the operative cell keeps a within-cell AUC of at least 0.62 and catches at least 4% more errors than agree_acc at 25%. The weak policy also removes the power problem (several times more errors in every tier).

**Change vs existing scripts:** copy exp/l8c_rollouts_fresh.py with --model Qwen2.5-0.5B-Instruct (E06's policy), same prompts (orig pids 2000..3999), T=0.25, G=8, max_tokens 512, seed 0; score with the Qwen3-4B Yes/No judge (exp/l3b_judge_score.py unchanged, accept = s_maj > 0) and with rm (exp/l5c_proxy_score.py --kinds rm; the rm accept fraction is set to the Qwen3-4B judge's accepted fraction on this same pool, label-free, never the 85.75% of the 1.5B policy). Copy exp/l8c_analyze.py and the forced boot script with one design change registered before labels: the operative cell is named label-free before any label is read (the (agreement, n_distinct) cell holding the 500th audit of the agree_acc key order on the accept set), and the AUC, the gate and the ratio are all computed in that cell. Frozen: BEST7 (rm agree_acc_soth, Qwen3-4B agree_acc_sdiff), signs, hyperparameters. No reselection.

**Command shape:** the same three-job pattern as L9a with `--model '$F'/models/Qwen2.5-0.5B-Instruct`, `--pid_start 2000 --n_prompts 2000`, `--ref_groups` the p0 groups table (pid 2000 is within its 3,000 rows), output under runs/L9c-weak-policy/; one job at a time after gpu_status shows under 2 GB; CPU replay at 25% (verdict) and 10%.

**Metric:** per proxy: operative-cell size, errors and AUC (point plus 2,000-resample group bootstrap CI), errors caught at 25% and 10% for agree_acc, both tie-breaks, greedy_blend7 (descriptive), oracle, and tie-break / agree_acc with percentile CI.

**Pass threshold:** on Qwen3-4B (the primary proxy, accept rule is natural for any policy): operative-cell AUC point at least 0.62 with CI lower end above 0.50 AND tie-break / agree_acc point ratio at least 1.04 at 25% AND at least 60 errors in the operative cell; rm must show a positive plug-in gain (secondary, not a pass condition).

**Kill threshold:** Qwen3-4B operative-cell AUC at most 0.57 AND ratio at most 1.02, with at least 60 errors in the operative cell (the mechanism does not transfer to a weak policy). Anything else is inconclusive; fewer than 60 operative-cell errors is UNDERPOWERED.

**Expected runtime:** about 20 minutes GPU in three jobs (UNVERIFIED for the 0.5B rollouts, which should be faster than the measured 737 s for the 1.5B model; scoring 361 s and 77 s measured in L8c-A), about 5 minutes CPU.

**Informs:** whether the label-free order is a property of the proxy-minority-completion mechanism or of this one policy, and starts the E06 regime-variation branch without waiting for a kill in L9a. Prior (judgment): 35% pass, 20% kill, 45% inconclusive. Caveats: one task (GSM8K train), Qwen-family policy and judge; a weak policy has a different error mix (more format and arithmetic slips), so a pass would be a regime-transfer point and not a general claim; a non-Qwen judge control is still the next step after any pass.

### Iteration 10 (proposed after L9c-weak-policy-tiebreak, 2026-10-19; verdict inconclusive, UNDERPOWERED by the 60-error gate; the spec's PASS ratio bar of 1.04 was unreachable in the operative cell, ceiling 1.014)

Iteration 10 = the loop run that executed L9c (GPU 519 s rollouts + 80 s rm + 399 s Qwen3-4B scoring, CPU replay about 3 minutes, 27 minutes wall). Weak policy Qwen2.5-0.5B-Instruct on the fresh prompts of L8c-A (1.5B values in brackets): majority-vote error 0.3245 [0.144], judge errors J 281 Qwen3-4B / 303 rm [192 / 186], accepted 1,583 [1,841 / 1,715], scorer sanity AUC 0.911 / 0.886 [0.848 / 0.807]. Operative cell at 25% (0.625, 2): 47 groups, 16 errors; within-cell AUC 0.905 [0.810, 0.973] Qwen3-4B, 0.811 [0.657, 0.936] rm; tie-break / agree_acc 1.014 [0.999, 1.045] and 1.011, against a perfect-order ceiling of 1.014 and 1.016. Findings that shape every spec below: (a) a weak policy raised majority-wrong groups 2.25x but judge errors only 1.5x because the proxies screen it better, so the premise "several times more errors per cell" failed and a cell-level error-count gate of 60 is unreachable on a 2,000-prompt pool (60 of 281 errors in one block); the tie-break test must use a budget-averaged or tier-stratified statistic. (b) Post hoc closed form: tie-break / agree_acc at 10 / 15 / 20 / 25 / 30 / 40% is 1.036 / 1.017 / 1.011 / 1.014 / 1.009 / 1.024 for Qwen3-4B (six of six positive, mean log ratio 0.0185) and 1.003 / 1.008 / 0.993 / 1.011 / 1.005 / 1.027 for rm (mean 0.0078). (c) The learner beats the frozen tie-break arm (1.128 [1.062, 1.175] and 1.132 [1.046, 1.191] at 25%) and agree_acc (1.144 on both), but a fixed blend rule (blend50_acc, 1.114 and 1.103 of agree_acc) matches most of it: greedy_blend7 / in-sample best fixed 1.027 [0.959, 1.065] and 1.037 [0.960, 1.086] at 25%, 0.983 and 1.072 at 10%, all four intervals covering parity. (d) Claim 3, l6a_cell_rate.py unchanged on the Qwen3-4B pool: confident half (792 groups, 47 errors) delta AUC +0.0144 [0.0033, 0.0300] for cell100, ACC-wide about 0, A_pass false. L9a (1.5B shard B) and L9b (static logit) from Iteration 9 stay queued and are unaffected; L8a is superseded by L10a.

### L10a: Is the learner's edge over the best fixed rule real once the fixed comparator is cross-fitted? Four pools, six budgets (CPU only; supersedes L8a)

**Hypothesis:** in every claim 2 read so far the comparator is the in-sample best of seven fixed rules (chosen on the same pool it is scored on), which flatters the fixed side, and the single-budget ratio SD (0.03 to 0.06) hides whether the learner (greedy_blend7) is ahead anywhere on the budget curve. With the comparator chosen on the OTHER half of the prompts, greedy_blend7 catches more corrupted-reward groups than the best fixed rule, averaged over budgets and pooled over four (policy, proxy) cells. Null: the learner's edge over the agreement rule (1.04 to 1.14) is a feature blend that a cross-fitted fixed rule already has, so the pooled edge is at most 1%.

**Change vs existing scripts:** copy exp/l8c_boot_forced.py to exp/l10a_boot_xfit_fixed.py and extend its XFIT=1 mode (sim_warm_boot_j4.py, lines 136 to 160: pids split by parity, the tie-break feature chosen on the other half) so that the chosen object is the best of the SEVEN fixed rules by 8-seed mean caught at the run budget on the other half, evaluated on the held half at the same fraction of that half; the learner greedy_blend7 runs on the held half warm-started from the frozen arm (FORCE_BEST7, no reselection: rm agree_acc_soth, Qwen3-4B agree_acc_sdiff), same default_rng(12345) resamples, B = 200 x 4 seeds, per-resample quantity = sum over the two held-out halves of the seed-mean caught (as j4). Pools: runs/L9c-weak-policy-tiebreak/pool/{q3_4b,rm}/feats_J.parquet (0.5B) and runs/L8c-A-fresh-tiebreak/pool/{q3_4b,rm}/feats_J.parquet (1.5B fresh), MIN_CELL=1, label judge_error. Budgets 10 / 15 / 20 / 25 / 30 / 40% of each half. Register the bins in the docstring before any label of the new statistic is read. Gate: every column that does not involve the cross-fit (agree_acc, blend50_acc, greedy_blend7 full-pool at 25%) reproduces the L9c / L8c-A boot_tb_f0.25.json values exactly (L9c Qwen3-4B: 192.25, 214.25, 220.0).

**Command shape:** `cd /homes/rdubash/avrl_live/FULL/avrl; for pool in L9c-weak-policy-tiebreak L8c-A-fresh-tiebreak; do for p in q3_4b rm; do for f in 0.1 0.15 0.2 0.25 0.3 0.4; do echo "XFIT=1 MIN_CELL=1 FORCE_BEST7=<arm> OMP_NUM_THREADS=2 BOOT_B=200 BOOT_SEEDS=4 BOOT_FRAC=$f /homes/rdubash/active-audit/.venv/bin/python exp/l10a_boot_xfit_fixed.py runs/$pool/pool/$p/feats_J.parquet judge_error cell100 s_maj,s_oth,s_diff,mean_tokens,std_tokens,any_truncated,log_qlen,agreement,n_distinct runs/L10a/$pool/$p/xfit_f$f.json"; done; done; done | xargs -P 4 -I{} sh -c '{}'` (24 jobs, 4 in parallel; no GPU, no gpu_run.sh), then a finalizer.

**Metric:** per cell and budget, cross-fitted greedy_blend7 / fixed ratio with percentile CI; the budget-averaged statistic C = mean over the six budgets of log(greedy_blend7 / cross-fitted best fixed), per cell and pooled as the unweighted mean of the four cells, with a paired resample CI; the same against agree_acc and against the frozen arm (context); the identity of the chosen fixed rule per half and budget.

**Pass threshold:** pooled C point >= +0.02 (a 2% edge) with percentile CI lower end > 0, AND a positive plug-in C in at least 3 of the 4 cells, all gates passing (claim 2 gains its first cross-fitted evidence).

**Kill threshold:** pooled C percentile CI upper end < +0.01 (the learner's edge over a cross-fitted fixed rule is below 1%, claim 2 as worded is dropped and the deliverable is the fixed blend plus tie-break). Anything else is inconclusive.

**Expected runtime:** about 20 minutes CPU (UNVERIFIED; L9c's eight replay jobs of one budget each took about 3 minutes wall with 4 in parallel, so 24 jobs of half-size pools should take 10 to 20 minutes).

**Informs:** claim 2 directly, the only open question that existing data can still move without GPU; it also fills the "allocator win as a function of proxy confidence quality" axis with two policies and two proxies. Prior (judgment): 25% pass, 20% kill, 55% inconclusive. Caveats: the CI conditions on the half-wise choices (the selection half is not resampled); the four cells share prompts within a policy and about 70% of their errors across proxies, so they are not four independent tests; the 0.5B and 1.5B pools share all 2,000 prompts; the learner has the frozen-arm warm start, which is the weaker side at 25% (blend50_acc is about 10% above the arm on the 0.5B Qwen3-4B pool).

### L10b: Cross-policy transfer of the prompt-cell structure, same prompts, two policies (CPU only)

**Hypothesis:** if the cell100 clustering of judge errors is structure in prompt space, it should not depend on which policy produced the completions: per-cell judge-error rates measured on one policy predict the judge errors of the other policy on the same prompts. Null: the within-pool cell signal (L9c: +0.0144 AUC [0.0033, 0.0300] on the post hoc confident half; L6a: killed; L8c-A fresh transfer 0.555 to 0.560) is pool-specific noise or prompt difficulty that agreement already carries.

**Change vs existing scripts:** new exp/l10b_cross_policy_cells.py reading runs/L8c-A-fresh-tiebreak/pool/q3_4b/feats_J.parquet (1.5B) and runs/L9c-weak-policy-tiebreak/pool/q3_4b/feats_J.parquet (0.5B); assert pid == arange and equal question columns; both carry the same cell100 (frozen centroids, runs/L8c-A-fresh-tiebreak/cells/cell100_fresh.npy). Registered in the docstring before any cross-policy statistic is computed. Statistics: (1) cell-level Spearman over the 100 cells between the shrunk judge-error rate of the two policies ((errors + 5 x base rate) / (members + 5) per policy), with a permutation null that shuffles the cell labels of prompts (2,000 permutations); (2) in each direction X from Y, the feature = Y's shrunk cell rate computed from Y's labels with the prompt's own row left out, AUC against X's judge_error on the confident half of X (top half by s_maj) and on all accepted groups, 1,000-resample group bootstrap CI; (3) references: the within-policy leave-one-out cell-rate AUC on X, and the AUC of the single prompt-level indicator "Y's majority is wrong on this prompt" (an upper bound on what any prompt-side feature can carry); (4) descriptive, the same cross-policy cell-rate AUC within accepted unanimous groups (agreement 1.0), where agreement carries no information.

**Command shape:** `cd /homes/rdubash/avrl_live/FULL/avrl; /homes/rdubash/active-audit/.venv/bin/python exp/l10b_cross_policy_cells.py --a runs/L8c-A-fresh-tiebreak/pool/q3_4b/feats_J.parquet --b runs/L9c-weak-policy-tiebreak/pool/q3_4b/feats_J.parquet --out runs/L10b-cross-policy-cells` (CPU in the shell, no gpu_run.sh).

**Metric:** the Spearman and its permutation p; the two directional confident-half AUCs with CIs and the references; the number of confident-half errors in each direction (47 and 54 are the L9c and L8c-A counts).

**Pass threshold:** cell-level Spearman >= 0.30 with permutation p < 0.01 AND both directional confident-half AUCs >= 0.58 with CI lower end > 0.50 (prompt-space cell structure is policy-independent, claim 3 gets its first positive out-of-sample read).

**Kill threshold:** Spearman <= 0.15 AND both directional confident-half AUC points <= 0.54 with CIs covering 0.5 (the cell structure does not transfer across policies, claim 3 is retired as a description of the allocator's edge). A bootstrap CI wider than 0.20 in either direction flags UNDERPOWERED. Anything else is inconclusive.

**Expected runtime:** about 3 minutes CPU (UNVERIFIED; two 2,000-row tables, 2,000 permutations and 1,000 bootstraps).

**Informs:** claim 3, with the first design that varies the policy and holds the prompts fixed, at zero GPU cost. Prior (judgment): 20% pass, 35% kill, 45% inconclusive. Caveats: a passing Spearman can reflect prompt difficulty that any agreement feature already carries (reference 3 and the unanimous-group read are there to expose that); 47 and 54 confident-half errors give an AUC SE of about 0.04 to 0.05, so the Spearman half carries the test; the 100 cells hold about 20 prompts each; one task, one prompt set, Qwen-family policies and judge.

### L10c: Blind weak-policy shard B for the budget-averaged tie-break test (GPU, then CPU; runs with or right after L9a)

**Hypothesis:** the label-free dissenting-score tie-break keeps a positive budget-averaged gain on a weak policy: on Qwen2.5-0.5B-Instruct over fresh prompts pids 4000..5999, with every rule frozen, S_tb = mean over the six budgets 10 / 15 / 20 / 25 / 30 / 40% of log(tie-break / agree_acc) is above 0 for Qwen3-4B. Null: the post hoc L9c shard-A curve (six of six budgets positive, mean log ratio 0.0185 Qwen3-4B, five of six and 0.0078 rm) is noise. The single-cell 25% test of L9c is dropped because its bars were unreachable (ceiling 1.014).

**Change vs existing scripts:** the L9a recipe (exp/l9a_rollouts_fresh.py with --ref_groups the orig-pid 0..3999 table, the prefix-equality and disjointness asserts on all 4,000 rows, the L9a analysis and replay copies, S_tb and the tier-stratified AUC registered before any shard-B label is read) with --model /homes/rdubash/avrl_live/FULL/models/Qwen2.5-0.5B-Instruct and the L9c accept rules (Qwen3-4B s_maj > 0; rm top fraction of s_maj equal to the Qwen3-4B accepted fraction on the same pool). Frozen: rm agree_acc_soth, Qwen3-4B agree_acc_sdiff. Pooled with the L9c pool (4,000 weak-policy prompts); shard B alone gives the blind sign check. The cell100 column for the new prompts follows the L9a handling; no claim 3 statistic is taken from this run.

**Command shape:** same three-job pattern as L9a under runs/L10c-weak-shardB/, `gpu_status.sh` under 2 GB before each job, jobs measured in L9c at 519 s (rollouts), 80 s (rm), 399 s (Qwen3-4B); CPU replay at the six budgets with BOOT_B=200 BOOT_SEEDS=4, MIN_CELL=1, FORCE_BEST7, 4 jobs in parallel.

**Metric:** S_tb per proxy on pooled A + B and on B alone with percentile CI; tier-stratified AUC over every partially audited block across the six budgets with bootstrap CI; tie-break / agree_acc at each budget; the measured pooled SD of S_tb.

**Pass threshold:** on Qwen3-4B pooled A + B: S_tb percentile CI lower end above 0 AND tier-stratified AUC point >= 0.62 with CI lower end > 0.50 AND S_tb on shard B alone has a positive plug-in AND measured pooled SD of S_tb <= 0.015 (a power gate, otherwise UNDERPOWERED), all gates passing; rm S_tb plug-in positive is a flag, not a pass condition.

**Kill threshold:** Qwen3-4B pooled S_tb plug-in <= +0.005 with gates passing and the SD gate met (the L9c shard-A gain did not hold blind). Anything else is inconclusive.

**Expected runtime:** about 20 minutes GPU in three jobs (519 + 80 + 399 s measured in L9c) plus about 5 minutes CPU for the replay (UNVERIFIED for six budgets).

**Informs:** the label-free-order pivot in a second error regime, with a statistic that can see a 1% to 3% effect, and the E06 regime-variation branch. Prior (judgment): 40% pass, 15% kill, 45% inconclusive. Caveats: shard A is not blind for S_tb because the closed-form curve was computed after the labels were read, which is why the pass needs a positive shard-B-alone sign; the weak policy is screened better by both proxies (judge sanity AUC 0.91), so the J pool grows only about 1.5x; both proxies are Qwen-family and a non-Qwen judge is not available locally (only OLMo-2-0425-1B and its instruct variant under active-audit/models, unlikely to clear the 0.70 sanity gate), so that control stays blocked.

### Iteration 11 (proposed after L10a-xfit-fixed-comparator, 2026-10-19; verdict inconclusive: pooled C plug-in +0.0134, percentile CI [-0.0243, +0.0251], 2 of 4 cells positive; PASS needed +0.02 with a positive lower end, KILL needed an upper end below +0.01)

Iteration 11 = the loop run that executed L10a (CPU only, 24 jobs of 200 x 4 resamples, 3.3 minutes wall, about 13 minutes in all). Findings that shape the specs below: (a) greedy_blend7 against the cross-fitted best fixed rule CH is +1.3% pooled over four (policy, proxy) cells and six budgets (ratio 1.013 [0.976, 1.025], SD 0.0131); the cross-fit removed only 1.3 caught errors (1% of the in-sample best), so the earlier in-sample comparator was nearly fair, and a cross-fitted fixed rule already holds +0.064 of the learner's +0.078 over agree_acc (5/6). (b) The edge is a mid-budget feature of the data: pooled log G/CH by budget 10 / 15 / 20 / 25 / 30 / 40% is -0.023 / +0.034 / +0.056 / +0.040 / +0.001 / -0.028; the 15 / 20 / 25% window reads +0.0432 [-0.0076, +0.0578] but was chosen after seeing the pattern. (c) The resampling instrument penalises the learner: bootstrap mean minus plug-in is -0.0118 for G/CH, -0.0098 for G/agree_acc, -0.0091 for G/frozen arm and about 0 (+0.002 to -0.0006) for every fixed-against-fixed contrast, so the percentile CI sits one SD left of the plug-in; the verdict is the same under percentile, basic and bias-centred intervals because the plug-in misses the +0.02 bar. (d) Power: pooled SD 0.0131 at 2,000 prompts per pool; separating a 1% from a 2% edge needs a SD near 0.005, about 13,700 prompts per pool (UNVERIFIED 1 / sqrt(n) scaling), more than GSM8K train holds (7,473), so only an edge of about 3% or more, or its absence, is decidable on this task. L10b (claim 3), L9a, L9b and L10c from Iteration 10 stay queued; L11c amends L9b and L11b registers a claim 2 statistic inside L9a and L10c.

### L11a: Duplicate-free uncertainty and a split-averaged cross-fit for the L10a statistic (CPU only; replaces follow-up (a) of the L10a ledger entry)

**Hypothesis:** the pooled C of L10a (+0.0134) is located and bounded too pessimistically: the parity split is one draw of the cross-fit, and the bootstrap resamples contain duplicated prompts that the learner refits on (learner-only bias -0.009 to -0.012). With 20 random 50/50 splits averaged and a duplicate-free subsampling scheme, the learner's edge over the cross-fitted best fixed rule is estimated with less bias and less noise. Null: the edge is still below 1% (split-averaged plug-in under +0.01) and the L10a interval was wide for ordinary reasons.

**Change vs existing scripts:** copy exp/l10a_boot_xfit_fixed.py to exp/l11a_xfit_nodup.py and add mode XFIT=3 after every existing random draw (gate: with XFIT=2 the copy reproduces runs/L10a-xfit-fixed-comparator/<pool>/<proxy>/xfit_f0.25.json exactly, all four pools). XFIT=3: (1) point estimate: S = 20 splits of each pool's 2,000 groups into two random halves of 1,000 (split 0 is the parity split of L10a, splits 1 to 19 come from permutations of default_rng(777)); each split is run in both directions exactly as XFIT=2 (CH = argmax of the 8-seed mean caught of the seven fixed rules on the other half, first listed wins ties; greedy_blend7 warm-started from the frozen arm, rm agree_acc_soth, Qwen3-4B agree_acc_sdiff, no reselection); plug-in C per cell = mean over splits and the six budgets of log(greedy_blend7 / CH) from the summed halves; pooled C = unweighted mean of the four cells. (2) uncertainty, split 0 only: 200 subsamples per half, each keeping 800 of the 1,000 held-out groups drawn WITHOUT replacement (default_rng(12345) per half, one permutation per subsample), 4 seeds each, CH fixed at the value chosen from the full other half (the conditioning of L10a, stated), SD_jack = sqrt((n - d) / d) x SD of the 200 subsample statistics = 2 x SD_sub for n = 1000, d = 200; CI = split-averaged plug-in +/- 1.96 SD_jack, with the L10a percentile CI repeated alongside. (3) bias diagnostic registered before the run: mean over the subsamples minus the full-half plug-in for G/CH, G/agree_acc, G/frozen arm, CH/agree_acc, frozen arm/agree_acc; expected |bias G/CH| <= 0.004 if duplicates were the cause; a bias that stays near -0.01 means the cause is the smaller effective sample (800 groups), not duplication, and the SD_jack is read with that offset. Register the bins in the docstring before any XFIT=3 job runs.

**Command shape:** same fan-out as exp/L10a-xfit-fixed-comparator.sh with the new script and XFIT=3 (24 jobs, 16 in parallel, OMP_NUM_THREADS=1, FORCE_BEST7 as in L10a), launched with gpu_run.sh because it is CPU but long running, then a finalizer exp/l11a_finalize.py. Features string as in runs/L9c-weak-policy-tiebreak/boot_jobs.txt.

**Metric:** split-averaged pooled C and per-cell C, the split-to-split SD of pooled C across the 20 splits, SD_jack and the CI half-width, the bias diagnostic table, C against agree_acc and against the frozen arm (context), stability of the CH choice across splits (share of (cell, budget) choices equal to the majority choice).

**Pass threshold:** split-averaged pooled C >= +0.02 AND CI lower end > 0 AND positive plug-in in at least 3 of 4 cells, all gates passing (the L10a bars, unchanged so the goalposts do not move).

**Kill threshold:** CI upper end < +0.01. Anything else is inconclusive; half-width above 0.03 with neither bin firing is POWER-LIMITED.

**Expected runtime:** about 10 minutes CPU (UNVERIFIED; L10a took 3.3 minutes for 24 jobs at 16 in parallel, and the 19 extra splits cost only full-half runs of 8 seeds each).

**Informs:** claim 2 and the instrument under every learner CI so far (L1a, L2a, L8c-A, L9c, L10a; L2a already saw a bootstrap mean of 0.999 against a plug-in of 1.084). It adds no data, so it moves the verdict only if the split-averaged plug-in leaves the 0.01 to 0.02 gap or the SD_jack is far below 0.0131. Prior (judgment): 10% pass, 20% kill, 70% inconclusive. Caveats: the delete-d jackknife is exact only for smooth statistics, CH is an argmax and is conditioned on; the four cells share prompts (error-set overlap 0.79 to 0.83 within a policy, 0.42 to 0.49 across policies); the 20 splits share the same 2,000 prompts, so averaging them removes split noise, not sampling noise.

### L11b: Blind cross-shard test of the mid-budget learner edge, registered inside L9a and L10c (GPU, then CPU)

**Hypothesis:** the learner's cross-fitted edge, which L10a located post hoc at budgets 15 / 20 / 25% (pooled +0.0432, nothing at 10 / 30 / 40%), holds on fresh prompts. With the comparator CH_A chosen on all of shard A (pids 2000..3999) and the learner warm-started from the frozen arm, C_B = mean over budgets 15 / 20 / 25% of log(greedy_blend7 / CH_A), scored on shard B (pids 4000..5999), is above zero pooled over four cells (1.5B from L9a, 0.5B from L10c, each with rm and Qwen3-4B). Null: the window is a winner's curse (3 of 6 budgets picked on the same data) and the true edge is at most 1%.

**Change vs existing scripts:** no new GPU recipe. Run L9a (1.5B shard B) and L10c (0.5B shard B) exactly as specified, then add a CPU analysis: exp/l11b_cross_shard.py = copy of exp/l11a_xfit_nodup.py with mode XFIT=4, in which the selection pool is shard A in full (runs/L8c-A-fresh-tiebreak/pool/<proxy>/feats_J.parquet for 1.5B, runs/L9c-weak-policy-tiebreak/pool/<proxy>/feats_J.parquet for 0.5B; 2,000 groups) and the held pool is shard B in full (2,000 groups, accept rules of L9a and L10c: Qwen3-4B s_maj > 0, rm top fraction of s_maj equal to the Qwen3-4B accepted fraction on the same pool, both label-free). CH_A is the argmax of the 8-seed mean caught of the seven fixed rules on shard A at each budget; the learner runs on all 2,000 shard B groups (twice its L10a data). Uncertainty by the L11a delete-d scheme (200 subsamples keeping 1,600 of 2,000 groups) if L11a shows the learner bias vanishes, otherwise by the L10a bootstrap with the bias stated. Budgets 15 / 20 / 25% primary (300 / 400 / 500 audits), 10 / 30 / 40% and the six-budget C_B context. Register everything in the docstring before any shard B label is read. Gate: with XFIT unset the copy reproduces the L8c-A and L9c boot_tb_f0.25.json values (agree_acc, blend50_acc, greedy_blend7) exactly.

**Command shape:** the L9a three-job GPU pattern for the 1.5B policy and the L10c pattern for the 0.5B policy (gpu_status.sh under 2 GB before each job, one job at a time, six jobs in all, each well under the 45 minute cap), then `XFIT=4 MIN_CELL=1 FORCE_BEST7=<arm> OMP_NUM_THREADS=1 BOOT_B=200 BOOT_SEEDS=4 BOOT_FRAC=<f> /homes/rdubash/active-audit/.venv/bin/python exp/l11b_cross_shard.py <shard B feats_J> judge_error cell100 <FEATS> <out.json>` per cell and budget, 24 jobs, 16 in parallel.

**Metric:** pooled C_B over 15 / 20 / 25% with CI and measured SD, per-cell C_B, the six-budget C_B and the 10 / 30 / 40% window as context, C_B against agree_acc and against the frozen arm, CH_A identity per budget, the gap between CH_A on shard B and the in-sample best on shard B (winner's curse removed).

**Pass threshold:** pooled C_B (15 / 20 / 25%) plug-in >= +0.025 AND CI lower end > 0 AND a positive plug-in in at least 3 of 4 cells AND measured SD <= 0.013 (power gate, otherwise UNDERPOWERED), all gates passing; claim 2 then has a blind cross-fitted confirmation in one budget window and the learner is carried into Phase 1 as an arm for audit fractions of about 15 to 25%.

**Kill threshold:** pooled C_B plug-in <= +0.01 AND at most 2 of 4 cells positive, gates passing and the SD gate met (the post hoc window did not replicate; claim 2 as worded is dropped and the deliverable is the agreement threshold plus the fixed blend and the dissenting-score tie-break). Anything else is inconclusive.

**Expected runtime:** GPU about 40 minutes in six jobs (737 + 77 + 361 s measured for the 1.5B pattern in L8c-A, 519 + 80 + 399 s for the 0.5B pattern in L9c), CPU about 10 minutes (UNVERIFIED for 2,000-group held pools).

**Informs:** claim 2, as the only follow-up that adds prompts, tests the post hoc window blind and removes the data deficit of caveat 5 of L10a; it shares its GPU cost with L9a and L10c, which are queued anyway for the tie-break, so the marginal cost is the CPU analysis. Expected pooled SD about 0.012 (UNVERIFIED: the L10a window SD 0.0175 divided by sqrt 2), so an edge of 4% would be seen at about 3 SD and an edge of 2% at about 1.6 SD. Prior (judgment): 25% pass, 35% kill, 40% inconclusive. Caveats: the 15 / 20 / 25% window was picked from the L10a pattern, which is why the bars are registered on shard B alone; the 0.5B policy is a different regime from the 1.5B one (the 0.5B cells were flat in L10a); both proxies are Qwen-family and share about 80% of their errors within a policy; one task.

### L11c: Natural-split cross-fit for the two 1.5B cells, an amendment to the queued L9b (CPU only)

**Hypothesis:** with the fixed comparator chosen on a disjoint, older pool (pids 0..1999) and both comparator and learner scored on shard A (pids 2000..3999, 2,000 held groups), the learner's edge is the same size as in the half-split of L10a (1.5B Qwen3-4B +0.042, 1.5B rm +0.017) and a frozen static logistic model reaches most of it. Null: the learner's edge over CH_0 is at most 1% and static_logit adds nothing over the best fixed rule.

**Change vs existing scripts:** run L9b as specified (exp/l9b_static_logit.py and exp/l9b_boot_static.py, fit on pids 0..1999, frozen, evaluated on shard A) and add two comparators to its replay, appended after every existing random draw: CH_0 = the best of the seven fixed rules by 8-seed mean caught on the pids 0..1999 pool (runs/L5c-proxies/rm/feats_J.parquet and runs/L3b-judge-proxy/feats_J.parquet; pre-check that the columns the seven rules need, agreement, n_distinct, s_maj, s_oth, s_diff, accept, exist there, and log the item infeasible if not), chosen separately for each budget 10 / 15 / 20 / 25 / 30 / 40%. Statistics: mean over six budgets of log(greedy_blend7 / CH_0) and log(static_logit / CH_0), per cell and averaged over the two cells, with the L11a delete-d interval (200 subsamples of 80% of shard A) or the L9b bootstrap if L11a has not run.

**Command shape:** CPU in the shell as in L9b (no gpu_run.sh), BOOT_B=200 BOOT_SEEDS=4, MIN_CELL=1, FORCE_BEST7, six budgets for each of the two proxies.

**Metric:** the two averaged log ratios and per-cell values with CI, the six per-budget ratios, CH_0 identity per budget, the L9b ratios (static_logit / greedy_blend7) at 10 / 25 / 40%.

**Pass threshold:** greedy_blend7 against CH_0, averaged over the two cells: log ratio >= +0.02 with CI lower end > 0 and positive in both cells (a blind, 2,000-group, disjoint-pool confirmation; still only the 1.5B policy). The L9b bins stay as registered and are judged separately.

**Kill threshold:** the same quantity with CI upper end < +0.01. Anything else is inconclusive.

**Expected runtime:** about 10 minutes CPU on top of the L9b 5 minutes (UNVERIFIED).

**Informs:** claim 2 from the cleanest split the existing data allow (selection pool disjoint from the evaluation prompts, learner on 2,000 groups), and whether the remaining edge needs online adaptation (static_logit at least 0.98 of greedy_blend7, the L9b pass, would say it does not). Two cells only and the shard A prompts are the same ones inside the 1.5B halves of L10a, so it cannot carry the verdict alone. Prior (judgment): 20% pass, 25% kill, 55% inconclusive. Caveats: the seven candidate rules include two tie-breaks found post hoc on the pids 0..1999 pools, so CH_0 is not free of selection on that pool (it is free of shard A); the frozen arm already equals one of the seven rules, so G/frozen and G/CH_0 will be close when the frozen arm is also CH_0 at that budget.

### Iteration 12 (proposed after L9a-shardB-tiebreak, 2026-10-19; verdict PASS by the registered bins on both proxies: pooled S_tb +0.0133 rm and +0.0281 Qwen3-4B, shard B alone +0.0192 and +0.0158, measured SD 0.0048)

Iteration 12 = the loop run that executed L9a (GPU 20 minutes, CPU replays 10 minutes, 38 minutes wall). Findings that shape the specs below: (a) the dissenting-score tie-break replicates on fresh prompts, but the gain is small (1.3% rm, 2.9% Qwen3-4B of what agree_acc catches, budget-averaged) and lives at the 25% and 30% budgets; the rm pooled pass is marginal under a bias-aware interval (basic CI [-0.0002, +0.0175]) and the Qwen3-4B shard-B-alone basic CI covers zero ([-0.0060, +0.0224]), so the two weaknesses sit in different cells. (b) The cumulative tier counts say the six budgets cut into the 7/8 agreement tier (516 groups, AUC 0.724 rm; 561 groups, AUC 0.763 Qwen3-4B) at 25% and 30%, at a boundary at 15% (rm) and past it at 40%: the gain looks like a function of where the cut lands. (c) The learner: Qwen3-4B 1.5B blind C_B (15/20/25%, comparator from shard A, scored on shard B) is +0.1344 [+0.0382, +0.1646] and greedy_blend7 is 1.09 to 1.13 times the in-sample best fixed rule at 15/20/25% pooled, while rm is -0.0001 and the L10a 0.5B cells were flat, so the learner edge is, so far, one cell. (d) L9b and L11c are still unrun and L10c is next on the GPU.

Queue order: L12a (CPU), then L10c (GPU, as written; run it with the L9a analysis copy so S_tb, the stratified AUC and C_B come out under one registration), then L12b (CPU).

### L12a: Frozen static logistic scorer and an out-of-pool fixed comparator on 4,000 fresh groups, both 1.5B cells (CPU only; run in place of L9b and L11c)

**Hypothesis:** the learner's blind edge for Qwen3-4B at 1.5B (L9a: C_B +0.1344 [+0.0382, +0.1646] over 15/20/25% against a comparator chosen on shard A; greedy_blend7 / in-sample best fixed 1.095 / 1.130 / 1.092 pooled) survives a fixed comparator chosen on the older pids 0..1999 pool, and it is a static function of the label-free columns, so one logistic model fitted once on pids 0..1999 and frozen reaches most of it. Null: the edge is specific to shard A as the selection pool or needs online adaptation.

**Change vs existing scripts:** run the L9b recipe (exp/l9b_static_logit.py and exp/l9b_boot_static.py as specified in L9b: LogisticRegression(C=1.0, max_iter=1000) on standardized [agreement, n_distinct, s_maj, s_oth], fitted once on the accepted groups of pids 0..1999, runs/L5c-proxies/rm/feats_J.parquet and runs/L3b-judge-proxy/feats_J.parquet, label judge_error, coefficients saved before any label of the evaluation pool is read) and add the L11c comparator CH_0 (per budget, the best of the seven fixed rules by 8-seed mean caught on the pids 0..1999 pool; the L9a gate (i) already replays those two files, so the columns exist). Evaluate on the pooled A + B pool, runs/L9a-shardB-tiebreak/pool/{rm,q3_4b}/feats_J.parquet (4,000 groups, pooled accept flags), and on shard B alone (runs/L9a-shardB-tiebreak/pool_B/<proxy>/feats_J.parquet) as the sign check. Gate: with the new arms removed every existing column reproduces runs/L9a-shardB-tiebreak/boot/<proxy>/boot_tb_f<f>.json exactly. Register everything in the docstring before the first replay; the static model and CH_0 are fixed before any pooled-pool number exists for them. Budgets 10/15/20/25/30/40%, 200 x 4 resamples with default_rng(12345), the L11a delete-d interval if L11a has run, otherwise the L9a bootstrap with the learner bias stated.

**Command shape:** CPU in the shell as in L9a (replay jobs chunked 4 x 50 resamples, parallel), no gpu_run.sh: `MIN_CELL=1 FORCE_BEST7=<arm> OMP_NUM_THREADS=1 BOOT_B=200 BOOT_START=<50j> BOOT_END=<50j+50> BOOT_SEEDS=4 BOOT_FRAC=<f> /homes/rdubash/active-audit/.venv/bin/python exp/l12a_boot_static.py <pooled feats_J> judge_error cell100 <FEATS> <out.json>` per proxy, budget and chunk (the job pattern of runs/L9a-shardB-tiebreak/jobs_AB.txt; exp/l12a_boot_static.py = copy of exp/l9a_boot_forced.py with the static_logit arm and the CH_0 comparator appended after every existing random draw), merged by the l9a_analyze.py --stage merge pattern.

**Metric:** per proxy: D_G = mean over 15/20/25% of log(greedy_blend7 / CH_0), D_S = same for static_logit, D_GS = mean log(static_logit / greedy_blend7), each with percentile CI and measured SD on the pooled pool and plug-in on shard B alone; six-budget versions and per-budget ratios as context; CH_0 identity per budget and the gap between CH_0 and the best rule in hindsight on A + B (winner's curse removed).

**Pass threshold:** Qwen3-4B, pooled A + B: D_G plug-in >= +0.05 AND percentile CI lower end > 0 AND D_G on shard B alone positive, gates passing and measured SD <= 0.02 (otherwise UNDERPOWERED). Then read the fork: D_GS >= -0.02 means static (the deliverable becomes a frozen scorer plus tie-break, no online learner, and claim 2 is reworded from a learned allocator to a learned scorer); D_GS <= -0.06 means online (claim 2 is supported in its original form for this cell, still one cell and one policy).

**Kill threshold:** Qwen3-4B pooled D_G plug-in <= +0.02 with gates passing (the blind +0.13 was specific to the shard-A comparator or to shard B, and the learner comparison joins the closed items). rm is reported and is not a bin (its learner edge is zero in L9a). Anything else is inconclusive.

**Expected runtime:** about 20 minutes CPU (UNVERIFIED; L9a's CPU replays took 365 s rm and 237 s Qwen3-4B for 24 chunk jobs each, this adds one arm, one more selection pass and a shard-B-alone run).

**Informs:** claim 2 only, through the largest blind effect in the loop, and it turns it into a fork that changes what is built (frozen scorer versus online learner). Prior (judgment): 50% pass, 15% kill, 35% inconclusive; given a pass, 50% static and 50% online (UNVERIFIED). Caveats: one cell is expected to carry it (rm and the 0.5B cells were flat); the proposer has seen the learner numbers on both shards (the bins are set before any static-model number exists, and the static model has never been scored); the instrument penalises the learner (C_B bootstrap mean 0.1035 against plug-in 0.1344), so percentile CIs are conservative for it; CH_0 is chosen on a pool where the two tie-breaks were found post hoc; Qwen-family policy and judge, one task.

### L12b: Continuous-budget tie-break gain and a first-order tier-landing prediction (CPU only)

**Hypothesis:** the tie-break gain is a mechanical function of where the budget cut lands in the agreement-tier structure: zero at tier boundaries and in the unanimous block, positive inside a tier whose within-tier AUC exceeds 0.5. Then the budget-averaged gain is not a post hoc pick of six budgets and can be predicted from the tier table. Null: the 25% and 30% pattern is chance on six points, and a dense grid shows no relation to the tier landing.

**Change vs existing scripts:** reuse exp/l9a_boot_forced.py unchanged (BOOT_FRAC per budget; FORCE_BEST7 as in L9a) on runs/L9a-shardB-tiebreak/pool/{rm,q3_4b}/feats_J.parquet. (1) Plug-in grid: budgets 5% to 45% in 1% steps, 8 seeds, resampling off (BOOT_B=0 is accepted by the script and writes the full-pool block only), for agree_acc and the verdict arm. (2) Interval grid: budgets 10% to 40% in 2% steps (16 budgets), BOOT_B=200 BOOT_START=0 BOOT_END=100 BOOT_SEEDS=2, i.e. the first 100 resamples of the same default_rng(12345) stream (same index arrays at every budget), giving S_dense = mean over the 16 budgets of log(tie-break / agree_acc) per resample. (3) Prediction, registered before any replay: for each grid budget b let t be the sub-unanimous tier containing the cut (tiers and counts from the pooled feats_J: n_t groups, E_t judge errors, AUC a_t from the L9a tier table), phi = fraction of tier t audited, and the predicted gain DeltaE_pred(b) = E_t x 4 x (a_t - 0.5) x phi x (1 - phi), zero outside sub-unanimous tiers (a scale-free first-order shape, used only for rank correlation). Spearman correlation across the 1% grid between DeltaE_pred and the observed gain in caught errors (tie-break minus agree_acc, 8-seed means).

**Command shape:** CPU in the shell, 31 plug-in jobs plus 16 interval jobs per proxy, 16 in parallel, OMP_NUM_THREADS=2, then a short finalizer exp/l12b_final.py. Gate: the 10%, 20%, 30%, 40% points coincide with the L9a plug-ins (agree_acc and verdict arm 8-seed means equal to the full_pool values in runs/L9a-shardB-tiebreak/boot/<proxy>/boot_tb_f<f>.json).

**Metric:** S_dense with percentile and basic CI, the observed gain curve (errors) over the 1% grid, the Spearman and its permutation p (2,000 shuffles of the budget labels), the per-tier mean observed gain against DeltaE_pred, the budget fractions where the tie-break gain is at least 1% of agree_acc catches.

**Pass threshold:** on BOTH proxies S_dense percentile CI lower end > 0 AND basic CI lower end > 0 (the rm weak point of L9a) AND Spearman >= 0.5 with permutation p < 0.01; gates passing.

**Kill threshold:** S_dense plug-in <= +0.005 on BOTH proxies (the six-budget gain was a lucky grid). Anything else is inconclusive.

**Expected runtime:** about 15 minutes CPU (UNVERIFIED; the 16 interval jobs at 100 x 2 resamples cost about half an L9a chunk job each, 31 plug-in jobs are cheap).

**Informs:** the deliverable's statement of when the tie-break pays (which audit fractions) and the two L9a weaknesses (post hoc budget list, rm basic CI), at zero GPU cost; it adds no data, so it cannot move claim 2 and cannot change the PASS. Prior (judgment): 30% pass, 5% kill, 65% inconclusive. Caveats: the tier AUCs a_t come from the same pooled data (in-sample for the mechanism, fine for a shape test, not for a gain forecast); the predictor is first order; the 2% grid is still a finite choice of budgets; Qwen-family proxies, one policy, one task.
