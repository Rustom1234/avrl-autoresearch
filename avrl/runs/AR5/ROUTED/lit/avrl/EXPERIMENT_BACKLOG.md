# AVRL Experiment Backlog

Pre-registered, 2026-09-30. Twenty experiments, E01 through E20, ordered by expected information value per GPU hour: cheapest, highest-leverage checks first, most expensive and most infrastructure-blocked last. This ordering means analysis-only reruns of already-collected data (near-zero marginal GPU cost) come before any new generation run, and new generation runs on this machine come before any GRPO training run on an A100.

## Claim numbering (from PROJECT.md)

- Claim 1: the learned allocator reaches the same policy quality with a fraction of expensive verifier calls.
- Claim 2: the learned allocator admits fewer corrupted rewards into training than any fixed rule at the same budget.
- Claim 3: the allocator's edge comes from structure in prompt space (proxy error clusters), the SOFisher condition (Li et al., 2026, Nature Communications, https://doi.org/10.1038/s41467-026-73404-6).

Source for the existing harness read this session: /homes/rdubash/avrl_live/ROUTED/avrl/PROJECT.md, /homes/rdubash/avrl_live/ROUTED/avrl/p0_rollouts.py, /homes/rdubash/avrl_live/ROUTED/avrl/p0_analyze.py, /homes/rdubash/avrl_live/ROUTED/avrl/allocators.py. Confirmed at read time: avrl/p1_grpo.py (the phase-1 GRPO training harness named in PROJECT.md's Files section) does not exist on disk yet, only p0_rollouts.py, p0_analyze.py, and allocators.py do. Every experiment below that needs an actual policy-training loop is marked NEEDS PHASE-1 HARNESS.

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

All paper citations above are carried forward unchanged from /homes/rdubash/avrl_live/ROUTED/lit/avrl/SYNTHESIS.md, read this session, which itself states its method: built via WebFetch against arXiv abstracts, the OpenAlex works API, and direct publisher DOIs, with WebSearch unavailable (budget error) and not retried. This document adds one new fact verified this session via WebFetch: HuggingFaceH4/MATH-500's existence, size, and content, https://huggingface.co/datasets/HuggingFaceH4/MATH-500. No paper, author, or number is invented in this document; every runtime figure not directly read from avrl/runs/p0/run.log is explicitly marked as an estimate or UNVERIFIED. G01-G04 were added after this session's three gap checks (GAP_already-done.md, GAP_cannot-work.md, GAP_not-a-thesis.md); their citations (VIP, https://arxiv.org/abs/2602.01601) are carried forward unchanged from SYNTHESIS.md and the gap-check files.

## Compute correction (2026-09-30, main session)
This machine's GPU is an NVIDIA RTX A4000 with 16 GB, not a 32 GB card. Measured with the phase-1 harness (Qwen2.5-1.5B, LoRA r=32, bf16): full settings (16 prompts x G=8, 512 tokens) run at 153 s per step with HF generation and peak 13.2 GB; vLLM colocate does not fit beside training on this card (out of memory at engine start). Smoke settings run at 20 s per step. vLLM-only rollout experiments (phase-0 style) are unaffected and fit in 40 minutes at 1,500 to 3,000 prompts.

## Feasibility note (this run)
Items marked A100, needing a larger cluster, a second machine, or the phase-1 GRPO harness (E15 to E20, G04, anything needing p1_grpo.py; about 89 s per step on this card, so 100 steps take 2.5 hours) are not runnable under the 45-minute cap and must not be planned.

## Proposed by the loop

Proposed 2026-10-18 after G01 (verdict KILL, avrl/runs/G01/, RESEARCH_LOG.md). Same format as E01 to E20. All three are analysis-only, CPU only, no GPU job. Ids are L<iteration><letter>; these come from iteration 1. E04 to E06 stay unlaunched, since their gate (G01 inconclusive or pass) is not met.

## L1a: Split robustness of the G01 replay (analysis-only)

**Status: done 2026-10-18, not rerun.** The G01 critic ran it on 16 split seeds (avrl/runs/G01_critic/split_robustness.txt): KILL on 16 of 16 splits. Largest greedy minus recal gap at 10% or 25% is +2.91 pp (split 14, 25%).

**Hypothesis:** the G01 kill (greedy within 3 pp of recal or below it at all four budgets) holds across different fit/sim splits, not only the one split used.

**Change vs existing scripts:** no change to the policies. Add an outer loop in a driver around avrl/exp/g01_label_free_baselines.py that changes the seed of the bucket shuffle (8 split seeds), reruns the four budgets, and records greedy minus recal per split.

**Command shape:** /homes/rdubash/active-audit/.venv/bin/python -I on feats.parquet, label proxy_error, group cell100, the G01 cheap features and budgets, once per split seed.

**Metric:** greedy minus recal, pp of hit rate, per budget, with the range over split seeds.

**Pass threshold:** the G01 pass rule (at least 5 pp at 10% or 25%, above 2x seed sd) holds on a majority of split seeds.

**Kill threshold:** the G01 kill rule holds on every split seed. Anything between is inconclusive.

**Expected runtime:** under 10 min, CPU.

**Informs:** claim 2. Expected information value high: it turns the single-split caveat into a measured number before the kill is treated as settled.

## L1b: Strongest label-free rule as the bar for claim 2 (analysis-only)

**Status: done 2026-10-18, PASS by the rule.** Held-out gap, tuned minus plain fixed: +3.96 pp at 5% and +5.38 pp at 10% (splits 4 to 7, all four splits positive at both). No pass at 25% (+0.84) or 50% (+0.24). Held-out splits share documents with the tune splits and one rollout pool, so the result is not independent of the pool. Details in avrl/runs/L1b/ and RESEARCH_LOG.md.

**Hypothesis:** a label-free rule with tuned settings (Beta prior strength, refit cadence R, stratum grid) beats the plain fixed agreement threshold by more than 2 pp at some budget, which would move the comparator for claim 2 away from the plain threshold.

**Change vs existing scripts:** a new driver that sweeps the prior pseudo-count (for example 2, 10, 50), R (budget/16, budget/8, budget/4), and the stratum grid (agreement only; agreement x n_distinct; agreement x n_distinct x any_truncated). Tune on split seeds 0 to 3 only; report on split seeds 4 to 7.

**Command shape:** /homes/rdubash/active-audit/.venv/bin/python -I on feats.parquet with the G01 arguments.

**Metric:** hit rate per budget for the tuned rule, the plain fixed threshold and recal, on the held-out split seeds.

**Pass threshold:** a tuned label-free rule beats the plain fixed threshold by more than 2 pp at some budget on held-out splits. The bar for claim 2 then becomes that rule.

**Kill threshold:** no tuned rule beats the plain fixed threshold by more than 2 pp at any budget. The plain fixed threshold stays the bar.

**Expected runtime:** under 15 min, CPU.

**Informs:** claim 2, which is stated against any fixed rule. Expected information value high: recal already loses to plain fixed at 10% (72.0 vs 65.8), so the comparator is not yet known.

## L1c: Learned-arm ceiling with a stronger scorer (analysis-only)

**Hypothesis:** a stronger scorer on the same cheap features (gradient-boosted trees in place of the logistic model) lifts greedy above recal by at least 5 pp at 10% or 25%. Null: it does not, because agreement carries almost all the signal.

**Change vs existing scripts:** a copy of g01_label_free_baselines.py with the greedy scorer swapped for a class-balanced gradient-boosted classifier. Also report held-out AUC for the logistic scorer, the boosted scorer, and agreement alone. The rest is untouched.

**Command shape:** /homes/rdubash/active-audit/.venv/bin/python -I on feats.parquet with the G01 arguments.

**Metric:** greedy minus recal, pp of hit rate, per budget; held-out AUC for the three scorers.

**Pass threshold:** the G01 pass rule holds.

**Kill threshold:** the G01 kill rule holds.

**Expected runtime:** under 15 min, CPU.

**Informs:** claims 2 and 3. Expected information value medium: the likely outcome is a kill, which would support claim 3 not supported on this pair. A pass would show headroom for the learned arm.

## Proposed by the loop, iteration 2 (after L1b)

Proposed 2026-10-18 after L1b (verdict PASS by the rule, avrl/runs/L1b/, RESEARCH_LOG.md). Same format as L1a to L1c. L2a and L2c are analysis-only, CPU only. L2b needs one GPU job through gpu_run.sh and is not launched.

## L2a: Disjoint held-out pool for the tuned rule (analysis-only)

**Hypothesis:** the L1b held-out gain (tuned rule minus plain fixed, +3.96 pp at 5% and +5.38 pp at 10%) survives when the tune and held-out documents are disjoint, not only when the splits share one rollout pool.

**Change vs existing scripts:** a copy of avrl/exp/l1b_tuned_threshold.py. Assign the 99 cell100 groups (after the MIN = 3 filter) to two disjoint pools, one for tuning and one for held-out, with no group in both. Run the same fit cut (first quarter of groups) inside each pool. Repeat with the roles swapped. Add greedy_online (the G01 learned arm) to the held-out evaluation so the 25% bar can be set. The rest is untouched.

**Command shape:** /homes/rdubash/active-audit/.venv/bin/python -I on feats.parquet with the L1b arguments, one run per direction.

**Metric:** gap of the selected tuned config minus plain fixed, pp of hit rate, per budget, paired by tie-break seed. Report the mean over tie-break seeds and a bucket-level bootstrap sd (1,000 resamples of groups) in the held-out pool. Also greedy_online minus plain fixed at 25%. Each pool has about 1,500 sim docs, so 5% budgets are about 55 calls and the intervals will be wider than in L1b.

**Pass threshold:** the held-out gap is above +2.0 pp at 5% or 10% in both directions.

**Kill threshold:** the held-out gap is at or below +2.0 pp at both budgets in both directions. Anything between is inconclusive.

**Expected runtime:** about 15 min, CPU.

**Informs:** claim 2 (the bar at 5% and 10%) and the pool caveat on L1b. Expected information value high: it is the cheap check on whether the L1b pass is pool-specific, and it gives the learned arm a held-out number at 25%.

## L2b: Fresh rollout pool for the tuned rule (GPU, one job)

**Hypothesis:** the 5% and 10% gains of the L1b tuned rule reproduce on a rollout pool that L1b did not use, with the two configs fixed in advance from L1b and no retuning.

**Change vs existing scripts:** copy avrl/p0_rollouts.py to a new script and run it on a different 3,000-prompt slice of GSM8K train, disjoint from the L1b pool (check the size of avrl/data first). Keep the rollout settings of p0 (Qwen2.5-1.5B, 8 rollouts, temperature 1.0). Then run a copy of the L1b script with the fixed configs only: k0 2, R = max(b//8, 10), agreement x n_distinct at 5%, and k0 50, R = max(b//8, 10), agreement x n_distinct x any_truncated at 10%. Plain fixed is the comparator. Start at 2,000 prompts to stay clear of the 45-minute cap.

**Command shape:** the rollout step through /homes/rdubash/avrl_live/ROUTED/bin/gpu_run.sh with the vLLM python (.venv-judge), then the CPU replay with /homes/rdubash/active-audit/.venv/bin/python -I on the new feats file.

**Metric:** held-out gap of each fixed config minus plain fixed, pp of hit rate, mean and sd over the held-out split seeds.

**Pass threshold:** the gap is above +2.0 pp at 5% or 10% (both configs fixed before the run).

**Kill threshold:** the gap is at or below +2.0 pp at both budgets. Anything between is inconclusive.

**Expected runtime:** about 40 min GPU (rollouts, under the 45-minute cap), then under 5 min CPU.

**Informs:** claim 2 at 5% and 10%. Expected information value highest of the three: it is the only test here that separates the L1b gains from the single rollout pool they came from.

## L2c: Prompt-space stratum for the tuned rule (analysis-only)

**Hypothesis:** adding a coarse prompt-space cell to the stratum key of the tuned rule beats the L1b selected config by more than 2 pp at 5% or 10% on held-out splits. Null: it does not.

**Change vs existing scripts:** a copy of avrl/exp/l1b_tuned_threshold.py with two added stratum keys, agreement x cell30 and agreement x cell30 x n_distinct. Agreement must stay in every key, because the Beta prior is defined per stratum from agreement. The sweep and tuning rule are otherwise unchanged (tune on split seeds 0 to 3, report on 4 to 7).

**Command shape:** /homes/rdubash/active-audit/.venv/bin/python -I on feats.parquet with the L1b arguments.

**Metric:** held-out gap of the selected config minus plain fixed, and minus the L1b selected config at the same budget, pp of hit rate, mean and sd over the held-out splits.

**Pass threshold:** a prompt-space stratum beats the L1b selected config by more than 2.0 pp at 5% or 10% on held-out splits.

**Kill threshold:** no prompt-space stratum beats the L1b selected config by more than 2.0 pp at either budget. Anything between is inconclusive.

**Expected runtime:** under 15 min, CPU.

**Informs:** claim 3. Expected information value medium: E01 (embedding AUC 0.606 alone, neighbor ratio about 1.0) and G01 predict no gain, so the likely result is a kill, which fits claim 3 not supported. A pass would be the first prompt-space signal on this pair.

## Proposed by the loop, iteration 3 (after L2b)

Proposed 2026-10-18 after L2b (verdict PASS by the rule, avrl/runs/L2b/, RESEARCH_LOG.md). Same format as L1a to L2c. L3a and L3b are analysis-only, CPU only. L3c needs one GPU job through gpu_run.sh and is not launched. Ids are L<iteration><letter>; these come from iteration 3 (the L2 ids came from iteration 2).

## L3a: R-floor sensitivity of the tuned rule on two pools (analysis-only)

**Hypothesis:** the 5% gain of the tuned rule does not depend on the refit floor. Its floor of R = 10 is a deliberate change from the G01 reference floor of 40, and under the reference floor the L1b pool gave +0.89 pp at 5%.

**Change vs existing scripts:** a copy of avrl/exp/l2b_replay.py with the R floor of configs A and B set to 40, so R = max(budget // 8, 40). Everything else is untouched, including the spend gate. This is a sensitivity check of the two pre-fixed configs, not a new selection.

**Command shape:** /homes/rdubash/active-audit/.venv/bin/python -I on avrl/runs/L2b/feats.parquet and avrl/runs/p0/feats.parquet, splits 0 to 7 on each.

**Metric:** held-out gap of config A at 5% minus plain fixed, and of config B at 10% minus plain fixed, pp of hit rate, mean and sd over the eight splits, on each pool.

**Pass threshold:** the mean gap at 5% is above +2.0 pp on the fresh pool at floor 40.

**Kill threshold:** the mean gap at 5% is at or below +2.0 pp on the fresh pool at floor 40. The two rules are complementary.

**Expected runtime:** under 10 min, CPU.

**Informs:** claim 2 at 5%. Expected information value high: the 5% claim is the one that depends on a floor chosen during the L1b sweep. The L1b pool (+0.89 pp at floor 40) predicts a kill there. If the fresh pool also collapses, the 5% gain is a refit-cadence effect and the claim has to say so.

## L3b: Learned arm on held-out splits 4 to 7 of both pools (analysis-only)

**Hypothesis:** the learned arm (greedy_online, the G01 logistic scorer on the cheap features) is below plain fixed on held-out splits at 5% and 10% on both rollout pools, so it stays below the tuned bar that claim 2 has to beat.

**Change vs existing scripts:** a copy of avrl/exp/l2b_replay.py with greedy_online from avrl/exp/g01_label_free_baselines.py added as an evaluated policy. Evaluate on split seeds 4 to 7 of both pools (the L1b held-out split indices), with 25% and 50% added as budgets. The policies are otherwise untouched.

**Command shape:** /homes/rdubash/active-audit/.venv/bin/python -I on avrl/runs/L2b/feats.parquet and avrl/runs/p0/feats.parquet.

**Metric:** held-out gap of greedy_online minus plain fixed, and of greedy_online minus the tuned config at the same budget, pp of hit rate, mean and sd over split seeds 4 to 7, per pool and budget.

**Pass threshold:** greedy_online beats the tuned config at the same budget by more than 2.0 pp at some budget on both pools.

**Kill threshold:** greedy_online is at or below plain fixed at every budget on both pools. Anything between is inconclusive.

**Expected runtime:** about 15 min, CPU.

**Informs:** claim 2 (the learned arm against the tuned bar on the same held-out splits) and the open 25% bar. Expected information value medium-high. On split 0 of the L1b pool the learned arm is 23 pp below plain fixed at 5% (59.98 against 83.14), so a kill is the likely result. That would confirm claim 2 as not supported on a wider sample. A pass would be the first learned-arm result above the tuned bar.

## L3c: Third rollout pool from the GSM8K test split (GPU, one job)

**Hypothesis:** the 5% and 10% gains of the two pre-fixed tuned configs reproduce on a third rollout pool drawn from the GSM8K test split, which no earlier pool used, with no retuning.

**Change vs existing scripts:** avrl/exp/l2b_rollouts_fresh.py with prompts drawn from the GSM8K test split, if it is in avrl/data. Check its row count first. It is about 1,300 problems, fewer than the 3,000 of earlier pools, so use all rows. Keep G = 8, the same generation settings and seed 1. Then the L2b feature script and the L2b replay (avrl/exp/l2b_replay.py) run unchanged, with the two configs fixed from L1b.

**Command shape:** one job through /homes/rdubash/avrl_live/ROUTED/bin/gpu_run.sh, with the vLLM python (.venv-judge) for rollouts and then the features step. The CPU replay runs afterwards with /homes/rdubash/active-audit/.venv/bin/python -I.

**Metric:** held-out gap of config A at 5% and config B at 10% minus plain fixed, pp of hit rate, mean and sd over split seeds 0 to 7, plus the verifier call count per split.

**Pass threshold:** the mean gap is above +2.0 pp at 5% or at 10% (both configs fixed before the run).

**Kill threshold:** the mean gap is at or below +2.0 pp at both budgets. Anything between is inconclusive.

**Expected runtime:** about 10 min GPU (about 10,400 rollouts, roughly 8 min of generation at the L2b rate), then under 5 min CPU.

**Informs:** claim 2 at 5% and 10% on a different prompt split. Expected information value medium. It tests whether the gains depend on the GSM8K train split. The pool is smaller, so 5% budgets are about 50 calls per split and the result is noisier.

## Proposed by the loop, iteration 4 (after L3a)

Proposed 2026-10-18 after L3a (verdict PASS by the rule on the fresh pool, but thin, and below the bar on the L1B pool at refit floor 40; avrl/runs/L3a/, RESEARCH_LOG.md). Same format as the earlier sections. L4a and L4c are analysis-only, CPU only. L4b needs one GPU job through gpu_run.sh and is not launched. Ids are L<iteration><letter>; these come from iteration 4.

## L4a: Refit-count curve at 5% on both pools (analysis-only)

**Hypothesis:** the 5% gain of config A tracks the number of refits per budget, not the refit floor as such. The floor-40 gain differs by pool (+3.16 pp on the fresh pool, +0.89 pp on splits 4 to 7 of the L1B pool) because of how often the rule refits.

**Change vs existing scripts:** a copy of avrl/exp/l3a_floor_sensitivity.py in which the refit rule is R = max(budget // k, 1) for k in 2, 4, 8, 16 and 32, so the rule refits about k times per 5% budget. Config A keeps k0 2 and the agreement x n_distinct stratum. Plain fixed is the baseline. Split seeds 0 to 7, 8 tie-break seeds, paired by seed, spend gate as before. Both pools, all splits. No selection: the curve is the output.

**Command shape:** /homes/rdubash/active-audit/.venv/bin/python -I on avrl/runs/L2b/feats.parquet and avrl/runs/p0/feats.parquet.

**Metric:** config A at 5% minus plain fixed, pp of hit rate, mean and sd over split seeds 0 to 7 for each k, per pool, with the refit count for each k.

**Pass threshold:** on the L1B pool, the mean gap is above +2.0 pp at k = 8 (about the floor-10 refit rate) and at or below +2.0 pp at k = 2 (about the floor-40 rate). This would mark the 5% gain as a refit-rate effect.

**Kill threshold:** on the L1B pool, the mean gap is above +2.0 pp at k = 2. The floor-40 failure on that pool is then not a refit-rate effect. Anything else is inconclusive.

**Expected runtime:** about 10 min, CPU.

**Informs:** claim 2 at 5%, and how the 5% claim is worded. Expected information value high: the L3a result leaves open whether the 5% gain is a cadence effect or specific to one pool.

## L4b: Third rollout pool from the GSM8K test split, both refit floors (GPU, one job)

**Hypothesis:** the 5% gain of config A and the 10% gain of config B hold on a third rollout pool drawn from the GSM8K test split, at refit floor 10 and at refit floor 40, with both configs fixed from L1b and no retuning.

**Change vs existing scripts:** avrl/exp/l2b_rollouts_fresh.py with prompts drawn from avrl/data/gsm8k_test.parquet (all rows, about 1,300, instead of the 3,000 train prompts), G = 8, same generation settings and seed 1. Then avrl/exp/l2b_feats.py runs unchanged. The replay is a copy of avrl/exp/l3a_floor_sensitivity.py with the floor passed as an argument, run at 10 and at 40.

**Command shape:** one job through /homes/rdubash/avrl_live/ROUTED/bin/gpu_run.sh with the vLLM python (.venv-judge) for rollouts and features. The CPU replay runs afterwards with /homes/rdubash/active-audit/.venv/bin/python -I.

**Metric:** config A at 5% and config B at 10% minus plain fixed, pp of hit rate, mean and sd over split seeds 0 to 7, at each floor, plus verifier calls per split.

**Pass threshold:** for the 5% claim, the mean gap of config A at 5% is above +2.0 pp at floor 40. For the 10% claim, the mean gap of config B at 10% is above +2.0 pp at floor 40.

**Kill threshold:** the same means are at or below +2.0 pp at floor 40. Pass and kill are complementary for each claim.

**Expected runtime:** about 10 min GPU, then under 5 min CPU.

**Informs:** the 5% claim across three pools, and the 10% claim at floor 40. Expected information value medium-high. About 1,300 prompts give 5% budgets near 50 calls per split, so the 5% read will be noisy and the 10% read is the firmer one.

## L4c: Learned arm against both tuned bars on held-out splits (analysis-only, replaces the bar in L3b)

**Hypothesis:** the learned arm (greedy_online) stays at or below plain fixed on held-out splits 4 to 7 at 5% and 10% on both pools, and below the tuned rule at each refit floor.

**Change vs existing scripts:** the L3b design unchanged (greedy_online added to a copy of avrl/exp/l2b_replay.py, with 25% and 50% added as budgets), except that the tuned rule is reported as two separate bars, at refit floor 10 and at refit floor 40.

**Command shape:** /homes/rdubash/active-audit/.venv/bin/python -I on avrl/runs/L2b/feats.parquet and avrl/runs/p0/feats.parquet.

**Metric:** greedy_online minus plain fixed, and greedy_online minus the tuned rule at the same budget and floor, pp of hit rate, mean and sd over split seeds 4 to 7, per pool and budget.

**Pass threshold:** greedy_online beats the tuned rule at the same budget and floor by more than 2.0 pp at some budget on both pools.

**Kill threshold:** greedy_online is at or below plain fixed at every budget on both pools. Anything between is inconclusive.

**Expected runtime:** about 15 min, CPU.

**Informs:** claim 2, the learned arm against the tuned bar on the same held-out splits, which is now floor-dependent. Expected information value medium-high. A kill is still the likely result: on split 0 of the L1B pool the learned arm is 23 pp below plain fixed at 5% (59.98 against 83.14).

## Proposed by the loop, iteration 5 (after L4b)

## L5a: Frozen-table control for config B (CPU, existing feats)

**Hypothesis:** the 10% gain of config B comes from a stratum ranking fixed once from labels, not from the online refit. If so, a fixed label-derived rule, not an adaptive one, carries the gain.

**Change vs existing scripts:** copy avrl/exp/l4b_replay.py. Fit config B's stratum rates (agreement x n_distinct x any_truncated, prior k0 50) once on the L1B pool (avrl/runs/p0/feats.parquet). Then rank docs on the L2b and L4b pools by that frozen table, with the same tie-breaks and no refits. Split seeds 0 to 7, 8 tie-break seeds, gap paired by seed, as in L3a.

**Command shape:** /homes/rdubash/active-audit/.venv/bin/python -I on the new copy, once per pool. CPU only.

**Metric:** frozen-table B at 10% minus plain fixed, pp of hit rate, mean over split seeds 0 to 7, per pool. The refitted B at 10% sits beside it (L4b +5.01, L2b +5.02).

**Pass threshold:** the frozen ranking's mean gap is above +2.0 pp on both the L2b and L4b pools. The gain then sits in a fixed ranking, and claim 2's "any fixed rule" comparison must be restated against label-derived fixed rules.

**Kill threshold:** the frozen ranking's mean gap is at or below +2.0 pp on either pool. The online refit then carries the gain.

**Expected runtime:** about 10 min, CPU.

**Informs:** the wording of claim 2 at 10%. Expected information value high: it decides whether the 10% gain is adaptive at all.

## L5b: Truncation ablation for config B (CPU, existing feats)

**Hypothesis:** the 10% gain of config B depends on any_truncated, the generation-side signal (a rollout reached max_tokens), which is the nearest rollout-side proxy for problem difficulty.

**Change vs existing scripts:** copy avrl/exp/l4b_replay.py and drop any_truncated from config B's stratum, leaving agreement x n_distinct with k0 50. Everything else is unchanged. Run at 10% and refit floor 40 on the L1B, L2b and L4b pools.

**Command shape:** /homes/rdubash/active-audit/.venv/bin/python -I, once per pool. CPU only.

**Metric:** ablated B minus plain fixed at 10%, pp of hit rate, mean over split seeds 0 to 7, per pool, beside the full B.

**Pass threshold:** the ablated gap is above +2.0 pp on all three pools, so truncation is not needed for the gain.

**Kill threshold:** the ablated gap is at or below +2.0 pp on at least one pool, so any_truncated carries the gain.

**Expected runtime:** about 10 min, CPU.

**Informs:** claim 3. Expected information value medium-high: it tests whether the one feature in B that tracks difficulty carries the edge.

## L5c: Learned arm against tuned bars on three pools (CPU, extends L4c)

**Hypothesis:** greedy_online stays at or below plain fixed on held-out splits 4 to 7 at 5% and 10% on all three pools, and below the tuned rule at each refit floor.

**Change vs existing scripts:** the L4c design unchanged, with the L4b pool (avrl/runs/L4b/feats.parquet) added as a third pool beside L2b and p0. The tuned rule is still reported as two bars, at refit floor 10 and at refit floor 40.

**Command shape:** /homes/rdubash/active-audit/.venv/bin/python -I on the three feats files. CPU only.

**Metric:** greedy_online minus plain fixed, and greedy_online minus the tuned rule at the same budget and floor, pp of hit rate, mean and sd over split seeds 4 to 7, per pool and budget.

**Pass threshold:** greedy_online beats the tuned rule at the same budget and floor by more than 2.0 pp at some budget on all three pools.

**Kill threshold:** greedy_online is at or below plain fixed at every budget on all three pools. Anything between is inconclusive.

**Expected runtime:** about 20 min, CPU.

**Informs:** claim 2 against the learned arm. Expected information value medium-high. The tuned rule's 5% gap at floor 40 is zero on the L4b pool, so there the learned arm is tested against plain fixed alone. A kill is still the likely result, because the learned arm is 23 pp below plain fixed at 5% on split 0 of the L1B pool.

## Proposed by the loop, iteration 6 (after L5a)

## L6a: Frozen table on the L4b composition split (CPU, existing feats)

**Hypothesis:** The frozen-table gain on L4b (+8.38 pp at 10%, L5a) survives when the eval is restricted to the registered test rows of that pool. The refit B gain collapsed there (+0.58 pp, L4b critic), so the frozen gain may collapse too.

**Change vs existing scripts:** avrl/exp/l5a_frozen_table.py unchanged in logic. The p0 table is fit once, as in L5a. The eval uses the 1,319 L4b test rows (pids 0 to 1318) and the 1,473 remaining train rows separately, plus 64 random 1,319-prompt halves of L4b and of L2b. Buckets: the same stand-in cell100 buckets the L4b critic used.

**Command shape:** /homes/rdubash/active-audit/.venv/bin/python -I on the existing feats files, with the subset selection done in a new script. CPU only.

**Metric:** frozen B minus plain fixed at 10%, pp of hit rate, mean and sd over split seeds 0 to 7, per subset.

**Pass threshold:** mean frozen gap above +2.0 pp on the test rows of L4b.

**Kill threshold:** mean frozen gap at or below +2.0 pp on the test rows of L4b.

**Expected runtime:** about 10 min, CPU.

**Informs:** claim 2 restated against label-derived fixed rules, and whether the L5a L4b PASS is a pool-composition effect. Expected information value high.

**Status: done 2026-10-18, 3,000-label table PASS (+3.39 pp), 200-label tables KILL (+1.22 pp, 8 of 20 subsets above +2.0) on the registered test rows.** Details in avrl/runs/L6a/ and avrl/RESEARCH_LOG.md (L6a section).

## L6b: Label-budget control for the frozen table (CPU, existing feats)

**Hypothesis:** A frozen table fitted on about 200 p0 labels, the same label budget as the refit, still beats plain fixed at 10% on L2b and L4b. If so, the 10% gain is a fixed ranking that needs little labelling.

**Change vs existing scripts:** avrl/exp/l5a_frozen_table.py with the fit step run on 20 random 200-prompt subsets of p0 (k0 50), each frozen and evaluated separately. The eval, gates and tie-break order are unchanged.

**Command shape:** /homes/rdubash/active-audit/.venv/bin/python -I on the existing feats files. CPU only.

**Metric:** frozen B minus plain fixed at 10%, pp of hit rate, mean over the 20 label subsets and split seeds 0 to 7, per eval pool, with sd.

**Pass threshold:** mean gap above +2.0 pp on both L2b and L4b.

**Kill threshold:** mean gap at or below +2.0 pp on either pool. Between is inconclusive.

**Expected runtime:** about 15 min, CPU.

**Informs:** claim 2. It puts the frozen table and the refit on the same label budget, which the L5a comparison cannot do. Expected information value high.

## L6c: Frozen 5% control for config A (CPU, existing feats)

**Hypothesis:** The refit A gain at 5% is zero on L4b (L4b log). A frozen A table (agreement x n_distinct, k0 2) fitted once on p0 gains over plain fixed at 5% on both L2b and L4b. If it does, the 5% failure comes from refit cadence, not from the ranking.

**Change vs existing scripts:** avrl/exp/l5a_frozen_table.py with stratum set agreement x n_distinct, k0 2, budget 5%. The gates and spend gate are unchanged.

**Command shape:** /homes/rdubash/active-audit/.venv/bin/python -I on the existing feats files. CPU only.

**Metric:** frozen A minus plain fixed at 5%, pp of hit rate, mean and sd over split seeds 0 to 7, per pool.

**Pass threshold:** mean gap above +2.0 pp on both L2b and L4b.

**Kill threshold:** mean gap at or below +2.0 pp on either pool.

**Expected runtime:** about 10 min, CPU.

**Informs:** claim 2 at 5%, which the refit A kill leaves open. Expected information value medium-high.

## Proposed by the loop, iteration 7 (after L6b)

## L7a: Learned arm at equal labels (CPU, extends L5c)

**Hypothesis:** greedy_online, fitted on the same 200 p0 prompts as the frozen table, gains more than +1.0 pp over that 200-label frozen table at 10% on both L2b and L4b. Null: it does not.

**Change vs existing scripts:** a copy of avrl/exp/l6b_frozen_label_budget.py in which the frozen table is replaced by greedy_online, fitted on each 200-prompt subset (RandomState(100 + i), i in 0 to 19). Splits, tie seeds, gates and spend gate unchanged. Existing scripts are not edited.

**Command shape:** /homes/rdubash/active-audit/.venv/bin/python -I on the L2b and L4b feats files with --subsets 20 --subset_size 200. CPU only.

**Metric:** greedy_online minus the 200-label frozen table at 10%, pp of hit rate, paired by split seed and tie seed, mean over 20 subsets x split seeds 0 to 7, with sd over subset means.

**Pass threshold:** mean difference above +1.0 pp on both L2b and L4b.

**Kill threshold:** mean difference at or below +1.0 pp on either pool.

**Expected runtime:** about 20 min, CPU.

**Informs:** claim 2 at equal labels, the first test of the learned allocator against a fixed rule on the same label budget. Expected information value high. A kill is the likely result, given the logged G01 kill, so the value lies in confirming it at matched labels.

## L7b: Label-count curve for the frozen table (CPU, existing feats)

**Hypothesis:** The frozen config B gain on L4b rises with label count (+5.12 at 200 labels, +8.38 at 3,000), and the gain at 50 labels stays above +2.0 pp on both pools. Null: at 50 labels the gain is at or below +2.0 pp on either pool.

**Change vs existing scripts:** avrl/exp/l6b_frozen_label_budget.py with --subset_size set to 50, 100, 200, 400, 800, 1600 and 3000 (20 subsets each; 3,000 is the full table). Also record the selected doc set per split and count the distinct selected sets per size.

**Command shape:** /homes/rdubash/active-audit/.venv/bin/python -I on the L2b and L4b feats files, one call per size with --subsets 20 --subset_size N. CPU only.

**Metric:** frozen gap at each size, mean over subsets and split seeds, sd over subset means, share of eval docs in the table, and the count of distinct selected sets.

**Pass threshold:** mean gap above +2.0 pp at 50 labels on both L2b and L4b.

**Kill threshold:** mean gap at or below +2.0 pp at 50 labels on either pool.

**Expected runtime:** about 30 min, CPU.

**Informs:** the label budget behind claim 2, and the non-independence of the 20 subsets. It separates strata coverage (92 to 99% of eval docs in the table at 200 labels) from posterior precision as the source of the L4b label dependence. Expected information value medium-high.

## L7c: Prompt-space cell in the frozen 200-label table (CPU, existing feats)

**Hypothesis:** Adding a coarse prompt-space cell (cell30) to the frozen stratum key lifts the 200-label gain by more than +1.0 pp over the rollout-only table on both L2b and L4b. Null: it does not.

**Change vs existing scripts:** a copy of avrl/exp/l6b_frozen_label_budget.py with the stratum key extended to agreement x n_distinct x any_truncated x cell30, agreement kept in every key as in L2c. The same 20 subsets, splits, tie seeds and gates. cell30 is present in the p0 and L4b feats files; confirm it in the L2b feats before the run.

**Command shape:** /homes/rdubash/active-audit/.venv/bin/python -I on the L2b and L4b feats files with --subsets 20 --subset_size 200. CPU only.

**Metric:** frozen gap with cell30 minus the rollout-only 200-label gap at 10%, paired by split seed and tie seed, mean over 20 subsets x split seeds 0 to 7 per pool, with stratum coverage reported per subset.

**Pass threshold:** mean difference above +1.0 pp on both L2b and L4b.

**Kill threshold:** mean difference at or below +1.0 pp on either pool. A null is weaker than it looks, because at 200 labels many expanded strata are empty and score the prior, so the coverage column must be read with it.

**Expected runtime:** about 15 min, CPU.

**Informs:** claim 3 in the frozen setting at equal labels. The online tuned rule with a prompt-space stratum (L2c) has no logged run. Expected information value high.

## Proposed by the loop, iteration 8 (after L6a)

## L8a: 200-label tables on random 1,319-prompt halves of L4b (CPU, existing feats)

**Hypothesis:** The 200-label KILL on the registered test rows (+1.22 pp) is a property of that cut, not of the recipe. Across random 1,319-prompt halves of the full L4b pool, the 200-label gap averages above +2.0 pp, as the 3,000-label table does (+7.60 pp mean over 64 halves). Null: the mean over halves is at or below +2.0 pp.

**Change vs existing scripts:** a copy of avrl/exp/l6a_test_rows.py in which the 20 registered 200-label subset tables (numpy RandomState(100 + i), fitted once, frozen) are evaluated on the 64 random 1,319-prompt halves of L4b (default_rng(100 + i).permutation, as in L6a context), instead of on the test rows. Gates, splits, tie seeds and spend gate unchanged. Existing scripts are not edited.

**Command shape:** /homes/rdubash/active-audit/.venv/bin/python -I on runs/L4b/feats.parquet and runs/p0/feats.parquet, with --subsets 20 --subset_size 200 --splits 0-7 --tie_seeds 8. CPU only.

**Metric:** frozen minus plain fixed at 10%, pp of hit rate, paired by split seed and tie seed. Mean over halves of the 200-label gap (each the mean over 20 subset tables and split seeds 0 to 7), with sd over halves and the count above +2.0.

**Pass threshold:** mean gap above +2.0 pp over the 64 halves.

**Kill threshold:** mean gap at or below +2.0 pp over the 64 halves.

**Expected runtime:** about 15 min, CPU.

**Informs:** whether the L6a KILL is a recipe property or a cut effect. The 3,000-label test-row gap is about 1.3 sd below the half mean. Expected information value high.

## L8b: Label-count curve on the registered test rows (CPU, existing feats)

**Hypothesis:** On the 1,319 registered test rows, the frozen gain rises with label count and clears +2.0 pp only at or above about 1,600 labels. Null: at 1,600 labels the test-row gap is at or below +2.0 pp, so only the full 3,000-label table clears the bar.

**Change vs existing scripts:** L7b (avrl/exp/l6b_frozen_label_budget.py with --subset_size set to 50, 100, 200, 400, 800, 1600 and 3000, 20 subsets each) with the L4b eval frame cut to pid 0 to 1318 as in L6a (--eval_pid_max 1318). Record the selected doc set per split and count the distinct selected sets per size. Existing scripts are not edited.

**Command shape:** /homes/rdubash/active-audit/.venv/bin/python -I on runs/L4b/feats.parquet and runs/p0/feats.parquet, one call per size with --subsets 20 --subset_size N --eval_pid_max 1318. CPU only.

**Metric:** frozen gap at each size, pp of hit rate, mean over subsets and split seeds 0 to 7, sd over subset means, share of test docs in the table, and the count of distinct selected sets per size.

**Pass threshold:** mean gap above +2.0 pp at 1,600 labels on the test rows.

**Kill threshold:** mean gap at or below +2.0 pp at 1,600 labels on the test rows.

**Expected runtime:** about 30 min, CPU.

**Informs:** the label cost the thesis can claim on held-out rows, replacing the full-pool curve in L7b. Expected information value high.

## L8c: Stratum-structure null on the test rows (CPU, existing feats)

**Hypothesis:** The frozen table's gain depends on which prompts share a stratum, not only on reordering the docs. Null: a table whose stratum labels are permuted across the p0 prompts (same stratum sizes) gains about as much as the true table on the test rows.

**Change vs existing scripts:** a copy of avrl/exp/l6a_test_rows.py. The 36 strata keep their sizes, but their labels are permuted across the 3,000 p0 prompts (seeded) before the fit. The permuted table is fitted once, frozen and evaluated on the test rows as in L6a. Gates and spend gate unchanged. Existing scripts are not edited.

**Command shape:** /homes/rdubash/active-audit/.venv/bin/python -I on runs/L4b/feats.parquet and runs/p0/feats.parquet, with --eval_pid_max 1318 and --splits 0-7 --tie_seeds 8. CPU only.

**Metric:** true table minus permuted table at 10%, pp of hit rate, paired by split seed and tie seed, mean over split seeds 0 to 7, with sd over splits. Repeat over 20 permutation seeds for a spread.

**Pass threshold:** mean difference above +2.0 pp (mean over 20 permutation seeds).

**Kill threshold:** mean difference at or below +2.0 pp.

**Expected runtime:** about 20 min, CPU.

**Informs:** claim 3. It tests whether the table's edge depends on which prompts share a stratum, using a size-matched null that no logged run has. Expected information value medium.

## Proposed by the loop, iteration 9 (after L7a)

## L9a: Table-initialised learned arm at equal labels (CPU, existing feats)

**Hypothesis:** A learned online allocator that starts from the frozen 200-label table's ranking and refits with eval labels gains more than +1.0 pp over that table at 10% on both L2b and L4b. Null: it does not.

**Change vs existing scripts:** a copy of avrl/exp/l7a_learned_equal_labels.py. The initial scorer is the table's posterior rate for each doc's stratum (the prior for strata not in the table), with no logistic fit. Every refit uses the 200 p0 labels plus the verified eval labels so far, with the table posterior added as a feature beside the six cheap features. Same 20 subsets, splits, tie seeds and gates. Existing scripts are not edited.

**Command shape:** /homes/rdubash/active-audit/.venv/bin/python -I on runs/p0/feats.parquet with --eval_feats runs/L2b/feats.parquet and then runs/L4b/feats.parquet, with --subsets 20 --subset_size 200 --splits 0-7 --tie_seeds 8. CPU only.

**Metric:** greedy minus the 200-label frozen table at 10%, pp of hit rate, paired by split seed and tie seed, mean over 20 subsets x split seeds 0 to 7 per pool, with sd over subset means.

**Pass threshold:** mean difference above +1.0 pp on both L2b and L4b.

**Kill threshold:** mean difference at or below +1.0 pp on either pool.

**Expected runtime:** about 20 min, CPU.

**Informs:** claim 2 in its strongest equal-label form, where the learned arm starts from the best label-fitted fixed rule. The L7a diagnostic puts most of the loss in the static ranking, so a kill is likely. A pass would be the first support for the learned arm at equal labels. Expected information value high.

## L9b: Learned arm at 3,000 p0 labels against the 3,000-label table (CPU, existing feats)

**Hypothesis:** With the logistic scorer fitted on all 3,000 p0 labels, the greedy learned arm gains more than +1.0 pp over the 3,000-label frozen table at 10% on both L2b and L4b. Null: it does not.

**Change vs existing scripts:** a copy of avrl/exp/l7a_learned_equal_labels.py with the fit set fixed at all 3,000 p0 rows (one fit, no subset draw) and the comparator set to the L5a 3,000-label table. Same split seeds, tie seeds and gates. Existing scripts are not edited.

**Command shape:** /homes/rdubash/active-audit/.venv/bin/python -I on runs/p0/feats.parquet with --eval_feats runs/L2b/feats.parquet runs/L4b/feats.parquet, --budget 0.10 --splits 0-7 --tie_seeds 8. CPU only.

**Metric:** greedy minus the 3,000-label frozen table at 10%, pp of hit rate, paired by split seed and tie seed, mean over split seeds 0 to 7 per pool, with sd over splits.

**Pass threshold:** mean difference above +1.0 pp on both L2b and L4b.

**Kill threshold:** mean difference at or below +1.0 pp on either pool.

**Registered 2026-10-19, before the run (rule fixed here, read the result only after this):** script avrl/exp/l9b_learned_3000_labels.py, launcher avrl/exp/L9b.sh, independent check avrl/exp/l9b_check.py (recomputes every split gap from the saved hit counts). Gates: plain fixed at 10% reproduces runs/L2b (config B) and runs/L4b (config B40) per split to 0.01 pp with the same calls; table minus plain fixed reproduces runs/L5a per split to 1e-6 pp; spend gates for fixed, table and greedy with zero failures; zero eval question text in p0. Deviation from L7a, stated in advance: the L6b per-subset gate does not apply because L9b has no subsets, so it is not run. The sd is over the 8 split gaps, which share one p0 fit, so it covers eval-side variation only. A gate failure is recorded as GATE_FAILED, not as a verdict.

**Result (2026-10-19): KILL on both pools.** Greedy minus the 3,000-label table: -2.38 pp on L2b (sd 0.67 over splits, 0 of 8 above +1.0) and -1.59 pp on L4b (sd 0.87, 0 of 8). All 16 split gaps negative. Gates PASS, independent check PASS. Read as the registered ceiling reading (see the L9b result entry in avrl/RESEARCH_LOG.md). Claim 2 unchanged, not_supported.

**Expected runtime:** about 15 min, CPU.

**Informs:** claim 2 at the label count the 3,000-label table uses (the L5a comparator). It separates scorer under-fitting from small samples (a pass here with a kill at 200 labels would point that way) from a ceiling in the six-feature logistic family (a kill at both sizes). Expected information value medium-high.

## L9c: Learned arm on the registered test rows (CPU, existing feats)

**Hypothesis:** On the L4b test rows (pid <= 1318), the greedy learned arm at 200 p0 labels gains more than +1.0 pp over the 200-label frozen table at 10%. Null: it does not.

**Change vs existing scripts:** a copy of avrl/exp/l7a_learned_equal_labels.py with the --eval_pid_max 1318 filter from avrl/exp/l6a_test_rows.py, applied to the L4b eval frame before the cell100 split. Same subsets, splits, tie seeds and gates. Existing scripts are not edited.

**Command shape:** /homes/rdubash/active-audit/.venv/bin/python -I on runs/p0/feats.parquet with --eval_feats runs/L4b/feats.parquet --eval_pid_max 1318 --subsets 20 --subset_size 200 --splits 0-7 --tie_seeds 8. CPU only.

**Metric:** greedy minus the 200-label frozen table at 10% on the test rows, pp of hit rate, mean over 20 subsets x split seeds 0 to 7, with sd over subset means.

**Pass threshold:** mean difference above +1.0 pp.

**Kill threshold:** mean difference at or below +1.0 pp.

**Expected runtime:** about 10 min, CPU.

**Informs:** claim 2 on the test rows that L6a registered and L7a did not cover. The L6a frozen table is only +1.22 pp above plain fixed there, so a kill is likely, and it would mean the learned arm sits below the table on the registered rows. Expected information value medium.

## Proposed by the loop, iteration 10 (after L9a)

## L10a: Table-pooled stratum posterior at equal labels (CPU, existing feats)

**Hypothesis:** A stratum-posterior learner that starts from the frozen 200-label table as prior counts and adds the verified eval labels as counts beats that same table by more than +1.0 pp at 10% on both L2b and L4b. Null: it does not.

**Change vs existing scripts:** a new script in avrl/exp/, copied from the stratum-posterior refit (refit B, floor 40) and extended so that the 200 p0 labels enter as prior counts at k0 = 50 per stratum. No logistic model. Same 20 subsets, split seeds, tie seeds and gates. Existing scripts are not edited.

**Command shape:** /homes/rdubash/active-audit/.venv/bin/python -I on runs/p0/feats.parquet with --eval_feats runs/L2b/feats.parquet, then runs/L4b/feats.parquet, with --budget 0.10 --subsets 20 --subset_size 200 --splits 0-7 --tie_seeds 8. CPU only.

**Metric:** pooled-posterior minus the 200-label frozen table at 10%, pp of hit rate, mean over 20 subsets x split seeds 0 to 7 per pool, with sd over subset means.

**Pass threshold:** mean difference above +1.0 pp on both L2b and L4b.

**Kill threshold:** mean difference at or below +1.0 pp on either pool.

**Expected runtime:** about 10 min, CPU.

**Informs:** claim 2 at equal labels, for the one learner that ties the table in the existing runs (refit B). L9a shows the logistic refit loses, so this is the remaining equal-label test of a learned allocator that fits no scorer. Expected information value high.

## L10b: Matched-precision control for L9a (CPU, existing feats)

**Hypothesis:** The L9a kill comes from the refit design, not from the posterior mismatch between training and scoring rows. With matched precision, the table-start refit beats the 200-label frozen table by more than +1.0 pp at 10% on both pools. Null: it does not.

**Change vs existing scripts:** a copy of avrl/exp/l9a_table_init_learned.py in which the 200 p0 rows keep their 5-fold out-of-fold posterior, and each eval row gets the table posterior from one randomly chosen fold table (160 labels) instead of the 200-label table. Everything else is as in L9a, including the frozen comparator. Existing scripts are not edited.

**Command shape:** as L9a: /homes/rdubash/active-audit/.venv/bin/python -I on runs/p0/feats.parquet with --eval_feats runs/L2b/feats.parquet and runs/L4b/feats.parquet, --budget 0.10 --subsets 20 --subset_size 200 --splits 0-7 --tie_seeds 8. CPU only.

**Metric:** table-start refit minus the 200-label frozen table at 10%, pp of hit rate, mean over 20 subsets x split seeds 0 to 7 per pool, with sd over subset means.

**Pass threshold:** mean difference above +1.0 pp on both pools.

**Kill threshold:** mean difference at or below +1.0 pp on either pool.

**Expected runtime:** about 5 min, CPU.

**Informs:** claim 2 at equal labels. The L9a kill has one confound it cannot rule out: the refit trains on 160-label posteriors and scores with 200-label posteriors. This run decides whether that mismatch drives the loss. Expected information value medium-high.

## L10c: Parsimonious refit for the table-start arm (CPU, existing feats)

**Hypothesis:** A refit on the table posterior plus agreement and n_distinct only (no mean_tokens, std_tokens, any_truncated or log_qlen) beats the 200-label frozen table by more than +1.0 pp at 10% on both pools. Null: it does not.

**Change vs existing scripts:** a copy of avrl/exp/l9a_table_init_learned.py with the refit feature list cut to agreement, n_distinct and the table posterior. Same refit schedule (R = 40), subsets, splits, tie seeds and gates. Existing scripts are not edited.

**Command shape:** as L9a: /homes/rdubash/active-audit/.venv/bin/python -I on runs/p0/feats.parquet with --eval_feats runs/L2b/feats.parquet and runs/L4b/feats.parquet, --budget 0.10 --subsets 20 --subset_size 200 --splits 0-7 --tie_seeds 8. CPU only.

**Metric:** table-start refit minus the 200-label frozen table at 10%, pp of hit rate, mean over 20 subsets x split seeds 0 to 7 per pool, with sd over subset means.

**Pass threshold:** mean difference above +1.0 pp on both pools.

**Kill threshold:** mean difference at or below +1.0 pp on either pool.

**Expected runtime:** about 5 min, CPU.

**Informs:** claim 2 at equal labels. With a few dozen proxy-error positives per 200-label subset, seven free coefficients may overfit, and this run tests that. Expected information value medium.

## Proposed by the loop, iteration 11 (after L10a)

## L11a: table_pooled on the registered L4b test rows (CPU, existing feats)

**Hypothesis:** The table-pooled stratum posterior (L10a) beats the 200-label frozen table by more than +1.0 pp at 10% on the 1,319 registered L4b test rows (pid <= 1318). Null: it does not. The full-pool L4b margin is 0.3 pp, and the L9a critic's stand-in replay gave the table-start logistic +1.44 on these rows, so this cut may favour table-start learners. It needs its own registration.

**Change vs existing scripts:** a copy of avrl/exp/l10a_table_pooled_posterior.py with the --eval_pid_max 1318 filter from avrl/exp/l6a_test_rows.py, applied to the L4b eval frame before the cell100 split, as in L9c. Same 20 subsets, split seeds, tie seeds, k0 = 50 and R = 40 (about 2 refits per split at this budget). Existing scripts are not edited.

**Command shape:** /homes/rdubash/active-audit/.venv/bin/python -I on runs/p0/feats.parquet with --eval_feats runs/L4b/feats.parquet --eval_pid_max 1318 --budget 0.10 --subsets 20 --subset_size 200 --splits 0-7 --tie_seeds 8. CPU only.

**Metric:** table_pooled minus the 200-label frozen table at 10% on the test rows, pp of hit rate, mean over 20 subsets (each over split seeds 0 to 7, 8 tie-break seeds each), with sd over subset means.

**Pass threshold:** mean difference above +1.0 pp.

**Kill threshold:** mean difference at or below +1.0 pp.

**Expected runtime:** about 5 min, CPU.

**Informs:** claim 2 on the registered rows, where no equal-label learner has been tested. Expected information value high.

## L11b: k0 and refit-cadence sensitivity for table_pooled (CPU, existing feats)

**Hypothesis:** The L2b result (+0.33 pp, L10a) depends on the table prior strength or the refit cadence, so some setting lifts the table-pooled gap above +1.0 pp on both pools. Null: no setting does.

**Change vs existing scripts:** a copy of avrl/exp/l10a_table_pooled_posterior.py with the refit cadence R exposed as a flag (default max(budget // 8, 40)). Grid: k0 in {10, 200} at R = 40, and R in {20, 80} at k0 = 50. The k0 = 50, R = 40 reference is the L10a result. Existing scripts are not edited.

**Command shape:** /homes/rdubash/active-audit/.venv/bin/python -I on runs/p0/feats.parquet with --eval_feats runs/L2b/feats.parquet and runs/L4b/feats.parquet, --budget 0.10 --subsets 20 --subset_size 200 --splits 0-7 --tie_seeds 8, one call per setting and pool. CPU only.

**Metric:** table_pooled minus the 200-label frozen table at 10%, pp of hit rate, mean over 20 subsets per pool and per setting, with sd over subset means.

**Pass threshold:** a setting above +1.0 pp on both pools is named as a candidate for a fresh-pool test, not as a claim pass.

**Kill threshold:** no setting is above +1.0 pp on both pools.

**Expected runtime:** about 15 min for 10 calls, CPU.

**Informs:** claim 2 at equal labels, and whether the L10a result depends on prior strength or cadence. Expected information value medium.

## L11c: fresh eval-cut replicate of L10a (CPU, existing feats)

**Hypothesis:** The L10a means (+0.33 pp on L2b, +1.30 pp on L4b) survive eval-side resampling: over 48 fresh bucket-shuffle cuts of each full pool, the mean gap is above +1.0 pp on both pools. Null: it is at or below +1.0 pp on at least one pool.

**Change vs existing scripts:** the L10a arms unchanged, run on 48 fresh bucket-shuffle cuts of each full pool (seeds 100 to 147, the cell100 column as the bucket stand-in, as in the L9a critic's replay), with tie seeds 0 to 2 and the mean over the same 20 p0 tables. A new cut driver is needed; the L10a script is not edited.

**Command shape:** /homes/rdubash/active-audit/.venv/bin/python -I on runs/p0/feats.parquet with --eval_feats runs/L2b/feats.parquet and runs/L4b/feats.parquet, through the new cut driver, 48 cuts per pool. CPU only.

**Metric:** table_pooled minus the 200-label frozen table at 10%, mean over the 20 tables for each cut, then mean and sd over the 48 cuts, per pool.

**Pass threshold:** mean over cuts above +1.0 pp on both pools.

**Kill threshold:** mean over cuts at or below +1.0 pp on either pool.

**Expected runtime:** about 20 min, CPU.

**Informs:** claim 2 at equal labels, since the 20 label draws do not vary the eval side. The L4b margin is 0.8 standard errors, so this decides whether the L4b pass is stable. Expected information value medium-high.

## Proposed by the loop, iteration 12 (after L11a)

## L12a: k0 and R sensitivity on the remaining-train rows (CPU, existing feats)

**Hypothesis:** The +0.34 pp table_pooled gap on the registered test rows (L11a) is set by prior strength or refit cadence, so some setting lifts the gap above +1.0 pp on the remaining-train rows. Null: no setting does.

**Change vs existing scripts:** a copy of avrl/exp/l11a_table_pooled_test_rows.py with the eval cut to pid > 1318 (1,473 rows, the development set) and the refit cadence R exposed as a flag. Grid: k0 in {10, 200} at R = 40, and R in {20, 80} at k0 = 50. Reference: k0 = 50, R = 40. The registered test rows are not used for the sweep. Existing scripts are not edited.

**Command shape:** /homes/rdubash/active-audit/.venv/bin/python -I on runs/p0/feats.parquet with --eval_feats runs/L4b/feats.parquet cut to pid > 1318, --budget 0.10 --subsets 20 --subset_size 200 --splits 0-7 --tie_seeds 8, one call per setting. CPU only.

**Metric:** table_pooled minus the 200-label frozen table at 10%, in pp of hit rate, mean over 20 subsets per setting, with sd over subset means.

**Pass threshold:** a development setting above +1.0 pp earns one pre-registered run on the registered test rows. This is not a claim pass.

**Kill threshold:** no development setting is above +1.0 pp.

**Expected runtime:** about 10 min, CPU.

**Informs:** claim 2 at equal labels, namely whether the +0.34 pp result is a prior-strength effect or a ceiling of the posterior learner. Expected information value medium.

## L12b: why the 200-label table's gain differs by row set (CPU, existing feats)

**Hypothesis:** The frozen 200-label table's gain over plain fixed differs by row set (+1.22 pp on the registered test rows, about +8.6 pp on the remaining-train rows, a stand-in from the L10a critic) because the stratum ordering does not transfer across rows. Null: the ordering transfers, and the difference comes from row mix or budget instead.

**Change vs existing scripts:** a new analysis script reading runs/p0/feats.parquet and runs/L4b/feats.parquet, split into pid <= 1318 and pid > 1318. For each of the 36 strata of the 3,000-prompt table, compute the error rate and occupancy on each row set, and the rank order of the 200-label table's stratum scores on each. No learner arm is run.

**Command shape:** /homes/rdubash/active-audit/.venv/bin/python -I on the new script. CPU only.

**Metric:** Spearman rank correlation of the 36 stratum error rates between the two row sets, plus the frozen-table gain on each row set as a check.

**Pass threshold:** Spearman correlation above 0.8.

**Kill threshold:** Spearman correlation below 0.5.

**Expected runtime:** under 5 min, CPU.

**Informs:** claim 3 (structure in prompt space). If the stratum order does not transfer, the table's gain is not a stable property of the structure, and claim 3 needs a different test. Expected information value high.

## L12c: fresh held-out rows on a second task (GPU rollouts, CPU analysis)

**Hypothesis:** On held-out rows from a second task, table_pooled beats the frozen 200-label table by more than +1.0 pp at 10% budget, so the L11a result is not specific to the GSM8K test rows. Null: it does not.

**Change vs existing scripts:** new rollouts on MATH problems with the same model (Qwen2.5-1.5B-Instruct) and G = 8, run through /homes/rdubash/avrl_live/ROUTED/bin/gpu_run.sh after a gpu_status check. A new answer extractor is needed, because p0_rollouts.py extracts only the GSM8K #### answer. The source needs a fit pool and a held-out eval pool. HuggingFaceH4/MATH-500 has only 500 problems, so the fit pool would need the MATH train split (not yet checked). Fix the split and the registration before any rollout runs.

**Command shape:** gpu_run.sh with the rollout command, then CPU analysis with the L11a script on the new feats files.

**Metric:** table_pooled minus the 200-label frozen table at 10%, in pp of hit rate, mean over 20 subsets on the held-out rows, with sd over subset means. Frozen minus plain fixed is reported as a control.

**Pass threshold:** mean above +1.0 pp.

**Kill threshold:** mean at or below +1.0 pp.

**Expected runtime:** about 5 to 10 min of GPU, extrapolated from the p0 throughput (MATH completions are longer), plus extractor and registration work.

**Informs:** claims 2 and 3 across tasks. The GSM8K registered rows have been used in several verdicts, so a fresh task is the only uncontaminated test here. Expected information value high.

## Proposed by the loop, iteration 13 (after L10b)

## L13a: shrunk table-start refit (CPU, existing feats)

**Hypothesis:** A refit that starts at the matched 160-label table ranking and moves only on evidence beats the 200-label frozen table by more than +1.0 pp at 10% on both L2b and L4b. Null: it does not.

**Change vs existing scripts:** a copy of avrl/exp/l10b_matched_precision.py. The refit scorer becomes a ridge-penalised logistic on the six cheap features with the table logit as a fixed offset, fitted with scipy.optimize (scipy is already a dependency of the venv's sklearn; no new package). One ridge strength, fixed before the run, no sweep. Matched start, refit cadence R = 40, subsets, seeds and gates are unchanged. Full pools, as in L9a and L10b, so the registered test rows are used again; report the pid > 1318 cut beside the result. Existing scripts are not edited.

**Command shape:** /homes/rdubash/active-audit/.venv/bin/python -I on runs/p0/feats.parquet with --eval_feats runs/L2b/feats.parquet, then with runs/L4b/feats.parquet, --budget 0.10 --k0 50 --subsets 20 --subset_size 200 --splits 0-7 --tie_seeds 8, then the combine step. CPU only.

**Metric:** table_refit minus the 200-label frozen table at 10%, in pp of hit rate, mean over 20 subsets per pool, with sd over subset means. table_zero minus frozen and the L10b values are reported beside it.

**Pass threshold:** mean above +1.0 pp on both pools.

**Kill threshold:** mean at or below +1.0 pp on either pool.

**Expected runtime:** about 10 min, CPU.

**Informs:** claim 2 at equal labels. No run has yet tested the shrinkage axis. Expected information value high: a kill would close the logistic family for the equal-label claim.

## L13b: next-pick diagnosis of the refit loss (CPU, existing feats, no new arm)

**Hypothesis:** The logistic refit's loss of about 1.1 pp against its own zero-refit start comes from ranking error on the docs it picks next, not from the trajectory it creates. Null: at each refit, the refit ranks the next picks as well as the table key does from the same seen set.

**Change vs existing scripts:** an analysis-only script that replays the L10b greedy loop (same subsets, split seeds, tie seeds, matched start, R = 40). At each refit event it scores the next 10 picks two ways from the same seen set: under the refit score, and under the table key (-score, agreement, -n_distinct, seeded draw). No new arm is run and no existing script is edited. The 60% diagnostic threshold below is set here, not in the L10b spec.

**Command shape:** /homes/rdubash/active-audit/.venv/bin/python -I on the new script, once per pool. CPU only.

**Metric:** per refit event, the hit rate of the next 10 picks under the refit score minus under the table key, in pp. Report the fraction of refit events where the refit is below the table key, per pool.

**Pass threshold:** the refit is below the table key in more than 60% of refit events on both pools. The loss is then local ranking error at the refit.

**Kill threshold:** at or below 60% on either pool. The refit is locally no worse, so the loss comes from the trajectory and shrinkage is the wrong fix.

**Expected runtime:** about 10 min, CPU.

**Informs:** claim 2. It picks the fix: if the refit is locally worse, L13a's shrinkage is the right next test; if not, the design needs a trajectory fix. Expected information value medium-high.

## L13c: matched table start on the remaining-train rows (CPU, existing feats)

**Hypothesis:** On the remaining-train rows (pid > 1318, 1,473 rows, the L12a dev set), the matched-precision table-start refit beats the 200-label frozen table by more than +1.0 pp at 10% budget. Null: it does not.

**Change vs existing scripts:** a copy of avrl/exp/l10b_matched_precision.py with the eval cut to pid > 1318 before the split, as the L11a copy did for pid <= 1318. The unmatched L9a stand-in on the same rows (-0.96 pp, from the L9a critic) is reported beside it. Existing scripts are not edited.

**Command shape:** /homes/rdubash/active-audit/.venv/bin/python -I on runs/p0/feats.parquet with --eval_feats runs/L4b/feats.parquet cut to pid > 1318, --budget 0.10 --k0 50 --subsets 20 --subset_size 200 --splits 0-7 --tie_seeds 8. CPU only.

**Metric:** table_refit minus the 200-label frozen table at 10%, in pp of hit rate, mean over 20 subsets, with sd over subset means and the count of subsets above +1.0.

**Pass threshold:** mean above +1.0 pp. This earns one pre-registered run on the registered test rows (pid <= 1318), not a claim pass.

**Kill threshold:** mean at or below +1.0 pp.

**Expected runtime:** about 5 min, CPU.

**Informs:** claim 2 on the row set where the table-start logistic reads +1.44 pp (unmatched stand-in on the test rows) against -0.96 pp on the remaining-train rows. If the matched version kills here too, the +1.44 reading does not justify a registered run, and the row-set gap belongs to L12b. Expected information value medium.

## Proposed by the loop, iteration 14 (after L13a)

## L14a: Eval-cut replicate of the shrunk arm on the registered test rows (CPU, existing feats)

**Hypothesis:** The offset-ridge refit's +1.43 pp lead over the 200-label frozen table on the registered test rows (pid <= 1318) survives fresh eval cuts. Null: it does not.

**Change vs existing scripts:** a copy of avrl/exp/l13a_shrunk_refit.py with the eval frame cut to pid <= 1318, run over 48 fresh eval cuts using the bucket-shuffle seeds 100 to 147 that the L10b critic used. Existing scripts are not edited.

**Command shape:** /homes/rdubash/active-audit/.venv/bin/python -I on runs/p0/feats.parquet with --eval_feats runs/L4b/feats.parquet cut to pid <= 1318, --budget 0.10 --k0 50 --subsets 20 --subset_size 200 --splits 0-7 --tie_seeds 8, once per cut seed. CPU only.

**Metric:** table_shrunk minus the 200-label frozen table at 10%, in pp of hit rate, mean over the 48 cuts (each cut is the mean over 20 subsets x split seeds 0 to 7), with the min, the max and the count of cuts above +1.0.

**Pass threshold:** mean above +1.0 pp over the 48 cuts. A pass earns a fresh-data test, not a claim pass, because the test rows have already been read by several registered cuts.

**Kill threshold:** mean at or below +1.0 pp.

**Expected runtime:** about 10 min, CPU.

**Informs:** claim 2 on the one row set where the offset-ridge refit reads positive (+1.43 pp, sd 2.16, 11 of 20 subsets above +1.0). It decides whether that lead is worth carrying at all. Expected information value medium-high: cheap, and the only positive number in L13a.

## L14b: Offset-ridge strength diagnostic at C = 0.1 and C = 0.01 (CPU, existing feats)

**Hypothesis:** A stronger offset-ridge penalty than L13a's C = 1 lifts the shrunk refit above +1.0 pp over the 200-label frozen table on both full pools, at one or both of the two strengths. Null: it does not at either strength.

**Change vs existing scripts:** two copies of avrl/exp/l13a_shrunk_refit.py, one with C_REFIT set to 0.1 and one with C_REFIT set to 0.01. Both strengths are fixed here, before any run. Existing scripts are not edited.

**Command shape:** /homes/rdubash/active-audit/.venv/bin/python -I on each copy, once per pool (L2b, then L4b), with the L13a arguments from avrl/exp/L13a.sh. CPU only.

**Metric:** table_shrunk minus the 200-label frozen table at 10%, in pp of hit rate, mean over 20 subsets per pool and strength, with sd over subset means. table_shrunk minus table_zero is reported beside it, so the curve from C = 1 to the zero-refit start is visible.

**Pass threshold:** at least one strength above +1.0 pp on both full pools. A pass earns a registered test on the registered test rows, not a claim pass.

**Kill threshold:** neither strength clears +1.0 pp on both pools. A kill is the expected result: with beta at zero the ranking is the matched table (table_zero, +0.36 on L2b and -0.08 on L4b), so a strength has to add gain over its own zero start to pass.

**Expected runtime:** about 15 min, CPU (four runs).

**Informs:** claim 2 at equal labels. L13a closed only C = 1. This either closes the offset-ridge family for the equal-label claim or shows that the penalty route is not the one to pursue. Expected information value low to medium.

## Proposed by the loop, iteration 15 (after L12b)

## L15a: Posterior-ordering transfer on the 200-label fits (CPU, existing feats)

**Hypothesis:** The frozen 200-label table's posterior ordering (not the raw per-stratum error rates that L12b used) transfers across the two L4b row sets. Null: it does not.

**Change vs existing scripts:** a new script, not an edit of avrl/exp/l5a_frozen_table.py or avrl/exp/l12b_stratum_transfer.py. For each of the 20 registered 200-prompt subsets (RandomState(100 + i), as in L13a), fit the frozen posterior on the 200 labels with k0 = 50 and prior 1 - agreement, as fit_table does in avrl/exp/l5a_frozen_table.py. Strata missing from the 200 labels score the prior, as attach_score does. Then compute Spearman rho between that posterior and the per-stratum proxy_error rate on the registered test rows (pid <= 1318), and separately on the remaining rows (pid > 1318), over strata with at least one doc on each side.

**Command shape:** /homes/rdubash/active-audit/.venv/bin/python -I on runs/p0/feats.parquet (fit) and runs/L4b/feats.parquet (eval), once per subset. CPU only.

**Metric:** mean over the 20 subsets of rho(posterior, test error rate) and of rho(posterior, remaining error rate), with the sd over subsets.

**Pass threshold:** both means above 0.8.

**Kill threshold:** either mean below 0.5. Values between 0.5 and 0.8 are inconclusive.

**Expected runtime:** about 5 min, CPU.

**Informs:** claim 2 indirectly, through the stability of the comparator. Expected information value high: L12b measured raw rates on the full row sets, which is not the object the hypothesis names.

## L15b: Occupancy-matched gain on the remaining rows (CPU, existing feats)

**Hypothesis:** The +8.58 pp frozen gain on the remaining rows (pid > 1318) is mostly row mix. Reweighting the remaining rows to the registered test-row stratum occupancy brings it near the +1.22 pp test-row gain. Null: it stays well above that.

**Change vs existing scripts:** a copy of avrl/exp/l13a_shrunk_refit.py with only the eval-row sampling changed. The remaining-row eval pool is resampled with probability proportional to the per-stratum weight (test occupancy divided by remaining occupancy). Strata with zero remaining docs are dropped from the weighting and listed. Existing scripts are not edited.

**Command shape:** /homes/rdubash/active-audit/.venv/bin/python -I, with the L13a arguments from avrl/exp/L13a.sh, once per weighted cut (seeds 100 to 147). CPU only.

**Metric:** frozen minus fixed at 10% on the weighted remaining rows, mean over the 48 cuts, with min and max.

**Pass threshold (row mix explains the gap):** mean at or below +3.2 pp, that is within 2 pp of the +1.22 pp test-row gain.

**Kill threshold (row mix does not explain it):** mean at or above +5.0 pp. Values between are inconclusive.

**Expected runtime:** about 10 min, CPU.

**Informs:** claim 2 through the comparator's gain on the remaining rows. Expected information value medium-high: it is the direct row-mix test that L12b could not run.

## L15c: Budget curve on both row sets (CPU, existing feats)

**Hypothesis:** The frozen table's gain over plain fixed converges across the two row sets at some budget, so the +1.22 versus +8.58 pp gap depends on budget. Null: the gap stays at 5 pp or more at both 5% and 20%.

**Change vs existing scripts:** the L13a eval loop, copied with only the frozen 200-label and plain fixed arms kept, run with --budget 0.05 and --budget 0.20 on each row set (pid <= 1318 and pid > 1318). Check first whether L8b (label-count curve on the registered test rows) already covers the test-row side. If it does, run only the remaining-row side.

**Command shape:** /homes/rdubash/active-audit/.venv/bin/python -I, 2 budgets by 2 row sets, 20 subsets each. CPU only.

**Metric:** frozen minus fixed at each budget on each row set, mean over the 20 subsets, and the difference between the two row sets at each budget.

**Pass threshold:** at one budget, the two row sets' gains differ by less than 2 pp.

**Kill threshold:** the two row sets differ by 5 pp or more at both budgets.

**Expected runtime:** about 10 min, CPU.

**Informs:** the budget half of the L12b question. Expected information value low to medium: it answers what L12b could not, but no registered claim hinges on a budget curve, so this is a second-order check.

## Proposed by the loop, iteration 16 (after L15b)

## L16a: Source-split null for the registered test-row gain (CPU, existing feats)

**Hypothesis:** The +1.22 pp frozen gain on the 1,319 registered test rows (GSM8K test questions) is within the spread of the same comparison on random 1,319-prompt cuts of GSM8K train questions, so the test split is not special. Null: the test-row gain lies below that spread, so the test split itself lowers the gain.

**Change vs existing scripts:** a new script, not an edit of avrl/exp/l15b_occupancy_weighted.py or avrl/exp/l13a_shrunk_refit.py. Keep the 200-label frozen table and plain fixed arms, with the L13a loop. Eval pool: runs/L2b/feats.parquet (3,000 GSM8K train prompts, disjoint from p0 and from the test questions), drawn without weights as 48 random cuts of 1,319 prompts (cut seeds 100 to 147). Also a prompt bootstrap of the 1,319 registered test rows (200 draws with replacement, seed 16), to give the test-row gain its own sampling interval. Existing scripts are not edited.

**Command shape:** /homes/rdubash/active-audit/.venv/bin/python -I with the L13a arguments from avrl/exp/L13a.sh. CPU only, in the shell, in batches of 8 cuts to stay under the shell limit.

**Metric:** frozen minus plain fixed at 10%, pp of hit rate: mean over each train cut (mean over 20 subsets x split seeds 0 to 7), reported as the 5th and 95th percentiles of the cut means and the min and max. Compare with the test-row value of +1.22 pp and its prompt-bootstrap 5th and 95th percentiles.

**Pass threshold:** +1.22 pp lies between the 5th and 95th percentiles of the train-cut means (the test split is not special at this sample size).

**Kill threshold:** +1.22 pp lies below the 5th percentile of the train-cut means (the test split lowers the gain).

**Expected runtime:** about 10 min, CPU.

**Informs:** the comparator for claim 2 on the registered rows. Expected information value high: it is the only check that separates the test split from sampling noise on existing data. Limit: the test split is one population, so a kill shows the gap is unlikely to be noise but cannot name its cause.

## L16b: Prompt-side scalar inside strata (CPU, existing feats)

**Hypothesis:** The within-stratum gap between registered test rows and remaining rows (25.9% against 16.5% proxy error, over 26 strata with at least 5 docs per side, from L15b) is carried by a prompt-side scalar, log_qlen. Null: it persists within log_qlen terciles.

**Change vs existing scripts:** a new script that reads runs/L4b/feats.parquet only. Stratum key = agreement x n_distinct x any_truncated x log_qlen tercile, with tercile cut points taken from the pooled L4b rows. No fit, no table, no new rollouts.

**Command shape:** /homes/rdubash/active-audit/.venv/bin/python -I on runs/L4b/feats.parquet. CPU only.

**Metric:** test-minus-remaining proxy error gap in pp, averaged over cells with at least 5 docs on each side, weighted by the number of test docs in each cell. Reference: 9.4 pp with no tercile split. Report the number of cells used.

**Pass threshold:** at or below 3 pp (a prompt-side scalar explains the gap).

**Kill threshold:** at or above 6 pp (the gap is not carried by log_qlen).

**Expected runtime:** about 5 min, CPU.

**Informs:** claim 3 (structure in prompt space). Expected information value medium: median question length is nearly the same on both row sets (222 and 217), so a pass is unlikely. A kill would support the source-split reading in L16a and would be the first direct prompt-side test of the gap.

Not re-proposed: L15a (posterior-ordering transfer, iteration 15) remains open and is the next claim-2 comparator check. L15c (budget curve) remains open.

## Proposed by the loop, iteration 17 (after L16a)

## L17a: Run-matched split null for the registered test rows (CPU, existing feats)

**Hypothesis:** Inside one rollout run, the registered test rows (pid <= 1318) and the remaining L4b rows (pid >= 1319) have the same frozen minus fixed gain. Null: the registered test rows gain less than the remaining rows, so the split is special within the run. L16a could not separate this, because its comparator came from the separate L2b rollout run.

**Change vs existing scripts:** a new script that reads runs/L4b/feats.parquet only, not an edit of avrl/exp/l16a_source_split_null.py or avrl/exp/l13a_shrunk_refit.py. Keep the 200-label frozen table (p0 fit, k0 = 50) and the plain fixed arm, with the L13a loop. Draw 200 subsets of 1,000 rows without replacement from each group (seeds 17 for the test rows, 18 for the remaining rows), and compute D = gain(test) minus gain(remaining) per draw. No new rollouts.

**Command shape:** /homes/rdubash/active-audit/.venv/bin/python -I on runs/L4b/feats.parquet, launched through a runbook in batches to stay under the shell limit. CPU only.

**Metric:** D in pp, with the 5th and 95th percentiles and the sign of each. Also report each group's own gain distribution. Reference: the test rows at +1.22 pp (L13a) and the remaining rows at +8.58 pp (L13a).

**Pass threshold:** 0 lies between the 5th and 95th percentiles of D (the split is not special within the run).

**Kill threshold:** the 95th percentile of D is below 0 (the registered test rows gain less than the remaining rows inside one run).

**Expected runtime:** about 5 min, CPU.

**Informs:** claim 2 (the registered comparator) and the reading of L16a. Expected information value high: this is the only check that holds the rollout run fixed, so it decides whether the registered-row shortfall is a split effect at all.

## L17b: Headroom-normalized comparator for the registered rows (CPU, existing numbers)

**Hypothesis:** The registered test-row gain, divided by headroom (100 minus the fixed hit rate), lies within the spread of the same ratio on the 48 L2b cuts from L16a. Null: it lies below that spread. The registered rows sit at a fixed hit rate of 84.5%, above every L2b cut (64.5% to 74.3%), so raw pp may compress their gain mechanically.

**Change vs existing scripts:** no new compute. Read runs/L16a/cuts and runs/L16a/result.json and compute the ratio per cut and for the registered rows. This ratio was first looked at after L16a, so this entry must be read as a new registration and the thresholds apply only to a run made under it.

**Command shape:** /homes/rdubash/active-audit/.venv/bin/python -I on the existing L16a result files. CPU only.

**Metric:** gain divided by (100 minus the fixed hit rate), for the registered rows and each of the 48 L2b cuts. Reference values from the L16a read: test rows 0.079; cut p5 0.058; cut p95 0.288.

**Pass threshold:** the registered-row ratio lies between the 5th and 95th percentiles of the cut ratios (the registered shortfall is mostly headroom).

**Kill threshold:** the registered-row ratio lies below the 5th percentile of the cut ratios (the shortfall persists after headroom).

**Expected runtime:** under 1 min, CPU.

**Informs:** claim 2 (how the registered comparator should be reported). Expected information value medium: a pass means claim 2 should be reported on the headroom scale, and a kill means the registered rows differ from the L2b cuts even after headroom.

## L17c: Second and third fixed rules against the frozen table (CPU, existing feats)

**Hypothesis:** The 200-label frozen table beats each of several other fixed rankings at a 10% budget. Claim 2 says "any fixed rule", but only one fixed rule (agreement ascending, then n_distinct descending) has been tested. Null: some other fixed rule matches or beats the frozen table.

**Change vs existing scripts:** a new script, not an edit of avrl/exp/l16a_source_split_null.py or avrl/exp/l13a_shrunk_refit.py. Two added fixed rules: n_distinct descending alone (ties by a seeded random key), and mean_tokens descending alone (same tie key). Frozen is the L13a table as in L16a. Score on the L2b full pool and the registered test rows, 20 subsets x split seeds 0 to 7 x tie seeds 0 to 7, budget int(0.10 x n_sim). No new rollouts.

**Command shape:** /homes/rdubash/active-audit/.venv/bin/python -I on runs/L2b/feats.parquet and runs/L4b/feats.parquet, through a runbook. CPU only.

**Metric:** frozen minus each fixed rule, pp of hit rate, on the L2b full pool and the registered rows.

**Pass threshold:** frozen minus each added rule is above +1.0 pp on the L2b full pool.

**Kill threshold:** the gap for any added rule is at or below 0 on the L2b full pool.

**Expected runtime:** about 10 min, CPU.

**Informs:** claim 2 ("any fixed rule"). Expected information value high for the scope of claim 2: it turns a one-rule comparison into a multi-rule comparison, and the claim is only as strong as the set of fixed rules tried.

## Proposed by the loop, iteration 18 (after L9b)

## L18a: Stratum-indicator logistic scorer at 3,000 labels (CPU, existing feats)

**Hypothesis:** The greedy arm's gap to the 3,000-label frozen table (L9b: -2.38 pp on L2b, -1.59 pp on L4b) comes from the six-feature logistic scorer lacking the agreement x n_distinct x any_truncated interactions. Adding the 36 one-hot stratum indicators to the scorer closes the gap on both pools. Null: it does not.

**Change vs existing scripts:** a copy of avrl/exp/l9b_learned_3000_labels.py, not an edit. The scorer becomes StandardScaler plus LogisticRegression(class_weight balanced, max_iter 1000) on the six cheap features plus 36 one-hot stratum indicators. Same single fit on all 3,000 p0 rows, same refits with verified eval labels, same splits, tie seeds, budget and gates.

**Command shape:** /homes/rdubash/active-audit/.venv/bin/python -I on runs/p0/feats.parquet with --eval_feats runs/L2b/feats.parquet runs/L4b/feats.parquet, --budget 0.10 --k0 50 --splits 0-7 --tie_seeds 8. CPU only, about 15 min.

**Metric:** per pool, greedy_strata minus the L5a 3,000-label frozen table at 10%, in pp of hit rate, mean over split seeds 0 to 7 (8 tie seeds each), with sd over splits.

**Pass threshold:** pool mean above +1.0 pp on both L2b and L4b.

**Kill threshold:** pool mean at or below +1.0 pp on either pool.

**Expected runtime:** about 15 min, CPU.

**Informs:** claim 2 and the L9b ceiling reading. A pass would register an interaction-aware learned family as a new arm, not a rescue of the six-feature arm. Expected information value high.

## L18b: One-shot logistic ranking with no online refit (CPU, existing feats)

**Hypothesis:** The L9b gap comes from the online refit loop, not from the scorer's ranking. Null: the scorer's one-shot ranking already trails the 3,000-label table, so the gap is in the scorer.

**Change vs existing scripts:** a copy of avrl/exp/l9b_learned_3000_labels.py, not an edit. The logistic scorer is fitted once on the 3,000 p0 rows, ranks each eval pool once, and the top int(0.10 x n_sim) docs are picked with the same per-seed tie key. No refits and no eval labels. Same splits, tie seeds and gates.

**Command shape:** as L18a. CPU only, about 10 min.

**Metric:** per pool, one-shot logistic minus the 3,000-label frozen table at 10%, in pp of hit rate, mean over split seeds 0 to 7, with sd over splits.

**Pass threshold:** pool mean above +1.0 pp on both pools (the refit loop is the cost).

**Kill threshold:** pool mean at or below +1.0 pp on either pool (the scorer's ranking is the cost).

**Expected runtime:** about 10 min, CPU.

**Informs:** claim 2, and the refit-loss reading from L13b. It picks the next step: L18a if the scorer is the cost, a refit fix if the loop is. Expected information value high.

## L18c: L9c at 3,000 labels on the registered test rows (CPU, existing feats)

**Hypothesis:** On the L4b registered test rows (pid <= 1318), the greedy logistic arm with the 3,000-label fit beats the 3,000-label frozen table by more than +1.0 pp at 10%. Null: it does not.

**Change vs existing scripts:** a copy of avrl/exp/l9b_learned_3000_labels.py with the L4b eval frame cut to pid <= 1318 before the cell100 split, as in L9c and L6a. Same fit, gates and seeds. Existing scripts are not edited.

**Command shape:** /homes/rdubash/active-audit/.venv/bin/python -I with --eval_feats runs/L4b/feats.parquet and the pid filter, --budget 0.10 --k0 50 --splits 0-7 --tie_seeds 8. CPU only, about 10 min.

**Metric:** greedy minus the 3,000-label frozen table at 10% on the test rows, in pp of hit rate, mean over split seeds 0 to 7, with sd over splits.

**Pass threshold:** mean above +1.0 pp.

**Kill threshold:** mean at or below +1.0 pp.

**Expected runtime:** about 10 min, CPU.

**Informs:** claim 2 on the row set where the frozen table's gain over plain fixed is smallest (+1.22 pp at 200 labels). Expected information value medium-low: every full-L4b split gap is already negative in L9b, so a kill is likely and mostly confirms L9b on the registered rows. L9c at 200 labels stays open.

## Proposed by the loop, iteration 19 (after L18a)

## L19a: Cell-posterior greedy with online updates (CPU, existing feats)

**Hypothesis:** A greedy arm that ranks by the L5a shrunken cell posterior (the table's own ranking), recomputed after each refit block from p0 rows plus verified eval labels, closes the gap to the frozen 3,000-label table on both pools. Null: it does not.

**Change vs existing scripts:** a copy of avrl/exp/l18a_strata_logistic_3000.py, not an edit. The scorer is the L5a cell posterior (k0 50, prior 1 minus agreement) instead of the logistic. Same splits, tie seeds, budget, refit schedule (R = max(budget // 8, 40)) and gates.

**Command shape:** /homes/rdubash/active-audit/.venv/bin/python -I on runs/p0/feats.parquet with --eval_feats runs/L2b/feats.parquet runs/L4b/feats.parquet, --budget 0.10 --k0 50 --splits 0-7 --tie_seeds 8. CPU only, about 10 min.

**Metric:** per pool, cell-posterior greedy minus the L5a 3,000-label frozen table at 10%, in pp of hit rate, mean over split seeds 0 to 7 (8 tie seeds each), with sd over splits.

**Pass threshold:** pool mean above +1.0 pp on both L2b and L4b.

**Kill threshold:** pool mean at or below +1.0 pp on either pool.

**Expected runtime:** about 10 min, CPU.

**Informs:** claim 2. It tests the reading L18a leaves open: whether a scorer that ranks by the cell posterior is outside the logistic family's ceiling. Expected information value medium-high.

## L19b: Stratum-cell decomposition of the one-shot gap (CPU, existing feats, no new labels)

**Hypothesis:** The one-shot gap between the logistic scorer and the frozen table (no refits) sits mainly in the sparse p0 cells (fewer than 10 p0 prompts), so the table's edge over the logistic is a shrinkage effect and fragile on new pools. Null: the gap sits in well-populated cells, so the edge is structural.

**Change vs existing scripts:** a new analysis script that reuses the L18a scorer fit and the L5a table fit on the same splits. No new arm, no edits to existing scripts.

**Command shape:** /homes/rdubash/active-audit/.venv/bin/python -I on runs/p0/feats.parquet with --eval_feats runs/L2b/feats.parquet runs/L4b/feats.parquet, --splits 0-7, one-shot picks only. CPU only, about 10 min.

**Metric:** per pool, the share of the summed hit difference (frozen table hits minus one-shot logistic hits, per stratum cell, over the picks at 10%) that falls in cells with fewer than 10 p0 prompts. Count those cells from the p0 fit set.

**Pass threshold:** share under 30 percent on both pools (the table's edge is structural).

**Kill threshold:** share at least 70 percent on either pool (the edge is a sparse-cell effect). Between the two: inconclusive.

**Expected runtime:** about 10 min, CPU.

**Informs:** claim 2, and how the frozen table's edge is reported as the deliverable. Expected information value high.

## L19c: GSM8K rollout replicate on the registered test questions (GPU, one gpu_run.sh job)

**Hypothesis:** The registered test-row gain (table minus plain fixed, +1.22 pp, against +8.58 pp for the remaining rows) comes from the question source, not from the rollout run. A fresh rollout on the same 1,319 test questions reproduces the gain. Null: it does not reproduce, so the registered rows are a question-source effect.

**Change vs existing scripts:** a fresh rollout with avrl/p0_rollouts.py (copied and modified, not edited) on the 1,319 GSM8K registered test questions, with the same model and prompt as the p0 rollouts. Then the L16a test-row comparison on the new rollouts.

**Command shape:** one job through /homes/rdubash/avrl_live/ROUTED/bin/gpu_run.sh <run_dir> '<command>', with the 45-minute cap, checked with gpu_status.sh before launch. About 8 min of generation by the p0 throughput estimate. Not yet registered for execution.

**Metric:** table minus plain fixed on the registered test rows at 10%, in pp of hit rate, compared with the 48 L16a train cuts (5th and 95th percentiles).

**Pass threshold:** the test-row gain above +1.0 pp and inside the 5th to 95th percentile of the train cuts.

**Kill threshold:** the gain at or below +1.0 pp, or below the 5th percentile of the train cuts.

**Expected runtime:** about 8 min of generation plus analysis.

**Informs:** claim 2 on the registered rows, and the L16a confound (rollout run against question source). Expected information value high.

## Proposed by the loop, iteration 20 (after L19c)

## L20a: Stratum-mix and headroom decomposition of the row-set contrast (CPU, existing feats)

**Hypothesis:** The 200-label contrast between the remaining rows and the registered test rows (+4.93 pp on runs/L19c, +7.36 pp on runs/L4b) comes mostly from the 36-stratum mix of the two row sets, not from question-level properties. Null: the contrast survives reweighting the remaining rows to the test rows' stratum mix.

**Change vs existing scripts:** a new script, avrl/exp/l20a_contrast_decomp.py, copied from avrl/exp/l19c_test_replicate.py (not an edit). It reuses the same 20 frozen 200-label tables, splits, tie seeds and 10% budget, and adds per-stratum hit counts and the reweighted remaining-row gain.

**Command shape:** /homes/rdubash/active-audit/.venv/bin/python -I on avrl/exp/l20a_contrast_decomp.py with --fit_feats runs/p0/feats.parquet, --eval_feats runs/L19c/feats.parquet runs/L4b/feats.parquet, --budget 0.10 --k0 50 --subsets 20 --subset_size 200 --splits 0-7 --tie_seeds 8. CPU only.

**Metric:** C = 200-label gain on pids 1319 and up minus the 200-label gain on pids 0 to 1318, in pp, mean over the 20 subsets (each the mean over split seeds 0 to 7). C_adj is the same quantity after reweighting the remaining rows so their 36-stratum shares equal the test rows' shares. Reported for both runs.

**Pass threshold:** C_adj at or below +1.0 pp on runs/L19c (stratum mix explains the contrast).

**Kill threshold:** C_adj above +3.0 pp on runs/L19c (the contrast survives the reweighting). Between the two: inconclusive.

**Expected runtime:** about 10 min, CPU.

**Informs:** claim 2 on the registered rows, and the L16a confound. If the contrast is mix or headroom, the test-row number is a sampling fact. If it survives, the row set has a question-level effect that a held-out claim has to carry. Expected information value high.

## L20b: Fresh fit-pool rollouts for the 3,000-label table (GPU, one gpu_run.sh job)

**Hypothesis:** The 3,000-label registered-row gain is stable across fit draws as well as eval draws. Its eval rollouts have been redrawn (+3.39 pp on runs/L6a, +3.46 pp on runs/L19c), but the p0 fit pool has one rollout draw. Null: a fresh p0 rollout draw drops the registered-row 3,000-label gain to at or below +2.0 pp.

**Change vs existing scripts:** a new rollout script, avrl/exp/l20b_p0_rollouts_seed2.py, copied from avrl/p0_rollouts.py with the sampling seed set to 2 and the 3,000 train prompts unchanged (the original is not edited). Then the unchanged feature script on the new run dir. Refit the 3,000-label table and the 20 subset tables on the new p0 features, and score on the L19c test rows. Check that the new feature columns match runs/p0/feats.parquet before scoring.

**Command shape:** /homes/rdubash/avrl_live/ROUTED/bin/gpu_run.sh /homes/rdubash/avrl_live/ROUTED/avrl/runs/L20b '<p0 rollout copy, seed 2, 3,000 prompts, G 8> && <feature script on runs/L20b>', after gpu_status.sh shows under 2 GB used. Then CPU: avrl/exp/l19c_test_replicate.py under the main venv with -I, --fit_feats runs/L20b/feats.parquet, --eval_feats runs/L19c/feats.parquet, --eval_pid_max 1318.

**Metric:** The 3,000-label table minus plain fixed at 10% on pids 0 to 1318 of runs/L19c/feats.parquet, mean over split seeds 0 to 7 (8 tie seeds each), with the fit on the new p0 draw. Context: the 200-label verdict value on the same new fit.

**Pass threshold:** 3,000-label registered-row gain above +2.0 pp (the L6a bar).

**Kill threshold:** at or below +2.0 pp.

**Expected runtime:** about 20 min GPU (3,000 prompts at the L19c rate of 968 s for 2,792 prompts), plus about 5 min CPU.

**Informs:** the deliverable's registered-row number, which has so far been tested against eval-side redraws only. Expected information value high.

**Status (2026-10-19):** run, with a different design from the one above (the L19c critic found seed 2 shares 6 of 8 sample streams with p0). Final design: avrl/exp/l20b_fit_rollouts.py (--seed 100) and avrl/exp/l20b_test_rollouts.py (--seed 200), both redrawn, then the 3,000-label verdict on runs/L20b/test. Result: +5.35 pp, PASS. See the L20b entries in avrl/RESEARCH_LOG.md.

## L20c: Second and third fixed rules on the fresh registered rows (CPU, existing feats)

**Hypothesis:** The 200-label frozen table beats each of two further fixed rankings on the registered rows of the fresh rollouts. Claim 2 says "any fixed rule", but only plain fixed (agreement ascending, then n_distinct descending) has been tested. Null: one of the two rules matches or beats the frozen table on those rows.

**Change vs existing scripts:** a new script, avrl/exp/l20c_fixed_rules.py, copied from avrl/exp/l19c_test_replicate.py (not an edit). It adds two fixed rules: n_distinct descending alone and mean_tokens descending alone, both with the same seeded tie key as plain fixed. This is the registered-row half of L17c (iteration 17, not run), moved to the fresh rollouts in runs/L19c. The frozen table and plain fixed are unchanged.

**Command shape:** /homes/rdubash/active-audit/.venv/bin/python -I on avrl/exp/l20c_fixed_rules.py with --fit_feats runs/p0/feats.parquet, --eval_feats runs/L19c/feats.parquet, --eval_pid_max 1318, --budget 0.10 --k0 50 --subsets 20 --subset_size 200 --splits 0-7 --tie_seeds 8. CPU only.

**Metric:** 200-label frozen minus each added rule at 10%, in pp of hit rate, on pids 0 to 1318, as the mean over 20 subsets (each the mean of split seeds 0 to 7, 8 tie seeds each). Context: frozen minus plain fixed (+2.17 pp).

**Pass threshold:** frozen minus each added rule above +1.0 pp.

**Kill threshold:** at or below 0 for either added rule. Between the two: inconclusive.

**Expected runtime:** about 10 min, CPU.

**Informs:** claim 2 scope. A rule that matches or beats the table would reduce claim 2 to a statement about plain fixed. Expected information value high.

## Proposed by the loop, iteration 21 (after L20b)

## L20d: Decomposition of the L20b gain into fit-draw and eval-draw effects (CPU, existing feats)

**Hypothesis:** The L20b registered-row gain (+5.35 pp, against +3.46 pp on L19c and +3.39 pp on L6a) moved because of the fresh fit draw, the fresh eval draw, or both. Null: the fresh fit draw alone, scored on the L19c test rows, keeps the registered-row gain above +2.0 pp.

**Change vs existing scripts:** a new CPU script, avrl/exp/l20d_decompose.py, copied from avrl/exp/l20b_replicate.py (not an edit). It runs the 3,000-label frozen table, k0 50, budget 0.10, split seeds 0 to 7, 8 tie seeds, in two cells: (a) fit on runs/L20b/fit/feats.parquet, eval on runs/L19c/feats.parquet pids 0 to 1318; (b) fit on runs/p0/feats.parquet, eval on runs/L20b/test/feats.parquet. The L20b cell (fresh fit, fresh eval) and the L6a and L19c cells are the reference.

**Command shape:** /homes/rdubash/active-audit/.venv/bin/python -I avrl/exp/l20d_decompose.py with the two cells above. CPU only, no GPU job.

**Metric:** the 3,000-label gap on the test rows, mean over split seeds 0 to 7, for cell (a) and cell (b).

**Pass threshold:** cell (a) above +2.0 pp (the fresh fit draw alone keeps the registered gain).

**Kill threshold:** cell (a) at or below +2.0 pp (the gain depends on the eval draw).

**Expected runtime:** about 2 min CPU.

**Informs:** which side of the L20b redraw moved the registered-row gain, and therefore which figure the deliverable should report. Expected information value high.

## L20a and L20c

Unchanged from iteration 20 above. Neither was run.

## Proposed by the loop, iteration 22 (after L20b verification)

## L22a: Question-level bootstrap of the L20b registered-row gain (CPU, existing feats)

**Hypothesis:** The +5.35 pp registered-row 3,000-label gain on runs/L20b/test survives resampling of the 1,319 registered questions. Null: the 2.5th percentile of the question-level bootstrap gain is at or below +2.0 pp.

**Change vs existing scripts:** a new script, avrl/exp/l22a_question_bootstrap.py, copied from the row-level bootstrap in avrl/exp/l16a_source_split_null.py (boot mode, RandomState draws over the 1,319 registered rows; not an edit). The 3,000-label table is fitted once on runs/L20b/fit/feats.parquet and frozen, as in avrl/exp/l20b_replicate.py.

**Command shape:** /homes/rdubash/active-audit/.venv/bin/python -I avrl/exp/l22a_question_bootstrap.py with --fit_feats runs/L20b/fit/feats.parquet --eval_feats runs/L20b/test/feats.parquet --eval_pid_max 1318 --boot 500 --budget 0.10 --k0 50 --splits 0-7 --tie_seeds 8. CPU only.

**Metric:** 3,000-label table minus plain fixed at 10% on the resampled registered questions, in pp, reported as the 2.5th, 50th and 97.5th percentiles.

**Pass threshold:** 2.5th percentile above +2.0 pp.

**Kill threshold:** 2.5th percentile at or below +2.0 pp.

**Expected runtime:** about 15 min, CPU.

**Informs:** claim 2's registered number. The 8-split sd (1.30) covers split variation only, so this is the row-sampling term the number lacks. Expected information value high.

## L22b: Eval-only redraw on the L20b fit pool (GPU, one gpu_run.sh job)

**Hypothesis:** The +5.35 pp registered-row gain holds on a third eval draw with the L20b fit pool and table held fixed. Null: a third eval draw at sampling seed 300 gives a gain at or below +2.0 pp.

**Change vs existing scripts:** a copy of avrl/exp/l20b_test_rollouts.py named avrl/exp/l22b_test_rollouts.py, run with --seed 300 (streams 300 to 307, outside seeds 0 to 9 and 100 to 207), output runs/L22b/test. The fit pool is not redrawn: the table is refitted only from runs/L20b/fit/feats.parquet, unchanged. Then avrl/exp/l20b_feats.py and avrl/exp/l20b_replicate.py run on the new rows, as in L20b.

**Command shape:** /homes/rdubash/avrl_live/ROUTED/bin/gpu_run.sh /homes/rdubash/avrl_live/ROUTED/avrl/runs/L22b '/homes/rdubash/active-audit/.venv-judge/bin/python /homes/rdubash/avrl_live/ROUTED/avrl/exp/l22b_test_rollouts.py --G 8 --seed 300 --out /homes/rdubash/avrl_live/ROUTED/avrl/runs/L22b/test', after gpu_status.sh shows under 2 GB used. Then the CPU feature and replicate steps as in L20b.

**Metric:** the L20b-fitted 3,000-label table minus plain fixed at 10% on the registered rows (pids 0 to 1318 of the new draw), in pp, mean over split seeds 0 to 7 with 8 tie seeds each.

**Pass threshold:** above +2.0 pp.

**Kill threshold:** at or below +2.0 pp.

**Expected runtime:** about 10 min GPU (one vLLM load and 1,319 prompts, L20b test rate) plus about 2 min CPU.

**Informs:** the eval-draw part of the L6a to L20b move. With L20d cell (b) it fills a 2x2 over fit draw and eval draw on fresh rollouts. Expected information value high.

## L22c: 3,000-label table against config B on the fresh registered rows (CPU, existing feats, gated)

**Hypothesis:** The 3,000-label table beats config B (the tuned, label-fed adaptive stratum rule at floor 40) at 10% on the registered rows of the fresh rollouts by more than +1.0 pp. Null: the margin is at or below 0.

**Change vs existing scripts:** a new script, avrl/exp/l22c_table_vs_configb.py, copied from avrl/exp/l20b_replicate.py (not an edit), with config B added exactly as run in avrl/exp/l3a_floor_sensitivity.py at floor 40. Same 3,000-label table, splits, tie seeds and 10% budget. Gate before scoring: config B must reproduce its recorded floor-40 value on an earlier pool (RESEARCH_LOG lines 277 and 337 give +5.02 pp on the fresh pool and +4.64 pp on the L1B pool). If it does not reproduce, the run is recorded as failed, not patched.

**Command shape:** /homes/rdubash/active-audit/.venv/bin/python -I avrl/exp/l22c_table_vs_configb.py --fit_feats runs/L20b/fit/feats.parquet --eval_feats runs/L20b/test/feats.parquet --eval_pid_max 1318 --budget 0.10 --k0 50 --splits 0-7 --tie_seeds 8. CPU only.

**Metric:** 3,000-label table minus config B at 10% on pids 0 to 1318, in pp, mean over split seeds 0 to 7 with 8 tie seeds each. Context: the same difference on runs/L19c/feats.parquet.

**Pass threshold:** above +1.0 pp.

**Kill threshold:** at or below 0. Between 0 and +1.0: inconclusive.

**Expected runtime:** about 10 min, CPU.

**Informs:** whether the 3,000-label table, the deliverable, adds anything over the adaptive rule the ledger has already tested. Claim 2 names a learned allocator against fixed rules, so this bears on how the deliverable is framed rather than on the claim as worded. Expected information value high.
