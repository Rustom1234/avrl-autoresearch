# AVRL Experiment Backlog

Pre-registered, 2026-09-30. Twenty experiments, E01 through E20, ordered by expected information value per GPU hour: cheapest, highest-leverage checks first, most expensive and most infrastructure-blocked last. This ordering means analysis-only reruns of already-collected data (near-zero marginal GPU cost) come before any new generation run, and new generation runs on this machine come before any GRPO training run on an A100.

## Claim numbering (from PROJECT.md)

- Claim 1: the learned allocator reaches the same policy quality with a fraction of expensive verifier calls.
- Claim 2: the learned allocator admits fewer corrupted rewards into training than any fixed rule at the same budget.
- Claim 3: the allocator's edge comes from structure in prompt space (proxy error clusters), the SOFisher condition (Li et al., 2026, Nature Communications, https://doi.org/10.1038/s41467-026-73404-6).

Source for the existing harness read this session: /homes/rdubash/avrl_live/CASCADE/avrl/PROJECT.md, /homes/rdubash/avrl_live/CASCADE/avrl/p0_rollouts.py, /homes/rdubash/avrl_live/CASCADE/avrl/p0_analyze.py, /homes/rdubash/avrl_live/CASCADE/avrl/allocators.py. Confirmed at read time: avrl/p1_grpo.py (the phase-1 GRPO training harness named in PROJECT.md's Files section) does not exist on disk yet, only p0_rollouts.py, p0_analyze.py, and allocators.py do. Every experiment below that needs an actual policy-training loop is marked NEEDS PHASE-1 HARNESS.

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

All paper citations above are carried forward unchanged from /homes/rdubash/avrl_live/CASCADE/lit/avrl/SYNTHESIS.md, read this session, which itself states its method: built via WebFetch against arXiv abstracts, the OpenAlex works API, and direct publisher DOIs, with WebSearch unavailable (budget error) and not retried. This document adds one new fact verified this session via WebFetch: HuggingFaceH4/MATH-500's existence, size, and content, https://huggingface.co/datasets/HuggingFaceH4/MATH-500. No paper, author, or number is invented in this document; every runtime figure not directly read from avrl/runs/p0/run.log is explicitly marked as an estimate or UNVERIFIED. G01-G04 were added after this session's three gap checks (GAP_already-done.md, GAP_cannot-work.md, GAP_not-a-thesis.md); their citations (VIP, https://arxiv.org/abs/2602.01601) are carried forward unchanged from SYNTHESIS.md and the gap-check files.

## Compute correction (2026-09-30, main session)
This machine's GPU is an NVIDIA RTX A4000 with 16 GB, not a 32 GB card. Measured with the phase-1 harness (Qwen2.5-1.5B, LoRA r=32, bf16): full settings (16 prompts x G=8, 512 tokens) run at 153 s per step with HF generation and peak 13.2 GB; vLLM colocate does not fit beside training on this card (out of memory at engine start). Smoke settings run at 20 s per step. vLLM-only rollout experiments (phase-0 style) are unaffected and fit in 40 minutes at 1,500 to 3,000 prompts.

## Feasibility note (this run)
Items marked A100, needing a larger cluster, a second machine, or the phase-1 GRPO harness (E15 to E20, G04, anything needing p1_grpo.py; about 89 s per step on this card, so 100 steps take 2.5 hours) are not runnable under the 45-minute cap and must not be planned.

---

## Proposed by the loop

Proposed from E03 (2026-10-18, kill for claim 2 against the label-free agreement threshold). All three are CPU only and need neither the GPU nor the phase-1 harness. Ids use loop iteration 3 (the iteration that produced E03).

---

## L3a: Robustness of the E03 kill (cluster bootstrap and cell30)

**Hypothesis:** the E03 kill on cell100 holds when the bootstrap respects bucket clustering and when the bucket granularity changes.

**Change vs existing scripts:** reuse avrl/exp/e03_threshold_bootstrap.py. Add a bucket-cluster bootstrap option (resample cell buckets with replacement, keep their groups). Run the same config with --cell cell30, adding the threshold arm (E01 already has cell30 cheap+emb numbers for the other arms).

**Metric:** D(ts_online) at 25% with a 95% CI under both bootstrap schemes, on both bucket granularities.

**Pass threshold:** the kill holds. The cluster-bootstrap CI stays below 0 and D stays negative on cell30.

**Kill threshold:** the cluster-bootstrap CI includes 0, or cell30 gives D(ts_online) > 0 with its CI excluding 0. Either reopens claim 2 on this pair.

**Expected runtime:** under 5 min, CPU.

**Informs:** the statistical support for claim 2, and whether the E03 kill can be cited without caveat.

---

## L3b: Warm-started bucket prior for the learned arm

**Hypothesis:** the bucket-level Thompson layer loses budget because its early bucket priors are flat. Initializing bucket priors from the label-free threshold ordering, then keeping the online refit, closes the gap to the threshold at 25%.

**Change vs existing scripts:** add a ts_warm arm to the E03 script. Its alpha and beta start from per-bucket mean agreement (label-free) instead of Beta(1,1), with the same refit cadence as ts_online. No change to the other arms.

**Metric:** admitted rate at 25% for ts_warm, and D(ts_warm) = admitted(threshold) minus admitted(ts_warm) with a 95% bootstrap CI as in E03.

**Pass threshold:** D > 0 with the 95% CI excluding 0 at 25%.

**Kill threshold:** the CI includes 0, or D <= 0. Then no bucket-based learned allocator tested here beats the label-free threshold, and claim 2 needs a different learner.

**Expected runtime:** under 10 min, CPU.

**Informs:** claim 2 directly. This is the only remaining design on this pair that could pass it.

---

## L3c: Out-of-sample ceiling for a labeled ranker

**Hypothesis:** a labeled ranker on the 22 E01 features cannot beat the label-free threshold at 25% by more than a few points of admitted rate, even out of sample.

**Change vs existing scripts:** five-fold cross-fit by cell100 bucket on the E03 sim pool, using the same logistic model and features as the static arm, ranking at 25%, compared with the threshold and static arms.

**Metric:** caught docs and admitted rate at 25% for the cross-fit ranker, and its difference from the threshold.

**Pass threshold (informative):** the cross-fit ranker catches at least 263 (admitted <= 0.149, the static reference). A labeled ranker then has room above the threshold, and L3b is worth running.

**Kill threshold:** the cross-fit ranker catches at most 252.5 (ties or loses to the threshold). No labeled ranker on these features beats the label-free rule, and phase 1 is not worth funding on this pair.

**Expected runtime:** under 2 min, CPU.

**Informs:** claims 1 and 2, and the phase-1 funding decision. It bounds the best case before any GPU run.

---

Proposed from L3c (2026-10-18, pass, marginal: caught 264 against the 263.0 line, one fold assignment). Ids use loop iteration 4 (the iteration that produced L3c). All three are CPU only.

---

## L4a: Fold-seed stability of the L3c cross-fit

**Hypothesis:** the L3c pass at 25% is not an artifact of the fold assignment. Averaged over fold seeds, the labeled cross-fit ranker still matches the labeled static reference (263.0).

**Change vs existing scripts:** reuse avrl/exp/l3c_crossfit_ranker.py with the fold seed as a loop variable (0 to 19). Keep the fit split at seed 0 and the pool unchanged. Report caught at 10, 25 and 50% per fold seed.

**Metric:** mean caught at 25% over the 20 fold seeds, with the range, and the admitted rate it implies (1 - caught/309).

**Pass threshold:** mean caught at 25% >= 263.0.

**Kill threshold:** mean caught at 25% <= 252.5 (the label-free threshold). Results strictly between are inconclusive.

**Expected runtime:** under 2 min, CPU.

**Informs:** whether the L3c PASS survives fold resampling. It gates L3b and the phase-1 funding decision. This is the check the L3c run report asked for.

---

## L4b: Cheap-only attribution for the labeled ranker (claim 3)

**Hypothesis:** the labeled cross-fit gain over the threshold comes from the cheap signals (agreement, n_distinct, token statistics), not from the embedding structure. If so, prompt-space structure adds nothing to a labeled ranker on this pair.

**Change vs existing scripts:** reuse avrl/exp/l3c_crossfit_ranker.py with COLS set to the 6 cheap features only. Same pool, same five folds per fold seed, same model. Run for fold seeds 0 to 19 and compare with the 22-feature result for the same seeds.

**Metric:** caught at 25% for 22 features minus caught at 25% for 6 features, mean over the 20 fold seeds.

**Pass threshold (claim 3 structure matters):** 22-feature minus 6-feature mean gap >= 5 caught at 25%.

**Kill threshold (claim 3 not supported here):** gap <= 2 caught. The labeled gain then comes from cheap signals, and the embedding structure adds nothing. Gaps between 2 and 5 are inconclusive.

**Expected runtime:** under 2 min, CPU.

**Informs:** claim 3 (edge from prompt-space structure). It is the only attribution test on this pair that needs no GPU.

---

## L4c: Paired bucket-cluster bootstrap of the L3c gap

**Hypothesis:** the cross-fit minus threshold gap at 25% (+11.5 caught in L3c) survives cluster resampling. The gain appears only at 25% and not at 10% or 50%, so its interval should be checked.

**Change vs existing scripts:** add a paired bootstrap to the L3c script. Resample the 75 sim buckets with replacement (seed 123, as in the E03 cluster bootstrap, duplicates relabeled), keep each bucket's groups, and recompute the cross-fit caught count and the threshold caught count (8-seed mean, E03 ordering) on each resample. Use the fold-seed 0 out-of-fold scores, fixed.

**Metric:** difference (cross-fit minus threshold) in caught at 10, 25 and 50%, with 95% percentile intervals from 200 resamples.

**Pass threshold:** the 95% interval at 25% excludes 0 on the positive side.

**Kill threshold:** the 95% interval at 25% includes 0. The L3c 25% gain is then within clustered noise, and the labeled ceiling is not established.

**Expected runtime:** under 3 min, CPU.

**Informs:** claim 2 (how much a labeled ranker can add over the threshold), and how much weight the L3c PASS can bear before L3b is run.

---

Proposed from L5a (2026-10-18, kill: the shipped LearnedAllocator admits more corrupted rewards than the label-free threshold at 25%, D -0.454). Ids use loop iteration 5 (the iteration that produced L5a), continuing at b because L5a is taken. All three are CPU only and each needs its own pre-registration before it runs. Run order: L5b, then L5c, then L5d.

---

## L5b: Shipped regression ranking without the Thompson layer (decomposition of L5a)

**Hypothesis:** the L5a loss sits in the Thompson sampling layer, not in the regression. The shipped regression, with its cold start, refit cadence and features, ranking by predicted error rate with no per-group Beta draw, ties or beats the label-free threshold at 25%.

**Change vs existing scripts:** reuse avrl/exp/l5a_learned_class.py with the arm swapped to the shipped GreedyOnlineAllocator from avrl/allocators.py (imported read-only). Same K = 1 whole-pool replay, budgets, seeds and both bootstraps. E03's greedy_online arm is E03's own run_online (greedy=True) and is not known to match the shipped class, so the pre-registration must say so.

**Metric:** D(greedy_online, shipped) at 25% with the group and bucket-cluster 95% CIs, plus caught at 10% and 50% (descriptive).

**Pass threshold:** point D > 0 at 25%, and both 95% CIs exclude 0 on the positive side.

**Kill threshold:** otherwise. A tie with a CI that includes 0 is a kill for the claim, but it localizes the loss to the Thompson layer (E03's own greedy arm tied the threshold at 25%, D -0.001).

**Expected runtime:** under 10 min, CPU.

**Informs:** which part of the shipped class fails, and whether L3b, which warm-starts the same Thompson layer, is worth running.

---

## L5c: Batch candidate sets for the shipped class (phase-1 shape)

**Hypothesis:** part of the L5a loss comes from the whole-pool candidate set (caveat 1 of the L5a pre-registration). With candidates drawn as a random batch per step, the shipped class's ranking admits fewer corrupted rewards than the threshold applied to the same batches.

**Change vs existing scripts:** copy avrl/exp/l5a_learned_class.py. Each step draws a random batch of the still-unverified groups, with the batch size fixed in the pre-registration. Per-step verify count = round(budget fraction x batch size). Run learned_class and threshold on the same batches. avrl/PROJECT.md does not state the batch size (checked), so it must come from the phase-1 harness spec.

**Metric:** D = admitted(batch threshold) minus admitted(batch learned_class) at 25%, with group and bucket-cluster 95% CIs.

**Pass threshold:** point D > 0 at 25%, and both 95% CIs exclude 0 on the positive side.

**Kill threshold:** otherwise.

**Expected runtime:** under 15 min, CPU.

**Informs:** whether the claim-2 kill holds under the candidate structure phase 1 uses. Medium-high information for the phase-1 design. It does not reopen the whole-pool verdict.

---

## L5d: Feature-scaling sensitivity for the shipped regression

**Hypothesis:** the unscaled 22-column logistic regression inside the shipped class is mis-specified, and standardizing the features (fit online on verified groups only) closes part of the L5a gap.

**Change vs existing scripts:** copy avrl/exp/l5a_learned_class.py. Wrap the shipped class's inputs in an online standardizer whose mean and sd come from verified groups only. Everything else as in L5a.

**Metric:** D(learned_class, scaled) at 25% with group and bucket-cluster 95% CIs.

**Pass threshold:** point D > 0 at 25%, and both 95% CIs exclude 0 on the positive side.

**Kill threshold:** otherwise. Scaling is the cheapest cause to rule out, since L3c's scaled, class-balanced regression beat the threshold by about 8 caught at 25% (fold-seed mean). This is a sensitivity variant, not a retune.

**Expected runtime:** under 10 min, CPU.

**Informs:** whether the shipped class is retired or only mis-specified. Medium-low information, and it should run after L5b and L5c.

---

## L6a: Batch-shape decomposition of the shipped class (greedy ranking, no Thompson layer)

**Hypothesis:** the L5c loss sits in the Thompson sampling layer, not in the regression. Under batch-16 candidate sets, the shipped GreedyOnlineAllocator (avrl/allocators.py: rank the batch by predicted corrupted-reward probability, no per-cell Beta draws) ties or beats threshold_batch at 25%.

**Change vs existing scripts:** copy avrl/exp/l5c_batch_learned_class.py and swap the learned_batch arm for GreedyOnlineAllocator, imported read-only. Keep the batch plan, budgets, seeds, both bootstraps and the pool exactly as in L5c, so the arm is paired with the L5c threshold rows. Launch through bin/gpu_run.sh as L5c was.

**Metric:** D = admitted(threshold_batch) minus admitted(greedy_batch) at 25%, with group and bucket-cluster 95% percentile CIs. Caught at 10% and 50% is descriptive.

**Pass threshold:** point D > 0 at 25%, and both 95% CIs exclude 0 on the positive side.

**Kill threshold:** otherwise. A pass would be a new allocator class (the greedy ranker, not the shipped LearnedAllocator) and would need its own pre-registration before any phase-1 use. It would not revive the shipped-class kill.

**Expected runtime:** under 5 min, CPU.

**Informs:** whether the phase-1 failure is the Thompson layer (fixable by changing the class) or the regression and its features (then L5d and L6c). This is the phase-1-shape version of L5b. L5b stays the whole-pool localization if L6a passes or ties.

---

## L6b: Selection diagnostic for the shipped class in batch shape (analysis-only, descriptive)

**Hypothesis (mechanism, untested):** in batch shape the shipped class's ranking is set mostly by Thompson draws from cell posteriors seeded at prior_strength 2.0 times the predicted error rate, not by the regression. Near the 0.14 base rate, an unseen cell's prior is Beta(0.28, 1.72), a draw with most of its mass near 0 and a heavy upper tail.

**Change vs existing scripts:** replay the L5c learned_batch arm with no change to its logic, and record per batch and per seed: (a) the overlap between the chosen b_i groups and the top-b_i groups by the model's current predicted probability; (b) the within-batch Spearman correlation between the Thompson score and the predicted probability; (c) the share of selections drawn from cells with no verified update yet.

**Metric:** the three quantities above, summarized over batches and seeds. No pass or kill rule. This is a descriptive run and makes no claim.

**Expected runtime:** under 5 min, CPU.

**Informs:** whether L6c is worth running. If (a) is near 1, the Thompson layer is not changing the decision and the loss is in the regression. If (a) is near the random overlap of b_i/16 (0.25 at 25%), the Thompson draws set the ranking. Medium information for the phase-1 design, at almost no compute cost.

---

## L6c: Prior-strength variant of the shipped class in batch shape (conditional on an L6a pass)

**Hypothesis:** if L6a passes, the Thompson layer can be made harmless by a stronger prior, so that the draws concentrate near the predicted rate and the regression ranking is kept with some exploration left in it. A prior_strength of 200 instead of the class default 2.0 gives this.

**Change vs existing scripts:** a new variant, not a retune of the shipped class. Copy avrl/exp/l5c_batch_learned_class.py and set LearnedAllocator(prior_strength=200.0), refit_every at its default. Everything else identical to L5c. Pre-register it as a separate entry before running.

**Metric:** D = admitted(threshold_batch) minus admitted(learned_batch, prior_strength 200) at 25%, with group and bucket-cluster 95% CIs.

**Pass threshold:** point D > 0 at 25%, and both 95% CIs exclude 0 on the positive side.

**Kill threshold:** otherwise. A pass would be a variant of the shipped class and would reopen claim 2 only for that variant, which needs a fresh pre-registration with its own claim-2 entry.

**Expected runtime:** under 10 min, CPU.

**Informs:** whether the shipped class's failure is the prior scale. Medium information. It runs only if L6a passes, because if the greedy ranker ties the threshold there is no regression signal for a stronger prior to preserve.

---

## L7a: Surface-feature diagnostic for the judge's pick inside split groups (analysis-only, descriptive)

**Hypothesis (mechanism, untested):** the 0.5B judge's pick is at chance among distinct answers in split groups (30.7% correct against 29.7% for a uniform pick over distinct answers, from E04). If its p_yes carries any usable signal, the signal should show up as a surface property of the candidate answer (the answer string copies a number from the question, its magnitude, its length, or whether it is an integer) and not in prompt embeddings.

**Change vs existing scripts:** new CPU script. Reads avrl/runs/E04/judge_scores.parquet, avrl/runs/p0/rollouts.parquet and avrl/runs/p0/groups.parquet (question text). For each split group and each distinct answer, record mean p_yes, correctness and the surface features above. Report, per feature, the within-group pick accuracy of the argmax rule and the within-group rank correlation of p_yes with correctness. No change to any E01 or E04 script.

**Metric:** within-split-group pick accuracy under each surface rule, against the uniform-over-distinct baseline (29.7% on this data); within-group Spearman correlation of p_yes with correctness.

**Pass threshold:** none. This is descriptive and makes no claim. A surface rule that beats the uniform baseline by at least 10 points would justify L7b only if it is pre-registered first.

**Kill threshold:** none.

**Expected runtime:** under 5 min, CPU.

**Informs:** whether the 0.5B judge has any usable signal inside split groups. If no surface feature beats chance, a rubric prompt or a larger judge (L7b) is the only route left for the judge-proxy lever. Medium information at almost no compute cost.

---

## L7b: Judge capacity check with a 4B judge (GPU, one job, same E04 protocol)

**Hypothesis:** the E04 failure is a capacity failure of the 0.5B judge. A 4B judge gets a pick accuracy in split groups at least 10 points above the uniform-over-distinct baseline (29.7%, so at least 39.7%), and its judge error then clears the E04 structure or claim-2 readout.

**Change vs existing scripts:** copy avrl/exp/e04_judge_score.py to a new script with --judge /homes/rdubash/avrl_live/CASCADE/models/Qwen3-4B, gpu_memory_utilization 0.85 (about 8 GB of bf16 weights does not fit at 0.5), and llm.chat called with chat_template_kwargs={"enable_thinking": False} so the first generated token is Yes or No. The prompt, the p_yes normalization and the top-k of 20 stay as in E04. Everything downstream (e04_judge_structure.py and the E03 replay copy with LABEL judge_error) is unchanged. Launch through bin/gpu_run.sh after gpu_status.sh shows under 2 GB used.

**Metric:** the E04 structure readout (AUC gain, Jaccard, neighbor ratio at cell100), the E04 claim-2 readout at 25% (D(ts_online) with group bootstrap 95% CI), and the split-group pick accuracy against the 29.7% uniform baseline.

**Pass threshold:** the E04 rules unchanged. Structure pass: AUC gain >= 0.05 and Jaccard < 0.5. Claim-2 pass: point D(ts_online) > 0 at 25% and the 95% CI excludes 0. A judge-proxy go needs both.

**Kill threshold:** the E04 kill rules unchanged. Also: if split-group pick accuracy is within 5 points of 29.7%, record the judge as no better than chance at 4B and close the judge-proxy lever on this task.

**Expected runtime:** about 10 min on GPU (the 0.5B run took 83 s for 24,000 prompts, and a 4B model is roughly 8 times the compute). Well inside the 45 min cap.

**Informs:** the most important follow-up for the proxy-type lever. It separates a capacity failure of the 0.5B judge from a failure shared by small judges on this task. The rollout policy is Qwen2.5-1.5B-Instruct, so using it as judge would be self-judging. Qwen3-4B avoids that confound. High information.

---

## L7c: Judge score as an extra feature for the majority-vote label (CPU, not as a verifier)

**Hypothesis:** the judge is not a usable verifier (E04), but the judge's mean p_yes for the majority answer, and the top-two gap in mean p_yes, may carry information about majority-vote errors that agreement misses. The E01 cheap-feature AUC for proxy_error is 0.923. Test whether two judge features move it.

**Change vs existing scripts:** new CPU script. Reads avrl/runs/E04/judge_scores.parquet and avrl/runs/E04/feats_judge.parquet. Adds two per-group features: mean p_yes of the majority answer and the top-two gap of mean p_yes over distinct answers. Fits 5-fold balanced logistic regression on cheap, cheap + judge and cheap + emb + judge with label proxy_error (E01 label). Then runs the E03 replay copy with LABEL proxy_error and the new columns at 25%.

**Metric:** AUC for proxy_error, cheap against cheap + judge (the gain), and the 25% claim-2 readout of D(ts_online) and D(greedy_online) with bootstrap CI, as in E03.

**Pass threshold:** proposed, to be pre-registered before the run: AUC gain >= 0.02, and the E03 claim-2 rule passes (point D(ts_online) > 0 at 25%, CI excludes 0).

**Kill threshold:** AUC gain < 0.02. Anything else is inconclusive.

**Expected runtime:** under 10 min, CPU.

**Informs:** whether the judge is useful as an allocator feature even though it fails as a verifier. Low to medium information: agreement alone gives AUC 0.91 for the majority label, and the judge is at chance inside split groups, so the expected gain is small. Run it only if L7a finds a surface signal.

---

## L8a: Warm-started bucket prior for ts_online (L3b, run now that the judge lever is closed)

**Hypothesis:** the only learner variant on the claim-2 chain that has not been run is ts_online with its initial bucket prior set from the label-free threshold (L3b). Every cold-start learner has lost to the threshold (E01, E03, L5a, L5c, and L7b on the judge label), and a warm prior is the one untested change to the Thompson layer.

**Change vs existing scripts:** the L3b design as written in this file. New CPU script (a copy of avrl/exp/e04_judge_replay.py, not an edit of it) in which the ts_online prior is the per-bucket mean agreement from the threshold rule instead of Beta(1,1). The online refit is unchanged. Input: avrl/runs/p0/feats.parquet (proxy label, E01 pool). No GPU.

**Metric:** D(ts_online) = admitted(threshold) minus admitted(ts_online, warm prior) at 25% on the proxy label (cell100, cheap+emb, 2,180 sim groups, 8 seeds), with the 200-resample group bootstrap (BOOT_SEED 0). Descriptive at 10% and 50%.

**Pass threshold:** point D above 0 at 25%, with the 95% CI excluding 0 (the L3b rule).

**Kill threshold:** point D at or below 0 at 25%, or a CI that includes 0. A kill closes the Thompson layer on this pair.

**Expected runtime:** under 15 min, CPU.

**Informs:** whether any deployable learner beats the threshold on this pair. High information for claim 2. Sequencing: the L3b note said to run after L4a. L4a is still unlogged, so its log should come first if the order matters.

---

## L8b: Judge features for the majority-vote label, rerun with the Qwen3-4B scores (L7c on L7b)

**Hypothesis:** the Qwen3-4B judge's mean p_yes for the majority answer, and the top-two gap of mean p_yes, add to the cheap features for the majority-vote error label. L7c was written when the judge was at chance inside split groups. The 4B judge is above chance (45.8% against 29.7%), so that premise is weaker now.

**Change vs existing scripts:** L7c as written, with avrl/runs/L7b/judge_scores.parquet and avrl/runs/L7b/feats_judge.parquet in place of the E04 files. New CPU script; no change to E01 or E04 scripts. The pre-registration must be written before the run, as L7c required.

**Metric:** AUC for proxy_error, cheap against cheap plus judge (the gain), and the 25% claim-2 readout of D(ts_online) with its group bootstrap CI.

**Pass threshold (to be pre-registered before the run):** AUC gain at least 0.02, and the E03 claim-2 rule passes (point D above 0 and CI excludes 0).

**Kill threshold:** AUC gain below 0.02. Anything else is inconclusive.

**Expected runtime:** under 10 min, CPU.

**Informs:** whether a judge score helps the allocator once the judge is above chance. Medium-low information. The 4B judge agrees with the majority answer in only 54.8% of groups, and 992 of its 1,327 errors are judge-only, so a gain is possible but probably small. This tests the judge as a feature, not as a verifier.

---

## L8c: Bucket-cluster bootstrap of the judge-label claim-2 readout

**Hypothesis:** the L7b claim-2 interval is a group bootstrap, which treats the 2,180 sim groups as independent even though they share 75 cell100 buckets. The E01 bucket-cluster interval for D(ts_online) was similar in width to its group interval (cluster [-0.225, -0.131] against group [-0.230, -0.142]), so the judge-label interval should be checked the same way.

**Change vs existing scripts:** new CPU script that repeats the E01 inline bucket-cluster bootstrap (resample the 75 buckets with replacement, duplicates relabeled, 200 resamples, seed 123, 25% budget) on avrl/runs/L7b/feats_judge.parquet with LABEL judge_error. Same arms, seeds and cell100 buckets as L7b. No GPU.

**Metric:** bucket-cluster 95% CI of D(ts_online) and D(greedy_online) at 25% on the judge label.

**Pass threshold:** the cluster CI for D(ts_online) lies entirely below 0. This confirms the L7b claim-2 kill.

**Kill threshold:** none for a new hypothesis. If the cluster CI includes 0, the L7b claim-2 readout is recorded as inconclusive in the ledger, not reversed. The point estimate (-0.070) is unchanged either way.

**Expected runtime:** under 5 min, CPU.

**Informs:** whether the claim-2 kill on the judge label depends on group dependence. Low information, because the point estimate is far from 0, but it closes the one open caveat on the kill's interval.

---

Proposed from E10 (2026-10-18, kill for claim 3 on OLMo-2-0425-1B; claim 2 gate fail, no verdict). Ids use loop iteration 9 (the iteration that produced E10). L9a and L9b are CPU only. L9c needs one GPU job through bin/gpu_run.sh.

---

## L9a: Cross-family prompt-difficulty check on the shared GSM8K prompts (CPU, diagnostic)

**Hypothesis:** the E01 and E10 policies share prompt-level difficulty (per-prompt group proxy error correlates 0.45 across the same 3,000 questions), so the claim-3 kill reflects the MiniLM cell, PC and cheap feature set, not the absence of prompt-level signal.

**Change vs existing scripts:** new CPU script that joins avrl/runs/p0/feats.parquet and avrl/runs/E10/feats.parquet on pid. It uses the E01 group proxy error as one extra feature for predicting the E10 group proxy error (and the reverse), with the 5-fold balanced logistic from e10_analyze.py and a cell100 bucket-cluster bootstrap for the interval. Diagnostic only: E01 rollouts come from a second policy, so the feature is not label-free for a single-policy allocator.

**Metric:** AUC of E10 proxy error for cheap alone, cheap plus E01 error, and cheap plus embedding. Report the incremental gain of E01 error over cheap, with a bucket-cluster 95% CI, and the same in the reverse direction.

**Pass threshold (reopens claim 3 as a representation question, not a rescue):** incremental gain from E01 error over cheap at least 0.05, with the CI excluding 0.

**Kill threshold:** incremental gain below 0.02. The claim-3 kill then stands as a finding about the representation tested, and the prompt-structure claim is recorded as not reachable with these features.

**Expected runtime:** under 5 min, CPU.

**Informs:** whether the claim-3 kill means no prompt-level structure or no structure reachable by MiniLM and cheap features. High information for how claim 3 is worded after two kills.

---

## L9b: Gate-reachability audit for the claim-2 confident subset (CPU, descriptive)

**Hypothesis:** the 3% confident-subset gate is not reachable for GSM8K majority-vote proxies at G 8 (E01 1.1%, E10 1.6%), so claim 2 cannot receive a verdict on this task whatever the policy.

**Change vs existing scripts:** new CPU script over the existing groups tables of avrl/runs/p0 and avrl/runs/E10 only. For agreement cutoffs 0.50, 0.625 and 0.75, it reports the group count, error count and error rate of the subset, per family.

**Metric:** confident-subset proxy error rate and error count at each cutoff, per family.

**Pass threshold:** none. This is descriptive and only informs a new pre-registration. It does not change the E01 or E10 verdicts, and the existing gate is not edited.

**Kill threshold:** no cutoff reaches 3% with at least 100 errors in the subset for either family. In that case the claim-2 gate cannot be met on GSM8K majority vote at G 8, and the next claim-2 test needs a different task or proxy.

**Expected runtime:** under 5 min, CPU.

**Informs:** whether the project needs a gate decision from Rustom or a different task. High information for the program, and it can be run before any GPU work.

---

## L9c: Truncation control, OLMo-2-0425-1B at max_tokens 1024 (GPU, one job, same E10 protocol)

**Hypothesis:** the 6.75% of E10 rollouts that hit the 512-token cap (1,620 of 24,000) create some proxy errors, so part of the confident-subset error and of the cheap-feature AUC reflects truncation rather than the policy's answers.

**Change vs existing scripts:** avrl/exp/E10.sh with max_tokens 1024 and the output directory changed to avrl/runs/L9c. Same model path, 3,000 prompts, G 8, seed 0. The analysis is e10_analyze.py and the E03 replay, both unchanged. Launch only through bin/gpu_run.sh after bin/gpu_status.sh shows under 2 GB used.

**Metric:** truncated rollout fraction, confident-subset error rate, AUC for cheap and cheap plus embedding, cell100 neighbor ratio, and D(ts_online) at 25% (descriptive).

**Pass threshold:** the E10 readings do not change. Confident-subset error moves by less than 0.5 percentage points and the claim-3 AUC gain stays below 0.05.

**Kill threshold:** none, because this is a control. If the confident-subset error reaches 3% or more, the E10 gate fail was a truncation artifact and must be re-examined before any verdict is recorded on that family.

**Expected runtime:** about 25 min GPU (E10 took 821 s at 512 tokens), under the 45 min cap, plus about 5 min CPU.

**Informs:** whether the E10 gate fail and the claim-3 reading are measurement artifacts. Medium-low information on the claims, but it removes a known confound before the family-level results are cited.

---

Proposed from L9a (2026-10-18, kill at the pre-registered point rule: cross-family residual 0.016 AUC, below 0.02). Ids use loop iteration 10 because L9a was run from the iteration 9 set.

---

## L10a: Oracle-ceiling replay for claim 2 (E01 proxy_error as an allocator feature on E10, CPU)

**Hypothesis:** even with the one cross-family feature that L9a found, the learned allocator does not beat the label-free agreement threshold at a 25% budget on the E10 pool.

**Change vs existing scripts:** copy avrl/exp/e03_threshold_bootstrap.py to a new script and append the E01 proxy_error column (joined on pid from avrl/runs/p0/feats.parquet) to the allocator features of the ts_online arm. Keep the threshold arm, the 2,180-group sim pool, the 8 seeds and the group bootstrap (200 resamples, BOOT_SEED 0) unchanged. Diagnostic only: the feature needs another policy's rollouts.

**Metric:** D = admitted(threshold) minus admitted(learned + E01 feature) at 10%, 25% and 50%, with a 95% group-bootstrap CI. Reference: the existing E10 D(ts_online) values.

**Pass threshold (diagnostic, not a deployable allocator):** D above 0 at 25% with the CI excluding 0, meaning the learned arm with the cross-family feature admits fewer corrupted rewards than the threshold.

**Kill threshold:** D at or below 0 at 25% (point estimate). Then the threshold still admits fewer corrupted rewards with a cross-policy oracle feature, and claim 2 is closed beyond cheap features.

**Expected runtime:** about 10 min, CPU.

**Informs:** how far the claim-2 loss is from the best a learned allocator could reach with the only residual signal L9a found. High information for how claim 2 is worded.

---

## L10b: G 16 rollouts on the E10 protocol (measurement-noise check for the gate and the residual, GPU, one job)

**Hypothesis:** the G 8 majority-vote proxy limits both the confident-subset gate (E10 at 1.6% against 3%) and the 0.016 AUC residual from L9a, so more rollouts per prompt would change one of them.

**Change vs existing scripts:** avrl/exp/E10.sh with --G 16 and the output directory changed to avrl/runs/L10b. Same model (OLMo-2-0425-1B-Instruct), same 3,000 GSM8K prompts, max_tokens 512, seed 0. Analysis is e10_analyze.py and the E03 replay, both unchanged. Launch only through bin/gpu_run.sh after bin/gpu_status.sh shows under 2 GB used. Projected generation about 28 min (E10 took 821 s for 24,000 rollouts, so 48,000 is about 1,640 s). If the projection exceeds 40 min, cut to 1,500 prompts before launch.

**Metric:** confident-subset (agreement at least 0.75) error rate and error count, AUC gain of cheap plus embedding over cheap, D(ts_online) at 25% (descriptive), and the L9a gain from the E01 feature (descriptive).

**Pass threshold:** confident-subset error at least 3% with at least 100 errors in the subset, so the claim-2 gate is reachable on this task at G 16.

**Kill threshold:** confident-subset error stays below 3% and the cheap plus embedding AUC gain stays below 0.05. Then the gate failure is a property of the task, not of the G 8 measurement.

**Expected runtime:** about 28 min GPU plus about 10 min CPU analysis, under the 45 min cap.

**Informs:** whether the claim-2 gate failure and the L9a residual are limited by rollout count. Medium information for the claims, and it decides whether a new task is needed.

---

## L10c: Bucket-definition sensitivity for L9a (CPU, descriptive)

**Hypothesis:** the L9a gain and its CI do not depend on whether the bootstrap resamples the 100 full-statistic buckets or the 75 sim-pool buckets that the pre-registration named.

**Change vs existing scripts:** copy avrl/exp/l9a_crossfamily_difficulty.py to a new script that restricts the statistic to the 2,180-row sim pool (75 buckets, the set used by avrl/runs/p0/sim_alloc.py) and bootstraps those 75 buckets with 1,000 resamples and seed 123. Report it beside the 100-bucket CI already in avrl/runs/L9a/result.json.

**Metric:** gain B minus A (point and 95% CI) on the sim pool, and the reverse gain on the same pool.

**Pass threshold:** none. This is descriptive and only closes the bucket record gap in the L9a entry. It does not change the L9a verdict, which is point-based.

**Kill threshold:** none.

**Expected runtime:** about 5 min, CPU.

**Informs:** whether the 0.016 residual survives the pre-registered bucket definition. Low information for the claims, and it removes a documented deviation from the record.

---

Proposed from L10a (2026-10-18, KILL at the pre-registered point rule: D(ts_online with E01 proxy_error) = -0.205 at 25%, CI [-0.263, -0.172]). Ids use loop iteration 11 because L10a was run from the iteration 10 set.

---

## L11a: Confident-subset test for claim 3 (CPU, descriptive, fixed effect bar)

**Hypothesis:** the cross-policy prompt-difficulty signal in the confident subset (agreement at least 0.75; L9a found 24 of 35 confident E10 errors shared with E01, AUC 0.721 to 0.916 against 0.715 for the MiniLM PCs) is real and is testable with a pre-set bar.

**Change vs existing scripts:** new copy of avrl/exp/l9a_crossfamily_difficulty.py, restricted to the agreement at least 0.75 subset. Same arms: A (cheap), B (cheap plus E01 proxy_error), C (cheap plus MiniLM PCs). Bucket-cluster bootstrap over cell100 buckets, 1,000 resamples, seed 123. Also report the share of shared errors with the identical wrong majority answer. Existing scripts unchanged. Replaces the L9a critic's suggested L10d.

**Metric:** AUC gain B minus A on the confident subset (point and 95% bucket-cluster CI); AUC gain C minus A; shared-error share; identical-wrong-answer share.

**Pass threshold:** point gain B minus A at least 0.05 with a 95% CI excluding 0. Claim 3's representation question then has a testable target in the confident-wrong region.

**Kill threshold:** point gain B minus A below 0.02. Then no confident-subset structure beyond cheap features is measurable on this task, and the representation question closes. Between 0.02 and 0.05, or a CI including 0 at 0.05 or above, is INCONCLUSIVE.

**Expected runtime:** under 5 min, CPU.

**Informs:** whether claim 3's open representation question has a measurable target. Medium information for the claim 3 wording. Power is low (35 errors in the subset), so the CI will be wide.

---

## L11b: Robustness of the labeled static arm with the E01 feature (CPU, descriptive, bounded)

**Hypothesis:** the one arm where the E01 feature moves the 25% comparison (static, labeled: admitted 0.108 with the feature, 0.157 without, against 0.134 for the threshold) reflects a real gap and not one fit split.

**Change vs existing scripts:** in a copy of avrl/exp/l10a_oracle_ceiling.py, rerun the static arm on the same 8 seeds as the other arms (the L10a copy fits it once on the seed-0 shuffle, n_runs 1), and bootstrap D(static) = admitted(threshold) minus admitted(static) at 25% with the same group bootstrap (200 resamples, BOOT_SEED 0). Report with and without e01_proxy_error. Existing scripts unchanged.

**Metric:** D(static) at 25% with and without E01, mean over 8 seeds, group-bootstrap 95% CI.

**Pass threshold:** D(static with E01) above 0 with the CI excluding 0. Then a labeled cross-policy ranker beats the threshold at 25%. This is a ceiling result, not a deployable allocator.

**Kill threshold:** point D(static with E01) at or below 0 across the seed mean. Then the 0.026 gap in the L10a run was one fit split.

**Expected runtime:** under 10 min, CPU.

**Informs:** how claim 2 should be worded for labeled learned allocators. High information: the L10a kill covers only the online ts_online learner, and this arm is the only evidence that a learned rule could beat the threshold with the cross-policy feature.

---

## L11c: Bucket-cluster bootstrap of the L10a 25% D (CPU, descriptive)

**Hypothesis:** the L10a KILL does not depend on resampling sim groups instead of buckets.

**Change vs existing scripts:** in a copy of avrl/exp/l10a_oracle_ceiling.py, bootstrap over the 75 sim-pool cell100 buckets instead of sim groups, 200 resamples, BOOT_SEED 0. Report D(ts_online) and D(greedy_online) at 25% with these CIs beside the group-bootstrap CIs in avrl/runs/L10a/result.json. Existing scripts unchanged.

**Metric:** D(ts_online) at 25% with the bucket-cluster 95% CI.

**Pass threshold:** none. Descriptive.

**Kill threshold:** none. The verdict is point-based (point D at or below 0), so resampling cannot change it.

**Expected runtime:** under 5 min, CPU.

**Informs:** the CI width for the claim 2 wording. Low information: the group-bootstrap CI [-0.263, -0.172] is far from 0. Mirrors L8c.

---

Proposed from L9b (2026-10-18, result: hypothesis not supported, E10 at agreement 0.50 meets both flags; verdict inconclusive). Ids use loop iteration 10 because L9b was run from the iteration 9 set, continuing at d because L10a to L10c are taken. The L9a critic's suggested L10d was replaced by L11a and never filed, so d is free. Both are CPU only. L10b (existing, G 16, GPU) is the other route to a different agreement grid and is not repeated here.

---

## L10d: Tie-rule and band split for the L9b 0.50 cell (CPU, descriptive)

**Hypothesis:** the one cell that meets both flags (E10 at agreement 0.50) is carried by unique 4-of-8 pluralities, not by the lexicographic tie rule in p0_rollouts.py.

**Change vs existing scripts:** new copy of avrl/exp/l9b_gate_audit.py. Split the exact-0.50 band (agreement equal to 0.50) by n_distinct. For the groups with n_distinct 2, recompute the majority under a random tie-break from the per-rollout answers in avrl/runs/E10/rollouts.parquet and avrl/runs/p0/rollouts.parquet. Report error rate per band with Wilson 95%. Existing scripts unchanged.

**Metric:** error count and rate per n_distinct band in the exact-0.50 band, for E01 and E10; expected proxy error of the 4-4 ties under a random tie-break.

**Pass threshold:** none. Descriptive.

**Kill threshold:** none. Descriptive, and it cannot change a verdict.

**Expected runtime:** under 5 min, CPU.

**Informs:** whether the 0.50 pass needs a tie rule in any future gate. Medium information. Only 21 (E10) and 13 (E01) groups are ties, so the tie rule can move at most that many errors. The band's high error rate (E10 29.7%) sits mostly in unique-plurality groups, which the tie rule does not touch.

---

## L10e: Sample size needed for the 100-error floor (CPU, planning arithmetic)

**Hypothesis:** at the observed rates, the 100-error floor cannot be met on the 0.75 confident subset of the existing 3,000 GSM8K prompts, and it needs a pool several times larger.

**Change vs existing scripts:** new CPU script over the groups tables of avrl/runs/p0 and avrl/runs/E10 only. For cutoffs 0.625 and 0.75, compute the confident groups needed for 100 errors (100 divided by the point rate), and the prompts needed at the observed confident share. Report at the observed point rate and at a fixed 3% rate, with the Wilson lower bound of the observed rate beside it. State that this assumes the observed rate and share hold on a larger pool. Existing scripts unchanged.

**Metric:** prompts needed for at least 100 errors in each cutoff subset, per family, at the observed point rate and at 3%.

**Pass threshold:** none. Descriptive planning number.

**Kill threshold:** none. Descriptive planning number.

**Expected runtime:** under 2 min, CPU.

**Informs:** whether a larger GSM8K prompt pool (new GPU rollouts) can satisfy the floor, or whether the next claim-2 test has to change task. Expected values from the L9b table: E10 at 0.75 about 6,200 confident groups and about 8,600 prompts; E01 at 0.75 about 9,100 groups and about 17,600 prompts; at 3% about 3,300 groups. High for the program decision, low cost.

---

Proposed from L11b (2026-10-18, PRIMARY KILL at point D -0.008; CEILING INCONCLUSIVE at point D +0.008). Ids use loop iteration 12 because L11b was run from the iteration 11 set. All three are CPU only.

---

## L12a: Confirmatory test of the labeled static arm on cheap six plus E01 (CPU, pre-registered)

**Hypothesis:** the static ranker on the six cheap columns plus e01_proxy_error beats the label-free agreement threshold at 25% across fit resamples. The L11b row (D +0.0354, CI [+0.0112, +0.0582]) is the best of the five static feature sets run there, so it is a lead, not a result.

**Change vs existing scripts:** new copy of avrl/exp/l11b_low_dim_ceiling.py that runs one static arm (cheap six plus e01_proxy_error) with the L11b refit (8 seeds, seeded 80% bucket subsample of the fit quarter) and the L11b group bootstrap (200 resamples, BOOT_SEED 0). Add a bucket-cluster bootstrap over the 75 cell100 buckets as a second CI. Report the other static feature sets as descriptive only. Existing scripts unchanged.

**Metric:** seed-mean D(static, cheap six plus E01) at 25%, with the group-bootstrap and bucket-cluster 95% CIs.

**Pass threshold:** seed-mean point D above 0 and both CIs exclude 0. A pass is a labeled cross-policy finding only and does not license a single-policy allocator.

**Kill threshold:** seed-mean point D at or below 0. Point D above 0 with either CI including 0 is INCONCLUSIVE, not a kill.

**Expected runtime:** under 5 min, CPU.

**Informs:** whether the only positive labeled row survives a fixed, pre-registered feature set. High information: it separates a real labeled gain from a feature-set selection effect.

---

## L12b: Hand-rule sensitivity to the E01 flagged count (CPU, descriptive)

**Hypothesis:** the hand rule's +0.042 at 25% (threshold order for 445 docs, then 100 E01-flagged docs) is not a knife edge at the untuned count of 100.

**Change vs existing scripts:** new copy of avrl/exp/l11b_low_dim_ceiling.py that sweeps the flagged count k over 0, 50, 100, 150 and 200 at 25%, filling the remaining slots in threshold order with the same seeded tie-break, 8 seeds, group bootstrap (200 resamples, BOOT_SEED 0). k = 0 must reproduce the threshold arm, as a built-in check. Existing scripts unchanged.

**Metric:** 8-seed mean D at 25% for each k, with the group-bootstrap 95% CI.

**Pass threshold:** none. Descriptive.

**Kill threshold:** none. Descriptive, and it cannot change a verdict.

**Expected runtime:** under 5 min, CPU.

**Informs:** how much of the oracle-flag gain depends on one untuned split, and so what bar a learned rule would have to clear. Medium information. The hand rule uses oracle E01 flags, so it is a reference ceiling and not an allocator candidate.

---

## L12c: Label-free greedy on cheap six, E01 GSM8K family (CPU, pre-registered)

**Hypothesis:** the label-free greedy_online arm on the six cheap columns beats the agreement threshold at 25% on the second GSM8K family (Qwen2.5-1.5B-Instruct rollouts in avrl/runs/p0/feats.parquet), unlike the L11b PRIMARY on the E10 pool (D -0.0079, CI [-0.0439, +0.0089]).

**Change vs existing scripts:** avrl/exp/l11b_low_dim_ceiling.py as it stands, run with --feats avrl/runs/p0/feats.parquet --cell cell100 --features cheap and no --e01_feats, 8 seeds, 200 resamples, BOOT_SEED 0. No code change. The threshold row is the baseline. Existing scripts unchanged.

**Metric:** 8-seed mean D(greedy_online, cheap six) at 25%, with the group-bootstrap 95% CI.

**Pass threshold:** point D above 0 at 25% and the CI excludes 0. A finding only, since the E01 family also fails the 3% confident-subset gate (1.10%, L9b).

**Kill threshold:** point D at or below 0 at 25%. Point D above 0 with a CI that includes 0 is INCONCLUSIVE.

**Expected runtime:** under 2 min, CPU.

**Informs:** whether the label-free null in L11b is specific to one GSM8K family. Medium information. Either outcome changes the claim 2 wording for label-free arms, and it cannot change the gate status.

---
Proposed from E06 (2026-10-18: gate inconclusive, claim 3 kill on the 0.5B policy, claim 2 no verdict). Ids use loop iteration 13 (the iteration that produced E06).

---

## L13a: Pre-registered cutoff-0.625 gate on all GSM8K train prompts, Qwen2.5-0.5B (GPU, one job)

**Hypothesis:** on all 7,473 GSM8K train prompts with the 0.5B policy (G 8, max_tokens 512, seed 0), the confident-subset proxy error rate at agreement at least 0.625 is at least 3% with k at least 100, so the gate opens for a claim 2 readout on GSM8K.

**Change vs existing scripts:** a copy of avrl/exp/E06.sh with --n_prompts 7473 and --out avrl/runs/L13a, run through bin/gpu_run.sh as one job after the usual gpu_status check. Then avrl/exp/e06_analyze.py and avrl/exp/e06_gate.py, with the primary cutoff set to 0.625 in a new copy of e06_gate.py. Existing p0 and E06 scripts unchanged.

**Metric:** confident-subset (agreement at least 0.625) proxy error rate and k, with Wilson 95%. The claim 2 readout (D(ts_online) at 25%, bucket-cluster CI) only if the gate opens.

**Pass threshold:** at cutoff 0.625, rate at least 0.03 and k at least 100. Then the claim 2 readout runs as its own pre-registered item.

**Kill threshold:** at cutoff 0.625, rate below 0.03. GSM8K majority vote then fails the gate for the 0.5B policy on both cutoffs, and the next step is a task change. Rate at least 0.03 with k below 100 is INCONCLUSIVE.

**Expected runtime:** about 31 min GPU at the E06 rate (747 s per 3,000 prompts, scaled by 7,473 / 3,000, so about 1,860 s plus vLLM startup), under the 45-minute cap. About 10 min CPU.

**Informs:** whether claim 2 can be tested on GSM8K at all. High information. Expected k about 199 and rate about 7.3% are extrapolations from E06 (1,089 groups, 80 errors). Choosing 0.625 after E06 is post hoc, so this must be pre-registered as a new gate before the run and cannot change the E06 verdict.

---

## L13b: L10a oracle-ceiling replay on the E06 pool, E01 proxy_error as the allocator feature (CPU)

**Hypothesis:** with the cross-policy E01 flag available as an allocator feature, a learned allocator beats the label-free agreement threshold at 25% on the E06 pool. If so, the claim 2 failure on E06 sits in the cheap features and not in the online learner.

**Change vs existing scripts:** avrl/exp/l10a_oracle_ceiling.py run on avrl/runs/E06/feats.parquet, with the E01 proxy_error joined by pid from avrl/runs/p0/groups.parquet. The E06 sim pool is the same 2,180 prompts as E10 (cell100 assignments agree on all 3,000). Existing scripts unchanged.

**Metric:** D(ts_online) at 25% with the E01 flag as a feature, with its bucket-cluster 95% CI. E06 CI half-width is about 0.02, against about 0.04 for E10.

**Pass threshold:** point D above 0 at 25% and the CI excludes 0.

**Kill threshold:** point D at or below 0 at 25%. A point D above 0 with a CI that includes 0 is INCONCLUSIVE.

**Expected runtime:** under 5 min, CPU.

**Informs:** whether the E06 claim 2 failure comes from weak features or from the learner. Medium information. Diagnostic only: while the gate is closed it cannot issue a claim 2 verdict.

---

## L13c: Confident-subset claim 3 check on the E06 pool, with a truncation cross-tab (CPU, descriptive)

**Hypothesis:** the confident-subset AUC gain of cheap+emb over cheap on the 0.5B pool (+0.060 on 31 errors) survives a bucket-cluster bootstrap, and truncation does not drive it (4 of 31 confident error groups against 30 of 695 confident non-error groups contain a truncated rollout).

**Change vs existing scripts:** the arms of L11a (cheap, and cheap plus MiniLM PCs; the E01 arm is not needed here), run on avrl/runs/E06/feats.parquet, restricted to agreement at least 0.75, with a bucket-cluster bootstrap over cell100 buckets. Plus the truncation cross-tab over the confident groups. Existing scripts unchanged.

**Metric:** AUC gain (PCs minus cheap) on the confident subset, with its 95% bucket-cluster CI. Truncation counts by error status.

**Pass threshold:** gain at least 0.05 with the CI excluding 0. Then the lead justifies the full-pool GPU run in L13a.

**Kill threshold:** point gain below 0.02. A gain between 0.02 and 0.05, or a CI that includes 0 at 0.05 or above, is INCONCLUSIVE.

**Expected runtime:** under 3 min, CPU.

**Informs:** whether the one confident-subset lead on the 0.5B pool is worth GPU time. Low information: with 31 errors the CI will probably include 0, so INCONCLUSIVE is the likely result.

---

## L14a: Fresh-prompt confirmation of the 0.625 gate on the GSM8K test split (GPU, one job)

**Hypothesis:** the 0.625 gate passes on prompts that did not choose the cutoff. Pool: the 1,319 GSM8K test prompts plus the 4,473 train prompts that E06 did not use (already generated in avrl/runs/L13a). Expected pooled k about 117 at a rate about 5.8%.

**Change vs existing scripts:** a copy of avrl/p0_rollouts.py (p0 unchanged) that reads the GSM8K test split from avrl/data/ instead of train, run with G 8, max_tokens 512, seed 0, through bin/gpu_run.sh. Then a pooled gate at cutoff 0.625 on the 4,473 non-E06 groups from avrl/runs/L13a/groups.parquet plus the new test groups. Pre-register the pool before the run.

**Metric:** pooled confident-subset (agreement at least 0.625) rate and k, with Wilson 95%.

**Pass threshold:** pooled rate at least 0.03 AND k at least 100. Then the claim 2 readout (L14b) runs on this pool.

**Kill threshold:** pooled rate below 0.03. Rate at least 0.03 with k below 100 is INCONCLUSIVE.

**Expected runtime:** about 8 min GPU (1,822 s x 1,319 / 7,473 is about 5.4 min, plus vLLM startup), under the 45-minute cap. Under 2 min CPU.

**Informs:** whether the L13a PASS survives without the prompts that chose the cutoff. High information. The 0.625 cutoff stays post hoc; this checks the pool, not the cutoff. Assumes the test prompts have a confident share and error rate similar to train.

**Status (2026-10-19): done, PASS at cutoff 0.625.** Pooled n 1,869, k 118, rate 6.31% (Wilson 5.30% to 7.51%). Train part (L13a pid 3000 and above): n 1,541, k 89, rate 5.78%. Test part: n 328, k 29, rate 8.84%. Truncated share 7.05%, not flagged. The gate opens for L14b on this pool, which still needs its own pre-registration entry before it runs. The assumption above did not fully hold: the test part has a confident share of 24.9% (train part 34.4%) and a higher error rate (8.8% against 5.8%), but both parts clear the 3% rate floor. Details, checks and critic note: avrl/RESEARCH_LOG.md, section "L14a 2026-10-19 result". Result file: avrl/runs/L14a/result.json.

---

## L14b: Claim 2 readout D(ts_online) at 25% with bucket-cluster CI on the gate-passing pool (CPU, pre-registered)

**Hypothesis:** on the GSM8K pool that passes the gate (the L13a pool, or the L14a pool if that one passes), the learned online allocator ts_online (cell100 buckets, cheap+emb features) admits fewer proxy errors than the label-free threshold at 25%, so D(ts_online) is above 0.

**Change vs existing scripts:** the E06 claim 2 replay (the E03 threshold bootstrap as run on E06: cell100, cheap+emb, budgets 10/25/50%, 8 seeds, 200 bucket-cluster resamples, BOOT_SEED 0), pointed at avrl/runs/L13a or the L14a pool. Existing scripts unchanged. The pool definition is pre-registered before the run.

**Metric:** D(ts_online) at 25%, defined as admitted(threshold) minus admitted(ts_online), with a bucket-cluster 95% CI. The sign convention is the one used in the E06 entry.

**Pass threshold:** point D above 0 and the CI excludes 0.

**Kill threshold:** point D at or below 0 at 25%. Point D above 0 with a CI that includes 0 is INCONCLUSIVE.

**Expected runtime:** under 15 min CPU.

**Informs:** whether claim 2 is killed on GSM8K. High information, and the only item in this series that can kill claim 2 on this pool. Prior: D(ts_online) at 25% was negative with CIs below 0 on E01 (-0.172), E06 (-0.124) and E10 (-0.203), so a kill is the likely result.

---

## L14c: Claim 3 and truncation check on the 161 confident errors (CPU, descriptive)

**Hypothesis:** with 161 confident errors on the 0.5B pool (31 in E06), the confident-subset AUC gain of cheap+emb over cheap is below 0.05, and truncation does not drive the confident errors.

**Change vs existing scripts:** the cheap and cheap+emb arms used in L13c, run on avrl/runs/L13a/feats.parquet restricted to agreement at least 0.625 (n 2,628, k 161), with a bucket-cluster bootstrap over cell100 buckets. Plus the truncation cross-tab over the confident groups as in L13c, with a Fisher exact test. Existing scripts unchanged.

**Metric:** AUC gain on the confident subset with its 95% bucket-cluster CI. Truncation counts by error status, with the Fisher exact p.

**Pass threshold (claim 3 lead):** gain at least 0.05 with the CI excluding 0. A claim 3 readout would then run as its own pre-registered item.

**Kill threshold:** point gain below 0.02. A gain between 0.02 and 0.05, or a CI that includes 0 at 0.05 or above, is INCONCLUSIVE.

**Expected runtime:** under 5 min CPU.

**Informs:** whether claim 3 has a confident-subset lead on the 7,473-prompt pool. Medium information. The descriptive 0.75 gain is +0.029 on 59 errors and the pooled gain is -0.0004, so a kill is the likely result. This is the only check on the confident subset here with more than 60 errors.

---


Proposed from L14a (2026-10-19, PASS at cutoff 0.625: pooled rate 6.31%, k 118). Ids use loop iteration 15 (the iteration after L14a). L15a and L15b need one GPU job each; L15c is CPU only. None needs the phase-1 harness.

---

## L15a: Policy versus prompts for the train-test accuracy gap (GPU, one job)

**Hypothesis:** the Qwen2.5-0.5B gap between train and test accuracy (majority vote 58.8% on the L13a train part against 46.5% on the GSM8K test split, 12.3 points) is specific to that policy and not a property of the prompts.

**Change vs existing scripts:** a copy of avrl/exp/l14a_rollouts_test.py with the OLMo-2-0425-1B-Instruct model path and the E10 settings (per avrl/exp/E10.sh), run on the 1,319 test prompts through bin/gpu_run.sh. Its majority-vote accuracy is compared with its own E10 train groups (3,000 prompts). Existing scripts unchanged.

**Metric:** OLMo majority-vote accuracy on the test prompts minus its accuracy on the E10 train prompts, set against the Qwen gap of 12.3 points.

**Pass threshold (reading, not a claim verdict):** OLMo gap at most 3 points means the gap is policy-specific, so the pooled rate is a mixture of two regimes.

**Kill threshold (reading):** OLMo gap at least 8 points means the test split is harder for both policies. A gap between 3 and 8 points is INCONCLUSIVE.

**Expected runtime:** about 6 min GPU, under the 45-minute cap; under 2 min CPU.

**Informs:** how to read the pooled rate on the L14a pool, as a representative sample or a mixture. High information for the pool, no direct effect on claims 2 or 3. Caveat: OLMo-2 may also have seen GSM8K train, so a small gap does not prove contamination.

---

## L15b: Sampling-seed replicate of the GSM8K test rollouts (GPU, one job)

**Hypothesis:** the L14a pooled k (118, 18 above the floor) survives a second sampling seed on the same 1,319 test prompts, so the k floor is not a seed artifact.

**Change vs existing scripts:** avrl/exp/L14a.sh with --seed 1 and --out avrl/runs/L15b, G 8 and max_tokens 512 unchanged, through bin/gpu_run.sh. The readout reuses avrl/exp/L14a.py with the test part taken from L15b and the train part (L13a, pid 3000 and above) unchanged. Existing scripts unchanged.

**Metric:** the seed-1 test part's k and rate at cutoff 0.625, and the pooled k (train part 89 plus the seed-1 test k).

**Pass threshold:** seed-1 test k at least 11, so that pooled k reaches 100, and pooled rate at least 0.03.

**Kill threshold:** pooled rate below 0.03 on seed 1. Seed-1 test k below 11 (pooled k below 100 with rate at least 0.03) is INCONCLUSIVE.

**Expected runtime:** about 6 min GPU, under the 45-minute cap; under 2 min CPU.

**Informs:** whether the L14a PASS margin on k is stable under resampling. Medium information. A kill is unlikely given the rate lower bound of 5.30%, but k is the weakest part of the current gate.

---

## L15c: Claim 3 check on the fresh pool (CPU, descriptive)

**Hypothesis:** on the 1,869 confident groups of the L14a pool (118 errors, none from E06), the confident-subset AUC gain of cheap+emb over cheap is below 0.05, as on L13a.

**Change vs existing scripts:** the L14c arms (cheap and cheap+emb, bucket-cluster bootstrap over cell100 buckets) run on the pooled L14a groups, with the test groups' features computed the same way as avrl/runs/L13a/feats.parquet, on CPU. Existing scripts unchanged.

**Metric:** AUC gain on the confident subset with its 95% bucket-cluster CI.

**Pass threshold (claim 3 lead):** gain at least 0.05 with the CI excluding 0, as in L14c. A claim 3 readout would then run as its own pre-registered item.

**Kill threshold:** point gain below 0.02. A gain between 0.02 and 0.05, or a CI that includes 0 at 0.05 or above, is INCONCLUSIVE.

**Expected runtime:** under 10 min CPU, including the test-question embeddings.

**Informs:** whether any claim 3 lead survives on prompts that chose nothing. Medium information. The prior is a kill (L13a descriptive gain -0.0004; E01, E06 and E10 kills), so this mainly replicates the kill on fresh prompts.

---

Proposed from L14b (2026-10-19, kill for claim 2: label-free ts_online at 25% on the L14a pool). Ids use loop iteration 16 because L15a to L15c are already taken.

---

## L16a: Noisier-proxy replay of the L14b readout (CPU, descriptive)

**Hypothesis:** the sign of D(ts_online) at 25% depends on proxy confidence quality. With a weaker majority proxy (4 of 8 rollouts per group), ts_online admits fewer corrupted rewards than the threshold.

**Change vs existing scripts:** copy avrl/exp/l14b_claim2_readout.py. Build a new pool table from avrl/runs/L13a/rollouts.parquet and avrl/runs/L14a/rollouts.parquet: a seeded subsample of 4 of the 8 rollouts per group, with proxy_error (majority over the 4 differs from gold), agreement, n_distinct, mean and std of tokens, and any_truncated recomputed on that subsample. Keep the cell100 and pc0 to pc15 columns unchanged. No GPU.

**Metric:** D(ts_online) at 25% with the primary bucket-cluster 95% interval (seed 123, 200 resamples). The group interval is secondary.

**Pass threshold:** point D > 0 and the primary interval excludes 0.

**Kill threshold:** point D <= 0. Point D > 0 with the interval including 0 is INCONCLUSIVE.

**Expected runtime:** under 10 min, CPU.

**Informs:** whether claim 2 has any proxy regime where a learned allocator wins. It is the only CPU check on the proxy-quality question in PROJECT.md before the GPU G 16 item (L10b). High information. Prior: a kill is likely, since the threshold also loses agreement signal when the proxy is noisier.

---

## L16b: Bucket-oracle ceiling for the cell100 allocator (CPU, descriptive)

**Hypothesis:** even with the true bucket error rates, a bucket-based allocator does not beat the threshold on the L14b pool at 25%.

**Change vs existing scripts:** add one arm to a copy of avrl/exp/l14b_claim2_readout.py. Rank sim-pool groups by the true error rate of their cell100 bucket (labels are used, so this is an upper bound for bucket-level allocators), with the threshold key (agreement ascending, n_distinct descending) inside each bucket. Budgets 10, 25 and 50%. No new features, no GPU.

**Metric:** D(bucket oracle) = admitted(threshold) - admitted(bucket oracle) at 25%, with the primary bucket-cluster interval.

**Pass threshold:** point D > 0 and the interval excludes 0. This would show headroom for bucket-level learners.

**Kill threshold:** point D <= 0. This closes the cell100 bucket design on this pool, not only the learner.

**Expected runtime:** under 5 min, CPU.

**Informs:** whether a failure is in the bucket design or in learning the buckets. Medium-high information. The arithmetic ceiling for any allocator at 25% on this pool is D = +0.085 (perfect labels, admitted 0.434 against the threshold's 0.519), so this item measures how much of that ceiling a bucket design can reach.

---

## L16c: Bucket granularity check for the L14b kill (CPU, descriptive)

**Hypothesis:** the L14b kill holds at cell30 and cell50 bucket granularity.

**Change vs existing scripts:** refit KMeans(30) and KMeans(50) on the same MiniLM question embeddings used for cell100 (n_init 3, random_state 0, fit on the 7,473 L13a train questions in pid order), predict the 1,319 test questions, and check the L13a cell30 column for pid 0 to 7472 before use if that column is stored in avrl/runs/L13a/feats.parquet. Then run avrl/exp/l14b_claim2_readout.py unchanged with cheap+emb at 25%.

**Metric:** D(ts_online) at 25% with the primary bucket-cluster interval, at each granularity.

**Pass threshold:** point D > 0 with the interval excluding 0 at both granularities.

**Kill threshold:** point D <= 0 at both granularities. Mixed signs or intervals including 0 are INCONCLUSIVE.

**Expected runtime:** under 10 min, CPU.

**Informs:** whether the kill depends on bucket granularity, the robustness check that L3a proposed for E03. Medium information.

---

Proposed from L16a (2026-10-19, KILL at the pre-registered point rule: D(ts_online) = -0.075 at 25% on the K = 4 proxy, primary CI [-0.091, -0.060]). Ids use loop iteration 17 because L16a, L16b and L16c are taken. Both are CPU only and each needs its own pre-registration before it runs. L10b (existing, G 16, GPU) is parked and not repeated here.

---

## L17a: Labeled static arm interval on the clean and noisier pools (CPU, descriptive)

**Hypothesis:** the labeled static arm's margin over the label-free threshold at 25% (+0.011 at K = 8, +0.010 at K = 4) has a bucket-cluster interval above 0 on both pools.

**Change vs existing scripts:** copy avrl/exp/l14b_claim2_readout.py and add a bucket-cluster bootstrap for the static arm (the readout bootstraps only ts_online and greedy_online today). Before running, confirm in the code that the static arm uses only fit-quarter labels. Run on avrl/runs/L14b/pool_feats.parquet (K = 8) and avrl/runs/L16a/pool_feats_noisy4.parquet (K = 4), CLUSTER_SEED 123, 200 resamples.

**Metric:** D(static) = admitted(threshold) minus admitted(static) at 25%, with the primary bucket-cluster 95% interval on each pool. The 10% and 50% points are descriptive.

**Pass threshold:** the primary interval lies above 0 on both pools. Claim 2 then reopens for labeled learners only.

**Kill threshold:** point D at or below 0 on either pool. Mixed results are inconclusive.

**Expected runtime:** under 5 min, CPU.

**Informs:** whether the only learned arm that beats the label-free threshold survives a bootstrap. High information for claim 2, since labeled learners are the part of the claim still untested. Prior: the interval includes 0. The E10 ceiling static arm in L11b had D +0.008 with CI [-0.017, +0.035].

---

## L17b: Proxy-quality points at K = 2 and K = 6 (CPU, descriptive)

**Hypothesis:** D(ts_online) at 25% stays at or below 0 at every subsample size tested, so the L16a sign is a plateau across proxy quality and not a feature of K = 4.

**Change vs existing scripts:** run avrl/exp/l16a_noisy_proxy_table.py with --keep 2 and --keep 6 (the K = 8 control must still pass), then run avrl/exp/l14b_claim2_readout.py unchanged on each table with cheap+emb, cell100, budgets 10, 25 and 50%, and the same bootstrap seeds. No GPU.

**Metric:** D(ts_online) at 25% with the primary bucket-cluster interval, for each K. The 10% and 50% points are descriptive.

**Pass threshold:** at some K, point D > 0 with the primary interval excluding 0. That would be the first regime where the label-free learner wins, and it would open a learner-level follow-up.

**Kill threshold:** point D at or below 0 at both K = 2 and K = 6. Point D > 0 with the interval including 0 is inconclusive.

**Expected runtime:** about 3 min per K, CPU.

**Informs:** whether L16a is a plateau or one point on a curve. At K = 2, agreement takes only the values 0.5 and 1.0, so read that point as a stress test. Medium information. Prior: kill at both K.

---

Proposed from L16b (2026-10-19, KILL at the pre-registered point rule: D(bucket_oracle) = -0.155 at 25%, primary CI [-0.182, -0.134]). Ids continue loop iteration 17 because L17a and L17b are taken. Both are CPU only and each needs its own pre-registration before it runs. L17a is the existing item that should run first, since L17d reuses its bootstrap.

---

## L17c: Doc-level labeled score with the cell100 rate (CPU, descriptive)

**Hypothesis:** a labeled doc-level score that combines the cell100 bucket error rate with agreement and n_distinct admits fewer corrupted rewards than the label-free threshold at 25%. The L16b kill covered rate-ranked whole-bucket allocation only.

**Change vs existing scripts:** copy avrl/exp/l16b_bucket_oracle.py and add one arm, score_rate. Fit a logistic regression of proxy_error on agreement, n_distinct and the cell100 bucket rate on the sim pool, then rank groups by fitted probability at 25%. Labels are used in-sample, so the rate includes each doc's own label (a small upward bias at about 57 docs per bucket); this is an upper bound. Keep the L16b bucket-cluster resampler, with rates rebuilt per resample. Run on avrl/runs/L14b/pool_feats.parquet. No GPU.

**Metric:** D(score_rate) = admitted(threshold) minus admitted(score_rate) at 25%, with the primary bucket-cluster 95% interval (CLUSTER_SEED 123, 200 resamples). The 10% and 50% points are descriptive.

**Pass threshold:** point D above 0 and the primary interval excludes 0. This would show that bucket rate adds doc-level information beyond agreement. It would justify a labeled-learner follow-up, but only on a split that holds out buckets within the pool, because the current bucket-level split cannot estimate a sim bucket's rate.

**Kill threshold:** point D at or below 0. Point D above 0 with the interval including 0 is inconclusive.

**Expected runtime:** under 5 min, CPU.

**Informs:** whether the cell100 structure has any doc-level value beyond agreement on this pool, which is the scope the L16b kill leaves open. High information. Prior: at most a small margin, since agreement already carries most of the ordering.

---

## L17d: Labeled static ablation by feature family (CPU, descriptive)

**Hypothesis:** the labeled static arm's margin over the threshold at 25% (+0.011 at K = 8, descriptive) comes from the cheap features, not from the 16 prompt-space PCs.

**Change vs existing scripts:** copy avrl/exp/l14b_claim2_readout.py and refit the static logistic three times on the same fit quarter with the same settings: cheap six only, cheap six plus pc0 to pc15 (the current arm, which must reproduce the L14b value of 938.0 caught at 25%), and pc0 to pc15 only. Bootstrap each with the bucket-cluster resampler from L17a (CLUSTER_SEED 123, 200 resamples). Run on avrl/runs/L14b/pool_feats.parquet. No GPU.

**Metric:** Delta = admitted(cheap only) minus admitted(cheap plus PCs) at 25% and at 50%, each with the primary bucket-cluster 95% interval. Positive Delta means the PCs admit fewer corrupted rewards. The PC-only variant is descriptive.

**Pass threshold:** at 25% or at 50%, Delta above 0 with the primary interval excluding 0. This would be the first labeled evidence for claim 3 and would place the labeled edge in prompt-space structure.

**Kill threshold:** Delta at or below 0 at both budgets, meaning the cheap features carry any labeled edge and the prompt-space PCs add nothing. Any other pattern is inconclusive.

**Expected runtime:** under 5 min, CPU.

**Informs:** claim 3 for the labeled learner, the only place the prompt-space part of the labeled lead can be tested. High information. Prior: the cheap-six model matches the full model, so a kill is likely.

---

Proposed from L17d (2026-10-19, INCONCLUSIVE under the pre-registered rule: 25% point Delta -0.009 with primary interval [-0.017, +0.002]; 50% point +0.002 with interval [-0.008, +0.011]). Ids continue loop iteration 18 because L17a to L17d are taken. L18a and L18c are CPU only. L18b is one GPU job and must go through gpu_run.sh. Each needs its own pre-registration before it runs.

---

## L18a: Fit-quarter sensitivity of the labeled static arms (CPU, descriptive)

**Hypothesis:** the labeled static margin at 25% (D(static cheap+PC) +0.011, cluster lower bound +0.0005) and the cheap-six versus cheap-plus-PC Delta depend on which 25 buckets are held out for the fit. The L17d intervals hold the fit quarter fixed at seed 0.

**Change vs existing scripts:** copy avrl/exp/l17d_static_ablation.py. Repeat the fit and the sim-pool scoring for 20 bucket-shuffle seeds (0 to 19), each with its own held-out quarter, its own sim pool and its own threshold tie-break. Skip the bootstrap in this pass. Run on avrl/runs/L14b/pool_feats.parquet. No GPU.

**Metric:** per fit seed, D(static cheap+PC) at 25% and 50%, and Delta (cheap six minus cheap plus PCs) at 25% and 50%. Report the distribution over seeds, the number of seeds with each sign, and the seed-0 values as a check (they must reproduce L17d: 938 caught at 25%).

**Pass threshold:** D(static cheap+PC) at 25% above 0 on at least 16 of 20 fit seeds. The labeled margin then survives the fit-split source of variance that the L17d interval omits.

**Kill threshold:** D(static cheap+PC) at 25% at or below 0 on at least 10 of 20 fit seeds. Any other count is inconclusive.

**Expected runtime:** a few minutes, CPU (the bootstrap is skipped).

**Informs:** whether the claim 2 labeled lead at 25% is a property of the pool or of one fit split. High information for claim 2. Prior: the 25% margin changes sign on a sizable share of fit quarters, since its point value is only +0.011 and the two cheap arms differ by -0.009 on the seed-0 quarter.

---

## L18b: Rollout-sampling replicate of the labeled ablation (GPU, one job)

**Hypothesis:** the L17d readings on the L14b pool (the cheap six carry the labeled margin at 25%; the PCs add no detectable margin at 25%) hold on a second rollout sample of the same GSM8K prompts with a fresh sampling seed.

**Change vs existing scripts:** first check whether the L15b sampling-seed replicate (GPU, one job) left rollouts and features that can be reused. If not, run one GPU job through /homes/rdubash/avrl_live/CASCADE/bin/gpu_run.sh with the L14b rollout protocol (G 8, same prompts, new sampling seed) using the existing rollout script, not an edited copy of it. Rebuild the pool features as L14b did, then run avrl/exp/l17d_static_ablation.py unchanged. The job must fit in the 45 minute cap; if the cap is hit, record it as failed.

**Metric:** Delta at 25% and 50% with the primary bucket-cluster interval, and D(static cheap+PC) at 25%.

**Pass threshold:** the L17d pass rule applied to the new sample: point Delta above 0 with the primary interval above 0 at 25% or 50%.

**Kill threshold:** the L17d kill rule applied to the new sample: point Delta at or below 0 at both budgets. Any other pattern is inconclusive.

**Expected runtime:** one GPU job within the 45 minute cap, plus about 5 min CPU. Check the logged runtime of L15b before launching.

**Informs:** the only test of the single-rollout-sample caveat, for both claim 2 (the labeled lead) and claim 3 (the PC margin). High information. Prior: no PC margin at 25% again, and the labeled D over the threshold is the more fragile of the two readings.

---

## L18c: Low-dimensional prompt-space check (CPU, descriptive)

**Hypothesis:** a few leading prompt-space PCs add margin to the cheap six that the 16-PC arm dilutes.

**Change vs existing scripts:** copy avrl/exp/l17d_static_ablation.py and add two static arms: cheap six plus pc0 to pc3, and cheap six plus pc0 to pc7, each fit on the same fit quarter. Use the L17d bucket-cluster bootstrap (CLUSTER_SEED 123, 200 resamples). Run on avrl/runs/L14b/pool_feats.parquet. No GPU.

**Metric:** Delta_k = admitted(cheap six) minus admitted(cheap six plus the first k PCs), for k = 4 and k = 8, at 25% and 50%, with the primary bucket-cluster interval.

**Pass threshold:** for k = 4 or k = 8, at 25% or 50%, point Delta_k above 0 with the primary interval above 0.

**Kill threshold:** point Delta_k at or below 0 for both k at both budgets. Any other pattern is inconclusive. Any "match" claim needs an equivalence margin pre-registered first; the current 25% interval [-0.017, +0.002] is too wide to show equivalence at +/-0.005.

**Expected runtime:** about 3 to 5 min, CPU.

**Informs:** whether claim 3 survives in a low-dimensional form, where the 16-PC arm may hide a margin. Low to medium information. Prior: kill, since the 16-PC arm already shows no 25% margin.

---
