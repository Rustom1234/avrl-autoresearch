# AVRL Experiment Backlog

Pre-registered, 2026-09-30. Twenty experiments, E01 through E20, ordered by expected information value per GPU hour: cheapest, highest-leverage checks first, most expensive and most infrastructure-blocked last. This ordering means analysis-only reruns of already-collected data (near-zero marginal GPU cost) come before any new generation run, and new generation runs on this machine come before any GRPO training run on an A100.

## Claim numbering (from PROJECT.md)

- Claim 1: the learned allocator reaches the same policy quality with a fraction of expensive verifier calls.
- Claim 2: the learned allocator admits fewer corrupted rewards into training than any fixed rule at the same budget.
- Claim 3: the allocator's edge comes from structure in prompt space (proxy error clusters), the SOFisher condition (Li et al., 2026, Nature Communications, https://doi.org/10.1038/s41467-026-73404-6).

Source for the existing harness read this session: /homes/rdubash/avrl_live/CHEAP/avrl/PROJECT.md, /homes/rdubash/avrl_live/CHEAP/avrl/p0_rollouts.py, /homes/rdubash/avrl_live/CHEAP/avrl/p0_analyze.py, /homes/rdubash/avrl_live/CHEAP/avrl/allocators.py. Confirmed at read time: avrl/p1_grpo.py (the phase-1 GRPO training harness named in PROJECT.md's Files section) does not exist on disk yet, only p0_rollouts.py, p0_analyze.py, and allocators.py do. Every experiment below that needs an actual policy-training loop is marked NEEDS PHASE-1 HARNESS.

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

All paper citations above are carried forward unchanged from /homes/rdubash/avrl_live/CHEAP/lit/avrl/SYNTHESIS.md, read this session, which itself states its method: built via WebFetch against arXiv abstracts, the OpenAlex works API, and direct publisher DOIs, with WebSearch unavailable (budget error) and not retried. This document adds one new fact verified this session via WebFetch: HuggingFaceH4/MATH-500's existence, size, and content, https://huggingface.co/datasets/HuggingFaceH4/MATH-500. No paper, author, or number is invented in this document; every runtime figure not directly read from avrl/runs/p0/run.log is explicitly marked as an estimate or UNVERIFIED. G01-G04 were added after this session's three gap checks (GAP_already-done.md, GAP_cannot-work.md, GAP_not-a-thesis.md); their citations (VIP, https://arxiv.org/abs/2602.01601) are carried forward unchanged from SYNTHESIS.md and the gap-check files.

## Compute correction (2026-09-30, main session)
This machine's GPU is an NVIDIA RTX A4000 with 16 GB, not a 32 GB card. Measured with the phase-1 harness (Qwen2.5-1.5B, LoRA r=32, bf16): full settings (16 prompts x G=8, 512 tokens) run at 153 s per step with HF generation and peak 13.2 GB; vLLM colocate does not fit beside training on this card (out of memory at engine start). Smoke settings run at 20 s per step. vLLM-only rollout experiments (phase-0 style) are unaffected and fit in 40 minutes at 1,500 to 3,000 prompts.

## Feasibility note (this run)
Items marked A100, needing a larger cluster, a second machine, or the phase-1 GRPO harness (E15 to E20, G04, anything needing p1_grpo.py; about 89 s per step on this card, so 100 steps take 2.5 hours) are not runnable under the 45-minute cap and must not be planned.

## Proposed by the loop

Proposed after E03 (kill for claim 2 on the phase-1 allocator as coded). Ids use iteration 3 to match this run's E03; the files carry no iteration counter, so rename if the loop numbers differently. Entries are CPU-only and reuse E01's feats.parquet.

---

## L3a: Thompson-sampling ablation of the phase-1 learned allocator (analysis-only)

**Hypothesis:** the E03 loss comes from the Thompson-over-cells step, not from the regression. In LearnedAllocator.select (allocators.py lines 140-145) every group in a cell draws from one shared Beta, so ranking inside a cell is random and the cell prior is set from the first group that touched it. A per-group score should recover threshold-level admission.

**Change vs existing scripts:** no new rollouts. Copy avrl/exp/e03_real_allocators.py and add three learned variants at 25%: (a) per-group Beta centered on the regression's p_i, no cell pooling; (b) cell pooling with the cell prior set from the cell's mean regression p rather than the first group's; (c) regression ranking with Thompson noise on the logit scale. Keep E03's fairness checks (identical verifier calls, gold only for update labels). allocators.py is not edited.

**Command shape:** the E03 command with the variant classes, output to runs/L3a, CPU in the local shell.

**Metric:** admitted rate at 25% for each variant minus threshold, paired over seeds 0-7, with the E03 bootstrap CI (B=100, seeds 0-3).

**Pass threshold:** a variant's bootstrap 95% CI upper bound is below 0 at identical calls. If (a) passes and the original E03 learned arm does not, the loss is cell pooling and phase 1 should use per-group sampling.

**Kill threshold:** no variant has a CI upper bound below 0. The regression then adds no ranking information beyond agreement at this pool size (E01 AUC 0.914 agreement alone vs 0.923 with all cheap features), and claim 2 needs a different allocator family rather than a tuned one.

**Expected runtime:** about 15 min CPU with the bootstrap (E03 ran about 0.65 s per replay over 1,300 replays); under 1 min without it.

**Expected information value:** high. It tests the mechanism E03's code reading points to.

**Informs:** claim 2 for the phase-1 allocator; decides whether phase 1 is re-specified before any GRPO spend.

---

## L3b: Scaled regression for the phase-1 learned allocator (analysis-only)

**Hypothesis:** the 640 ConvergenceWarnings in E03 (unscaled LogisticRegression, max_iter=1000) mean the phase-1 regression is poorly fit, so its cell priors and ranking are weak. Standardizing the 22 features inside the regression changes the learned arm's admission rate.

**Change vs existing scripts:** rerun E03 unchanged except that the regression is wrapped in a StandardScaler pipeline inside a copy of LearnedAllocator (no edit to allocators.py). Same seeds, bootstrap and fairness checks.

**Command shape:** the E03 command with the scaled copy of the allocator, output to runs/L3b, CPU in the local shell.

**Metric:** learned minus threshold admitted rate at 25%, bootstrap 95% CI, and the ConvergenceWarning count.

**Pass threshold:** CI upper bound below 0 at identical calls, as in E03.

**Kill threshold:** CI includes 0 or lies above 0 with zero warnings. The E03 kill is then not an artifact of the unscaled fit, and L3a decides between mechanism explanations.

**Expected runtime:** about 15 min CPU.

**Expected information value:** medium-high. It removes the main caveat on E03 as written.

**Informs:** claim 2 for the phase-1 allocator.

---

## L3c: E11 with an extended prior_strength grid (analysis-only)

**Hypothesis:** the first-touch Beta prior (prior_strength x regression p) is where the learned arm fails, so a stronger or weaker prior changes the admission rate more than refit_every does.

**Change vs existing scripts:** E11 as written in this file (refit_every in {5, 20, 50, 100} x prior_strength in {0.5, 2.0, 5.0}, 25% budget, E01 feats.parquet, 8 seeds), with prior_strength 32.0 added. Learned arm only, threshold as the reference.

**Command shape:** the E11 driver instantiating allocators.LearnedAllocator(refit_every=r, prior_strength=p) per grid cell, CPU in the local shell.

**Metric:** learned minus threshold admitted rate at 25% for each cell, with the best cell's paired bootstrap CI.

**Pass threshold:** a grid cell's bootstrap 95% CI upper bound is below 0. That is a tuning result and must be re-pre-registered before any GRPO spend.

**Kill threshold:** no cell passes. The E03 kill is then robust to refit_every and prior_strength.

**Expected runtime:** about 2 min CPU for 128 replays.

**Expected information value:** medium. A flat grid makes the kill robust to hyperparameters at near-zero cost.

**Informs:** claim 2 for the phase-1 allocator, and the E11 step that E03's kill path requires before E15.

---

## L3d: Noise and prior-concentration dose-response for the regression variants (analysis-only)

**Hypothesis:** the gap between the regression-based variants and greedy (no sampling, 321.0 caught) is a noise cost: more ranking noise means fewer caught errors, and a higher Beta concentration recovers part of it.

**Change vs existing scripts:** copy avrl/exp/l3a_thompson_ablation.py and sweep LOGIT_SIGMA in {0, 0.25, 0.5, 1.0, 2.0} for logit_noise, and prior_strength in {2, 8, 32} for pergroup_beta. 25% budget, seeds 0-7, E01 feats.parquet. Main seeds only; a bootstrap only for a setting whose caught mean exceeds 334.25. Sigma 0 is a control: it should reproduce greedy's 321.0, apart from ties created by the 1e-3 clip.

**Command shape:** the L3a command with the sweep flags, output to runs/L3d, CPU in the local shell.

**Metric:** caught mean and admitted rate at 25% per setting, with threshold (334.25) and greedy (321.0) as references.

**Pass threshold:** a setting whose bootstrap 95% CI upper bound on (admitted minus threshold) is below 0 at identical calls. This is tuning of the same family, so it needs re-pre-registration before any GRPO spend.

**Kill threshold:** no setting exceeds the threshold's caught mean of 334.25. The sweep then gives a dose-response only and cannot rescue this family.

**Expected runtime:** about 2 min CPU.

**Expected information value:** medium. It describes the noise dose-response, but it cannot produce a pass on its own, since the best case (sigma 0) equals greedy, which is below threshold.

**Informs:** the noise-versus-regression reading in the L3a entry, and claim 2 for the regression family.

---

## L3e: Noise-cost control, threshold ranking with matched noise (analysis-only)

**Hypothesis:** the noise cost seen in L3a is generic. Adding Gaussian noise to the threshold ranking (agreement, with the n_distinct tie-break) at the same mean within-step rank displacement as logit_noise loses about as many caught errors as logit_noise does (about 32 below threshold).

**Change vs existing scripts:** add a ThresholdNoise arm to a copy of avrl/exp/l3a_thompson_ablation.py. Its score is -agreement plus N(0, s) noise on the agreement scale. Choose s on a pilot (seed 0, 25%) so the mean within-step rank displacement matches logit_noise's, then run seeds 0-7 at 25% with identical 750 calls. No bootstrap needed for the control.

**Command shape:** the L3a command with the ThresholdNoise arm added, output to runs/L3e, CPU in the local shell.

**Metric:** caught mean at 25% for ThresholdNoise, against threshold (334.25) and logit_noise (302.6), with the rank displacement used for matching.

**Pass threshold:** not a pass/kill test. The generic-noise reading is supported if ThresholdNoise loses at least as many caught errors as logit_noise does. The regression-specific reading is supported if it loses little.

**Kill threshold:** not applicable (control).

**Expected runtime:** about 1 min CPU, main seeds only.

**Expected information value:** high. The two readings point to different next steps: a generic noise cost means any learned allocator must rank with little or no sampling, while a regression-specific cost means a different feature family is needed.

**Informs:** claim 2 for the regression family, and the choice between tuning (E11, L3b) and a family change.

---

## L3f: Fresh rollout pool for the allocator comparison (GPU for rollouts, analysis-only after)

**Hypothesis:** the L3a and E03 comparisons hold on a fresh pool of GSM8K train groups, not only on the E01 pool used for every claim-2 run so far.

**Change vs existing scripts:** generate a new 3,000-prompt GSM8K train slice with a different rollout seed, using a copy of avrl/p0_rollouts.py (not edited) and p0_analyze.py to build feats.parquet. Then run the L3a arms, with the E03 learned arm rerun on the new pool rather than taken from E03, with the L3a bootstrap. The 3,000-prompt size is what keeps the rollout inside the 45-minute cap (2026-09-30 compute note).

**Command shape:** the vLLM rollout step through /homes/rdubash/avrl_live/CHEAP/bin/gpu_run.sh <run_dir> '<command>'; then the L3a analysis on the new feats.parquet, CPU in the local shell, output to runs/L3f.

**Metric:** as L3a: admitted rate at 25% for each arm, caught counts, and paired bootstrap CIs (B=100, seeds 0-3), plus the in-pool learned arm.

**Pass threshold:** as L3a. A variant's bootstrap 95% CI upper bound below 0 at identical 750 calls, with the 98.3% Bonferroni bound reported alongside. A pass on a second pool is still a candidate re-specification, pending pre-registration.

**Kill threshold:** as L3a. No variant's upper bound below 0, and the learned arm still above threshold on the new pool. Claim 2 for this family is then killed on two pools.

**Expected runtime:** about 40 min GPU for rollouts, plus about 45 min CPU for the arms.

**Expected information value:** high. Every claim-2 kill so far rests on one pool (E01's 3,000 GSM8K groups), so this is the test of whether the result generalizes. Not run in L3a, since the spec puts it out of scope.

**Informs:** claim 2 for all allocator families, and the phase-1 kill path.

---

Added after L3e (2026-10-18), under the same "Proposed by the loop" heading.

## L3g: Noise dose-response on a shared displacement axis (analysis-only)

**Hypothesis:** the noise cost seen in L3e is generic. At matched mean within-step rank displacement, threshold ranking and regression ranking lose the same number of caught errors. L3e matched only one point, and the match was 11.6 percent high.

**Change vs existing scripts:** copy avrl/exp/l3e_threshold_noise.py and sweep threshold_noise over s in {0, 0.05, 0.10, 0.125, 0.15, 0.20, 0.30}, and logit_noise over sigma in {0, 0.25, 0.5, 1.0, 2.0}, at 25%, seeds 0-7, E01 feats.parquet. Record caught and measured displacement for every setting. No bootstrap.

**Command shape:** the L3e command with the sweep flags, output to runs/L3g, CPU in the local shell.

**Metric:** caught mean at 25% against mean measured displacement, for both families on one plot, with threshold (334.25) and greedy (321.0) as references.

**Pass threshold:** not a pass/kill test (control). Fixed reading before the run: generic if the two curves differ by less than 5 caught at every displacement they share (linear interpolation between sweep points), family-specific otherwise.

**Kill threshold:** not applicable (control). Cannot change any claim verdict.

**Expected runtime:** about 1 min CPU.

**Expected information value:** high. It removes the matching caveat in L3e and tests the generic reading at several noise levels. It supersedes the logit half of L3d.

**Informs:** claim 2 for the regression and threshold families, and whether a deterministic ranker is required.

---

## L3h: Deterministic tie-break inside the threshold order (analysis-only)

**Hypothesis:** the regression carries some within-level signal that threshold ignores. Breaking exact (agreement, n_distinct) ties by the regression's p, with no sampling, beats the threshold at 25%.

**Change vs existing scripts:** a copy of avrl/exp/e03_real_allocators.py (or the L3a script) with a ThresholdAllocator subclass. Order is the threshold key (agreement ascending, then -n_distinct), with exact ties broken by the regression p (higher p first). No noise, no Thompson step, no cells. Cold start (before the first fit) uses the plain threshold order. E03 fairness checks and gold use kept.

**Command shape:** the L3a command with this class, output to runs/L3h, CPU in the local shell.

**Metric:** admitted rate at 25% minus threshold, paired over seeds 0-7, with the L3a paired bootstrap (B = 100, seeds 0-3).

**Pass threshold:** paired bootstrap 95% CI upper bound on (admitted minus threshold) below 0, at identical 750 calls.

**Kill threshold:** CI includes 0 or lies above 0. A CI that collapses to 0 counts as a kill.

**Expected runtime:** about 10 min CPU with the bootstrap (estimate; one learned variant, scaled from L3a's 1979 s for three).

**Expected information value:** medium-high. It is the first learned variant with no sampling in it, so a pass or kill here is not confounded by the noise cost measured in L3e.

**Informs:** claim 2 for the regression family without sampling; decides whether any regression-based allocator is still worth tuning.

---

## L3i: Deterministic regression without the embedding features (analysis-only)

**Hypothesis:** the 16 question-embedding components (pc0 to pc15) are part of why the regression ranking (greedy, 321.0) sits 13.25 below the threshold (334.25). A deterministic regression on the six non-embedding features ranks at threshold level or better.

**Change vs existing scripts:** a copy of the E03 greedy arm with the feature vector reduced to agreement, n_distinct, mean_tokens, std_tokens, any_truncated and log_qlen (pc0 to pc15 dropped). No noise, no cells. Otherwise identical to E03 (refit schedule, cold start, 750 calls). allocators.py is not edited.

**Command shape:** the E03 command with this feature set, output to runs/L3i, CPU in the local shell.

**Metric:** caught at 25% against greedy (321.0, 22 features) and threshold (334.25); paired bootstrap on admitted minus threshold (B = 100, seeds 0-3).

**Pass threshold:** paired bootstrap 95% CI upper bound on (admitted minus threshold) below 0, at identical 750 calls.

**Kill threshold:** CI includes 0 or lies above 0.

**Expected runtime:** about 10 min CPU with the bootstrap (estimate).

**Expected information value:** medium. If the six-feature regression reaches threshold level, the embedding components account for the regression's gap, which bears on claim 3 directly. If it stays below threshold, the regression ranking is worse than agreement with or without the embedding, so the embedding is not the cause.

**Informs:** claim 2 (regression family) and claim 3 (whether prompt-space features help or hurt ranking on this pool).

Added after L3i (2026-10-18), under the same "Proposed by the loop" heading.

## L3j: Fresh GSM8K pool for the regression family (L3f with the L3i arms; GPU for rollouts, analysis-only after)

**Hypothesis:** the kill of the regression family (E03, L3a, L3i) holds on a fresh pool of GSM8K train groups, not only on the E01 pool.

**Change vs existing scripts:** the backlog L3f design, with threshold, greedy6 (six non-embedding features, as in L3i) and the E03 greedy arm (22 features) rerun on the new pool. Rollouts use a copy of avrl/p0_rollouts.py with a different rollout seed on a new 3,000-prompt GSM8K train slice. feats.parquet is built with a copy of p0_analyze.py. Arms are scored with a copy of the L3i script. No file in avrl/ is edited.

**Command shape:** the vLLM rollout step through /homes/rdubash/avrl_live/CHEAP/bin/gpu_run.sh <run_dir> '<command>'; then the L3i analysis on the new feats.parquet, CPU in the local shell, output to runs/L3j.

**Metric:** admitted rate at 25% for threshold, greedy6 and greedy22 on the new pool, caught counts over seeds 0-7, and the paired bootstrap of greedy6 minus threshold (B=100, seeds 0-3).

**Pass threshold:** paired bootstrap 95% CI upper bound on (admitted greedy6 minus admitted threshold) below 0, at identical 750 calls. A pass is a candidate re-specification and needs a further pre-registration before any claim-2 credit.

**Kill threshold:** the CI includes 0 or lies above 0. With the E01 result, that makes the regression family killed on two pools.

**Expected runtime:** about 40 min GPU for rollouts, plus about 5 min CPU for the arms.

**Expected information value:** high. The regression family's kill (E03, L3a, L3i) rests on one pool. This is the test of whether it generalizes.

**Informs:** claim 2 for the regression family, and the phase-1 kill path.

---

## L3k: Offline ranking diagnostic with no sequential loop (analysis-only)

**Hypothesis:** the regression ranking's loss to threshold (L3i, about 12.5 caught) comes from the ranking itself, not from the online loop (verified-only labels, refit cadence, cold start). Also: pc0 to pc15 carry no proxy-error signal on their own on this pool.

**Change vs existing scripts:** a new script in avrl/exp/ (copy of the L3i loader, no edits to allocators.py). Five-fold out-of-fold logistic regression on all 3,000 groups with full labels (StratifiedKFold, shuffle, random_state 0), no sequential selection. For each feature set, select the global top 750 groups by out-of-fold probability and count proxy errors caught. Feature sets: threshold order (agreement ascending, then n_distinct descending, as in ThresholdAllocator), six cheap features, 22 features (six plus pc0 to pc15), and pc0 to pc15 alone. Report out-of-fold AUC as a secondary number.

**Command shape:** /homes/rdubash/active-audit/.venv/bin/python -B avrl/exp/l3k_offline_ranking.py --pool avrl/runs/p0/feats.parquet --out avrl/runs/L3k, CPU in the local shell; runner avrl/exp/L3k.sh with timeout 2700, writing run.log, result.json, exit_code and done.flag.

**Metric:** caught out of 436 proxy errors for each feature set, against the threshold order and against a chance level of about 109 (750 x 436 / 3000).

**Pass threshold:** not applicable (diagnostic). Pre-registered reads: (i) if the six-feature regression catches fewer than the threshold order with full labels, the ranking itself is worse, and L3k does not implicate the online loop. (ii) If it reaches the threshold order, the loss comes from the online loop, and L3l is the next test. (iii) If pc0 to pc15 alone are within about 10 caught of chance (about 109), the embedding components carry no proxy-error signal on this pool, which tests claim 3 directly rather than through the allocator.

**Kill threshold:** not applicable.

**Expected runtime:** about 5 min CPU (estimate).

**Expected information value:** high. It separates a ranking problem from a sequential-learning problem, and it measures claim 3's mechanism without an allocator in the loop.

**Informs:** claim 2 (regression family: ranking versus loop) and claim 3 (prompt-space signal).

---

## L3l: Per-step gap trace and refit-cadence sensitivity for greedy6 (analysis-only)

**Hypothesis:** the greedy6 gap to threshold opens at the first post-cold-start refit and grows, rather than appearing gradually or only in the early steps.

**Change vs existing scripts:** a copy of the L3i script that records caught per step for the threshold and greedy6 arms at 25%, seeds 0-7, and reruns greedy6 with refit_every in {10, 20, 40} (the L3i setting is 20). No edits to allocators.py.

**Command shape:** runner avrl/exp/L3l.sh, CPU in the local shell, output to runs/L3l with run.log, result.json, exit_code and done.flag.

**Metric:** cumulative caught by step for each arm, the step at which greedy6 first falls below threshold on each seed, and caught at 25% for each refit_every.

**Pass threshold:** not applicable (diagnostic). Read: if the gap opens at the first refit and then stays flat, a cadence fix is worth a test. If the gap grows steadily after the first refit at every cadence, the regression ranking itself is the problem.

**Kill threshold:** not applicable.

**Expected runtime:** about 3 min CPU (estimate).

**Expected information value:** medium. It separates a cadence problem from a ranking problem for the regression family, and it bears on whether L3k is needed at all.

**Informs:** claim 2 (regression family), with no claim-3 bearing.

---

Added after L3k (2026-10-18), under the same "Proposed by the loop" heading.

L3k read (ii) passed on the fold-seed mean: the six-feature full-label top 750 caught 362.6, against 354 for the threshold order (see RESEARCH_LOG, L3k). L3l stays the next test as written. L3k's scaled secondary arm (369.5 caught) raises the weight of L3b, which is already listed above. No new ids were added.

## L3m: Unbiased labels inside the loop (analysis-only)

**Hypothesis:** the greedy6 gap to threshold (L3l: about 12.5 caught at r=20, decelerating after the first refit) comes from the regression's training set being made of its own picks, so its ranking drifts. Mixing uniformly sampled verified groups into the training data removes most of the gap.

**Change vs existing scripts:** a copy of avrl/exp/l3l_refit_cadence_trace.py (not edited). In each 16-group step, a fraction e of the 4 verifier calls goes to groups drawn uniformly from the unverified pool. All verified labels train the regression, and greedy6 spends the rest at refit_every 20. e in {0.25, 0.5}. Total calls stay at 750. No edits to allocators.py.

**Command shape:** a copy of avrl/exp/L3l.sh with the new script, output to runs/L3m, CPU in the local shell.

**Metric:** caught mean at 25% over seeds 0-7, paired against threshold (334.25) and against the as-coded r=20 arm (321.75).

**Pass threshold:** caught mean at least 331.25 at some e (within 3 of threshold, about 2 paired se).

**Kill threshold:** caught mean at or below 325 at both e.

**Expected runtime:** about 5 min CPU (estimate).

**Expected information value:** high. It tests the leading loop mechanism directly. A pass says the sampled-label loop is the one L3j should run. A kill says selection bias in the training set is not the cause, and the next step is L3n.

**Informs:** claim 2 (regression family, loop versus ranking). No claim-3 bearing.

---

## L3n: Regression choice restricted to a threshold window (analysis-only)

**Hypothesis:** a learned choice that stays inside the threshold order's best candidates keeps the threshold's caught count, so the learned component can be kept as a bounded refinement.

**Change vs existing scripts:** a copy of avrl/exp/l3l_refit_cadence_trace.py (not edited). Within each 16-group step, the regression may pick its 4 verifies only among the W lowest-agreement groups of that step (threshold key: agreement ascending, then n_distinct descending). W in {8, 12}. greedy6 at refit_every 20 with the six non-embedding features.

**Command shape:** a copy of avrl/exp/L3l.sh with the new script, output to runs/L3n, CPU in the local shell.

**Metric:** caught mean at 25% over seeds 0-7, paired against threshold (334.25).

**Pass threshold:** caught mean at least 334 at either W.

**Kill threshold:** caught mean at or below 325 at both W.

**Expected runtime:** about 5 min CPU (estimate).

**Expected information value:** medium-high. It asks whether any bounded learned reordering helps near the threshold cut. A pass gives a threshold-plus-learned recipe to report. A kill says the regression reorders badly even locally, which fits a ranking-level problem.

**Informs:** claim 2 (learned allocator versus the fixed threshold recipe that PROJECT.md names as the fallback deliverable).

---

## L3o: Seed power for the cadence contrast (analysis-only, optional)

**Hypothesis:** the r=40 lead over r=20 (L3l: +1.9 caught, paired se 1.2, n=8) is noise, and no refit_every beats r=20 by a margin that matters.

**Change vs existing scripts:** the L3l script with 32 seeds for threshold and greedy6 at refit_every 10, 20 and 40. Replay only.

**Command shape:** a copy of avrl/exp/L3l.sh with 32 seeds, output to runs/L3o, CPU in the local shell.

**Metric:** paired difference r=40 minus r=20 over 32 seeds, with a t interval; the same for r=10 minus r=20.

**Pass threshold:** not applicable (diagnostic). The r=40 minus r=20 interval excludes 0.

**Kill threshold:** the interval includes 0.

**Expected runtime:** about 2 min CPU (estimate).

**Expected information value:** low. Cadence is already not the fix (every cadence ends about 11 below threshold). Run this only if a later cadence question needs a tighter estimate.

**Informs:** claim 2 (regression family). No claim-3 bearing.

---

Added after L3l (2026-10-18), under the same "Proposed by the loop" heading.

L3l verdict: inconclusive (see the L3l verdict entry in RESEARCH_LOG, which relabels the run's Read B label). Cadence is not the fix: caught means 321.6 to 323.6 at r=10 to 40, all about 11 below threshold. L3j stays as written (refit_every 20) unless L3m or L3n gives a reason to change the loop first. New ids: L3m, L3n and L3o.

## L3p: Scaled greedy loop with no Thompson step (analysis-only)

**Hypothesis:** the feature scaling that L3k showed helps full-label ranking (369.5 vs 362.6 for the six features) also helps the online greedy loop, now that the Thompson step is removed. L3b showed scaling does nothing inside the Thompson loop, which leaves the greedy loop as the untested case.

**Change vs existing scripts:** a copy of avrl/exp/l3i_no_embedding_regression.py (the greedy6 arm of L3i, not edited) with the regression wrapped in a StandardScaler pipeline, the six non-embedding features, refit_every 20, cold start on the threshold order. No edits to allocators.py.

**Command shape:** a copy of avrl/exp/L3b.sh pointed at the new script, output to runs/L3p, CPU in the local shell.

**Metric:** admitted rate at 25% for greedy_scaled minus threshold, paired over seeds 0-7 at 750 calls, with a paired bootstrap 95% CI (B=100, seeds 0-3). Also the caught mean and the ConvergenceWarning counts.

**Pass threshold:** bootstrap CI upper bound below 0 at identical 750 calls.

**Kill threshold:** the CI includes 0 or lies above 0.

**Expected runtime:** about 6 min CPU (estimate).

**Expected information value:** medium. It closes the scaling question for the loop that L3i and L3k place closest to the threshold. Expected outcome is a kill, since unscaled greedy6 is 12.5 caught below the per-step threshold.

**Informs:** claim 2 (regression family, online loop without Thompson). No claim-3 bearing.

---

## L3q: L3m redesigned with a matched-budget control (analysis-only)

**Hypothesis:** the loop's gap to threshold comes partly from the regression's training set being made of its own picks. Mixing uniformly drawn verified groups into training removes some of the gap, measured against a control that spends the same budget on the same uniform draws.

**Change vs existing scripts:** a copy of avrl/exp/l3l_refit_cadence_trace.py, the base named in the L3m entry (not edited). Adds a mixed-threshold arm: in each 16-group step, a fraction e of the 4 verifier calls goes to uniform draws and the rest to the threshold order. greedy6 with label mixing uses the same e. The pass comparison is greedy6 with mixing against this mixed-threshold arm, not against pure threshold. Total calls stay at 750. No edits to allocators.py.

**Command shape:** a copy of avrl/exp/L3l.sh with the new script, output to runs/L3q, CPU in the local shell.

**Metric:** paired caught difference at 25%, greedy6 with mixing minus the mixed-threshold arm at the same e, over seeds 0-7, with a paired bootstrap CI (B=100, seeds 0-3). Also the caught means of all arms.

**Pass threshold:** at some e in {0.25, 0.5}, the bootstrap CI lower bound for greedy6-with-mixing minus mixed-threshold caught is above 0 (the mixing helps relative to the matched control).

**Kill threshold:** at both e, the CI includes 0 or lies below 0.

**Expected runtime:** about 6 min CPU (estimate).

**Expected information value:** high. It tests the leading loop mechanism without the budget confound in L3m's kill threshold. A pass says the sampled-label loop is what L3j should run. A kill says selection bias in the training set is not the cause, and the next step is the ranking-level test in L3n.

**Informs:** claim 2 (regression family, loop versus ranking). No claim-3 bearing.

---

## L3r: L3j on a fresh GSM8K pool with the scaled arm added (GPU for rollouts, analysis-only after)

**Hypothesis:** the regression-family result on the E01 pool (L3k's marginal full-label pass, the L3i and L3b kills) holds on a second pool, and the scaled greedy loop from L3p behaves the same way on that pool.

**Change vs existing scripts:** the L3j plan (a new 3,000-prompt GSM8K slice with a different rollout seed through p0_rollouts.py and p0_analyze.py), with the L3i arms plus the L3p scaled arm. If run, this replaces L3j rather than adding to it. No edits to allocators.py or the p0 scripts.

**Command shape:** the rollout step through gpu_run.sh (about 40 min, 45-minute cap), then the analysis script on runs/L3r, CPU in the local shell.

**Metric:** admitted rate at 25% for each regression arm minus threshold, paired over seeds 0-7 at 750 calls, with the paired bootstrap CI (B=100, seeds 0-3). Also the full-label ranking read from L3k on the new pool.

**Pass threshold:** a regression arm with a bootstrap CI upper bound below 0 on this pool, and the L3k full-label pass replicating on the new pool.

**Kill threshold:** no regression arm passes, and the full-label pass does not replicate.

**Expected runtime:** about 40 min GPU for rollouts, plus about 45 min CPU (estimate).

**Expected information value:** high, but only after L3p and L3q have run, since their results decide which arms L3r needs. L3k's full-label pass has a pool bootstrap interval of [-1.0, +16.0] on this pool, and claim 2 has been decided on one pool.

**Informs:** claim 2 (regression family, second pool). Claim 3 only if the pc0 to pc15 arms are kept in the new run.

---

Added after L3b (2026-10-18), under the same "Proposed by the loop" heading.

L3b verdict: kill (see the L3b entry in RESEARCH_LOG). Scaling the regression does not change the learned arm at 25%: the bootstrap CI for (learned_scaled minus threshold) admitted is [+0.417, +0.488], and the unscaled control reproduces E03 exactly. The E03 kill is not an artifact of the unscaled fit. New ids: L3p, L3q and L3r.

---

Added after L3q (2026-10-18).

L3q verdict: kill (see the L3q entry in RESEARCH_LOG). With 25% or 50% of the 750 calls spent on uniform draws, greedy6 with label mixing sits 9.4 caught below its matched mixed-threshold control at both shares (bootstrap 95% CIs [-14.51, -4.37] and [-12.50, -2.37]). The matched control spends the same uniform calls, so the budget arithmetic is not the cause. Selection bias in the regression's training set does not explain the loop gap at these shares on this pool. Next test: L3n (threshold-window restriction). L3r still waits on L3p, which has no entry yet.

Added after L3p (2026-10-18), under the same "Proposed by the loop" heading.

L3p verdict: pass under the pre-registered rule (see the L3p entry in RESEARCH_LOG). The scaled greedy loop admits 0.0178 fewer corrupted rewards than threshold at 25% (seeds 0-7), and the primary percentile bootstrap upper bound is -0.0050. The pass is fragile: the basic interval contains 0 and the bootstrap mean is shifted about 0.011. Not credited to claim 2. New ids: L3s, L3t and L3u. L3r stays gated and must carry the scaled arm.

## L3s: Bootstrap centring check for the L3p pass (analysis-only)

**Hypothesis:** the L3p pass depends on the percentile interval method. The bootstrap mean of (greedy6s minus threshold) admitted sits 0.011 below the seeds 0-3 point estimate, mostly from the threshold arm, so the percentile upper bound is not a clean test of the observed difference.

**Change vs existing scripts:** a copy of avrl/exp/l3p_scaled_greedy_loop.py (not edited). The bootstrap is replaced by without-replacement subsamples of 1,500 of the 3,000 groups, B=200, seeds 0-3, default_rng(10000+b). Each subsample uses its own error count as the denominator. Report the percentile CI, the basic (reverse-percentile) CI and the bias. The main seeds 0-7 run is unchanged.

**Command shape:** a copy of avrl/exp/L3p.sh pointed at the new script, output to runs/L3s, CPU in the local shell.

**Metric:** greedy6s minus threshold admitted at 25%, paired over seeds 0-3, bootstrap bias (mean of resamples minus the observed seeds 0-3 point estimate), percentile and basic CIs.

**Pass threshold:** both the percentile and the basic upper bounds are below 0, and the absolute bias is below 0.003.

**Kill threshold:** either upper bound is 0 or above. The L3p pass is then not robust to the interval method, and L3r is expected to fail.

**Expected runtime:** about 2 min CPU.

**Expected information value:** high. It tests the one weak point of the L3p pass before any GPU time is spent on L3r.

## L3t: Scaled greedy loop with the embedding components (analysis-only)

**Hypothesis:** with the scaling that L3p shows matters, the 16 question-embedding components (pc0 to pc15) add proxy-error signal to the online loop. Unscaled, they added 0.75 caught (E03 greedy22 at 321.0 against L3i greedy6 at 321.75).

**Change vs existing scripts:** a copy of avrl/exp/l3p_scaled_greedy_loop.py with FEAT_COLS extended to the 22 E03 features (six plus pc0 to pc15). Arms: threshold (gated to E03), greedy22u (unscaled control, gated to the E03 greedy22 per-seed caught 331, 325, 322, 327, 317, 319, 311, 316), and greedy22s (scaled, tested). The L3p bootstrap (B=100, seeds 0-3) is kept. No edits to allocators.py.

**Command shape:** a copy of avrl/exp/L3p.sh pointed at the new script, output to runs/L3t, CPU in the local shell.

**Metric:** greedy22s minus greedy6s admitted at 25%, paired over seeds 0-7, with the paired bootstrap CI (B=100, seeds 0-3).

**Pass threshold:** greedy22s beats greedy6s with bootstrap upper bound below 0, and both controls reproduce. This would give claim 3 loop-level support.

**Kill threshold:** upper bound 0 or above. Claim 3 stays not supported in the loop.

**Expected runtime:** about 2 min CPU.

**Expected information value:** medium-high. Claim 3 rests on E01's embedding AUC (0.606 alone, 0.922 with cheap features against 0.923 without) and on the unscaled comparison. The scaled embeddings in the loop were never tested.

## L3u: L2 penalty versus feature scaling (analysis-only)

**Hypothesis:** the L3p gain comes from the default L2 penalty acting differently on small-scale features (agreement, log_qlen) than on mean_tokens, not from standardization as such. A weak-penalty unscaled fit should then match the scaled fit.

**Change vs existing scripts:** a copy of avrl/exp/l3p_scaled_greedy_loop.py (not edited). Arms: greedy6u at C in {0.01, 0.1, 10, 100} (unscaled; C=1 is the L3i control and stays gated to L3i per seed) and greedy6s at C in {0.1, 10} (scaled). Threshold and the L3p checks are unchanged. Seeds 0-7 only, no bootstrap.

**Command shape:** a copy of avrl/exp/L3p.sh pointed at the new script, output to runs/L3u, CPU in the local shell.

**Metric:** caught mean and paired admitted difference against threshold, seeds 0-7, for each C and scaling choice.

**Pass threshold:** not applicable (diagnostic). If unscaled C=100 lands within 2 caught of greedy6s, the mechanism is the penalty and L3r should set C. If it does not, scaling itself is needed and L3r keeps the StandardScaler step.

**Kill threshold:** not applicable.

**Expected runtime:** about 3 min CPU.

**Expected information value:** medium. It sets which knob L3r and any production loop must choose. It does not change the L3p verdict or claim 2.

Added after L3s (2026-10-18), under the same "Proposed by the loop" heading.

L3s verdict: kill (see the L3s entry in RESEARCH_LOG). Without-replacement half-pool subsamples give a percentile upper bound of +0.0023 and a basic upper bound of +0.0080, so the L3p pass does not survive. The bias (-0.0032) is just outside tolerance, and the centring shift comes from both arms moving up on the half pool, not from duplicated groups. L3r stays gated with no passing candidate. L3t and L3u wait on L3v. New ids: L3v, L3w and L3x.

## L3v: Null calibration of the subsample bootstrap (analysis-only)

**Hypothesis:** the L3s design (half pool, B=200, seeds 0-3) is calibrated. On data with no label signal its bias is near zero and its upper bounds rarely fall below 0.

**Change vs existing scripts:** a copy of avrl/exp/l3s_subsample_centring.py (not edited). Null pools are built by permuting the rows of X, agreement and n_distinct together, relative to proxy_error, so no arm can see the labels. About 20 null pools, each run with the full L3s design.

**Command shape:** a copy of avrl/exp/L3s.sh pointed at the new script, output to runs/L3v, CPU in the local shell.

**Metric:** per null pool, the paired bias, the per-arm subsample level shifts, the percentile and basic upper bounds, and the rule outcome. Aggregate the PASS and KILL rates.

**Pass threshold:** diagnostic. The design is calibrated if the mean null bias is below 0.003 in absolute value and the null PASS rate is at most 0.025.

**Kill threshold:** not applicable. If the null bias is systematically negative by 0.003 or more, the bias criterion is a design artifact and must be dropped before any further L3 bootstrap verdict.

**Expected runtime:** about 12 min CPU.

**Expected information value:** high. Every verdict on this design, including the L3s KILL, depends on whether the design is calibrated.

## L3w: Second pool for the scaled arm (GPU for rollouts, analysis-only after)

**Hypothesis:** the scaled greedy advantage over threshold, if real, replicates on a fresh GSM8K pool.

**Change vs existing scripts:** L3r re-specified after L3s and L3v, with the arm set threshold and greedy6s on the L3s design. Only one feature pool exists (runs/p0), so a second pool needs fresh rollouts through gpu_run.sh.

**Command shape:** gpu_run.sh for the rollouts (about 40 GPU min), then a copy of avrl/exp/L3s.sh on the new pool, CPU in the local shell.

**Metric:** the paired admitted difference at 25% on the new pool, with the L3s percentile and basic upper bounds and the L3v null-calibrated bias.

**Pass threshold:** both upper bounds below 0 on the new pool, with a calibrated design (L3v). This would make claim 2 testable in the form it was stated.

**Kill threshold:** either upper bound at or above 0, or L3v shows the design is not calibrated.

**Expected runtime:** about 40 GPU min plus CPU. Do not launch until L3v is read.

**Expected information value:** high for claim 2 if run. The expected pass probability is low given L3s.

## L3x: Penalty versus scaling (analysis-only, reframed L3u)

**Hypothesis:** any scaled-arm gain comes from the default L2 penalty acting on small-scale features, not from standardization as such.

**Change vs existing scripts:** as L3u (C sweep of unscaled greedy6 and C in {0.1, 10} for scaled greedy6, seeds 0-7). Its value is lower now, because the gain it would explain is no longer established.

**Command shape:** as L3u, CPU in the local shell.

**Metric:** as L3u, caught mean and paired admitted difference against threshold.

**Pass threshold:** not applicable (diagnostic). Run only after L3v.

**Kill threshold:** not applicable.

**Expected runtime:** about 3 min CPU.

**Expected information value:** low to medium. It would set the knob for a re-specified arm but does not change claim 2 or claim 3.

## L3y: Re-score of the L3v and L3s draws under alternative bias references (analysis-only)

**Hypothesis:** under the L3s rule, the bias gate (abs(bias) < 0.003 against a 4-seed point) is the binding constraint on any PASS, not the upper bounds. The 0.003 tolerance sits below the noise of the 4-seed point (SE about 0.006 on the real pool).

**Change vs existing scripts:** none to the pipeline. A new analysis script reads runs/L3v/null_k/result.json (k = 0..19) and runs/L3s/result.json, and re-scores the stored bootstrap draws (greedy6s_minus_threshold_admitted_per_resample) and the per-seed paired differences under three alternative references: the 8-seed main mean in place of the 4-seed point, the bias gate removed, and the mean bias over pools. It also reports the per-seed SE needed for a 0.003 tolerance.

**Command shape:** CPU in the local shell with the active-audit .venv python. No GPU, no new bootstrap, no new arm runs.

**Metric:** for each reference, the number of the 20 null pools that would return PASS, INCONCLUSIVE and KILL, and the real L3s verdict under that reference. Also the bootstrap SD of the threshold and greedy6s arms, computed separately on the null pools and the real pool.

**Pass threshold:** diagnostic only, not a claim test. Informative if some reference gives a null PASS share at or below 0.025 with a bias bin that resolves.

**Kill threshold:** not applicable. Any reference chosen here must be pre-registered before any L3w read. A reference picked after seeing L3s is not creditable.

**Expected runtime:** under 1 min CPU.

**Expected information value:** high. It decides the gate that every L3 PASS depends on (L3w, L3t, L3u, L3x), at no compute cost.

## L3z: Planted-effect power check of the L3s gate (analysis-only, CPU)

**Hypothesis:** the L3s gate has little power at this pool size. Even when greedy6s truly admits fewer errors than threshold, PASS is rare, so a second-pool PASS would need a larger real effect than the L3p and L3s points suggest.

**Change vs existing scripts:** a copy of avrl/exp/l3v_null_calibration.py (not edited). Pools are built with a planted, known feature-label association: label information is injected into the six features at a strength set by a pilot, so that the expected greedy6s minus threshold admitted difference at 25% is about -0.013, -0.02 and -0.03. The rest of the L3s design is unchanged.

**Command shape:** a copy of avrl/exp/L3v.sh, one run per planted level (20 pools each, under the 45-min cap), CPU in the local shell. No GPU.

**Metric:** per planted level, the share of pools returning PASS, INCONCLUSIVE (bias gate) and KILL, and the share where both upper bounds are below 0 regardless of the bias check.

**Pass threshold:** the gate is usable for L3w if the PASS share at the -0.03 level is at least 0.5.

**Kill threshold:** the PASS share at -0.03 is below 0.5. Then L3w does not launch under the current gate, and the loop must re-specify the gate (pre-registered) or change the test before spending GPU time.

**Expected runtime:** about 13 min CPU per level unloaded (about 50 min per 20 pools under the load seen in L3v, so keep each level in its own run).

**Expected information value:** high. It decides whether the 40 GPU min for L3w can credit claim 2 under this gate at all.

Added after L3h (2026-10-19), under the same "Proposed by the loop" heading.

L3h verdict: kill (see the L3h entry in RESEARCH_LOG). The sampling-free tie-break inside the threshold order does not beat threshold on the E01 pool: design A U_A is +0.0027, and the design B percentile and basic upper bounds are +0.0059 and +0.0036, so the tie-break family is closed on this pool. Claim 2 and claim 3 stay not supported. The open question is whether greedy6s's gain (L3s point -0.0126, seeds 0-7 -0.0178) sits within (agreement, n_distinct) blocks or across them. New ids: L3aa, L3ab and L3ac (letters a to z are used by L3a to L3z, so these continue the L3 series).

## L3aa: Oracle ceiling for within-block ordering (analysis-only, diagnostic)

**Hypothesis:** the threshold's (agreement, n_distinct) blocks leave little headroom for any within-block ordering on this pool. An oracle that orders within each block by the gold proxy_error bounds what any within-block ranker can gain.

**Change vs existing scripts:** a copy of avrl/exp/l3h_tiebreak_threshold.py (not edited). In select(), replace p with the gold proxy_error inside each block. Gold is used only as this ceiling, never as an allocator input. Designs A and B and the verdict rule are unchanged.

**Command shape:** as L3h (avrl/exp/L3h.sh pattern), CPU in the local shell. No GPU.

**Metric:** paired admitted diff (oracle minus threshold) at 25%, seeds 0-7 at 750 calls, plus the design A U_A.

**Pass threshold:** not applicable (diagnostic).

**Kill threshold:** not applicable. Reading rule, fixed before the run: if the seeds 0-7 paired diff is above -0.005, within-block ordering is capped on this pool whatever the ranker, and the tie-break family is closed for good. If it is below -0.005, the loss sits in the ranker, and L3ab decides which part.

**Expected runtime:** about 1 min CPU.

**Expected information value:** high. It decides whether any within-block ranker is worth designing before more GPU spend.

## L3ab: Within-block discrimination of the regression (analysis-only)

**Hypothesis:** the six-feature regression's p does not rank corrupted groups within (agreement, n_distinct) blocks on this pool better than chance. That would explain the L3h null without any sampling issue.

**Change vs existing scripts:** a new analysis script (no pipeline change). Out-of-fold p from five folds on the full pool, using the same six features and Pipeline(StandardScaler, LogisticRegression(max_iter=1000)). Compute the pooled within-block AUC of p against proxy_error over blocks containing both labels, weighted by block size, with a bootstrap CI over blocks. Also compute the same AUC restricted to the 4th and 5th groups of each 16-group step (the cut).

**Command shape:** CPU in the local shell with the active-audit .venv python. No GPU.

**Metric:** pooled within-block AUC and its CI, the all-group AUC for reference, and the share of blocks with mixed labels.

**Pass threshold:** not applicable (diagnostic).

**Kill threshold:** not applicable. Reading rule, fixed before the run: if the within-block AUC CI includes 0.5, the L3h null is explained by the ranker, and no within-block design is worth building.

**Expected runtime:** under 1 min CPU.

**Expected information value:** medium-high. It explains the L3h null before any GPU spend that L3w or L3r would need.

## L3ac: Larger-B design A for the tie-break arm (analysis-only, conditional)

**Hypothesis:** the L3h design A bound (U_A = +0.0027 at B = 100) stays above zero at larger B.

**Change vs existing scripts:** as avrl/exp/l3h_tiebreak_threshold.py with B = 1000 for design A. Design B is not rerun.

**Command shape:** CPU in the local shell, about 6 min. Run only if L3aa shows headroom (diff below -0.005).

**Metric:** U_A at B = 1000, and the share of resamples with an upper bound below 0.

**Pass threshold:** diagnostic. Only a U_A below 0 would change the L3h verdict, and that would need a pre-registered re-read.

**Kill threshold:** not applicable.

**Expected runtime:** about 6 min CPU.

**Expected information value:** low, unless L3aa shows headroom. The sign of U_A is already several resample spreads above zero (about 0.0007 spread across resamples), so this changes the verdict only if the true effect is near zero.

Added after L3aa (2026-10-19), under the same "Proposed by the loop" heading.

L3aa verdict: inconclusive (diagnostic, reading HEADROOM; see the L3aa entry in RESEARCH_LOG). The oracle within-block ceiling at 25%, seeds 0-7, is -0.0192 (control passes; design A U_A -0.0140; design B percentile upper bound -0.0104). The pre-registered hypothesis (at most 0.005) is refuted on this pool. Claim 2 and claim 3 stay not supported. The within-block question is open again. L3ab (above) is next by the fixed rule and is not repeated here. L3ac's condition (headroom) is now met, but its value is low, so it waits behind L3ab. New ids: L3ad and L3ae.

## L3ad: Full-step gold ceiling and across-block headroom (analysis-only)

**Hypothesis:** the full-step gold ceiling is at or below -0.0242 at 25% on this pool, so block choice carries at least 0.005 of headroom beyond the within-block ceiling of -0.0192.

**Change vs existing scripts:** a copy of avrl/exp/l3aa_oracle_ceiling.py (not edited). The arm sorts each 16-group step by gold proxy_error descending with no block key. Same seeds, permutations, 750 calls and threshold control.

**Command shape:** CPU in the local shell with the active-audit .venv python, runner as avrl/exp/L3aa.sh. No GPU.

**Metric:** paired admitted diff (full-step oracle minus threshold) at 25%, seeds 0-7. Across-block headroom is that diff minus -0.0192.

**Pass threshold:** not applicable (diagnostic). A full-step ceiling at or below -0.0242 means block choice has room on this pool.

**Kill threshold:** not applicable.

**Expected runtime:** about 1 min CPU.

**Expected information value:** high. It bounds any step-level selector, not only threshold-style ones. If block choice has room, a block-level re-specification is worth designing. If not, the within-block questions (L3ab, L3ae) are the whole remaining step-level problem.

## L3ae: Per-seed split of greedy6s's gain into within-block and block-crossing pairs (analysis-only)

**Hypothesis:** on seeds 3, 4, 5 and 7, where greedy6s's caught count exceeds the within-block oracle, most of greedy6s's caught gain comes from block-crossing pairs.

**Change vs existing scripts:** a new analysis script that reuses the L3p greedy6s arm unchanged (no gold in the arm). In each step, the groups greedy6s swaps in are paired with the groups it swaps out relative to threshold's top-4. A pair is within-block when both groups share (agreement, n_distinct), and block-crossing otherwise. The matching rule (within-block pairs first) is fixed before the run.

**Command shape:** CPU in the local shell, about 2 min. No GPU.

**Metric:** per seed, the caught difference from within-block pairs and from block-crossing pairs, and the block-crossing share of greedy6s's total caught difference.

**Pass threshold:** not applicable (diagnostic). A block-crossing share above one half on seeds 3, 4, 5 and 7 supports the block-change reading.

**Kill threshold:** not applicable.

**Expected runtime:** about 2 min CPU.

**Expected information value:** high. It separates the two sources of greedy6s's gain, which decides whether the next design is a block-level re-specification or a within-block ranker.

Added after L3ab (2026-10-19), under the same "Proposed by the loop" heading. L3ae was already listed above as the rule-fixed next test. The new ids are L3af and L3ag.

## L3af: Power check for the L3ab within-block read (analysis-only, diagnostic)

**Hypothesis:** on the same 18 mixed blocks with their real sizes and label counts, the L3ab block-bootstrap CI excludes 0.5 with at least 80% power only for a within-block AUC of about 0.58 or higher. If so, the L3ab KILL bounds the within-block signal, rather than showing chance.

**Change vs existing scripts:** a copy of avrl/exp/l3ab_within_block_auc.py (not edited). For each planted level (0.53, 0.55, 0.58, 0.60), simulate scores within each real mixed block as two normal distributions whose separation is set so the expected within-block AUC matches the level, with a fixed seed chosen before the run. Run the same block bootstrap (B=2000, default_rng(20000)) on each of 200 replicates per level.

**Command shape:** CPU in the local shell with the active-audit .venv python, runner as avrl/exp/L3ab.sh. No GPU.

**Metric:** per planted level, the share of replicates whose 95% CI excludes 0.5, and the smallest level with at least 80% share.

**Pass threshold:** not applicable (diagnostic).

**Kill threshold:** not applicable.

**Expected runtime:** about 2 min CPU.

**Expected information value:** high. It turns the L3ab KILL into a bound on the size of any within-block signal this design could detect, and says whether the within-block route is closed or only not shown.

## L3ag: Single-model check of the L3ab within-block AUC (analysis-only, diagnostic)

**Hypothesis:** the pooled out-of-fold within-block AUC (0.5280) is not an artifact of pooling five fold models. One logistic fit on all 3000 groups, six parameters so in-sample optimism is small, gives a within-block AUC within 0.01 of the pooled value.

**Change vs existing scripts:** a copy of avrl/exp/l3ab_within_block_auc.py (not edited). Replace the out-of-fold score with the in-sample score of one Pipeline(StandardScaler, LogisticRegression(max_iter=1000)) fitted on all groups. Same blocks, same bootstrap (B=2000, default_rng(20000)), same cut.

**Command shape:** CPU in the local shell, runner as avrl/exp/L3ab.sh. No GPU.

**Metric:** full-pool and cut within-block AUC with CI, against the L3ab values (0.5280 and 0.5689).

**Pass threshold:** not applicable (diagnostic).

**Kill threshold:** not applicable. If the single-model AUC differs from 0.5280 by more than 0.01, the L3ab point estimate is scale-sensitive. Report that as a caveat on the L3ab read; the pre-registered KILL verdict stands.

**Expected runtime:** under 1 min CPU.

**Expected information value:** low-medium. It rules out a scale artifact in the point estimate. This matters only if the point estimate, rather than the CI, carries the reading.

Added after L3z (2026-10-19), under the same "Proposed by the loop" heading. The L3z run spec numbers this queue as iteration 15, so the new ids are L15a to L15c; earlier entries carry the L3 prefix.

## L15a: Fine-grid pilot and re-run of the L3z power check on fresh pools (CPU pilot, one GPU run)

**Hypothesis:** the L3z miss came from a 2-pool pilot that undersamples pool-to-pool spread (eval SD 0.0209) on a steep, non-monotone strength curve. A 16-pool pilot on a 0.02 grid lands the planted mean closer to -0.030, and the L3s gate then gives a power reading at that effect.

**Change vs existing scripts:** a copy of avrl/exp/l3z_planted_power.py (not edited). Pilot pool seeds 30002 to 30017 (disjoint from 30000 and 30001). Pilot grid s in {0.30, 0.32, ..., 0.50}. Choose s* as the grid value whose 16-pool pilot mean is closest to -0.030, and pre-register the tolerance before the pilot (at least 0.008, because a 16-pool pilot mean has SE near 0.005). Evaluation on 20 new pool seeds 50000 to 50019, disjoint from 20000, 30000 and 40000. The L3s design is unchanged.

**Command shape:** the pilot in the local shell with the active-audit .venv python (about 3 min CPU). The evaluation through bin/gpu_run.sh with runs/L15a as run_dir (about 17 min wall, one run).

**Metric:** PASS, bounds-only and INCONCLUSIVE counts out of 20, mean bias with SD, and the achieved A over seeds 0-7 against the pre-registered tolerance.

**Pass threshold:** the L3z reading rules R2 to R4 apply on the achieved effect, with the tolerance fixed before the pilot.

**Kill threshold:** the pilot cannot reach the tolerance (record INFEASIBLE_PLANT and launch nothing).

**Expected runtime:** about 3 min CPU plus about 17 min GPU wall.

**Expected information value:** high. It is the only route to a power reading at -0.030 for the L3s gate, which decides whether any future L3 verdict on the E01 pool carries weight, and whether L3w can be funded at all.

## L15b: Bias-source check on the existing 20 planted draws (analysis-only, descriptive)

**Hypothesis:** the positive bias in the L3z planted pools (mean +0.0078, positive in 16 of 20 pools) comes from the half-pool subsample (1500 of 3000 groups without replacement) rather than from the seeds 0-3 point estimate, so a reference built on the same subsample size removes most of it.

**Change vs existing scripts:** a new CPU script under avrl/exp that reads the stored arrays in avrl/runs/L3z/planted_k/result.json (threshold_subsample_mean_seeds_0_to_3 and greedy6s_minus_threshold_admitted_per_resample) and decomposes each pool's bias into a subsample-size term and a seed-variance term. No new draws. No edits to the L3s code.

**Command shape:** the active-audit .venv python in the local shell, with -B. No GPU.

**Metric:** per-pool bias split into its two terms, the correlation of the split with the planted effect, and the count of pools with |bias| below 0.003 under each reference.

**Pass threshold:** not applicable (descriptive). Its output sets the reference that a pre-registered L3y must use.

**Kill threshold:** not applicable. The L3z and L3v draws have now been read, so this check cannot produce a verdict. Any verdict needs fresh pools (L15a) scored under a reference pre-registered before its read.

**Expected runtime:** about 1 min CPU.

**Expected information value:** medium-high. It decides whether the next step is L3y (the R3 route, a new bias reference) or a change to the bounds, and it tests the leading descriptive pattern in the L3z entry (11 of 12 bounds-only pools failed the bias check).

## L15c: Owed L3v calibration re-run through bin/gpu_run.sh (analysis-only, compute-rule fix)

**Hypothesis:** the L3v null calibration (point-estimate bias 0.00216, 0 of 20 PASS) reproduces when the run goes through bin/gpu_run.sh under the helper's rules, so the L3s gate's null behaviour is established before any L3w credit.

**Change vs existing scripts:** none. Run the existing avrl/exp/L3v.sh unchanged through the helper, with a fresh run_dir under avrl/runs/L3v_rerun.

**Command shape:** bin/gpu_run.sh avrl/runs/L3v_rerun 'bash avrl/exp/L3v.sh' (about 13 min wall, one run). Check bin/gpu_status.sh first.

**Metric:** PASS count, bias point estimate and the per-pool verdict counts, compared with the L3v entry.

**Pass threshold:** not applicable (calibration check). A reproduction of 0 of 20 PASS and a bias near 0.002 is the expected result.

**Kill threshold:** the re-run disagrees with L3v on PASS count or on the bias point beyond the L3v pre-registered tolerance. Then L3w stays gated and the gate is re-examined.

**Expected runtime:** about 13 min GPU wall.

**Expected information value:** medium. It does not change claim 2 or claim 3, but it is required before any L3w credit, so it is on the critical path.

Added after L3f (2026-10-19), under the same "Proposed by the loop" heading. The L3z queue was numbered iteration 15, so these ids are L16a and L16b (iteration 16; rename if the loop numbers differently).

## L16a: Cross-model replication of the L3f kills on the same 3,000 prompts (GPU rollouts, CPU analysis)

**Hypothesis:** the claim-2 kill (learned minus threshold admitted, 95% CI entirely above 0) and the claim-3 kill (AUC gain below 0.05 and cell100 ratio below 1.5) replicate when the rollout model is Qwen2.5-0.5B-Instruct instead of Qwen2.5-1.5B-Instruct, on the same 3,000 prompts. Both kills so far rest on one model and one task.

**Change vs existing scripts:** copy exp/L3f_rollouts.py to exp/L16a_rollouts.py and pass --model models/Qwen2.5-0.5B-Instruct (the flag already exists). Take the 3,000 questions from runs/L3f/pool/groups.parquet instead of drawing new ones, so the only change is the model. Copy exp/L3f_analyze.py with the pool path set to runs/L16a/pool. E03 (e03_real_allocators.py) and exp/L3f.py run unchanged on the new pool.

**Command shape:** bin/gpu_run.sh runs/L16a/gpu 'cd /homes/rdubash/avrl_live/CHEAP/avrl && /homes/rdubash/active-audit/.venv-judge/bin/python -B exp/L16a_rollouts.py --model models/Qwen2.5-0.5B-Instruct --n_prompts 3000 --G 8 --max_tokens 512 --seed 1 --out runs/L16a/pool && /homes/rdubash/active-audit/.venv/bin/python -B exp/L16a_analyze.py'. Then the CPU runner pattern from exp/L3f.sh with the pool path and output directory changed.

**Metric:** as L3f. Admitted by arm at 25% (750 calls, seeds 0-7), paired differences with the 100-resample group bootstrap (seeds 0-3), and claim-3 AUC gain and cell100 neighbor ratio on the full pool. Report the proxy error rate, which will differ from 0.1637.

**Pass threshold:** candidate only, not credited without a further pre-registration. The 95% CI of learned minus threshold admitted lies entirely below 0. A PASS-candidate would conflict with both L3f and E01 and would need its own follow-up.

**Kill threshold:** the claim-2 kill replicates if the 95% lower bound of learned minus threshold admitted is above 0. The claim-3 kill replicates if AUC gain is below 0.05 and cell100 ratio is below 1.5. A CI that straddles 0 is INCONCLUSIVE. Proposed degenerate-run guard, fixed before launch: fewer than 100 proxy errors out of 3,000 groups is recorded as failed, not as a verdict.

**Expected runtime:** about 15 min GPU for rollouts (estimate: the 1.5B run took 1,143 s, and the 0.5B model should be faster), plus about 26 min CPU for E03 with the bootstrap (L3f took 1,578 s).

**Expected information value:** high. It is the only backlog test that varies the rollout model. A replicated kill answers the one-model objection to both kills. A PASS-candidate would show that the kill depends on the model, which would reopen claim 2.

## L16b: Bootstrap CIs for the claim-3 structure metrics on both pools (analysis-only, CPU)

**Hypothesis:** the claim-3 kill holds as intervals, not only as point estimates. On the fresh pool, the AUC gain (point -0.0038) and the cell100 neighbor ratio (point 1.007) have 95% CIs that stay on the kill side of 0.05 and 1.5.

**Change vs existing scripts:** a new script that resamples groups with replacement (100 resamples, seeds 0-3, the E03 bootstrap convention) and recomputes auc_cheap, auc_cheap_emb and the cell100 neighbor ratio, reusing the functions in avrl/p0_analyze.py. Run on the fresh pool (runs/L3f/pool) and on the E01 pool (runs/p0).

**Command shape:** local shell, /homes/rdubash/active-audit/.venv/bin/python -B on the new script, under 2 min CPU.

**Pass threshold:** not applicable (descriptive). The kill is already met on both pools by point estimates.

**Kill threshold:** not applicable. A verdict flip would need the cell100 ratio CI to reach 1.5 or the AUC gain CI to reach 0.05, far from the current point estimates.

**Expected runtime:** under 2 min CPU.

**Expected information value:** low-medium. It removes the caveat that claim 3 has point estimates only. It is unlikely to change the status, so run it only if the claim-3 kill needs a stated interval in the write-up.
