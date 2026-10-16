# D6: A product framework for budgeted verification in RL post-training

## The idea in plain language

Package the project's own two-step playbook as a library: before training, run a short diagnostic that checks whether a cheap grader's confidence predicts its own errors well and whether the errors it misses cluster by prompt. That single check tells you which of two policies to use: a fixed, periodically recalibrated threshold (cheap, safe, near-optimal when the cheap signal is strong), or a learned allocator that targets prompt regions where the cheap grader is exploitable (worth the complexity only when the cheap signal is weak and errors cluster). Ship this as a drop-in reward wrapper for TRL and verl, not a new RL algorithm.

## The mechanism, precisely

Four pieces:

1. **Grader interface.** Two functions with the same signature, `cheap(prompts, completions) -> (score, confidence_features)` and `expensive(prompts, completions) -> score`, where `expensive` is metered.
2. **Phase-0 diagnostic.** Run once on a sample of prompts before training: compute AUC of `confidence_features` for predicting cheap-vs-expensive disagreement, and the neighbor ratio of errors in prompt-embedding space restricted to the confident subset. This project's own data gives working thresholds: embedding-AUC gain under 0.05 and neighbor ratio under 1.5 means pick the fixed threshold; cheap AUC under about 0.80 with neighbor ratio over 1.5 in the confident subset means pick a learned allocator.
3. **Allocation policies.** `FixedThreshold` (recalibrated every N steps), `SelfTrainingAllocator` and `CellBandit` (the two replay winners), plus `Full`, `Random`, `Oracle` as dashboard baselines.
4. **Budget accounting and dashboard.** A per-step counter enforcing `expensive_calls_per_step`, logging an estimated corrupted-reward count (cheap score used where cheap and expensive disagreed), exposed through TRL's `log_metric`/`log_extra` hooks or an equivalent verl callback.

The reward wrapper matches TRL's `reward_funcs` signature (`completions, prompts, log_metric, **kwargs -> list[float]`, confirmed against the current GRPOTrainer docs) and a parallel wrapper matches verl's pluggable reward manager. Internally it calls the diagnostic once, picks a policy, metres expensive calls, blends the reward, and logs the corrupted-reward curve.

## Prior art, with URLs

- **TRL `reward_funcs`** (https://huggingface.co/docs/trl/main/en/grpo_trainer): supports arbitrary sync/async reward functions or pretrained reward models, multiple functions combined by a weighted sum, and logging hooks. No budget concept, no choice between graders, just addition. This project's wrapper slots into exactly this API.
- **verl / HybridFlow** (arXiv:2409.19256, https://arxiv.org/abs/2409.19256): pluggable reward manager, but the paper's contribution is single/multi-controller placement for throughput, not deciding which prompts get the expensive grader.
- **OpenRLHF**: ray-based distributed RLHF with a remote reward-model server. UNVERIFIED in detail (not fetched this session), but its documented pattern treats the reward-model call as mandatory per sample, not optional or budgeted.
- **Unsloth**: wraps TRL's GRPOTrainer with faster LoRA kernels. UNVERIFIED whether it alters `reward_funcs` semantics; public docs suggest it reuses TRL's path unchanged.
- **VIP, variance-informed rollout allocation** (arXiv:2602.01601, https://arxiv.org/abs/2602.01601): learns in-training budget allocation, but for how many rollouts to sample per prompt under a single grader, to cut gradient variance. It never chooses between a cheap and an expensive grader.
- **Trust or Escalate** (arXiv:2407.18370, https://arxiv.org/abs/2407.18370): a cheap-then-expensive judge cascade with human-agreement guarantees, but static and evaluation-time, no in-training budget-vs-corrupted-reward accounting.
- **Heteroskedastic Signals in Budgeted LLM Verification** (arXiv:2606.15841, https://arxiv.org/abs/2606.15841). Closest prior art, confirmed via OpenAlex and arXiv. Proposes Heterogeneity-Gated Allocation (HGA): a warm-up comparability test decides between global thresholding and cost-stratified thresholding for selective-compute verification, because uncertainty quality is heteroskedastic across cost strata. This is a regime diagnostic in the same spirit as phase 0 here, but it is general selective-compute verification, not an RL training-loop reward wrapper, and its choice is between two kinds of threshold, never between a threshold and a learned allocator that co-adapts with a drifting policy.
- **Instance-Optimal Estimation with Multiple LLM Judges on a Budget** (arXiv:2605.23362, https://arxiv.org/abs/2605.23362): budgeted allocation across several judges to estimate a quantity. UNVERIFIED beyond title/abstract; appears aimed at evaluation/estimation, not RL training.

## The exact novelty claim that survives

Not "learned beats fixed" in general (false here, and HGA already shows regime-gating beats either fixed choice in its own setting). What survives: (1) the regime test applied specifically to reward-model-vs-judge pairs inside an RL reward wrapper, scored by corrupted rewards admitted during training rather than held-out error rate; (2) the finding that the regime can change once training starts, because the allocator's own selection bias and the policy's drift interact (seed-0 GSM8K: the fixed threshold's corrupted admissions fell over training while the learned allocator's did not), which a one-shot pre-training warm-up test like HGA's cannot see; (3) no existing framework packages the diagnostic plus both policies as a drop-in TRL/verl reward wrapper.

## Smallest experiment, on the existing harnesses

Add `Phase0Diagnostic` to `avrl/allocators.py`. Call it at step 0 in `avrl/p1_grpo.py` and `avrl/p1_judge_grpo.py`, before the training loop, logging the predicted regime alongside the per-step corrupted-reward curves those harnesses already produce. Pass criterion: the diagnostic's step-0 label matches which allocator has fewer corrupted admissions by step 100+, on the two runs already done (P1-goldberg-s0, P1-judge-s0) and, once it completes, the hannah job 78883 run. Files to change: `avrl/allocators.py`, `avrl/p1_grpo.py`, `avrl/p1_judge_grpo.py`, a new entry in `avrl/RESEARCH_LOG.md`. No new data collection; this reuses existing rollouts and logs.

## Publication shape

Venue: a workshop paper (an efficient-ML or LLM-post-training workshop), given this is a measurement result, not a new mechanism. Title: "When to Trust the Cheap Grader: A Regime Diagnostic for Budgeted Verification in RL Post-Training." Headline figure: a scatter with cheap-grader AUC on the x-axis and allocator gain over the budget-matched threshold on the y-axis, showing the project's measured points split into two clusters (GSM8K/MBPP self-consistency near zero gain at AUC 0.87 to 0.93; open-ended reward-model-vs-judge at plus 19 to 31 percent at AUC 0.75 to 0.80), with HGA's warm-up test marked as the nearest existing diagnostic.

## Product angle

Working name (a suggestion, not decided): `gradewise`. A user wraps their existing cheap grader (a regex check, a small reward model, a self-consistency vote) and their expensive grader (an API call to a bigger judge, a human queue, ground truth) in the two-method interface, calls `diagnose()` on a prompt sample, and gets back a printed regime label and a recommended policy. They pass `as_trl_reward_fn()` straight into `GRPOTrainer(reward_funcs=...)` or `as_verl_reward_fn()` into verl's reward manager config, set `expensive_calls_per_step`, and watch a small dashboard: cumulative expensive calls spent, estimated corrupted rewards admitted, and the regime call. Smallest demo: a single-GPU notebook training a 0.5B model on GSM8K (threshold wins, dashboard shows it) side by side with HelpSteer2 plus a judge (learned allocator wins, dashboard shows that instead), so the whole demo is the tool picking the right policy in both of this project's own measured regimes without the user hand-tuning anything.
