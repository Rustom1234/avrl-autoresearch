# D4: Spend the verification budget at the learning frontier

## The idea in plain language

GRPO only learns from a rollout group when the group disagrees with itself. If every rollout in a group gets the same reward, the advantage (reward minus group mean) is zero for all of them and the gradient vanishes. So a group only teaches the policy anything when its pass rate, under whatever grader produced the rewards, sits away from 0 and 1. The proposal is to spend the scarce expensive-grader budget only on groups whose pass rate (as measured by the cheap grader) falls in that band, roughly 0.2 to 0.8, and to skip expensive grading entirely on groups the cheap grader already calls unanimous. Pair that with two existing ideas: DAPO's dynamic sampling, which throws away or resamples prompts that turn out to be unanimous rather than wasting a training step on them, and learnability filters (PCL, MoPPS), which try to predict before rollout whether a prompt will land in that band, so compute is not spent generating rollouts for prompts that were never going to teach the policy anything.

## The mechanism, precisely

1. Roll out a group of k samples per prompt, score every sample with the cheap grader only.
2. Compute the group's cheap pass rate p_hat.
3. If p_hat is 0 or 1, treat the group as saturated: either drop it (DAPO-style resampling, redraw a prompt to refill the batch) or keep it but never send it to the expensive grader, since zero-advantage groups cannot move the policy regardless of reward correctness.
4. If p_hat is in [0.2, 0.8], the group is at the frontier: these are the only groups eligible for expensive grading, because they are the only groups where a corrupted cheap reward actually changes which direction the policy moves.
5. Within the frontier set, spend the expensive-grader budget using the project's existing cheap-confidence ranking (verify the least-confident, most split groups first), which is the fixed rule already shown near-optimal in the self-consistency regime (ledger E01, E05, E06, E08).
6. Optionally add a pre-rollout learnability predictor (MoPPS's online Bayesian estimate, or PCL's learned value model) to skip generating rollouts at all for prompts predicted to be saturated, before step 1 ever runs.

## Prior art, and what each already did

- DAPO, "An Open-Source LLM Reinforcement Learning System at Scale" (Yu et al., 2025, arXiv:2503.14476, https://arxiv.org/abs/2503.14476). Introduces dynamic sampling: filter out prompts whose group gets all-correct or all-wrong rewards and resample to refill the batch with nonzero-advantage groups. It assumes rewards are already computed and cheap (rule-based verifiers); it never considers a two-tier grader or a budget on grading itself.
- PCL, "Prompt Curriculum Learning for Efficient LLM Post-Training" (Gao, Kim, Sun, Joachims, Wang, Pang, Tan, 2025, arXiv:2510.01135, https://arxiv.org/abs/2510.01135). Trains a value model to predict which prompts are at intermediate difficulty and skips rollouts for the rest, saving generation compute. Single reward channel, no expensive/cheap split; the thing it is predicting is difficulty, not grader disagreement.
- MoPPS, "Can Prompt Difficulty be Online Predicted for Accelerating RL Finetuning of Reasoning Models?" (Qu, Wang, Mao, Hu, Ommer, Ji, 2025, arXiv:2507.04632, https://arxiv.org/abs/2507.04632). A Bayesian risk-predictive framework that estimates prompt difficulty online without running the model, used the same way as PCL: to decide what to roll out, not what to grade.
- VIP, "Adaptive Rollout Allocation for Online Reinforcement Learning with Verifiable Rewards" (Nguyen, Nguyen, Ma, Zhao, She, Nguyen, 2026, arXiv:2602.01601, https://arxiv.org/abs/2602.01601). Allocates rollout count per prompt to minimize gradient variance; this is the paper already flagged in PROJECT.md as doing in-training budget allocation, but for how many rollouts to draw, not for whether to pay for a second, more trustworthy grader.
- Trust or Escalate (Jung, Brahman, Choi, 2024, arXiv:2407.18370, https://arxiv.org/abs/2407.18370), the fixed cheap-then-expensive cascade used at evaluation time, already cited as the project's closest comparison.

None of these five papers combine a two-tier grader cost with the GRPO zero-advantage condition. DAPO/PCL/MoPPS/VIP operate on a single reward channel and decide what to sample; AVRL's own result (ledger E01-E09d) is a budget rule on a second grader, fit to data rather than derived from the optimization.

## The novelty claim that survives

Tying the AVRL fixed threshold to the GRPO zero-advantage band is mostly a reframing in the self-consistency regime: a split-vote group and a low-cheap-confidence group are close to the same set on GSM8K and MBPP, so this should reproduce E01/E08, not beat them. The part that is new and testable: (a) a hyperparameter-light version of the existing threshold, derived from the optimization structure (0.2-0.8 from GRPO's advantage, not fit on held-out data), which removes a tuning knob the current harness has; (b) a compounding two-stage budget, pre-rollout learnability filtering (PCL/MoPPS, saves generation cost) composed with post-rollout grading filtering (AVRL's threshold, saves grading cost), which nobody has measured together; (c) whether, in the weak-signal open-ended regime (E09d, neighbor ratio 2.0), DAPO-style resampling changes which groups land in the frontier band enough to change the allocator-vs-threshold gap found in L9d. UNVERIFIED: whether the two filters are uncorrelated enough for the savings to compound multiplicatively rather than overlapping.

## Smallest experiment, on the existing harnesses

Add a pass-rate-band bucketing function to avrl/p0_analyze.py: label each group saturated-low (p_hat < 0.2), frontier (0.2-0.8), saturated-high (p_hat > 0.8) by the cheap grader's own group score, and diff this selection against the existing confidence-threshold selection on the E01 (GSM8K) and E09d (HelpSteer2, 7B judge) data already on disk. Add a "frontier" entry to avrl/allocators.py that restricts verification to the frontier bucket, ranked by existing cheap confidence, and run it through the same replay and bootstrap used for L9d so the comparison is budget-matched. Add a DAPO-style resample step to avrl/p1_grpo.py and avrl/p1_judge_grpo.py (redraw a prompt when its realized group is saturated, before any verification budget is spent) and rerun the in-training comparison already queued (hannah job 78883) with and without this step as a fourth arm.

## Publication shape

Not a standalone paper on these numbers; this is one ablation inside the main AVRL submission, answering "does tying the grading budget to the training-loss frontier beat the already-tuned fixed threshold." Venue: an efficient-training or LLM-post-training-economics workshop at ICML/NeurIPS, or as a figure in the main AVRL paper if the in-training open-ended result confirms. Title for the ablation: "The grading budget and the GRPO frontier are nearly the same rule." Headline figure: x-axis cheap-grader AUC (E01 through E09d), y-axis gain of frontier-band allocator over tuned-confidence threshold at matched budget, both near zero on GSM8K/MBPP, open question mark on the open-ended points pending the in-training rerun.

## The product angle

A user training a small model with TRL GRPO plus a cheap grader and a budgeted expensive grader calls one function, verify_budget(group_rewards, cheap_scores, budget), which buckets groups by the GRPO-frontier rule (no tuning), resamples saturated prompts instead of wasting a step, and spends expensive calls on frontier groups ranked by cheap-grader confidence. They would see a per-step panel, reusing the project's own ledger table format, showing calls spent, corrupted rewards caught, and the frontier share of the batch: a one-line, no-tuning policy instead of the twenty-experiment sweep this repo needed to find and validate the same rule.
