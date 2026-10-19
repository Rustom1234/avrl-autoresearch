# Verification-aware reinforcement learning (AVRL)

Started 2026-09-24. Owner: Rustom. Compute: one RTX A4000 16 GB workstation.

## Claim
Reward computation in RL post-training is a budgeted active-measurement problem. A learned allocator that decides, per rollout group, whether to pay for the expensive verifier or accept a cheap proxy, and that learns where the proxy is exploitable, (1) reaches the same policy quality with a fraction of expensive verifier calls, (2) admits fewer corrupted rewards into training than any fixed rule at the same budget, and (3) gets its edge from structure in prompt space (proxy error clusters), which is the SOFisher condition. If (3) fails, the honest result is a measurement of proxy-error structure plus a fixed-threshold recipe.

## Why this could matter to a lab
Every frontier reasoning model is trained with graded rollouts. Grading is the cost centre and the reward-hacking surface. Nobody has put a learned, budget-capped router in front of the expensive grader inside the training loop; the closest work is a fixed cheap-then-expensive cascade used at evaluation time (Trust or Escalate, ICLR 2025) and a 2026 result that a fixed threshold captures most evaluation-time gains (arXiv 2606.15841). The in-training case, where the policy and its failure modes drift, is open.

## Phase 0 (this machine): is there structure to exploit?
Qwen2.5-1.5B-Instruct, 3,000 GSM8K train prompts, 8 rollouts each. Cheap proxy = agreement with the group majority. Expensive verifier = gold answer. Measures: proxy-error rate overall and among confident (agreement >= 0.75) groups; AUC of cheap features and of prompt embeddings for proxy error; neighbor ratio of proxy error by embedding cell; and an allocation replay at 10 / 25 / 50 percent verification budgets comparing random, threshold (static cheap features), learned online, and oracle on proxy errors caught per verifier call.
Pre-registered reads: if proxy error among confident groups is under 3 percent, budgeting is moot on this task (move to a harder task or a noisier proxy). If embeddings add less than 0.05 AUC over cheap features and the neighbor ratio is under 1.5x, claim (3) is dead and only claims (1) and (2) remain, to be won by a fixed rule.

Phase 0 outcome (2026-09-30, ledger E01): the threshold regime. Agreement AUC 0.91, no embedding structure (neighbor ratio 1.0), fixed threshold near-optimal. Claim 3 is dead for GSM8K with majority vote; the project now measures where the allocator wins as a function of proxy confidence quality, per the revised proposal.

## Phase 1 (deferred: needs a larger GPU, not available in this run): does it change training?
GRPO with LoRA on the same model, 400 steps, four arms at a 25 percent verification budget: full (verify all, upper bound), random, threshold, learned. Logged per step: expensive calls, corrupted-reward groups admitted (gold used for logging only), and greedy accuracy on 500 held-out test prompts every 50 steps.
Kill test: if the learned arm does not beat the threshold arm on final accuracy at equal expensive calls by more than the seed spread across 3 seeds, the allocator is not worth a year; the fixed threshold recipe is the deliverable.

## Phase 2 (if phase 1 survives)
Real expensive judge (7B or API model) on open-ended and code tasks where the proxy is a small reward model; joint training of allocator and policy; reward-hacking study (does concentrating verification where the proxy is exploitable reduce measured hacking); transfer of the allocator across task families.

## Files
avrl/p0_rollouts.py, avrl/p0_analyze.py, avrl/p1_grpo.py (harness, built and reviewed by the workflow), avrl/runs/.
