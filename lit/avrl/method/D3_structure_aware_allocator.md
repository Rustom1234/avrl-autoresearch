# D3: Structure-aware allocator for the weak-signal regime

## The idea in plain language

When the cheap grader's own confidence already predicts its errors (AUC 0.87 to 0.93, the math and code regime), a fixed rule that verifies the split-vote groups is close to optimal, and nothing learned should be trusted with the training signal. When the cheap grader is a reward model scoring open-ended text (AUC 0.75 to 0.80), its confidence is a weaker signal, and the groups it gets wrong cluster by prompt, not randomly. A router that knows which prompt neighborhood tends to produce corrupted rewards can spend its budget there instead of by confidence alone. The idea is two cooperating pieces: a bandit over prompt neighborhoods that learns which neighborhoods are dangerous, and a classifier that ranks groups inside a neighborhood using the reward model's own weak signal. A third piece, a regime detector, checks whether the cheap signal is actually weak before trusting either of them, and falls back to the fixed split-vote rule the moment it is not.

## The mechanism, precisely

Cells. Partition the embedding space of training prompts into k cells (k-means on a frozen sentence embedding, matching the "cell100" grouping already used in the replay harness).

Cell Thompson sampling. Each cell c keeps a Beta(a_c, b_c) posterior over its corrupted-reward rate. At each allocation step, sample theta_c per cell and rank cells by the sample; this spends more budget on cells whose posterior says "dangerous," while still occasionally probing low-posterior cells. That exploration is what the pure-exploitation "learned online" allocator lacked; P1-goldberg-s0 found its corrupted admissions rising over training, likely from feeding on its own biased sample.

Online classifier. Within the cells the bandit chooses, a small logistic model scores each group on cheap features (margin, entropy, length) to pick which groups get the real budget, using the group-level signal instead of discarding it.

Regime detector. Every N steps, compute AUC of cheap features for predicting corrupted reward on the groups verified so far (the same statistic as E09 through E09d). At or above an AUC of 0.85, the detector forces the fixed split-vote rule for the next window; the bandit and classifier keep updating but do not control the budget while the gate is closed. D3 is a strict superset of the fixed-rule baseline, not a competitor to it.

Joint training with the policy. All three parts refit from the same (features, cell id, verified-or-not, label) stream GRPO already produces; no separate held-out phase. Cell posteriors update after every verified group; the classifier refits every M steps on a sliding window, bounding the selection-bias feedback loop P1-goldberg-s0 diagnosed. The gate's AUC estimate is corrected for the bandit's non-uniform sampling by inverse-propensity weighting (propensity equals theta_c at selection time), since otherwise it is biased toward whatever the bandit already believes is easy.

## Prior art

VIP, "Adaptive Rollout Allocation for Online Reinforcement Learning with Verifiable Rewards" (arXiv 2602.01601, https://arxiv.org/abs/2602.01601). Learns a per-prompt rollout budget during RLVR to minimize gradient variance. In-training and learned like D3, but allocates rollout count, never verifier spend, with no prompt clusters or regime switch.

ProbeLLM, "ProbeLLM: Automating Principled Diagnosis of LLM Failures" (arXiv 2602.12966, https://arxiv.org/abs/2602.12966). Uses tree search to find clusters of inputs where a model fails, the clustering structure D3 exploits, but as an offline diagnostic, not a budgeted router trained inside an RL loop.

"Ask a Strong LLM Judge when Your Reward Model is Uncertain" (arXiv 2510.20369, https://arxiv.org/abs/2510.20369). The closest single paper: routes uncertain reward-model predictions to a stronger judge under a budget. UNVERIFIED whether its rule is learned jointly with training or fit once offline; the summary reads as a per-example threshold, no prompt-cluster structure, no regime gate.

arXiv 2606.15841 (https://arxiv.org/abs/2606.15841), UNVERIFIED author and venue, fetched as "Heteroskedastic Signals in Budgeted LLM Verification: Structural Heterogeneity Limits Optimization Gains." Argues uncertainty quality varies across cost strata, capping any optimizer's gain, and recommends testing decision-comparability first. This is effectively the paper-length case for D3's regime detector, but it proposes no online self-switching gate and no cell-level bandit.

Trust or Escalate, "LLM Judges with Provable Guarantees for Human Agreement" (arXiv 2407.18370, https://arxiv.org/abs/2407.18370; ICLR 2025 per project notes, UNVERIFIED here since the fetched record shows only a 2024 arXiv date). A fixed, calibrated cascade with guarantees, used at evaluation time, not inside training, not cluster-aware.

Active search and budgeted-bandit theory: GUTS (arXiv 2304.02075, https://arxiv.org/abs/2304.02075) and budgeted Thompson sampling (arXiv 1505.00146, https://arxiv.org/abs/1505.00146; arXiv 2408.15535, https://arxiv.org/abs/2408.15535). General machinery for spending a budget to find rare positives; none specific to LLM verification or a policy drifting underneath the search.

## The exact novelty claim that survives

No located paper combines a cell-level bandit over where corrupted rewards cluster, an online classifier using the cheap grader's weak signal within a cell, a regime detector that measures its own precondition (cheap-feature AUC) and disables itself when that fails, and all three refit online from the stream GRPO already generates, with inverse-propensity correction for the bandit's effect on labeling. The ingredients are each published; the combination, specifically the self-disabling gate trained alongside the policy, is not.

## Smallest experiment

Reuse the existing judge harness. Files to change: add a StructureAwareAllocator class to avrl/allocators.py (posteriors, classifier, gate, alongside the existing threshold and self-training allocators); add it as a fifth arm in avrl/p1_judge_grpo.py; add avrl/exp/D3_structure_aware.py as the launch script, cloned from the E09d-era script behind the L9d replay numbers. First test in replay only, on the collected E09d rollouts and labels, no retraining: does the gated cell-bandit-plus-classifier beat the plain cell bandit's +31 percent [14, 37] and the self-trainer's +19 percent [8, 26] at a 25 percent budget, and does the gate stay closed throughout (E09d's measured AUC, 0.795 to 0.80, sits close to the proposed 0.85, so this also checks whether the threshold needs tuning). If it clears replay, run it in training on the A100 slot already queued for the judge comparison.

## Publication shape

Venue: a training or RL workshop (ICLR or NeurIPS), not a main-track claim, since the contribution is a measurement-backed design choice plus one combination, not a new learning principle. Title: "When to Learn and When to Threshold: A Self-Gating Allocator for Budgeted Verification in RL." Headline figure: one plot, cheap-feature AUC on the x-axis, allocator gain over the budget-matched fixed rule on the y-axis, the two measured regimes as points (math and code near zero, open-ended near plus 20 to 30 percent), the gate's threshold marked, and D3's allocator overlaid, tracking the fixed rule's zero gain on the left and the cell bandit's gain on the right.

## The product angle

A small open-source wrapper around TRL GRPO: avrl train, with flags for policy, cheap grader, expensive grader, and budget fraction. The user supplies both graders and a budget; the tool runs the phase-0 diagnostics (the AUC and neighbor-ratio statistics already in p0_analyze.py), tells the user which regime their task sits in, and turns on the fixed rule or the structure-aware allocator accordingly, no allocator-design decision left to the user. What they see: a one-page dashboard showing corrupted rewards avoided per budget spent, the regime verdict, and a running count of expensive-grader calls against their quota.
