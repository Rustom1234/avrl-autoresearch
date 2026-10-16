# Judge pass: novelty and prior art

Lens: does this direction add something not already published, and does the cited prior art actually hold up. Scores are 1 to 10. A direction whose core mechanism is already published scores novelty 2 or less, even if the application target is new.

## Cross-cutting finding before the scores

WebFetch on https://arxiv.org/abs/2510.20369 confirms "Ask a Strong LLM Judge when Your Reward Model is Uncertain" (Xu, Lu, Zhang, Qiu, Hong, Yu, Yao, Liu, Jiang, Li, Yun, Zhao; submitted Oct 23, 2025) is a real paper whose abstract states it is "an uncertainty-based routing framework that efficiently complements a fast RM with a strong but costly LLM judge," formulates advantage estimation as pairwise preference classification, routes uncertain pairs to the judge and confident ones to the RM, and reports results on "improving online RLHF." That is cheap-RM-plus-expensive-judge routing under a budget, inside RL training, roughly a year before this project's open-ended-regime work. UNVERIFIED beyond the abstract (full text not fetched, no confirmation of whether its router is learned jointly with the policy, which task families it used, or whether it measures corrupted-reward admission or prompt clustering), but the abstract alone is close enough to the project's own core combination that it caps novelty for any direction that leans on "nobody routes a cheap RM to an expensive judge inside RL" as its claim. D3 and D6 lean on exactly that.

## D1: Consistency gating (self-consistency as the general cheap grader)

Novelty: 3. The confidence statistic is Self-Consistency (Wang et al. 2022, https://arxiv.org/abs/2203.11171), using agreement as the reward is TTRL (https://arxiv.org/abs/2504.16084), and fixed vote-correction rules exist in Dual Consensus (https://arxiv.org/abs/2603.16223); the file's own "exact novelty claim" section admits no new estimator, only an untested extension (embedding similarity for free text) and an online percentile threshold, and percentile-relative recalibration under drift is standard practice outside this niche, not a new idea.
Feasibility: 8. Reuses already-sampled E09d completions and an existing embedding model; the smallest experiment is an analysis script, no new RL run.
Publishability: 3. The file itself says this is not a standalone paper, only an ablation inside the main submission; as a section it is fine, as a judged "direction" it has little to publish on its own.
Product value: 4. One more candidate cheap-grader feature inside the allocator; useful but narrow, and only pays off if it beats the reward model's 0.80 AUC, which is unproven.

## D2: Verifier cascades in RL, recalibrated online

Novelty: 3. Trust or Escalate (https://arxiv.org/abs/2407.18370) already does a calibrated cheap-to-expensive cascade; Heteroskedastic Signals (https://arxiv.org/abs/2606.15841, confirmed by WebFetch: Jinlong Yang, submitted June 2026, proposes Heterogeneity-Gated Allocation with a warm-up comparability test, evaluation-time and static per the fetched abstract) already argues for and tests stratified-versus-global thresholds; what is left, periodically refitting a threshold from a buffer decoupled from the policy's own routing decisions plus a drift trigger, is standard concept-drift-detection methodology (the ADWIN/DDM family of ideas) relabeled for this setting, not a new mechanism.
Feasibility: 7. Clear, small, reuses the E09d setup and existing allocator scaffolding; the three-arm rerun is more expensive than D1 but still fits goldberg.
Publishability: 4. Workshop-tier by the file's own admission; the in-training falsifiable test (does decoupled recalibration dodge the self-reinforcement failure seen on GSM8K) gives it a cleaner story than D1.
Product value: 5. A cascade with a visible recalibration log is a believable feature, but it is competing with D6 for the same dashboard, not a separate product.

## D3: Structure-aware allocator (cell bandit + classifier + regime gate)

Novelty: 2. Every individual piece is cited and published by the file's own account: VIP (https://arxiv.org/abs/2602.01601, confirmed by WebFetch: rollout-count allocation to cut gradient variance, no grader cost at all) for in-training learned allocation, ProbeLLM (https://arxiv.org/abs/2602.12966) for failure clustering, Heteroskedastic Signals for a self-gating regime test, and general budgeted-bandit theory (GUTS, https://arxiv.org/abs/2304.02075; budgeted Thompson sampling, https://arxiv.org/abs/1505.00146 and https://arxiv.org/abs/2408.15535) for the cell-level exploration mechanism itself; worse, the cross-cutting finding above shows a paper already routes a cheap RM to an expensive judge inside online RLHF under uncertainty, which undercuts the file's claim that "no located paper combines" these pieces under one roof. A combination of five published components, three of which (bandit, classifier, gate) each individually duplicate existing mechanisms, does not clear the bar for a method novelty above 2.
Feasibility: 5. Three interacting learned pieces (cell posteriors, an online classifier, a regime gate with inverse-propensity correction) is real engineering complexity and more failure surface than D1/D2; the propensity-weighting correctness is subtle and easy to get wrong silently.
Publishability: 4. As a workshop ablation, matching the file's own framing; a main-track reviewer who knows arXiv:2510.20369 would likely call this an application of known routing ideas, not a new design.
Product value: 4. Higher ceiling if it works, but higher risk of silent bugs (biased propensity weights) than the simpler allocators, which is a liability in a shipped product.

## D4: Frontier budget (tie verification to the GRPO zero-advantage band)

Novelty: 3. DAPO (https://arxiv.org/abs/2503.14476), PCL (https://arxiv.org/abs/2510.01135), and MoPPS (https://arxiv.org/abs/2507.04632) already exist exactly as described and are correctly distinguished (they filter what to roll out, not what to grade); the file's own novelty section calls this "mostly a reframing" on the self-consistency regime and flags that it should reproduce, not beat, existing ledger numbers. Deriving the existing threshold's shape from GRPO's advantage formula instead of fitting it is a reasonable removal of one tuning knob, not a new mechanism, and the paper's own honesty here (it says so directly) is correctly harsh self-assessment.
Feasibility: 8. Cheapest experiment of the six: a bucketing function against data already on disk.
Publishability: 3. By the file's own admission, not a standalone result, one ablation figure inside the main paper.
Product value: 3. "No-tuning heuristic" is a nice line for a demo but a minor feature, not a product on its own.

## D5: Canary divergence alarm (reward-hacking defense)

Novelty: 5. The highest of the six from this lens. Baker et al.'s CoT monitor (https://arxiv.org/abs/2503.11926) penalizes at a constant rate and documents monitorability tax, not adaptive budget reallocation; Spurious Rewards (https://arxiv.org/abs/2506.10947) and Gao, Schulman, Hilton's reward-overoptimization scaling laws (https://arxiv.org/abs/2210.10760) are end-of-training or lawful-trend diagnostics, not a live alarm; Trust or Escalate is static. No cited paper combines an unbiased, never-selected canary sample with a per-cluster divergence trajectory that actively reallocates a scarce verification budget toward the alarming cluster, with the claim checked against an injected, ground-truth-labeled exploit rather than inferred. The mechanism itself (held-out monitoring set, trend-triggered intervention) is standard general ML-monitoring practice, which is why this is a 5 and not higher, but the application and the falsifiable injected-exploit test are the most genuinely underexplored combination of the six.
Feasibility: 7. Needs new injected-exploit infrastructure (a marker-string bonus) that the other directions do not, but it is a small, well-specified addition to an existing harness.
Publishability: 5. Has the cleanest standalone story of the six: one mechanism, one ground-truth-labeled test, one falsifiable claim, matched to a safety/monitoring workshop rather than needing to win the main AVRL thesis argument.
Product value: 6. A hacking-risk alarm is a more differentiated pitch than another allocator variant, and it is the one direction whose "what the user sees" story (a rising risk number before held-out accuracy breaks) is not just a repackaging of the project's existing dashboard.

## D6: Product framework (phase-0 diagnostic plus TRL/verl reward wrapper)

Novelty: 2. The core mechanism, a regime diagnostic that decides which allocation policy to run, is already published: Heteroskedastic Signals' Heterogeneity-Gated Allocation (confirmed by WebFetch, https://arxiv.org/abs/2606.15841) is a warm-up comparability test that gates between two allocation strategies for budgeted verification; that D6 gates between a different pair (fixed threshold vs. learned allocator, for RL reward wrappers specifically) is a change of application target, not a new idea, and the cross-cutting finding (arXiv:2510.20369 already routes RM-to-judge inside online RLHF) further narrows what is left to claim as new. TRL's `reward_funcs` and verl's reward manager (https://arxiv.org/abs/2409.19256) already supply the plumbing this wraps.
Feasibility: 8. Mostly glue code around interfaces and allocators that already exist in this repo; lowest technical risk of the six.
Publishability: 2. A reviewer who has read HGA would read this as "HGA's gating idea, applied to RL training with a software wrapper," which is a weak paper contribution on its own; it is a packaging effort, not a research result, and the file's own publication section already downgrades it to workshop-measurement-paper status.
Product value: 8. Highest of the six, and the only direction that squarely answers what the owner asked for in (b): a usable open framework, not a paper contribution. Novelty score should not be read as a verdict against shipping this; it is a verdict against calling it a paper.

## Summary table

| direction | novelty | feasibility | publishability | product value |
|---|---|---|---|---|
| D1 consistency gating | 3 | 8 | 3 | 4 |
| D2 cascade in RL | 3 | 7 | 4 | 5 |
| D3 structure-aware allocator | 2 | 5 | 4 | 4 |
| D4 frontier budget | 3 | 8 | 3 | 3 |
| D5 canary divergence alarm | 5 | 7 | 5 | 6 |
| D6 product framework | 2 | 8 | 2 | 8 |

## Bottom line from this lens

Nothing here clears novelty 6. The project's actual defensible claim, already stated in STATUS.md, is a measurement (which regime a cheap grader is in, and that the regime predicts whether a learned allocator helps), not a new allocation mechanism; every direction that tries to add a new mechanism on top (D2, D3, D4) runs into prior art that already occupies most of the same ground, and D3 in particular runs into a paper (arXiv:2510.20369) that already does the project's central cheap-RM-to-expensive-judge-under-RL combination. D5 is the one direction whose combination (unbiased canary plus reallocation plus an injected, ground-truth-checked exploit) was not found published anywhere in the six files' prior-art sections, which is why it is the only one above a 4. D6 is honest that it is a product, not a discovery, and should be pitched and evaluated as exactly that rather than inflated into a paper claim.
