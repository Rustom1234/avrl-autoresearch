# H9: Budgeted Verification in RL from Verifiable or Judged Rewards

## 1. Hypothesis and SOFisher Mapping

In RL post-training (GRPO, RLVR, RLAIF), a learned policy that decides which rollouts or prompts receive an expensive scorer (LLM judge, code sandbox, long tool run, human), using cheap proxies (small reward model, self-consistency, length, entropy) and exploiting autocorrelation of reward and reward-model error across prompts and rollout clusters, achieves the same policy improvement per unit of scoring cost with far fewer expensive calls. This also reduces reward hacking by concentrating expensive checks where the cheap proxy is least trustworthy.

SOFisher mapping: field of view -> prompt or rollout cluster; rare target -> reward-hacking or reward-model-error event; expensive fluorescence imaging -> LLM judge / sandbox execution; cheap state features -> small proxy reward model, self-consistency, entropy, length; spatial autocorrelation of tissue -> autocorrelation of reward and RM error across semantically or difficulty-clustered prompts.

## 2. Prior Art

- SOFisher, Ng et al., Nature Communications 2026: RL-guided (DQN) field-of-view selection for spatial omics imaging; outperforms conventional dense-sampling across metrics in simulation on real spatial datasets. https://www.nature.com/articles/s41467-026-73404-6
- DeepSeekMath / GRPO, Shao et al. 2024: introduces GRPO, critic-free group-relative RL; DeepSeekMath 7B reaches 51.7% on MATH without tools, 60.9% with 64-sample self-consistency. https://arxiv.org/abs/2402.03300
- DeepSeek-R1, DeepSeek-AI 2025 (Nature 2025): pure RLVR with rule-based rewards; R1-Zero AIME 2024 pass@1 rises 15.6% to 77.9% (86.7% with self-consistency); final R1 scores 79.8% on AIME 2024. https://arxiv.org/abs/2501.12948
- Constitutional AI, Bai et al. 2022 (Anthropic): defines RLAIF, training a preference model from AI-generated judgments in place of human labels for the RL phase. https://arxiv.org/abs/2212.08073
- Trust or Escalate, Jung et al., ICLR 2025: cascaded judge escalation (Mistral-7B to GPT-3.5/4) with provable human-agreement guarantees; up to 87% cost savings, 80%+ coverage at over 80% agreement, as low as 12.6% of full GPT-4 cost in the cheapest cascade. https://arxiv.org/abs/2407.18370
- On Cost-Effective LLM-as-a-Judge Improvement Techniques, 2025: surveys cascade, sampling, and caching techniques; reports a deployed cascade example cutting monthly judge spend from $283K to $34K. https://arxiv.org/html/2604.13717
- Uncertainty-Aware Reward Modeling for Stable RLHF, 2026: reward-model uncertainty estimation used to down-weight or flag untrustworthy reward predictions during PPO/RLHF training. https://arxiv.org/html/2606.19818v1
- Active Reward Modeling, 2025 (ICML): active-learning selection of which preference pairs to label using uncertainty and cross-prompt comparison, reducing labeling cost versus random selection. https://arxiv.org/abs/2502.04354
- Prompt Curriculum Learning, 2025 (ICLR 2026 submission): learned value model selects "Goldilocks" intermediate-difficulty prompts for RLVR, cutting the filtering cost by up to 17x while accelerating convergence. https://arxiv.org/abs/2510.01135 (ADS id 2025arXiv251001135G)
- VCRL: Variance-based Curriculum Reinforcement Learning, 2025: shows GRPO group-reward variance is low for too-easy/too-hard prompts and high for moderate difficulty, and schedules training on variance. https://arxiv.org/abs/2509.19803
- No Prompt Left Behind, 2025: reframes zero-variance (uniform-reward) prompts as still informative via entropy-based advantage assignment rather than discarding them. https://arxiv.org/pdf/2509.21880v1
- Adaptive Rollout Allocation (VIP), ICLR 2026: variance-informed allocation of a fixed rollout budget across prompts in online RLVR; reports higher sample efficiency than uniform GRPO allocation but the fetched abstract gives no exact percentage (UNVERIFIED magnitude). https://arxiv.org/abs/2602.01601
- Heteroskedastic Signals in Budgeted LLM Verification, 2026: argues structural heterogeneity in reward-signal noise across verification instances creates an irreducible ceiling on gains from adaptive/budgeted allocation strategies; direct counter-evidence to be addressed, not just supporting prior art. https://arxiv.org/pdf/2606.15841
- Fast Best-of-N Decoding via Speculative Rejection, 2024: prunes low-promise generations early during best-of-N reward scoring to cut inference cost, a speculative/early-exit analog for verification. https://arxiv.org/abs/2410.20290
- Inference-Aware Fine-Tuning for Best-of-N Sampling, 2024: cost/accuracy tradeoffs of best-of-N sampling and how to fine-tune policies aware of the sampling budget used at inference. https://arxiv.org/abs/2412.15287

## 3. What Is Genuinely Open as of 2026

No paper found directly casts "which rollout gets the expensive judge" as a sequential decision-making / bandit-with-budget problem trained jointly with the RL post-training loop itself, the way SOFisher trains a DQN jointly with the imaging campaign. Existing work splits into two disconnected lines: (a) cascade/escalation systems (Trust or Escalate, cost-effective judge surveys) that are static, offline, and built for one-shot evaluation, not for a policy that also updates the LLM being trained; and (b) curriculum/allocation work (PCL, VCRL, VIP, No Prompt Left Behind) that decides how many rollouts or which prompts to sample, but assumes the reward for every sampled rollout is already computed, not which rollouts additionally deserve the expensive scorer versus a cheap proxy. Nobody has combined "select which rollouts to sample" with "select which sampled rollouts get verified expensively" as a single budgeted-verification policy, and nobody has measured whether reward-model error is autocorrelated across GRPO's own rollout groups or across semantically similar prompts within a training run, as opposed to across the whole dataset (VCRL and PCL use difficulty and variance signals, not RM-error clustering, as the controller). The Heteroskedastic Signals paper (2606.15841) is a live objection: it argues heterogeneous, possibly non-autocorrelated noise structure caps the value of adaptive verification budgets. A year is needed because this requires (i) building the RM-error-autocorrelation measurement infra that does not currently exist for LLM RL, (ii) training the budgeted-verification policy jointly with an actual GRPO run rather than post hoc, (iii) directly testing against the heteroskedasticity objection on real RLVR/RLAIF data rather than the synthetic settings in 2606.15841, and (iv) demonstrating the reward-hacking reduction claim, which no cited paper attempts to measure as a byproduct of budgeted verification.

## 4. Three-Condition Check

1. Autocorrelation of reward/RM-error across prompt or rollout clusters: partially supported. VCRL (2509.19803) shows group-reward variance clusters by difficulty tier; No Prompt Left Behind (2509.21880) shows zero-variance clustering is systematic, not random. But 2606.15841 directly disputes that this structure is exploitable for verification budgeting, calling it heteroskedastic rather than smoothly autocorrelated. UNVERIFIED whether RM-error specifically (not just reward variance) is autocorrelated in this way; no paper found measures it directly.
2. Expensive measurement costs far more than cheap features: strongly supported. Self-consistency at 64 samples on MATH costs roughly $2000 in GPT-4 API calls per the cost-effective-judge literature, and a deployed cascade cut judge spend from $283K to $34K/month (2604.13717); Trust or Escalate shows cheap judges (Mistral-7B) handle 75-80% of cases at a fraction of GPT-4 cost (2407.18370).
3. Cheap features alone cannot rank targets: supported indirectly. Reward-model uncertainty work (2606.19818, 2502.04354) exists precisely because a single cheap RM score is known to be unreliable and needs active selection or ensembling; if cheap proxies alone sufficed, none of this active/uncertainty-aware literature would be necessary.

## 5. Year Plan

Phase 1 (weeks 1-6, 4x A100 PCIe 40GB): GRPO on a 1.5B model (e.g. Qwen2.5-1.5B) on a math dataset (GSM8K/MATH subset), with a simulated expensive judge (a larger model, e.g. 7-8B, called only when the budgeted-verification policy requests it) and a cheap proxy (small RM or self-consistency-of-4). Compare fixed-budget random/uniform verification versus a learned budgeting policy (contextual bandit over RM disagreement, entropy, prompt cluster). Metric: accuracy gain per expensive call (delta in eval accuracy divided by number of expensive-judge invocations), matched at equal total expensive-call budget against the strongest baseline.
Phase 2 (months 2-6): scale to 7-8B policy models, add a code-execution sandbox as the expensive scorer (RLVR-for-code setting), and test the reward-hacking-reduction claim by measuring exploit rate under matched budget.
Phase 3 (months 6-12): move to an RLAIF/judged-reward setting with a real large-model judge, test transfer across task families, and directly engage the heteroskedasticity objection from 2606.15841 by ablating the autocorrelation assumption on held-out prompt distributions.

Strongest existing baseline to beat: Trust or Escalate's cascaded selective evaluation (2407.18370) adapted to the RL-in-the-loop setting, since it is the most rigorous existing cost-aware verification method with a provable guarantee, though it was not built for a policy that changes during training.

## 6. Kill Criteria

Kill if, at matched expensive-call budget, the learned budgeting policy does not beat uniform-random verification allocation by a pre-registered margin (e.g. >=10% relative accuracy-per-call) by end of Phase 1. Kill if RM-error autocorrelation across rollout clusters is not measurably above the autocorrelation expected under the null (shuffled clusters), which would validate the heteroskedasticity objection in 2606.15841 for this setting. Kill if the reward-hacking-reduction effect is not distinguishable from noise in Phase 2's exploit-rate measurement.

## 7. Impact

A lab that validates H9 would stop sending every rollout through its most expensive judge/sandbox by default and instead run a lightweight learned router alongside the RM in its GRPO/RLAIF loop, cutting judge/sandbox calls per training run by a large factor (Trust or Escalate-scale savings, 75-87% range, would be the target) while holding policy-improvement-per-step roughly constant, and would gain a built-in, auditable signal (verification-budget allocation over time) for detecting where reward hacking is emerging.

SCORECARD
novelty: 6
feasibility_1yr: 7
lab_impact: 7
sofisher_fit: 8
strongest_baseline: Trust or Escalate cascaded selective evaluation (Jung et al., ICLR 2025)
one_line_pitch: Train a cheap-proxy-driven router that decides which GRPO/RLAIF rollouts earn the expensive judge, cutting scoring cost per unit of policy improvement the way SOFisher cuts imaging cost per unit of rare-target capture.
