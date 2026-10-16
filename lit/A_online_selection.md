# Model-aware, online, and adaptive data selection for LLM pretraining (2022-2026)

## Per-method summary

**RHO-LOSS** (Mindermann et al. 2022): selects points with highest "reducible holdout loss" (train loss minus a fixed irreducible-loss reference model's loss), down-weighting redundant/noisy/already-learned points. Expensive score: per-example loss under a small holdout-trained reference model, computed per candidate batch, not the whole corpus. Gain: 18x speedup, +2% accuracy vs uniform on noisy Clothing-1M. https://arxiv.org/abs/2206.07137

**MATES** (Yu, Das, Xiong 2024): a small data-influence model, retrained periodically during pretraining, predicts each example's effect on the current model and picks data for the next chunk. Expensive score: true influence (reference-set loss change) is only measured on a sampled subset per round; a distilled proxy then scores the full pool cheaply. Gain: on Pythia/C4, doubles gains of prior static reference-model selection, halves FLOPs to target performance. https://arxiv.org/abs/2406.06046

**DoReMi** (Xie et al. 2023): trains a small proxy model with Group DRO across domains to set domain reweighting, then resamples the full corpus for the final large run. No per-document score; the proxy training run (280M for an 8B target) is the one-time cost, computed offline, not adapted online. Gain: +6.5 pts avg few-shot accuracy vs Pile default weights, 2.6x fewer steps to baseline. https://arxiv.org/abs/2305.10429

**ODM / Online Data Mixing** (Albalak et al. 2023): casts domain mixing as a multi-armed bandit (Exp3-style), updating domain weights online from per-domain loss/information-gain reward. No document scoring, only domain-level arms; reward reuses training loss already computed. https://arxiv.org/abs/2312.02406

**ADO** (Jiang, Zhou, Feng, Malladi, Kolter, ICLR 2025): fits per-domain scaling laws online to estimate each domain's current "learning potential" and adjusts mixture weights, no proxy model needed. Domain-level, in-loop, negligible overhead; comparable or better than DoReMi/ODM. https://arxiv.org/abs/2410.11820

**Skill-it** (NeurIPS 2023 Spotlight): builds a "skills graph" over data-associated skills, online-samples exploiting prerequisite ordering between skills. Skill-level, not per-document. Gain: +36.5 pts over random on LEGO synthetic; -13.6% target-skill val loss vs training on that skill alone (fine-tuning setting). https://arxiv.org/abs/2307.14430

**DSIR** (Xie, Santurkar, Ma, Liang 2023): estimates document importance weights against a target distribution using hashed n-gram features, then importance-resamples. Deliberately cheap: 100M docs selected from full Pile in 4.5 hours. https://arxiv.org/abs/2302.03169

**QuRating** (Wettig, Gupta, Malik, Chen, ICML 2024): an LLM (GPT-3.5-turbo) makes pairwise quality judgments (style, expertise, facts, educational value); a distilled "QuRater" scalar model then rates the full 260B-token corpus. Expensive step is the pairwise LLM judgments, sampled not exhaustive. UNVERIFIED exact downstream deltas. https://arxiv.org/abs/2402.09739

**Ask-LLM** (Sachdeva et al. 2024): prompts an instruction-tuned LLM to directly judge keep/reject per example using its own reasoning. Does NOT avoid full scoring, the judge runs over the entire dataset; efficiency gain is in downstream training speed, not scoring cost. Gain: training on 60% Ask-LLM-selected data beats 100%-data training, 70% faster; robust even at 90% rejection. https://arxiv.org/abs/2402.09668

**Rho-1** (Lin, Gou et al. 2024): token-level (not document-level) selective language modeling; scores individual tokens with a reference model and concentrates loss on high-utility tokens. Reference model trained once; scoring is a cheap forward pass. Gain: up to +30% few-shot on 9 math tasks after 15B-token continual pretraining, matches DeepSeekMath with ~3% of tokens. https://arxiv.org/abs/2404.07965

**Aioli** (Chen et al. 2024, HazyResearch): shows prior mixing methods implicitly assume different "mixing laws"; derives a unified online method estimating mixing-law parameters live and adjusting domain proportions. Domain-level. UNVERIFIED precise deltas beyond "matches or beats" prior baselines. https://arxiv.org/abs/2411.05735

**AutoScale** (2024): two-stage, scale-aware data composition adapting mixture as training scale grows. UNVERIFIED beyond abstract-level detail in this pass. https://arxiv.org/abs/2407.20177

**AC-ODM** (2025): reframes online mixing as RL, Actor-Critic agent adjusts domain weights using a gradient-alignment reward. Domain-level, no per-document cost. UNVERIFIED precise numbers. https://arxiv.org/abs/2505.23878

**DsDm** (2024, MIT): uses "datamodels" (learned linear approximations of how a training subset affects target-task performance) to select data directly for target performance. Fitting datamodels requires many training runs, amortized once. Gain: 2x compute multiplier over baselines on held-out benchmarks (UNVERIFIED beyond abstract). https://arxiv.org/abs/2401.12926

**BLISS** (2025): bilevel optimization, upper level trains a scorer model to assign importance weights for pretraining selection. Gain: 1.7x speedup to matched performance vs prior SOTA at 1B scale. https://arxiv.org/abs/2510.06048

## Answers

**1. Region/arm-based online adapters:** ODM (explicit MAB/Exp3 over domains), ADO (online per-domain scaling laws), Aioli (online mixing-law estimation), AC-ODM (RL actor-critic over domain weights), Skill-it (online sampling over a skills graph). DoReMi is domain-level but fit offline, not adapted during the final run. MATES and Rho-1 adapt online at the example/token level rather than domain/arm level. https://arxiv.org/abs/2312.02406 https://arxiv.org/abs/2410.11820 https://arxiv.org/abs/2411.05735 https://arxiv.org/abs/2505.23878 https://arxiv.org/abs/2307.14430 https://arxiv.org/abs/2406.06046 https://arxiv.org/abs/2404.07965

**2. Genuinely open as of 2026:** robust document-level (not domain-level) online bandit selection at trillion-token scale without a cheap proxy is not clearly solved. MATES is the closest example-level adaptive method but still relies on a distilled proxy, not a bandit over raw documents. Every method found either distills to a cheap proxy (QuRating, MATES, DSIR) or works at coarser domain/skill granularity (ODM, ADO, Aioli, AC-ODM). Consistent with the survey "Data Mixing for LLM Pretraining" treating domain mixing and document filtering as largely separate literatures. https://arxiv.org/abs/2604.16380

**3. Budgeted sequential search deciding which documents get an expensive score:** UNVERIFIED, not found in this pass. Methods reviewed either score every document with a cheap proxy (DSIR, distilled QuRater, Rho-1 reference model, MATES distilled influence model) or score at domain/skill level only, with no per-document budget problem. Ask-LLM runs its expensive judge over the full dataset rather than budgeting it. RHO-LOSS and MATES budget expensive computation over a sampled subset used to train a proxy, which is adjacent but not a sequential/bandit search that chooses which specific documents get scored based on prior results. Off-topic but structurally similar budgeted-sequential-scoring work exists for LLM-judge evaluation cost reduction and scaling-law experiment selection, neither applied to pretraining document selection. https://arxiv.org/abs/2602.15481 https://arxiv.org/abs/2604.22753

This looks like a genuine gap rather than a search miss, but confidence is moderate given search-tool limits.
