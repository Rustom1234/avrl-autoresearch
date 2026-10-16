# H8: Targeted Synthetic Data Generation — Thesis Justification

## 1. Hypothesis and SOFisher Mapping

A policy that decides which region of skill space to spend the next expensive generate-and-verify budget on, using cheap per-region probe signals (student loss or error rate on a small held-out set), improves student capability per verified example more than uniform sampling or static error-ranked generation, because model errors and the marginal gains from fixing them are autocorrelated across neighbouring skills. This is testable directly: fix a generation-and-verification budget, compare final student accuracy per verified example across policies that differ only in how they allocate that budget across skill regions.

SOFisher mapping: field of view -> skill/task region in a taxonomy (e.g., math sub-skills, code error categories); rare target density -> student error rate in that region; cheap state feature -> student's loss on a small cached probe set per region (near-free, no teacher call); expensive measurement -> teacher generation plus verifier/judge scoring of a full synthetic batch for that region; spatial autocorrelation of targets -> skill-graph adjacency (Skill-it's ordered/interdependent skills); the DQN choosing the next FOV -> a bandit or RL policy choosing the next skill region to generate for.

## 2. Prior Art

- Skill-it (Chen et al. 2023, https://arxiv.org/abs/2307.14430) — ordered, interdependent skill structure exists in LM training data; skills-graph-guided sampling reaches higher LM-Eval-Harness accuracy with 1B tokens than uniform sampling with 3B tokens on a 3B-parameter continual-pretraining run.
- Textbooks Are All You Need / phi-1 (Gunasekar et al. 2023, https://arxiv.org/abs/2306.11644) — 1.3B model trained on 6B web tokens plus 1B GPT-3.5-synthesized textbook/exercise tokens (8 A100s, 4 days) reaches 50.6% pass@1 HumanEval, 55.5% MBPP; establishes synthetic-data quality over volume as a lever, but sampling is static, not policy-driven.
- Self-Instruct (Wang et al. 2022, https://arxiv.org/abs/2212.10560) — bootstrapped instruction generation from a model's own outputs gives a 33-point absolute gain on Super-NaturalInstructions over vanilla GPT-3, on par with InstructGPT-001; no targeting by student weakness.
- WizardLM / Evol-Instruct (Xu et al. 2023, https://arxiv.org/abs/2304.12244) — instruction complexity is evolved via LLM rewriting; human evaluation on a complexity-balanced test bed rates Evol-Instruct outputs above human-written ones, and WizardLM preferred over ChatGPT on high-complexity items. Complexity is escalated, not routed by measured student error.
- Magpie (Xu et al. 2024, https://arxiv.org/abs/2406.08464, ICLR 2025) — alignment data synthesized from an aligned model's own pre-query template with no seed prompts; demonstrates cheap high-volume generation but with no student-in-the-loop targeting.
- LEMA (An et al. 2023, https://arxiv.org/abs/2310.20689) — collects incorrect reasoning paths, uses GPT-4 as corrector to build mistake-correction pairs; improves CoT-only fine-tuning across multiple LLMs and reasoning benchmarks. Closest in spirit to error-targeted generation but corrects existing mistakes rather than allocating new-generation budget by region.
- STaR (Zelikman et al. 2022, https://arxiv.org/abs/2203.14465) — iterative rationale bootstrapping (generate, keep correct, fine-tune, repeat) performs comparably to a 30x larger model directly fine-tuned on CommonsenseQA; establishes the generate-verify-retrain loop this thesis reuses, without region-level allocation.
- ReST-EM (Singh et al. 2023, https://arxiv.org/abs/2312.06585, "Beyond Human Data") — EM-style rejection-sampling self-training; larger gains on harder APPS/MATH benchmarks as PaLM 2 scale increases, further gains with multiple iterations. Confirms verifier-gated self-generated data compounds, but sampling across problems is not skill-region-adaptive.
- DataEnvGym (https://arxiv.org/abs/2410.06215) — agent generates data inside a "teacher environment" conditioned on student feedback, using skills to identify weaknesses; adjacent to H8, exact efficiency numbers UNVERIFIED here.
- SwS: Self-aware Weakness-driven Problem Synthesis (Liang et al. 2025, https://arxiv.org/abs/2506.08989) — identifies weakness regions, synthesizes targeted RL problems there, including weak-to-strong teaching; numeric gain UNVERIFIED here.
- Skill-Targeted Adaptive Training (https://arxiv.org/abs/2510.10023) — adapts synthetic data toward skills the model still struggles with post-training, unlike prior new-skill-instilling work; numbers UNVERIFIED here.
- Towards Active Synthetic Data Generation for Finetuning LMs (https://arxiv.org/abs/2512.00884) — student-in-the-loop generation beats static one-shot generation at matched budget; simple active-learning criteria (student's own loss) beat LLM-judge seed selection. Closest existing baseline to H8.
- Multi-Armed Bandit for Optimizing Training on Synthetic Data (https://arxiv.org/abs/2412.05466) — UCB-based usability metric selects which synthetic data to keep; bandit precedent for online budget allocation, on image data not LM skill regions.
- MASS: Mathematical Data Selection via Skill Graphs (https://arxiv.org/abs/2503.14917) — skill-graph-guided selection for math pretraining, extending Skill-it's graph to a harder domain, the thesis's proposed testbed.
- Skill-Aware Data Selection for Reasoning Distillation (https://arxiv.org/abs/2601.10109) — skill-aware selection improves data efficiency for reasoning distillation (2026), generalizing to a teacher-student distillation setting.

## 3. What Is Genuinely Open as of 2026

No paper found here trains an explicit sequential policy (bandit or RL, à la SOFisher's DQN) that allocates a *generation-and-verification* budget across skill regions using *only* cheap probe signals, then reports **student gain per verified example** against both a uniform baseline and a static error-ranked baseline in one budget-matched experiment. The closest work (2512.00884, DataEnvGym, SwS, 2510.10023) each has one or two of: student-in-the-loop feedback, skill-region weakness targeting, or active-learning-style seed selection, but none confirmed here to combine all three with an autocorrelation-exploiting policy tested against a *static* skill-ranked baseline (the harder comparison; uniform alone is largely already beaten in this literature). A year is needed because: building a taxonomy with a reliable cheap probe and trustworthy verifier is nontrivial engineering; isolating the policy's causal effect from confounds (verifier quality, teacher strength, taxonomy granularity) needs multiple controlled runs; and whether skill-space autocorrelation beats a well-tuned static ranking, not just uniform sampling, is a genuinely open empirical question needing iteration.

## 4. Three-Condition Check

1. Errors/gains autocorrelated across neighbouring skills: supported by Skill-it's demonstrated skill ordering and interdependency (2307.14430), MASS's skill-graph structure in math (2503.14917), and persistence of clustered weaknesses post-training in Skill-Targeted Adaptive Training (2510.10023).
2. Expensive measurement costs far more than cheap features: directionally true (teacher generation plus LLM-judge/verifier scoring vs. a forward pass on a cached probe set) but no paper found here quantifies the exact cost ratio for this specific pipeline — UNVERIFIED as a precise number, plausible from known inference API pricing asymmetries.
3. Cheap features alone cannot rank targets well (a policy is needed, not just a static ranking): supported by 2512.00884's finding that active, in-the-loop use of student loss beats static/one-shot use of similar signals, implying the value is in adaptive allocation, not the raw signal alone.

## 5. Year Plan

Phase 1 (months 1-3, feasibility): build a skill taxonomy (math word-problem sub-skills or code-error categories, 10-30 regions), a cached cheap probe per region, and a verifier (rule-based for math, unit tests for code). First experiment, under one week on 4 A100s: open 7B teacher (Qwen2.5-7B-Instruct or Llama-3.1-8B-Instruct) generates and verifies for a 1B student (Qwen2.5-1.5B or Llama-3.2-1B) on GSM8K/MATH-subset or MBPP/APPS-subset. Compare three allocators at matched verified-example budget: uniform, static error-rate-ranked, and a simple contextual bandit (UCB or Thompson sampling) using probe loss as state. Metric: accuracy gain per verified example (and per dollar of compute). Baseline to beat: the active criterion from 2512.00884, plus a static skill-ranked baseline in the spirit of 2510.10023.

Phase 2 (months 4-8, robustness): scale taxonomy granularity, test a second domain (code) for transfer, swap the bandit for a small RL policy if the bandit underperforms, ablate verifier noise and teacher strength, test whether skill-graph adjacency (not just marginal loss) adds value over a flat bandit.

Phase 3 (months 9-12, scale and writeup): scale student/teacher sizes toward a more realistic midtraining budget if compute allows, test on a third domain, finalize comparison against the strongest 2025-2026 baselines, write up.

## 6. Kill Criteria

Kill if, at matched budget, the policy fails to beat the static error-ranked baseline by a prespecified margin (5% relative gain per verified example) after Phase 1 tuning. Kill if probe loss per region is uncorrelated with actual post-generation gain in that region (breaks condition 1). Kill if verifier/teacher cost is not meaningfully larger than probe cost in the chosen domain (breaks condition 2). Kill if a static one-shot ranking (computed once, never updated) matches the adaptive policy, meaning the policy adds nothing over a cheap precomputed ranking.

## 7. Impact

If it works, a lab would stop generating synthetic data uniformly or via one fixed pass over a skill/error ranking, and instead run generation as an online loop: after every N verified batches, re-score the student on cheap per-region probes and let a lightweight bandit/RL policy reallocate the next batch's region, closing the loop between teacher spend and actual student weakness. This changes only the *scheduler* around an existing generate-verify-train pipeline (STaR/ReST-EM/Evol-Instruct-style), not the generation or verification method itself, so it is a low-risk drop-in wherever a measurable probe set and per-example verifier already exist.

SCORECARD
novelty: 5
feasibility_1yr: 8
lab_impact: 7
sofisher_fit: 8
strongest_baseline: Towards Active Synthetic Data Generation for Finetuning Language Models (arXiv:2512.00884)
one_line_pitch: A cheap per-skill-region probe plus a bandit policy over where to spend the next expensive generate-and-verify budget should beat uniform and static error-ranked synthetic data generation per verified example, because student weaknesses cluster and drift across neighbouring skills the way rare targets cluster across a tissue slide.
