# H1: Active Evaluation During Training

## 1. Hypothesis and SOFisher mapping

A learned policy that adaptively chooses which evaluation items or benchmarks to run next on the current training checkpoint, exploiting correlation structure among items and across checkpoints, can estimate a model's capability curve and detect regressions at a fraction of the judge-call and compute cost of exhaustive periodic benchmarking. Because this policy is cheap to run relative to the judge-based measurement it schedules, it can sit inside the training loop continuously rather than as a periodic offline sweep.

SOFisher mapping:
- Slide: the full training run, i.e. the sequence of checkpoints produced over time.
- Field of view: one (checkpoint, benchmark item) pair that could be evaluated next.
- Expensive measurement: running the item through the model and scoring it with an LLM judge or exact-match harness (judge calls, generation compute).
- Cheap state features: item metadata (difficulty/discrimination from IRT, embedding cluster, benchmark tag), the model's own logits/uncertainty on that item, and prior scores from nearby checkpoints.
- Autocorrelation: items cluster by latent skill (IRT discrimination/difficulty), and scores are highly correlated checkpoint-to-checkpoint (a model that scored X at step t rarely swings wildly by step t+1).
- Policy: a sequential selector (bandit, RL, or Fisher-information rule) choosing the next item/checkpoint to spend judge budget on.
- Reward: reduction in estimation error of the capability curve, or early/cheap detection of a true regression, per unit of judge-call budget spent.

## 2. Prior art

- tinyBenchmarks, arXiv 2402.14992. IRT-based item clustering picks ~100 examples per benchmark; estimation error under 2% versus full Open LLM Leaderboard/MMLU/HELM/AlpacaEval scores.
- Active Testing: Sample-Efficient Model Evaluation, Kossen et al., ICML 2021, arXiv 2103.05331. Acquisition-based unbiased risk estimator for test-time labeling; foundational framework, validated on WideResNets/Gaussian processes, not LLMs.
- Active Surrogate Estimators, arXiv 2202.06881. Extends active testing with a learned surrogate to further cut labels needed.
- Active Evaluation Acquisition for Efficient LLM Benchmarking, arXiv 2410.05952 (ICML 2025). Models dependencies across test items and trains an RL-based policy to pick the next benchmark item; closest architectural analog to a SOFisher-style policy for LLM eval, though no cross-checkpoint reuse reported.
- Reliable and Efficient Amortized Model-based Evaluation, arXiv 2503.13335. IRT difficulty predictor plus item generator for adaptive question selection; evaluated on 22 benchmarks and 172 language models.
- ATLAS: Adaptive Testing for LLM Evaluation, arXiv 2511.04689. Fisher-information-guided IRT item selection; up to 90% fewer items needed (e.g. 41 of 5,600 HellaSwag items match whole-bank ability estimates); flags that 23-31% of >3,000 evaluated models shift rank by more than 10 positions once ability is estimated correctly, i.e. static leaderboards mis-rank models.
- Towards Explainable Computerized Adaptive Testing with LLM, ACL Findings EMNLP 2024, aclanthology.org/2024.findings-emnlp.149. LLM inside a CAT loop, comparable-or-better accuracy plus improved transparency versus classic CAT item-selection rules.
- Leveraging computerized adaptive testing for cost-effective evaluation of LLMs in medical benchmarking, npj Digital Medicine 2026, nature.com/articles/s41746-026-02671-w. IRT-grounded CAT reaches near-perfect correlation with full item bank using 1.3% of items; eval time drops from 6.85 hours to 8.4 minutes, cost from about $1,475 to under $5.
- Position: AI Evaluation Should Learn from How We Test Humans, arXiv 2306.10512. Position paper arguing for psychometric/CAT-style AI evaluation; motivates the line, no new numbers.
- Towards Reliable LLM Evaluation: Correcting the Winner's Curse in Adaptive Benchmarking, arXiv 2605.05973. Shows adaptive/selection-based benchmarking introduces an optimistic bias and proposes a correction; relevant to any policy that adaptively narrows items per checkpoint.
- Efficient Benchmarking Is Just Feature Selection and Multiple Regression, arXiv 2605.25773. Reframes efficient benchmarking as regression over item features, a simpler baseline family to beat.
- Consistent and Distinctive: Benchmark Efficiency via Maximum Independent Set on Similarity Graphs, arXiv 2606.01400. Graph-based item selection exploiting item-similarity structure, an alternative to IRT for the "autocorrelation" structure.
- EffiEval: Efficient and Generalizable Model Evaluation via Capability Coverage Maximization, arXiv 2508.09662. Subset selection maximizing capability coverage rather than IRT ability; UNVERIFIED whether it reports checkpoint reuse.
- CELEUS: Certifiable and Efficient LLM Evaluation via E-Processes, arXiv 2606.20820. Anytime-valid sequential inference for LLM evaluation with a certifiable stopping rule, relevant to running continuously and stopping when confident.
- Predicting LLM Reasoning Performance with Small Proxy Model (rBridge), arXiv 2509.21013 (ICLR 2026). Sub-1B proxy models predict large-model reasoning scores, cutting ranking cost over 100x; a cheap-feature analog rather than an item-selection policy.

## 3. What is genuinely open as of 2026

The CAT/IRT/active-acquisition literature solves single-checkpoint, single-benchmark adaptive item selection well (ATLAS, tinyBenchmarks, the amortized IRT paper). Not established: a policy that carries information across checkpoints within one run, using the fact that checkpoint t+1 is highly correlated with checkpoint t, to decide not just which items but whether to evaluate a checkpoint at all, and when to widen versus narrow the probe. The RL-based active-acquisition paper (2410.05952) is architecturally closest to a SOFisher-style policy but was evaluated as a static per-benchmark selector, not a continuous in-loop scheduler, and does not address the winner's-curse bias adaptive selection introduces over repeated use (2605.05973). No paper found combines: cross-checkpoint correlation modeling, a live policy embedded in a training loop, and regression detection with a calibrated false-positive rate under repeated adaptive querying. A year is needed to build and validate the policy across a real multi-checkpoint trajectory, apply the winner's-curse correction online, and confirm it holds on judge-based, not just exact-match, benchmarks where item correlation is noisier.

## 4. Three-condition check

- Expensive measurement really expensive relative to cheap features: yes. Judge-based benchmark runs cost hundreds to thousands of dollars per sweep (PaperBench: ~$400/rollout, ~$8,000 for 20 papers, per huggingface.co/blog/evaleval/eval-costs-bottleneck), while cheap features (IRT item parameters, embeddings, prior-checkpoint scores) are near-free or already cached.
- Cheap features weak on their own: partially confirmed. rBridge (2509.21013) shows small proxies predict some reasoning scores well, so cheap features are not useless, but ATLAS (2511.04689) shows raw accuracy alone mis-ranks 23-31% of models by more than 10 positions, so naive cheap signals are insufficient. UNVERIFIED how far this extends to judge-based benchmarks specifically.
- Exploitable structure: yes for items (IRT discrimination/difficulty clustering underlies tinyBenchmarks, ATLAS, and the amortized-evaluation paper, all with concrete gains). Checkpoint-to-checkpoint correlation is the weaker leg: no cited paper directly measures inter-checkpoint score correlation within one training run; UNVERIFIED, exactly what phase 1 must establish first.

## 5. Year plan

Phase 1 (weeks 1-6, 4 A100s, under a week for the first experiment): using an open checkpoint suite with cached full-benchmark scores (Pythia or OLMo intermediate checkpoints), first measure checkpoint-to-checkpoint score correlation and item-cluster stability directly, no training needed. Then, on the 4 A100s, run a subset of a judge-based benchmark (an AlpacaEval-style subset) across several checkpoints to get real judge-cost numbers, and train a lightweight sequential item-selection policy (bandit or small RL) that reuses prior-checkpoint scores as state. Metric: mean absolute error of the estimated capability curve at fixed judge-call budget, and detection latency (checkpoints until flagged) for injected synthetic regressions at fixed false-positive rate. Baseline to beat: ATLAS-style per-checkpoint Fisher-information item selection run independently with no cross-checkpoint reuse, the strongest published method, which the policy must beat by exploiting correlation structure.

Phase 2 (months 2-6): extend to a real multi-week training run (or continued pretraining) with the policy running online, add the winner's-curse correction from 2605.05973, and test on judge-based, noisier benchmarks, not just exact-match ones.

Phase 3 (months 6-12): stress-test with adversarial and non-stationary conditions, a data-mix change, a sudden capability jump, a reward-hacking-induced regression, to see whether the policy's cross-checkpoint prior misleads it, and package as a scheduler service with a calibrated stopping rule.

## 6. Kill criteria

Kill if, in phase 1: (a) checkpoint-to-checkpoint correlation is too weak or non-stationary, i.e. a static per-checkpoint IRT baseline (ATLAS-style) matches the learned policy within noise; (b) judge-noise variance dominates signal such that no policy detects target regression sizes faster than random sampling at matched budget; (c) the winner's-curse correction costs so much budget that net gain over ATLAS is under roughly 20%, too small to justify a new system over published static CAT.

## 7. Impact

If it works, a lab like Anthropic would replace fixed periodic full-benchmark sweeps with a lightweight, always-on eval scheduler that decides per checkpoint which small item subset to run and only triggers a full judge-based sweep when uncertainty or a regression signal crosses a threshold. Evaluation would run continuously at training-loop frequency rather than at expensive periodic checkpoints, catching regressions (including reward-hacking-style ones) closer to the step that caused them, while cutting judge-API and generation compute spent on evaluation, similar in spirit to how loss-spike and gradient monitors already run continuously and cheaply inside training.

SCORECARD
novelty: 5
feasibility_1yr: 7
lab_impact: 6
sofisher_fit: 7
strongest_baseline: ATLAS (Fisher-information IRT adaptive item selection, arXiv 2511.04689)
one_line_pitch: A policy that reuses checkpoint-to-checkpoint correlation, not just item-level IRT structure, could beat static per-checkpoint adaptive testing at catching regressions during training for a fraction of the judge-call budget.
