# H5: Budgeted Memorisation and Extraction Auditing

## 1. Hypothesis and SOFisher mapping

Verbatim memorisation of training data (personal data, copyrighted text, benchmark items) in a trained language model clusters by source, duplication count, and content type, rather than being spread uniformly across the training corpus. Given a fixed query budget against the model, where each extraction probe or membership-inference test costs a generation plus a judging pass, a learned policy that chooses which training-data region to probe next, updating on earlier hits, will recover more memorised sequences per query than random or static-ranked probing, and can run continuously across checkpoints as a memorisation monitor during training.

SOFisher mapping: field of view to image next -> training-data region (shard, source, duplication bucket) to probe next; rare targets -> memorised sequences, which are rare relative to total probes; expensive measurement -> a generation-plus-judge extraction attempt; cheap state features -> duplication count, source metadata, perplexity or Min-K% score computed once offline; spatial autocorrelation -> memorisation clusters by source and duplication rather than scattering randomly, which is exactly the condition SOFisher exploits for imaging.

## 2. Prior art

- Carlini et al. 2021, Extracting Training Data from Large Language Models. Extracted 600+ memorised training examples from GPT-2 including PII, IRC logs, and code, out of 1800 candidates tested. https://www.usenix.org/system/files/sec21-carlini-extracting.pdf
- Carlini, Ippolito, Jagielski, Lee, Tramer, Zhang 2023 (ICLR), Quantifying Memorization Across Neural Language Models. Log-linear growth of memorisation with model capacity, duplication count, and context length; GPT-J (6B) memorises at least 1% of The Pile. https://arxiv.org/abs/2202.07646
- Nasr, Carlini, Hayase et al. 2023, Scalable Extraction of Training Data from (Production) Language Models. Recovered 10,000+ ChatGPT training examples for about $200 in queries using a divergence attack that raised the emission rate 150x. https://arxiv.org/abs/2311.17035
- Biderman, Prashanth, Sutawika et al. 2023 (NeurIPS), Emergent and Predictable Memorization in Large Language Models. Shows memorisation in Pythia checkpoints is predictable by extrapolating from smaller/earlier runs, but individual sequences that will be memorised are hard to identify in advance (emergent at the sequence level). https://arxiv.org/abs/2304.11158
- Shi, Zhang, Azaria et al. 2024 (ICLR), Detecting Pretraining Data from Large Language Models (Min-K% Prob). Reference-free membership inference using the average log-likelihood of the k% lowest-probability tokens; introduces the WikiMIA benchmark. https://proceedings.iclr.cc/paper_files/paper/2024/file/e32ad85fa27be4a9868d55703f01323e-Paper-Conference.pdf
- Zhang et al. 2025 (ICLR Spotlight), Min-K%++: Improved Baseline for Detecting Pre-Training Data of LLMs. Improves on Min-K% by normalizing with the local maximum of expected log-likelihood, reports higher AUROC across the WikiMIA and MIMIR benchmarks. https://proceedings.iclr.cc/paper_files/paper/2025/file/a2e3b4132ab2e0b7a21e6e75da7f91a9-Paper-Conference.pdf
- Lee, Ippolito, Nystrom et al. 2022 (ACL), Deduplicating Training Data Makes Language Models Better. Deduplication cuts verbatim emission roughly 10x and removes examples like a 61-word sentence duplicated 60,000+ times in C4. https://arxiv.org/abs/2107.06499
- Kandpal, Wallace, Raffel 2022 (ICML), Deduplicating Training Data Mitigates Privacy Risks in Language Models. Regeneration rate is superlinear in duplicate count (roughly 1000x more likely to regenerate a sequence duplicated 10x than one seen once); memorisation detectors are near chance on non-duplicated sequences. https://proceedings.mlr.press/v162/kandpal22a/kandpal22a.pdf
- Hans, Wen, Jain et al. 2024 (NeurIPS), Be Like a Goldfish, Don't Memorize! Mitigating Memorization in Generative LLMs. Dropping a random subset of tokens from the training loss (goldfish mask) suppresses extractable verbatim memorisation in billion-scale LLaMA-2 training with little benchmark cost. https://arxiv.org/abs/2406.10209
- Ahmed et al. 2025, Extracting Memorized Pieces of (Copyrighted) Books from Open-Weight Language Models. Llama 3.1 70B can be prompted with a book's opening words to deterministically regenerate large fractions of some in-copyright books near-verbatim. https://arxiv.org/abs/2505.12546
- Chang, Cramer, Soni, Bamman 2024, DE-COP: Detecting Copyrighted Content in Language Models' Training Data. Benchmark pairing verbatim passages against paraphrases to test whether a model can identify the original, used for copyright-membership auditing. https://arxiv.org/pdf/2402.09910
- Morris, Naik, Zhao et al. 2025, How Much Do Language Models Memorize? Estimates GPT-style models store about 3.6 bits of unintended memorisation per parameter before capacity saturates and generalisation begins ("grokking" transition). https://arxiv.org/abs/2505.24832
- ACL BlackboxNLP 2024, Insights from Tracking LLM Memorization During Training. Tracking held-in vs held-out log-likelihood across Pythia checkpoints shows an instantaneous memorisation spike after LR warm-up followed by long persistence in larger models. https://aclanthology.org/2024.blackboxnlp-1.4.pdf
- Biderman et al. 2025, Hubble: A Model Suite to Advance the Study of LLM Memorization. Public model suite with controlled, injected memorisation targets (PII, copyrighted excerpts, benchmark contamination) across checkpoints, built explicitly to support memorisation research. https://arxiv.org/pdf/2510.19811
- 2026, On the Evidentiary Limits of Membership Inference for Copyright Auditing. Argues current MIA methods (including Min-K% family) have weak, sometimes near-chance, per-example reliability when used as legal/forensic evidence of memorisation, especially at low duplication counts. UNVERIFIED author list and venue at time of writing. https://arxiv.org/pdf/2601.12937

## 3. What is genuinely open as of 2026

No published method learns an adaptive, checkpoint-persistent policy over training-data regions for extraction/MIA probing under an explicit query budget. Existing work either (a) proposes a single membership-inference score (Min-K%, Min-K%++) applied uniformly or to a fixed candidate set with no sequential budget allocation, or (b) proposes static, offline extraction pipelines (Carlini, Nasr) that pick candidates once by heuristic ranking (duplication count, perplexity) rather than updating a policy from observed hits. Biderman 2023 predicts aggregate memorisation rates but explicitly does not solve per-sequence targeting. The BlackboxNLP and Hubble lines track memorisation across checkpoints but as passive measurement on a known target set, not as an online search problem over an unknown target set with a probe budget. The 2026 evidentiary-limits paper raises open doubt about whether any current cheap-feature score (duplication, perplexity, Min-K%) is a reliable enough prior to seed a policy, which is exactly the open empirical question a bandit/RL policy must resolve. A year is needed because: building the injected-memorisation testbed or wiring Pythia's public training order to a probe harness takes real engineering; validating that the policy transfers across model scales and checkpoint spacing requires multiple training runs; and demonstrating utility as a continuous training-time monitor requires integrating with an actual training loop, not a single post-hoc audit, which is a materially different systems problem than one-shot extraction.

## 4. Three-condition check

- Probe is expensive: yes, evidenced directly. Nasr et al. spent about $200 in API queries plus a 9TB matching pass to recover 10,000 examples; each extraction attempt requires generation plus judging against a reference corpus (Nasr et al. 2023, arxiv 2311.17035).
- Cheap predictors of memorisation are weak: yes. Kandpal et al. 2022 report memorisation detectors are near chance accuracy on non-duplicated sequences; the 2026 evidentiary-limits paper argues Min-K%-family scores are unreliable per-example evidence (arxiv 2601.12937). Duplication count alone is a strong but noisy signal, superlinear but not deterministic (Kandpal et al. 2022, Carlini et al. 2023).
- Memorisation clusters by source/duplication: yes. Carlini et al. 2023 show log-linear scaling with duplication count; Lee et al. 2022 show a small number of highly-duplicated sources account for disproportionate verbatim emission; Hubble's design assumes and exploits source-level clustering of injected targets.

## 5. Year plan

Phase 1 (months 1-3): build the probe harness on Pythia (EleutherAI, public training data order and checkpoints, 8 sizes from 70M-12B). First experiment on 4 A100s in under a week: fix Pythia-1.4B or 2.8B, use the known Pile duplication counts per document as ground truth, run a bandit/DQN policy over document-cluster "regions" (grouped by source + duplication decile) that chooses which region to extraction-probe next, using cheap features (duplication count, source tag, one-time Min-K% pass) as state. Metric: memorised sequences recovered per query (hits at fixed budget, e.g. 5,000 probes), against random probing and static duplication-rank probing as baselines. Strongest existing baseline to beat: static ranking by duplication count (the closest thing to a policy in Carlini/Kandpal), and Min-K%++ pre-filtering.

Phase 2 (months 4-8): extend the policy to run across checkpoints during a live or replayed training run (continuous monitor), test transfer across Pythia model sizes, and add copyrighted-book and PII-style targets (using Hubble's injected targets for ground truth).

Phase 3 (months 9-12): validate on one larger open-weight model with reconstructable training order, write up detection-rate-per-dollar comparisons, and package as a monitoring tool a lab could run alongside training.

## 6. Kill criteria

Kill if, at matched query budget, the learned policy does not beat static duplication-rank probing by a clear margin (e.g. less than 15% more hits) on the Phase 1 experiment; that would mean duplication count alone is already near-sufficient and a learned policy adds engineering cost without benefit. Also kill if cheap state features carry so little signal that the policy cannot outperform random within the first 3 months, since that undermines the premise that structure exists to exploit at the "cheap feature" level.

## 7. Impact

If it works, a lab would add a standing, budgeted memorisation-probing service that runs against checkpoints during pretraining (not just post-hoc), reallocating a fixed weekly query budget toward the highest-risk data regions (duplication clusters, sensitive sources) as evidence accumulates, and would use recovered hits to trigger targeted deduplication or goldfish-loss masking on just the flagged regions rather than blanket deduplication of the whole corpus. It also gives a defensible, budget-bounded audit trail for regulatory or copyright risk review, replacing one-off, unaudited extraction sweeps.

SCORECARD
novelty: 6
feasibility_1yr: 7
lab_impact: 6
sofisher_fit: 7
strongest_baseline: static duplication-rank probing with Min-K%++ pre-filter
one_line_pitch: A budget-aware policy that learns which training-data regions to extraction-probe next finds more memorised sequences per dollar than static duplication-ranked probing, and can run continuously across checkpoints as a training-time memorisation monitor.
