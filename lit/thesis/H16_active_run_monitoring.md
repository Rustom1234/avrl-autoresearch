# H16: Active Monitoring of Large Training Runs (SOFisher-Inspired Diagnostic Policy)

## 1. Hypothesis and SOFisher mapping

Training instabilities in large model runs (loss spikes, attention-logit blowups, dead experts, silent data corruption) are rare, spatially and temporally autocorrelated events preceded by localized precursor signals in specific layers, steps, or data shards, and a learned policy that allocates a fixed diagnostic budget toward the layers, steps, and shards most likely to show precursors will detect impending failures earlier per unit of observability cost than fixed-interval monitoring. This maps directly onto SOFisher: fields of view become (layer, step, data-shard) triples, the rare targets are precursor signatures instead of rare cells, the expensive measurement is a full diagnostic pass (per-layer activation/gradient stats, a held-out eval, a checkpoint save) instead of high-resolution imaging, and the cheap state feature is a running scalar (loss, grad norm, router entropy) instead of a cheap low-res scan.

## 2. Prior art

One figure per paper. Marked UNVERIFIED where the exact number could not be confirmed against primary text in this pass (search snippets and abstracts only, no full-PDF read).

- PaLM (Chowdhery et al. 2022), https://arxiv.org/abs/2204.02311: roughly 20 loss spikes over the 540B run, each handled by rolling back to a checkpoint about 100 steps earlier and skipping 200-500 data batches. UNVERIFIED exact spike count, drawn from search summary not primary text.
- OPT-175B logbook (Zhang et al. 2022), https://arxiv.org/abs/2205.01068: 35 manual restarts and 100+ hosts cycled over 56 days of training, documented in a public logbook.
- Wortsman et al. 2023, "Small-scale proxies for large-scale Transformer training instabilities," https://arxiv.org/abs/2309.14322 (ICLR 2024 oral): reproduces attention-logit-growth and output-logit-divergence instabilities in small models by training at high learning rate, and shows large-scale mitigations (qk-layernorm) transfer down to small scale. Number: accepted as an ICLR 2024 oral.
- Molybog et al. 2023, "A Theory on Adam Instability in Large-Scale Machine Learning," https://arxiv.org/abs/2304.09871: 17 Meta co-authors formalize Adam-driven divergence as a large-batch, large-model phenomenon, explaining PaLM/OPT-style spikes theoretically.
- Dehghani et al. 2023, "Scaling Vision Transformers to 22B Parameters" (origin of qk-layernorm, cited secondhand via Wortsman et al. 2023): introduces qk-layernorm to control attention logit growth at 22B-parameter scale. UNVERIFIED exact arXiv ID, not directly fetched this pass.
- Zoph et al. 2022, ST-MoE, https://arxiv.org/abs/2202.08906: introduces router z-loss to stabilize mixture-of-experts training. UNVERIFIED exact z-loss coefficient, not confirmed from primary text this pass.
- Llama 3 Herd of Models (2024), https://arxiv.org/abs/2407.21783: abstract claims "techniques for achieving better training stability" for the 405B run. UNVERIFIED specific mechanisms, could not confirm beyond the abstract in this pass.
- OLMo 2 (2025), https://arxiv.org/abs/2501.00656: abstract claims improved training stability techniques for the OLMo 2 family. UNVERIFIED specific mechanisms, could not confirm beyond the abstract in this pass.
- "Understanding Silent Data Corruption in LLM Training" (2025, ACL), https://arxiv.org/abs/2502.12340: analyzes SDC at submodule, optimizer-step, and training-period granularity; reports 6 unplanned job interruptions attributed to SDC in a 54-day Meta pretraining snapshot, and cites a Google estimate of roughly one Gemini SDC event every 1-2 weeks.
- ZClip (2025), "Adaptive Spike Mitigation for LLM Pre-Training," https://arxiv.org/abs/2504.02507: adaptive gradient-norm clipping that reacts to spikes in real time, automating the manual PaLM/OPT-style rollback loop.
- LLM-PRISM (2026), https://arxiv.org/abs/2604.10390: characterizes silent data corruption arising from permanent GPU faults during LLM training, extending the 2025 SDC line toward fault localization.
- "Training instability in deep learning follows low-dimensional dynamical principles" (2026), https://arxiv.org/abs/2601.13160: argues instability trajectories are governed by a small number of latent dynamical variables, i.e. a low-dimensional, potentially predictable precursor structure.
- "Revisiting Reliability in Large-Scale Machine Learning Research Clusters" (2024), https://arxiv.org/abs/2410.21680: infrastructure-level reliability study of large training clusters, contextualizing how much interruption budget is hardware/SDC-driven versus loss-instability-driven.
- "Initialization of Large Language Models via Reparameterization to Mitigate Loss Spikes" (2024), https://arxiv.org/abs/2410.05052: shows spike frequency can be reduced by initialization choice alone, meaning some precursor risk is knowable before step one.

## 3. What is genuinely open as of 2026

All of the prior art above is either a fixed heuristic for reacting after instability appears (PaLM/OPT rollback, ZClip's real-time clip), a static architectural fix applied uniformly everywhere (qk-layernorm, z-loss, reparameterized init), or a passive characterization of instability or SDC after the fact (Wortsman et al., Molybog et al., the SDC papers, the 2026 low-dimensional-dynamics paper). None of them pose the problem as a sequential decision: given a fixed diagnostic budget per step, which (layer, step, shard) to probe next, updated online as evidence accumulates. I found no paper in this pass that trains a policy over that allocation problem, which is the gap H16 targets, analogous to how SOFisher's novelty was the policy rather than the imaging modality itself. A year is needed because: (i) inducing rare, controllable instabilities cheaply and repeatably is itself unsolved outside the specific high-LR recipe in Wortsman et al., which has not been shown to extend to MoE dead-expert or SDC-style corruption at small scale (UNVERIFIED that no one has tried this); (ii) a working active-diagnostic policy needs a replay or simulation harness before it can be trained cheaply, and no such harness exists off the shelf; (iii) transferring a policy trained on 125M-350M runs to production scale is an unproven step, mirroring the open scaling question Wortsman et al. themselves leave unresolved.

## 4. Three-condition check

SOFisher needs: rare targets, spatial/temporal autocorrelation of precursors, and an expensive measurement that dominates cost while cheap features alone cannot substitute.

- Rarity: confirmed. PaLM saw about 20 spikes across a full 540B run and OPT needed only 35 restarts across 56 days, so instabilities are a small fraction of total steps (Chowdhery et al. 2022; Zhang et al. 2022).
- Autocorrelation and localization: partially confirmed. The 2026 low-dimensional-dynamics paper (arXiv:2601.13160) argues instability follows a small number of latent trajectories, and Wortsman et al. (2023) localize instability to specific mechanisms (attention logits, output logits) rather than the whole network uniformly, supporting layer-level localization. Temporal autocorrelation, meaning a precursor is visible several steps before visible divergence, is implied by PaLM's practice of rolling back about 100 steps, but no paper found in this pass directly measures a precursor lead-time curve. UNVERIFIED as a quantified claim.
- Cost asymmetry: confirmed qualitatively. Full diagnostic suites (held-out eval, checkpoint save/rollback, per-layer activation and gradient logging) are treated as too expensive to run every step everywhere, implicit in why PaLM and OPT relied on rollback-after-the-fact rather than continuous full diagnostics, while cheap scalars (loss, grad norm) are cheap but insufficient alone to predict divergence before it happens, since per Molybog et al. (2023) the failure mode is a state-dependent optimizer interaction not visible in the loss curve until it is already diverging.

## 5. Year plan

- Phase 1, weeks 1-6, first experiment on 4 A100s in under a week: train 125M-350M GPT-style models using the high-LR instability-induction recipe from Wortsman et al. (2023), inject an artificial SDC channel (bit-flip or gradient corruption on a random shard) and, if using MoE, an artificial dead-expert trigger, and log full per-layer activation and gradient stats at every step as ground truth. Compare a fixed-interval monitor against an oracle upper bound and a random policy. Metric: precursor lead-time per unit diagnostic cost, i.e. steps of advance warning gained divided by compute spent on diagnostics, at matched total diagnostic budget. Strongest existing baseline: fixed-interval full-suite monitoring at PaLM/OPT-style cadence (roughly every 100-500 steps, UNVERIFIED exact cadence) combined with rollback-after-detection, plus ZClip as an always-on cheap-signal baseline.
- Phase 2, months 2-6: train the DQN-style policy on the Phase 1 simulator, extend from offline replay to online decision-making inside a live small run, and test transfer across model sizes from 125M to 1B.
- Phase 3, months 6-12: attempt one real, larger-scale validation (partner lab or bigger internal run) comparing the policy's diagnostic allocation against the lab's existing fixed-interval monitoring on wall-clock time to first intervention.

## 6. Kill criteria

Kill if: (a) Phase 1 shows fixed-interval monitoring at matched budget ties or beats the oracle upper bound, meaning there is no autocorrelation structure worth exploiting; (b) the artificial instability-injection recipe fails to produce instabilities distinguishable from noise at 125M-350M scale, meaning the small-scale proxy premise from Wortsman et al. does not extend to SDC or dead-expert failure modes; (c) measured precursor lead time is under about 5 steps, too short for any policy-driven diagnostic to act on before a fixed-interval monitor would have caught it anyway.

## 7. Impact on a lab's stack

If it works even partially, a lab replaces blanket per-step or fixed-cadence diagnostic logging with a learned diagnostic scheduler, cutting observability compute and storage overhead while catching instabilities earlier, shortening the loop from detection to rollback or learning-rate cut, and reducing wasted GPU-hours on runs that diverge unnoticed for many steps. Phase 1 also produces a reusable small-scale instability-injection harness that is useful on its own for testing any future stability intervention, independent of whether the RL policy succeeds.

SCORECARD
novelty: 6
feasibility_1yr: 5
lab_impact: 6
sofisher_fit: 7
strongest_baseline: fixed-interval full-diagnostic monitoring with PaLM/OPT-style checkpoint rollback, plus ZClip as a cheap always-on baseline
one_line_pitch: Train an RL policy to choose which layers, steps, and data shards to run expensive training diagnostics on, so rare instabilities are caught earlier per unit of observability cost than fixed-interval monitoring.
